use crate::ast::{
    BinOp, Block as Block_, Expr as Expr_, Field, Literal, Stmt as Stmt_, SymbolPath, Typed,
};
use crate::bytecode::{opcode, BuilderError, BytecodeBuilder, Function, Label};
use crate::error::{Error, ErrorList};
use crate::id::{reserved_id, Id, IdMap};
use crate::sema::{TypedFunction, TypedProgram};
use crate::span::{Span, Spanned};
use crate::ty::{expand_all, Type, TypeCon, Unique};
use crate::utils::HashMapWithScope;
use rustc_hash::FxHashMap;

type TExpr = Expr_<Type>;
type TypedExpr = Typed<TExpr, Type>;
type TypedStmt = Stmt_<Type>;
type TypedBlock = Block_<Type>;

macro_rules! try_b {
    ($result:expr, $span:expr) => {
        match $result {
            Ok(value) => Some(value),
            Err(err) => {
                error!($span, "{}", err);
                None
            }
        }
    };
}

fn i2u(i: i64) -> u64 {
    u64::from_le_bytes(i.to_le_bytes())
}

fn sort_by_field_order(order: &[Id], fields: &mut Vec<(Spanned<Id>, TypedExpr)>) {
    'a: for i in 0..order.len() {
        let (expr_name, _) = &fields[i];
        if expr_name.kind != order[i] {
            for j in i + 1..fields.len() {
                let (expr_name, _) = &fields[j];
                if expr_name.kind == order[i] {
                    fields.swap(i, j);
                    continue 'a;
                }
            }
            panic!();
        }
    }
}

fn binop_opcode(binop: &BinOp, ty: &Type) -> opcode::Opcode {
    use opcode::*;

    match ty {
        Type::Int => match binop {
            BinOp::Add => BINOP_IADD,
            BinOp::Sub => BINOP_ISUB,
            BinOp::Mul => BINOP_IMUL,
            BinOp::Div => BINOP_IDIV,
            BinOp::Mod => BINOP_IMOD,
            BinOp::Equal => BINOP_IEQ,
            BinOp::NotEqual => BINOP_INE,
            BinOp::LessThan => BINOP_ILT,
            BinOp::LessThanOrEqual => BINOP_ILE,
            BinOp::GreaterThan => BINOP_IGT,
            BinOp::GreaterThanOrEqual => BINOP_IGE,
            BinOp::LShift => BINOP_ASHL,
            BinOp::RShift => BINOP_ASHR,
            BinOp::BitAnd => BINOP_AND,
            BinOp::BitOr => BINOP_OR,
            BinOp::BitXor => BINOP_XOR,
            _ => panic!(),
        },
        Type::Float => match binop {
            BinOp::Add => BINOP_FADD,
            BinOp::Sub => BINOP_FSUB,
            BinOp::Mul => BINOP_FMUL,
            BinOp::Div => BINOP_FDIV,
            BinOp::Equal => BINOP_FEQ,
            BinOp::NotEqual => BINOP_FNE,
            BinOp::LessThan => BINOP_FLT,
            BinOp::LessThanOrEqual => BINOP_FLE,
            BinOp::GreaterThan => BINOP_FGT,
            BinOp::GreaterThanOrEqual => BINOP_FGE,
            _ => panic!(),
        },
        Type::UInt => match binop {
            BinOp::Add => BINOP_IADD,
            BinOp::Sub => BINOP_ISUB,
            BinOp::Mul => BINOP_IMUL,
            BinOp::Div => BINOP_IDIV,
            BinOp::Mod => BINOP_IMOD,
            BinOp::Equal => BINOP_IEQ,
            BinOp::NotEqual => BINOP_INE,
            BinOp::LessThan => BINOP_ILT,
            BinOp::LessThanOrEqual => BINOP_ILE,
            BinOp::GreaterThan => BINOP_IGT,
            BinOp::GreaterThanOrEqual => BINOP_IGE,
            BinOp::LShift => BINOP_LSHL,
            BinOp::RShift => BINOP_LSHR,
            BinOp::BitAnd => BINOP_AND,
            BinOp::BitOr => BINOP_OR,
            BinOp::BitXor => BINOP_XOR,
            _ => panic!(),
        },
        Type::App(TypeCon::Pointer(..), ..) => match binop {
            BinOp::Equal => BINOP_REF_EQ,
            BinOp::NotEqual => BINOP_REF_NE,
            _ => panic!(),
        },
        _ => panic!(),
    }
}

type AllTypes = FxHashMap<SymbolPath, FxHashMap<Unique, TypeCon>>;

#[derive(Debug, PartialEq, Eq, Clone)]
struct Variable {
    loc: u32,
    level: u32,
    is_escaped: bool,
}

struct Generator<'a> {
    module: TypedProgram,
    builder: BytecodeBuilder,
    types: &'a AllTypes,
    vars: HashMapWithScope<Id, Variable>,
    current_level: u32,
    temp_var_num: usize,
}

impl<'a> Generator<'a> {
    fn new(module: TypedProgram, types: &'a AllTypes) -> Self {
        Self {
            builder: BytecodeBuilder::new(&module.module_path),
            module,
            types,
            vars: HashMapWithScope::new(),
            current_level: 0,
            temp_var_num: 0,
        }
    }

    fn gen_temp_var_name(&mut self) -> Id {
        let name = format!("$temp{}", self.temp_var_num);
        let name = IdMap::new_id(&name);
        self.temp_var_num += 1;
        name
    }

    fn expand_name(&self, ty: Type) -> Type {
        match ty {
            Type::App(TypeCon::Named(path, uniq), types) => {
                // TODO: Add support for external module types
                Type::App(self.types[&path][&uniq].clone(), types)
            }
            ty => ty,
        }
    }

    fn expand(&self, ty: Type) -> Type {
        let ty = self.expand_name(ty);
        let ty = expand_all(ty);
        ty
    }

    fn gen_expr_with_copy(&mut self, func: &mut Function, expr: TypedExpr) -> Option<()> {
        let is_lvalue = expr.is_lvalue;
        self.gen_expr(func, expr)?;

        if is_lvalue {
            func.push_noarg(opcode::DEREF);
        }

        Some(())
    }

    fn push_int(&mut self, func: &mut Function, n: impl Into<i64>) -> Result<(), BuilderError> {
        self.push_uint(func, i2u(n.into()))
    }

    fn push_uint(&mut self, func: &mut Function, n: impl Into<u64>) -> Result<(), BuilderError> {
        let id = self.builder.push_int(n.into())?;
        func.push(opcode::INT, id)?;
        Ok(())
    }

    fn gen_var(&mut self, func: &mut Function, var: &Variable, span: &Span) {
        if var.level == self.current_level {
            if var.is_escaped {
                try_b!(func.push(opcode::LOAD_REF_EV, var.loc), span);
            } else {
                try_b!(func.push(opcode::LOAD_REF_LV, var.loc), span);
            }
        } else {
            assert!(var.level < self.current_level);
            let relative_level = self.current_level - var.level;
            try_b!(self.push_uint(func, relative_level), span);
            try_b!(func.push(opcode::LOAD_REF_EV_OUTER, var.loc), span);
        }
    }

    fn gen_convertion(&mut self, _from: &Type, _to: &Type) {}

    // TODO: support in_heap
    fn gen_expr(&mut self, func: &mut Function, expr: TypedExpr) -> Option<()> {
        match expr.kind {
            TExpr::Literal(Literal::Number(n)) => {
                try_b!(self.push_int(func, n), &expr.span);
            }
            TExpr::Literal(Literal::UnsignedNumber(n)) => {
                try_b!(self.push_uint(func, n), &expr.span);
            }
            TExpr::Literal(Literal::Float(n)) => {
                let id = try_b!(self.builder.push_float(n), &expr.span)?;
                try_b!(func.push(opcode::FLOAT, id), &expr.span);
            }
            TExpr::Literal(Literal::String(s)) => {
                let id = try_b!(self.builder.push_string(s), &expr.span)?;
                try_b!(func.push(opcode::STRING, id), &expr.span);
            }
            TExpr::Literal(Literal::Char(ch)) => {
                try_b!(self.push_uint(func, ch as u64), &expr.span);
            }
            TExpr::Literal(Literal::Unit) => {
                // XXX: もしかしたら何かプッシュしないといけないかも
            }
            TExpr::Literal(Literal::True) => {
                func.push_noarg(opcode::TRUE);
            }
            TExpr::Literal(Literal::False) => {
                func.push_noarg(opcode::FALSE);
            }
            TExpr::Literal(Literal::Null) => {
                func.push_noarg(opcode::NULL);
            }
            TExpr::Tuple(exprs) => {
                let exprs_len = exprs.len();
                for expr in exprs {
                    self.gen_expr_with_copy(func, expr);
                }
                try_b!(func.push(opcode::ALLOC, exprs_len as u32), &expr.span);
            }
            TExpr::Struct(_, mut fields) => {
                let fields_order = match &expr.ty {
                    Type::App(TypeCon::Named(path, uniq), ..) => match &self.types[&path][&uniq] {
                        TypeCon::Fun(_, box Type::App(TypeCon::Struct(fields), ..)) => fields,
                        _ => panic!(),
                    },
                    _ => panic!(),
                };

                sort_by_field_order(fields_order, &mut fields);

                let fields_count = fields.len();
                for (_, field_expr) in fields {
                    self.gen_expr_with_copy(func, field_expr);
                }
                try_b!(func.push(opcode::ALLOC, fields_count as u32), &expr.span);
            }
            TExpr::Array(elem_expr, size) => {
                try_b!(self.push_uint(func, size as u64), &expr.span);

                self.gen_expr_with_copy(func, *elem_expr);
                try_b!(func.push(opcode::DUP, size as u32 - 1), &expr.span);
                try_b!(func.push(opcode::ALLOC, size as u32 + 1), &expr.span);
            }
            TExpr::Field(comp_expr, field) => {
                let internal_ty = self.expand(comp_expr.ty.clone());
                let internal_ty = match internal_ty {
                    Type::App(TypeCon::Pointer(..), types) => types[0].clone(),
                    internal_ty => internal_ty,
                };

                if comp_expr.is_lvalue {
                    self.gen_expr(func, *comp_expr);
                } else {
                    self.gen_expr_with_copy(func, *comp_expr);
                }

                match field {
                    Field::Number(n) => {
                        try_b!(func.push(opcode::OFFSET_CONST, n as u32), &expr.span);
                    }
                    Field::Id(name) => {
                        let fields = match internal_ty {
                            Type::App(TypeCon::Struct(fields), ..) => fields,
                            _ => panic!(),
                        };
                        let pos = fields.iter().position(|n| *n == name).unwrap();
                        try_b!(func.push(opcode::OFFSET_CONST, pos as u32), &expr.span);
                    }
                }
            }
            TExpr::Subscript(array_expr, subscript_expr) => {
                if array_expr.is_lvalue {
                    self.gen_expr(func, *array_expr);
                } else {
                    self.gen_expr_with_copy(func, *array_expr);
                }

                self.gen_expr_with_copy(func, *subscript_expr);
                func.push_noarg(opcode::OFFSET);
            }
            TExpr::Range(..) => panic!(),
            TExpr::BinOp(BinOp::Or, lhs, rhs) => {
                let end_l = Label::new();
                let true_l = Label::new();

                self.gen_expr_with_copy(func, *lhs);
                func.push_jump(opcode::JUMP_IF_TRUE, true_l);
                self.gen_expr_with_copy(func, *rhs);
                func.push_jump(opcode::JUMP_IF_TRUE, true_l);
                func.push_noarg(opcode::FALSE);
                func.push_jump(opcode::JUMP, end_l);
                func.set_label_here(true_l);
                func.push_noarg(opcode::TRUE);
                func.set_label_here(end_l);
            }
            TExpr::BinOp(BinOp::And, lhs, rhs) => {
                let end_l = Label::new();
                let false_l = Label::new();

                self.gen_expr_with_copy(func, *lhs);
                func.push_jump(opcode::JUMP_IF_FALSE, false_l);
                self.gen_expr_with_copy(func, *rhs);
                func.push_jump(opcode::JUMP_IF_FALSE, false_l);
                func.push_noarg(opcode::TRUE);
                func.push_jump(opcode::JUMP, end_l);
                func.set_label_here(false_l);
                func.push_noarg(opcode::FALSE);
                func.set_label_here(end_l);
            }
            TExpr::BinOp(binop, lhs, rhs) => {
                self.gen_expr_with_copy(func, *lhs);
                self.gen_expr_with_copy(func, *rhs);
                let op = binop_opcode(&binop, &expr.ty);
                func.push_noarg(op);
            }
            TExpr::Variable(name, ..) => {
                let var = self.vars.get(&name).unwrap().clone();
                self.gen_var(func, &var, &expr.span);
            }
            TExpr::Path(..) => {
                // TODO: sema.rs側でモジュールと関数名を取得しておく
                // let module_path, func_name = ?
            }
            TExpr::Call(func_expr, arg_expr) => {
                let mut arg_exprs = vec![arg_expr];
                let mut current_func_expr = func_expr;
                while let TExpr::Call(func_expr, arg_expr) = current_func_expr.kind {
                    arg_exprs.push(arg_expr);
                    current_func_expr = func_expr;
                }

                while let Some(arg_expr) = arg_exprs.pop() {
                    self.gen_expr_with_copy(func, *arg_expr);
                }
                self.gen_expr_with_copy(func, *current_func_expr);

                func.push_noarg(opcode::CALL_REF);
            }
            TExpr::Dereference(ptr_expr) => {
                self.gen_expr_with_copy(func, *ptr_expr);
            }
            TExpr::Address(inner_expr, ..) => {
                if inner_expr.is_lvalue {
                    self.gen_expr(func, *inner_expr);
                } else {
                    // TODO: エスケープする可能性がある
                    let loc = try_b!(func.define_local_variable(), &inner_expr.span)?;
                    let name = self.gen_temp_var_name();
                    let var = Variable {
                        loc,
                        is_escaped: false,
                        level: self.current_level,
                    };
                    self.vars.insert(name, var.clone());
                    self.gen_var(func, &var, &inner_expr.span);
                }
            }
            TExpr::Negative(inner_expr) => {
                self.gen_expr_with_copy(func, *inner_expr);
                match &expr.ty {
                    Type::Int => func.push_noarg(opcode::INEG),
                    Type::Float => func.push_noarg(opcode::FNEG),
                    _ => panic!(),
                }
            }
            TExpr::Not(inner_expr) => {
                self.gen_expr_with_copy(func, *inner_expr);
                func.push_noarg(opcode::NOT);
            }
            TExpr::Block(block) => {
                self.gen_block(func, block);
            }
            TExpr::If(cond_expr, then_expr, None) => {
                let end_l = Label::new();

                self.gen_expr_with_copy(func, *cond_expr);
                func.push_jump(opcode::JUMP_IF_FALSE, end_l);
                self.gen_expr(func, *then_expr);
                func.push(opcode::POP, 1u32).unwrap();
                func.set_label_here(end_l);
            }
            TExpr::If(cond_expr, then_expr, Some(else_expr)) => {
                let end_l = Label::new();
                let else_l = Label::new();

                self.gen_expr_with_copy(func, *cond_expr);
                func.push_jump(opcode::JUMP_IF_FALSE, else_l);
                self.gen_expr_with_copy(func, *then_expr);
                func.push_jump(opcode::JUMP, end_l);
                func.set_label_here(else_l);
                self.gen_expr_with_copy(func, *else_expr);
                func.set_label_here(end_l);
            }
            TExpr::App(inner_expr, ..) => {
                self.gen_expr_with_copy(func, *inner_expr);
            }
        }

        if let Some(converted_from) = &expr.converted_from {
            self.gen_convertion(converted_from, &expr.ty);
        }

        Some(())
    }

    fn gen_block(&mut self, func: &mut Function, block: TypedBlock) {
        for stmt in block.stmts {
            self.gen_stmt(func, stmt);
        }
        self.gen_expr_with_copy(func, *block.result_expr);

        for func_name in block.function_ids {
            let func = self.module.functions.remove(&func_name).unwrap();
            let span = func.name.span.clone();
            if let Some(func) = self.gen_func(func) {
                try_b!(self.builder.define_function(func_name, func), &span);
            }
        }
    }

    fn gen_stmt(&mut self, func: &mut Function, stmt: Spanned<TypedStmt>) -> Option<()> {
        match stmt.kind {
            TypedStmt::Bind(name, _, init_expr, _, is_escaped, _is_in_heap) => {
                self.gen_expr_with_copy(func, *init_expr);

                let loc = if is_escaped {
                    try_b!(func.define_escaped_variable(), &stmt.span)?
                } else {
                    try_b!(func.define_local_variable(), &stmt.span)?
                };

                self.vars.insert(
                    name,
                    Variable {
                        loc,
                        level: self.current_level,
                        is_escaped,
                    },
                );
            }
            TypedStmt::Expr(expr) => {
                self.gen_expr_with_copy(func, expr);
                try_b!(func.push(opcode::POP, 1u32), &stmt.span);
            }
            TypedStmt::Return(expr) => {
                if let Some(expr) = expr {
                    self.gen_expr_with_copy(func, expr);
                } else {
                    try_b!(self.push_uint(func, 0u32), &stmt.span);
                }
                func.push_noarg(opcode::RETURN);
            }
            TypedStmt::While(cond_expr, body_stmt) => {
                let start_l = Label::new();
                let end_l = Label::new();

                func.set_label_here(start_l);
                self.gen_expr(func, cond_expr);
                func.push_jump(opcode::JUMP_IF_FALSE, end_l);
                self.gen_stmt(func, *body_stmt);
                func.push_jump(opcode::JUMP, start_l);
                func.set_label_here(end_l);
            }
            TypedStmt::Assign(lhs, rhs) => {
                self.gen_expr_with_copy(func, *rhs);
                self.gen_expr(func, lhs);
                func.push_noarg(opcode::STORE_REF);
            }
            TypedStmt::Import(..) => {}
        }

        Some(())
    }

    fn gen_func(&mut self, ast_func: TypedFunction) -> Option<Function> {
        let mut func = Function::new(ast_func.params.len() as u32);

        self.vars.push_scope();

        for param in &ast_func.params {
            let loc = if param.is_escaped {
                try_b!(func.define_escaped_variable(), &ast_func.name.span)
            } else {
                try_b!(func.define_local_variable(), &ast_func.name.span)
            }?;
            self.vars.insert(
                param.name,
                Variable {
                    loc,
                    is_escaped: param.is_escaped,
                    level: self.current_level,
                },
            );
        }

        self.gen_expr_with_copy(&mut func, ast_func.body);
        func.push_noarg(opcode::RETURN);

        self.vars.pop_scope();

        Some(func)
    }

    fn gen(mut self) -> Option<BytecodeBuilder> {
        let main_func = self
            .module
            .functions
            .remove(&*reserved_id::MAIN_FUNC)
            .unwrap();
        let func = self.gen_func(main_func)?;
        self.builder
            .define_function(*reserved_id::MAIN_FUNC, func)
            .unwrap();

        Some(self.builder)
    }
}

pub fn gen_bytecodes(
    modules: FxHashMap<SymbolPath, TypedProgram>,
) -> Option<FxHashMap<SymbolPath, BytecodeBuilder>> {
    let mut all_types: AllTypes = FxHashMap::default();
    for program in modules.values() {
        all_types.insert(program.module_path.clone(), program.all_types.clone());
    }

    let modules_count = modules.len();

    let mut builders = FxHashMap::default();
    for (path, module) in modules {
        assert_eq!(path, module.module_path);
        let generator = Generator::new(module, &all_types);
        if let Some(builder) = generator.gen() {
            builders.insert(path, builder);
        }
    }

    if builders.len() < modules_count {
        None
    } else {
        Some(builders)
    }
}
