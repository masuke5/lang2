use std::io::{Read, Write, Seek};
use std::collections::HashMap;

use crate::ty::Type;
use crate::ast::*;
use crate::error::Error;
use crate::span::{Span, Spanned};
use crate::id::{Id, IdMap};
use crate::bytecode::{Function, opcode, BytecodeBuilder, BytecodeStream, InstList};
use crate::module::{FunctionHeader, ModuleHeader};
use crate::translate;

macro_rules! error {
    ($self:ident, $span:expr, $fmt: tt $(,$arg:expr)*) => {
        $self.errors.push(Error::new(&format!($fmt $(,$arg)*), $span));
    };
}

macro_rules! try_some {
    ($($var:ident),*) => {
        $(let $var = $var?;)*
    };
}

macro_rules! fn_to_expect {
    ($fn_name:ident, $type_name:tt, $ty:ty, $pat:pat => $expr:expr,) => {
        fn $fn_name<'a>(errors: &mut Vec<Error>, ty: &'a Type, span: Span) -> Option<&'a $ty> {
            match ty {
                $pat => $expr,
                _ => {
                    let msg = format!(concat!("expected type `", $type_name, "` but got type `{}`"), ty);
                    let error = Error::new(&msg, span);
                    errors.push(error);
                    None
                },
            }
        }
    };
}

fn check_type(errors: &mut Vec<Error>, expected: &Type, actual: &Type, span: Span) -> bool {
    // Don't add an error if type of either `lhs` and `rhs` is invalid
    if *expected == Type::Invalid || *actual == Type::Invalid {
        return false;
    }

    // A null can assign to a pointer
    if let Type::Pointer(_, _) = expected {
        if *actual == Type::Null {
            return true;
        }
    }

    // A mutable pointer can assign to a immutable pointer
    if let Type::Pointer(expected, false) = expected {
        if let Type::Pointer(actual, _) = actual {
            if *expected == *actual {
                return true;
            }
        }
    }

    if expected != actual {
        let error = Error::new(&format!("expected type `{}` but got type `{}`", expected, actual), span);
        errors.push(error);
        false
    } else {
        true
    }
}

fn_to_expect! {
    expect_tuple, "tuple", Vec<Type>,
    Type::Tuple(types) => Some(types),
}

fn_to_expect! {
    expect_struct, "struct", Vec<(Id, Type)>,
    Type::Struct(fields) => Some(fields),
}

// Return size of specified type.
fn type_size(types: &HashMap<Id, Type>, ty: &Type) -> usize {
    match ty {
        Type::Named(id) => {
            types.get(id)
                .map(|ty| type_size(types, ty))
                .unwrap_or(1)
        },
        Type::Tuple(tys) => tys.iter().fold(0, |acc, ty| acc + type_size(types, ty)),
        Type::Struct(fields) => fields.iter().fold(0, |acc, (_, ty)| acc + type_size(types, ty)),
        Type::Array(ty, size) => type_size(types, ty) * size,
        _ => 1,
    }
}

#[derive(Debug)]
struct ExprInfo {
    pub ty: Type,
    pub span: Span,
    pub insts: InstList,
    pub is_lvalue: bool,
    pub is_mutable: bool,
}

impl ExprInfo {
    fn new(insts: InstList, ty: Type, span: Span) -> Self {
        Self {
            ty,
            span,
            insts,
            is_lvalue: false,
            is_mutable: false,
        }
    }

    fn new_lvalue(insts: InstList, ty: Type, span: Span, is_mutable: bool) -> Self {
        Self {
            ty,
            span,
            insts,
            is_lvalue: true,
            is_mutable,
        }
    }
}

#[derive(Debug)]
struct Variable {
    ty: Type,
    is_mutable: bool,
    loc: isize,
}

impl Variable {
    fn new(ty: Type, is_mutable: bool, loc: isize) -> Self {
        Self {
            ty,
            is_mutable,
            loc,
        }
    }
}

#[derive(Debug)]
pub struct Analyzer<'a> {
    function_headers: HashMap<Id, FunctionHeader>,
    types: HashMap<Id, Type>,
    variables: Vec<HashMap<Id, Variable>>,
    errors: Vec<Error>,
    main_func_id: Id,
    return_value_id: Id,
    current_func: Id,
    next_temp_num: u32,
    std_module: ModuleHeader,
    _phantom: &'a std::marker::PhantomData<Self>,
}

impl<'a> Analyzer<'a> {
    pub fn new(std_module: ModuleHeader) -> Self {
        let main_func_id = IdMap::new_id("$main");
        let return_value_id = IdMap::new_id("$rv");

        Self {
            function_headers: HashMap::new(),
            variables: Vec::with_capacity(5),
            types: HashMap::new(),
            errors: Vec::new(),
            main_func_id,
            return_value_id,
            current_func: main_func_id, 
            next_temp_num: 0,
            std_module,
            _phantom: &std::marker::PhantomData,
        }
    }

    fn add_error(&mut self, msg: &str, span: Span) {
        self.errors.push(Error::new(msg, span));
    }

    fn push_scope(&mut self) {
        self.variables.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.variables.pop().unwrap();
    }

    // Insert parameters and return value as variables to `self.variables`
    fn insert_params(&mut self, params: Vec<(Id, Type, bool)>, return_ty: &Type) {
        let last_map = self.variables.last_mut().unwrap();
        let mut loc = -3isize; // fp, ip
        for (id, ty, is_mutable) in params.iter().rev() {
            loc -= type_size(&self.types, ty) as isize;
            last_map.insert(*id, Variable::new(ty.clone(), *is_mutable, loc));
        }

        loc -= type_size(&self.types, return_ty) as isize;
        last_map.insert(self.return_value_id, Variable::new(return_ty.clone(), false, loc));
    }

    fn get_return_var(&self) -> &Variable {
        self.find_var(self.return_value_id).unwrap()
    }

    // ====================================
    //  Variable
    // ====================================

    fn new_var(&mut self, current_func: &mut Function, id: Id, ty: Type, is_mutable: bool) -> isize {
        let last_map = self.variables.last_mut().unwrap();
        let new_var_size = type_size(&self.types, &ty);

        let loc = match last_map.get(&id) {
            // If the same scope contains the same size variable, use the variable location
            Some(var) if new_var_size == type_size(&self.types, &var.ty) => {
                var.loc
            },
            _ => {
                let loc = current_func.stack_size as isize;
                current_func.stack_size += new_var_size as u8;
                loc
            },
        };

        last_map.insert(id, Variable::new(ty.clone(), is_mutable, loc));

        loc
    }

    fn gen_temp_id(&mut self) -> Id {
        let id = IdMap::new_id(&format!("$comp{}", self.next_temp_num));
        self.next_temp_num += 1;
        id
    }

    fn find_var(&self, id: Id) -> Option<&Variable> {
        for variables in self.variables.iter().rev() {
            if let Some(var) = variables.get(&id) {
                return Some(var);
            }
        }

        None
    }

    // Named =>
    //   Struct => ok
    //   _ => error
    // Pointer
    //   Struct => ok
    //   Named =>
    //     Struct => ok
    //     _ => error
    //   _ => error
    // Struct => ok
    // _ => error
    fn get_struct_fields<'b>(&'b mut self, ty: &'b Type, span: &Span, is_mutable: bool) -> Option<(&'b Vec<(Id, Type)>, bool)> {
        match ty {
            Type::Struct(fields) => Some((fields, is_mutable)),
            Type::Named(name) => {
                let ty = match self.types.get(name) {
                    Some(ty) => ty,
                    None => {
                        error!(self, span.clone(), "undefined type");
                        return None;
                    },
                };

                expect_struct(&mut self.errors, ty, span.clone()).map(|fields| (fields, is_mutable))
            },
            Type::Pointer(ty, is_mutable) => self.get_struct_fields(&ty, span, *is_mutable),
            ty => {
                error!(self, span.clone(), "expected type `struct` or `*struct` but got type `{}`", ty);
                None
            },
        }
    }

    // ====================================
    //  Expression
    // ====================================

    // Return true if `walk_expr` passed `expr` may push multiple values
    fn expr_push_multiple_values(expr: &Expr) -> bool {
        match expr {
            // always
            Expr::Tuple(_) | Expr::Struct(_, _) | Expr::Array(_, _) => true,
            // only if the return value is a compound value
            Expr::Call(_, _) => true,
            _ => false,
        }
    }

    fn walk_expr<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, expr: Spanned<Expr>) -> Option<ExprInfo> {
        let (insts, ty) = match expr.kind {
            Expr::Literal(Literal::Number(n)) => {
                (translate::literal_int(n), Type::Int)
            },
            Expr::Literal(Literal::String(i)) => {
                let ty = Type::Pointer(Box::new(Type::String), false);
                (translate::literal_str(i), ty)
            },
            Expr::Literal(Literal::Unit) => {
                (translate::literal_unit(), Type::Unit)
            },
            Expr::Literal(Literal::True) => {
                (translate::literal_true(), Type::Bool)
            },
            Expr::Literal(Literal::False) => {
                (translate::literal_false(), Type::Bool)
            },
            Expr::Literal(Literal::Null) => {
                (translate::literal_null(), Type::Null)
            },
            Expr::Tuple(exprs) => {
                let mut insts = InstList::new();
                let mut types = Vec::new();
                for expr in exprs {
                    let expr = self.walk_expr(code, expr);
                    if let Some(expr) = expr {
                        types.push(expr.ty);
                        insts.append(expr.insts);
                    } else {
                        types.push(Type::Invalid);
                    }
                }

                (insts, Type::Tuple(types))
            },
            Expr::Struct(name, field_exprs) => {
                let ty = self.types.get(&name)?;
                let fields = match &ty {
                    Type::Struct(fields) => fields,
                    ty => {
                        error!(self, expr.span.clone(), "expected struct but got type `{}`", ty);
                        return None;
                    },
                };

                let mut insts = InstList::new();

                // Push instructions to `insts` in order
                for (field_id, field_ty) in fields.clone() {
                    let field_expr = field_exprs.iter().find(|(id, _)| id.kind == field_id);
                    match field_expr {
                        Some((_, expr)) => {
                            // TODO: Avoid clone()
                            if let Some(expr) = self.walk_expr(code, expr.clone()) {
                                insts.append(expr.insts);
                                check_type(&mut self.errors, &field_ty, &expr.ty, expr.span);
                            }
                        },
                        None => {
                            error!(self, expr.span.clone(), "missing field `{}`", IdMap::name(field_id));
                        },
                    }
                }

                (insts, Type::Named(name))
            },
            Expr::Array(expr, size) => {
                let expr = self.walk_expr(code, *expr)?;

                (translate::literal_array(expr.insts, type_size(&self.types, &expr.ty), size), Type::Array(Box::new(expr.ty), size))
            },
            Expr::Field(comp_expr, field) => {
                let should_store = Self::expr_push_multiple_values(&comp_expr.kind);
                let comp_expr = self.walk_expr(code, *comp_expr)?;
                let comp_expr_size = type_size(&self.types, &comp_expr.ty);
                let mut is_mutable = comp_expr.is_mutable;

                let loc = if should_store {
                    let id = self.gen_temp_id();
                    Some(self.new_var(code.current_func_mut(), id, comp_expr.ty.clone(), false))
                } else {
                    None
                };

                let should_deref = match &comp_expr.ty {
                    Type::Pointer(_, _) => true,
                    _ => false,
                };

                // Get the field type and offset
                let (field_ty, offset) = match field {
                    Field::Number(i) => {
                        let types = match &comp_expr.ty {
                            Type::Pointer(ty, is_mutable_) => {
                                is_mutable = *is_mutable_;
                                expect_tuple(&mut self.errors, &ty, comp_expr.span.clone())?
                            },
                            ty => expect_tuple(&mut self.errors, ty, comp_expr.span.clone())?,
                        };
                        
                        match types.get(i) {
                            Some(ty) => {
                                let offset = types.iter().take(i).fold(0, |acc, ty| acc + type_size(&self.types, ty));
                                (ty.clone(), offset)
                            },
                            None => {
                                error!(self, expr.span, "error");
                                return None;
                            },
                        }
                    },
                    Field::Id(name) => {
                        let (fields, is_mutable_) = self.get_struct_fields(&comp_expr.ty, &comp_expr.span, is_mutable)?;
                        let fields = fields.clone();
                        is_mutable = is_mutable_;

                        let i = match fields.iter().position(|(id, _)| *id == name) {
                            Some(i) => i,
                            None => {
                                error!(self, expr.span, "no field in `{}`: `{}`", comp_expr.ty, IdMap::name(name));
                                return None;
                            },
                        };

                        let offset = fields.iter().take(i).fold(0, |acc, (_, ty)| acc + type_size(&self.types, ty));
                        (fields[i].1.clone(), offset)
                    }
                };

                let insts = translate::field(loc, should_deref, comp_expr.insts, comp_expr_size, offset);
                let ty = field_ty.clone();
                return Some(ExprInfo::new_lvalue(insts, ty, expr.span, is_mutable));
            },
            Expr::Subscript(expr, subscript_expr) => {
                let should_store = Self::expr_push_multiple_values(&expr.kind);

                let expr = self.walk_expr(code, *expr);
                let subscript_expr = self.walk_expr(code, *subscript_expr);
                try_some!(expr, subscript_expr);

                let mut expr = expr;

                let loc = if should_store {
                    let id = self.gen_temp_id();
                    Some(self.new_var(code.current_func_mut(), id, expr.ty.clone(), false))
                } else {
                    None
                };

                let (ty, should_deref) = match expr.ty.clone() {
                    Type::Array(ty, _) => (*ty, false),
                    Type::Pointer(ty, is_mutable) => {
                        expr.is_mutable = is_mutable;

                        match *ty {
                            Type::Array(ty, _) => (*ty, true),
                            ty => {
                                error!(self, expr.span.clone(), "expected array but got type `{}`", ty);
                                return None;
                            },
                        }
                    }
                    ty => {
                        error!(self, expr.span.clone(), "expected array but got type `{}`", ty);
                        return None;
                    },
                };

                check_type(&mut self.errors, &Type::Int, &subscript_expr.ty, subscript_expr.span);

                expr.insts = translate::subscript(
                    loc,
                    should_deref,
                    expr.insts,
                    type_size(&self.types, &expr.ty),
                    subscript_expr.insts,
                    type_size(&self.types, &subscript_expr.ty),
                );
                expr.ty = ty;
                return Some(expr);
            },
            Expr::Variable(name) => {
                let var = match self.find_var(name) {
                    Some(v) => v,
                    None => {
                        self.add_error("undefined variable", expr.span.clone());
                        return None;
                    },
                };

                let insts = translate::variable(var.loc);
                return Some(ExprInfo::new_lvalue(insts, var.ty.clone(), expr.span, var.is_mutable));
            },
            Expr::BinOp(BinOp::And, lhs, rhs) => {
                let lhs = self.walk_expr(code, *lhs);
                let rhs = self.walk_expr(code, *rhs);
                try_some!(lhs, rhs);

                // Type check
                match (&lhs.ty, &rhs.ty) {
                    (Type::Bool, Type::Bool) => {},
                    (Type::Invalid, _) | (_, Type::Invalid) => {},
                    (lty, rty) => {
                        error!(self, expr.span.clone(), "{} && {}", lty, rty);
                    },
                }

                let lhs_size = type_size(&self.types, &lhs.ty);
                let rhs_size = type_size(&self.types, &rhs.ty);
                (translate::binop_and(lhs.insts, lhs_size, rhs.insts, rhs_size), Type::Bool)
            },
            Expr::BinOp(BinOp::Or, lhs, rhs) => {
                let lhs = self.walk_expr(code, *lhs);
                let rhs = self.walk_expr(code, *rhs);
                try_some!(lhs, rhs);

                // Type check
                match (&lhs.ty, &rhs.ty) {
                    (Type::Bool, Type::Bool) => {},
                    (Type::Invalid, _) | (_, Type::Invalid) => {},
                    (lty, rty) => {
                        error!(self, expr.span.clone(), "{} || {}", lty, rty);
                    },
                }

                let lhs_size = type_size(&self.types, &lhs.ty);
                let rhs_size = type_size(&self.types, &rhs.ty);
                (translate::binop_or(lhs.insts, lhs_size, rhs.insts, rhs_size), Type::Bool)
            },
            Expr::BinOp(binop, lhs, rhs) => {
                let lhs = self.walk_expr(code, *lhs);
                let rhs = self.walk_expr(code, *rhs);
                try_some!(lhs, rhs);

                let binop_symbol = binop.to_symbol();
                let ty = match (&binop, &lhs.ty, &rhs.ty) {
                    (BinOp::Add, Type::Int, Type::Int) => Type::Int,
                    (BinOp::Sub, Type::Int, Type::Int) => Type::Int,
                    (BinOp::Mul, Type::Int, Type::Int) => Type::Int,
                    (BinOp::Div, Type::Int, Type::Int) => Type::Int,
                    (BinOp::Equal, Type::Int, Type::Int) => Type::Bool,
                    (BinOp::NotEqual, Type::Int, Type::Int) => Type::Bool,
                    (BinOp::LessThan, Type::Int, Type::Int) => Type::Bool,
                    (BinOp::LessThanOrEqual, Type::Int, Type::Int) => Type::Bool,
                    (BinOp::GreaterThan, Type::Int, Type::Int) => Type::Bool,
                    (BinOp::GreaterThanOrEqual, Type::Int, Type::Int) => Type::Bool,

                    (BinOp::Equal, Type::Pointer(_, _), Type::Pointer(_, _)) => Type::Bool,
                    (BinOp::Equal, Type::Null, Type::Pointer(_, _)) => Type::Bool,
                    (BinOp::Equal, Type::Pointer(_, _), Type::Null) => Type::Bool,
                    (BinOp::NotEqual, Type::Pointer(_, _), Type::Pointer(_, _)) => Type::Bool,
                    (BinOp::NotEqual, Type::Null, Type::Pointer(_, _)) => Type::Bool,
                    (BinOp::NotEqual, Type::Pointer(_, _), Type::Null) => Type::Bool,
                    _ => {
                        self.add_error(&format!("`{} {} {}`", lhs.ty, binop_symbol, rhs.ty), expr.span.clone());
                        Type::Invalid
                    }
                };

                let lhs_size = type_size(&self.types, &lhs.ty);
                let rhs_size = type_size(&self.types, &rhs.ty);
                (translate::binop(binop, lhs.insts, lhs_size, rhs.insts, rhs_size), ty)
            },
            Expr::Call(name, args) => {
                let (return_ty, params, code_id, module_id) = {
                    // Get the callee function
                    let (callee_func, code_id, module_id) = match self.function_headers.get(&name) {
                        Some(func) => (func, code.get_function(name).unwrap().code_id, None),
                        None => {
                            if let Some((id, func)) = self.std_module.find_func(name) {
                                (func, *id, Some(0))
                            } else {
                                error!(self, expr.span.clone(), "undefined function");
                                return None;
                            }
                        },
                    };

                    (
                        callee_func.return_ty.clone(),
                        callee_func.params.clone(),
                        code_id,
                        module_id,
                    )
                };

                // Check parameter length
                if args.len() != params.len() {
                    error!(self, expr.span.clone(),
                        "the function takes {} parameters. but got {} arguments",
                        params.len(),
                        args.len());
                    return None;
                }

                let mut insts = Vec::new();
                for (arg, param_ty) in args.into_iter().zip(params.iter()) {
                    let arg = self.walk_expr(code, arg);
                    if let Some(arg) = arg {
                        insts.push((arg.insts, type_size(&self.types, &arg.ty)));
                        check_type(&mut self.errors, &param_ty, &arg.ty, arg.span.clone());
                    }
                }

                let insts = translate::call(code_id, module_id, insts, type_size(&self.types, &return_ty));
                (insts, return_ty)
            },
            Expr::Address(expr, is_mutable) => {
                let expr = self.walk_expr(code, *expr)?;

                if !expr.is_lvalue {
                    error!(self, expr.span, "this expression is not lvalue");
                    return None;
                } else if is_mutable && !expr.is_mutable {
                    error!(self, expr.span, "this expression is immutable");
                    return None;
                } else {
                    let insts = translate::address(expr.insts);
                    (insts, Type::Pointer(Box::new(expr.ty), is_mutable))
                }
            },
            Expr::Dereference(expr) => {
                let expr = self.walk_expr(code, *expr)?;
                let expr_size = type_size(&self.types, &expr.ty);

                match expr.ty {
                    Type::Pointer(ty, is_mutable) => {
                        let insts = translate::dereference(expr.insts, expr_size);
                        return Some(ExprInfo::new_lvalue(insts, *ty, expr.span, is_mutable));
                    }
                    Type::Invalid => return None,
                    ty => {
                        error!(self, expr.span, "expected type `pointer` but got type `{}`", ty);
                        return None;
                    }
                }
            },
            Expr::Negative(expr) => {
                let expr = self.walk_expr(code, *expr)?;
                let expr_size = type_size(&self.types, &expr.ty);

                match expr.ty {
                    ty @ Type::Int /* | Type::Float */ => {
                        (translate::negative(expr.insts, expr_size), ty)
                    },
                    ty => {
                        error!(self, expr.span, "expected type `int` or `float` but got type `{}`", ty);
                        return None;
                    },
                }
            },
            Expr::Alloc(expr, is_mutable) => {
                let expr = self.walk_expr(code, *expr)?;

                let insts = translate::alloc(expr.insts, type_size(&self.types, &expr.ty));
                (insts, Type::Pointer(Box::new(expr.ty), is_mutable))
            },
        };

        Some(ExprInfo::new(insts, ty, expr.span))
    }

    fn walk_stmt<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, stmt: Spanned<Stmt>) -> Option<InstList> {
        let insts = match stmt.kind {
            Stmt::Expr(expr) => {
                let expr = self.walk_expr(code, expr)?;

                translate::expr_stmt(expr.insts, type_size(&self.types, &expr.ty))
            },
            Stmt::If(cond, stmt, None) => {
                let cond = self.walk_expr(code, cond);
                let then_insts = self.walk_stmt(code, *stmt);
                try_some!(cond, then_insts);

                check_type(&mut self.errors, &Type::Bool, &cond.ty, cond.span);

                translate::if_stmt(cond.insts, type_size(&self.types, &cond.ty), then_insts)
            },
            Stmt::If(cond, then_stmt, Some(else_stmt)) => {
                let cond = self.walk_expr(code, cond);
                let then = self.walk_stmt(code, *then_stmt);
                let els = self.walk_stmt(code, *else_stmt);
                try_some!(cond, then, els);

                check_type(&mut self.errors, &Type::Bool, &cond.ty, cond.span);

                translate::if_else_stmt(cond.insts, type_size(&self.types, &cond.ty), then, els)
            },
            Stmt::While(cond, stmt) => {
                let cond = self.walk_expr(code, cond);
                let body = self.walk_stmt(code, *stmt);
                try_some!(cond, body);

                check_type(&mut self.errors, &Type::Bool, &cond.ty, cond.span);

                translate::while_stmt(cond.insts, type_size(&self.types, &cond.ty), body)
            },
            Stmt::Block(stmts) => {
                self.push_scope();
                let mut insts = InstList::new();
                for stmt in stmts {
                    if let Some(t) = self.walk_stmt(code, stmt) {
                        insts.append(t);
                    }
                }
                self.pop_scope();

                insts
            },
            Stmt::Bind(name, expr, is_mutable) => {
                let expr = self.walk_expr(code, expr)?;
                let loc = self.new_var(code.current_func_mut(), name, expr.ty.clone(), is_mutable);

                translate::bind_stmt(loc, expr.insts, type_size(&self.types, &expr.ty))
            },
            Stmt::Assign(lhs, rhs) => {
                let lhs = self.walk_expr(code, lhs);
                let rhs = self.walk_expr(code, rhs);
                try_some!(lhs, rhs);

                if !lhs.is_lvalue {
                    error!(self, lhs.span, "unassignable expression");
                    return None;
                }

                if !lhs.is_mutable {
                    error!(self, lhs.span, "immutable expression");
                    return None;
                }

                check_type(&mut self.errors, &lhs.ty, &rhs.ty, rhs.span);

                translate::assign_stmt(lhs.insts, rhs.insts, type_size(&self.types, &rhs.ty))
            },
            Stmt::Return(expr) => {
                let func_name = code.current_func().name;

                // Check if is outside function
                if func_name == self.main_func_id {
                    error!(self, stmt.span, "return statement outside function");
                    return None;
                }

                let expr = match expr {
                    Some(expr) => Some(self.walk_expr(code, expr)?),
                    None => None,
                };

                // Check type
                let return_var = self.find_var(self.return_value_id).unwrap();
                let ty = return_var.ty.clone();
                let loc = return_var.loc;

                let return_ty = expr.as_ref().map_or(&Type::Unit, |expr| &expr.ty);
                check_type(&mut self.errors, &ty, return_ty, stmt.span);

                translate::return_stmt(loc, expr.map(|expr| (expr.insts, type_size(&self.types, &expr.ty))))
            },
        };

        Some(insts)
    }

    // Check if specified type exists
    fn walk_type(&mut self, ty: &Type, span: &Span) {
        match ty {
            Type::Named(name) => {
                if !self.types.contains_key(name) {
                    error!(self, span.clone(), "undefined type `{}`", IdMap::name(*name));
                }
            },
            Type::Struct(fields) => {
                for (_, ty) in fields {
                    self.walk_type(ty, span);
                }
            },
            Type::Tuple(types) => {
                for ty in types {
                    self.walk_type(ty, span);
                }
            },
            Type::Array(ty, _) => {
                self.walk_type(ty, span);
            },
            Type::Pointer(ty, _) => self.walk_type(ty, span),
            Type::Int | Type::Bool | Type::String | Type::Unit | Type::Invalid | Type::Null => {},
        }
    }

    fn walk_toplevel<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, toplevel: Spanned<TopLevel>) {
        match toplevel.kind {
            TopLevel::Function(name, params, return_ty, stmt) => {
                self.current_func = name;

                self.push_scope();

                // params
                self.insert_params(params, &return_ty);

                code.begin_function(name);
                // `None` is not returned because `stmt` is always a block statement
                let mut insts = self.walk_stmt(code, stmt).unwrap();

                // Push a return instruction if the return value type is unit
                let return_var = self.get_return_var();
                if let Type::Unit = return_var.ty {
                    insts.push_inst_noarg(opcode::RETURN);
                }

                code.end_function(name, insts);

                self.pop_scope();
            },
            TopLevel::Type(_, ty) => {
                self.walk_type(&ty, &toplevel.span);
            },
            _ => {},
        }
    }

    fn walk_main_stmt<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, toplevel: Spanned<TopLevel>) -> Option<InstList> {
        if let TopLevel::Stmt(stmt) = toplevel.kind {
            self.walk_stmt(code, stmt)
        } else {
            None
        }
    }

    fn insert_function_header<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, toplevel: &TopLevel) {
        match toplevel {
            TopLevel::Function(name, params, return_ty, _) => {
                let param_types: Vec<Type> = params.iter().map(|(_, ty, _)| ty.clone()).collect();
                let param_size = param_types.iter().fold(0, |acc, ty| acc + type_size(&self.types, ty));

                // Insert a header of the function
                let header = FunctionHeader {
                    params: param_types,
                    return_ty: return_ty.clone(),
                };
                self.function_headers.insert(*name, header);

                // Insert function
                let func = Function::new(*name, param_size);
                code.new_function(func);
            },
            TopLevel::Type(name, ty) => {
                self.types.insert(*name, ty.clone());
            },
            _ => {},
        }
    }

    pub fn analyze<W: Read + Write + Seek>(mut self, code: W, mut program: Program) -> Result<BytecodeStream<W>, Vec<Error>> {
        let mut code = BytecodeBuilder::new(BytecodeStream::new(code), &program.strings, &[&self.std_module]);

        // Insert main function header
        let header = FunctionHeader {
            params: Vec::new(),
            return_ty: Type::Unit,
        };
        self.function_headers.insert(self.main_func_id, header);

        // Insert main function
        let func = Function::new(self.main_func_id, 0);
        code.new_function(func);

        // Insert function headers
        for toplevel in program.top.iter() {
            self.insert_function_header(&mut code, &toplevel.kind);
        }
        code.end_new_function();

        self.push_scope();

        for toplevel in program.top.drain_filter(|toplevel| match toplevel.kind { TopLevel::Stmt(_) => false, _ => true }) {
            self.walk_toplevel(&mut code, toplevel);
        }

        code.begin_function(self.main_func_id);
        let mut insts = InstList::new();
        for toplevel in program.top {
            if let Some(stmt_insts) = self.walk_main_stmt(&mut code, toplevel) {
                insts.append(stmt_insts);
            }
        }
        code.end_function(self.main_func_id, insts);

        self.pop_scope();

        if !self.errors.is_empty() {
            Err(self.errors)
        } else {
            Ok(code.build())
        }
    }
}
