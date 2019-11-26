use std::collections::HashMap;

use crate::ty::Type;
use crate::ast::*;
use crate::error::Error;
use crate::span::{Span, Spanned};
use crate::id::{Id, IdMap};
use crate::inst::{Inst, Function, NativeFunctionBody, BinOp as IBinOp};
use crate::stdlib::NativeFuncMap;

macro_rules! error {
    ($self:ident, $span:expr, $fmt: tt $(,$arg:expr)*) => {
        $self.errors.push(Error::new(&format!($fmt $(,$arg)*), $span));
    };
}

macro_rules! check_type {
    ($self:ident, $ty1:expr, $ty2:expr, $format:tt, $span:expr) => {
        {
            if $ty1 != Type::Invalid && $ty2 != Type::Invalid {
                if $ty1 != $ty2 {
                    $self.errors.push(Error::new(&format!($format, expected = $ty1, actual = $ty2), $span));
                    false
                } else {
                    true
                }
            } else {
                false
            }
        }
    };
}

#[derive(Debug)]
struct FunctionHeader {
    pub params: Vec<Type>,
    pub return_ty: Type,
}

#[derive(Debug)]
pub struct Analyzer<'a> {
    stdlib_funcs: &'a NativeFuncMap,
    functions: HashMap<Id, Function>,
    function_headers: HashMap<Id, FunctionHeader>,
    variables: Vec<HashMap<Id, (isize, Type)>>,
    errors: Vec<Error>,
    main_func_id: Id,
    current_func: Id,
}

impl<'a> Analyzer<'a> {
    pub fn new(stdlib_funcs: &'a NativeFuncMap) -> Self {
        let main_func_id = IdMap::new_id("$main");

        Self {
            stdlib_funcs,
            functions: HashMap::new(),
            function_headers: HashMap::new(),
            variables: Vec::with_capacity(5),
            errors: Vec::new(),
            main_func_id,
            current_func: main_func_id, 
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

    fn insert_params(&mut self, params: Vec<(Id, Type)>) {
        let last_map = self.variables.last_mut().unwrap();
        let mut loc = -2isize; // fp, ip
        for (id, ty) in params.iter().rev() {
            loc -= ty.size() as isize;
            last_map.insert(*id, (loc, ty.clone()));
        }
    }

    fn new_var(&mut self, id: Id, ty: Type) -> isize {
        let last_map = self.variables.last_mut().unwrap();
        let current_func = self.functions.get_mut(&self.current_func).unwrap();

        current_func.stack_size += ty.size();

        let loc = current_func.stack_size as isize;
        last_map.insert(id, (loc, ty.clone()));

        loc
    }

    fn find_var(&self, id: Id) -> Option<&(isize, Type)> {
        for variables in self.variables.iter().rev() {
            if let Some(var) = variables.get(&id) {
                return Some(var);
            }
        }

        None
    }

    #[allow(unused_variables)]
    fn call_native(name: Id, body: NativeFunctionBody, params: usize) -> Inst {
        #[cfg(debug_assertions)]
        {
            Inst::CallNative(name, body, params)
        }
        #[cfg(not(debug_assertions))]
        {
            Inst::CallNative(body, params)
        }
    }

    fn expr_is_lvalue(expr: &Expr) -> bool {
        match expr {
            Expr::Variable(_) | Expr::Dereference(_) | Expr::Field(_, _) => true,
            _ => false,
        }
    }

    fn walk_expr(&mut self, insts: &mut Vec<Inst>, expr: Spanned<Expr>) -> (Type, Span) {
        let ty = match expr.kind {
            Expr::Literal(Literal::Number(n)) => {
                insts.push(Inst::Int(n));
                Type::Int
            },
            Expr::Literal(Literal::String(s)) => {
                insts.push(Inst::String(s));
                Type::String
            },
            Expr::Literal(Literal::True) => {
                insts.push(Inst::True);
                Type::Bool
            },
            Expr::Literal(Literal::False) => {
                insts.push(Inst::False);
                Type::Bool
            },
            Expr::Tuple(exprs) => {
                let mut types = Vec::new();
                for expr in exprs {
                    let (ty, _) = self.walk_expr(insts, expr);
                    types.push(ty);
                }

                insts.push(Inst::Record(types.len()));

                Type::Tuple(types)
            },
            Expr::Field(tuple_expr, Field::Number(i)) => {
                let (ty, tuple_span) = self.walk_expr(insts, *tuple_expr);

                let types = match ty {
                    Type::Tuple(types) => types,
                    ty => {
                        error!(self, tuple_span.clone(), "expected type `tuple` but got type `{}`", ty);
                        return (Type::Invalid, expr.span);
                    },
                };

                let ty = match types.get(i) {
                    Some(ty) => ty,
                    None => {
                        error!(self, tuple_span.clone(), "error");
                        return (Type::Invalid, expr.span);
                    },
                };

                insts.push(Inst::Field(i));

                ty.clone()
            },
            Expr::Variable(name) => {
                let (loc, ty) = match self.find_var(name) {
                    Some(r) => r,
                    None => {
                        self.add_error("undefined variable", expr.span.clone());
                        return (Type::Invalid, expr.span);
                    },
                };

                insts.push(Inst::Load(*loc));

                ty.clone()
            },
            Expr::BinOp(binop, lhs, rhs) => {
                let (lty, _) = self.walk_expr(insts, *lhs);
                let (rty, _) = self.walk_expr(insts, *rhs);

                // Insert an instruction
                let ibinop = match binop {
                    BinOp::Add => IBinOp::Add,
                    BinOp::Sub => IBinOp::Sub,
                    BinOp::Mul => IBinOp::Mul,
                    BinOp::Div => IBinOp::Div,
                    BinOp::LessThan => IBinOp::LessThan,
                    BinOp::LessThanOrEqual => IBinOp::LessThanOrEqual,
                    BinOp::GreaterThan => IBinOp::GreaterThan,
                    BinOp::GreaterThanOrEqual => IBinOp::GreaterThanOrEqual,
                    BinOp::Equal => IBinOp::Equal,
                    BinOp::NotEqual => IBinOp::NotEqual,
                    BinOp::And => IBinOp::And,
                    BinOp::Or => IBinOp::Or,
                };
                insts.push(Inst::BinOp(ibinop));

                // Type check
                if !check_type!(self, lty, rty, "different types `{expected}` and `{actual}`", expr.span.clone()) {
                    return (lty, expr.span);
                }

                let binop_symbol = binop.to_symbol();
                match (binop, &lty) {
                    (BinOp::Add, Type::Int) => Type::Int,
                    (BinOp::Sub, Type::Int) => Type::Int,
                    (BinOp::Mul, Type::Int) => Type::Int,
                    (BinOp::Div, Type::Int) => Type::Int,
                    (BinOp::Equal, Type::Int) => Type::Bool,
                    (BinOp::NotEqual, Type::Int) => Type::Bool,
                    (BinOp::LessThan, Type::Int) => Type::Bool,
                    (BinOp::LessThanOrEqual, Type::Int) => Type::Bool,
                    (BinOp::GreaterThan, Type::Int) => Type::Bool,
                    (BinOp::GreaterThanOrEqual, Type::Int) => Type::Bool,
                    (BinOp::And, Type::Bool) => Type::Bool,
                    (BinOp::Or, Type::Bool) => Type::Bool,
                    _ => {
                        self.add_error(&format!("`{} {} {}` is not possible", lty, binop_symbol, rty), expr.span.clone());
                        Type::Invalid
                    }
                }
            },
            Expr::Call(name, args) => {
                let name_str = IdMap::name(name);

                let (return_ty, params, inst) = match self.stdlib_funcs.get(&*name_str) {
                    Some(func) => {
                        (func.return_ty.clone(), func.params.clone(), Self::call_native(name, func.body.clone(), func.params.len()))
                    },
                    None => {
                        // Get the callee function
                        let callee_func = match self.function_headers.get(&name) {
                            Some(func) => func,
                            None => {
                                error!(self, expr.span.clone(), "undefined function");
                                return (Type::Invalid, expr.span);
                            },
                        };

                        (callee_func.return_ty.clone(), callee_func.params.clone(), Inst::Call(name))
                    },
                };

                // Check parameter length
                if args.len() != params.len() {
                    error!(self, expr.span.clone(),
                        "the function takes {} parameters. but got {} arguments",
                        params.len(),
                        args.len());
                    return (return_ty, expr.span);
                }

                // Check parameter types
                for (arg, param_ty) in args.into_iter().zip(params.iter()) {
                    let (arg_ty, span) = self.walk_expr(insts, arg);
                    check_type!(self, *param_ty, arg_ty, "the parameter type is `{expected}`. but got `{actual}` type", span.clone()); 
                }

                // Insert an instruction
                insts.push(inst);

                return_ty
            },
            Expr::Address(expr) => {
                if !Self::expr_is_lvalue(&expr.kind) {
                    error!(self, expr.span, "this expression is not lvalue");
                    Type::Invalid
                } else {
                    let (ty, _) = self.walk_expr(insts, *expr);
                    insts.push(Inst::Pointer);

                    Type::Pointer(Box::new(ty))
                }
            },
            Expr::Dereference(expr) => {
                let (ty, span) = self.walk_expr(insts, *expr);
                match ty {
                    Type::Pointer(ty) => {
                        insts.push(Inst::Dereference);
                        *ty
                    }
                    Type::Invalid => Type::Invalid,
                    ty => {
                        error!(self, span, "expected type `pointer` but got type `{}`", ty);
                        Type::Invalid
                    }
                }
            },
        };

        (ty, expr.span)
    }

    fn walk_stmt(&mut self, insts: &mut Vec<Inst>, stmt: Stmt) {
        match stmt {
            Stmt::Expr(expr) => {
                let (ty, _) = self.walk_expr(insts, expr);

                let pop_count = ty.size();
                for _ in 0..pop_count {
                    insts.push(Inst::Pop);
                }
            },
            Stmt::If(cond, stmt, else_stmt) => {
                // Condition
                let (ty, span) = self.walk_expr(insts, cond);
                check_type!(self, Type::Bool, ty, "expected type `{expected}` but got type `{actual}`", span);

                // Insert dummy instruction to jump to else-clause or end
                let jump_to_else = insts.len();
                insts.push(Inst::Int(0));

                // Then-clause
                self.walk_stmt(insts, stmt.kind);

                if let Some(else_stmt) = else_stmt {
                    // Insert dummy instruction to jump to end
                    let jump_to_end = insts.len();
                    insts.push(Inst::Int(0));

                    insts[jump_to_else] = Inst::JumpIfZero(insts.len());

                    // Insert else-clause instructions
                    self.walk_stmt(insts, else_stmt.kind);

                    insts[jump_to_end] = Inst::Jump(insts.len());
                } else {
                    // Insert instruction to jump to end
                    insts[jump_to_else] = Inst::JumpIfZero(insts.len());
                }
            },
            Stmt::While(cond, stmt) => {
                let begin = insts.len();

                // Insert condition expression instruction
                let (ty, span) = self.walk_expr(insts, cond);
                check_type!(self, Type::Bool, ty, "expected type `{expected}` but got type `{actual}`", span);

                // Insert dummy instruction to jump to end
                let jump_to_end = insts.len();
                insts.push(Inst::Int(0));

                // Insert body statement instruction
                self.walk_stmt(insts, stmt.kind);

                // Jump to begin
                insts.push(Inst::Jump(begin));

                // Insert instruction to jump to end
                insts[jump_to_end] = Inst::JumpIfZero(insts.len());
            },
            Stmt::Block(stmts) => {
                self.push_scope();
                for stmt in stmts {
                    self.walk_stmt(insts, stmt.kind);
                }
                self.pop_scope();
            },
            Stmt::Bind(name, expr) => {
                let (ty, _) = self.walk_expr(insts, expr);

                let loc = self.new_var(name, ty.clone());
                insts.push(Inst::Load(loc as isize));
                insts.push(Inst::Store);
            },
            Stmt::Assign(lhs, rhs) => {
                if !Self::expr_is_lvalue(&lhs.kind) {
                    error!(self, lhs.span, "unassignable expression");
                    return;
                }

                let (rhs_ty, rhs_span) = self.walk_expr(insts, rhs);
                let (lhs_ty, _) = self.walk_expr(insts, lhs);

                check_type!(self, lhs_ty, rhs_ty, "expected type `{expected}` but got type `{actual}`", rhs_span);

                insts.push(Inst::Store);
            },
            Stmt::Return(expr) => {
                let main_id = self.main_func_id;

                let (ty, span) = self.walk_expr(insts, expr);

                let func_name = self.functions[&self.current_func].name;
                let return_ty = &self.function_headers[&self.current_func].return_ty;

                // Check if is outside function
                if func_name == main_id {
                    error!(self, span, "return statement outside function");
                    return;
                }

                // Check type
                check_type!(self, *return_ty, ty, "expected `{expected}` type, but got `{actual}` type", span);

                insts.push(Inst::Return);
            },
        }
    }

    fn walk_toplevel(&mut self, main_insts: &mut Vec<Inst>, toplevel: TopLevel) {
        match toplevel {
            TopLevel::Stmt(stmt) => {
                self.current_func = self.main_func_id;
                self.walk_stmt(main_insts, stmt.kind);
            },
            TopLevel::Function(name, params, _, stmt) => {
                self.current_func = name;

                self.push_scope();

                // params
                self.insert_params(params);

                // body
                let mut insts = Vec::new();
                self.walk_stmt(&mut insts, stmt.kind);
                self.functions.get_mut(&name).unwrap().insts = insts;

                self.pop_scope();
            },
        }
    }

    fn insert_function_header(&mut self, toplevel: &TopLevel) {
        if let TopLevel::Function(name, params, return_ty, _) = toplevel {
            let param_types: Vec<Type> = params.iter().map(|(_, ty)| ty.clone()).collect();
            let param_count = param_types.len();

            // Insert a header of the function
            let header = FunctionHeader {
                params: param_types,
                return_ty: return_ty.clone(),
            };
            self.function_headers.insert(*name, header);

            // Insert function
            let func = Function::new(*name, param_count);
            self.functions.insert(*name, func);
        }
    }

    pub fn analyze(mut self, program: Program) -> Result<HashMap<Id, Function>, Vec<Error>> {
        // Insert main function header
        let header = FunctionHeader {
            params: Vec::new(),
            return_ty: Type::Int, // TODO: Type::Unit
        };
        self.function_headers.insert(self.main_func_id, header);

        // Insert main function
        let func = Function::new(self.main_func_id, 0);
        self.functions.insert(self.main_func_id, func);

        // Insert function headers
        for toplevel in program.top.iter() {
            self.insert_function_header(&toplevel.kind);
        }

        self.push_scope();

        let mut main_insts = Vec::new();
        for toplevel in program.top {
            self.walk_toplevel(&mut main_insts, toplevel.kind);
        }

        self.pop_scope();

        self.functions.get_mut(&self.main_func_id).unwrap().insts = main_insts;

        if !self.errors.is_empty() {
            Err(self.errors)
        } else {
            Ok(self.functions)
        }
    }
}
