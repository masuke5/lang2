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
pub struct Analyzer<'a> {
    stdlib_funcs: &'a NativeFuncMap,
    functions: HashMap<Id, Function>,
    variables: Vec<HashMap<Id, (isize, Type)>>,
    errors: Vec<Error>,
    main_func_id: Id,
    temp_var_id: Id,
    current_func: Id,
}

impl<'a> Analyzer<'a> {
    pub fn new(stdlib_funcs: &'a NativeFuncMap) -> Self {
        let main_func_id = IdMap::new_id("$main");
        let temp_var_id = IdMap::new_id("$temp");

        Self {
            stdlib_funcs,
            functions: HashMap::new(),
            variables: Vec::new(),
            errors: Vec::new(),
            main_func_id,
            temp_var_id,
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
        let mut loc = -3isize; // fp, ip
        for (id, ty) in params.iter().rev() {
            loc -= ty.size() as isize;
            last_map.insert(*id, (loc, ty.clone()));
        }
    }

    fn new_var(&mut self, id: Id, ty: Type) -> isize {
        let last_map = self.variables.last_mut().unwrap();
        let current_func = self.functions.get_mut(&self.current_func).unwrap();

        let loc = current_func.stack_size as isize;
        last_map.insert(id, (loc, ty.clone()));

        current_func.stack_size += ty.size();

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

    fn temp_var(&mut self) -> isize {
        match self.find_var(self.temp_var_id) {
            Some((loc, _)) => *loc,
            None => self.new_var(self.temp_var_id, Type::Int),
        }
    }

    fn call_native(name: Id, body: NativeFunctionBody, params: usize) -> Inst {
        Inst::CallNative(name, body, params)
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

                Type::Tuple(types)
            },
            Expr::Field(expr, field) => {
                match field {
                    Field::Number(i) => {
                        type GenFunc = Box<dyn FnMut(&mut Vec<Inst>, &Vec<Type>)>;
                        let (ty, span, mut gen): (Type, Span, GenFunc) = match expr.kind {
                            Expr::Variable(name) => {
                                let (loc, ty) = match self.find_var(name) {
                                    Some(r) => r,
                                    None => {
                                        self.add_error("undefined variable", expr.span.clone());
                                        return (Type::Invalid, expr.span);
                                    },
                                };
                                let loc = *loc;

                                (ty.clone(), expr.span.clone(), Box::new(move |insts, _| {
                                    insts.push(Inst::Load(loc, i));
                                }))
                            },
                            _ => {
                                let (ty, span) = self.walk_expr(insts, *expr);
                                let temp_var_loc = self.temp_var();

                                (ty.clone(), span, Box::new(move |insts, types| {
                                    let pop_count = types.len() - i - 1;
                                    for _ in 0..pop_count {
                                        insts.push(Inst::Pop);
                                    }

                                    if i > 0 {
                                        insts.push(Inst::Save(temp_var_loc, 0));

                                        let pop_count = i;
                                        for _ in 0..pop_count {
                                            insts.push(Inst::Pop);
                                        }

                                        insts.push(Inst::Load(temp_var_loc, 0));
                                    }
                                }))
                            },
                        };
                        
                        match ty {
                            Type::Tuple(types) => {
                                if let Some(ty) = types.get(i) {
                                    gen(insts, &types);
                                    ty.clone()
                                } else {
                                    error!(self, span, "error");
                                    Type::Invalid
                                }
                            },
                            ty => {
                                error!(self, span, "expected type `tuple` but got type `{}`", ty);
                                Type::Invalid
                            },
                        }
                    },
                }
            },
            Expr::Variable(name) => {
                let (loc, ty) = match self.find_var(name) {
                    Some(r) => r,
                    None => {
                        self.add_error("undefined variable", expr.span.clone());
                        return (Type::Invalid, expr.span);
                    },
                };

                for i in 0..ty.size() {
                    insts.push(Inst::Load(*loc, i));
                }

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
                let name_str = IdMap::name(&name);

                let (return_ty, params, inst) = match self.stdlib_funcs.get(&*name_str) {
                    Some(func) => {
                        (func.return_ty.clone(), func.params.clone(), Self::call_native(name, func.body.clone(), func.params.len()))
                    },
                    None => {
                        // Get the callee function
                        let callee_func = match self.functions.get(&name) {
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
        };

        (ty, expr.span)
    }

    fn walk_stmt(&mut self, insts: &mut Vec<Inst>, stmt: Stmt) {
        match stmt {
            Stmt::Expr(expr) => {
                self.walk_expr(insts, expr);
                insts.push(Inst::Pop);
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

                match ty {
                    Type::Tuple(types) => {
                        for i in 0..types.len() {
                            let offset = types.len() - i - 1;
                            insts.push(Inst::Save(loc as isize, offset));
                        }
                    },
                    _ => insts.push(Inst::Save(loc as isize, 0)),
                };
            },
            Stmt::Return(expr) => {
                let main_id = self.main_func_id;

                let (ty, span) = self.walk_expr(insts, expr);

                let current_func = &self.functions[&self.current_func];

                // Check if is outside function
                if current_func.name == main_id {
                    error!(self, span, "return statement outside function");
                    return;
                }

                // Check type
                check_type!(self, current_func.return_ty, ty, "expected `{expected}` type, but got `{actual}` type", span);

                insts.push(Inst::Return(current_func.return_ty.size()));
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
            let param_types = params.iter().map(|(_, ty)| ty.clone()).collect();
            let func = Function::new(*name, param_types, return_ty.clone());

            self.functions.insert(*name, func);
        }
    }

    pub fn analyze(mut self, program: Program) -> Result<HashMap<Id, Function>, Vec<Error>> {
        // self.functions.insert(id_map.new_id("printi"), Function {
        //     args: vec![(id_map.new_id("n"), Type::Int)],
        //     return_ty: Type::Int,
        // });
        // self.functions.insert(id_map.new_id("printlf"), Function {
        //     args: vec![],
        //     return_ty: Type::Int,
        // });

        // Insert main function header
        let main_func = Function::new(self.main_func_id, Vec::new(), Type::Int);
        self.functions.insert(self.main_func_id, main_func);

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
