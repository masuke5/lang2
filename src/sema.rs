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
    current_func: Id,

    tuple_var_id: Option<Id>,
    should_push_tuple: bool,
    next_tuple_id_num: u32,
}

impl<'a> Analyzer<'a> {
    pub fn new(stdlib_funcs: &'a NativeFuncMap) -> Self {
        let main_func_id = IdMap::new_id("$main");

        Self {
            stdlib_funcs,
            functions: HashMap::new(),
            variables: Vec::new(),
            errors: Vec::new(),
            main_func_id,
            current_func: main_func_id, 
            tuple_var_id: None,
            next_tuple_id_num: 0,
            should_push_tuple: false,
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

    fn gen_tuple_id(&mut self) -> Id {
        let id = IdMap::new_id(&format!("$tuple{}", self.next_tuple_id_num));
        self.next_tuple_id_num += 1;
        id
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

    fn call_native(name: Id, body: NativeFunctionBody, params: usize) -> Inst {
        Inst::CallNative(name, body, params)
    }

    fn expect_tuple<'b>(&mut self, ty: &'b Type, span: &Span) -> Option<&'b [Type]> {
        match ty {
            Type::Tuple(types) => Some(types),
            ty => {
                error!(self, span.clone(), "expected type `tuple` but got type `{}`", ty);
                None
            },
        }
    }

    fn check_tuple_index<'b>(&mut self, types: &'b [Type], i: usize, span: &Span) -> Option<&'b Type> {
        if let Some(ty) = types.get(i) {
            Some(ty)
        } else {
            error!(self, span.clone(), "error");
            None
        }
    }

    fn field_offset(&mut self, insts: &mut Vec<Inst>, tuple_expr: Spanned<Expr>, i: usize) -> Option<(Type, usize)> {
        let span = tuple_expr.span.clone();
        let (ty, offset) = match tuple_expr.kind {
            Expr::Field(expr, Field::Number(j)) => {
                self.field_offset(insts, *expr, j)?
            },
            Expr::Variable(name) => {
                self.tuple_var_id = Some(name);

                let (_, ty) = match self.find_var(name) {
                    Some(r) => r,
                    None => {
                        self.add_error("undefined variable", span);
                        return None;
                    },
                };

                (ty.clone(), 0)
            },
            _ => {
                self.should_push_tuple = false;
                self.tuple_var_id = None;
                let (ty, _) = self.walk_expr(insts, tuple_expr);
                (ty, 0)
            }
        };
        
        let types = self.expect_tuple(&ty, &span)?; // TODO: Fix span
        let ty = self.check_tuple_index(types, i, &span)?;

        let next_offset = types.iter().take(i).fold(0, |acc, ty| acc + ty.size());
        Some((ty.clone(), offset + next_offset))
    }

    // Convert tuple expression to instructions and return save count
    fn tuple(
        &mut self,
        insts: &mut Vec<Inst>,
        types: &mut Vec<Type>,
        exprs: Vec<Spanned<Expr>>,
    ) -> usize {
        let mut save_count = 0;

        for expr in exprs {
            match expr.kind {
                Expr::Tuple(exprs) => {
                    let mut tuple_types = Vec::new();
                    save_count += self.tuple(insts, &mut tuple_types, exprs);
                    types.push(Type::Tuple(tuple_types));
                },
                _ => {
                    let (ty, _) = self.walk_expr(insts, expr);
                    save_count += ty.size();
                    types.push(ty);
                },
            }
        }

        save_count
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
                let save_count = self.tuple(insts, &mut types, exprs);
                let ty = Type::Tuple(types);

                if !self.should_push_tuple {
                    // save to memory
                    let id = self.tuple_var_id.unwrap_or(self.gen_tuple_id());
                    let loc = self.find_var(id)
                        .map(|(loc, _)| *loc)
                        .unwrap_or_else(|| self.new_var(id, ty.clone()));

                    for offset in (0..save_count).rev() {
                        insts.push(Inst::Save(loc, offset));
                    }

                    self.tuple_var_id = Some(id);
                }

                ty
            },
            Expr::Field(tuple_expr, Field::Number(i)) => {
                // insert instructions and get field offset
                let (ty, offset) = match self.field_offset(insts, *tuple_expr, i) {
                    Some(t) => t,
                    None => return (Type::Invalid, expr.span),
                };

                // read from memory
                let (loc, _) = self.find_var(self.tuple_var_id.unwrap()).unwrap();
                for offset in offset..offset + ty.size() {
                    insts.push(Inst::Load(*loc, offset));
                }

                ty
            },
            Expr::Variable(name) => {
                self.tuple_var_id = Some(name);

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
                    self.should_push_tuple = true;
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
                self.should_push_tuple = true;
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
                self.tuple_var_id = Some(name);
                self.should_push_tuple = false;
                let (ty, _) = self.walk_expr(insts, expr);
                self.tuple_var_id = None;

                match ty {
                    Type::Tuple(_) => {},
                    _ => {
                        let loc = self.new_var(name, ty.clone());
                        insts.push(Inst::Save(loc as isize, 0));
                    }
                };
            },
            Stmt::Assign(Spanned { kind: Expr::Variable(id), span: var_span }, rhs) => {
                let (loc, var_ty) = match self.find_var(id) {
                    Some(t) => t.clone(),
                    None => {
                        error!(self, var_span, "undefined variable");
                        return;
                    },
                };

                self.tuple_var_id = Some(id);
                self.should_push_tuple = false;
                let (rhs_ty, rhs_span) = self.walk_expr(insts, rhs);

                check_type!(self, var_ty, rhs_ty, "expected type `{expected}` but got type `{actual}`", rhs_span);

                match rhs_ty {
                    Type::Tuple(_) => {},
                    ty => {
                        for offset in (0..ty.size()).rev() {
                            insts.push(Inst::Save(loc, offset));
                        }
                    },
                }
            },
            Stmt::Assign(Spanned { kind: Expr::Field(expr, Field::Number(i)), .. }, rhs) => {
                // insert instructions and get field offset
                let (var_ty, offset) = match self.field_offset(insts, *expr, i) {
                    Some(t) => t,
                    None => return,
                };

                let (rhs_ty, rhs_span) = self.walk_expr(insts, rhs);

                check_type!(self, var_ty, rhs_ty, "expected type `{expected}` but got type `{actual}`", rhs_span);

                match rhs_ty {
                    Type::Tuple(_) => {},
                    ty => {
                        let (loc, _) = self.find_var(self.tuple_var_id.unwrap()).unwrap();
                        for offset in offset..offset + ty.size() {
                            insts.push(Inst::Save(*loc, offset));
                        }
                    },
                }
            }
            Stmt::Assign(Spanned { span, .. }, _) => {
                error!(self, span, "unassignable expression");
            },
            Stmt::Return(expr) => {
                let main_id = self.main_func_id;

                self.should_push_tuple = true;
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
