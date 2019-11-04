use std::collections::HashMap;

use crate::ty::Type;
use crate::ast::*;
use crate::error::Error;
use crate::span::{Span, Spanned};
use crate::id::{Id, IdMap};
use crate::inst::{Inst, Function, BinOp as IBinOp};
use crate::stdlib::NativeFuncMap;

macro_rules! error {
    ($self:ident, $span:expr, $fmt: tt $(,$arg:expr)*) => {
        $self.errors.push(Error::new(&format!($fmt $(,$arg)*), $span));
    };
}

#[derive(Debug)]
pub struct Analyzer<'a> {
    stdlib_funcs: &'a NativeFuncMap,
    functions: HashMap<Id, Function>,
    errors: Vec<Error>,
    main_func_id: Id,
    current_func: Id,
    id_map: &'a IdMap,
}

impl<'a> Analyzer<'a> {
    pub fn new(stdlib_funcs: &'a NativeFuncMap, id_map: &'a mut IdMap) -> Self {
        let main_id = id_map.new_id("$main");
        Self {
            stdlib_funcs,
            functions: HashMap::new(),
            errors: Vec::new(),
            main_func_id: main_id,
            current_func: main_id, 
            id_map,
        }
    }

    fn add_error(&mut self, msg: &str, span: Span) {
        self.errors.push(Error::new(msg, span));
    }

    fn walk_expr(&mut self, insts: &mut Vec<Inst>, expr: Spanned<Expr>) -> (Type, Span) {
        let ty = match expr.kind {
            Expr::Literal(Literal::Number(n)) => {
                insts.push(Inst::Int(n));
                Type::Int
            }
            Expr::Literal(Literal::True) => {
                insts.push(Inst::True);
                Type::Bool
            },
            Expr::Literal(Literal::False) => {
                insts.push(Inst::False);
                Type::Bool
            },
            Expr::Variable(name) => {
                let func = &self.functions[&self.current_func];
                let (loc, ty) = match func.locals.get(&name) {
                    Some(r) => r,
                    None => {
                        self.add_error("undefined variable", expr.span.clone());
                        return (Type::Invalid, expr.span);
                    },
                };

                insts.push(Inst::Load(*loc, 0));

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
                };
                insts.push(Inst::BinOp(ibinop));

                // Type check
                if lty != rty {
                    self.add_error(&format!("different types `{}` and `{}`", lty, rty), expr.span.clone());
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
                    _ => {
                        self.add_error(&format!("`{} {} {}` is not possible", lty, binop_symbol, rty), expr.span.clone());
                        Type::Invalid
                    }
                }
            },
            Expr::Call(name, args) => {
                let name_str = self.id_map.name(&name);

                let (return_ty, params, inst) = match self.stdlib_funcs.get(name_str) {
                    Some(func) => {
                        (func.return_ty.clone(), func.params.clone(), Inst::CallNative(func.body.clone(), func.params.len()))
                    },
                    None => {
                        // Get the callee function
                        let callee_func = match self.functions.get(&name) {
                            Some(func) => func,
                            None => {
                                self.add_error(&format!("undefined function"), expr.span.clone());
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
                    if arg_ty != *param_ty {
                        error!(self, span.clone(), "the parameter type is `{}`. but got `{}` type", param_ty, arg_ty);
                    }
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
                // Check if condition expression is bool type
                match ty {
                    Type::Bool => {},
                    _ => self.add_error(&format!("expected type `bool` but got type `{}`", ty), span),
                };

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
                // Check if condition expression is bool type
                match ty {
                    Type::Bool => {},
                    _ => self.add_error(&format!("expected type `bool` but got type `{}`", ty), span),
                };

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
                for stmt in stmts {
                    self.walk_stmt(insts, stmt.kind);
                }
            },
            Stmt::Bind(name, expr) => {
                let (ty, _) = self.walk_expr(insts, expr);

                let current_func = self.functions.get_mut(&self.current_func).unwrap();

                let loc = current_func.stack_size;
                current_func.locals.insert(name, (loc as isize, ty.clone()));
                current_func.stack_size += 1;

                insts.push(Inst::Save(loc as isize, 0));
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
                if ty != current_func.return_ty {
                    error!(self, span, "expected `{}` type, but got `{}` type", current_func.return_ty, ty);
                }

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
            TopLevel::Function(name, _, _, stmt) => {
                self.current_func = name;

                let mut insts = Vec::new();
                self.walk_stmt(&mut insts, stmt.kind);
                self.functions.get_mut(&name).unwrap().insts = insts;
            },
        }
    }

    fn insert_function_header(&mut self, toplevel: &TopLevel) {
        match toplevel {
            TopLevel::Function(name, params, return_ty, _) => {
                let param_types = params.iter().map(|(_, ty)| ty.clone()).collect();
                let mut func = Function::new(*name, param_types, return_ty.clone());
                let mut loc = -3isize; // 2 is fp and ip
                for (id, ty) in params.iter().rev() {
                    loc -= 1;
                    func.locals.insert(*id, (loc, ty.clone()));
                }

                self.functions.insert(*name, func);
            },
            _ => {},
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

        let mut main_insts = Vec::new();
        for toplevel in program.top {
            self.walk_toplevel(&mut main_insts, toplevel.kind);
        }

        self.functions.get_mut(&self.main_func_id).unwrap().insts = main_insts;

        if self.errors.len() > 0 {
            Err(self.errors)
        } else {
            Ok(self.functions)
        }
    }
}
