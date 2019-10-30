use std::collections::HashMap;

use crate::ty::Type;
use crate::ast::*;
use crate::error::Error;
use crate::span::Span;
use crate::id::{Id, IdMap};
use crate::inst::{Inst, Function, BinOp as IBinOp};

macro_rules! error {
    ($self:ident, $span:expr, $fmt: tt $(,$arg:expr)*) => {
        $self.errors.push(Error::new(&format!($fmt $(,$arg)*), $span));
    };
}

#[derive(Debug)]
pub struct Analyzer {
    functions: HashMap<Id, Function>,
    errors: Vec<Error>,
    main_func_id: Id,
    current_func: Id,
}

impl Analyzer {
    pub fn new(id_map: &mut IdMap) -> Self {
        let main_id = id_map.new_id("$main");
        Self {
            functions: HashMap::new(),
            errors: Vec::new(),
            main_func_id: main_id,
            current_func: main_id, 
        }
    }

    #[inline]
    fn current_func_mut(&mut self) -> &mut Function {
        self.functions.get_mut(&self.current_func).unwrap()
    }

    #[inline]
    fn current_func_imm(&self) -> &Function {
        self.functions.get(&self.current_func).unwrap()
    }

    fn add_error(&mut self, msg: &str, span: Span) {
        self.errors.push(Error::new(msg, span));
    }

    fn walk_expr(&mut self, expr: SpannedTyped<Expr>) -> (Type, Span) {
        let ty = match expr.kind {
            Expr::Literal(Literal::Number(n)) => {
                let func = self.current_func_mut();
                func.insts.push(Inst::Int(n));
                Type::Int
            }
            Expr::Literal(Literal::True) => {
                let func = self.current_func_mut();
                func.insts.push(Inst::True);
                Type::Bool
            },
            Expr::Literal(Literal::False) => {
                let func = self.current_func_mut();
                func.insts.push(Inst::False);
                Type::Bool
            },
            Expr::Variable(name) => {
                let func = self.current_func_mut();
                let (loc, ty) = match func.locals.get(&name) {
                    Some(r) => r,
                    None => {
                        self.add_error("undefined variable", expr.span.clone());
                        return (Type::Invalid, expr.span);
                    },
                };

                func.insts.push(Inst::Load(*loc, 0));

                ty.clone()
            },
            Expr::BinOp(binop, lhs, rhs) => {
                let (lty, _) = self.walk_expr(*lhs);
                let (rty, _) = self.walk_expr(*rhs);

                let func = self.current_func_mut();

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
                func.insts.push(Inst::BinOp(ibinop));

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
                let (return_ty, params) = {
                    // Get the callee function
                    let callee_func = match self.functions.get(&name) {
                        Some(func) => func,
                        None => {
                            self.add_error(&format!("undefined function"), expr.span.clone());
                            return (Type::Invalid, expr.span);
                        },
                    };

                    // Check parameter length
                    if args.len() != callee_func.params.len() {
                        error!(self, expr.span.clone(),
                            "the function takes {} parameters. but got {} arguments.",
                            callee_func.params.len(),
                            args.len());
                        return (callee_func.return_ty.clone(), expr.span);
                    }

                    (callee_func.return_ty.clone(), callee_func.params.clone())
                };

                // Check parameter types
                for (arg, param_ty) in args.into_iter().zip(params.iter()) {
                    let (arg_ty, span) = self.walk_expr(arg);
                    if arg_ty != *param_ty {
                        error!(self, span.clone(), "the parameter type is `{}`. but got `{}` type", param_ty, arg_ty);
                    }
                }

                let func = self.current_func_mut();

                // Insert an instruction
                func.insts.push(Inst::Call(name));

                return_ty
            },
        };

        (ty, expr.span)
    }

    fn walk_stmt(&mut self, stmt: Stmt) {
        match stmt {
            Stmt::Expr(expr) => { self.walk_expr(expr); },
            Stmt::If(cond, stmt) => {
                // Condition
                let (ty, span) = self.walk_expr(cond);
                // Check if condition expression is bool type
                match ty {
                    Type::Bool => {},
                    _ => self.add_error(&format!("expected type `bool` but got type `{}`", ty), span),
                };

                // Insert dummy instruction to jump to else-clause or end
                let func = self.current_func_mut();
                let jump_to_else = func.insts.len();
                func.insts.push(Inst::Int(0));

                // Then-clause
                // Check if then-clause statement is block
                match stmt.kind {
                    Stmt::Block(_) => self.walk_stmt(stmt.kind),
                    _ => self.add_error("expected block statement", stmt.span)
                };

                /* if with_else {
                    // Insert dummy instruction to jump to end
                    let jump_to_end = func.insts.len();
                    func.insts.push(Inst::Int(0));

                    func.insts[jump_to_else] = Inst::JumpIfZero(func.insts.len());

                    // Insert else-clause instructions

                    func.insts[jump_to_end] = Inst::Jump(func.insts.len());
                } else { */

                // Insert instruction to jump to end
                let func = self.current_func_mut();
                func.insts[jump_to_else] = Inst::JumpIfZero(func.insts.len());

                //}
            },
            Stmt::While(cond, stmt) => {
                let func = self.current_func_mut();
                let begin = func.insts.len();

                // Insert condition expression instruction
                let (ty, span) = self.walk_expr(cond);
                // Check if condition expression is bool type
                match ty {
                    Type::Bool => {},
                    _ => self.add_error(&format!("expected type `bool` but got type `{}`", ty), span),
                };

                // Insert dummy instruction to jump to end
                let func = self.current_func_mut();
                let jump_to_end = func.insts.len();
                func.insts.push(Inst::Int(0));

                // Insert body statement instruction
                // Check if body statement is block
                match stmt.kind {
                    Stmt::Block(_) => self.walk_stmt(stmt.kind),
                    _ => self.add_error("expected block statement", stmt.span)
                };

                // Jump to begin
                let func = self.current_func_mut();
                func.insts.push(Inst::Jump(begin));

                // Insert instruction to jump to end
                func.insts[jump_to_end] = Inst::JumpIfZero(func.insts.len());
            },
            Stmt::Block(stmts) => {
                for stmt in stmts {
                    self.walk_stmt(stmt.kind);
                }
            },
            Stmt::Bind(name, expr) => {
                let (ty, _) = self.walk_expr(expr);

                let func = self.current_func_mut();

                let loc = func.stack_size;
                func.locals.insert(name, (loc as isize, ty.clone()));
                func.stack_size += 1;

                func.insts.push(Inst::Save(loc as isize, 0));
            },
            Stmt::Return(expr) => {
                let main_id = self.main_func_id;
                let func = self.current_func_imm();

                if func.name == main_id {
                    return;
                }

                let (ty, span) = self.walk_expr(expr);

                // Check type
                let return_ty = self.current_func_imm().return_ty.clone();
                if ty != return_ty {
                    error!(self, span, "expected `{}` type, but got `{}` type", return_ty, ty);
                }

                let func = self.current_func_mut();
                func.insts.push(Inst::Return);
            },
        }
    }

    fn walk_toplevel(&mut self, toplevel: TopLevel) {
        match toplevel {
            TopLevel::Stmt(stmt) => {
                self.current_func = self.main_func_id;
                self.walk_stmt(stmt.kind);
            },
            TopLevel::Function(name, _, _, stmt) => {
                self.current_func = name;
                self.walk_stmt(stmt.kind);
            },
        }
    }

    fn insert_function_header(&mut self, toplevel: &TopLevel) {
        match toplevel {
            TopLevel::Function(name, params, return_ty, _) => {
                let param_types = params.iter().map(|(_, ty)| ty.clone()).collect();
                let mut func = Function::new(*name, param_types, return_ty.clone());
                let mut loc = -2isize; // 2 is fp and ip
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

        for toplevel in program.top {
            self.walk_toplevel(toplevel.kind);
        }

        if self.errors.len() > 0 {
            Err(self.errors)
        } else {
            Ok(self.functions)
        }
    }
}
