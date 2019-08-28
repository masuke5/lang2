use std::collections::HashMap;

use crate::ty::Type;
use crate::ast::*;
use crate::error::Error;
use crate::span::Span;

#[derive(Debug, Clone)]
struct Function<'a> {
    args: Vec<(&'a str, Type)>,
    return_ty: Type,
}

#[derive(Debug)]
pub struct Analyzer<'a> {
    functions: HashMap<&'a str, Function<'a>>,
    errors: Vec<Error>,
    vars: Vec<HashMap<&'a str, Type>>,
    current_func: Option<&'a str>,
}

impl<'a> Analyzer<'a> {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            errors: Vec::new(),
            vars: Vec::new(),
            current_func: None,
        }
    }

    fn add_error(&mut self, msg: &str, span: Span) {
        self.errors.push(Error::new(msg, span));
    }

    fn push_scope(&mut self) {
        self.vars.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.vars.pop().unwrap();
    }

    fn new_var(&mut self, name: &'a str, ty: Type) {
        self.vars.last_mut().unwrap().insert(name, ty);
    }

    fn find_var(&mut self, name: &'a str) -> Option<Type> {
        for vars in self.vars.iter().rev() {
            match vars.get(name) {
                Some(ty) => return Some(ty.clone()),
                None => {},
            }
        }

        None
    }

    fn analyze_expr(&mut self, expr: &'a mut SpannedTyped<Expr>) -> (&'a Type, &'a Span) {
        match &mut expr.kind {
            Expr::Literal(Literal::Number(_)) => expr.ty = Type::Int,
            Expr::Literal(Literal::True) => expr.ty = Type::Bool,
            Expr::Literal(Literal::False) => expr.ty = Type::Bool,
            Expr::Variable(name) => {
                match self.find_var(name) {
                    Some(ty) => expr.ty = ty.clone(),
                    None => self.add_error(&format!("undefined variable `{}`", name), expr.span.clone()),
                }
            },
            Expr::BinOp(binop, lhs, rhs) => {
                let (lty, _) = self.analyze_expr(lhs);
                let (rty, _) = self.analyze_expr(rhs);

                if *lty != *rty {
                    self.add_error(&format!("different types `{}` and `{}`", lty, rty), expr.span.clone());
                    return (&lty, &expr.span);
                }

                let binop_symbol = binop.to_symbol();
                match (binop, lty) {
                    (BinOp::Add, Type::Int) => expr.ty = Type::Int,
                    (BinOp::Sub, Type::Int) => expr.ty = Type::Int,
                    (BinOp::Mul, Type::Int) => expr.ty = Type::Int,
                    (BinOp::Div, Type::Int) => expr.ty = Type::Int,
                    (BinOp::Equal, Type::Int) => expr.ty = Type::Bool,
                    (BinOp::NotEqual, Type::Int) => expr.ty = Type::Bool,
                    (BinOp::LessThan, Type::Int) => expr.ty = Type::Bool,
                    (BinOp::LessThanOrEqual, Type::Int) => expr.ty = Type::Bool,
                    (BinOp::GreaterThan, Type::Int) => expr.ty = Type::Bool,
                    (BinOp::GreaterThanOrEqual, Type::Int) => expr.ty = Type::Bool,
                    _ => self.add_error(&format!("`{} {} {}` is not possible", lty, binop_symbol, rty), expr.span.clone()),
                }
            },
            Expr::Call(name, args) => {
                let func = match self.functions.get(name) {
                    Some(func) => func.clone(),
                    None => {
                        self.add_error(&format!("undefined function `{}`", name), expr.span.clone());
                        return (&expr.ty, &expr.span)
                    },
                };

                expr.ty = func.return_ty;

                if args.len() != func.args.len() {
                    self.add_error(&format!("function `{}` takes {} parameters. but got {} arguments.", name, func.args.len(), args.len()), expr.span.clone());
                    return (&expr.ty, &expr.span);
                }

                for (arg, (param_name, param_ty)) in args.iter_mut().zip(func.args.into_iter()) {
                    let (arg_ty, span) = self.analyze_expr(arg);
                    if *arg_ty != param_ty {
                        self.add_error(&format!("parameter `{}` type is `{}`. but got `{}` type", param_name, param_ty, arg_ty), span.clone());
                    }
                }
            },
        };

        (&expr.ty, &expr.span)
    }

    fn analyze_stmt(&mut self, stmt: &'a mut Stmt) {
        match stmt {
            Stmt::Expr(expr) => { self.analyze_expr(expr); },
            Stmt::If(cond, stmt) | Stmt::While(cond, stmt) => {
                let (ty, span) = self.analyze_expr(cond);
                match ty {
                    Type::Bool => {},
                    ty => self.add_error(&format!("expected bool type, but got `{}` type", ty), span.clone()),
                };

                let span = &stmt.span;
                match stmt.kind {
                    Stmt::Block(_) | Stmt::If(_, _) => self.analyze_stmt(&mut stmt.kind),
                    _ => self.add_error("expected block or if statement", span.clone()),
                };
            },
            Stmt::Block(stmts) => {
                self.push_scope();

                for stmt in stmts {
                    self.analyze_stmt(&mut stmt.kind);
                }

                self.pop_scope();
            },
            Stmt::Bind(name, expr) => {
                let (ty, _) = self.analyze_expr(expr);
                self.new_var(name, ty.clone());
            },
            Stmt::Return(expr) => {
                let (ty, span) = self.analyze_expr(expr);
                if let Some(name) = self.current_func {
                    let return_ty = self.functions[name].return_ty.clone();
                    if *ty != return_ty {
                        self.add_error(&format!("expected `{}` type, but got `{}` type", return_ty, ty), span.clone());
                    }
                } else {
                    self.add_error("'return' statement outside function", span.clone());
                }
            },
        }
    }

    fn analyze_toplevel(&mut self, toplevel: &'a mut TopLevel) {
        match toplevel {
            TopLevel::Stmt(stmt) => self.analyze_stmt(&mut stmt.kind),
            TopLevel::Function(name, args, return_ty, stmt) => {
                let function = Function::<'a> {
                    args: args.clone(),
                    return_ty: return_ty.clone(),
                };
                self.functions.insert(name, function);

                self.push_scope();

                for (name, ty) in args {
                    self.new_var(name, ty.clone());
                }

                self.current_func = Some(name);
                self.analyze_stmt(&mut stmt.kind);
                self.current_func = None;

                self.pop_scope();
            },
        }
    }

    pub fn analyze(mut self, program: &'a mut Program) -> Result<(), Vec<Error>> {
        self.functions.insert("printi", Function {
            args: vec![("n", Type::Int)],
            return_ty: Type::Int,
        });
        self.functions.insert("printlf", Function {
            args: vec![],
            return_ty: Type::Int,
        });

        self.push_scope();

        for toplevel in &mut program.top {
            self.analyze_toplevel(&mut toplevel.kind);
        }

        self.pop_scope();

        if self.errors.len() > 0 {
            Err(self.errors)
        } else {
            Ok(())
        }
    }
}
