use crate::ast::*;
use crate::id::Id;
use crate::utils::HashMapWithScope;

struct Finder<'a> {
    variables: HashMapWithScope<Id, &'a mut bool>,
}

impl<'a> Finder<'a> {
    fn new() -> Self {
        Finder {
            variables: HashMapWithScope::new(),
        }
    }

    fn push_scope(&mut self) {
        self.variables.push_scope();
    }

    fn pop_scope(&mut self) {
        self.variables.pop_scope();
    }

    fn find_expr(&mut self, expr: &'a mut Expr) {
        match expr {
            Expr::Tuple(exprs) => {
                for expr in exprs {
                    self.find_expr(&mut expr.kind);
                }
            }
            Expr::Struct(_, fields) => {
                for (_, expr) in fields {
                    self.find_expr(&mut expr.kind);
                }
            }
            Expr::Array(expr, _) => self.find_expr(&mut expr.kind),
            Expr::Subscript(array_expr, index_expr) => {
                self.find_expr(&mut array_expr.kind);
                self.find_expr(&mut index_expr.kind);
            }
            Expr::BinOp(_, lhs, rhs) => {
                self.find_expr(&mut lhs.kind);
                self.find_expr(&mut rhs.kind);
            }
            Expr::Variable(name, is_escaped) => {
                let current_level = self.variables.level();
                if let Some((escaped, level)) = self.variables.get_mut_with_level(&name) {
                    if level < current_level {
                        **escaped = true;
                        *is_escaped = true;
                    }
                }
            }
            Expr::Call(func_expr, arg_expr) => {
                self.find_expr(&mut func_expr.kind);
                self.find_expr(&mut arg_expr.kind);
            }
            Expr::Dereference(expr)
            | Expr::Negative(expr)
            | Expr::Alloc(expr, _)
            | Expr::App(expr, _) => self.find_expr(&mut expr.kind),
            Expr::Block(stmts, expr) => {
                for stmt in stmts {
                    self.find_stmt(&mut stmt.kind);
                }
                self.find_expr(&mut expr.kind);
            }
            Expr::If(cond, then, els) => {
                self.find_expr(&mut cond.kind);
                self.find_expr(&mut then.kind);
                if let Some(els) = els {
                    self.find_expr(&mut els.kind);
                }
            }
            _ => {}
        }
    }

    fn find_stmt(&mut self, stmt: &'a mut Stmt) {
        match stmt {
            Stmt::Expr(expr) => self.find_expr(&mut expr.kind),
            Stmt::Bind(name, _, expr, _, is_escaped) => {
                self.find_expr(&mut expr.kind);
                self.variables.insert(*name, is_escaped);
            }
            Stmt::Return(Some(expr)) => {
                self.find_expr(&mut expr.kind);
            }
            Stmt::While(expr, stmt) => {
                self.find_expr(&mut expr.kind);
                self.find_stmt(&mut stmt.kind);
            }
            Stmt::Assign(lhs, rhs) => {
                self.find_expr(&mut lhs.kind);
                self.find_expr(&mut rhs.kind);
            }
            Stmt::FnDef(func) => {
                self.push_scope();

                for param in &mut func.params {
                    self.variables.insert(param.name, &mut param.is_escaped);
                }
                self.find_expr(&mut func.body.kind);

                self.pop_scope();
            }
            _ => {}
        }
    }

    fn find(&mut self, program: &'a mut Program) {
        self.push_scope();

        for stmt in &mut program.main_stmts {
            self.find_stmt(&mut stmt.kind);
        }

        self.pop_scope();
    }
}

pub fn find(program: &mut Program) {
    let mut finder = Finder::new();
    finder.find(program);
}
