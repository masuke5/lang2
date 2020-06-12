use crate::ast::*;
use crate::id::Id;
use crate::span::Spanned;
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

    fn find_func(&mut self, func: &'a mut AstFunction) {
        self.push_scope();

        for param in &mut func.params {
            self.variables.insert(param.name, &mut param.is_in_heap);
        }
        self.find_expr(&mut func.body.kind);

        self.pop_scope();
    }

    fn find_block(&mut self, block: &'a mut Block) {
        for stmt in &mut block.stmts {
            self.find_stmt(&mut stmt.kind);
        }
        self.find_expr(&mut block.result_expr.kind);

        for func in &mut block.functions {
            self.find_func(func);
        }
    }

    fn mark(&mut self, expr: &'a mut Expr) {
        match expr {
            Expr::Variable(name, _) => {
                if let Some(is_heapvar) = self.variables.get_mut(&name) {
                    **is_heapvar = true;
                }
            }
            Expr::Field(expr, _) => {
                self.mark(&mut expr.kind);
            }
            _ => {}
        }
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
            Expr::Call(func_expr, arg_expr) => {
                self.find_expr(&mut func_expr.kind);
                self.find_expr(&mut arg_expr.kind);
            }
            Expr::Dereference(expr) | Expr::Negative(expr) | Expr::App(expr, _) => {
                self.find_expr(&mut expr.kind)
            }
            Expr::Block(block) => {
                self.find_block(block);
            }
            Expr::If(cond, then, els) => {
                self.find_expr(&mut cond.kind);
                self.find_expr(&mut then.kind);
                if let Some(els) = els {
                    self.find_expr(&mut els.kind);
                }
            }
            Expr::Address(expr, _) => {
                if let Expr::Subscript(
                    ref mut expr,
                    box Spanned {
                        kind: Expr::Range(..),
                        ..
                    },
                ) = expr.kind
                {
                    self.mark(&mut expr.kind);
                } else {
                    self.mark(&mut expr.kind);
                }
            }
            _ => {}
        }
    }

    fn find_stmt(&mut self, stmt: &'a mut Stmt) {
        match stmt {
            Stmt::Expr(expr) => self.find_expr(&mut expr.kind),
            Stmt::Bind(name, _, expr, _, _, is_in_heap) => {
                self.find_expr(&mut expr.kind);
                self.variables.insert(*name, is_in_heap);
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
            _ => {}
        }
    }

    fn find(&mut self, program: &'a mut Program) {
        self.push_scope();

        self.find_block(&mut program.main);

        for imple in &mut program.impls {
            for func in &mut imple.functions {
                self.find_func(func);
            }
        }

        self.pop_scope();
    }
}

pub fn find(program: &mut Program) {
    let mut finder = Finder::new();
    finder.find(program);
}
