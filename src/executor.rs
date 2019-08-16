use std::collections::HashMap;

use crate::ast::*;
use crate::env::*;

pub struct Executor<'a> {
    var_map: HashMap<&'a str, Value>,
}

impl<'a> Executor<'a> {
    pub fn new() -> Self {
        Self {
            var_map: HashMap::new(),
        }
    }

    fn run_binop(&mut self, binop: BinOp, lhs: Expr<'a>, rhs: Expr<'a>) -> Value {
        let left = self.run_expr(lhs).int();
        let right = self.run_expr(rhs).int();

        Value::Int(match binop {
            BinOp::Add => left + right,
            BinOp::Sub => left - right,
            BinOp::Mul => left * right,
            BinOp::Div => left / right,
        })
    }

    fn run_expr(&mut self, expr: Expr<'a>) -> Value {
        #[allow(unreachable_patterns)]
        match expr {
            Expr::Literal(Literal::Number(n)) => Value::Int(n),
            Expr::BinOp(binop, lhs, rhs) => self.run_binop(binop, lhs.kind, rhs.kind),
            Expr::Variable(name) => self.var_map[name].clone(),
            _ => unimplemented!(),
        }
    }

    fn run_bind_stmt(&mut self, name: &'a str, expr: Expr<'a>) {
        let value = self.run_expr(expr);
        self.var_map.insert(name, value);
    }

    fn run_stmt(&mut self, stmt: Stmt<'a>) {
        #[allow(unreachable_patterns)]
        match stmt {
            Stmt::Bind(name, expr) => self.run_bind_stmt(name, expr.kind),
            Stmt::Expr(expr) => { self.run_expr(expr.kind); },
            Stmt::Block(stmts) => {
                for stmt in stmts {
                    self.run_stmt(stmt.kind);
                }
            },
            _ => unimplemented!(),
        };
    }

    pub fn exec(&mut self, program: Program<'a>) -> i64 {
        for stmt in program.stmt {
            self.run_stmt(stmt.kind);
        }

        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    #[test]
    fn exec() {
        let lexer = Lexer::new(r#"
            {
                let foo = 10 + 3 * 5 + 20;
                let bar = 30;
                let baz = foo + bar;
            }"#);
        let tokens = lexer.lex().unwrap();
        let parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        let mut executor = Executor::new();
        let _ = executor.exec(program);

        let mut expected = HashMap::new();
        expected.insert("foo", Value::Int(45));
        expected.insert("bar", Value::Int(30));
        expected.insert("baz", Value::Int(75));

        for name in executor.var_map.keys() {
            assert_eq!(executor.var_map[name], expected[name]);
        }
    }
}
