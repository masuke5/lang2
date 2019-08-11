use crate::ast::*;
use crate::env::*;

pub struct Executor {
}

impl Executor {
    pub fn new() -> Self {
        Self {
        }
    }

    fn run_binop(&mut self, binop: BinOp, lhs: Expr, rhs: Expr) -> Value {
        let left = self.run_expr(lhs).int();
        let right = self.run_expr(rhs).int();

        Value::Int(match binop {
            BinOp::Add => left + right,
            BinOp::Sub => left - right,
            BinOp::Mul => left * right,
            BinOp::Div => left / right,
        })
    }

    fn run_expr(&mut self, expr: Expr) -> Value {
        match expr {
            Expr::Literal(Literal::Number(n)) => Value::Int(n),
            Expr::BinOp(binop, lhs, rhs) => self.run_binop(binop, lhs.kind, rhs.kind),
        }
    }

    pub fn exec(&mut self, program: Program) -> i64 {
        self.run_expr(program.expr.kind).int()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    #[test]
    fn exec() {
        let lexer = Lexer::new("10 + 3 * 5 + 20");
        let tokens = lexer.lex().unwrap();
        let parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        let mut executor = Executor::new();
        let result = executor.exec(program);

        assert_eq!(result, 45);
    }
}
