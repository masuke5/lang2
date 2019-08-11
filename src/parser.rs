use crate::span::{Span, Spanned};
use crate::error::Error;
use crate::token::*;
use crate::ast::*;

fn spanned<T>(kind: T, span: Span) -> Spanned<T> {
    Spanned::<T>::new(kind, span)
}

macro_rules! binop {
    ($self:ident, $func:ident, { $($token:path => $binop:path),* $(,)? }) => {
        {
            let mut expr = $self.$func()?;

            loop {
                if false {
                } $(else if $self.consume($token) {
                    let rhs = $self.$func()?;
                    let span = Span::merge(&expr.span, &rhs.span);
                    expr = spanned(Expr::BinOp($binop, Box::new(expr), Box::new(rhs)), span);
                })* else {
                    break;
                }
            }

            Ok(expr)
        }
    }
}

type ExprResult = Result<Spanned<Expr>, Error>;

pub struct Parser {
    tokens: Vec<Spanned<Token>>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Spanned<Token>>) -> Self {
        Self {
            tokens,
            pos: 0,
        }
    }

    fn next(&mut self) -> &Spanned<Token> {
        self.pos += 1;
        &self.tokens[self.pos]
    }

    fn peek(&self) -> &Spanned<Token> {
        &self.tokens[self.pos]
    }

    fn consume(&mut self, token: Token) -> bool {
        if self.peek().kind == token {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_primary(&mut self) -> ExprResult {
        let token = self.peek();
        let span = &token.span;

        let result = match &token.kind {
            Token::Number(n) => Ok(spanned(Expr::Literal(Literal::Number(*n)), span.clone())),
            Token::Lparen => {
                self.next();
                self.parse_expr()
            },
            token => Err(Error::new(&format!("Unexpected token `{}`", token), span.clone())),
        };

        self.next();
        result
    }

    fn parse_mul(&mut self) -> ExprResult {
        binop!(self, parse_primary, {
            Token::Asterisk => BinOp::Mul,
            Token::Div => BinOp::Div,
        })
    }

    fn parse_add(&mut self) -> ExprResult {
        binop!(self, parse_mul, {
            Token::Add => BinOp::Add,
            Token::Sub => BinOp::Sub,
        })
    }

    fn parse_expr(&mut self) -> ExprResult {
        self.parse_add()
    }

    pub fn parse(mut self) -> Result<Program, Vec<Error>> {
        match self.parse_expr() {
            Ok(expr) => Ok(Program { expr }),
            Err(err) => Err(vec![err]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use pretty_assertions::assert_eq;

    #[test]
    fn parse() {
        fn new<T>(kind: T, start_line: u32, start_col: u32, end_line: u32, end_col: u32) -> Box<Spanned<T>> {
            Box::new(Spanned::new(kind, Span {
                start_line,
                start_col,
                end_line,
                end_col,
            }))
        }

        let lexer = Lexer::new("10 + 3 * (5 + 20)");
        let tokens = lexer.lex().unwrap();
        let parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let expected = Program {
            expr: *new(Expr::BinOp(BinOp::Add,
                      new(Expr::Literal(Literal::Number(10)), 0, 0, 0, 2),
                      new(Expr::BinOp(BinOp::Mul,
                          new(Expr::Literal(Literal::Number(3)), 0, 5, 0, 6),
                          new(Expr::BinOp(BinOp::Add,
                              new(Expr::Literal(Literal::Number(5)), 0, 10, 0, 11),
                              new(Expr::Literal(Literal::Number(20)), 0, 14, 0, 16)),
                              0, 10, 0, 16)),
                          0, 5, 0, 16)),
                      0, 0, 0, 16),
        };

        assert_eq!(program, expected);
    }
}
