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

macro_rules! expect {
    ($self:ident, $pat:pat => $expr:expr) => {
        {
            let token = $self.peek();
            let res = match token.kind {
                $pat => Ok($expr),
                _ => Err($self.unexpected_token(token)),
            };
            $self.next();
            res
        }
    };
    ($self:ident, $pat:pat) => (expect!($self, $pat => ()));
}

macro_rules! skip_if_err {
    ($self:ident, $result:expr, $token:expr) => {
        $result.or_else(|err| { $self.skip_until($token); Err(err) })
    }
}

type ExprResult<'a> = Result<Spanned<Expr<'a>>, Error>;
type StmtResult<'a> = Result<Spanned<Stmt<'a>>, Error>;

pub struct Parser<'a> {
    tokens: Vec<Spanned<Token<'a>>>,
    pos: usize,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: Vec<Spanned<Token<'a>>>) -> Parser<'a> {
        Self {
            tokens,
            pos: 0,
        }
    }

    fn next(&mut self) -> &Spanned<Token<'a>> {
        self.pos += 1;
        &self.tokens[self.pos]
    }

    #[inline]
    fn peek(&self) -> &Spanned<Token<'a>> {
        &self.tokens[self.pos]
    }

    #[inline]
    fn unexpected_token(&self, token: &Spanned<Token>) -> Error {
        Error::new(&format!("Unexpected token `{}`", token.kind), token.span.clone())
    }

    fn skip_until(&mut self, token: &Token) {
        while self.peek().kind != *token && self.peek().kind != Token::EOF {
            self.next();
        }

        if self.peek().kind != Token::EOF {
            self.next();
        }
    }

    fn consume(&mut self, token: Token) -> bool {
        if self.peek().kind == token {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_primary(&mut self) -> ExprResult<'a> {
        let token = self.peek();
        let span = &token.span;

        let result = match &token.kind {
            Token::Number(n) => Ok(spanned(Expr::Literal(Literal::Number(*n)), span.clone())),
            Token::Identifier(name) => Ok(spanned(Expr::Variable(name), span.clone())),
            Token::Lparen => {
                self.next();
                let mut expr = self.parse_expr()?;

                expect!(self, Token::Rparen)?;

                // Adjust to parentheses
                expr.span.start_col -= 1;
                expr.span.end_col += 1;

                return Ok(expr);
            },
            _ => Err(self.unexpected_token(token)),
        };
        self.next();
        result
    }

    fn parse_mul(&mut self) -> ExprResult<'a> {
        binop!(self, parse_primary, {
            Token::Asterisk => BinOp::Mul,
            Token::Div => BinOp::Div,
        })
    }

    fn parse_add(&mut self) -> ExprResult<'a> {
        binop!(self, parse_mul, {
            Token::Add => BinOp::Add,
            Token::Sub => BinOp::Sub,
        })
    }

    fn parse_expr(&mut self) -> ExprResult<'a> {
        self.parse_add()
    }

    fn parse_bind_stmt(&mut self) -> StmtResult<'a> {
        // Eat "let"
        let let_span = self.peek().span.clone();
        self.next();

        let name = skip_if_err!(self, expect!(self, Token::Identifier(name) => name), &Token::Semicolon)?;
        skip_if_err!(self, expect!(self, Token::Assign), &Token::Semicolon)?;
        let expr = skip_if_err!(self, self.parse_expr(), &Token::Semicolon)?;

        expect!(self, Token::Semicolon)?;

        let mut span = Span::merge(&let_span, &expr.span);
        // Adjust to semicolon
        span.end_col += 1;

        Ok(spanned(Stmt::Bind(name, expr), span))
    }

    fn parse_expr_stmt(&mut self) -> StmtResult<'a> {
        let expr = skip_if_err!(self, self.parse_expr(), &Token::Semicolon)?;

        expect!(self, Token::Semicolon)?;

        let mut span = expr.span.clone();
        // Adjust to semicolon
        span.end_col += 1;

        Ok(spanned(Stmt::Expr(expr), span))
    }

    fn parse_stmt(&mut self) -> StmtResult<'a> {
        let token = self.peek().clone();

        match token.kind {
            Token::Let => self.parse_bind_stmt(),
            _ => self.parse_expr_stmt(),
        }
    }

    pub fn parse(mut self) -> Result<Program<'a>, Vec<Error>> {
        let mut stmts = Vec::new();
        let mut errors = Vec::new();

        while self.peek().kind != Token::EOF {
            match self.parse_stmt() {
                Ok(stmt) => stmts.push(stmt),
                Err(err) => errors.push(err),
            };
        }

        if errors.len() > 0 {
            Err(errors)
        } else {
            Ok(Program { stmt: stmts })
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

        let lexer = Lexer::new("let abc = 10 + 3 * (5 + 20); abc;");
        let tokens = lexer.lex().unwrap();
        let parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let expected = Program {
            stmt: vec![
                *new(Stmt::Bind(
                    "abc",
                    *new(Expr::BinOp(BinOp::Add,
                          new(Expr::Literal(Literal::Number(10)), 0, 10, 0, 12),
                          new(Expr::BinOp(BinOp::Mul,
                              new(Expr::Literal(Literal::Number(3)), 0, 15, 0, 16),
                              new(Expr::BinOp(BinOp::Add,
                                  new(Expr::Literal(Literal::Number(5)), 0, 20, 0, 21),
                                  new(Expr::Literal(Literal::Number(20)), 0, 24, 0, 26)),
                                  0, 19, 0, 27)),
                              0, 15, 0, 27)),
                          0, 10, 0, 27)),
                    0, 0, 0, 28),
                *new(Stmt::Expr(
                    *new(Expr::Variable("abc"), 0, 29, 0, 32)),
                    0, 29, 0, 33),
            ],
        };

        assert_eq!(program, expected);
    }
}
