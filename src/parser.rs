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
                _ => Err(Self::unexpected_token(token)),
            };
            $self.next();
            res
        }
    };
    ($self:ident, $pat:pat) => (expect!($self, $pat => ()));
}

macro_rules! push_if_error {
    ($result:expr, $error_vec:expr, $default:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                $error_vec.push(err);
                $default
            },
        }
    };
    ($result:expr, $error_vec:expr) => {
        match $result {
            Ok(_) => (),
            Err(err) => $error_vec.push(err),
        };
    };
}

fn result_from_error<T, E>(value: T, errors: Vec<E>) -> Result<T, Vec<E>> {
    if errors.len() > 0 {
        Err(errors)
    } else {
        Ok(value)
    }
}

type ExprResult = Result<Spanned<Expr>, Error>;
type StmtResult<'a> = Result<Spanned<Stmt<'a>>, Vec<Error>>;

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

    fn peek(&self) -> &Spanned<Token<'a>> {
        &self.tokens[self.pos]
    }

    fn unexpected_token(token: &Spanned<Token>) -> Error {
        Error::new(&format!("Unexpected token `{}`", token.kind), token.span.clone())
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
            _ => Err(Self::unexpected_token(token)),
        }?;

        self.next();
        Ok(result)
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

    fn parse_bind_stmt(&mut self) -> StmtResult<'a> {
        let mut errors = Vec::new();

        // Eat "let"
        let let_span = self.peek().span.clone();
        self.next();

        // Identifier 
        let name = expect!(self, Token::Identifier(name) => name);
        let name = push_if_error!(name, errors, "error");

        // =
        push_if_error!(expect!(self, Token::Assign), errors);

        // Expression
        let expr = match self.parse_expr() {
            Ok(expr) => expr,
            Err(err) => {
                errors.push(err);
                return Err(errors);
            },
        };

        push_if_error!(expect!(self, Token::Semicolon), errors);

        let span = Span::merge(&let_span, &expr.span);
        result_from_error(
            spanned(Stmt::Bind(name, expr), span),
            errors)
    }

    fn parse_expr_stmt(&mut self) -> StmtResult<'a> {
        let mut errors = Vec::new();

        let expr = match self.parse_expr() {
            Ok(expr) => expr,
            Err(err) => {
                errors.push(err);
                return Err(errors);
            },
        };

        push_if_error!(expect!(self, Token::Semicolon), errors);

        let span = expr.span.clone();
        result_from_error(
            spanned(Stmt::Expr(expr), span),
            errors)
    }

    fn parse_stmt(&mut self) -> StmtResult<'a> {
        let token = self.peek().clone();

        match token.kind {
            Token::Let => self.parse_bind_stmt(),
            _ => self.parse_expr_stmt(),
        }
    }

    pub fn parse(mut self) -> Result<Program<'a>, Vec<Error>> {
        let mut errors = Vec::new();
        let mut stmts = Vec::new();

        while self.peek().kind != Token::EOF {
            match self.parse_stmt() {
                Ok(stmt) => stmts.push(stmt),
                Err(new_errors) => {
                    errors.extend(new_errors.into_iter());
                },
            };
        }

        result_from_error(Program { stmt: stmts }, errors)
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

        let lexer = Lexer::new("let abc = 10 + 3 * (5 + 20);");
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
                                  0, 20, 0, 26)),
                              0, 15, 0, 26)),
                          0, 10, 0, 26)),
                    0, 0, 0, 26),
            ],
        };

        assert_eq!(program, expected);
    }
}
