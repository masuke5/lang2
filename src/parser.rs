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

macro_rules! unwrap_or_skip {
    ($self:ident, $result:expr, $token:pat $(| $pat:pat),*) => {
        $result.or_else(|err| {
            while $self.peek().kind != Token::EOF {
                match $self.peek().kind {
                    $token $(| $pat)* => break,
                    _ => $self.next(),
                };
            }

            if $self.peek().kind != Token::EOF {
                $self.next();
            }
            Err(err)
        })
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
    fn prev(&self) -> &Spanned<Token<'a>> {
        &self.tokens[self.pos - 1]
    }

    #[inline]
    fn unexpected_token(&self, token: &Spanned<Token>) -> Error {
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

    fn parse_call(&mut self, name: &'a str, name_span: Span) -> ExprResult<'a> {
        // Parse arguments
        let mut args = Vec::new();

        let mut parse_args = || {
            if !self.consume(Token::Rparen) {
                loop {
                    args.push(self.parse_expr()?);

                    if self.consume(Token::Rparen) {
                        return Ok(self.prev().span.clone());
                    } else if !self.consume(Token::Comma) {
                        return Err(self.unexpected_token(self.peek()));
                    }
                }
            } else {
                Ok(self.prev().span.clone())
            }
        };
        let rparen_span = parse_args()?;

        Ok(spanned(Expr::Call(name, args), Span::merge(&name_span, &rparen_span)))
    }

    fn parse_var_or_call(&mut self, ident: &'a str, ident_span: Span) -> ExprResult<'a> {
        self.next();

        if self.consume(Token::Lparen) {
            self.parse_call(ident, ident_span)
        } else {
            Ok(spanned(Expr::Variable(ident), ident_span.clone()))
        }
    }

    fn parse_primary(&mut self) -> ExprResult<'a> {
        let token = self.peek().clone();

        match token.kind {
            Token::Number(n) => {
                self.next();
                Ok(spanned(Expr::Literal(Literal::Number(n)), token.span))
            },
            Token::Identifier(name) => self.parse_var_or_call(name, token.span),
            Token::Lparen => {
                self.next();
                let mut expr = self.parse_expr()?;

                expect!(self, Token::Rparen)?;

                // Adjust to parentheses
                expr.span.start_col -= 1;
                expr.span.end_col += 1;

                return Ok(expr);
            },
            _ => Err(self.unexpected_token(&token)),
        }
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

        let name = unwrap_or_skip!(self, expect!(self, Token::Identifier(name) => name), Token::Semicolon)?;
        unwrap_or_skip!(self, expect!(self, Token::Assign), Token::Semicolon)?;
        let expr = unwrap_or_skip!(self, self.parse_expr(), Token::Semicolon)?;

        expect!(self, Token::Semicolon)?;

        let mut span = Span::merge(&let_span, &expr.span);
        // Adjust to semicolon
        span.end_col += 1;

        Ok(spanned(Stmt::Bind(name, expr), span))
    }

    fn parse_expr_stmt(&mut self) -> StmtResult<'a> {
        let expr = unwrap_or_skip!(self, self.parse_expr(), Token::Semicolon)?;

        expect!(self, Token::Semicolon)?;

        let mut span = expr.span.clone();
        // Adjust to semicolon
        span.end_col += 1;

        Ok(spanned(Stmt::Expr(expr), span))
    }

    fn parse_multiple_statements(&mut self, end_token: &Token) -> Result<Vec<Spanned<Stmt<'a>>>, Vec<Error>> {
        let mut stmts = Vec::new();
        let mut errors = Vec::new();

        while self.peek().kind != *end_token {
            if let Token::EOF = self.peek().kind {
                errors.push(Error::new(&format!("Expected `{}`, but got EOF", end_token), self.peek().span.clone()));
                break;
            }

            match self.parse_stmt() {
                Ok(stmt) => stmts.push(stmt),
                Err(err) => errors.extend(err),
            }
        }

        if errors.len() > 0 {
            Err(errors)
        } else {
            Ok(stmts)
        }
    }

    fn parse_block(&mut self) -> Result<Spanned<Stmt<'a>>, Vec<Error>> {
        // Eat "{"
        let lbrace_span = self.peek().span.clone();
        self.next();

        let stmts = self.parse_multiple_statements(&Token::Rbrace)?;

        // Eat "}"
        let rbrace_span = self.peek().span.clone();
        self.next();

        let span = Span::merge(&lbrace_span, &rbrace_span);
        Ok(spanned(Stmt::Block(stmts), span))
    }

    fn parse_return(&mut self) -> StmtResult<'a> {
        // Eat "return"
        let return_token_span = self.peek().span.clone();
        self.next();

        let expr = self.parse_expr()?;

        expect!(self, Token::Semicolon)?;

        let mut span = Span::merge(&return_token_span, &expr.span);
        // Adjust to a semicolon
        span.end_col += 1;

        Ok(spanned(Stmt::Return(expr),span))
    }

    fn parse_stmt(&mut self) -> Result<Spanned<Stmt<'a>>, Vec<Error>> {
        let token = self.peek();

        match token.kind {
            Token::Let => self.parse_bind_stmt().map_err(|err| vec![err]),
            Token::Lbrace => self.parse_block(),
            Token::Return => self.parse_return().map_err(|err| vec![err]),
            _ => self.parse_expr_stmt().map_err(|err| vec![err]),
        }
    }

    fn parse_type(&mut self) -> Result<Type, Error> {
        let result = match self.peek().kind {
            Token::Int => Ok(Type::Int),
            _ => Err(self.unexpected_token(self.peek())),
        };

        self.next();

        result
    }

    fn parse_fn_decl(&mut self) -> Result<Spanned<TopLevel<'a>>, Vec<Error>> {
        // Eat "fn"
        let fn_span = self.peek().span.clone();
        self.next();

        // name(
        let name = expect!(self, Token::Identifier(name) => name).map_err(|err| vec![err])?;
        unwrap_or_skip!(self, expect!(self, Token::Lparen), Token::Rparen).map_err(|err| vec![err])?;

        let mut errors = Vec::new();
        let mut params = Vec::new();
        loop {
            let mut parse_param = || -> Result<(&'a str, Type), Error> {
                // name: type
                let name = unwrap_or_skip!(self, expect!(self, Token::Identifier(name) => name), Token::Comma | Token::Rparen)?;
                unwrap_or_skip!(self, expect!(self, Token::Colon), Token::Comma | Token::Rparen)?;
                let ty = self.parse_type()?;
                Ok((name, ty))
            };

            match parse_param() {
                Ok(param) => params.push(param),
                Err(err) => errors.push(err),
            };

            if self.consume(Token::Rparen) {
                break;
            } else if !self.consume(Token::Comma) {
                return Err(vec![self.unexpected_token(self.peek())]);
            }
        }

        if errors.len() > 0 {
            return Err(errors);
        }

        expect!(self, Token::Colon).map_err(|err| vec![err])?;
        let return_ty = self.parse_type().map_err(|err| vec![err])?;

        let body = self.parse_stmt()?;

        let span = Span::merge(&fn_span, &body.span);
        Ok(spanned(TopLevel::Function(name, params, return_ty, body), span))
    }

    fn parse_toplevel(&mut self) -> Result<Spanned<TopLevel<'a>>, Vec<Error>> {
        match self.peek().kind {
            Token::Fn => self.parse_fn_decl(),
            _ => self.parse_stmt().map(|stmt| {
                let span = stmt.span.clone();
                spanned(TopLevel::Stmt(stmt), span)
            }),
        }
    }

    pub fn parse(mut self) -> Result<Program<'a>, Vec<Error>> {
        let mut toplevels = Vec::new();
        let mut errors = Vec::new();

        while self.peek().kind != Token::EOF {
            match self.parse_toplevel() {
                Ok(toplevel) => toplevels.push(toplevel),
                Err(err) => errors.extend(err),
            }
        }

        if errors.len() > 0 {
            Err(errors)
        } else {
            Ok(Program { top: toplevels })
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

        let lexer = Lexer::new(r#"let abc = 10 + 3 * (5 + 20); abc; { abc; 10; }
fn add(a: int, b: int): int { a + b; }
add(3, 5 + 8);
return abc;"#);
        let tokens = lexer.lex().unwrap();
        let parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let expected = Program {
            top: vec![
                *new(TopLevel::Stmt(
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
                        0, 0, 0, 28)),
                    0, 0, 0, 28),
                *new(TopLevel::Stmt(
                    *new(Stmt::Expr(
                        *new(Expr::Variable("abc"), 0, 29, 0, 32)),
                        0, 29, 0, 33)),
                    0, 29, 0, 33),
                *new(TopLevel::Stmt(
                    *new(Stmt::Block(vec![
                        *new(Stmt::Expr(
                            *new(Expr::Variable("abc"), 0, 36, 0, 39)),
                            0, 36, 0, 40),
                        *new(Stmt::Expr(
                            *new(Expr::Literal(Literal::Number(10)), 0, 41, 0, 43)),
                            0, 41, 0, 44)]),
                        0, 34, 0, 46)),
                    0, 34, 0, 46),
                *new(TopLevel::Function("add", vec![("a", Type::Int), ("b", Type::Int)], Type::Int,
                    *new(Stmt::Block(vec![
                        *new(Stmt::Expr(
                            *new(Expr::BinOp(BinOp::Add,
                                new(Expr::Variable("a"), 1, 30, 1, 31),
                                new(Expr::Variable("b"), 1, 34, 1, 35)),
                                1, 30, 1, 35)),
                            1, 30, 1, 36)]),
                        1, 28, 1, 38)),
                    1, 0, 1, 38),
                *new(TopLevel::Stmt(
                    *new(Stmt::Expr(
                        *new(Expr::Call("add", vec![
                            *new(Expr::Literal(Literal::Number(3)), 2, 4, 2, 5),
                            *new(Expr::BinOp(BinOp::Add,
                                new(Expr::Literal(Literal::Number(5)), 2, 7, 2, 8),
                                new(Expr::Literal(Literal::Number(8)), 2, 11, 2, 12)),
                                2, 7, 2, 12)]),
                            2, 0, 2, 13)),
                        2, 0, 2, 14)),
                    2, 0, 2, 14),
                *new(TopLevel::Stmt(
                    *new(Stmt::Return(
                        *new(Expr::Variable("abc"), 3, 7, 3, 10)),
                        3, 0, 3, 11)),
                    3, 0, 3, 11),
            ],
        };

        assert_eq!(program, expected);
    }
}
