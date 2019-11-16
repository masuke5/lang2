use crate::span::{Span, Spanned};
use crate::error::Error;
use crate::token::*;
use crate::ast::*;
use crate::ty::Type;
use crate::id::Id;

fn spanned<T>(kind: T, span: Span) -> Spanned<T> {
    Spanned::<T>::new(kind, span)
}

macro_rules! error {
    ($self:ident, $span:expr, $fmt: tt $(,$arg:expr)*) => {
        $self.errors.push(Error::new(&format!($fmt $(,$arg)*), $span));
    };
}


pub struct Parser {
    tokens: Vec<Spanned<Token>>,
    pos: usize,
    errors: Vec<Error>,
}

impl Parser {
    pub fn new(tokens: Vec<Spanned<Token>>) -> Parser {
        Self {
            tokens,
            pos: 0,
            errors: Vec::new(),
        }
    }

    fn next(&mut self) -> &Spanned<Token> {
        self.pos += 1;
        &self.tokens[self.pos]
    }

    #[inline]
    fn peek(&self) -> &Spanned<Token> {
        &self.tokens[self.pos]
    }

    #[inline]
    fn prev(&self) -> &Spanned<Token> {
        &self.tokens[self.pos - 1]
    }

    #[inline]
    fn consume(&mut self, token: &Token) -> bool {
        if &self.peek().kind == token {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: &Token, skip: &[Token]) -> Option<()> {
        if !self.consume(&expected) {
            let token = self.peek().clone();
            error!(self, token.span, "expected `{}` but got `{}`", expected, token.kind);

            self.skip_to(skip);
            if self.peek().kind == *expected {
                self.next();
            }

            None
        } else {
            Some(())
        }
    }

    fn expect_identifier(&mut self, skip: &[Token]) -> Option<Id> {
        match self.peek().kind {
            Token::Identifier(name) => {
                self.next();
                Some(name)
            }
            _ => {
                let token = self.peek().clone();
                error!(self, token.span, "expected `identifier` but got `{}`", token.kind);

                self.skip_to(skip);
                if let Token::Identifier(_) = self.peek().kind {
                    self.next();
                }

                None
            },
        }
    }

    #[inline]
    fn skip_to(&mut self, tokens: &[Token]) {
        while !tokens.contains(&self.peek().kind) && self.peek().kind != Token::EOF {
            self.next();
        }
    }

    // Parse something using `func`. skip to `tokens` if fail.
    fn parse_skip<T, F>(&mut self, mut func: F, tokens: &[Token]) -> Option<T>
        where F: FnMut(&mut Self,) -> Option<T>
    {
        let res = func(self);
        if let None = res {
            self.skip_to(tokens);
        }

        res
    }

    #[inline]
    fn parse_binop<F>(&mut self, allow_join: bool, mut func: F, rules: &[(&Token, &BinOp)]) -> Option<Spanned<Expr>>
        where F: FnMut(&mut Self) -> Option<Spanned<Expr>>
    {
        let mut expr = func(self)?;

        if allow_join {
            'outer: loop {
                for (token, binop) in rules {
                    if self.consume(token) {
                        let rhs = func(self)?;
                        let span = Span::merge(&expr.span, &rhs.span);

                        expr = spanned(Expr::BinOp((*binop).clone(), Box::new(expr), Box::new(rhs)), span);
                        continue 'outer;
                    }
                }

                break;
            }
        } else {
            for (token, binop) in rules {
                if self.consume(token) {
                    let rhs = func(self)?;
                    let span = Span::merge(&expr.span, &rhs.span);

                    expr = spanned(Expr::BinOp((*binop).clone(), Box::new(expr), Box::new(rhs)), span);
                    break;
                }
            }
        }

        Some(expr)
    }

    fn parse_args(&mut self) -> Option<Vec<Spanned<Expr>>> {
        if !self.consume(&Token::Rparen) {
            let mut args = Vec::new();

            // Parse a first argument 
            match self.parse_expr() {
                Some(expr) => args.push(expr),
                None => self.skip_to(&[Token::Comma, Token::Rparen]),
            };

            // Parse other arguments
            while !self.consume(&Token::Rparen) {
                if let None = self.expect(&Token::Comma, &[Token::Rparen]) {
                    break;
                }

                match self.parse_expr() {
                    Some(expr) => args.push(expr),
                    None => self.skip_to(&[Token::Comma, Token::Rparen]),
                };
            }

            if self.peek().kind == Token::EOF {
                None
            } else {
                Some(args)
            }
        } else {
            Some(Vec::new())
        }
    }

    fn parse_call(&mut self, name: Id, name_span: Span) -> Option<Spanned<Expr>> {
        let args = self.parse_args()?;
        let rparen_span = &self.prev().span;

        Some(spanned(Expr::Call(name, args), Span::merge(&name_span, &rparen_span)))
    }

    fn parse_var_or_call(&mut self, ident: Id, ident_span: Span) -> Option<Spanned<Expr>> {
        self.next();

        if self.consume(&Token::Lparen) {
            self.parse_call(ident, ident_span)
        } else {
            Some(spanned(Expr::Variable(ident), ident_span.clone()))
        }
    }

    fn parse_tuple(&mut self, lparen_span: &Span, first: Spanned<Expr>) -> Option<Spanned<Expr>> {
        let mut inner = vec![first];

        while self.peek().kind != Token::Rparen && self.consume(&Token::Comma) {
            if self.peek().kind == Token::Rparen {
                break;
            }

            let expr = self.parse_skip(Self::parse_expr, &[Token::Comma, Token::Rparen])?;
            inner.push(expr);
        }

        self.expect(&Token::Rparen, &[Token::Rparen])?;

        let rparen_span = &self.prev().span;
        let span = Span::merge(&lparen_span, rparen_span);

        Some(spanned(Expr::Tuple(inner), span))
    }

    fn parse_primary(&mut self) -> Option<Spanned<Expr>> {
        let token = self.peek().clone();

        match token.kind {
            Token::Number(n) => {
                self.next();
                Some(spanned(Expr::Literal(Literal::Number(n)), token.span))
            },
            Token::String(s) => {
                self.next();
                Some(spanned(Expr::Literal(Literal::String(s)), token.span))
            },
            Token::Identifier(name) => self.parse_var_or_call(name, token.span),
            Token::True => {
                self.next();
                Some(spanned(Expr::Literal(Literal::True), token.span))
            },
            Token::False => {
                self.next();
                Some(spanned(Expr::Literal(Literal::False), token.span))
            },
            Token::Lparen => {
                let lparen_span = self.peek().span.clone();
                self.next();

                let mut expr = self.parse_expr()?;

                if self.peek().kind == Token::Comma {
                    self.parse_tuple(&lparen_span, expr)
                } else {
                    self.expect(&Token::Rparen, &[Token::Rparen])?;
                    let rparen_span = &self.prev().span;

                    expr.span = Span::merge(&lparen_span, rparen_span);

                    Some(expr)
                }
            },
            _ => {
                error!(self, token.span, "expected `number`, `identifier`, `true`, `false` or `(` but got `{}`", self.peek().kind);
                None
            }
        }
    }

    fn parse_mul(&mut self) -> Option<Spanned<Expr>> {
        self.parse_binop(true, Self::parse_primary, &[
            (&Token::Asterisk, &BinOp::Mul),
            (&Token::Div, &BinOp::Div),
        ])
    }

    fn parse_add(&mut self) -> Option<Spanned<Expr>> {
        self.parse_binop(true, Self::parse_mul, &[
            (&Token::Add, &BinOp::Add),
            (&Token::Sub, &BinOp::Sub),
        ])
    }

    fn parse_relational(&mut self) -> Option<Spanned<Expr>> {
        self.parse_binop(false, Self::parse_add, &[
            (&Token::LessThan, &BinOp::LessThan),
            (&Token::LessThanOrEqual, &BinOp::LessThanOrEqual),
            (&Token::GreaterThan, &BinOp::GreaterThan),
            (&Token::GreaterThanOrEqual, &BinOp::GreaterThanOrEqual),
        ])
    }

    fn parse_equality(&mut self) -> Option<Spanned<Expr>> {
        self.parse_binop(false, Self::parse_relational, &[
            (&Token::Equal, &BinOp::Equal),
            (&Token::NotEqual, &BinOp::NotEqual),
        ])
    }

    fn parse_and(&mut self) -> Option<Spanned<Expr>> {
        self.parse_binop(true, Self::parse_equality, &[
            (&Token::And, &BinOp::And),
        ])
    }

    fn parse_or(&mut self) -> Option<Spanned<Expr>> {
        self.parse_binop(true, Self::parse_and, &[
            (&Token::Or, &BinOp::Or),
        ])
    }

    fn parse_expr(&mut self) -> Option<Spanned<Expr>> {
        self.parse_or()
    }

    fn parse_bind_stmt(&mut self) -> Option<Spanned<Stmt>> {
        // Eat "let"
        let let_span = self.peek().span.clone();
        self.next();

        // Bind name
        let name = self.expect_identifier(&[Token::Semicolon])?;

        self.expect(&Token::Assign, &[Token::Semicolon])?;

        // Initial expression
        let expr = match self.parse_expr() {
            Some(expr) => expr,
            None => {
                self.skip_to(&[Token::Semicolon]);
                return None;
            },
        };

        self.expect(&Token::Semicolon, &[Token::Semicolon])?;
        let semicolon_span = &self.prev().span;

        let span = Span::merge(&let_span, semicolon_span);

        Some(spanned(Stmt::Bind(name, expr), span))
    }

    fn parse_expr_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let expr = match self.parse_expr() {
            Some(expr) => expr,
            None => {
                self.skip_to(&[Token::Semicolon]);
                return None;
            },
        };

        self.expect(&Token::Semicolon, &[Token::Semicolon])?;
        let semicolon_span = &self.prev().span;

        let span = Span::merge(&expr.span, semicolon_span);

        Some(spanned(Stmt::Expr(expr), span))
    }

    // Return None if reach EOF
    fn parse_multiple_statements(&mut self, end_token: &Token) -> Option<Vec<Spanned<Stmt>>> {
        let mut stmts = Vec::new();

        while self.peek().kind != *end_token {
            if let Token::EOF = self.peek().kind {
                error!(self, self.peek().span.clone(), "expected `{}`, but got EOF", end_token);
                return None;
            }

            // Skip semicolon
            if self.consume(&Token::Semicolon) {
                continue;
            }

            if let Some(stmt) = self.parse_stmt() {
                stmts.push(stmt);
            }
        }

        Some(stmts)
    }

    fn parse_block(&mut self) -> Option<Spanned<Stmt>> {
        // Eat "{"
        let lbrace_span = self.peek().span.clone();
        self.next();

        let stmts = self.parse_multiple_statements(&Token::Rbrace)?;

        // Eat "}"
        let rbrace_span = self.peek().span.clone();
        self.next();

        let span = Span::merge(&lbrace_span, &rbrace_span);
        Some(spanned(Stmt::Block(stmts), span))
    }

    fn expect_block(&mut self) -> Option<Spanned<Stmt>> {
        if self.peek().kind != Token::Lbrace {
            error!(self, self.peek().span.clone(), "expected block");
            return None;
        }

        let block = self.parse_block()?;

        Some(block)
    }

    fn parse_return(&mut self) -> Option<Spanned<Stmt>> {
        // Eat "return"
        let return_token_span = self.peek().span.clone();
        self.next();

        let expr = self.parse_skip(Self::parse_expr, &[Token::Semicolon])?;

        self.expect(&Token::Semicolon, &[Token::Semicolon])?;
        let semicolon_span = &self.prev().span;

        let span = Span::merge(&return_token_span, &semicolon_span);

        Some(spanned(Stmt::Return(expr),span))
    }

    fn parse_if_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let if_token_span = self.peek().span.clone();
        self.next();

        // Parse condition expression
        let expr = self.parse_skip(Self::parse_expr, &[Token::Lbrace, Token::Else]);
        // Parse then-clause
        let stmt = self.expect_block()?;

        // Parse else-clause
        let else_stmt = if self.consume(&Token::Else) {
            if self.peek().kind == Token::If {
                Some(self.parse_skip(Self::parse_if_stmt, &[Token::Rbrace])?)
            } else {
                Some(self.expect_block()?)
            }
        } else {
            None
        }.map(|stmt| Box::new(stmt));

        let span = Span::merge(&if_token_span, &stmt.span);
        Some(spanned(Stmt::If(expr?, Box::new(stmt), else_stmt), span))
    }

    fn parse_while_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let while_token_span = self.peek().span.clone();
        self.next();

        // Parse condition expression
        let cond = self.parse_skip(Self::parse_expr, &[Token::Lbrace]);
        // Parse body
        let stmt = self.expect_block()?;

        let span = Span::merge(&while_token_span, &stmt.span);
        Some(spanned(Stmt::While(cond?, Box::new(stmt)), span))
    }

    fn parse_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let token = self.peek();

        match token.kind {
            Token::Let => self.parse_bind_stmt(),
            Token::Lbrace => self.parse_block(),
            Token::Return => self.parse_return(),
            Token::If => self.parse_if_stmt(),
            Token::While => self.parse_while_stmt(),
            _ => self.parse_expr_stmt(),
        }
    }

    fn parse_type(&mut self) -> Option<Type> {
        let result = match self.peek().kind {
            Token::Int => Some(Type::Int),
            Token::Bool => Some(Type::Bool),
            Token::StringType => Some(Type::String),
            Token::Lparen => {
                self.next();

                if self.consume(&Token::Rparen) {
                    error!(self, self.prev().span.clone(), "tuple has to one type at least");
                    None
                } else {
                    let mut inner = Vec::new();

                    let ty = self.parse_skip(Self::parse_type, &[Token::Rparen])?;
                    inner.push(ty);

                    while self.consume(&Token::Comma) && !self.consume(&Token::Rparen) {
                        if self.consume(&Token::Rparen) {
                            break;
                        }

                        let ty = self.parse_skip(Self::parse_type, &[Token::Comma, Token::Rparen])?;
                        inner.push(ty);
                    }

                    Some(Type::Tuple(inner))
                }
            },
            _ => {
                error!(self, self.peek().span.clone(), "expected `int`, `bool`, `string` or `(` but got `{}`", self.peek().kind);
                None
            },
        };

        self.next();

        result
    }

    fn parse_param(&mut self) -> Option<(Id, Type)> {
        let tokens_to_skip = [Token::Comma, Token::Rparen];

        // Parse the parameter name
        let name = self.expect_identifier(&tokens_to_skip)?;

        // Parse the parameter type
        self.expect(&Token::Colon, &tokens_to_skip)?;
        let ty = self.parse_skip(Self::parse_type, &tokens_to_skip)?;

        Some((name, ty))
    }

    fn parse_param_list(&mut self) -> Option<Vec<(Id, Type)>> {
        let mut params = Vec::new();

        // Parse the first parameter
        if let Some(param) = self.parse_param() {
            params.push(param);
        }

        // Parse other parameters
        while !self.consume(&Token::Rparen) {
            if let None = self.expect(&Token::Comma, &[Token::Rparen]) {
                break;
            }

            if let Some(param) = self.parse_param() {
                params.push(param);
            }
        }

        if self.peek().kind == Token::EOF {
            None
        } else {
            Some(params)
        }
    }

    fn parse_fn_decl(&mut self) -> Option<Spanned<TopLevel>> {
        // Eat "fn"
        let fn_span = self.peek().span.clone();
        self.next();

        // Parse the function name
        let name = self.expect_identifier(&[Token::Lparen]);

        // Parse parameters
        self.expect(&Token::Lparen, &[Token::Lparen]);
        let params = self.parse_param_list()?;

        // Parse the return type
        let return_ty = if self.consume(&Token::Colon) {
            self.parse_skip(Self::parse_type, &[Token::Lbrace])
        } else {
            // If omit type, the return type is void
            Some(Type::Int) // TODO: Void
        };

        let body = self.expect_block()?;

        let span = Span::merge(&fn_span, &body.span);
        Some(spanned(TopLevel::Function(name?, params, return_ty?, body), span))
    }

    fn parse_toplevel(&mut self) -> Option<Spanned<TopLevel>> {
        match self.peek().kind {
            Token::Fn => self.parse_fn_decl(),
            _ => self.parse_stmt().map(|stmt| {
                let span = stmt.span.clone();
                spanned(TopLevel::Stmt(stmt), span)
            }),
        }
    }

    pub fn parse(mut self) -> Result<Program, Vec<Error>> {
        let mut toplevels = Vec::new();

        while self.peek().kind != Token::EOF {
            if self.consume(&Token::Semicolon) {
                continue;
            }

            if let Some(toplevel) = self.parse_toplevel() {
                toplevels.push(toplevel);
            }
        }

        if self.errors.len() > 0 {
            Err(self.errors)
        } else {
            Ok(Program { top: toplevels })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::id::{Id, IdMap};
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

        fn newt<T>(kind: T, start_line: u32, start_col: u32, end_line: u32, end_col: u32) -> Box<Spanned<T>> {
            let spanned = new(kind, start_line, start_col, end_line, end_col);
            Box::new(Spanned::new(spanned.kind, spanned.span))
        }

        let mut id_map = IdMap::new();
        let lexer = Lexer::new(r#"let abc = 10 + 3 * (5 + 20); abc; { abc; 10; }
fn add(a: int, b: int): int { a + b; }
add(3, 5 + 8)  ;
return abc;"#, &mut id_map);
        let tokens = lexer.lex().unwrap();

        let mut id = |id: &str| -> Id {
            id_map.new_id(id)
        };

        let parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let expected = Program {
            top: vec![
                *new(TopLevel::Stmt(
                    *new(Stmt::Bind(
                        id("abc"),
                        *newt(Expr::BinOp(BinOp::Add,
                              newt(Expr::Literal(Literal::Number(10)), 0, 10, 0, 12),
                              newt(Expr::BinOp(BinOp::Mul,
                                  newt(Expr::Literal(Literal::Number(3)), 0, 15, 0, 16),
                                  newt(Expr::BinOp(BinOp::Add,
                                      newt(Expr::Literal(Literal::Number(5)), 0, 20, 0, 21),
                                      newt(Expr::Literal(Literal::Number(20)), 0, 24, 0, 26)),
                                      0, 19, 0, 27)),
                                  0, 15, 0, 27)),
                              0, 10, 0, 27)),
                        0, 0, 0, 28)),
                    0, 0, 0, 28),
                *new(TopLevel::Stmt(
                    *new(Stmt::Expr(
                        *newt(Expr::Variable(id("abc")), 0, 29, 0, 32)),
                        0, 29, 0, 33)),
                    0, 29, 0, 33),
                *new(TopLevel::Stmt(
                    *new(Stmt::Block(vec![
                        *new(Stmt::Expr(
                            *newt(Expr::Variable(id("abc")), 0, 36, 0, 39)),
                            0, 36, 0, 40),
                        *new(Stmt::Expr(
                            *newt(Expr::Literal(Literal::Number(10)), 0, 41, 0, 43)),
                            0, 41, 0, 44)]),
                        0, 34, 0, 46)),
                    0, 34, 0, 46),
                *new(TopLevel::Function(id("add"), vec![(id("a"), Type::Int), (id("b"), Type::Int)], Type::Int,
                    *new(Stmt::Block(vec![
                        *new(Stmt::Expr(
                            *newt(Expr::BinOp(BinOp::Add,
                                newt(Expr::Variable(id("a")), 1, 30, 1, 31),
                                newt(Expr::Variable(id("b")), 1, 34, 1, 35)),
                                1, 30, 1, 35)),
                            1, 30, 1, 36)]),
                        1, 28, 1, 38)),
                    1, 0, 1, 38),
                *new(TopLevel::Stmt(
                    *new(Stmt::Expr(
                        *newt(Expr::Call(id("add"), vec![
                            *newt(Expr::Literal(Literal::Number(3)), 2, 4, 2, 5),
                            *newt(Expr::BinOp(BinOp::Add,
                                newt(Expr::Literal(Literal::Number(5)), 2, 7, 2, 8),
                                newt(Expr::Literal(Literal::Number(8)), 2, 11, 2, 12)),
                                2, 7, 2, 12)]),
                            2, 0, 2, 13)),
                        2, 0, 2, 16)),
                    2, 0, 2, 16),
                *new(TopLevel::Stmt(
                    *new(Stmt::Return(
                        *newt(Expr::Variable(id("abc")), 3, 7, 3, 10)),
                        3, 0, 3, 11)),
                    3, 0, 3, 11),
            ],
        };

        assert_eq!(program, expected);
    }
}
