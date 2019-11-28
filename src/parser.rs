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
        if res.is_none() {
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
            while self.peek().kind != Token::Rparen && self.consume(&Token::Comma) {
                if self.peek().kind == Token::Rparen {
                    break;
                }

                match self.parse_expr() {
                    Some(expr) => args.push(expr),
                    None => self.skip_to(&[Token::Comma, Token::Rparen]),
                };
            }

            self.expect(&Token::Rparen, &[Token::Rparen])?;
            Some(args)
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

                if self.consume(&Token::Rparen) {
                    // Unit literal
                    let span = Span::merge(&lparen_span, &self.prev().span);
                    Some(spanned(Expr::Literal(Literal::Unit), span))
                } else {
                    let mut expr = self.parse_expr()?;

                    if self.peek().kind == Token::Comma {
                        self.parse_tuple(&lparen_span, expr)
                    } else {
                        self.expect(&Token::Rparen, &[Token::Rparen])?;
                        let rparen_span = &self.prev().span;

                        expr.span = Span::merge(&lparen_span, rparen_span);

                        Some(expr)
                    }
                }
            },
            _ => {
                error!(self, token.span, "expected `number`, `identifier`, `true`, `false` or `(` but got `{}`", self.peek().kind);
                None
            }
        }
    }

    fn parse_field(&mut self) -> Option<Spanned<Expr>> {
        let parse = Self::parse_primary;
        let mut expr = parse(self)?;

        loop {
            if self.consume(&Token::Dot) {
                match self.peek().kind {
                    Token::Number(n) => {
                        self.next();

                        if n < 0 {
                            error!(self, self.peek().span.clone(), "field of negative number");
                            return None;
                        }

                        let span = Span::merge(&expr.span, &self.prev().span);
                        expr = spanned(Expr::Field(Box::new(expr), Field::Number(n as usize)), span);
                    },
                    _ => {
                        error!(self, self.peek().span.clone(), "expected `number` but got `{}`", self.peek().kind);
                    }
                }
            } else {
                break;
            }
        }

        Some(expr)
    }

    #[inline]
    fn parse_unary_op<P, F>(&mut self, mut parse: P, f: F) -> Option<Spanned<Expr>>
        where P: FnMut(&mut Self) -> Option<Spanned<Expr>>,
              F: Fn(Box<Spanned<Expr>>) -> Expr
    {
        let symbol_span = self.peek().span.clone();
        self.next();

        let expr = parse(self)?;
        let span = Span::merge(&symbol_span, &expr.span);
        Some(spanned(f(Box::new(expr)), span))
    }

    fn parse_unary(&mut self) -> Option<Spanned<Expr>> {
        let parse = Self::parse_field;

        match self.peek().kind {
            Token::Ampersand => self.parse_unary_op(parse, |expr| Expr::Address(expr)),
            Token::Asterisk => self.parse_unary_op(parse, |expr| Expr::Dereference(expr)),
            Token::Sub => self.parse_unary_op(parse, |expr| Expr::Negative(expr)),
            _ => parse(self),
        }
    }

    fn parse_mul(&mut self) -> Option<Spanned<Expr>> {
        self.parse_binop(true, Self::parse_unary, &[
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

        let is_mutable = self.consume(&Token::Mut);

        // Bind name
        let name = self.expect_identifier(&[Token::Semicolon])?;

        self.expect(&Token::Equal, &[Token::Semicolon])?;

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

        Some(spanned(Stmt::Bind(name, expr, is_mutable), span))
    }

    fn parse_expr_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let expr = self.parse_skip(Self::parse_expr, &[Token::Semicolon, Token::Assign])?;

        if self.consume(&Token::Assign) {
            let rhs = self.parse_skip(Self::parse_expr, &[Token::Semicolon])?;

            self.expect(&Token::Semicolon, &[Token::Semicolon])?;
            let semicolon_span = &self.prev().span;

            let span = Span::merge(&expr.span, semicolon_span);

            Some(spanned(Stmt::Assign(expr, rhs), span))
        } else {
            self.expect(&Token::Semicolon, &[Token::Semicolon])?;
            let semicolon_span = &self.prev().span;

            let span = Span::merge(&expr.span, semicolon_span);

            Some(spanned(Stmt::Expr(expr), span))
        }
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

        if self.consume(&Token::Semicolon) {
            let span = Span::merge(&return_token_span, &self.prev().span);
            Some(spanned(Stmt::Return(None), span))
        } else {
            let expr = self.parse_skip(Self::parse_expr, &[Token::Semicolon])?;

            self.expect(&Token::Semicolon, &[Token::Semicolon])?;
            let semicolon_span = &self.prev().span;

            let span = Span::merge(&return_token_span, &semicolon_span);

            Some(spanned(Stmt::Return(Some(expr)), span))
        }
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
        }.map(Box::new);

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

    fn parse_type_pointer(&mut self) -> Option<Type> {
        self.next(); // eat '*'
        let ty = self.parse_type()?;

        // FIXME: remove later
        self.pos -= 1;

        Some(Type::Pointer(Box::new(ty)))
    }

    fn parse_type_tuple(&mut self) -> Option<Type> {
        self.next(); // eat "("

        if self.peek().kind == Token::Rparen {
            Some(Type::Unit)
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
    }

    fn parse_struct_field(&mut self) -> Option<(Id, Type)> {
        let tokens_to_skip = &[Token::Comma, Token::Rbrace];

        // name: ty
        let name = self.expect_identifier(tokens_to_skip)?;
        self.expect(&Token::Colon, tokens_to_skip)?;
        let ty = self.parse_skip(Self::parse_type, tokens_to_skip)?;

        Some((name, ty))
    }

    fn parse_type_struct(&mut self) -> Option<Type> {
        // Eat "struct"
        self.next();

        self.expect(&Token::Lbrace, &[Token::Rbrace])?;

        if self.consume(&Token::Rbrace) {
            Some(Type::Struct(Vec::new()))
        } else {
            let mut fields = Vec::new();

            let first = self.parse_struct_field();
            if let Some(first) = first {
                fields.push(first);
            }

            while self.consume(&Token::Comma) && self.peek().kind != Token::Rbrace {
                if self.peek().kind == Token::Rbrace {
                    break;
                }

                let field = self.parse_struct_field();
                if let Some(field) = field {
                    fields.push(field);
                }
            }

            Some(Type::Struct(fields))
        }
    }

    fn parse_type(&mut self) -> Option<Type> {
        let result = match self.peek().kind {
            Token::Int => Some(Type::Int),
            Token::Bool => Some(Type::Bool),
            Token::StringType => Some(Type::String),
            Token::Identifier(name) => Some(Type::Named(name)),
            Token::Asterisk => self.parse_type_pointer(),
            Token::Lparen => self.parse_type_tuple(), // tuple
            Token::Struct => self.parse_type_struct(), // struct
            _ => {
                error!(self,
                    self.peek().span.clone(),
                    "expected `int`, `bool`, `string`, `(`, `struct` or `identifier` but got `{}`",
                    self.peek().kind
                );
                None
            },
        };

        self.next();

        result
    }

    fn parse_param(&mut self) -> Option<(Id, Type, bool)> {
        let tokens_to_skip = [Token::Comma, Token::Rparen];

        let is_mutable = self.consume(&Token::Mut);

        // Parse the parameter name
        let name = self.expect_identifier(&tokens_to_skip)?;

        // Parse the parameter type
        self.expect(&Token::Colon, &tokens_to_skip)?;
        let ty = self.parse_skip(Self::parse_type, &tokens_to_skip)?;

        Some((name, ty, is_mutable))
    }

    fn parse_param_list(&mut self) -> Option<Vec<(Id, Type, bool)>> {
        let mut params = Vec::new();

        if self.consume(&Token::Rparen) {
            return Some(params);
        }

        // Parse the first parameter
        if let Some(param) = self.parse_param() {
            params.push(param);
        }

        // Parse other parameters
        while self.peek().kind != Token::Rparen && self.consume(&Token::Comma) {
            if self.peek().kind == Token::Rparen {
                break;
            }

            if let Some(param) = self.parse_param() {
                params.push(param);
            }
        }

        self.expect(&Token::Rparen, &[Token::Rparen])?;
        Some(params)
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
            // return unit if omit a type
            Some(Type::Unit)
        };

        let body = self.expect_block()?;

        let span = Span::merge(&fn_span, &body.span);
        Some(spanned(TopLevel::Function(name?, params, return_ty?, body), span))
    }

    fn parse_def_type(&mut self) -> Option<Spanned<TopLevel>> {
        let type_span = self.peek().span.clone();
        self.next(); // eat "type"

        let name = self.expect_identifier(&[Token::Semicolon])?;
        let ty = self.parse_skip(Self::parse_type, &[Token::Semicolon])?;

        self.expect(&Token::Semicolon, &[Token::Semicolon])?;

        let span = Span::merge(&type_span, &self.prev().span);
        Some(spanned(TopLevel::Type(name, ty), span))
    }

    fn parse_toplevel(&mut self) -> Option<Spanned<TopLevel>> {
        match self.peek().kind {
            Token::Fn => self.parse_fn_decl(),
            Token::Type => self.parse_def_type(),
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

        if !self.errors.is_empty() {
            Err(self.errors)
        } else {
            Ok(Program { top: toplevels })
        }
    }
}
