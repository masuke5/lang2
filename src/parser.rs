use std::convert::TryFrom;

use crate::span::{Span, Spanned};
use crate::error::Error;
use crate::token::*;
use crate::ast::*;
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

    main_stmts: Vec<Spanned<Stmt>>,
    functions: Vec<AstFunction>,
    types: Vec<AstTypeDef>,
    strings: Vec<String>,
}

impl Parser {
    pub fn new(tokens: Vec<Spanned<Token>>) -> Parser {
        Self {
            tokens,
            pos: 0,
            errors: Vec::new(),
            main_stmts: Vec::new(),
            functions: Vec::new(),
            types: Vec::new(),
            strings: Vec::new(),
        }
    }

    #[inline]
    fn next(&mut self) -> &Spanned<Token> {
        self.pos += 1;
        &self.tokens[self.pos]
    }

    #[inline]
    fn next_and<T>(&mut self, value: T) -> T {
        self.next();
        value
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
    fn next_token(&self) -> &Spanned<Token> {
        &self.tokens[self.pos + 1]
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

    fn parse_call(&mut self, name: Id, name_span: Span, tyargs: Vec<Spanned<AstType>>) -> Option<Spanned<Expr>> {
        let args = self.parse_args()?;
        let rparen_span = &self.prev().span;

        Some(spanned(Expr::Call(name, args, tyargs), Span::merge(&name_span, &rparen_span)))
    }

    fn parse_field_init(&mut self) -> Option<(Spanned<Id>, Spanned<Expr>)> {
        let tokens_to_skip = &[Token::Comma, Token::Rbrace];

        let name = self.expect_identifier(tokens_to_skip)?;
        let name = spanned(name, self.prev().span.clone());

        self.expect(&Token::Colon, tokens_to_skip)?;

        let expr = self.parse_skip(Self::parse_expr, tokens_to_skip)?;

        Some((name, expr))
    }

    fn parse_struct(&mut self, name: Id, name_span: Span) -> Option<Spanned<Expr>> {
        let ty = self.parse_type_app(spanned(name, name_span.clone()));

        self.expect(&Token::Lbrace, &[Token::Lbrace]);

        if self.consume(&Token::Rbrace) {
            let span = Span::merge(&name_span, &self.prev().span);
            Some(spanned(Expr::Struct(ty, Vec::new()), span))
        } else {
            let mut fields = Vec::new();

            match self.parse_field_init() {
                Some(field) => fields.push(field),
                None => self.skip_to(&[Token::Comma, Token::Rbrace]),
            };

            while self.peek().kind != Token::Rbrace && self.consume(&Token::Comma) {
                if self.peek().kind == Token::Rbrace {
                    break;
                }

                match self.parse_field_init() {
                    Some(field) => fields.push(field),
                    None => self.skip_to(&[Token::Comma, Token::Rbrace]),
                };
            }

            self.expect(&Token::Rbrace, &[Token::Rbrace])?;
            

            let span = Span::merge(&name_span, &self.prev().span);
            Some(spanned(Expr::Struct(ty, fields), span))
        }
    }

    fn parse_var_or_call(&mut self, ident: Id, ident_span: Span) -> Option<Spanned<Expr>> {
        self.next();

        if self.peek().kind == Token::Dot && self.next_token().kind == Token::LessThan {
            self.next();

            // parse_type_args() returns None only when the current token is not Token::LessThan
            let tyargs = self.parse_type_args().unwrap();
            self.expect(&Token::Lparen, &[Token::Lparen]);

            self.parse_call(ident, ident_span, tyargs)
        } else if self.consume(&Token::Lparen) {
            self.parse_call(ident, ident_span, Vec::new())
        } else if self.consume(&Token::Colon) {
            self.parse_struct(ident, ident_span)
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

    fn parse_array(&mut self) -> Option<Spanned<Expr>> {
        let lbracket_span = self.peek().span.clone();
        self.next(); // eat "["

        let init_expr = self.parse_expr()?;

        self.expect(&Token::Semicolon, &[Token::Semicolon]);

        let size = match self.peek().kind {
            Token::Number(size) => match usize::try_from(size) {
                Ok(size) => size,
                Err(_) => {
                    error!(self, self.peek().span.clone(), "too large");
                    0
                },
            },
            _ => return None,
        };
        self.next();

        self.expect(&Token::Rbracket, &[Token::Rbracket]);

        let span = Span::merge(&lbracket_span, &self.prev().span);
        Some(spanned(Expr::Array(Box::new(init_expr), size), span))
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
                self.strings.push(s);
                Some(spanned(Expr::Literal(Literal::String(self.strings.len() - 1)), token.span))
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
            Token::Null => {
                self.next();
                Some(spanned(Expr::Literal(Literal::Null), token.span))
            },
            Token::Lbracket => self.parse_array(),
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
                    Token::Identifier(id) => {
                        self.next();
                        let span = Span::merge(&expr.span, &self.prev().span);
                        expr = spanned(Expr::Field(Box::new(expr), Field::Id(id)), span);
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

    fn parse_subscript(&mut self) -> Option<Spanned<Expr>> {
        let parse = Self::parse_field;

        let mut expr = parse(self)?;

        loop {
            if self.consume(&Token::Lbracket) {
                let subscript = parse(self)?;

                self.expect(&Token::Rbracket, &[Token::Rbracket])?;
                
                let span = Span::merge(&expr.span, &self.prev().span);
                expr = spanned(Expr::Subscript(Box::new(expr), Box::new(subscript)), span);
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
        let parse = Self::parse_subscript;

        match self.peek().kind {
            Token::Asterisk => self.parse_unary_op(parse, |expr| Expr::Dereference(expr)),
            Token::Sub => self.parse_unary_op(parse, |expr| Expr::Negative(expr)),
            Token::Ampersand => {
                let symbol_span = self.peek().span.clone();
                self.next();

                let is_mutable = self.consume(&Token::Mut);
                let expr = parse(self)?;

                let span = Span::merge(&symbol_span, &expr.span);
                Some(spanned(Expr::Address(Box::new(expr), is_mutable), span))
            },
            Token::New => {
                let symbol_span = self.peek().span.clone();
                self.next();

                let is_mutable = self.consume(&Token::Mut);
                let expr = parse(self)?;

                let span = Span::merge(&symbol_span, &expr.span);
                Some(spanned(Expr::Alloc(Box::new(expr), is_mutable), span))
            },
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

    fn parse_type_pointer(&mut self) -> Option<Spanned<AstType>> {
        let asterisk_span = self.peek().span.clone();
        self.next(); // eat '*'

        let is_mutable = self.consume(&Token::Mut);
        let ty = self.parse_type()?;

        let span = Span::merge(&asterisk_span, &self.prev().span);
        Some(spanned(AstType::Pointer(Box::new(ty), is_mutable), span))
    }

    fn parse_type_tuple(&mut self) -> Option<Spanned<AstType>> {
        let lparen_span = self.peek().span.clone();
        self.next(); // eat "("

        if self.consume(&Token::Rparen) {
            let span = Span::merge(&lparen_span, &self.prev().span);
            Some(spanned(AstType::Unit, span))
        } else {
            let mut inner = Vec::new();

            let ty = self.parse_skip(Self::parse_type, &[Token::Comma, Token::Rparen]);
            if let Some(ty) = ty {
                inner.push(ty);
            }

            while self.peek().kind != Token::Rparen && self.consume(&Token::Comma) {
                if self.peek().kind == Token::Rparen {
                    break;
                }

                let ty = self.parse_skip(Self::parse_type, &[Token::Comma, Token::Rparen]);
                if let Some(ty) = ty {
                    inner.push(ty);
                }
            }

            self.expect(&Token::Rparen, &[Token::Rparen]);

            let span = Span::merge(&lparen_span, &self.prev().span);
            Some(spanned(AstType::Tuple(inner), span))
        }
    }

    fn parse_struct_field(&mut self) -> Option<(Spanned<Id>, Spanned<AstType>)> {
        let tokens_to_skip = &[Token::Comma, Token::Rbrace];

        // name: ty
        let span = self.peek().span.clone();
        let name = self.expect_identifier(tokens_to_skip)?;
        let name = spanned(name, span);

        self.expect(&Token::Colon, tokens_to_skip)?;
        let ty = self.parse_skip(Self::parse_type, tokens_to_skip)?;

        Some((name, ty))
    }

    fn parse_type_struct(&mut self) -> Option<Spanned<AstType>> {
        // Eat "struct"
        let struct_span = self.peek().span.clone();
        self.next();

        self.expect(&Token::Lbrace, &[Token::Rbrace])?;

        if self.consume(&Token::Rbrace) {
            // Empty structure
            let span = Span::merge(&struct_span, &self.peek().span);
            Some(spanned(AstType::Struct(Vec::new()), span))
        } else {
            let mut fields = Vec::new();

            let first = self.parse_struct_field();
            if let Some(field) = first {
                fields.push(field);
            }

            while self.peek().kind != Token::Rbrace && self.consume(&Token::Comma) {
                if self.peek().kind == Token::Rbrace {
                    break;
                }

                let field = self.parse_struct_field();
                if let Some(field) = field {
                    fields.push(field);
                }
            }

            self.expect(&Token::Rbrace, &[Token::Rbrace]);

            let span = Span::merge(&struct_span, &self.peek().span);
            Some(spanned(AstType::Struct(fields), span))
        }
    }

    fn parse_type_array(&mut self) -> Option<Spanned<AstType>> {
        let lbracket_span = self.peek().span.clone();
        self.next(); // eat "["

        let ty = self.parse_skip(Self::parse_type, &[Token::Rbracket])?;

        self.expect(&Token::Semicolon, &[Token::Rbracket])?;

        let size = match self.peek().kind {
            Token::Number(size) => match usize::try_from(size) {
                Ok(size) => size,
                Err(_) => {
                    error!(self, self.peek().span.clone(), "too large");
                    0
                },
            },
            _ => return None,
        };
        self.next();

        self.expect(&Token::Rbracket, &[Token::Rbracket]);

        let span = Span::merge(&lbracket_span, &self.peek().span);
        Some(spanned(AstType::Array(Box::new(ty), size), span))
    }

    fn parse_type_args(&mut self) -> Option<Vec<Spanned<AstType>>> {
        if !self.consume(&Token::LessThan) {
            return None;
        }

        if self.consume(&Token::GreaterThan) {
            Some(Vec::new())
        } else {
            let mut types = Vec::new();

            let first = self.parse_skip(Self::parse_type, &[Token::GreaterThan, Token::Comma]);
            if let Some(first) = first {
                types.push(first);
            }

            while self.peek().kind != Token::GreaterThan && self.consume(&Token::Comma) {
                if self.peek().kind == Token::GreaterThan {
                    break;
                }

                if let Some(ty) = self.parse_skip(Self::parse_type, &[Token::GreaterThan, Token::Comma]) {
                    types.push(ty);
                }
            }

            self.expect(&Token::GreaterThan, &[Token::GreaterThan]);

            Some(types)
        }
    }

    fn parse_type_app(&mut self, id: Spanned<Id>) -> Spanned<AstType> {
        let lessthan_span = self.peek().span.clone();
        let args = self.parse_type_args().unwrap_or(Vec::new());

        if args.is_empty() {
            spanned(AstType::Named(id.kind), id.span)
        } else {
            let span = Span::merge(&lessthan_span, &self.peek().span);
            spanned(AstType::App(id, args), span)
        }
    }

    fn parse_type(&mut self) -> Option<Spanned<AstType>> {
        let first_span = self.peek().span.clone();
        match self.peek().kind {
            Token::Int => self.next_and(Some(spanned(AstType::Int, first_span))),
            Token::Bool => self.next_and(Some(spanned(AstType::Bool, first_span))),
            Token::StringType => self.next_and(Some(spanned(AstType::String, first_span))),
            Token::Identifier(name) => {
                let id = self.next_and(spanned(name, first_span));
                let ty = self.parse_type_app(id);
                Some(ty)
            },
            Token::Asterisk => self.parse_type_pointer(),
            Token::Lparen => self.parse_type_tuple(), // tuple
            Token::Struct => self.parse_type_struct(), // struct
            Token::Lbracket => self.parse_type_array(), // array
            _ => {
                error!(self,
                    self.peek().span.clone(),
                    "expected `(`, `[`, `*`, `int`, `bool`, `string`, `struct` or `identifier` but got `{}`",
                    self.peek().kind
                );
                self.next();
                None
            },
        }
    }

    fn parse_param(&mut self) -> Option<Param> {
        let tokens_to_skip = [Token::Comma, Token::Rparen];

        let is_mutable = self.consume(&Token::Mut);

        // Parse the parameter name
        let name = self.expect_identifier(&tokens_to_skip)?;

        // Parse the parameter type
        self.expect(&Token::Colon, &tokens_to_skip)?;
        let ty = self.parse_skip(Self::parse_type, &tokens_to_skip)?;

        Some(Param {
            name,
            ty,
            is_mutable,
        })
    }

    fn parse_param_list(&mut self) -> Option<Vec<Param>> {
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

        self.expect(&Token::Rparen, &[Token::Rparen]);

        Some(params)
    }

    fn parse_fn_decl(&mut self) -> Option<AstFunction> {
        // Eat "fn"
        self.next();

        // Parse the function name
        let name = self.expect_identifier(&[Token::Lparen]);
        // Parse the type parameters
        let ty_params = self.parse_type_vars();

        // Parse parameters
        self.expect(&Token::Lparen, &[Token::Lparen]);
        let params = self.parse_param_list()?;

        // Parse the return type
        let return_ty = if self.consume(&Token::Colon) {
            self.parse_skip(Self::parse_type, &[Token::Lbrace])
        } else {
            // return unit if omit a type
            Some(spanned(AstType::Unit, self.prev().span.clone()))
        };

        let body = self.expect_block()?;

        Some(AstFunction {
            name: name?,
            params,
            return_ty: return_ty?,
            body,
            ty_params,
        })
    }

    fn parse_type_vars(&mut self) -> Vec<Spanned<Id>> {
        if !self.consume(&Token::LessThan) {
            return Vec::new();
        }

        if self.consume(&Token::GreaterThan) {
            Vec::new()
        } else {
            let mut vars = Vec::new();

            let first = self.expect_identifier(&[Token::GreaterThan, Token::Comma]);
            if let Some(first) = first {
                vars.push(spanned(first, self.prev().span.clone()));
            }

            while self.peek().kind != Token::GreaterThan && self.consume(&Token::Comma) {
                if self.peek().kind == Token::GreaterThan {
                    break;
                }

                if let Some(var) = self.expect_identifier(&[Token::GreaterThan, Token::Comma]) {
                    vars.push(spanned(var, self.prev().span.clone()));
                }
            }

            self.expect(&Token::GreaterThan, &[Token::GreaterThan]);

            vars
        }
    }

    fn parse_def_type(&mut self) -> Option<AstTypeDef> {
        self.next(); // eat "type"

        let name = self.expect_identifier(&[Token::Semicolon])?;
        let var_ids = self.parse_type_vars();

        let ty = self.parse_skip(Self::parse_type, &[Token::Semicolon])?;

        self.expect(&Token::Semicolon, &[Token::Semicolon])?;

        Some(AstTypeDef {
            name,
            ty,
            var_ids,
        })
    }

    fn parse_toplevel(&mut self) -> Option<()> {
        match self.peek().kind {
            Token::Fn => {
                let func = self.parse_fn_decl()?;
                self.functions.push(func);
            },
            Token::Type => {
                let tydef = self.parse_def_type()?;
                self.types.push(tydef);
            },
            _ => {
                let stmt = self.parse_stmt()?;
                self.main_stmts.push(stmt);
            },
        };

        Some(())
    }

    pub fn parse(mut self) -> Result<Program, Vec<Error>> {
        while self.peek().kind != Token::EOF {
            if self.consume(&Token::Semicolon) {
                continue;
            }

            self.parse_toplevel();
        }

        if !self.errors.is_empty() {
            Err(self.errors)
        } else {
            Ok(Program {
                functions: self.functions,
                types: self.types,
                main_stmts: self.main_stmts,
                strings: self.strings,
            })
        }
    }
}
