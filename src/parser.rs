use std::convert::TryFrom;
use std::path::{Path, PathBuf};

use rustc_hash::FxHashSet;

use crate::ast::{
    AstFunction as AstFunction_, Block as Block_, Expr as Expr_, Impl as Impl_,
    Program as Program_, Stmt as Stmt_, *,
};
use crate::error::{Error, ErrorList};
use crate::id::{reserved_id, Id};
use crate::span::{Span, Spanned};
use crate::token::*;

type Expr = Expr_<Empty>;
type UntypedExpr = Typed<Expr, Empty>;
type Stmt = Stmt_<Empty>;
type Block = Block_<Empty>;
type Impl = Impl_<Empty>;
type AstFunction = AstFunction_<Empty>;
type Program = Program_<Empty>;

fn spanned<T>(kind: T, span: Span) -> Spanned<T> {
    Spanned::<T>::new(kind, span)
}

fn new_expr(kind: Expr, span: Span) -> UntypedExpr {
    UntypedExpr::new(kind, span, Empty)
}

fn needs_semicolon(expr: &Expr) -> bool {
    match expr {
        Expr::Block(..) | Expr::If(..) => false,
        _ => true,
    }
}

fn expr_is_callable(expr: &Expr) -> bool {
    needs_semicolon(expr)
}

struct DefinitionInBlock {
    functions: Vec<AstFunction>,
    types: Vec<AstTypeDef>,
}

impl DefinitionInBlock {
    fn new() -> Self {
        Self {
            functions: Vec::new(),
            types: Vec::new(),
        }
    }
}

struct BlockBuilder {
    defs: Vec<DefinitionInBlock>,
}

impl BlockBuilder {
    fn new() -> Self {
        Self { defs: Vec::new() }
    }

    fn push(&mut self) {
        self.defs.push(DefinitionInBlock::new());
    }

    fn pop_and_build(&mut self, stmts: Vec<Spanned<Stmt>>, result_expr: UntypedExpr) -> Block {
        let def = self.defs.pop().unwrap();
        Block {
            functions: def.functions,
            types: def.types,
            stmts,
            result_expr: Box::new(result_expr),
        }
    }

    fn def(&mut self) -> &mut DefinitionInBlock {
        self.defs.last_mut().unwrap()
    }
}

pub struct Parser {
    root_path: PathBuf,
    tokens: Vec<Spanned<Token>>,
    pos: usize,

    main_stmts: Vec<Spanned<Stmt>>,
    impls: Vec<Impl>,

    blocks_builder: BlockBuilder,
}

impl Parser {
    pub fn new(
        root_path: &Path,
        tokens: Vec<Spanned<Token>>,
        loaded_modules: FxHashSet<SymbolPath>,
    ) -> Self {
        Self {
            root_path: root_path.to_path_buf(),
            tokens,
            pos: 0,
            main_stmts: Vec::new(),
            impls: Vec::new(),
            blocks_builder: BlockBuilder::new(),
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
            error!(
                &token.span,
                "expected `{}` but got `{}`", expected, token.kind
            );

            self.skip_until(skip);
            // TODO: Improve it
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
                error!(
                    &token.span,
                    "expected `identifier` but got `{}`", token.kind
                );

                self.skip_until(skip);
                // TODO: Improve it
                if let Token::Identifier(_) = self.peek().kind {
                    self.next();
                }

                None
            }
        }
    }

    #[inline]
    fn skip_until(&mut self, tokens: &[Token]) {
        let tokens_to_dec: Vec<&Token> =
            tokens.iter().filter(|t| t.is_close_parenthese()).collect();
        let tokens_to_inc: Vec<Token> = tokens_to_dec
            .iter()
            .map(|t| t.matching_parenthese().unwrap())
            .collect();
        let mut depth = 0;

        while depth > 0 || (!tokens.contains(&self.peek().kind) && self.peek().kind != Token::EOF) {
            let curr = &self.peek().kind;

            if tokens_to_dec.contains(&curr) {
                depth -= 1;
            } else if tokens_to_inc.contains(&curr) {
                depth += 1;
            }

            self.next();
        }
    }

    // Parse something using `func`. skip to `tokens` if fail.
    fn parse_skip<T, F>(&mut self, mut func: F, tokens: &[Token]) -> Option<T>
    where
        F: FnMut(&mut Self) -> Option<T>,
    {
        let res = func(self);
        if res.is_none() {
            self.skip_until(tokens);
        }

        res
    }

    fn parse_path_with_first_segment(
        &mut self,
        first: Option<SymbolPathSegment>,
        first_span: Span,
    ) -> Option<Spanned<SymbolPath>> {
        let mut path = SymbolPath::new();

        if let Some(first) = first {
            path.segments.push(first);
        }

        while self.consume(&Token::Scope) {
            let id = self.expect_identifier(&[Token::Scope]);
            if let Some(id) = id {
                path.segments.push(SymbolPathSegment::new(id));
            }
        }

        let span = Span::merge(&first_span, &self.prev().span);
        Some(spanned(path, span))
    }

    #[allow(dead_code)]
    fn parse_path(&mut self) -> Option<Spanned<SymbolPath>> {
        let first = self
            .expect_identifier(&[Token::Scope])
            .map(SymbolPathSegment::new);

        self.parse_path_with_first_segment(first, self.prev().span.clone())
    }

    #[inline]
    fn parse_binop<F>(
        &mut self,
        allow_join: bool,
        mut func: F,
        rules: &[(&Token, &BinOp)],
    ) -> Option<UntypedExpr>
    where
        F: FnMut(&mut Self) -> Option<UntypedExpr>,
    {
        let mut expr = func(self)?;

        if allow_join {
            'outer: loop {
                for (token, binop) in rules {
                    if self.consume(token) {
                        let rhs = func(self)?;
                        let span = Span::merge(&expr.span, &rhs.span);

                        expr = new_expr(
                            Expr::BinOp((*binop).clone(), Box::new(expr), Box::new(rhs)),
                            span,
                        );
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

                    expr = new_expr(
                        Expr::BinOp((*binop).clone(), Box::new(expr), Box::new(rhs)),
                        span,
                    );
                    break;
                }
            }
        }

        Some(expr)
    }

    fn parse_field_init(&mut self) -> Option<(Spanned<Id>, UntypedExpr)> {
        let tokens_to_skip = &[Token::Comma, Token::Rbrace];

        let name = self.expect_identifier(tokens_to_skip)?;
        let name = spanned(name, self.prev().span.clone());

        self.expect(&Token::Colon, tokens_to_skip)?;

        let expr = self.parse_skip(Self::parse_expr, tokens_to_skip)?;

        Some((name, expr))
    }

    fn parse_struct(&mut self, name: Id, name_span: Span) -> Option<UntypedExpr> {
        let ty = self.parse_type_app(spanned(name, name_span.clone()));

        self.expect(&Token::Lbrace, &[Token::Lbrace]);

        if self.consume(&Token::Rbrace) {
            let span = Span::merge(&name_span, &self.prev().span);
            Some(new_expr(Expr::Struct(ty, Vec::new()), span))
        } else {
            let mut fields = Vec::new();

            match self.parse_field_init() {
                Some(field) => fields.push(field),
                None => self.skip_until(&[Token::Comma, Token::Rbrace]),
            };

            while self.peek().kind != Token::Rbrace && self.consume(&Token::Comma) {
                if self.peek().kind == Token::Rbrace {
                    break;
                }

                match self.parse_field_init() {
                    Some(field) => fields.push(field),
                    None => self.skip_until(&[Token::Comma, Token::Rbrace]),
                };
            }

            self.expect(&Token::Rbrace, &[Token::Rbrace])?;

            let span = Span::merge(&name_span, &self.prev().span);
            Some(new_expr(Expr::Struct(ty, fields), span))
        }
    }

    fn parse_var_or_call(&mut self, ident: Id, ident_span: Span) -> Option<UntypedExpr> {
        self.next();

        match &self.peek().kind {
            Token::Scope => {
                let path = self.parse_path_with_first_segment(
                    Some(SymbolPathSegment::new(ident)),
                    ident_span,
                )?;
                Some(new_expr(Expr::Path(path.kind), path.span))
            }
            Token::Colon => {
                self.next();
                self.parse_struct(ident, ident_span)
            }
            _ => Some(new_expr(Expr::Variable(ident, false), ident_span)),
        }
    }

    fn parse_tuple(&mut self, lparen_span: &Span, first: UntypedExpr) -> Option<UntypedExpr> {
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

        Some(new_expr(Expr::Tuple(inner), span))
    }

    fn parse_array(&mut self) -> Option<UntypedExpr> {
        let lbracket_span = self.peek().span.clone();
        self.next(); // eat "["

        let init_expr = self.parse_expr()?;

        self.expect(&Token::Semicolon, &[Token::Semicolon]);

        let size = match self.peek().kind {
            Token::Number(size) => match usize::try_from(size) {
                Ok(size) => size,
                Err(_) => {
                    error!(&self.peek().span, "too large");
                    0
                }
            },
            _ => return None,
        };
        self.next();

        self.expect(&Token::Rbracket, &[Token::Rbracket]);

        let span = Span::merge(&lbracket_span, &self.prev().span);
        Some(new_expr(Expr::Array(Box::new(init_expr), size), span))
    }

    fn parse_primary(&mut self) -> Option<UntypedExpr> {
        let token = self.peek().clone();

        match token.kind {
            Token::Number(n) => {
                self.next();
                Some(new_expr(Expr::Literal(Literal::Number(n)), token.span))
            }
            Token::UnsignedNumber(n) => {
                self.next();
                Some(new_expr(
                    Expr::Literal(Literal::UnsignedNumber(n)),
                    token.span,
                ))
            }
            Token::Float(n) => {
                self.next();
                Some(new_expr(Expr::Literal(Literal::Float(n)), token.span))
            }
            Token::String(s) => {
                self.next();
                Some(new_expr(Expr::Literal(Literal::String(s)), token.span))
            }
            Token::Char(ch) => {
                self.next();
                Some(new_expr(Expr::Literal(Literal::Char(ch)), token.span))
            }
            Token::Identifier(name) => self.parse_var_or_call(name, token.span),
            Token::Keyword(Keyword::True) => {
                self.next();
                Some(new_expr(Expr::Literal(Literal::True), token.span))
            }
            Token::Keyword(Keyword::False) => {
                self.next();
                Some(new_expr(Expr::Literal(Literal::False), token.span))
            }
            Token::Keyword(Keyword::Null) => {
                self.next();
                Some(new_expr(Expr::Literal(Literal::Null), token.span))
            }
            Token::Lbracket => self.parse_array(),
            Token::Lparen => {
                let lparen_span = self.peek().span.clone();
                self.next();

                if self.consume(&Token::Rparen) {
                    // Unit literal
                    let span = Span::merge(&lparen_span, &self.prev().span);
                    Some(new_expr(Expr::Literal(Literal::Unit), span))
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
            }
            Token::Lbrace => self.parse_block_expr(),
            Token::Keyword(Keyword::If) => self.parse_if_expr(),
            _ => {
                error!(
                    &token.span,
                    "expected `number`, `identifier`, `true`, `false` or `(` but got `{}`",
                    self.peek().kind
                );
                None
            }
        }
    }

    fn parse_field(&mut self) -> Option<UntypedExpr> {
        let parse = Self::parse_primary;
        let mut expr = parse(self)?;

        loop {
            if self.consume(&Token::Dot) {
                match self.peek().kind {
                    Token::Number(n) => {
                        self.next();

                        if n < 0 {
                            error!(&self.peek().span.clone(), "field of negative number");
                            return None;
                        }

                        let span = Span::merge(&expr.span, &self.prev().span);
                        expr =
                            new_expr(Expr::Field(Box::new(expr), Field::Number(n as usize)), span);
                    }
                    Token::Identifier(id) => {
                        self.next();
                        let span = Span::merge(&expr.span, &self.prev().span);
                        expr = new_expr(Expr::Field(Box::new(expr), Field::Id(id)), span);
                    }
                    _ => {
                        error!(
                            &self.peek().span.clone(),
                            "expected `number` but got `{}`",
                            self.peek().kind
                        );
                    }
                }
            } else {
                break;
            }
        }

        Some(expr)
    }

    fn parse_expr_app(&mut self) -> Option<UntypedExpr> {
        let parse = Self::parse_field;

        let mut expr = parse(self)?;

        if self.consume(&Token::LTypeArgs) {
            let tyargs = self.parse_type_args();

            let span = Span::merge(&expr.span, &self.prev().span);
            expr = new_expr(Expr::App(Box::new(expr), tyargs), span);
        }

        Some(expr)
    }

    fn parse_subscript(&mut self) -> Option<UntypedExpr> {
        let parse = Self::parse_expr_app;

        let mut expr = parse(self)?;

        loop {
            if self.consume(&Token::Lbracket) {
                let subscript = self.parse_expr()?;

                self.expect(&Token::Rbracket, &[Token::Rbracket])?;

                let span = Span::merge(&expr.span, &self.prev().span);
                expr = new_expr(Expr::Subscript(Box::new(expr), Box::new(subscript)), span);
            } else {
                break;
            }
        }

        Some(expr)
    }

    fn next_is_arg(&self) -> bool {
        match &self.peek().kind {
            Token::Number(_)
            | Token::UnsignedNumber(_)
            | Token::Char(_)
            | Token::String(_)
            | Token::Identifier(_)
            | Token::Float(_)
            | Token::Keyword(Keyword::True)
            | Token::Keyword(Keyword::False)
            | Token::Keyword(Keyword::Null)
            | Token::Lparen => true,
            _ => false,
        }
    }

    fn parse_call(&mut self) -> Option<UntypedExpr> {
        let parse = Self::parse_subscript;

        let mut expr = parse(self)?;

        while expr_is_callable(&expr.kind) && self.next_is_arg() {
            let arg_expr = parse(self)?;
            let span = Span::merge(&expr.span, &arg_expr.span);
            expr = new_expr(Expr::Call(Box::new(expr), Box::new(arg_expr)), span);
        }

        Some(expr)
    }

    fn parse_range_end(
        &mut self,
        parse: fn(&mut Self) -> Option<UntypedExpr>,
        start: Option<UntypedExpr>,
        start_span: &Span,
    ) -> Option<UntypedExpr> {
        let (end, span) = if self.next_is_arg() {
            let end = parse(self)?;
            let span = Span::merge(start_span, &end.span);
            (Some(end), span)
        } else {
            (None, Span::merge(start_span, &self.prev().span))
        };

        Some(new_expr(
            Expr::Range(start.map(Box::new), end.map(Box::new)),
            span,
        ))
    }

    fn parse_range(&mut self) -> Option<UntypedExpr> {
        let parse = Self::parse_call;

        if self.consume(&Token::DoubleDot) {
            let start_span = self.prev().span.clone();
            return self.parse_range_end(parse, None, &start_span);
        }

        let mut expr = parse(self)?;
        if self.consume(&Token::DoubleDot) {
            let start_span = expr.span.clone();
            expr = self.parse_range_end(parse, Some(expr), &start_span)?;
        }

        Some(expr)
    }

    #[inline]
    fn parse_unary_op<P, F>(&mut self, mut parse: P, f: F) -> Option<UntypedExpr>
    where
        P: FnMut(&mut Self) -> Option<UntypedExpr>,
        F: Fn(Box<UntypedExpr>) -> Expr,
    {
        let symbol_span = self.peek().span.clone();
        self.next();

        let expr = parse(self)?;
        let span = Span::merge(&symbol_span, &expr.span);
        Some(new_expr(f(Box::new(expr)), span))
    }

    fn parse_unary(&mut self) -> Option<UntypedExpr> {
        let parse = Self::parse_range;

        match self.peek().kind {
            Token::Asterisk => self.parse_unary_op(parse, Expr::Dereference),
            Token::Sub => self.parse_unary_op(parse, Expr::Negative),
            Token::Keyword(Keyword::Not) => self.parse_unary_op(parse, Expr::Not),
            Token::Ampersand => {
                let symbol_span = self.peek().span.clone();
                self.next();

                let is_mutable = self.consume(&Token::Keyword(Keyword::Mut));
                let expr = parse(self)?;

                let span = Span::merge(&symbol_span, &expr.span);
                Some(new_expr(Expr::Address(Box::new(expr), is_mutable), span))
            }
            _ => parse(self),
        }
    }

    fn parse_mul(&mut self) -> Option<UntypedExpr> {
        self.parse_binop(
            true,
            Self::parse_unary,
            &[
                (&Token::Asterisk, &BinOp::Mul),
                (&Token::Div, &BinOp::Div),
                (&Token::Percent, &BinOp::Mod),
            ],
        )
    }

    fn parse_add(&mut self) -> Option<UntypedExpr> {
        self.parse_binop(
            true,
            Self::parse_mul,
            &[(&Token::Add, &BinOp::Add), (&Token::Sub, &BinOp::Sub)],
        )
    }

    fn parse_shift(&mut self) -> Option<UntypedExpr> {
        self.parse_binop(
            true,
            Self::parse_add,
            &[
                (&Token::LShift, &BinOp::LShift),
                (&Token::RShift, &BinOp::RShift),
            ],
        )
    }

    fn parse_bit_and(&mut self) -> Option<UntypedExpr> {
        self.parse_binop(
            true,
            Self::parse_shift,
            &[(&Token::Ampersand, &BinOp::BitAnd)],
        )
    }

    fn parse_bit_xor(&mut self) -> Option<UntypedExpr> {
        self.parse_binop(true, Self::parse_bit_and, &[(&Token::Xor, &BinOp::BitXor)])
    }

    fn parse_bit_or(&mut self) -> Option<UntypedExpr> {
        self.parse_binop(
            true,
            Self::parse_bit_xor,
            &[(&Token::VerticalBar, &BinOp::BitOr)],
        )
    }

    fn parse_relational(&mut self) -> Option<UntypedExpr> {
        self.parse_binop(
            false,
            Self::parse_bit_or,
            &[
                (&Token::LessThan, &BinOp::LessThan),
                (&Token::LessThanOrEqual, &BinOp::LessThanOrEqual),
                (&Token::GreaterThan, &BinOp::GreaterThan),
                (&Token::GreaterThanOrEqual, &BinOp::GreaterThanOrEqual),
            ],
        )
    }

    fn parse_equality(&mut self) -> Option<UntypedExpr> {
        self.parse_binop(
            false,
            Self::parse_relational,
            &[
                (&Token::Equal, &BinOp::Equal),
                (&Token::NotEqual, &BinOp::NotEqual),
            ],
        )
    }

    fn parse_and(&mut self) -> Option<UntypedExpr> {
        self.parse_binop(true, Self::parse_equality, &[(&Token::And, &BinOp::And)])
    }

    fn parse_or(&mut self) -> Option<UntypedExpr> {
        self.parse_binop(true, Self::parse_and, &[(&Token::Or, &BinOp::Or)])
    }

    fn parse_expr(&mut self) -> Option<UntypedExpr> {
        self.parse_or()
    }

    fn parse_bind_stmt(&mut self) -> Option<Spanned<Stmt>> {
        // Eat "let"
        let let_span = self.peek().span.clone();
        self.next();

        let is_mutable = self.consume(&Token::Keyword(Keyword::Mut));

        // Bind name
        let name = self.expect_identifier(&[Token::Semicolon])?;

        let ty = if self.consume(&Token::Colon) {
            let ty = self.parse_skip(Self::parse_type, &[Token::Equal])?;
            Some(ty)
        } else {
            None
        };

        self.expect(&Token::Equal, &[Token::Semicolon])?;

        // Initial expression
        let expr = match self.parse_expr() {
            Some(expr) => expr,
            None => {
                self.skip_until(&[Token::Semicolon]);
                return None;
            }
        };

        self.expect(&Token::Semicolon, &[Token::Semicolon])?;
        let semicolon_span = &self.prev().span;

        let span = Span::merge(&let_span, semicolon_span);

        Some(spanned(
            Stmt::Bind(name, ty, Box::new(expr), is_mutable, false, false),
            span,
        ))
    }

    fn parse_block_expr(&mut self) -> Option<UntypedExpr> {
        let lbrace_span = self.peek().span.clone();
        self.next();

        let mut result_expr = None;
        let mut stmts = Vec::new();

        self.blocks_builder.push();

        while !self.consume(&Token::Rbrace) {
            if self.consume(&Token::Semicolon) {
                continue;
            }

            let (is_expr, stmt) = self.parse_stmt_without_expr();
            if is_expr {
                let expr = self.parse_skip(Self::parse_expr, &[Token::Semicolon])?;

                // Assign statement
                let expr = match self.parse_assign_operators(expr) {
                    Ok(stmt) => {
                        if let Some(stmt) = stmt {
                            stmts.push(stmt);
                        }
                        continue;
                    }
                    Err(expr) => expr,
                };

                if self.consume(&Token::Rbrace) {
                    result_expr = Some(expr);
                    break;
                }

                if needs_semicolon(&expr.kind) {
                    self.expect(&Token::Semicolon, &[Token::Semicolon])?;
                }

                let span = expr.span.clone();
                stmts.push(spanned(Stmt::Expr(expr), span));
            } else if let Some(stmt) = stmt {
                stmts.push(stmt);
            }
        }

        let span = Span::merge(&lbrace_span, &self.prev().span);

        // Push literal unit if there is no result expression
        let result_expr =
            result_expr.unwrap_or_else(|| new_expr(Expr::Literal(Literal::Unit), span.clone()));

        let block = self.blocks_builder.pop_and_build(stmts, result_expr);

        Some(new_expr(Expr::Block(block), span))
    }

    fn expect_block_expr(&mut self) -> Option<UntypedExpr> {
        if self.peek().kind != Token::Lbrace {
            error!(&self.peek().span.clone(), "expected block");
            return None;
        }

        self.parse_block_expr()
    }

    fn expect_block(&mut self) -> Option<Spanned<Stmt>> {
        let block = self.expect_block_expr()?;
        let span = block.span.clone();
        let block = spanned(Stmt::Expr(block), span);

        Some(block)
    }

    fn parse_stmt_assign(&mut self, lhs: UntypedExpr) -> Option<Spanned<Stmt>> {
        let rhs = self.parse_skip(Self::parse_expr, &[Token::Semicolon])?;

        self.expect(&Token::Semicolon, &[Token::Semicolon])?;

        let span = Span::merge(&lhs.span, &self.prev().span);
        Some(spanned(Stmt::Assign(lhs, Box::new(rhs)), span))
    }

    fn parse_compound_assignment(
        &mut self,
        binop: BinOp,
        lhs: UntypedExpr,
    ) -> Option<Spanned<Stmt>> {
        self.next();

        let rhs = self.parse_skip(Self::parse_expr, &[Token::Semicolon])?;
        self.expect(&Token::Semicolon, &[Token::Semicolon]);

        let span = Span::merge(&lhs.span, &self.prev().span);
        Some(spanned(
            Stmt::Assign(
                lhs.clone(),
                Box::new(new_expr(
                    Expr::BinOp(binop, Box::new(lhs), Box::new(rhs)),
                    span.clone(),
                )),
            ),
            span,
        ))
    }

    fn parse_assign_operators(
        &mut self,
        lhs: UntypedExpr,
    ) -> Result<Option<Spanned<Stmt>>, UntypedExpr> {
        match &self.peek().kind {
            Token::Assign => {
                self.next();
                Ok(self.parse_stmt_assign(lhs))
            }
            Token::AddAssign => Ok(self.parse_compound_assignment(BinOp::Add, lhs)),
            Token::SubAssign => Ok(self.parse_compound_assignment(BinOp::Sub, lhs)),
            Token::MulAssign => Ok(self.parse_compound_assignment(BinOp::Mul, lhs)),
            Token::DivAssign => Ok(self.parse_compound_assignment(BinOp::Div, lhs)),
            Token::ModAssign => Ok(self.parse_compound_assignment(BinOp::Mod, lhs)),
            _ => Err(lhs),
        }
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

    fn parse_if_expr(&mut self) -> Option<UntypedExpr> {
        let if_token_span = self.peek().span.clone();
        self.next();

        // Parse condition expression
        let expr = self.parse_skip(
            Self::parse_expr,
            &[Token::Lbrace, Token::Keyword(Keyword::Else)],
        );
        // Parse then-clause
        let then_expr = self.expect_block_expr()?;

        // Parse else-clause
        let else_expr = if self.consume(&Token::Keyword(Keyword::Else)) {
            if self.peek().kind == Token::Keyword(Keyword::If) {
                Some(self.parse_skip(Self::parse_if_expr, &[Token::Rbrace])?)
            } else {
                Some(self.expect_block_expr()?)
            }
        } else {
            None
        }
        .map(Box::new);

        let span = Span::merge(
            &if_token_span,
            else_expr
                .as_ref()
                .map(|e| &e.span)
                .unwrap_or(&then_expr.span),
        );
        Some(new_expr(
            Expr::If(Box::new(expr?), Box::new(then_expr), else_expr),
            span,
        ))
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

    fn parse_import_range(&mut self) -> Option<Spanned<ImportRange>> {
        let mut stack = Vec::new();
        let mut top = None;

        let first = self.expect_identifier(&[Token::Scope])?;
        let first_span = self.prev().span.clone();

        if self.consume(&Token::Keyword(Keyword::As)) {
            let renamed = self.expect_identifier(&[Token::Semicolon])?;
            let span = Span::merge(&first_span, &self.prev().span);
            return Some(spanned(ImportRange::Renamed(first, renamed), span));
        }

        stack.push(first);

        while self.consume(&Token::Scope) {
            match &self.peek().kind {
                Token::Identifier(id) => {
                    let id = *id;
                    self.next();

                    if self.consume(&Token::Keyword(Keyword::As)) {
                        let renamed = self.expect_identifier(&[Token::Semicolon])?;
                        top = Some(ImportRange::Renamed(id, renamed));

                        break;
                    } else {
                        stack.push(id);
                    }
                }
                Token::Lbrace => {
                    self.next();

                    let mut ranges = Vec::new();

                    if let Some(first) =
                        self.parse_skip(Self::parse_import_range, &[Token::Rbrace, Token::Comma])
                    {
                        ranges.push(first.kind);
                    }

                    while self.peek().kind != Token::Rbrace && self.consume(&Token::Comma) {
                        if self.peek().kind == Token::Rbrace {
                            break;
                        }

                        if let Some(range) = self
                            .parse_skip(Self::parse_import_range, &[Token::Rbrace, Token::Comma])
                        {
                            ranges.push(range.kind);
                        }
                    }

                    self.expect(&Token::Rbrace, &[Token::Rbrace]);

                    top = Some(ImportRange::Multiple(ranges));

                    break;
                }
                Token::Asterisk => {
                    self.next();
                    top = Some(ImportRange::All);
                    break;
                }
                _ => break,
            }
        }

        let end_span = self.prev().span.clone();

        let mut range = top.unwrap_or_else(|| ImportRange::Symbol(stack.pop().unwrap()));
        while let Some(id) = stack.pop() {
            range = ImportRange::Scope(id, Box::new(range));
        }

        let span = Span::merge(&first_span, &end_span);
        Some(spanned(range, span))
    }

    fn parse_import_stmt(&mut self) -> Option<Spanned<Stmt>> {
        self.next();

        let import_range = self.parse_skip(Self::parse_import_range, &[Token::Semicolon])?;

        self.expect(&Token::Semicolon, &[Token::Semicolon])?;

        Some(spanned(Stmt::Import(import_range.kind), import_range.span))
    }

    fn parse_stmt_without_expr(&mut self) -> (bool, Option<Spanned<Stmt>>) {
        let token = self.peek();
        let stmt = match token.kind {
            Token::Keyword(Keyword::Let) => self.parse_bind_stmt(),
            Token::Keyword(Keyword::Return) => self.parse_return(),
            Token::Keyword(Keyword::While) => self.parse_while_stmt(),
            Token::Keyword(Keyword::Import) => self.parse_import_stmt(),
            Token::Keyword(Keyword::Fn) => {
                let func = self.parse_fn_decl();
                if let Some(func) = func {
                    self.blocks_builder.def().functions.push(func);
                }
                None
            }
            Token::Keyword(Keyword::Type) => {
                let tydef = self.parse_def_type();
                if let Some(tydef) = tydef {
                    self.blocks_builder.def().types.push(tydef);
                }
                None
            }
            _ => return (true, None),
        };

        (false, stmt)
    }

    fn parse_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let (is_expr, stmt) = self.parse_stmt_without_expr();
        if is_expr {
            let expr = self.parse_skip(Self::parse_expr, &[Token::Semicolon])?;

            let expr = match self.parse_assign_operators(expr) {
                Ok(stmt) => return stmt,
                Err(expr) => expr,
            };

            if needs_semicolon(&expr.kind) {
                self.expect(&Token::Semicolon, &[Token::Semicolon])?;
            }

            let span = expr.span.clone();
            Some(spanned(Stmt::Expr(expr), span))
        } else {
            stmt
        }
    }

    fn parse_type_pointer(&mut self) -> Option<Spanned<AstType>> {
        let asterisk_span = self.peek().span.clone();
        self.next(); // eat '*'

        let is_mutable = self.consume(&Token::Keyword(Keyword::Mut));
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
                    error!(&self.peek().span.clone(), "too large");
                    0
                }
            },
            _ => return None,
        };
        self.next();

        self.expect(&Token::Rbracket, &[Token::Rbracket]);

        let span = Span::merge(&lbracket_span, &self.peek().span);
        Some(spanned(AstType::Array(Box::new(ty), size), span))
    }

    fn parse_type_args(&mut self) -> Vec<Spanned<AstType>> {
        if self.consume(&Token::GreaterThan) {
            Vec::new()
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

                if let Some(ty) =
                    self.parse_skip(Self::parse_type, &[Token::GreaterThan, Token::Comma])
                {
                    types.push(ty);
                }
            }

            self.expect(&Token::GreaterThan, &[Token::GreaterThan]);

            types
        }
    }

    fn parse_type_app(&mut self, id: Spanned<Id>) -> Spanned<AstType> {
        let args = if self.consume(&Token::LessThan) {
            self.parse_type_args()
        } else {
            Vec::new()
        };

        if args.is_empty() {
            spanned(AstType::Named(id.kind), id.span)
        } else {
            let span = Span::merge(&id.span, &self.prev().span);
            spanned(AstType::App(id, args), span)
        }
    }

    fn parse_type_arrow(&mut self, ty: Spanned<AstType>) -> Option<Spanned<AstType>> {
        if self.consume(&Token::Arrow) {
            let new_ty = self.parse_type()?;
            let span = Span::merge(&ty.span, &new_ty.span);
            Some(spanned(
                AstType::Arrow(Box::new(ty), Box::new(new_ty)),
                span,
            ))
        } else {
            Some(ty)
        }
    }

    fn parse_type_slice(&mut self) -> Option<Spanned<AstType>> {
        // Eat "&"
        let and_token_span = self.peek().span.clone();
        self.next();

        let is_mutable = self.consume(&Token::Keyword(Keyword::Mut));

        self.expect(&Token::Lbracket, &[Token::Lbracket])?;
        let inner_type = self.parse_type()?;
        self.expect(&Token::Rbracket, &[Token::Rbracket])?;

        let span = Span::merge(&and_token_span, &self.prev().span);
        Some(spanned(
            AstType::Slice(Box::new(inner_type), is_mutable),
            span,
        ))
    }

    fn parse_type(&mut self) -> Option<Spanned<AstType>> {
        let first_span = self.peek().span.clone();
        let ty = match self.peek().kind {
            Token::Keyword(Keyword::UInt) => {
                self.next_and(Some(spanned(AstType::UInt, first_span)))
            }
            Token::Keyword(Keyword::Int) => self.next_and(Some(spanned(AstType::Int, first_span))),
            Token::Keyword(Keyword::Float) => {
                self.next_and(Some(spanned(AstType::Float, first_span)))
            }
            Token::Keyword(Keyword::Char) => {
                self.next_and(Some(spanned(AstType::Char, first_span)))
            }
            Token::Keyword(Keyword::Bool) => {
                self.next_and(Some(spanned(AstType::Bool, first_span)))
            }
            Token::Keyword(Keyword::String) => {
                self.next_and(Some(spanned(AstType::String, first_span)))
            }
            Token::Identifier(name) => {
                let id = self.next_and(spanned(name, first_span));
                let ty = self.parse_type_app(id);
                Some(ty)
            }
            Token::Asterisk => self.parse_type_pointer(),
            Token::Lparen => self.parse_type_tuple(), // tuple
            Token::Keyword(Keyword::Struct) => self.parse_type_struct(), // struct
            Token::Lbracket => self.parse_type_array(), // array
            Token::Ampersand => self.parse_type_slice(),
            _ => {
                error!(
                    &self.peek().span.clone(),
                    "expected `(`, `[`, `*`, `&[`, `int`, `bool`, `string`, `struct` or `identifier` but got `{}`",
                    self.peek().kind
                );
                self.next();
                None
            }
        }?;

        let ty = self.parse_type_arrow(ty)?;

        Some(ty)
    }

    fn parse_param(&mut self) -> Option<Param> {
        let tokens_to_skip = [Token::Comma, Token::Rparen];

        let is_mutable = self.consume(&Token::Keyword(Keyword::Mut));

        // Parse the parameter name
        let name = self.expect_identifier(&tokens_to_skip)?;

        // Parse the parameter type
        self.expect(&Token::Colon, &tokens_to_skip)?;
        let ty = self.parse_skip(Self::parse_type, &tokens_to_skip)?;

        Some(Param {
            name,
            ty,
            is_mutable,
            // These are set in module "escape" and "heapvar"
            is_escaped: false,
            is_in_heap: false,
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
        let name = name.map(|name| spanned(name, self.prev().span.clone()));
        // Parse the type parameters
        let ty_params = self.parse_type_vars();

        // Parse parameters
        self.expect(&Token::Lparen, &[Token::Lparen]);
        let mut params = self.parse_param_list()?;

        // Insert unit param if no params
        if params.is_empty() {
            params.push(Param {
                name: *reserved_id::DUMMY_PARAM,
                ty: spanned(AstType::Unit, self.prev().span.clone()),
                is_mutable: false,
                is_escaped: false,
                is_in_heap: false,
            });
        }

        // Parse the return type
        let return_ty = if self.consume(&Token::Colon) {
            self.parse_skip(Self::parse_type, &[Token::Lbrace])
        } else {
            // return unit if omit a type
            Some(spanned(AstType::Unit, self.prev().span.clone()))
        };

        self.expect(&Token::Equal, &[Token::Equal]);

        let body = self.parse_skip(Self::parse_expr, &[Token::Semicolon, Token::Rbrace])?;
        if needs_semicolon(&body.kind) {
            self.expect(&Token::Semicolon, &[Token::Semicolon])?;
        }

        Some(AstFunction {
            name: name?,
            params,
            return_ty: return_ty?,
            body,
            ty_params,
            has_escaped_variables: false,
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

        Some(AstTypeDef { name, ty, var_ids })
    }

    fn parse_defs_in_impl(&mut self, imple: &mut Impl) -> Option<()> {
        if let Token::Keyword(Keyword::Fn) = self.peek().kind {
            let func = self.parse_fn_decl()?;
            imple.add_function(func);
        }

        Some(())
    }

    fn parse_impl(&mut self) -> Option<Impl> {
        let target = self.expect_identifier(&[Token::Rbrace])?;
        let target = Spanned::new(target, self.prev().span.clone());

        self.expect(&Token::Lbrace, &[Token::Lbrace]);

        let mut impl_ = Impl::new(target);

        while !self.consume(&Token::Rbrace) {
            if self.peek().kind == Token::EOF {
                error!(&self.peek().span, "unexpected EOF");
                break;
            }

            self.parse_defs_in_impl(&mut impl_);
        }

        Some(impl_)
    }

    pub fn parse(mut self, module_path: &SymbolPath) -> Program {
        assert!(module_path.is_absolute);
        self.blocks_builder.push();

        while self.peek().kind != Token::EOF {
            if self.consume(&Token::Semicolon) {
                continue;
            }

            if self.consume(&Token::Keyword(Keyword::Impl)) {
                if let Some(impl_) = self.parse_impl() {
                    self.impls.push(impl_);
                }
                continue;
            }

            if let Some(stmt) = self.parse_stmt() {
                self.main_stmts.push(stmt);
            }
        }

        let dummy_result_expr = new_expr(Expr::Literal(Literal::Unit), self.peek().span.clone());
        let block = self
            .blocks_builder
            .pop_and_build(self.main_stmts, dummy_result_expr);

        Program {
            module_path: module_path.clone(),
            main: block,
            imported_modules: Vec::new(),
            impls: self.impls,
        }
    }
}
