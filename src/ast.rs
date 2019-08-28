use crate::span::{Span, Spanned};
use crate::ty::Type;

#[derive(Debug, PartialEq, Clone)]
pub struct SpannedTyped<T> {
    pub kind: T,
    pub ty: Type,
    pub span: Span,
}

impl<T> SpannedTyped<T> {
    pub fn new(kind: T, span: Span, ty: Type) -> SpannedTyped<T> {
        Self {
            kind,
            span,
            ty,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl BinOp {
    pub fn to_symbol(&self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Equal => "==",
            BinOp::NotEqual => "!=",
            BinOp::LessThan => "<",
            BinOp::LessThanOrEqual => "<=",
            BinOp::GreaterThan => ">",
            BinOp::GreaterThanOrEqual => ">=",
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Number(i64),
    True,
    False,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr<'a> {
    Literal(Literal),
    BinOp(BinOp, Box<SpannedTyped<Expr<'a>>>, Box<SpannedTyped<Expr<'a>>>),
    Variable(&'a str),
    Call(&'a str, Vec<SpannedTyped<Expr<'a>>>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Stmt<'a> {
    Bind(&'a str, SpannedTyped<Expr<'a>>),
    Expr(SpannedTyped<Expr<'a>>),
    Block(Vec<Spanned<Stmt<'a>>>),
    Return(SpannedTyped<Expr<'a>>),
    If(SpannedTyped<Expr<'a>>, Box<Spanned<Stmt<'a>>>),
    While(SpannedTyped<Expr<'a>>, Box<Spanned<Stmt<'a>>>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum TopLevel<'a> {
    Stmt(Spanned<Stmt<'a>>),
    Function(&'a str, Vec<(&'a str, Type)>, Type, Spanned<Stmt<'a>>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program<'a> {
    pub top: Vec<Spanned<TopLevel<'a>>>,
}
