use crate::span::{Span, Spanned};
use crate::ty::Type;
use crate::id::Id;

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
pub enum Expr {
    Literal(Literal),
    BinOp(BinOp, Box<SpannedTyped<Expr>>, Box<SpannedTyped<Expr>>),
    Variable(Id),
    Call(Id, Vec<SpannedTyped<Expr>>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Stmt {
    Bind(Id, SpannedTyped<Expr>),
    Expr(SpannedTyped<Expr>),
    Block(Vec<Spanned<Stmt>>),
    Return(SpannedTyped<Expr>),
    If(SpannedTyped<Expr>, Box<Spanned<Stmt>>, Option<Box<Spanned<Stmt>>>),
    While(SpannedTyped<Expr>, Box<Spanned<Stmt>>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum TopLevel {
    Stmt(Spanned<Stmt>),
    Function(Id, Vec<(Id, Type)>, Type, Spanned<Stmt>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub top: Vec<Spanned<TopLevel>>,
}
