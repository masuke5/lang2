use crate::span::Spanned;

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int,
}

#[derive(Debug, PartialEq, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div
}

impl BinOp {
    pub fn to_symbol(&self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Number(i64),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr<'a> {
    Literal(Literal),
    BinOp(BinOp, Box<Spanned<Expr<'a>>>, Box<Spanned<Expr<'a>>>),
    Variable(&'a str),
    Call(&'a str, Vec<Spanned<Expr<'a>>>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Stmt<'a> {
    Bind(&'a str, Spanned<Expr<'a>>),
    Expr(Spanned<Expr<'a>>),
    Block(Vec<Spanned<Stmt<'a>>>),
    Return(Spanned<Expr<'a>>),
    If(Spanned<Expr<'a>>, Box<Spanned<Stmt<'a>>>),
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
