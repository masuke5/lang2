use crate::span::Spanned;

#[derive(Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
pub enum Literal {
    Number(i64),
}

#[derive(Debug, PartialEq)]
pub enum Expr<'a> {
    Literal(Literal),
    BinOp(BinOp, Box<Spanned<Expr<'a>>>, Box<Spanned<Expr<'a>>>),
    Variable(&'a str),
}

#[derive(Debug, PartialEq)]
pub enum Stmt<'a> {
    Bind(&'a str, Spanned<Expr<'a>>),
    Expr(Spanned<Expr<'a>>),
    Block(Vec<Spanned<Stmt<'a>>>),
}

#[derive(Debug, PartialEq)]
pub struct Program<'a> {
    pub stmt: Vec<Spanned<Stmt<'a>>>,
}
