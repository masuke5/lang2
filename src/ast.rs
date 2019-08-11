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
pub enum Expr {
    Literal(Literal),
    BinOp(BinOp, Box<Spanned<Expr>>, Box<Spanned<Expr>>),
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub expr: Spanned<Expr>,
}
