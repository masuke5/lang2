use crate::span::Spanned;
use crate::ty::Type;
use crate::id::Id;

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
    And,
    Or,
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
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Number(i64),
    String(String),
    True,
    False,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Literal(Literal),
    BinOp(BinOp, Box<Spanned<Expr>>, Box<Spanned<Expr>>),
    Variable(Id),
    Call(Id, Vec<Spanned<Expr>>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Stmt {
    Bind(Id, Spanned<Expr>),
    Expr(Spanned<Expr>),
    Block(Vec<Spanned<Stmt>>),
    Return(Spanned<Expr>),
    If(Spanned<Expr>, Box<Spanned<Stmt>>, Option<Box<Spanned<Stmt>>>),
    While(Spanned<Expr>, Box<Spanned<Stmt>>),
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
