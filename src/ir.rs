use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::utils::{escape_string, format_iter};

pub static NEXT_LABEL: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(usize);

impl Label {
    pub fn new() -> Self {
        Self(NEXT_LABEL.fetch_add(1, Ordering::AcqRel))
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.as_usize())
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariableLoc {
    Local(isize),
    Heap(usize, usize),
}

impl fmt::Display for VariableLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local(loc) => write!(f, "local[{}]", loc),
            Self::Heap(loc, level) => write!(f, "heap[{}][{}]", level, loc),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Int(i64),
    String(String),
    True,
    False,
    Null,
    Pointer(Box<Expr>),
    Copy(Box<Expr>, usize),
    Offset(Box<Expr>, Box<Expr>),
    Duplicate(Box<Expr>, usize),
    LoadCopy(VariableLoc, usize),
    LoadRef(VariableLoc),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    Alloc(Box<Expr>),
    Record(Vec<Expr>),
    Wrap(Box<Expr>),
    Unwrap(Box<Expr>, usize),
    Call(Box<Expr>, Box<Expr>),
    FuncPos(usize, usize),
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(n) => write!(f, "{}", n),
            Self::String(s) => write!(f, "\"{}\"", escape_string(s)),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
            Self::Null => write!(f, "null"),
            Self::Pointer(expr) => write!(f, "&({})", expr),
            Self::Copy(expr, size) => write!(f, "copy({}, {})", expr, size),
            Self::Offset(expr, offset) => write!(f, "({})[{}]", expr, offset),
            Self::Duplicate(expr, count) => write!(f, "dup({}, {})", expr, count),
            Self::LoadCopy(loc, size) => write!(f, "lcopy({}, {})", loc, size),
            Self::LoadRef(loc) => write!(f, "{}", loc),
            Self::BinOp(binop, lhs, rhs) => write!(f, "{} {} {}", lhs, binop.to_symbol(), rhs),
            Self::Alloc(expr) => write!(f, "alloc({})", expr),
            Self::Record(exprs) => write!(f, "({})", format_iter(exprs.iter())),
            Self::Wrap(expr) => write!(f, "wrap({})", expr),
            Self::Unwrap(expr, size) => write!(f, "unwrap({}, {})", expr, size),
            Self::Call(func, arg) => write!(f, "({}) ({})", func, arg),
            Self::FuncPos(module, func) => write!(f, "func({}, {})", module, func),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Store(VariableLoc, Expr),
    Return(Expr),
    Label(Label),
    JumpIfFalse(Label, Expr),
    JumpIfTrue(Label, Expr),
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(loc, expr) => write!(f, "{} <- {}", loc, expr),
            Self::Return(expr) => write!(f, "return {}", expr),
            Self::Label(label) => write!(f, "{}:", label),
            Self::JumpIfFalse(label, expr) => write!(f, "if !({}) => {}", expr, label),
            Self::JumpIfTrue(label, expr) => write!(f, "if ({}) => {}", expr, label),
        }
    }
}
