use std::collections::LinkedList;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::utils::{escape_string, format_iter, format_iter_delimiter};

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
    Unit,
    Pointer(Box<Expr>),
    Dereference(Box<Expr>),
    Copy(Box<Expr>, usize),
    Offset(Box<Expr>, Box<Expr>),
    Duplicate(Box<Expr>, usize),
    LoadCopy(VariableLoc, usize),
    LoadRef(VariableLoc),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    Negative(Box<Expr>),
    Alloc(Box<Expr>),
    Record(Vec<Expr>),
    Wrap(Box<Expr>),
    Unwrap(Box<Expr>, usize),
    Call(Box<Expr>, Box<Expr>, usize),
    FuncPos(Option<usize>, usize),
    EP,
    TOS,

    Seq(Vec<Stmt>, Box<Expr>),
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Int(n) => write!(f, "{}", n),
            Expr::String(s) => write!(f, "\"{}\"", escape_string(s)),
            Expr::True => write!(f, "true"),
            Expr::False => write!(f, "false"),
            Expr::Null => write!(f, "null"),
            Expr::Unit => write!(f, "()"),
            Expr::Pointer(expr) => write!(f, "&{}", expr),
            Expr::Dereference(expr) => write!(f, "*{}", expr),
            Expr::Copy(expr, size) => write!(f, "copy({}, {})", expr, size),
            Expr::Offset(expr, offset) => write!(f, "{}[{}]", expr, offset),
            Expr::Duplicate(expr, count) => write!(f, "dup({}, {})", expr, count),
            Expr::LoadCopy(loc, size) => write!(f, "lcopy({}, {})", loc, size),
            Expr::LoadRef(loc) => write!(f, "&{}", loc),
            Expr::BinOp(binop, lhs, rhs) => write!(f, "({} {} {})", lhs, binop.to_symbol(), rhs),
            Expr::Negative(expr) => write!(f, "-{}", expr),
            Expr::Alloc(expr) => write!(f, "alloc({})", expr),
            Expr::Record(exprs) => write!(f, "[{}]", format_iter(exprs.iter())),
            Expr::Wrap(expr) => write!(f, "wrap({})", expr),
            Expr::Unwrap(expr, size) => write!(f, "unwrap({}, {})", expr, size),
            Expr::Call(func, arg, rv_size) => write!(f, "({} {} rv_size={})", func, arg, rv_size),
            Expr::FuncPos(Some(module), func) => write!(f, "func({}, {})", module, func),
            Expr::FuncPos(None, func) => write!(f, "self_func({})", func),
            Expr::EP => write!(f, "$ep"),
            Expr::TOS => write!(f, "$TOS"),
            Expr::Seq(stmts, expr) => write!(
                f,
                "{{ {}; {} }}",
                format_iter_delimiter(stmts.iter(), "; "),
                expr
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Discard(Expr),
    Store(VariableLoc, Expr),
    StoreFromRef(Expr, Expr),
    Return(Option<Expr>),
    Label(Label),
    Jump(Label),
    JumpIfFalse(Label, Expr),
    JumpIfTrue(Label, Expr),
    Push(Expr),
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discard(expr) => write!(f, "discard {}", expr),
            Self::Store(loc, expr) => write!(f, "{} <- {}", loc, expr),
            Self::StoreFromRef(lhs, expr) => write!(f, "*{} <- {}", lhs, expr),
            Self::Return(expr) => {
                write!(f, "return")?;
                if let Some(expr) = expr {
                    write!(f, " {}", expr)?;
                }

                Ok(())
            }
            Self::Label(label) => write!(f, "{}:", label),
            Self::Jump(label) => write!(f, "=> {}", label),
            Self::JumpIfFalse(label, expr) => write!(f, "if !{} => {}", expr, label),
            Self::JumpIfTrue(label, expr) => write!(f, "if {} => {}", expr, label),
            Self::Push(expr) => write!(f, "push {}", expr),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeBuf {
    stmts: LinkedList<Stmt>,
}

impl CodeBuf {
    pub fn new() -> Self {
        Self {
            stmts: LinkedList::new(),
        }
    }

    pub fn append(&mut self, mut buf: Self) {
        self.stmts.append(&mut buf.stmts);
    }

    pub fn push(&mut self, stmt: Stmt) {
        self.stmts.push_back(stmt);
    }

    pub fn iter(&self) -> impl Iterator<Item = &Stmt> {
        self.stmts.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = Stmt> {
        self.stmts.into_iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub stack_size: usize,
    pub stack_in_heap_size: usize,
    pub body: Expr,
}

pub fn dump_expr(expr: &Expr) {
    let raw = format!("{}", expr);

    // Format
    let mut dump = String::with_capacity(raw.len());
    let mut level = 0;
    let mut skip_ws = false;

    for ch in raw.chars() {
        match ch {
            '{' => {
                skip_ws = true;
                level += 1;

                dump.push('{');
                dump.push('\n');
                dump += &"  ".repeat(level);

                continue;
            }
            '}' => {
                level -= 1;

                if dump.as_bytes().last().map_or('\0', |b| *b as char) != '\n' {
                    dump.push('\n');
                    dump += &"  ".repeat(level);
                }
                dump.push('}');
            }
            ';' => {
                skip_ws = true;

                dump.push(';');
                dump.push('\n');
                dump += &"  ".repeat(level);

                continue;
            }
            ' ' | '\t' if skip_ws => {}
            _ => dump.push(ch),
        }

        skip_ws = false;
    }

    println!("{}", dump);
}
