use std::collections::LinkedList;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::id::Id;
use crate::utils::{escape_string, format_iter, format_iter_delimiter};

pub static NEXT_LABEL: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(usize);

impl Label {
    pub fn new() -> Self {
        Self(NEXT_LABEL.fetch_add(1, Ordering::AcqRel))
    }

    pub fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.as_usize())
    }
}

pub static NEXT_SEQ_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(usize);

impl SeqId {
    pub fn new() -> Self {
        Self(NEXT_SEQ_ID.fetch_add(1, Ordering::AcqRel))
    }

    #[allow(dead_code)]
    pub unsafe fn from_raw(raw: usize) -> Self {
        Self(raw)
    }

    #[allow(dead_code)]
    pub fn raw(self) -> usize {
        self.0
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BinOp {
    FloatAdd,
    FloatSub,
    FloatMul,
    FloatDiv,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    LShift,
    RShift,
    LogicalLShift,
    LogicalRShift,
    BitAnd,
    BitOr,
    BitXor,
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
            BinOp::FloatAdd => "+.",
            BinOp::FloatSub => "-.",
            BinOp::FloatMul => "*.",
            BinOp::FloatDiv => "/.",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::LShift => "<<",
            BinOp::RShift => ">>",
            BinOp::LogicalLShift => "<<.",
            BinOp::LogicalRShift => ">>.",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
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

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    String(String),
    True,
    False,
    Null,
    Unit,
    Pointer(Box<Expr>),
    Dereference(Box<Expr>),
    Copy(Box<Expr>, usize),
    Offset(Box<Expr>, Box<Expr>),
    OffsetSlice(Box<Expr>, Box<Expr>, usize),
    Duplicate(Box<Expr>, usize),
    LoadCopy(VariableLoc, usize),
    LoadRef(VariableLoc),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    Negative(Box<Expr>),
    Not(Box<Expr>),
    Alloc(Box<Expr>),
    Record(Vec<Expr>),
    Wrap(Box<Expr>),
    Unwrap(Box<Expr>, usize),
    Call(Box<Expr>, Box<Expr>, usize),
    FuncPos(Option<usize>, usize),
    EP,
    TOS(usize),

    Seq(Vec<Stmt>, Box<Expr>),
    SeqId(SeqId, Box<Expr>),
}

impl Expr {
    pub fn size(&self) -> usize {
        match self {
            Expr::Int(..)
            | Expr::Float(..)
            | Expr::String(..)
            | Expr::True
            | Expr::False
            | Expr::Null => 1,
            Expr::Unit => 0,
            Expr::Pointer(..) | Expr::Dereference(..) => 1,
            Expr::Copy(_, size) => *size,
            Expr::Offset(..) | Expr::OffsetSlice(..) => 1,
            Expr::Duplicate(expr, count) => expr.size() * count,
            Expr::LoadCopy(_, size) => *size,
            Expr::LoadRef(..) => 1,
            Expr::BinOp(..) => 1,    // bool or int
            Expr::Negative(..) => 1, // int
            Expr::Not(..) => 1,      // int or uint
            Expr::Alloc(..) => 1,
            Expr::Record(exprs) => exprs.iter().map(Expr::size).sum(),
            Expr::Wrap(..) => 1,
            Expr::Unwrap(_, size) => *size,
            Expr::Call(_, _, size) => *size,
            Expr::FuncPos(..) => 1,
            Expr::EP => 1,
            Expr::TOS(size) => *size,
            Expr::Seq(_, expr) => expr.size(),
            Expr::SeqId(..) => panic!("unknown size"),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Int(n) => write!(f, "{}", n),
            Expr::Float(n) => write!(f, "{}", n),
            Expr::String(s) => write!(f, "\"{}\"", escape_string(s)),
            Expr::True => write!(f, "true"),
            Expr::False => write!(f, "false"),
            Expr::Null => write!(f, "null"),
            Expr::Unit => write!(f, "()"),
            Expr::Pointer(expr) => write!(f, "&{}", expr),
            Expr::Dereference(expr) => write!(f, "*{}", expr),
            Expr::Copy(expr, size) => write!(f, "copy({}, {})", expr, size),
            Expr::Offset(expr, offset) => write!(f, "offset({}, {})", expr, offset),
            Expr::OffsetSlice(slice, offset, size) => {
                write!(f, "s_offset({}, {}, size={})", slice, offset, size)
            }
            Expr::Duplicate(expr, count) => write!(f, "dup({}, {})", expr, count),
            Expr::LoadCopy(loc, size) => write!(f, "lcopy({}, {})", loc, size),
            Expr::LoadRef(loc) => write!(f, "&{}", loc),
            Expr::BinOp(binop, lhs, rhs) => write!(f, "({} {} {})", lhs, binop.to_symbol(), rhs),
            Expr::Negative(expr) => write!(f, "-{}", expr),
            Expr::Not(expr) => write!(f, "!{}", expr),
            Expr::Alloc(expr) => write!(f, "alloc({})", expr),
            Expr::Record(exprs) => write!(f, "[{}]", format_iter(exprs.iter())),
            Expr::Wrap(expr) => write!(f, "wrap({})", expr),
            Expr::Unwrap(expr, size) => write!(f, "unwrap({}, {})", expr, size),
            Expr::Call(func, arg, rv_size) => write!(f, "({} {} rv_size={})", func, arg, rv_size),
            Expr::FuncPos(Some(module), func) => write!(f, "@{}-{}", module, func),
            Expr::FuncPos(None, func) => write!(f, "@{}", func),
            Expr::EP => write!(f, "$ep"),
            Expr::TOS(size) => write!(f, "$TOS size={}", size),
            Expr::Seq(stmts, expr) => write!(
                f,
                "{{ {}; {} }}",
                format_iter_delimiter(stmts.iter(), "; "),
                expr
            ),
            Expr::SeqId(id, expr) => write!(f, "seqid({:?}, {})", id, expr),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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

    BeginSeq(SeqId),
    EndSeq(SeqId),
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
            Self::BeginSeq(id) => write!(f, "begin_seq {:?}", id),
            Self::EndSeq(id) => write!(f, "end_seq {:?}", id),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub stack_size: usize,
    pub stack_in_heap_size: usize,
    pub param_size: usize,
    pub body: Expr,
}

impl Function {
    pub fn new() -> Self {
        Self {
            stack_size: 0,
            stack_in_heap_size: 0,
            param_size: 0,
            body: Expr::Unit,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub functions: Vec<(Id, Function)>,
    pub imported_modules: Vec<String>,
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
