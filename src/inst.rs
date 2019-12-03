use std::fmt;

use crate::ty::Type;
use crate::id::{Id, IdMap};
use crate::value::Value;
use crate::utils;

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: Id,
    pub stack_size: usize,
    pub param_size: usize,
    pub insts: Vec<Inst>,
}

#[derive(Clone)]
pub struct NativeFunctionBody(pub fn(&[Value]) -> Value);

impl fmt::Debug for NativeFunctionBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[funcition pointer]") 
    }
}

#[derive(Debug, Clone)]
pub struct NativeFunction {
    pub params: Vec<Type>,
    pub return_ty: Type,
    pub body: NativeFunctionBody,
}

impl Function {
    pub fn new(name: Id, param_size: usize) -> Self {
        Self {
            name,
            param_size,
            stack_size: 0,
            insts: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Inst {
    Int(i64),
    String(String),
    True,
    False,
    // make a pointer from a reference
    Pointer,
    // dereference a pointer
    Dereference,
    Negative,
    Copy(usize),
    Offset,
    Duplicate(usize, usize),
    Load(isize),
    StoreWithSize(usize),
    BinOp(BinOp),
    Call(Id),
    Pop,
    #[cfg(debug_assertions)]
    CallNative(Id, NativeFunctionBody, usize),
    #[cfg(not(debug_assertions))]
    CallNative(NativeFunctionBody, usize),

    Jump(usize),
    JumpIfZero(usize),
    JumpIfNonZero(usize),
    Return(usize),

}

impl fmt::Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Inst::Int(n) => write!(f, "int {}", n),
            Inst::String(s) => write!(f, "string \"{}\"", utils::escape_string(s)),
            Inst::True => write!(f, "true"),
            Inst::False => write!(f, "false"),
            Inst::Load(loc) => write!(f, "load_ref {}", loc),
            Inst::Pointer => write!(f, "pointer"),
            Inst::Dereference => write!(f, "deref"),
            Inst::Negative => write!(f, "neg"),
            Inst::Copy(size) => write!(f, "copy size={}", size),
            Inst::Duplicate(size, count) => write!(f, "duplicate size={}, count={}", size, count),
            Inst::Offset => write!(f, "offset"),
            Inst::BinOp(binop) => {
                match binop {
                    BinOp::Add => write!(f, "add"),
                    BinOp::Sub => write!(f, "sub"),
                    BinOp::Mul => write!(f, "mul"),
                    BinOp::Div => write!(f, "div"),
                    BinOp::Mod => write!(f, "mod"),
                    BinOp::LessThan => write!(f, "less_than"),
                    BinOp::LessThanOrEqual => write!(f, "less_than_or_equal"),
                    BinOp::GreaterThan => write!(f, "greater_than"),
                    BinOp::GreaterThanOrEqual => write!(f, "greater_than_or_equal"),
                    BinOp::Equal => write!(f, "equal"),
                    BinOp::NotEqual => write!(f, "not_equal"),
                }
            },
            Inst::StoreWithSize(size) => write!(f, "store size={}", size),
            Inst::Call(name) => {
                write!(f, "call {}", IdMap::name(*name))
            },
            #[cfg(debug_assertions)]
            Inst::CallNative(name, _, param_count) => {
                write!(f, "call_native {} params={}", IdMap::name(*name), param_count)
            },
            #[cfg(not(debug_assertions))]
            Inst::CallNative(_, param_count) => {
                write!(f, "call_native params={}", param_count);
            },
            Inst::Pop => write!(f, "pop"),
            Inst::Jump(i) => write!(f, "jump {}", i),
            Inst::JumpIfZero(i) => write!(f, "jump_if_zero {}", i),
            Inst::JumpIfNonZero(i) => write!(f, "jump_if_non_zero {}", i),
            Inst::Return(size) => write!(f, "return size={}", size),
        }
    }
}

pub fn dump_insts(insts: &[Inst]) {
    let index_len = format!("{}", insts.len()).len();

    for (i, inst) in insts.iter().enumerate() {
        println!("{:<width$} {}", i, inst, width = index_len);
    }
}
