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
    pub param_count: usize,
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
    pub fn new(name: Id, param_count: usize) -> Self {
        Self {
            name,
            param_count,
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
    Record(usize),
    // make a pointer from a reference
    Pointer,
    // dereference a pointer
    Dereference,

    Load(isize),
    Store,

    Field(usize),
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
    Return,

}

pub fn dump_insts(insts: &[Inst]) {
    let index_len = format!("{}", insts.len()).len();

    for (i, inst) in insts.iter().enumerate() {
        print!("{:<width$} ", i, width = index_len);

        match inst {
            Inst::Int(n) => println!("int {}", n),
            Inst::String(s) => println!("string \"{}\"", utils::escape_string(s)),
            Inst::True => println!("true"),
            Inst::False => println!("false"),
            Inst::Load(loc) => println!("load_ref {}", loc),
            Inst::Record(size) => println!("record size={}", size),
            Inst::Pointer => println!("pointer"),
            Inst::Dereference => println!("deref"),
            Inst::Field(i) => println!("field {}", i),
            Inst::BinOp(binop) => {
                match binop {
                    BinOp::Add => println!("add"),
                    BinOp::Sub => println!("sub"),
                    BinOp::Mul => println!("mul"),
                    BinOp::Div => println!("div"),
                    BinOp::Mod => println!("mod"),
                    BinOp::LessThan => println!("less_than"),
                    BinOp::LessThanOrEqual => println!("less_than_or_equal"),
                    BinOp::GreaterThan => println!("greater_than"),
                    BinOp::GreaterThanOrEqual => println!("greater_than_or_equal"),
                    BinOp::Equal => println!("equal"),
                    BinOp::NotEqual => println!("not_equal"),
                };
            },
            Inst::Store => println!("store"),
            Inst::Call(name) => {
                println!("call {}", IdMap::name(*name));
            },
            #[cfg(debug_assertions)]
            Inst::CallNative(name, _, param_count) => {
                println!("call_native {} params={}", IdMap::name(*name), param_count);
            },
            #[cfg(not(debug_assertions))]
            Inst::CallNative(_, param_count) => {
                println!("call_native params={}", param_count);
            },
            Inst::Pop => println!("pop"),
            Inst::Jump(i) => println!("jump {}", i),
            Inst::JumpIfZero(i) => println!("jump_if_zero {}", i),
            Inst::JumpIfNonZero(i) => println!("jump_if_non_zero {}", i),
            Inst::Return => println!("return"),
        }
    }
}
