use std::fmt;

use crate::ty::Type;
use crate::id::{Id, IdMap};
use crate::value::Value;

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
    And,
    Or,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: Id,
    pub stack_size: usize,
    pub params: Vec<Type>,
    pub insts: Vec<Inst>,
    pub return_ty: Type,
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
    pub fn new(name: Id, params: Vec<Type>, return_ty: Type) -> Self {
        Self {
            name,
            params,
            stack_size: 0,
            insts: Vec::new(),
            return_ty,
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
    Load(isize, usize),

    BinOp(BinOp),
    Save(isize, usize),
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

pub fn dump_insts(insts: &[Inst], id_map: &IdMap) {
    let index_len = format!("{}", insts.len()).len();

    for (i, inst) in insts.iter().enumerate() {
        print!("{:<width$} ", i, width = index_len);

        match inst {
            Inst::Int(n) => println!("int {}", n),
            Inst::String(s) => println!("string \"{}\"", s),
            Inst::True => println!("true"),
            Inst::False => println!("false"),
            Inst::Load(loc, offset) => {
                print!("load {}", loc);
                if *offset > 0 {
                    println!(" offset={}", offset);
                } else {
                    println!();
                }
            },
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
                    BinOp::And => println!("and"),
                    BinOp::Or => println!("or"),
                };
            },
            Inst::Save(loc, offset) => {
                print!("save {}", loc);
                if *offset > 0 {
                    println!(" offset={}", offset);
                } else {
                    println!();
                }
            },
            Inst::Call(name) => {
                println!("call {}", id_map.name(&name));
            },
            #[cfg(debug_assertions)]
            Inst::CallNative(name, _, param_count) => {
                println!("call_native {} params={}", id_map.name(&name), param_count);
            },
            #[cfg(not(debug_assertions))]
            Inst::CallNative(_, param_count) => {
                println!("call_native params={}", param_count);
            },
            Inst::Pop => println!("pop"),
            Inst::Jump(i) => println!("jump {}", i),
            Inst::JumpIfZero(i) => println!("jump_if_zero {}", i),
            Inst::JumpIfNonZero(i) => println!("jump_if_non_zero {}", i),
            Inst::Return(size) => println!("return size={}", size),
        }
    }
}
