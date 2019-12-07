use std::fmt;
use std::ptr;
use std::mem;
use std::collections::HashMap;

use crate::ty::Type;
use crate::id::{Id, IdMap};
use crate::value::Value;
use crate::utils;
use crate::vm::Context;

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
    pub pos: usize,
    pub ref_start: usize,
    pub insts: Vec<Inst>, // TODO: remove later
}

#[derive(Clone)]
pub struct NativeFunctionBody(pub fn(&mut Context) -> Vec<Value>);

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
            pos: 0,
            ref_start: 0,
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
    Null,
    // make a pointer from a reference
    Pointer,
    // dereference a pointer
    Dereference,
    Negative,
    Copy(usize),
    Offset,
    Duplicate(usize, usize),
    Load(isize),
    LoadCopy(isize, usize),
    StoreWithSize(usize),
    BinOp(BinOp),
    Pop,
    Alloc(usize),

    Call(Id),
    CallNative(Id, NativeFunctionBody, usize),

    Jump(usize),
    JumpIfZero(usize),
    JumpIfNonZero(usize),
    Return,
}

impl fmt::Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Inst::Int(n) => write!(f, "int {}", n),
            Inst::String(s) => write!(f, "string \"{}\"", utils::escape_string(s)),
            Inst::True => write!(f, "true"),
            Inst::False => write!(f, "false"),
            Inst::Null => write!(f, "__null__"),
            Inst::Load(loc) => write!(f, "load_ref {}", loc),
            Inst::LoadCopy(loc, size) => write!(f, "load_copy {} size={}", loc, size),
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
            Inst::Alloc(size) => write!(f, "alloc size={}", size),
            Inst::Call(name) => {
                write!(f, "call {}", IdMap::name(*name))
            },
            Inst::CallNative(name, _, param_count) => {
                write!(f, "call_native {} params={}", IdMap::name(*name), param_count)
            },
            Inst::Pop => write!(f, "pop"),
            Inst::Jump(i) => write!(f, "jump {}", i),
            Inst::JumpIfZero(i) => write!(f, "jump_if_zero {}", i),
            Inst::JumpIfNonZero(i) => write!(f, "jump_if_non_zero {}", i),
            Inst::Return => write!(f, "return"),
        }
    }
}

pub fn dump_insts(insts: &[Inst]) {
    let index_len = format!("{}", insts.len()).len();

    for (i, inst) in insts.iter().enumerate() {
        println!("{:<width$} {}", i, inst, width = index_len);
    }
}

pub mod opcode {
    pub const NOP: u8 = 0x0;
    pub const INT: u8 = 0x1;
    pub const STRING: u8 = 0x2;
    pub const TRUE: u8 = 0x3;
    pub const FALSE: u8 = 0x4;
    pub const NULL: u8 = 0x5;
    pub const POINTER: u8 = 0x6;
    pub const DEREFERENCE: u8 = 0x7;
    pub const NEGATIVE: u8 = 0x8;
    pub const COPY: u8 = 0x9;
    pub const OFFSET: u8 = 0xa;
    pub const DUPLICATE: u8 = 0xb;
    pub const LOAD_REF: u8 = 0xc;
    pub const LOAD_COPY: u8 = 0xd;
    pub const STORE: u8 = 0xe;
    pub const BINOP_ADD: u8 = 0xf;
    pub const BINOP_SUB: u8 = 0x10;
    pub const BINOP_MUL: u8 = 0x11;
    pub const BINOP_DIV: u8 = 0x12;
    pub const BINOP_MOD: u8 = 0x13;
    pub const BINOP_LT: u8 = 0x14;
    pub const BINOP_LE: u8 = 0x15;
    pub const BINOP_GT: u8 = 0x16;
    pub const BINOP_GE: u8 = 0x17;
    pub const BINOP_EQ: u8 = 0x18;
    pub const BINOP_NEQ: u8 = 0x19;
    pub const POP: u8 = 0x1a;
    pub const ALLOC: u8 = 0x1b;
    pub const CALL: u8 = 0x1c;
    pub const CALL_NATIVE: u8 = 0x1d;
    pub const JUMP: u8 = 0x1e;
    pub const JUMP_IF_FALSE: u8 = 0x1f;
    pub const JUMP_IF_TRUE: u8 = 0x20;
    pub const RETURN: u8 = 0x21;
    pub const ZERO: u8 = 0x22;
}

macro_rules! impl_toref {
    ($ty:ty) => {
        impl ToRef for $ty {
            fn convert(self) -> u64 {
                // TODO: Add check
                unsafe { mem::transmute(self.to_le()) }
            }
        }
    };
}

pub trait ToRef {
    fn convert(self) -> u64;
}

impl_toref!(u64);
impl_toref!(i64);
impl_toref!(usize);
impl_toref!(isize);

#[derive(Debug, Copy, Clone)]
pub struct Jump(usize);

#[derive(Debug)]
pub struct Bytecode {
    pub code: Vec<u8>,
    functions: HashMap<Id, Function>,
    refs: Vec<u64>,
    current_func_id: Option<Id>,
}

impl Bytecode {
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            functions: HashMap::new(),
            refs: Vec::with_capacity(50),
            current_func_id: None,
        }
    }

    // Function

    pub fn new_function(&mut self, func: Function) {
        self.functions.insert(func.name, func);
    }

    pub fn begin_function(&mut self, id: Id) {
        self.functions.get_mut(&id).unwrap().pos = self.code.len();
        self.refs.clear();
        self.current_func_id = Some(id);
    }

    pub fn end_function(&mut self, id: Id) {
        self.functions.get_mut(&id).unwrap().ref_start = self.code.len();
        self.code.push(self.refs.len() as u8);

        // alignment
        let mut padding = 8 - self.code.len() % 8;
        if padding == 8 {
            padding = 0
        }

        self.code.resize(self.code.len() + padding, 0);

        // Push refs
        let refs_size = self.refs.len() * 8; // in bytes
        self.code.reserve(refs_size);

        unsafe {
            let dst = self.code.as_mut_ptr().add(self.code.len());
            let src = self.refs.as_ptr() as *const u8;
            ptr::copy_nonoverlapping(src, dst, refs_size);

            self.code.set_len(self.code.len() + refs_size);
        }

        self.current_func_id = None;
    }

    pub fn get_function(&self, id: Id) -> Option<&Function> {
        self.functions.get(&id)
    }

    pub fn current_func(&self) -> &Function {
        self.functions.get(self.current_func_id.as_ref().unwrap()).unwrap()
    }

    pub fn current_func_mut(&mut self) -> &mut Function {
        self.functions.get_mut(self.current_func_id.as_ref().unwrap()).unwrap()
    }

    // Insert

    #[inline]
    pub fn insert_inst_noarg(&mut self, opcode: u8) {
        self.code.push(opcode);
        self.code.push(0);
    }

    #[inline]
    pub fn insert_inst(&mut self, opcode: u8, arg: u8) {
        self.code.push(opcode);
        self.code.push(arg);
    }

    #[inline]
    pub fn insert_inst_ref(&mut self, opcode: u8, arg: impl ToRef) {
        let arg = self.new_ref(arg);
        // TODO: Add support for values above u8
        self.insert_inst(opcode, arg as u8);
    }

    pub fn new_ref(&mut self, value: impl ToRef) -> usize {
        self.refs.push(value.convert());
        self.refs.len() - 1
    }

    // Jump

    pub fn jump(&mut self) -> Jump {
        self.insert_inst_noarg(opcode::NOP);
        Jump(self.code.len() - 3)
    }

    pub fn insert_jump_inst(&mut self, jump: Jump) {
        let index = jump.0;
        self.code[index] = opcode::JUMP;
        self.code[index + 1] = self.new_ref(index) as u8;
    }


    pub fn insert_jump_if_false_inst(&mut self, jump: Jump) {
        let index = jump.0;
        self.code[index] = opcode::JUMP_IF_FALSE;
        self.code[index + 1] = self.new_ref(index) as u8;
    }

    pub fn insert_jump_if_true_inst(&mut self, jump: Jump) {
        let index = jump.0;
        self.code[index] = opcode::JUMP_IF_TRUE;
        self.code[index + 1] = self.new_ref(index) as u8;
    }

    pub fn dump(&self) {
        let index_len = format!("{}", self.code.len()).len();
        let mut i = 0;

        loop {
            // Find the function
            let mut current_func = None;
            for func in self.functions.values() {
                if i == func.pos {
                    current_func = Some(func);
                    break;
                }
            }

            if current_func.is_none() {
                break;
            }

            let current_func = current_func.unwrap();

            println!("{}:", IdMap::name(current_func.name));

            while i < current_func.ref_start {
                print!("{:<width$}  ", i, width = index_len);
                match self.code[i] {
                    opcode::NOP => println!("NOP"),
                    opcode::INT => println!("INT &{}", self.code[i + 1]),
                    opcode::STRING => println!("STRING &{}", self.code[i + 1]),
                    opcode::TRUE => println!("TRUE"),
                    opcode::FALSE => println!("FALSE"),
                    opcode::NULL => println!("NULL"),
                    opcode::POINTER => println!("POINTER"),
                    opcode::DEREFERENCE => println!("DEREFERENCE"),
                    opcode::NEGATIVE => println!("NEGATIVE"),
                    opcode::COPY => println!("COPY size={}", self.code[i + 1]),
                    opcode::OFFSET => println!("OFFSET"),
                    opcode::DUPLICATE => println!("DUPLICATE"),
                    opcode::LOAD_REF => println!("LOAD_REF {}", self.code[i + 1]),
                    opcode::LOAD_COPY => println!("LOAD_COPY unimplemented"),
                    opcode::STORE => println!("STORE size={}", self.code[i + 1]),
                    opcode::BINOP_ADD => println!("BINOP_ADD"),
                    opcode::BINOP_SUB => println!("BINOP_SUB"),
                    opcode::BINOP_MUL => println!("BINOP_MUL"),
                    opcode::BINOP_DIV => println!("BINOP_DIV"),
                    opcode::BINOP_MOD => println!("BINOP_MOD"),
                    opcode::BINOP_LT => println!("BINOP_LT"),
                    opcode::BINOP_LE => println!("BINOP_LE"),
                    opcode::BINOP_GT => println!("BINOP_GT"),
                    opcode::BINOP_GE => println!("BINOP_GE"),
                    opcode::BINOP_EQ => println!("BINOP_EQ"),
                    opcode::BINOP_NEQ => println!("BINOP_NEQ"),
                    opcode::POP => println!("POP"),
                    opcode::ALLOC => println!("ALLOC size={}", self.code[i + 1]),
                    opcode::CALL => println!("CALL unimplemented"),
                    opcode::CALL_NATIVE => println!("CALL_NATIVE unimplemented"),
                    opcode::JUMP => println!("JUMP &{}", self.code[i + 1]),
                    opcode::JUMP_IF_FALSE => println!("JUMP_IF_FALSE &{}", self.code[i + 1]),
                    opcode::JUMP_IF_TRUE => println!("JUMP_IF_TRUE &{}", self.code[i + 1]),
                    opcode::RETURN => println!("RETURN"),
                    opcode::ZERO => println!("ZERO"),
                    _ => println!("UNKNOWN"),
                };

                i += 2;
            }

            let ref_count = self.code[i] as usize;
            i += 1;

            // Skip padding
            while i % 8 != 0 {
                i += 1;
            }

            let index_start = i;
            let limit = i + ref_count * 8;
            while i < limit {
                let index = (i - index_start) / 8;
                let value = unsafe { *((self.code.as_ptr().add(i)) as *const i64) };
                println!("{:<width$}({})  {}", i, index, value, width = index_len);
                i += 8;
            }
        }
    }
}

