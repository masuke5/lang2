use crate::ast::SymbolPath;
use crate::id::Id;
use opcode::Opcode;
use rustc_hash::FxHashMap;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use thiserror::Error;

#[allow(dead_code)]
pub mod opcode {
    #[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
    pub struct Opcode(u8);

    pub const NOP: Opcode = Opcode(0x1);
    pub const INT: Opcode = Opcode(0x2);
    pub const TINY_INT: Opcode = Opcode(0x3);
    pub const FLOAT: Opcode = Opcode(0x4);
    pub const STRING: Opcode = Opcode(0x5);
    pub const TRUE: Opcode = Opcode(0x6);
    pub const FALSE: Opcode = Opcode(0x7);
    pub const NULL: Opcode = Opcode(0x8);
    pub const DUP: Opcode = Opcode(0x9);
    pub const POP: Opcode = Opcode(0xa);
    pub const ALLOC: Opcode = Opcode(0xb);
    // LV = Local Variable
    // EV = Escaped Variable
    pub const LOAD_LV: Opcode = Opcode(0xc);
    pub const LOAD_EV: Opcode = Opcode(0xd);
    pub const LOAD_EV_OUTER: Opcode = Opcode(0xe);
    pub const LOAD_REF_LV: Opcode = Opcode(0xf);
    pub const LOAD_REF_EV: Opcode = Opcode(0x10);
    pub const LOAD_REF_EV_OUTER: Opcode = Opcode(0x11);
    pub const STORE_LV: Opcode = Opcode(0x12);
    pub const STORE_EV: Opcode = Opcode(0x13);
    pub const STORE_EV_OUTER: Opcode = Opcode(0x14);
    pub const STORE_REF: Opcode = Opcode(0x15);
    pub const DEREF: Opcode = Opcode(0x16);
    pub const OFFSET: Opcode = Opcode(0x17);
    pub const OFFSET_CONST: Opcode = Opcode(0x18);
    pub const INEG: Opcode = Opcode(0x19);
    pub const FNEG: Opcode = Opcode(0x1a);
    pub const NOT: Opcode = Opcode(0x1b);
    pub const BINOP_IADD: Opcode = Opcode(0x1c);
    pub const BINOP_ISUB: Opcode = Opcode(0x1d);
    pub const BINOP_IMUL: Opcode = Opcode(0x1e);
    pub const BINOP_IDIV: Opcode = Opcode(0x1f);
    pub const BINOP_IMOD: Opcode = Opcode(0x20);
    pub const BINOP_FADD: Opcode = Opcode(0x21);
    pub const BINOP_FSUB: Opcode = Opcode(0x22);
    pub const BINOP_FMUL: Opcode = Opcode(0x23);
    pub const BINOP_FDIV: Opcode = Opcode(0x24);
    pub const BINOP_ILT: Opcode = Opcode(0x25);
    pub const BINOP_ILE: Opcode = Opcode(0x26);
    pub const BINOP_IGT: Opcode = Opcode(0x27);
    pub const BINOP_IGE: Opcode = Opcode(0x28);
    pub const BINOP_IEQ: Opcode = Opcode(0x29);
    pub const BINOP_INE: Opcode = Opcode(0x2a);
    pub const BINOP_FLT: Opcode = Opcode(0x2b);
    pub const BINOP_FLE: Opcode = Opcode(0x2c);
    pub const BINOP_FGT: Opcode = Opcode(0x2d);
    pub const BINOP_FGE: Opcode = Opcode(0x2e);
    pub const BINOP_FEQ: Opcode = Opcode(0x2f);
    pub const BINOP_FNE: Opcode = Opcode(0x30);
    pub const BINOP_REF_EQ: Opcode = Opcode(0x31);
    pub const BINOP_REF_NE: Opcode = Opcode(0x32);
    pub const BINOP_LSHL: Opcode = Opcode(0x33);
    pub const BINOP_LSHR: Opcode = Opcode(0x34);
    pub const BINOP_ASHL: Opcode = Opcode(0x35);
    pub const BINOP_ASHR: Opcode = Opcode(0x36);
    pub const BINOP_AND: Opcode = Opcode(0x37);
    pub const BINOP_OR: Opcode = Opcode(0x38);
    pub const BINOP_XOR: Opcode = Opcode(0x39);
    pub const CALL: Opcode = Opcode(0x3a);
    pub const CALL_REF: Opcode = Opcode(0x3b);
    pub const CALL_NATIVE: Opcode = Opcode(0x3c);
    pub const RETURN: Opcode = Opcode(0x3d);
    pub const JUMP: Opcode = Opcode(0x3e);
    pub const JUMP_IF_TRUE: Opcode = Opcode(0x3f);
    pub const JUMP_IF_FALSE: Opcode = Opcode(0x40);
}

const MAX_ARG: u32 = 0b1111_1111_1111_1111_1111_1111;

static NEXT_LABEL: AtomicU32 = AtomicU32::new(0);

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct Label(u32);

impl Label {
    pub fn new() -> Self {
        let id = Self(NEXT_LABEL.load(Ordering::Acquire));
        NEXT_LABEL.fetch_add(1, Ordering::Acquire);
        id
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unique({})", self.0)
    }
}

impl fmt::Debug for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum BuilderError {
    #[error("out of range (> {0})")]
    OutOfRange(u64),
    #[error("{0} (> {1})")]
    TooMany(&'static str, u64),
}

#[derive(Debug, Clone, PartialEq)]
enum Inst {
    Normal(Opcode, u32),
    Jump(Opcode, Label),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub escaped_stack_size: u32,
    pub stack_size: u32,
    labels: FxHashMap<Label, usize>,
    insts: Vec<Inst>,
}

impl Function {
    pub fn new() -> Self {
        Self {
            escaped_stack_size: 0,
            stack_size: 0,
            insts: Vec::new(),
            labels: FxHashMap::default(),
        }
    }

    pub fn define_local_variable(&mut self) -> Result<u32, BuilderError> {
        if self.stack_size > MAX_ARG {
            return Err(BuilderError::TooMany(
                "too many local variables",
                MAX_ARG as u64,
            ));
        }

        let loc = self.stack_size;
        self.stack_size += 1;
        Ok(loc)
    }

    pub fn define_escaped_variable(&mut self) -> Result<u32, BuilderError> {
        if self.stack_size > MAX_ARG {
            return Err(BuilderError::TooMany(
                "too many escaped variables",
                MAX_ARG as u64,
            ));
        }

        let loc = self.stack_size;
        self.stack_size += 1;
        Ok(loc)
    }

    pub fn push_noarg(&mut self, opcode: Opcode) {
        self.insts.push(Inst::Normal(opcode, 0));
    }

    pub fn push<T: Into<u32>>(&mut self, opcode: Opcode, arg: T) -> Result<(), BuilderError> {
        let arg = arg.into();
        if arg > MAX_ARG {
            return Err(BuilderError::OutOfRange(MAX_ARG as u64));
        }

        self.insts.push(Inst::Normal(opcode, arg));
        Ok(())
    }

    pub fn push_jump(&mut self, opcode: Opcode, label: Label) {
        self.insts.push(Inst::Jump(opcode, label));
    }

    pub fn set_label_here(&mut self, label: Label) {
        self.labels.insert(label, self.insts.len());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BytecodeBuilder {
    module_path: SymbolPath,
    integers: Vec<u64>,
    floats: Vec<f64>,
    strings: Vec<String>,
    functions: FxHashMap<Id, Function>,
}

impl BytecodeBuilder {
    pub fn new(module_path: &SymbolPath) -> Self {
        Self {
            module_path: module_path.clone(),
            integers: Vec::new(),
            floats: Vec::new(),
            strings: Vec::new(),
            functions: FxHashMap::default(),
        }
    }

    pub fn push_int(&mut self, n: u64) -> Result<u32, BuilderError> {
        if self.integers.len() >= MAX_ARG as usize {
            return Err(BuilderError::TooMany(
                "too many integer literals",
                MAX_ARG as u64,
            ));
        }

        self.integers.push(n);
        Ok((self.integers.len() - 1) as u32)
    }

    pub fn push_float(&mut self, n: f64) -> Result<u32, BuilderError> {
        if self.floats.len() >= MAX_ARG as usize {
            return Err(BuilderError::TooMany(
                "too many floating number literals",
                MAX_ARG as u64,
            ));
        }

        self.floats.push(n);
        Ok((self.floats.len() - 1) as u32)
    }

    pub fn push_string(&mut self, s: String) -> Result<u32, BuilderError> {
        if self.strings.len() >= MAX_ARG as usize {
            return Err(BuilderError::TooMany(
                "too many string literals",
                MAX_ARG as u64,
            ));
        }

        self.strings.push(s);
        Ok((self.strings.len() - 1) as u32)
    }

    pub fn get_func(&self, name: Id) -> &Function {
        self.functions.get(&name).unwrap()
    }

    pub fn get_func_mut(&mut self, name: Id) -> &mut Function {
        self.functions.get_mut(&name).unwrap()
    }
}
