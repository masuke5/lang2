use std::ptr;
use std::mem;
use std::slice;
use std::str;
use std::collections::HashMap;

use crate::id::{Id, IdMap};
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
    pub code_id: u16,
    pub stack_size: usize,
    pub param_size: usize,
    pub pos: usize,
    pub ref_start: usize,
}

impl Function {
    pub fn new(name: Id, param_size: usize) -> Self {
        Self {
            name,
            code_id: 0,
            param_size,
            stack_size: 0,
            pos: 0,
            ref_start: 0,
        }
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

    pub const END: u8 = 0x23;
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
    next_func_id: u16,
    funcmap_start: usize,
    string_map_start: usize,
    strings: Vec<String>,
}

impl Bytecode {
    pub fn new(strings: Vec<String>) -> Self {
        let mut slf = Self {
            code: vec![
                0x4c, 0x32, 0x42, 0x43, // "L2BC"
                0, // the number of function
                0, // the number of string
                0, // funcmap_start (2byte)
                0,
                0, // string_map_start (2byte)
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            functions: HashMap::new(),
            refs: Vec::with_capacity(50),
            current_func_id: None,
            next_func_id: 0,
            funcmap_start: 0,
            string_map_start: 16,
            strings,
        };

        slf.code[5] = slf.strings.len() as u8;

        // Reserve string map
        slf.code[8] = 16;
        slf.code.resize(slf.code.len() + slf.strings.len() * 2, 0);

        // Write string entities
        unsafe {
            slf.code.resize(utils::align(slf.code.len(), 8), 0);

            for (i, string) in slf.strings.iter().enumerate() {
                slf.code.reserve(string.len() + 8);

                // Write string location
                let map_ptr = slf.code.as_mut_ptr().add(slf.string_map_start) as *mut u16;
                *map_ptr.add(i) = slf.code.len() as u16;

                // Write string length
                let len = slf.code.as_mut_ptr().add(slf.code.len()) as *mut usize;
                *len = string.len().to_le();

                // Write string bytes
                let bytes_ptr = len.add(1) as *mut u8;
                let src = string.as_bytes().as_ptr();
                ptr::copy_nonoverlapping(src, bytes_ptr, string.len());

                slf.code.set_len(slf.code.len() + string.len() + 8);
                slf.code.resize(utils::align(slf.code.len(), 8), 0);
            }
        }

        // Write funcmap_start
        slf.funcmap_start = slf.code.len();
        unsafe {
            let funcmap_start = &mut slf.code[6] as *mut u8 as *mut u16;
            *funcmap_start = slf.funcmap_start as u16;
        }

        slf
    }

    // Function

    pub fn new_function(&mut self, mut func: Function) {
        if self.code.len() % 4 != 0 {
            panic!("not algined");
        }

        if self.next_func_id > std::u16::MAX {
            panic!("too many functions");
        }

        let id = self.next_func_id;
        func.code_id = id;
        self.functions.insert(func.name, func);

        self.code.reserve(8);
        unsafe {
            let ptr = self.code.as_mut_ptr().add(self.code.len()) as *mut u16;
            *ptr = id.to_le();

            *ptr.add(1) = 0;
            *ptr.add(2) = 0;
            *ptr.add(3) = 0;

            self.code.set_len(self.code.len() + 8);
        }

        self.next_func_id += 1;
    }

    pub fn end_new_function(&mut self) {
        self.code.resize(utils::align(self.code.len(), 8), 0);

        // Set function count
        self.code[4] = self.functions.len() as u8;
    }

    pub fn begin_function(&mut self, id: Id) {
        let func = self.functions.get_mut(&id).unwrap();
        func.pos = self.code.len();
        self.refs.clear();
        self.current_func_id = Some(id);
    }

    pub fn end_function(&mut self, id: Id) {
        self.insert_inst_noarg(opcode::END);

        self.code.resize(utils::align(self.code.len(), 8), 0);

        let func = self.functions.get_mut(&id).unwrap();
        func.ref_start = self.code.len();

        // Push refs
        let refs_size = self.refs.len() * 8; // in bytes
        self.code.reserve(refs_size);

        unsafe {
            let dst = self.code.as_mut_ptr().add(self.code.len());
            let src = self.refs.as_ptr() as *const u8;
            ptr::copy_nonoverlapping(src, dst, refs_size);

            self.code.set_len(self.code.len() + refs_size);
        }

        // Set function infomations
        unsafe {
            let ptr = self.code.as_mut_ptr().add(self.funcmap_start) as *mut u8;
            let key = ptr.add(func.code_id as usize * 8) as *mut u16;

            let stack_size = key.add(1) as *mut u8;
            *stack_size = func.stack_size.to_le() as u8;

            let param_size = stack_size.add(1);
            *param_size = func.param_size.to_le() as u8;

            let pos = param_size.add(1) as *mut u16;
            *pos = func.pos.to_le() as u16;

            let ref_start = pos.add(1);
            *ref_start = func.ref_start.to_le() as u16;
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
        Jump(self.code.len() - 2)
    }

    pub fn insert_jump_inst(&mut self, jump: Jump) {
        let index = jump.0;
        self.code[index] = opcode::JUMP;
        self.code[index + 1] = index as u8;
    }


    pub fn insert_jump_if_false_inst(&mut self, jump: Jump) {
        let index = jump.0;
        self.code[index] = opcode::JUMP_IF_FALSE;
        self.code[index + 1] = index as u8;
    }

    pub fn insert_jump_if_true_inst(&mut self, jump: Jump) {
        let index = jump.0;
        self.code[index] = opcode::JUMP_IF_TRUE;
        self.code[index + 1] = index as u8;
    }

    pub fn dump(&self) {
        let index_len = format!("{}", self.code.len()).len();

        fn read<T: Copy>(code: &[u8], pos: usize) -> T {
            if pos + mem::size_of::<T>() >= code.len() {
                panic!("out of bounds");
            }

            unsafe { *(code.as_ptr().add(pos) as *const T) }
        }
        
        fn get_ptr<T>(code: &[u8], pos: usize) -> *const T {
            if pos + mem::size_of::<T>() >= code.len() {
                panic!("out of bounds");
            }

            unsafe { code.as_ptr().add(pos) as *const T }
        }

        // String map
        let string_count = read::<u8>(&self.code, 5) as usize;
        let count_len = format!("{}", string_count).len();
        for i in 0..string_count {
            let loc = read::<u16>(&self.code, self.string_map_start + i * 2) as usize;
            print!("{:<width$}  ", loc, width = index_len);

            let len = read::<usize>(&self.code, loc);
            let bytes = get_ptr::<u8>(&self.code, loc + 8);
            let slice = unsafe { slice::from_raw_parts(bytes, len) };
            let raw = str::from_utf8(slice).unwrap();

            println!("{:<width$} \"{}\"", i, utils::escape_string(raw), width = count_len);
        }

        // Function map
        let func_count = self.code[4] as usize;

        type Func = (u16, u8, u8, u16, u16); // id, stack_size, param_size, pos, ref_start
        let mut functions: Vec<Func> = Vec::new();

        unsafe {
            for j in 0..func_count {
                let loc = self.funcmap_start + j * 8;
                let key = self.code.as_ptr().add(loc) as *const u16;
                let func_id = key;
                let stack_size = key.add(1) as *const u8;
                let param_size = stack_size.add(1);
                let pos = param_size.add(1) as *const u16;
                let ref_start = pos.add(1);

                println!("{:<width$}", loc, width = index_len);
                println!("  id: {}", *func_id);
                println!("  stack_size: {}", *stack_size);
                println!("  param_size: {}", *param_size);
                println!("  pos: {}", *pos);
                println!("  ref_start: {}", *ref_start);

                functions.push((*func_id, *stack_size, *param_size, *pos, *ref_start));
            }
        }

        for (id, _, _, pos, ref_start) in functions {
            let mut i = pos as usize;

            let func = self.functions.values().find(|func| func.code_id == id).unwrap();
            println!("{} ({}):", IdMap::name(func.name), id);

            while self.code[i] != opcode::END {
                print!("{:<width$}  ", i, width = index_len);
                match self.code[i] {
                    opcode::NOP => println!("NOP"),
                    opcode::INT => {
                        let value = &self.code[ref_start as usize + self.code[i + 1] as usize * 8] as *const u8 as *const i64;
                        let value = unsafe { *value };
                        println!("INT {} ({})", self.code[i + 1], value);
                    },
                    opcode::STRING => {
                        let (s, loc) = {
                            let string_id = read::<u8>(&self.code, i + 1) as usize;
                            let loc = read::<u16>(&self.code, self.string_map_start + string_id * 2) as usize;
                            let len = read::<usize>(&self.code, loc);
                            let bytes = get_ptr::<u8>(&self.code, loc + 8);

                            let slice = unsafe { slice::from_raw_parts(bytes, len) };
                            let raw = str::from_utf8(slice).unwrap();

                            (raw, loc)
                        };

                        let escaped_string = utils::escape_string(s);
                        println!("STRING {} (\"{}\") ({})", self.code[i + 1], escaped_string, loc);
                    },
                    opcode::TRUE => println!("TRUE"),
                    opcode::FALSE => println!("FALSE"),
                    opcode::NULL => println!("NULL"),
                    opcode::POINTER => println!("POINTER"),
                    opcode::DEREFERENCE => println!("DEREFERENCE"),
                    opcode::NEGATIVE => println!("NEGATIVE"),
                    opcode::COPY => println!("COPY size={}", self.code[i + 1]),
                    opcode::OFFSET => println!("OFFSET"),
                    opcode::DUPLICATE => println!("DUPLICATE"),
                    opcode::LOAD_REF => println!("LOAD_REF {}", i8::from_le_bytes([self.code[i + 1]])),
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
                    opcode::CALL => println!("CALL {}", self.code[i + 1]),
                    opcode::CALL_NATIVE => println!("CALL_NATIVE unimplemented"),
                    opcode::JUMP => println!("JUMP &{}", self.code[i + 1]),
                    opcode::JUMP_IF_FALSE => println!("JUMP_IF_FALSE &{}", self.code[i + 1]),
                    opcode::JUMP_IF_TRUE => println!("JUMP_IF_TRUE &{}", self.code[i + 1]),
                    opcode::RETURN => println!("RETURN"),
                    opcode::ZERO => println!("ZERO count={}", self.code[i + 1]),
                    _ => println!("UNKNOWN (0x{:x})", self.code[i]),
                };

                i += 2;
            }
        }
    }
}

