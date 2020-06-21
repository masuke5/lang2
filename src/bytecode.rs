use std::collections::{hash_map::Entry, HashMap, LinkedList};
use std::convert::TryInto;
use std::mem;
use std::ptr;
use std::str;

use rustc_hash::FxHashMap;

use crate::id::Id;
use crate::utils;
use crate::value::Lang2Str;

#[allow(dead_code)]
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
    // Logical shift
    pub const BINOP_L_LSHIFT: u8 = 0x1a;
    pub const BINOP_L_RSHIFT: u8 = 0x1b;
    // Arithmetic shift
    pub const BINOP_A_LSHIFT: u8 = 0x1c;
    pub const BINOP_A_RSHIFT: u8 = 0x1d;
    pub const BINOP_BITAND: u8 = 0x1e;
    pub const BINOP_BITOR: u8 = 0x1f;
    pub const BINOP_BITXOR: u8 = 0x20;
    pub const POP: u8 = 0x21;
    pub const ALLOC: u8 = 0x22;
    pub const CALL: u8 = 0x23;
    pub const CALL_NATIVE: u8 = 0x24;
    pub const JUMP: u8 = 0x25;
    pub const JUMP_IF_FALSE: u8 = 0x26;
    pub const JUMP_IF_TRUE: u8 = 0x27;
    pub const RETURN: u8 = 0x28;
    pub const ZERO: u8 = 0x29;
    pub const CALL_EXTERN: u8 = 0x2a;
    pub const TINY_INT: u8 = 0x2b;
    pub const WRAP: u8 = 0x2c;
    pub const UNWRAP: u8 = 0x2d;
    pub const CONST_OFFSET: u8 = 0x2e;
    pub const CALL_POS: u8 = 0x2f;
    pub const CALL_EXTERN_POS: u8 = 0x30;
    pub const LOAD_HEAP: u8 = 0x31;
    pub const LOAD_HEAP_TRACE: u8 = 0x32;
    pub const EP: u8 = 0x33;
    pub const OFFSET_SLICE: u8 = 0x34;
    pub const NOT: u8 = 0x35;
    pub const EXTEND_ARG: u8 = 0x36;
    pub const FLOAT: u8 = 0x37;
    pub const BINOP_FLOAT_ADD: u8 = 0x38;
    pub const BINOP_FLOAT_SUB: u8 = 0x39;
    pub const BINOP_FLOAT_MUL: u8 = 0x4a;
    pub const BINOP_FLOAT_DIV: u8 = 0x4b;

    pub const END: u8 = 0x50;
}

#[inline]
pub fn opcode_name(opcode: u8) -> &'static str {
    match opcode {
        opcode::NOP => "NOP",
        opcode::EXTEND_ARG => "EXTEND_ARG",
        opcode::INT => "INT",
        opcode::TINY_INT => "TINY_INT",
        opcode::FLOAT => "FLOAT",
        opcode::STRING => "STRING",
        opcode::TRUE => "TRUE",
        opcode::FALSE => "FALSE",
        opcode::NULL => "NULL",
        opcode::POINTER => "POINTER",
        opcode::DEREFERENCE => "DEREFERENCE",
        opcode::NEGATIVE => "NEGATIVE",
        opcode::NOT => "NOT",
        opcode::COPY => "COPY",
        opcode::OFFSET => "OFFSET",
        opcode::CONST_OFFSET => "CONST_OFFSET",
        opcode::OFFSET_SLICE => "OFFSET_SLICE",
        opcode::DUPLICATE => "DUPLICATE",
        opcode::LOAD_REF => "LOAD_REF",
        opcode::LOAD_COPY => "LOAD_COPY",
        opcode::EP => "EP",
        opcode::LOAD_HEAP => "LOAD_HEAP",
        opcode::LOAD_HEAP_TRACE => "LOAD_HEAP_TRACE",
        opcode::STORE => "STORE",
        opcode::BINOP_ADD => "BINOP_ADD",
        opcode::BINOP_SUB => "BINOP_SUB",
        opcode::BINOP_MUL => "BINOP_MUL",
        opcode::BINOP_DIV => "BINOP_DIV",
        opcode::BINOP_MOD => "BINOP_MOD",
        opcode::BINOP_LT => "BINOP_LT",
        opcode::BINOP_LE => "BINOP_LE",
        opcode::BINOP_GT => "BINOP_GT",
        opcode::BINOP_GE => "BINOP_GE",
        opcode::BINOP_EQ => "BINOP_EQ",
        opcode::BINOP_NEQ => "BINOP_NEQ",
        opcode::BINOP_L_LSHIFT => "BINOP_L_LSHIFT",
        opcode::BINOP_L_RSHIFT => "BINOP_L_RSHIFT",
        opcode::BINOP_A_LSHIFT => "BINOP_A_LSHIFT",
        opcode::BINOP_A_RSHIFT => "BINOP_A_RSHIFT",
        opcode::BINOP_BITAND => "BINOP_BITAND",
        opcode::BINOP_BITOR => "BINOP_BITOR",
        opcode::BINOP_BITXOR => "BINOP_BITXOR",
        opcode::BINOP_FLOAT_ADD => "BINOP_FLOAT_ADD",
        opcode::BINOP_FLOAT_SUB => "BINOP_FLOAT_SUB",
        opcode::BINOP_FLOAT_MUL => "BINOP_FLOAT_MUL",
        opcode::BINOP_FLOAT_DIV => "BINOP_FLOAT_DIV",
        opcode::POP => "POP",
        opcode::ALLOC => "ALLOC",
        opcode::CALL => "CALL",
        opcode::CALL_POS => "CALL_POS",
        opcode::CALL_NATIVE => "CALL_NATIVE",
        opcode::JUMP => "JUMP",
        opcode::JUMP_IF_FALSE => "JUMP_IF_FALSE",
        opcode::JUMP_IF_TRUE => "JUMP_IF_TRUE",
        opcode::RETURN => "RETURN",
        opcode::ZERO => "ZERO",
        opcode::CALL_EXTERN => "CALL_EXTERN",
        opcode::WRAP => "WRAP",
        opcode::UNWRAP => "UNWRAP",
        opcode::END => "END",
        _ => "UNKNOWN",
    }
}

pub const HEADER: [u8; 4] = *b"L2BC";

macro_rules! int_from_bytes {
    ($ty:ty, $bytes:expr, $pos:expr) => {{
        let size = mem::size_of::<$ty>();
        <$ty>::from_le_bytes($bytes[$pos..$pos + size].try_into().unwrap())
    }};
}

macro_rules! write_to_bytes {
    ($bytes:expr, $pos:expr, $value:expr) => {{
        let value_bytes = $value.to_le_bytes();
        for i in 0..value_bytes.len() {
            $bytes[$pos + i] = value_bytes[i];
        }
    }};
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeHeader {
    pub header: [u8; 4],
    pub func_count: u8,
    pub string_count: u8,
    pub func_map_start: u16,
    pub string_map_start: u16,
    pub module_map_start: u32,
    pub module_count: u32,
    pub ref_start: u16,
    pub len: u32,
}

impl BytecodeHeader {
    const POS_FUNC_COUNT: usize = 4;
    const POS_STRING_COUNT: usize = 5;
    const POS_FUNC_MAP_START: usize = 6;
    const POS_STRING_MAP_START: usize = 8;
    const POS_MODULE_MAP_START: usize = 10;
    const POS_MODULE_COUNT: usize = 14;
    const POS_REF_START: usize = 18;
    const POS_LEN: usize = 20;

    pub fn from_bytes(bytes: &[u8]) -> Option<BytecodeHeader> {
        let size = mem::size_of::<Self>();
        if bytes.len() < size {
            return None;
        }

        Some(Self {
            header: bytes[0..=3].try_into().unwrap(),
            func_count: bytes[Self::POS_FUNC_COUNT],
            string_count: bytes[Self::POS_STRING_COUNT],
            func_map_start: int_from_bytes!(u16, bytes, Self::POS_FUNC_MAP_START),
            string_map_start: int_from_bytes!(u16, bytes, Self::POS_STRING_MAP_START),
            module_map_start: int_from_bytes!(u32, bytes, Self::POS_MODULE_MAP_START),
            module_count: int_from_bytes!(u32, bytes, Self::POS_MODULE_COUNT),
            ref_start: int_from_bytes!(u16, bytes, Self::POS_REF_START),
            len: int_from_bytes!(u32, bytes, Self::POS_LEN),
        })
    }

    pub fn to_bytes(&self) -> [u8; mem::size_of::<Self>()] {
        let mut bytes = [0; mem::size_of::<Self>()];

        bytes[0] = self.header[0];
        bytes[1] = self.header[1];
        bytes[2] = self.header[2];
        bytes[3] = self.header[3];

        write_to_bytes!(&mut bytes, Self::POS_FUNC_COUNT, self.func_count);
        write_to_bytes!(&mut bytes, Self::POS_STRING_COUNT, self.string_count);
        write_to_bytes!(&mut bytes, Self::POS_FUNC_MAP_START, self.func_map_start);
        write_to_bytes!(
            &mut bytes,
            Self::POS_STRING_MAP_START,
            self.string_map_start
        );
        write_to_bytes!(
            &mut bytes,
            Self::POS_MODULE_MAP_START,
            self.module_map_start
        );
        write_to_bytes!(&mut bytes, Self::POS_MODULE_COUNT, self.module_count);
        write_to_bytes!(&mut bytes, Self::POS_REF_START, self.ref_start);
        write_to_bytes!(&mut bytes, Self::POS_LEN, self.len);

        bytes
    }
}

macro_rules! bfn_read {
    ($ty:ty, $name:ident) => {
        #[allow(dead_code, trivial_casts)]
        pub fn $name(&self, pos: usize) -> $ty {
            let ptr = self.bytes.as_ptr();
            unsafe {
                let ptr = ptr.add(pos) as *const $ty;
                *ptr
            }
        }
    };
}

macro_rules! bfn_write {
    ($ty:ty, $push:ident, $write:ident) => {
        #[allow(dead_code)]
        pub fn $write(&mut self, pos: usize, value: $ty) {
            if pos >= self.bytes.len() {
                panic!("out of bounds");
            }

            let bytes = value.to_le_bytes();
            self.write_bytes(pos, &bytes);
        }

        #[allow(dead_code)]
        pub fn $push(&mut self, value: $ty) {
            let bytes = value.to_le_bytes();
            self.push_bytes(&bytes);
        }
    };
}

pub struct Bytecode {
    bytes: Vec<u8>,
}

impl Bytecode {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn push_header(&mut self) {
        if self.len() > 0 {
            panic!("pushed the header already");
        }

        self.push_bytes(&HEADER);
        self.reserve(mem::size_of::<BytecodeHeader>() - 4);
    }

    pub fn write_header(&mut self, header: &BytecodeHeader) {
        let header_bytes = header.to_bytes();
        for i in 0..header_bytes.len() {
            self.bytes[i] = header_bytes[i];
        }
    }

    pub fn read_header(&self) -> BytecodeHeader {
        BytecodeHeader::from_bytes(&self.bytes).unwrap()
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    bfn_read!(u8, read_u8);
    bfn_read!(i8, read_i8);
    bfn_read!(u16, read_u16);
    bfn_read!(i16, read_i16);
    bfn_read!(u32, read_u32);
    bfn_read!(i32, read_i32);
    bfn_read!(u64, read_u64);
    bfn_read!(i64, read_i64);
    bfn_read!(u128, read_u128);
    bfn_read!(i128, read_i128);
    bfn_read!(f64, read_f64);

    pub fn read_bytes(&self, pos: usize, bytes: &mut [u8]) {
        unsafe {
            let src = self.bytes.as_ptr().add(pos);
            let dst = bytes.as_mut_ptr();
            ptr::copy_nonoverlapping(src, dst, bytes.len());
        }
    }

    pub unsafe fn read_str(&self, pos: usize) -> Lang2Str {
        Lang2Str::from_bytes_ptr(&self.bytes[pos])
    }

    bfn_write!(u8, push_u8, write_u8);
    bfn_write!(i8, push_i8, write_i8);
    bfn_write!(u16, push_u16, write_u16);
    bfn_write!(i16, push_i16, write_i16);
    bfn_write!(u32, push_u32, write_u32);
    bfn_write!(i32, push_i32, write_i32);
    bfn_write!(u64, push_u64, write_u64);
    bfn_write!(i64, push_i64, write_i64);
    bfn_write!(u128, push_u128, write_u128);
    bfn_write!(i128, push_i128, write_i128);
    bfn_write!(f64, push_f64, write_f64);

    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.bytes.reserve(self.bytes.len() + bytes.len());
        unsafe {
            let dst = self.bytes.as_mut_ptr().add(self.bytes.len());
            let src = bytes.as_ptr();
            ptr::copy_nonoverlapping(src, dst, bytes.len());

            self.bytes.set_len(self.bytes.len() + bytes.len());
        }
    }

    pub fn push_str(&mut self, s: &str) {
        self.push_u64(s.len() as u64);
        self.push_bytes(s.as_bytes());
    }

    pub fn write_bytes(&mut self, pos: usize, bytes: &[u8]) {
        unsafe {
            let dst = self.bytes.as_mut_ptr().add(pos);
            let src = bytes.as_ptr();
            ptr::copy_nonoverlapping(src, dst, bytes.len());
        }
    }

    #[allow(dead_code)]
    pub fn write_str(&mut self, pos: usize, s: &str) {
        let necessary_bytes = mem::size_of::<u64>() + s.len();
        if pos + necessary_bytes >= self.len() {
            panic!("not enough bytes");
        }

        self.write_u64(pos, s.len() as u64);
        self.write_bytes(pos + mem::size_of::<u64>(), s.as_bytes());
    }

    pub fn reserve(&mut self, size_in_bytes: usize) {
        for _ in 0..size_in_bytes {
            self.bytes.push(0);
        }
    }

    pub fn align(&mut self, n: usize) {
        let new_len = utils::align(self.bytes.len(), n);
        if self.bytes.len() != new_len {
            // Write padding
            for _ in 0..new_len - self.bytes.len() {
                self.bytes.push(0);
            }
        }
    }

    pub fn dump_inst(
        &self,
        opcode: u8,
        arg: u8,
        ip: usize,
        string_map_start: usize,
        ref_start: usize,
    ) {
        print!("{:02x} {:02x}  ", opcode, arg);
        print!("{} ", opcode_name(opcode));

        match opcode {
            opcode::NOP => println!(),
            opcode::EXTEND_ARG => println!(),
            opcode::INT => {
                let value = self.read_i64(ref_start + arg as usize * 8);
                println!("{} ({:x})", arg, value);
            }
            opcode::FLOAT => {
                let value = self.read_f64(ref_start + arg as usize * 8);
                println!("{} ({})", arg, value);
            }
            opcode::TINY_INT => {
                println!("{}", arg);
            }
            opcode::STRING => {
                let string_id = arg as usize;

                let loc = self.read_u16(string_map_start + string_id * 2) as usize;
                let s = unsafe { self.read_str(loc) };

                let escaped_string = utils::escape_string(s.as_str());
                println!("{} (\"{}\") ({})", string_id, escaped_string, loc);
            }
            opcode::TRUE => println!(),
            opcode::FALSE => println!(),
            opcode::NULL => println!(),
            opcode::POINTER => println!(),
            opcode::DEREFERENCE => println!(),
            opcode::NEGATIVE => println!(),
            opcode::NOT => println!(),
            opcode::COPY => println!("size={}", arg),
            opcode::OFFSET => println!(),
            opcode::CONST_OFFSET => println!("{}", arg),
            opcode::OFFSET_SLICE => println!("elem_size={}", arg),
            opcode::DUPLICATE => {
                let value = self.read_u64(ref_start + arg as usize * 8);
                let size = (value >> 32) as usize; // upper 32 bits
                let count = (value as u32) as usize; // lower 32 bits

                println!("{} (size={} count={})", arg, size, count);
            }
            opcode::LOAD_REF => println!("{}", i8::from_le_bytes([arg])),
            opcode::LOAD_COPY => {
                let loc = i8::from_le_bytes([arg & 0b1111_1000]) >> 3;
                let size = arg & 0b0000_0111;
                println!("{} size={}", loc, size);
            }
            opcode::EP => println!(),
            opcode::LOAD_HEAP => println!("{}", i8::from_le_bytes([arg])),
            opcode::LOAD_HEAP_TRACE => println!("{}", i8::from_le_bytes([arg])),
            opcode::STORE => println!("size={}", arg),
            opcode::BINOP_ADD => println!(),
            opcode::BINOP_SUB => println!(),
            opcode::BINOP_MUL => println!(),
            opcode::BINOP_DIV => println!(),
            opcode::BINOP_MOD => println!(),
            opcode::BINOP_LT => println!(),
            opcode::BINOP_LE => println!(),
            opcode::BINOP_GT => println!(),
            opcode::BINOP_GE => println!(),
            opcode::BINOP_EQ => println!(),
            opcode::BINOP_NEQ => println!(),
            opcode::BINOP_L_LSHIFT => println!(),
            opcode::BINOP_L_RSHIFT => println!(),
            opcode::BINOP_A_LSHIFT => println!(),
            opcode::BINOP_A_RSHIFT => println!(),
            opcode::BINOP_BITAND => println!(),
            opcode::BINOP_BITOR => println!(),
            opcode::BINOP_BITXOR => println!(),
            opcode::BINOP_FLOAT_ADD => println!(),
            opcode::BINOP_FLOAT_SUB => println!(),
            opcode::BINOP_FLOAT_MUL => println!(),
            opcode::BINOP_FLOAT_DIV => println!(),
            opcode::POP => println!(),
            opcode::ALLOC => println!("size={}", arg),
            opcode::CALL => println!("{}", arg),
            opcode::CALL_POS => println!(),
            opcode::CALL_EXTERN_POS => println!(),
            opcode::CALL_NATIVE => println!(" unimplemented"),
            opcode::CALL_EXTERN => {
                let module = (arg & 0b1111_0000) >> 4;
                let func = arg & 0b0000_1111;
                println!("{} module={}", func, module);
            }
            opcode::JUMP | opcode::JUMP_IF_FALSE | opcode::JUMP_IF_TRUE => {
                let loc = i8::from_le_bytes([arg]);
                println!("{} ({})", loc, ip as isize + loc as isize * 2);
            }
            opcode::RETURN => println!(),
            opcode::ZERO => println!("count={}", arg),
            opcode::WRAP => println!("size={}", arg),
            opcode::UNWRAP => println!("size={}", arg),
            _ => println!("(0x{:x})", opcode),
        };
    }

    pub fn dump(&self) {
        // Check header
        if self.bytes.len() < HEADER.len() || self.bytes[..HEADER.len()] != HEADER {
            println!(
                "Invalid header `{}`",
                str::from_utf8(&self.bytes[..HEADER.len()]).unwrap_or("")
            );
            return;
        }

        let index_len = format!("{}", self.bytes.len()).len();

        let header = match BytecodeHeader::from_bytes(&self.bytes) {
            Some(bh) => bh,
            None => {
                println!("Invalid header");
                return;
            }
        };

        let string_count = header.string_count as usize;
        let string_map_start = header.string_map_start as usize;

        // String map
        let count_len = format!("{}", string_count).len();
        for i in 0..string_count {
            let loc = self.read_u16(string_map_start + i * 2) as usize;
            print!("{:<width$}  ", loc, width = index_len);

            // Read the string
            let s = unsafe { self.read_str(loc) };

            println!(
                "{:<width$} \"{}\"",
                i,
                utils::escape_string(s.as_str()),
                width = count_len
            );
        }

        // Module map
        let module_map_start = header.module_map_start as usize;
        let module_count = header.module_count as usize;

        for i in 0..module_count {
            let loc = self.read_u16(module_map_start + i * 2) as usize;
            print!("{:<width$}  ", loc, width = index_len);

            // Read the module name
            let s = unsafe { self.read_str(loc) };

            println!(
                "{:<width$} import {}",
                i,
                utils::escape_string(s.as_str()),
                width = count_len
            );
        }

        // Function map
        let func_map_start = header.func_map_start as usize;
        let func_count = header.func_count as usize;

        let mut functions: Vec<Function> = Vec::new();

        for j in 0..func_count {
            let loc = func_map_start + j * 8;
            let func_id = j as u16;

            let mut bytes = [0; 8];
            self.read_bytes(loc, &mut bytes);

            let func = Function::from_bytes(func_id, bytes);

            println!("{:<width$}", loc, width = index_len);
            println!("  id: {}", func_id);
            println!("  stack_in_heap_size: {}", func.stack_in_heap_size);
            println!("  stack_size: {}", func.stack_size);
            println!("  param_size: {}", func.param_size);
            println!("  pos: {}", func.pos);

            functions.push(func);
        }

        let ref_start = header.ref_start as usize;

        for Function { code_id, pos, .. } in functions {
            println!("FUNC {}:", code_id);

            let mut inst = [0u8; 2];
            let mut i = 0;

            // Load first instruction
            self.read_bytes(pos + i * 2, &mut inst);

            while inst[0] != opcode::END {
                print!("{:<width$}  ", pos + i * 2, width = index_len);

                self.dump_inst(inst[0], inst[1], pos + i * 2, string_map_start, ref_start);

                i += 1;
                self.read_bytes(pos + i * 2, &mut inst);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: Option<Id>,
    pub code_id: u16,
    pub stack_in_heap_size: usize,
    pub stack_size: usize,
    pub param_size: usize,
    pub pos: usize,
}

impl Function {
    pub const OFFSET_STACK_IN_HEAP_SIZE: usize = 0;
    pub const OFFSET_STACK_SIZE: usize = 1;
    pub const OFFSET_PARAM_SIZE: usize = 2;
    pub const OFFSET_POS: usize = 4;

    #[inline]
    pub fn name(&self) -> Id {
        self.name.unwrap()
    }

    pub fn from_bytes(code_id: u16, bytes: [u8; 8]) -> Self {
        Self {
            name: None,
            code_id,
            param_size: bytes[Self::OFFSET_PARAM_SIZE] as usize,
            stack_in_heap_size: bytes[Self::OFFSET_STACK_IN_HEAP_SIZE] as usize,
            stack_size: bytes[Self::OFFSET_STACK_SIZE] as usize,
            pos: u16::from_le_bytes(
                bytes[Self::OFFSET_POS..Self::OFFSET_POS + 2]
                    .try_into()
                    .unwrap(),
            ) as usize,
        }
    }

    pub fn to_bytes(&self) -> [u8; 8] {
        let mut bytes = [0u8; 8];

        bytes[Self::OFFSET_STACK_IN_HEAP_SIZE] = self.stack_in_heap_size as u8;
        bytes[Self::OFFSET_STACK_SIZE] = self.stack_size as u8;
        bytes[Self::OFFSET_PARAM_SIZE] = self.param_size as u8;
        bytes[Self::OFFSET_POS..Self::OFFSET_POS + 2]
            .copy_from_slice(&(self.pos as u16).to_le_bytes());

        bytes
    }
}

#[derive(Debug)]
pub struct InstList {
    pub insts: LinkedList<[u8; 2]>,
    pub labels: FxHashMap<usize, usize>,
    jumps: FxHashMap<usize, usize>,
}

impl InstList {
    pub fn new() -> Self {
        InstList {
            insts: LinkedList::new(),
            labels: FxHashMap::default(),
            jumps: FxHashMap::default(),
        }
    }

    pub fn add_label(&mut self, label: usize) {
        self.labels.insert(label, self.len());
    }

    pub fn jumps(&self) -> &FxHashMap<usize, usize> {
        &self.jumps
    }

    // Insert an instruction

    #[inline]
    pub fn append(&mut self, mut insts: InstList) {
        for (name, label) in insts.labels {
            self.labels.insert(name, label + self.len());
        }

        for (index, label) in insts.jumps {
            self.jumps.insert(index + self.len(), label);
        }

        self.insts.append(&mut insts.insts);
    }

    #[inline]
    #[allow(dead_code)]
    pub fn prepend(&mut self, mut insts: InstList) {
        let insts_len = insts.len();

        mem::swap(&mut self.insts, &mut insts.insts);
        self.insts.append(&mut insts.insts);

        for label in self.labels.values_mut() {
            *label += insts_len;
        }
        self.labels.extend(insts.labels);

        let old_jumps: Vec<(usize, usize)> = self.jumps.drain().collect();
        for (index, label) in old_jumps {
            self.jumps.insert(index + insts_len, label);
        }
        self.jumps.extend(insts.jumps);
    }

    #[inline]
    pub fn push(&mut self, opcode: u8, arg: u32) {
        if arg >= std::u8::MIN as u32 && arg <= std::u8::MAX as u32 {
            self.insts.push_back([opcode, arg as u8]);
        } else {
            // Panic if arg is 24-bit
            if arg > ((1 << 24) - 1) {
                panic!("too large arg: {}", arg);
            }

            let first_arg = arg >> 16;

            let more_arg = arg as u16;
            let second_arg = more_arg >> 8;
            let third_arg = more_arg as u8;

            self.push_noarg(opcode::EXTEND_ARG);
            self.insts.push_back([opcode, first_arg as u8]);
            self.insts.push_back([second_arg as u8, third_arg]);
        }
    }

    #[inline]
    pub fn pushi(&mut self, opcode: u8, arg: i32) {
        if arg >= std::i8::MIN as i32 && arg <= std::i8::MAX as i32 {
            self.insts.push_back([opcode, arg as u8]);
        } else {
            let first_arg = (arg >> 16) as i8;

            let more_arg = arg as i16;
            let second_arg = (more_arg >> 8) as i8;
            let third_arg = more_arg as i8;

            self.push_noarg(opcode::EXTEND_ARG);
            self.insts.push_back([opcode, first_arg.to_le_bytes()[0]]);
            self.insts
                .push_back([second_arg.to_le_bytes()[0], third_arg.to_le_bytes()[0]]);
        }
    }

    #[inline]
    pub fn push_jump(&mut self, opcode: u8, label: usize) {
        self.jumps.insert(self.len(), label);
        self.push_noarg(opcode);
    }

    #[inline]
    pub fn push_noarg(&mut self, opcode: u8) {
        self.push(opcode, 0);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.insts.len()
    }
}

pub struct BytecodeBuilder {
    pub code: Bytecode,

    header: BytecodeHeader,
    functions: HashMap<Id, Function>,
    func_count: usize,
    modules: Vec<String>,
    refs: Vec<u64>,
}

impl BytecodeBuilder {
    pub fn new() -> Self {
        let mut code = Bytecode::new();
        code.push_header();

        BytecodeBuilder {
            header: BytecodeHeader::from_bytes(&code.bytes).unwrap(),
            code,
            functions: HashMap::new(),
            func_count: 0,
            modules: Vec::new(),
            refs: Vec::new(),
        }
    }

    pub fn new_ref_u64(&mut self, value: u64) -> usize {
        self.refs.push(value);
        self.refs.len() - 1
    }

    pub fn new_ref_i64(&mut self, value: i64) -> usize {
        self.refs.push(unsafe { mem::transmute(value) });
        self.refs.len() - 1
    }

    pub fn new_ref_f64(&mut self, value: f64) -> usize {
        self.refs.push(u64::from_le_bytes(value.to_le_bytes()));
        self.refs.len() - 1
    }

    fn write_strings(&mut self, strings: &[String]) {
        if strings.len() > std::u16::MAX as usize {
            panic!("too many strings");
        }

        self.header.string_count = strings.len() as u8;

        self.code.align(8);

        let string_map_start = self.code.len();
        self.header.string_map_start = string_map_start as u16;

        type StringId = u16;

        // Reserve for string map
        self.code
            .reserve(strings.len() * mem::size_of::<StringId>());
        self.code.align(8);

        // Write strings
        for (i, string) in strings.iter().enumerate() {
            // Write the string location to string map
            self.code.write_u16(
                string_map_start + i * mem::size_of::<StringId>(),
                self.code.len() as u16,
            );

            // Write the string
            self.code.push_str(&string);
            self.code.align(8);
        }
    }

    fn write_modules(&mut self) {
        self.header.module_count = self.modules.len() as u32;

        self.code.align(8);

        let module_map_start = self.code.len();
        self.header.module_map_start = self.code.len() as u32;

        type ModuleId = u16;

        self.code
            .reserve(self.modules.len() * mem::size_of::<ModuleId>());
        self.code.align(8);

        for (id, name) in self.modules.iter().enumerate() {
            self.code.write_u16(
                module_map_start + id * mem::size_of::<ModuleId>(),
                self.code.len() as u16,
            );

            self.code.push_str(name);
            self.code.align(8);
        }
    }

    fn write_funcs(&mut self) {
        self.code.align(8);

        // Write function map start
        self.header.func_map_start = self.code.len() as u16;
        // Write the number of functions
        self.header.func_count = self.func_count as u8;

        let mut functions: Vec<Function> = self.functions.values().cloned().collect();
        functions.sort_by_key(|f| f.code_id);

        for func in &functions {
            let base = self.code.len();

            self.code.reserve(8);

            let bytes = func.to_bytes();
            self.code.write_bytes(base, &bytes);
        }
    }

    fn write_refs(&mut self) {
        self.code.align(8);

        self.header.ref_start = self.code.len() as u16;

        for value in &self.refs {
            self.code.push_u64(*value);
        }
    }

    // Function

    pub fn insert_function_header(&mut self, mut func: Function) {
        if self.func_count > std::u16::MAX as usize {
            panic!("too many functions");
        }

        if let Entry::Vacant(entry) = self.functions.entry(func.name()) {
            // Set ID
            func.code_id = self.func_count as u16;
            self.func_count += 1;

            entry.insert(func);
        }
    }

    pub fn push_function_body(&mut self, func_id: Id, mut insts: InstList) {
        insts.push_noarg(opcode::END);

        let func = self
            .functions
            .get_mut(&func_id)
            .expect("the function header must be added");
        func.pos = self.code.len();

        // Write instruction bytes
        for inst in &insts.insts {
            self.code.push_bytes(inst);
        }

        self.code.align(8);
    }

    pub fn push_module(&mut self, name: &str) -> u16 {
        self.modules.push(name.to_string());
        self.modules.len() as u16 - 1
    }

    pub fn build(mut self, strings: &[String]) -> Bytecode {
        self.write_strings(strings);
        self.write_modules();
        self.write_funcs();
        self.write_refs();

        self.header.len = self.code.len() as u32;
        self.code.write_header(&self.header);

        self.code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytecode_write() {
        let mut bytecode = Bytecode::new();
        bytecode.push_header();
        let base = bytecode.len();
        bytecode.push_u64(123123123123);
        bytecode.push_u64(9);
        bytecode.push_u64(789789789789);
        bytecode.write_u64(base + 8, 456456456456);

        assert_eq!(bytecode.read_u64(base), 123123123123);
        assert_eq!(bytecode.read_u64(bytecode.len() - 16), 456456456456);
    }

    #[test]
    fn instlist_append() {
        let mut insts = InstList::new();
        insts.push(opcode::TINY_INT, 30);
        insts.add_label(0);
        insts.push(opcode::TINY_INT, 50);
        insts.push_jump(opcode::JUMP_IF_FALSE, 0);

        let mut insts2 = InstList::new();
        insts2.push_noarg(opcode::BINOP_ADD);
        insts2.add_label(1);
        insts2.push(opcode::TINY_INT, 60);
        insts2.push_jump(opcode::JUMP, 1);
        insts2.push_noarg(opcode::BINOP_MUL);

        insts.append(insts2);

        let expected: LinkedList<[u8; 2]> = vec![
            [opcode::TINY_INT, 30],
            [opcode::TINY_INT, 50],
            [opcode::JUMP_IF_FALSE, 0],
            [opcode::BINOP_ADD, 0],
            [opcode::TINY_INT, 60],
            [opcode::JUMP, 0],
            [opcode::BINOP_MUL, 0],
        ]
        .into_iter()
        .collect();

        assert_eq!(expected, insts.insts);
        assert_eq!(insts.labels, vec![(0, 1), (1, 4),].into_iter().collect());
        assert_eq!(insts.jumps, vec![(2, 0), (5, 1),].into_iter().collect());
    }

    #[test]
    fn instlist_prepend() {
        let mut insts = InstList::new();
        insts.push_noarg(opcode::BINOP_ADD);
        insts.push_jump(opcode::JUMP, 0);
        insts.push(opcode::TINY_INT, 50);
        insts.add_label(0);
        insts.push_noarg(opcode::BINOP_MUL);

        let mut insts2 = InstList::new();
        insts2.push_jump(opcode::JUMP, 1);
        insts2.push(opcode::TINY_INT, 30);
        insts2.add_label(1);
        insts2.push(opcode::TINY_INT, 60);

        insts.prepend(insts2);

        let expected: LinkedList<[u8; 2]> = vec![
            [opcode::JUMP, 0],
            [opcode::TINY_INT, 30],
            [opcode::TINY_INT, 60],
            [opcode::BINOP_ADD, 0],
            [opcode::JUMP, 0],
            [opcode::TINY_INT, 50],
            [opcode::BINOP_MUL, 0],
        ]
        .into_iter()
        .collect();

        assert_eq!(expected, insts.insts);
        assert_eq!(insts.labels, vec![(0, 6), (1, 2)].into_iter().collect());
        assert_eq!(insts.jumps, vec![(0, 1), (4, 0)].into_iter().collect());
    }

    #[test]
    fn func_to_bytes() {
        let func = Function {
            name: None,
            code_id: 30,
            stack_in_heap_size: 10,
            stack_size: 20,
            param_size: 12,
            pos: 19,
        };
        let bytes = func.to_bytes();
        let actual = Function::from_bytes(30, bytes);

        assert_eq!(func, actual);
    }
}
