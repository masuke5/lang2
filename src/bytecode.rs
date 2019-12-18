use std::mem;
use std::ptr;
use std::str;
use std::collections::HashMap;
use std::io;
use std::io::{Read, Write, Seek, SeekFrom};

use crate::id::{Id, IdMap};
use crate::utils;
use crate::module::ModuleHeader;

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
    pub const POP: u8 = 0x1a;
    pub const ALLOC: u8 = 0x1b;
    pub const CALL: u8 = 0x1c;
    pub const CALL_NATIVE: u8 = 0x1d;
    pub const JUMP: u8 = 0x1e;
    pub const JUMP_IF_FALSE: u8 = 0x1f;
    pub const JUMP_IF_TRUE: u8 = 0x20;
    pub const RETURN: u8 = 0x21;
    pub const ZERO: u8 = 0x22;
    pub const CALL_EXTERN: u8 = 0x23;

    pub const END: u8 = 0x24;
}

#[inline]
pub fn opcode_name(opcode: u8) -> &'static str {
    match opcode {
        opcode::NOP => "NOP",
        opcode::INT => "INT",
        opcode::STRING => "STRING",
        opcode::TRUE => "TRUE",
        opcode::FALSE => "FALSE",
        opcode::NULL => "NULL",
        opcode::POINTER => "POINTER",
        opcode::DEREFERENCE => "DEREFERENCE",
        opcode::NEGATIVE => "NEGATIVE",
        opcode::COPY => "COPY",
        opcode::OFFSET => "OFFSET",
        opcode::DUPLICATE => "DUPLICATE",
        opcode::LOAD_REF => "LOAD_REF",
        opcode::LOAD_COPY => "LOAD_COPY",
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
        opcode::POP => "POP",
        opcode::ALLOC => "ALLOC",
        opcode::CALL => "CALL",
        opcode::CALL_NATIVE => "CALL_NATIVE",
        opcode::JUMP => "JUMP",
        opcode::JUMP_IF_FALSE => "JUMP_IF_FALSE",
        opcode::JUMP_IF_TRUE => "JUMP_IF_TRUE",
        opcode::RETURN => "RETURN",
        opcode::ZERO => "ZERO",
        opcode::CALL_EXTERN => "CALL_EXTERN",
        opcode::END => "END",
        _ => "UNKNOWN",
    }
}

pub const HEADER: [u8; 4] = *b"L2BC";
pub const POS_FUNC_COUNT: u64 = 4;
pub const POS_STRING_COUNT: u64 = 5;
pub const POS_MODULE_COUNT: u64 = 13;
pub const POS_FUNC_MAP_START: u64 = 6;
pub const POS_STRING_MAP_START: u64 = 8;
pub const POS_MODULE_MAP_START: u64 = 10;

pub const FUNC_OFFSET_STACK_SIZE: u64 = 2;
pub const FUNC_OFFSET_PARAM_SIZE: u64 = 3;
pub const FUNC_OFFSET_POS: u64 = 4;
pub const FUNC_OFFSET_REF_START: u64 = 6;

macro_rules! bfn_read {
    ($ty:ty, $name:ident) => {
        #[allow(dead_code)]
        pub fn $name(&self, pos: usize) -> $ty {
            let ptr = self.bytes.as_ptr();
            unsafe {
                let ptr = ptr.add(pos) as *const $ty;
                *ptr
            }
        }
    };
}

pub struct Bytecode {
    bytes: Vec<u8>,
}

impl Bytecode {
    pub fn from_stream<W: Read + Seek>(mut stream: BytecodeStream<W>) -> Self {
        // TODO: with_capacity
        let mut bytecode = Self { bytes: Vec::new() };
        stream.read_to_end(&mut bytecode.bytes);

        bytecode
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

    pub fn read_bytes(&self, pos: usize, bytes: &mut [u8]) {
        unsafe {
            let src = self.bytes.as_ptr().add(pos);
            let dst = bytes.as_mut_ptr();
            ptr::copy_nonoverlapping(src, dst, bytes.len());
        }
    }

    pub fn dump_inst(&self, opcode: u8, arg: u8, pos: usize, ref_start: usize, string_map_start: usize) {
        print!("{} ", opcode_name(opcode));

        match opcode {
            opcode::NOP => println!(),
            opcode::INT => {
                let value = self.read_i64(ref_start + arg as usize * 8);
                println!("{} ({})", arg, value);
            },
            opcode::STRING => {
                let string_id = arg as usize;
                // Read string location from string map
                let loc = self.read_u16(string_map_start + string_id * 2) as usize;
                // Read string length
                let len = self.read_u64(loc) as usize;

                // Read string bytes
                let mut buf = Vec::with_capacity(len);
                buf.resize(len, 0);
                self.read_bytes(loc + 8, &mut buf[..]);

                // Convert to string
                let raw = str::from_utf8(&buf).unwrap();

                let escaped_string = utils::escape_string(&raw);
                println!("{} (\"{}\") ({})", string_id, escaped_string, loc);
            },
            opcode::TRUE => println!(),
            opcode::FALSE => println!(),
            opcode::NULL => println!(),
            opcode::POINTER => println!(),
            opcode::DEREFERENCE => println!(),
            opcode::NEGATIVE => println!(),
            opcode::COPY => println!("size={}", arg),
            opcode::OFFSET => println!(),
            opcode::DUPLICATE => println!(" unimplemented"),
            opcode::LOAD_REF => println!("{}", i8::from_le_bytes([arg])),
            opcode::LOAD_COPY => {
                let loc = i8::from_le_bytes([arg & 0b11111000]) >> 3;
                let size = arg & 0b00000111;
                println!("{} size={}", loc, size);
            },
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
            opcode::POP => println!(),
            opcode::ALLOC => println!("size={}", arg),
            opcode::CALL => println!("{}", arg),
            opcode::CALL_NATIVE => println!(" unimplemented"),
            opcode::CALL_EXTERN => {
                let module = (arg & 0b11110000) >> 4;
                let func = arg & 0b00001111;
                println!("{} module={}", func, module);
            },
            opcode::JUMP | opcode::JUMP_IF_FALSE | opcode::JUMP_IF_TRUE => {
                println!("{} ({})", arg, pos + arg as usize);
            },
            opcode::RETURN => println!(),
            opcode::ZERO => println!("count={}", arg),
            _ => println!("(0x{:x})", opcode),
        };
    }

    pub fn dump(&self) {
        let index_len = format!("{}", self.bytes.len()).len();

        // String map
        let string_map_start = self.read_u16(POS_STRING_MAP_START as usize) as usize;
        let string_count = self.read_u8(POS_STRING_COUNT as usize) as usize;

        let count_len = format!("{}", string_count).len();
        for i in 0..string_count {
            let loc = self.read_u16(string_map_start + i * 2) as usize;
            print!("{:<width$}  ", loc, width = index_len);

            // Read the string length
            let len = self.read_u64(loc) as usize;

            // Read the string bytes
            let mut buf = Vec::with_capacity(len);
            buf.resize(len, 0);
            self.read_bytes(loc + 8, &mut buf[..]);

            let raw = str::from_utf8(&buf).unwrap();

            println!("{:<width$} \"{}\"", i, utils::escape_string(raw), width = count_len);
        }

        // Module map
        let module_map_start = self.read_u16(POS_MODULE_MAP_START as usize) as usize;
        let module_count = self.read_u8(POS_MODULE_COUNT as usize) as usize;

        for i in 0..module_count {
            let loc = self.read_u16(module_map_start + i * 2) as usize;
            print!("{:<width$}  ", loc, width = index_len);

            // Read the string length
            let len = self.read_u16(loc) as usize;

            // Read the string bytes
            let mut buf = Vec::with_capacity(len);
            buf.resize(len, 0);
            self.read_bytes(loc + 2, &mut buf[..]);

            let raw = str::from_utf8(&buf).unwrap();

            println!("{:<width$} import {}", i, utils::escape_string(raw), width = count_len);
        }

        // Function map
        let func_map_start = self.read_u16(POS_FUNC_MAP_START as usize) as usize;
        let func_count = self.read_u8(POS_FUNC_COUNT as usize) as usize;

        type Func = (u16, u8, u8, usize, usize); // id, stack_size, param_size, pos, ref_start
        let mut functions: Vec<Func> = Vec::new();

        for j in 0..func_count {
            let loc = func_map_start + j * 8;
            let func_id = self.read_u16(loc);
            let stack_size = self.read_u8(loc + FUNC_OFFSET_STACK_SIZE as usize);
            let param_size = self.read_u8(loc + FUNC_OFFSET_PARAM_SIZE as usize);
            let pos = self.read_u16(loc + FUNC_OFFSET_POS as usize) as usize;
            let ref_start = self.read_u16(loc + FUNC_OFFSET_REF_START as usize) as usize;

            println!("{:<width$}", loc, width = index_len);
            println!("  id: {}", func_id);
            println!("  stack_size: {}", stack_size);
            println!("  param_size: {}", param_size);
            println!("  pos: {}", pos);
            println!("  ref_start: {}", ref_start);

            functions.push((func_id, stack_size, param_size, pos, ref_start));
        }

        for (id, _, _, pos, ref_start) in functions {
            println!("FUNC {}:", id);

            let mut inst = [0u8; 2];
            let mut i = 0;

            // Load first instruction
            self.read_bytes(pos + i * 2, &mut inst);

            while inst[0] != opcode::END {
                print!("{:<width$}  ", pos + i * 2, width = index_len);

                self.dump_inst(inst[0], inst[1], pos, ref_start, string_map_start);

                i += 1;
                self.read_bytes(pos + i * 2, &mut inst);
            }
        }
    }
}

// FIXME: Avoid unwrap()
macro_rules! fn_write {
    ($ty:ty, $write:ident, $push:ident) => {
        #[allow(dead_code)]
         pub fn $write(&mut self, pos: u64, value: $ty) {
             self.code.seek(SeekFrom::Start(pos)).unwrap();
             self.code.write_all(&value.to_le_bytes()).unwrap();
         }

        #[allow(dead_code)]
         pub fn $push(&mut self, value: $ty) {
             self.code.seek(SeekFrom::End(0)).unwrap();
             self.code.write_all(&value.to_le_bytes()).unwrap();
             self.len += mem::size_of::<$ty>() as u64;
         }
    };
}

// FIXME: Avoid unwrap()
macro_rules! fn_read {
    ($ty:ty, $read:ident) => {
        #[allow(dead_code)]
        pub fn $read(&mut self, pos: u64) -> $ty {
            let mut buf = [0; mem::size_of::<$ty>()];
            self.code.seek(SeekFrom::Start(pos)).unwrap();
            self.code.read_exact(&mut buf).unwrap();
            <$ty>::from_le_bytes(buf)
        }
    };
}

#[derive(Debug)]
pub struct BytecodeStream<W> {
    code: W,
    len: u64,
}

impl<W: Seek> BytecodeStream<W> {
    pub fn new(code: W) -> Self {
        BytecodeStream {
            code,
            len: 0,
        }
    }
    
    pub fn len(&self) -> u64 {
        self.len
    }
}

// FIXME: Avoid unwrap()
impl<W: Write + Seek> BytecodeStream<W> {
    pub fn push_header(&mut self) {
        if self.len > 0 {
            panic!("pushed the header already");
        }

        self.push_bytes(&HEADER);
        self.push_bytes(&[
            0, // the number of function
            0, // the number of string
            0, // funcmap_start (2byte)
            0,
            0, // string_map_start (2byte)
            0,
            0, // the number of module
            0,
            0,
            0,
            0,
            0,
        ]);
    }

    fn_write!(i8, write_i8, push_i8);
    fn_write!(u8, write_u8, push_u8);
    fn_write!(i16, write_i16, push_i16);
    fn_write!(u16, write_u16, push_u16);
    fn_write!(i32, write_i32, push_i32);
    fn_write!(u32, write_u32, push_u32);
    fn_write!(i64, write_i64, push_i64);
    fn_write!(u64, write_u64, push_u64);
    fn_write!(i128, write_i128, push_i128);
    fn_write!(u128, write_u128, push_u128);

    pub fn write_bytes(&mut self, pos: u64, bytes: &[u8]) {
        self.code.seek(SeekFrom::Start(pos as u64)).unwrap();
        self.code.write_all(bytes).unwrap();
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.code.seek(SeekFrom::End(0)).unwrap();
        self.code.write_all(bytes).unwrap();
        self.len += bytes.len() as u64;
    }

    pub fn reserve(&mut self, size_in_bytes: u64) {
        self.code.seek(SeekFrom::End(0)).unwrap();
        for _ in 0..size_in_bytes {
            self.code.write_all(&[0]).unwrap();
        }

        self.len += size_in_bytes;
    }

    pub fn align(&mut self, n: u64) {
        let new_len = utils::align(self.len as usize, n as usize) as u64;
        if self.len != new_len {
            // Write padding
            self.code.seek(SeekFrom::End(0)).unwrap();
            for _ in 0..new_len - self.len {
                self.code.write_all(&[0]).unwrap();
            }

            self.len = new_len;
        }
    }
}

impl<R: Read + Seek> BytecodeStream<R> {
    // FIXME: Avoid panic!
    #[allow(dead_code)]
    pub fn from(mut code: R) -> Result<Self, io::Error> {
        let mut header = [0u8; 4];
        code.read_exact(&mut header)?;

        if header != HEADER {
            panic!("different header");
        }

        let len = code.seek(SeekFrom::End(0)).unwrap();

        Ok(BytecodeStream {
            code,
            len,
        })
    }

    fn_read!(i8, read_i8);
    fn_read!(u8, read_u8);
    fn_read!(i16, read_i16);
    fn_read!(u16, read_u16);
    fn_read!(i32, read_i32);
    fn_read!(u32, read_u32);
    fn_read!(i64, read_i64);
    fn_read!(u64, read_u64);
    fn_read!(i128, read_i128);
    fn_read!(u128, read_u128);

    pub fn read_to_end(&mut self, buf: &mut Vec<u8>) {
        self.code.seek(SeekFrom::Start(0)).unwrap();
        self.code.read_to_end(buf).unwrap();
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: Id,
    pub code_id: u16,
    pub stack_size: u8,
    pub param_size: u8,
    pub pos: u16,
    pub ref_start: u16,
}

impl Function {
    pub fn new(name: Id, param_size: usize) -> Self {
        Self {
            name,
            code_id: 0,
            param_size: param_size as u8,
            stack_size: 0,
            pos: 0,
            ref_start: 0,
        }
    }
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
pub struct Jump(u64);

#[derive(Debug)]
pub struct BytecodeBuilder<W: Read + Write + Seek> {
    pub code: BytecodeStream<W>,
    functions: HashMap<Id, Function>,
    refs: Vec<u64>,
    current_func_id: Option<Id>,
    next_func_id: u16,
    funcmap_start: u16,
    string_map_start: u16,
    module_map_start: u16,
    prev_inst: [u8; 2],
}

impl<W: Read + Write + Seek> BytecodeBuilder<W> {
    pub fn new(mut code: BytecodeStream<W>, strings: &[String], modules: &[&ModuleHeader]) -> Self {
        if code.len() > 0 {
            panic!("written bytecode already");
        }

        code.push_header();

        let mut slf = BytecodeBuilder {
            code,
            functions: HashMap::new(),
            refs: Vec::with_capacity(50),
            current_func_id: None,
            next_func_id: 0,
            funcmap_start: 0,
            string_map_start: 16,
            module_map_start: 0,
            prev_inst: [0; 2],
        };
        slf.write_strings(strings);
        slf.write_modules(modules);

        // Write function map start
        slf.funcmap_start = slf.code.len() as u16;
        slf.code.write_u16(POS_FUNC_MAP_START, slf.funcmap_start as u16);

        slf
    }

    fn write_strings(&mut self, strings: &[String]) {
        if strings.len() > std::u16::MAX as usize {
            panic!("too many strings");
        }

        self.code.write_u8(POS_STRING_COUNT, strings.len() as u8);
        self.code.write_u16(POS_STRING_MAP_START, self.code.len() as u16);

        // Reserve for string map
        self.code.reserve(strings.len() as u64 * 2);

        // Write strings
        self.code.align(8);

        for (i, string) in strings.iter().enumerate() {
            // Write the string location to string map
            self.code.write_u16(self.string_map_start as u64 + i as u64 * 2, self.code.len() as u16);

            // Write the string length and bytes
            self.code.push_u64(string.len() as u64);
            self.code.push_bytes(string.as_bytes());

            self.code.align(8);
        }
    }

    fn write_modules(&mut self, modules: &[&ModuleHeader]) {
        if modules.len() > 0b1111 {
            panic!("too many modules");
        }

        self.code.write_u8(POS_MODULE_COUNT, modules.len() as u8);

        self.code.align(8);

        self.module_map_start = self.code.len() as u16;
        self.code.write_u16(POS_MODULE_MAP_START, self.code.len() as u16);

        self.code.reserve(modules.len() as u64 * 2);
        self.code.align(8);

        for (id, module) in modules.iter().enumerate() {
            self.code.write_u16(self.module_map_start as u64 + id as u64 * 2, self.code.len() as u16);

            let module_name = IdMap::name(module.id);

            self.code.push_u16(module_name.len() as u16);
            self.code.push_bytes(module_name.as_bytes());

            self.code.align(8);
        }
    }

    pub fn prev_inst(&self) -> [u8; 2] {
        self.prev_inst
    }
    
    // Function

    pub fn new_function(&mut self, mut func: Function) {
        if self.next_func_id > std::u16::MAX {
            panic!("too many functions");
        }

        // Generate ID and set it
        let id = self.next_func_id;
        func.code_id = id;
        self.next_func_id += 1;

        self.functions.insert(func.name, func);

        // Reserve for function infomations
        self.code.push_u16(id);
        self.code.reserve(6);
    }

    pub fn end_new_function(&mut self) {
        self.code.align(8);

        // Set function count
        self.code.write_u8(POS_FUNC_COUNT, self.functions.len() as u8);
    }

    pub fn begin_function(&mut self, id: Id) {
        let func = self.functions.get_mut(&id).unwrap();
        // Set function position in bytecode
        func.pos = self.code.len() as u16;

        self.current_func_id = Some(id);
    }

    pub fn end_function(&mut self, id: Id) {
        self.insert_inst_noarg(opcode::END);

        self.code.align(8);

        let func = self.functions.get_mut(&id).unwrap();
        func.ref_start = self.code.len() as u16;

        // Set function infomations
        let func = self.get_function(id).unwrap().clone();
        let base = self.funcmap_start as u64 + func.code_id as u64 * 8;
        self.code.write_u8(base + FUNC_OFFSET_STACK_SIZE, func.stack_size);
        self.code.write_u8(base + FUNC_OFFSET_PARAM_SIZE, func.param_size);
        self.code.write_u16(base + FUNC_OFFSET_POS, func.pos);
        self.code.write_u16(base + FUNC_OFFSET_REF_START, func.ref_start);

        // Push refs
        for int_ref in self.refs.drain(..) {
            self.code.push_u64(int_ref);
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
    pub fn replace_last_inst_with(&mut self, opcode: u8, arg: u8) {
        self.code.write_bytes(self.code.len() - 2, &[opcode, arg]);
    }

    #[inline]
    pub fn insert_inst_noarg(&mut self, opcode: u8) {
        self.code.push_bytes(&[opcode, 0]);
        self.prev_inst = [opcode, 0];
    }

    #[inline]
    pub fn insert_inst(&mut self, opcode: u8, arg: u8) {
        self.code.push_bytes(&[opcode, arg]);
        self.prev_inst = [opcode, arg];
    }

    #[inline]
    pub fn insert_inst_ref(&mut self, opcode: u8, arg: impl ToRef) {
        let arg = self.new_ref(arg);
        // TODO: Add support for values above u8
        self.insert_inst(opcode, arg as u8);
        self.prev_inst = [opcode, arg as u8];
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
        let func_pos = self.current_func().pos as u64;
        self.code.write_bytes(index, &[opcode::JUMP, (self.code.len() - func_pos) as u8]);
    }

    pub fn insert_jump_if_false_inst(&mut self, jump: Jump) {
        let index = jump.0;
        let func_pos = self.current_func().pos as u64;
        self.code.write_bytes(index, &[opcode::JUMP_IF_FALSE, (self.code.len() - func_pos) as u8]);
    }

    pub fn insert_jump_if_true_inst(&mut self, jump: Jump) {
        let index = jump.0;
        let func_pos = self.current_func().pos as u64;
        self.code.write_bytes(index, &[opcode::JUMP_IF_TRUE, (self.code.len() - func_pos) as u8]);
    }

    pub fn build(self) -> BytecodeStream<W> {
        self.code
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use super::*;

    #[test]
    fn bytecode_write() {
        let bytecode: Vec<u8> = Vec::new();
        let bytecode = Cursor::new(bytecode);
        let mut bytecode = BytecodeStream::new(bytecode);
        bytecode.push_header();
        let base = bytecode.len();
        bytecode.push_u64(123123123123);
        bytecode.push_u64(9);
        bytecode.push_u64(789789789789);
        bytecode.write_u64(base + 8, 456456456456);

        assert_eq!(bytecode.read_u64(base), 123123123123);
        assert_eq!(bytecode.read_u64(bytecode.len() - 16), 456456456456);
    }
}
