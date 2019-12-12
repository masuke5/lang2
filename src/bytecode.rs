use std::mem;
use std::str;
use std::collections::HashMap;
use std::io;
use std::io::{Read, Write, Seek, SeekFrom};

use crate::id::Id;
use crate::utils;

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

    pub const END: u8 = 0x23;
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

pub const HEADER: [u8; 4] = *b"L2BC";
pub const POS_FUNC_COUNT: u64 = 4;
pub const POS_STRING_COUNT: u64 = 5;
pub const POS_FUNC_MAP_START: u64 = 6;
pub const POS_STRING_MAP_START: u64 = 8;

pub const FUNC_OFFSET_STACK_SIZE: u64 = 2;
pub const FUNC_OFFSET_PARAM_SIZE: u64 = 3;
pub const FUNC_OFFSET_POS: u64 = 4;
pub const FUNC_OFFSET_REF_START: u64 = 6;

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
            0,
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

    pub fn read_bytes(&mut self, pos: u64, bytes: &mut [u8]) {
        self.code.seek(SeekFrom::Start(pos)).unwrap();
        self.code.read_exact(bytes).unwrap();
    }

    pub fn read_to_end(&mut self, buf: &mut Vec<u8>) {
        self.code.seek(SeekFrom::Start(0)).unwrap();
        self.code.read_to_end(buf).unwrap();
    }

    pub fn dump(&mut self) {
        let index_len = format!("{}", self.len).len();

        // String map
        let string_map_start = self.read_u16(POS_STRING_MAP_START) as u64;
        let string_count = self.read_u8(POS_STRING_COUNT) as u64;

        let count_len = format!("{}", string_count).len();
        for i in 0..string_count {
            let loc = self.read_u16(string_map_start + i * 2) as u64;
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

        // Function map
        let func_map_start = self.read_u16(POS_FUNC_MAP_START) as u64;
        let func_count = self.read_u8(POS_FUNC_COUNT) as u64;

        type Func = (u16, u8, u8, u64, u64); // id, stack_size, param_size, pos, ref_start
        let mut functions: Vec<Func> = Vec::new();

        for j in 0..func_count {
            let loc = func_map_start + j * 8;
            let func_id = self.read_u16(loc);
            let stack_size = self.read_u8(loc + FUNC_OFFSET_STACK_SIZE);
            let param_size = self.read_u8(loc + FUNC_OFFSET_PARAM_SIZE);
            let pos = self.read_u16(loc + FUNC_OFFSET_POS) as u64;
            let ref_start = self.read_u16(loc + FUNC_OFFSET_REF_START) as u64;

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
                let opcode = inst[0];
                let arg = inst[1];

                print!("{:<width$}  ", pos + i * 2, width = index_len);
                match opcode {
                    opcode::NOP => println!("NOP"),
                    opcode::INT => {
                        let value = self.read_i64(ref_start + arg as u64 * 8);
                        println!("INT {} ({})", arg, value);
                    },
                    opcode::STRING => {
                        let string_id = arg as u64;
                        // Read string location from string map
                        let loc = self.read_u16(string_map_start + string_id * 2) as u64;
                        // Read string length
                        let len = self.read_u64(loc) as usize;

                        // Read string bytes
                        let mut buf = Vec::with_capacity(len);
                        buf.resize(len, 0);
                        self.read_bytes(loc + 8, &mut buf[..]);

                        // Convert to string
                        let raw = str::from_utf8(&buf).unwrap();

                        let escaped_string = utils::escape_string(&raw);
                        println!("STRING {} (\"{}\") ({})", string_id, escaped_string, loc);
                    },
                    opcode::TRUE => println!("TRUE"),
                    opcode::FALSE => println!("FALSE"),
                    opcode::NULL => println!("NULL"),
                    opcode::POINTER => println!("POINTER"),
                    opcode::DEREFERENCE => println!("DEREFERENCE"),
                    opcode::NEGATIVE => println!("NEGATIVE"),
                    opcode::COPY => println!("COPY size={}", arg),
                    opcode::OFFSET => println!("OFFSET"),
                    opcode::DUPLICATE => println!("DUPLICATE"),
                    opcode::LOAD_REF => println!("LOAD_REF {}", i8::from_le_bytes([arg])),
                    opcode::LOAD_COPY => println!("LOAD_COPY unimplemented"),
                    opcode::STORE => println!("STORE size={}", arg),
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
                    opcode::ALLOC => println!("ALLOC size={}", arg),
                    opcode::CALL => println!("CALL {}", arg),
                    opcode::CALL_NATIVE => println!("CALL_NATIVE unimplemented"),
                    opcode::JUMP | opcode::JUMP_IF_FALSE | opcode::JUMP_IF_TRUE => {
                        let opcode = match opcode {
                            opcode::JUMP => "JUMP",
                            opcode::JUMP_IF_FALSE => "JUMP_IF_FALSE",
                            opcode::JUMP_IF_TRUE => "JUMP_IF_TRUE",
                            _ => unreachable!(),
                        };

                        println!("{} {} ({})", opcode, arg, pos + arg as u64);
                    },
                    opcode::RETURN => println!("RETURN"),
                    opcode::ZERO => println!("ZERO count={}", arg),
                    _ => println!("UNKNOWN (0x{:x})", opcode),
                };

                i += 1;
                self.read_bytes(pos + i * 2, &mut inst);
            }
        }
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
    prev_opcode: u8,
}

impl<W: Read + Write + Seek> BytecodeBuilder<W> {
    pub fn new(mut code: BytecodeStream<W>, strings: &[String]) -> Self {
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
            prev_opcode: 0,
        };
        slf.write_strings(strings);

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

        // Write function map start
        self.funcmap_start = self.code.len() as u16;
        self.code.write_u16(POS_FUNC_MAP_START, self.funcmap_start as u16);
    }

    pub fn prev_opcode(&self) -> u8 {
        self.prev_opcode
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
    pub fn insert_inst_noarg(&mut self, opcode: u8) {
        self.code.push_bytes(&[opcode, 0]);
        self.prev_opcode = opcode;
    }

    #[inline]
    pub fn insert_inst(&mut self, opcode: u8, arg: u8) {
        self.code.push_bytes(&[opcode, arg]);
        self.prev_opcode = opcode;
    }

    #[inline]
    pub fn insert_inst_ref(&mut self, opcode: u8, arg: impl ToRef) {
        let arg = self.new_ref(arg);
        // TODO: Add support for values above u8
        self.insert_inst(opcode, arg as u8);
        self.prev_opcode = opcode;
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
