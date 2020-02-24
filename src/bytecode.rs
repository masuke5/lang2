use std::collections::{HashMap, LinkedList, hash_map::Entry};
use std::mem;
use std::ptr;
use std::str;

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
    pub const CALL_EXTERN: u8 = 0x23;
    pub const TINY_INT: u8 = 0x24;
    pub const WRAP: u8 = 0x25;
    pub const UNWRAP: u8 = 0x26;
    pub const CONST_OFFSET: u8 = 0x27;
    pub const CALL_POS: u8 = 0x28;
    pub const CALL_EXTERN_POS: u8 = 0x29;
    pub const LOAD_HEAP: u8 = 0x2a;
    pub const LOAD_HEAP_TRACE: u8 = 0x2b;

    pub const END: u8 = 0x50;
}

#[inline]
pub fn opcode_name(opcode: u8) -> &'static str {
    match opcode {
        opcode::NOP => "NOP",
        opcode::INT => "INT",
        opcode::TINY_INT => "TINY_INT",
        opcode::STRING => "STRING",
        opcode::TRUE => "TRUE",
        opcode::FALSE => "FALSE",
        opcode::NULL => "NULL",
        opcode::POINTER => "POINTER",
        opcode::DEREFERENCE => "DEREFERENCE",
        opcode::NEGATIVE => "NEGATIVE",
        opcode::COPY => "COPY",
        opcode::OFFSET => "OFFSET",
        opcode::CONST_OFFSET => "CONST_OFFSET",
        opcode::DUPLICATE => "DUPLICATE",
        opcode::LOAD_REF => "LOAD_REF",
        opcode::LOAD_COPY => "LOAD_COPY",
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
pub const POS_FUNC_COUNT: usize = 4;
pub const POS_STRING_COUNT: usize = 5;
pub const POS_MODULE_COUNT: usize = 13;
pub const POS_FUNC_MAP_START: usize = 6;
pub const POS_STRING_MAP_START: usize = 8;
pub const POS_MODULE_MAP_START: usize = 10;

pub const FUNC_OFFSET_STACK_IN_HEAP_SIZE: usize = 0;
pub const FUNC_OFFSET_STACK_SIZE: usize = 1;
pub const FUNC_OFFSET_PARAM_SIZE: usize = 2;
pub const FUNC_OFFSET_POS: usize = 4;
pub const FUNC_OFFSET_REF_START: usize = 6;

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

    pub fn push_header(&mut self) {
        if self.len() > 0 {
            panic!("pushed the header already");
        }

        self.push_bytes(&HEADER);
        self.push_bytes(&[
            0, // the number of function
            0, // the number of string
            0, // funcmap_start (2byte)
            0, 0, // string_map_start (2byte)
            0, 0, // the number of module
            0, 0, 0, 0, 0,
        ]);
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

    pub fn read_bytes(&self, pos: usize, bytes: &mut [u8]) {
        unsafe {
            let src = self.bytes.as_ptr().add(pos);
            let dst = bytes.as_mut_ptr();
            ptr::copy_nonoverlapping(src, dst, bytes.len());
        }
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

    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.bytes.reserve(self.bytes.len() + bytes.len());
        unsafe {
            let dst = self.bytes.as_mut_ptr().add(self.bytes.len());
            let src = bytes.as_ptr();
            ptr::copy_nonoverlapping(src, dst, bytes.len());

            self.bytes.set_len(self.bytes.len() + bytes.len());
        }
    }

    pub fn write_bytes(&mut self, pos: usize, bytes: &[u8]) {
        unsafe {
            let dst = self.bytes.as_mut_ptr().add(pos);
            let src = bytes.as_ptr();
            ptr::copy_nonoverlapping(src, dst, bytes.len());
        }
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
        ref_start: usize,
        string_map_start: usize,
    ) {
        print!("{} ", opcode_name(opcode));

        match opcode {
            opcode::NOP => println!(),
            opcode::INT => {
                let value = self.read_i64(ref_start + arg as usize * 8);
                println!("{} ({:x})", arg, value);
            }
            opcode::TINY_INT => {
                println!("{}", arg);
            }
            opcode::STRING => {
                let string_id = arg as usize;
                // Read string location from string map
                let loc = self.read_u16(string_map_start + string_id * 2) as usize;
                // Read string length
                let len = self.read_u64(loc) as usize;

                // Read string bytes
                let mut buf = vec![0; len];
                self.read_bytes(loc + 8, &mut buf[..]);

                // Convert to string
                let raw = str::from_utf8(&buf).unwrap();

                let escaped_string = utils::escape_string(&raw);
                println!("{} (\"{}\") ({})", string_id, escaped_string, loc);
            }
            opcode::TRUE => println!(),
            opcode::FALSE => println!(),
            opcode::NULL => println!(),
            opcode::POINTER => println!(),
            opcode::DEREFERENCE => println!(),
            opcode::NEGATIVE => println!(),
            opcode::COPY => println!("size={}", arg),
            opcode::OFFSET => println!(),
            opcode::CONST_OFFSET => println!("{}", arg),
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

        // String map
        let string_map_start = self.read_u16(POS_STRING_MAP_START) as usize;
        let string_count = self.read_u8(POS_STRING_COUNT) as usize;

        let count_len = format!("{}", string_count).len();
        for i in 0..string_count {
            let loc = self.read_u16(string_map_start + i * 2) as usize;
            print!("{:<width$}  ", loc, width = index_len);

            // Read the string length
            let len = self.read_u64(loc) as usize;

            // Read the string bytes
            let mut buf = vec![0; len];
            self.read_bytes(loc + 8, &mut buf[..]);

            let raw = str::from_utf8(&buf).unwrap();

            println!(
                "{:<width$} \"{}\"",
                i,
                utils::escape_string(raw),
                width = count_len
            );
        }

        // Module map
        let module_map_start = self.read_u16(POS_MODULE_MAP_START) as usize;
        let module_count = self.read_u8(POS_MODULE_COUNT) as usize;

        for i in 0..module_count {
            let loc = self.read_u16(module_map_start + i * 2) as usize;
            print!("{:<width$}  ", loc, width = index_len);

            // Read the string length
            let len = self.read_u16(loc) as usize;

            // Read the string bytes
            let mut buf = vec![0; len];
            self.read_bytes(loc + 2, &mut buf[..]);

            let raw = str::from_utf8(&buf).unwrap();

            println!(
                "{:<width$} import {}",
                i,
                utils::escape_string(raw),
                width = count_len
            );
        }

        // Function map
        let func_map_start = self.read_u16(POS_FUNC_MAP_START) as usize;
        let func_count = self.read_u8(POS_FUNC_COUNT) as usize;

        type Func = (u16, u8, u8, usize, usize); // id, stack_size, param_size, pos, ref_start
        let mut functions: Vec<Func> = Vec::new();

        for j in 0..func_count {
            let loc = func_map_start + j * 8;
            let func_id = j as u16;
            let stack_in_heap_size = self.read_u8(loc + FUNC_OFFSET_STACK_IN_HEAP_SIZE);
            let stack_size = self.read_u8(loc + FUNC_OFFSET_STACK_SIZE);
            let param_size = self.read_u8(loc + FUNC_OFFSET_PARAM_SIZE);
            let pos = self.read_u16(loc + FUNC_OFFSET_POS) as usize;
            let ref_start = self.read_u16(loc + FUNC_OFFSET_REF_START) as usize;

            println!("{:<width$}", loc, width = index_len);
            println!("  id: {}", func_id);
            println!("  stack_in_heap_size: {}", stack_in_heap_size);
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

                self.dump_inst(inst[0], inst[1], pos + i * 2, ref_start, string_map_start);

                i += 1;
                self.read_bytes(pos + i * 2, &mut inst);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: Id,
    pub code_id: u16,
    pub stack_in_heap_size: u8,
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
            stack_in_heap_size: 0,
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
                if mem::size_of::<$ty>() > 8 {
                    panic!("too big value");
                }

                unsafe { mem::transmute(self.to_le()) }
            }
        }
    };
}

pub trait ToRef {
    fn convert(self) -> u64;
}

#[derive(Debug)]
pub struct InstList {
    pub insts: LinkedList<[u8; 2]>,
    refs: Vec<u64>,
}

impl InstList {
    pub fn new() -> Self {
        InstList {
            insts: LinkedList::new(),
            refs: Vec::new(),
        }
    }

    pub fn prev_inst(&self) -> [u8; 2] {
        *self.insts.back().unwrap()
    }

    // Insert an instruction

    #[inline]
    pub fn append(&mut self, mut insts: InstList) {
        // Update ref id and label ids
        for [opcode, arg] in &mut insts.insts {
            match *opcode {
                opcode::INT | opcode::DUPLICATE => {
                    *arg += self.refs.len() as u8;
                }
                _ => {}
            }
        }

        self.insts.append(&mut insts.insts);
        self.refs.append(&mut insts.refs);
    }

    #[inline]
    #[allow(dead_code)]
    pub fn prepend(&mut self, mut insts: InstList) {
        // Update ref id and label ids
        for [opcode, arg] in &mut insts.insts {
            match *opcode {
                opcode::INT | opcode::DUPLICATE => {
                    *arg += self.refs.len() as u8;
                }
                _ => {}
            }
        }

        mem::swap(&mut self.insts, &mut insts.insts);
        self.insts.append(&mut insts.insts);
        self.refs.append(&mut insts.refs);
    }

    #[inline]
    pub fn replace_last_inst_with(&mut self, opcode: u8, arg: u8) {
        let last = self.insts.back_mut().unwrap();
        *last = [opcode, arg];
    }

    #[inline]
    pub fn push_inst(&mut self, opcode: u8, arg: u8) {
        self.insts.push_back([opcode, arg]);
    }

    #[inline]
    pub fn push_jump(&mut self, opcode: u8, label: u8) {
        let opcode = opcode | 0b1000_0000;
        self.push_inst(opcode, label);
    }

    #[inline]
    pub fn push_inst_noarg(&mut self, opcode: u8) {
        self.push_inst(opcode, 0);
    }

    #[inline]
    pub fn push_inst_ref(&mut self, opcode: u8, arg: impl ToRef) {
        let arg = self.new_ref(arg);
        // TODO: Add support for values above u8
        self.push_inst(opcode, arg as u8);
    }

    pub fn new_ref(&mut self, value: impl ToRef) -> usize {
        self.refs.push(value.convert());
        self.refs.len() - 1
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.insts.len()
    }
}

impl_toref!(u64);
impl_toref!(i64);
impl_toref!(usize);
impl_toref!(isize);

pub struct BytecodeBuilder {
    pub code: Bytecode,

    functions: HashMap<Id, Function>,
    func_count: usize,
    modules: Vec<String>,
}

impl BytecodeBuilder {
    pub fn new() -> Self {
        let mut code = Bytecode::new();
        code.push_header();

        BytecodeBuilder {
            code,
            functions: HashMap::new(),
            func_count: 0,
            modules: Vec::new(),
        }
    }

    fn write_strings(&mut self, strings: &[String]) {
        if strings.len() > std::u16::MAX as usize {
            panic!("too many strings");
        }

        self.code.write_u8(POS_STRING_COUNT, strings.len() as u8);

        let string_map_start = self.code.len();
        self.code
            .write_u16(POS_STRING_MAP_START, string_map_start as u16);

        type StringId = u16;

        // Reserve for string map
        self.code
            .reserve(strings.len() * mem::size_of::<StringId>());

        // Write strings
        self.code.align(8);

        for (i, string) in strings.iter().enumerate() {
            // Write the string location to string map
            self.code.write_u16(
                string_map_start + i * mem::size_of::<StringId>(),
                self.code.len() as u16,
            );

            // Write the string length and bytes
            self.code.push_u64(string.len() as u64);
            self.code.push_bytes(string.as_bytes());

            self.code.align(8);
        }
    }

    fn write_modules(&mut self) {
        // Limit to 4 bits for CALL_EXTERN
        if self.modules.len() > 0b1111 {
            panic!("too many modules");
        }

        self.code
            .write_u8(POS_MODULE_COUNT, self.modules.len() as u8);

        self.code.align(8);

        let module_map_start = self.code.len();
        self.code
            .write_u16(POS_MODULE_MAP_START, module_map_start as u16);

        type ModuleId = u16;

        self.code
            .reserve(self.modules.len() * mem::size_of::<ModuleId>());
        self.code.align(8);

        for (id, name) in self.modules.iter().enumerate() {
            self.code.write_u16(
                module_map_start + id * mem::size_of::<ModuleId>(),
                self.code.len() as u16,
            );

            self.code.push_u16(name.len() as u16);
            self.code.push_bytes(name.as_bytes());

            self.code.align(8);
        }
    }

    fn write_funcs(&mut self) {
        self.code.align(8);

        // Write function map start
        self.code
            .write_u16(POS_FUNC_MAP_START, self.code.len() as u16);
        // Write the number of functions
        self.code.write_u8(POS_FUNC_COUNT, self.func_count as u8);

        let mut functions: Vec<Function> = self.functions.values().cloned().collect();
        functions.sort_by_key(|f| f.code_id);

        for func in &functions {
            let base = self.code.len();
            self.code.reserve(8);
            self.code.write_u8(
                base + FUNC_OFFSET_STACK_IN_HEAP_SIZE,
                func.stack_in_heap_size,
            );
            self.code
                .write_u8(base + FUNC_OFFSET_STACK_SIZE, func.stack_size);
            self.code
                .write_u8(base + FUNC_OFFSET_PARAM_SIZE, func.param_size);
            self.code.write_u16(base + FUNC_OFFSET_POS, func.pos);
            self.code
                .write_u16(base + FUNC_OFFSET_REF_START, func.ref_start);
        }
    }

    // Function

    pub fn insert_function_header(&mut self, mut func: Function) {
        if self.func_count > std::u16::MAX as usize {
            panic!("too many functions");
        }

        if let Entry::Vacant(entry) = self.functions.entry(func.name) {
            // Set ID
            func.code_id = self.func_count as u16;
            self.func_count += 1;

            entry.insert(func);
        }
    }

    pub fn push_function_body(&mut self, func_id: Id, mut insts: InstList) {
        insts.push_inst_noarg(opcode::END);

        let func = self
            .functions
            .get_mut(&func_id)
            .expect("the function header must be added");
        func.pos = self.code.len() as u16;

        // Write instruction bytes
        for inst in &insts.insts {
            self.code.push_bytes(inst);
        }

        self.code.align(8);

        func.ref_start = self.code.len() as u16;

        // Push refs
        for int_ref in insts.refs.drain(..) {
            self.code.push_u64(int_ref);
        }
    }

    pub fn get_function(&self, id: Id) -> Option<&Function> {
        self.functions.get(&id)
    }

    pub fn get_function_mut(&mut self, id: Id) -> Option<&mut Function> {
        self.functions.get_mut(&id)
    }

    pub fn push_module(&mut self, name: &str) -> u16 {
        self.modules.push(name.to_string());
        self.modules.len() as u16 - 1
    }

    pub fn build(mut self, strings: &[String]) -> Bytecode {
        self.write_strings(strings);
        self.write_modules();
        self.write_funcs();

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
        insts.push_inst(opcode::TINY_INT, 30);
        insts.push_inst_ref(opcode::INT, 505050 as u64);

        let mut insts2 = InstList::new();
        insts2.push_inst_noarg(opcode::BINOP_ADD);
        insts2.push_inst_ref(opcode::INT, 606060 as u64);
        insts2.push_inst_noarg(opcode::BINOP_MUL);

        insts.append(insts2);

        let expected: LinkedList<[u8; 2]> = vec![
            [opcode::TINY_INT, 30],
            [opcode::INT, 0],
            [opcode::BINOP_ADD, 0],
            [opcode::INT, 1],
            [opcode::BINOP_MUL, 0],
        ]
        .into_iter()
        .collect();

        assert_eq!(expected, insts.insts);
        assert_eq!(vec![505050, 606060], insts.refs);
    }

    #[test]
    fn instlist_prepend() {
        let mut insts = InstList::new();
        insts.push_inst_noarg(opcode::BINOP_ADD);
        insts.push_inst_ref(opcode::INT, 606060 as u64);
        insts.push_inst_noarg(opcode::BINOP_MUL);

        let mut insts2 = InstList::new();
        insts2.push_inst(opcode::TINY_INT, 30);
        insts2.push_inst_ref(opcode::INT, 505050 as u64);

        insts.prepend(insts2);

        let expected: LinkedList<[u8; 2]> = vec![
            [opcode::TINY_INT, 30],
            [opcode::INT, 1],
            [opcode::BINOP_ADD, 0],
            [opcode::INT, 0],
            [opcode::BINOP_MUL, 0],
        ]
        .into_iter()
        .collect();

        assert_eq!(expected, insts.insts);
        assert_eq!(vec![606060, 505050], insts.refs);
    }
}
