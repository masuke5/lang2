use crate::ast::SymbolPath;
use crate::id::{Id, IdMap};
use crate::utils;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use opcode::Opcode;
use rustc_hash::FxHashMap;
use std::fmt;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::str;
use std::sync::atomic::{AtomicU32, Ordering};
use thiserror::Error;

#[allow(dead_code)]
pub mod opcode {
    #[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
    pub struct Opcode(u8);

    impl Opcode {
        #[inline]
        pub fn from(code: u8) -> Self {
            Self(code)
        }

        #[inline]
        pub fn code(self) -> u8 {
            self.0
        }

        #[inline]
        pub fn is_extended(self) -> bool {
            (self.0 >> 7) != 0
        }

        #[inline]
        pub fn extended(self) -> Self {
            Self(self.0 | (1 << 7))
        }
    }

    pub const NOP: Opcode = Opcode(0x0);
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
    pub const MODULE: Opcode = Opcode(0x3b);
    pub const FUNC: Opcode = Opcode(0x3c);
    pub const SELF_FUNC: Opcode = Opcode(0x3d);
    pub const CALL_REF: Opcode = Opcode(0x3e);
    pub const CALL_NATIVE: Opcode = Opcode(0x3f);
    pub const RETURN: Opcode = Opcode(0x40);
    pub const JUMP: Opcode = Opcode(0x41);
    pub const JUMP_IF_TRUE: Opcode = Opcode(0x42);
    pub const JUMP_IF_FALSE: Opcode = Opcode(0x43);
    pub const END: Opcode = Opcode(0x44);

    impl Opcode {
        pub fn name(self) -> &'static str {
            match self {
                NOP => "NOP",
                INT => "INT",
                TINY_INT => "TINY_INT",
                FLOAT => "FLOAT",
                STRING => "STRING",
                TRUE => "TRUE",
                FALSE => "FALSE",
                NULL => "NULL",
                DUP => "DUP",
                POP => "POP",
                ALLOC => "ALLOC",
                LOAD_LV => "LOAD_LV",
                LOAD_EV => "LOAD_EV",
                LOAD_EV_OUTER => "LOAD_EV_OUTER",
                LOAD_REF_LV => "LOAD_REF_LV",
                LOAD_REF_EV => "LOAD_REF_EV",
                LOAD_REF_EV_OUTER => "LOAD_REF_EV_OUTER",
                STORE_LV => "STORE_LV",
                STORE_EV => "STORE_EV",
                STORE_EV_OUTER => "STORE_EV_OUTER",
                STORE_REF => "STORE_REF",
                DEREF => "DEREF",
                OFFSET => "OFFSET",
                OFFSET_CONST => "OFFSET_CONST",
                INEG => "INEG",
                FNEG => "FNEG",
                NOT => "NOT",
                BINOP_IADD => "BINOP_IADD",
                BINOP_ISUB => "BINOP_ISUB",
                BINOP_IMUL => "BINOP_IMUL",
                BINOP_IDIV => "BINOP_IDIV",
                BINOP_IMOD => "BINOP_IMOD",
                BINOP_FADD => "BINOP_FADD",
                BINOP_FSUB => "BINOP_FSUB",
                BINOP_FMUL => "BINOP_FMUL",
                BINOP_FDIV => "BINOP_FDIV",
                BINOP_ILT => "BINOP_ILT",
                BINOP_ILE => "BINOP_ILE",
                BINOP_IGT => "BINOP_IGT",
                BINOP_IGE => "BINOP_IGE",
                BINOP_IEQ => "BINOP_IEQ",
                BINOP_INE => "BINOP_INE",
                BINOP_FLT => "BINOP_FLT",
                BINOP_FLE => "BINOP_FLE",
                BINOP_FGT => "BINOP_FGT",
                BINOP_FGE => "BINOP_FGE",
                BINOP_FEQ => "BINOP_FEQ",
                BINOP_FNE => "BINOP_FNE",
                BINOP_REF_EQ => "BINOP_REF_EQ",
                BINOP_REF_NE => "BINOP_REF_NE",
                BINOP_LSHL => "BINOP_LSHL",
                BINOP_LSHR => "BINOP_LSHR",
                BINOP_ASHL => "BINOP_ASHL",
                BINOP_ASHR => "BINOP_ASHR",
                BINOP_AND => "BINOP_AND",
                BINOP_OR => "BINOP_OR",
                BINOP_XOR => "BINOP_XOR",
                MODULE => "MODULE",
                FUNC => "FUNC",
                SELF_FUNC => "SELF_FUNC",
                CALL_REF => "CALL_REF",
                CALL_NATIVE => "CALL_NATIVE",
                RETURN => "RETURN",
                JUMP => "JUMP",
                JUMP_IF_TRUE => "JUMP_IF_TRUE",
                JUMP_IF_FALSE => "JUMP_IF_FALSE",
                END => "END",
                _ => "UNKNOWN",
            }
        }
    }
}

// 24-bit
const MAX_ARG: u32 = 0b1111_1111_1111_1111_1111_1111;
const MAX_ARG_IMAX: i32 = 0b111_1111_1111_1111_1111_1111;
const MAX_ARG_IMIN: i32 = -0b1000_0000_0000_0000_0000_0000;

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
    Function(SymbolPath, Id),
    Normal(Opcode, u32),
    Jump(Opcode, Label),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub escaped_stack_size: u32,
    pub stack_size: u32,
    labels: FxHashMap<Label, usize>,
    param_count: u32,
    insts: Vec<Inst>,
}

impl Function {
    pub fn new(param_count: u32) -> Self {
        Self {
            escaped_stack_size: 0,
            stack_size: 0,
            param_count,
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

#[derive(Debug, Clone)]
pub struct BytecodeBuilder {
    module_path: SymbolPath,
    imported_modules: Vec<SymbolPath>,
    integers: Vec<u64>,
    floats: Vec<f64>,
    strings: Vec<String>,
    functions: FxHashMap<Id, Function>,
}

impl BytecodeBuilder {
    pub fn new(module_path: &SymbolPath) -> Self {
        Self {
            module_path: module_path.clone(),
            imported_modules: Vec::new(),
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

    pub fn define_function(&mut self, name: Id, func: Function) -> Result<(), BuilderError> {
        if self.functions.len() >= MAX_ARG as usize {
            return Err(BuilderError::TooMany("too many functions", MAX_ARG as u64));
        }
        if func.param_count >= MAX_ARG {
            return Err(BuilderError::TooMany("too many parameters", MAX_ARG as u64));
        }

        self.functions.insert(name, func);
        Ok(())
    }

    pub fn import_module(&mut self, path: &SymbolPath) -> Result<(), BuilderError> {
        if self.imported_modules.len() >= MAX_ARG as usize {
            return Err(BuilderError::TooMany(
                "too many imported modules",
                MAX_ARG as u64,
            ));
        }

        self.imported_modules.push(path.clone());
        Ok(())
    }

    pub fn get_func(&self, name: Id) -> &Function {
        self.functions.get(&name).unwrap()
    }

    pub fn get_func_mut(&mut self, name: Id) -> &mut Function {
        self.functions.get_mut(&name).unwrap()
    }

    fn allocate_ids_for_functions(&self) -> FxHashMap<Id, u32> {
        let mut function_ids = FxHashMap::default();
        let mut next_id = 0;

        for (name, _) in &self.functions {
            function_ids.insert(*name, next_id);
            next_id += 1;
        }

        function_ids
    }

    fn build(mut self, function_ids: &FxHashMap<SymbolPath, FxHashMap<Id, u32>>) -> Vec<u8> {
        // Insert END for all functions
        for func in self.functions.values_mut() {
            func.push_noarg(opcode::END);
        }

        // Allocate IDs for modules
        let modules = {
            let mut next_id = 0;
            let mut modules = FxHashMap::default();
            for module in &self.imported_modules {
                modules.insert(module.clone(), next_id);
                next_id += 1;
            }
            modules
        };

        let module_names = self
            .imported_modules
            .iter()
            .map(|path| (format!("{}", path), modules[path] as usize))
            .collect();

        // Create functions
        let mut functions = Vec::with_capacity(self.functions.len());
        for (name, function) in &self.functions {
            functions.push(BytecodeFunction {
                id: function_ids[&self.module_path][&name] as usize,
                name: IdMap::name(*name),
                stack_size: function.stack_size as usize,
                escaped_stack_size: function.escaped_stack_size as usize,
                param_count: function.param_count as usize,
                pos: 0,
            });
        }

        let mut function_insts = FxHashMap::default();

        for (name, function) in self.functions {
            fn push_u32(insts: &mut Vec<(opcode::Opcode, u8)>, op: opcode::Opcode, arg: u32) {
                assert!(arg <= MAX_ARG);

                if arg > 0xff {
                    insts.push((op.extended(), (arg >> 16) as u8));
                    insts.push((opcode::Opcode::from((arg >> 8) as u8), arg as u8));
                } else {
                    insts.push((op, arg as u8));
                }
            }

            fn push_i32(insts: &mut Vec<(opcode::Opcode, u8)>, op: opcode::Opcode, arg: i32) {
                fn i2u(n: i8) -> u8 {
                    u8::from_le_bytes(n.to_le_bytes())
                }

                assert!(arg >= MAX_ARG_IMIN && arg <= MAX_ARG_IMAX);

                if arg < -0x80 && arg > 0x7f {
                    let raw = u32::from_le_bytes(arg.to_le_bytes());
                    insts.push((op.extended(), i2u((arg >> 16) as i8)));
                    insts.push((opcode::Opcode::from((raw >> 8) as u8), raw as u8));
                } else {
                    insts.push((op, i2u(arg as i8)));
                }
            }

            let mut insts = Vec::with_capacity(function.insts.len());
            for (i, inst) in function.insts.into_iter().enumerate() {
                match inst {
                    Inst::Function(module_path, name) => {
                        if module_path != self.module_path {
                            push_u32(&mut insts, opcode::MODULE, modules[&module_path]);
                            push_u32(&mut insts, opcode::FUNC, function_ids[&module_path][&name])
                        } else {
                            push_u32(
                                &mut insts,
                                opcode::SELF_FUNC,
                                function_ids[&module_path][&name],
                            )
                        }
                    }
                    Inst::Jump(op, label) => {
                        push_i32(
                            &mut insts,
                            op,
                            (i as i32 - function.labels[&label] as i32) / 2,
                        );
                    }
                    Inst::Normal(op, arg) => {
                        push_u32(&mut insts, op, arg);
                    }
                }
            }

            function_insts.insert(function_ids[&self.module_path][&name], insts);
        }

        let module = BytecodeModule {
            path: format!("{}", self.module_path),
            modules: module_names,
            functions,
            integers: self.integers,
            floats: self.floats,
            strings: self.strings,
        };

        module.to_bytes(function_insts)
    }
}

pub fn build_bytecode(
    builders_per_module: FxHashMap<SymbolPath, BytecodeBuilder>,
) -> FxHashMap<String, Vec<u8>> {
    // Allocate IDs for functions in each modules
    let mut function_ids = FxHashMap::default();
    for (module_path, builder) in &builders_per_module {
        function_ids.insert(module_path.clone(), builder.allocate_ids_for_functions());
    }

    builders_per_module
        .into_iter()
        .map(|(path, builder)| (format!("{}", path), builder.build(&function_ids)))
        .collect()
}

#[derive(Debug, Clone)]
pub struct BytecodeFunction {
    pub id: usize,
    pub name: String,
    pub stack_size: usize,
    pub escaped_stack_size: usize,
    pub param_count: usize,
    pub pos: usize,
}

#[derive(Debug, Clone)]
pub struct BytecodeModule {
    pub path: String,
    pub modules: Vec<(String, usize)>,
    pub functions: Vec<BytecodeFunction>,
    pub integers: Vec<u64>,
    pub floats: Vec<f64>,
    pub strings: Vec<String>,
}

pub const HEADER: &[u8; 4] = b"LB02";

impl BytecodeModule {
    pub fn from_bytes(bytes: Vec<u8>, path: &str) -> Option<Self> {
        type LE = LittleEndian;

        fn align(bytes: &mut Cursor<Vec<u8>>) {
            bytes.set_position(bytes.position() + (8 - bytes.position() % 8));
        }

        let mut bc = Cursor::new(bytes);

        // Check header
        let mut buf = [0u8; 4];
        bc.read_exact(&mut buf).ok()?;
        if &buf != HEADER {
            return None;
        }

        align(&mut bc);

        // Read modules
        let modules_count = bc.read_u64::<LE>().ok()?;
        let mut modules = Vec::with_capacity(modules_count as usize);
        for _ in 0..modules_count {
            let id = bc.read_u32::<LE>().ok()?;

            // Read module path
            let len = bc.read_u32::<LE>().ok()? as usize;
            let mut buf = vec![0; len];
            bc.read_exact(&mut buf).ok()?;
            let path = str::from_utf8(&buf).ok()?;
            align(&mut bc);

            modules.push((path.to_string(), id as usize));
        }

        // Read functions
        let func_count = bc.read_u64::<LE>().ok()?;
        let mut functions = Vec::with_capacity(func_count as usize);
        for _ in 0..func_count {
            let id = bc.read_u32::<LE>().ok()?;

            // Read function name
            let len = bc.read_u32::<LE>().ok()? as usize;
            let mut buf = vec![0; len];
            bc.read_exact(&mut buf).ok()?;
            let name = str::from_utf8(&buf).ok()?.to_string();
            align(&mut bc);

            let stack_size = bc.read_u32::<LE>().ok()?;
            let escaped_stack_size = bc.read_u32::<LE>().ok()?;
            let param_count = bc.read_u32::<LE>().ok()?;
            let pos = bc.read_u64::<LE>().ok()?;
            align(&mut bc);

            let func = BytecodeFunction {
                id: id as usize,
                name,
                stack_size: stack_size as usize,
                escaped_stack_size: escaped_stack_size as usize,
                param_count: param_count as usize,
                pos: pos as usize,
            };
            functions.push(func);
        }

        // Read integers
        let integer_count = bc.read_u32::<LE>().ok()?;
        align(&mut bc);
        let mut integers = Vec::with_capacity(integer_count as usize);
        for _ in 0..integer_count {
            let n = bc.read_u64::<LE>().ok()?;
            integers.push(n);
        }

        // Read floats
        let float_count = bc.read_u32::<LE>().ok()?;
        align(&mut bc);
        let mut floats = Vec::with_capacity(float_count as usize);
        for _ in 0..float_count {
            let n = bc.read_f64::<LE>().ok()?;
            floats.push(n);
        }

        // Read strings
        let string_count = bc.read_u32::<LE>().ok()?;
        align(&mut bc);
        let mut strings = Vec::with_capacity(string_count as usize);
        for _ in 0..string_count {
            let len = bc.read_u64::<LE>().ok()? as usize;
            let mut buf = vec![0; len];
            bc.read_exact(&mut buf).ok()?;
            let s = str::from_utf8(&buf).ok()?.to_string();
            align(&mut bc);

            strings.push(s);
        }

        Some(Self {
            path: path.to_string(),
            modules,
            functions,
            integers,
            floats,
            strings,
        })
    }

    fn to_bytes(&self, insts: FxHashMap<u32, Vec<(opcode::Opcode, u8)>>) -> Vec<u8> {
        type LE = LittleEndian;

        fn align(bytes: &mut Cursor<Vec<u8>>) {
            for _ in 0..8 - bytes.position() % 8 {
                bytes.write_all(&[0]).unwrap();
            }
        }

        let mut bc = Cursor::new(Vec::new());
        bc.write_all(HEADER).unwrap();
        align(&mut bc);

        // Write modules
        bc.write_u64::<LE>(self.modules.len() as u64).unwrap();
        for (path, id) in &self.modules {
            bc.write_u32::<LE>(*id as u32).unwrap();
            bc.write_u32::<LE>(path.len() as u32).unwrap();
            bc.write_all(path.as_bytes()).unwrap();
            align(&mut bc);
        }

        // Write functions
        let mut func_pos = FxHashMap::<u32, u64>::default();
        bc.write_u64::<LE>(self.functions.len() as u64).unwrap();
        for func in &self.functions {
            bc.write_u32::<LE>(func.id as u32).unwrap();
            bc.write_u32::<LE>(func.name.len() as u32).unwrap();
            bc.write_all(func.name.as_bytes()).unwrap();
            align(&mut bc);

            bc.write_u32::<LE>(func.stack_size as u32).unwrap();
            bc.write_u32::<LE>(func.escaped_stack_size as u32).unwrap();
            bc.write_u32::<LE>(func.param_count as u32).unwrap();
            func_pos.insert(func.id as u32, bc.position());
            bc.write_u64::<LE>(!0).unwrap();
            align(&mut bc);
        }

        // Write integers
        bc.write_u32::<LE>(self.integers.len() as u32).unwrap();
        align(&mut bc);
        for n in &self.integers {
            bc.write_u64::<LE>(*n).unwrap();
        }

        // Write floats
        bc.write_u32::<LE>(self.floats.len() as u32).unwrap();
        align(&mut bc);
        for n in &self.floats {
            bc.write_f64::<LE>(*n).unwrap();
        }

        // Write strings
        bc.write_u32::<LE>(self.strings.len() as u32).unwrap();
        align(&mut bc);
        for s in &self.strings {
            bc.write_u64::<LE>(s.len() as u64).unwrap();
            bc.write_all(s.as_bytes()).unwrap();
            align(&mut bc);
        }

        // Write instructions
        for (id, insts) in insts {
            // Write pos
            let pos = bc.position();
            bc.seek(SeekFrom::Start(func_pos[&id])).unwrap();
            bc.write_u64::<LE>(pos).unwrap();

            // Write instructions
            bc.seek(SeekFrom::End(0)).unwrap();
            for (op, arg) in insts {
                bc.write_u8(op.code()).unwrap();
                bc.write_u8(arg).unwrap();
            }

            align(&mut bc);
        }

        bc.into_inner()
    }

    pub fn dump_inst(&self, bytes: &[u8], pos: usize) {
        use opcode::*;

        let op = opcode::Opcode::from(bytes[pos]);
        let arg: u32 = if op.is_extended() {
            let arg = ((bytes[pos + 1] as u32) << 16)
                | ((bytes[pos + 2] as u32) << 8)
                | bytes[pos + 3] as u32;
            print!("{:<5} {:02x}{:06x}  {}* ", pos, op.code(), arg, op.name());
            arg
        } else {
            let arg = bytes[pos + 1];
            print!(
                "{:<5} {:02x}{:02x}      {}  ",
                pos,
                op.code(),
                arg,
                op.name()
            );
            arg as u32
        };

        match op {
            NOP | TRUE | FALSE | NULL | DEREF | OFFSET | INEG | FNEG | NOT | RETURN | CALL_REF
            | CALL_NATIVE => println!(),
            op if (BINOP_IADD..=BINOP_XOR).contains(&op) => println!(),
            INT => println!("{} ({})", arg, self.integers[arg as usize]),
            FLOAT => println!("{} ({})", arg, self.floats[arg as usize]),
            STRING => println!(
                "{} (\"{}\")",
                arg,
                utils::escape_string(&self.strings[arg as usize])
            ),
            MODULE => println!("{} ({})", arg, self.modules[arg as usize].0),
            SELF_FUNC => println!("{} ({})", arg, self.functions[arg as usize].name),
            JUMP | JUMP_IF_TRUE | JUMP_IF_FALSE => {
                let arg = if op.is_extended() {
                    // Move sign
                    let sign_mask = arg & (1 << 23);
                    let arg = arg & !sign_mask | (sign_mask << 8);
                    i32::from_le_bytes(arg.to_le_bytes())
                } else {
                    i8::from_le_bytes([arg as u8]) as i32
                };
                println!("{} ({})", arg, pos as i64 + arg as i64 * 2);
            }
            _ => println!("{}", arg),
        }
    }

    pub fn dump_metadata(&self) {
        println!("Module {}", self.path);

        println!("Imported modules:");
        for (name, id) in &self.modules {
            println!("  {:<3} {}", id, name);
        }

        println!("Functions:");
        for func in &self.functions {
            println!("  {:<3} {}", func.id, func.name);
            println!("    stack_size: {}", func.stack_size);
            println!("    escaped_stack_size: {}", func.escaped_stack_size);
            println!("    param_count: {}", func.param_count);
            println!("    pos: {}", func.pos);
        }
    }

    pub fn dump_values(&self) {
        println!("Intergers:");
        for (i, n) in self.integers.iter().enumerate() {
            println!("  {:<4} {}", i, n);
        }

        println!("Floats:");
        for (i, n) in self.floats.iter().enumerate() {
            println!("  {:<4} {}", i, n);
        }

        println!("Strings:");
        for (i, s) in self.strings.iter().enumerate() {
            println!("  {:<4} \"{}\"", i, utils::escape_string(s));
        }
    }

    pub fn dump_instructions(&self, bytes: &[u8]) {
        use opcode::*;

        for func in &self.functions {
            println!("Function {} ({})", func.name, func.id);

            let mut pos = func.pos;
            while bytes[pos] != END.code() {
                self.dump_inst(bytes, pos);

                let op = Opcode::from(bytes[pos]);
                if op.is_extended() {
                    pos += 4;
                } else {
                    pos += 2;
                }
            }
        }
    }
}
