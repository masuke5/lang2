use std::time::Instant;
use std::mem;
use std::mem::size_of;
use std::slice;
use std::str;
use std::ptr;

use rustc_hash::FxHashMap;

use crate::bytecode;
use crate::bytecode::{Bytecode, opcode, opcode_name};
use crate::value::{Value, Lang2String};
use crate::gc::Gc;
use crate::module::Module;
use crate::sema::ModuleBody;

pub const SELF_MODULE_ID: usize = 0x7fffffff;

const STACK_SIZE: usize = 10000;

macro_rules! pop {
    ($self:ident) => {
        {
            let v = $self.stack[$self.sp];
            $self.sp -= 1;
            v
        }
    };
}

macro_rules! push {
    ($self:ident, $value:expr) => {
        $self.sp += 1;
        $self.stack[$self.sp] = $value;
    };
}

#[derive(Debug, Clone)]
struct Function {
    stack_size: usize,
    param_size: usize,
    pos: usize,
    ref_start: usize,
}

#[derive(Debug)]
pub struct InstPerformance {
    count: u32,
    average: f32,
    total: f32,
}

impl InstPerformance {
    fn new() -> Self {
        Self {
            count: 0,
            average: 0.0,
            total: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct Performance {
    insts: FxHashMap<u8, InstPerformance>,
    current_opcode: u8,
    started_time: Instant,
    total: f32,
}

impl Performance {
    pub fn new() -> Self {
        Performance {
            insts: FxHashMap::default(),
            current_opcode: opcode::NOP,
            started_time: Instant::now(),
            total: 0.0,
        }
    }

    pub fn new_inst(&mut self, opcode: u8) {
        self.current_opcode = opcode;
        self.started_time = Instant::now();
    }

    pub fn end_inst(&mut self) {
        let now = Instant::now();
        let elapsed = now - self.started_time;
        let elapsed = elapsed.as_nanos() as f32;

        let p = self.insts.entry(self.current_opcode).or_insert(InstPerformance::new());
        let count = p.count as f32;

        p.average = 1.0 / (count + 1.0) * (count * p.average + elapsed);
        p.count += 1;
        p.total += elapsed;

        self.total += elapsed;
    }
}

pub struct VM {
    performance: Performance,
    
    // garbage collector
    gc: Gc,

    // instruction pointer
    ip: usize,
    // frame pointer
    fp: usize,
    // stack pointer
    sp: usize,

    stack: [Value; STACK_SIZE],

    // global id -> all functions in the module
    functions: Vec<Vec<Function>>,
    current_func: usize,
    current_module: usize,
}

impl VM {
    pub fn new() -> Self {
        Self {
            performance: Performance::new(),
            gc: Gc::new(),
            ip: 0,
            fp: 0,
            sp: 0,
            stack: unsafe { mem::zeroed() },
            functions: Vec::new(),
            current_func: 0,
            current_module: 0,
        }
    }

    pub fn arg_loc(&self, n: usize, args_size: usize) -> usize {
        self.fp - args_size - n
    }

    pub fn get_value(&self, loc: usize) -> Value {
        self.stack[loc]
    }

    pub fn get_string(&self, loc: usize) -> &Lang2String {
        let value = self.stack[loc];
        if !value.is_heap_ptr() {
            self.panic("no heap ptr");
        }

        unsafe { &*value.as_ptr() }
    }

    fn dump_value(value: &Value, depth: usize) {
        print!("{}{:x}", "  ".repeat(depth), value.as_u64());
        if value.is_heap_ptr() {
            println!(" (HEAP)");
        } else {
            println!();
        }
    }

    fn dump_stack(&self, stop: usize) {
        let current_is_main = self.current_func == 0;
        let saved_module_id = if !current_is_main { Some(self.fp - 4) } else { None };
        let saved_func_id = if !current_is_main { Some(self.fp - 3) } else { None };
        let saved_ip = if !current_is_main { Some(self.fp - 2) } else { None };
        let saved_fp = if !current_is_main { Some(self.fp - 1) } else { None };

        println!("-------- STACK DUMP --------");
        for (i, value) in self.stack.iter().enumerate() {
            if i == self.fp {
                print!("(fp) ");
            }

            if i == self.sp {
                print!("(sp) ");
            }

            if Some(i) == saved_func_id {
                print!("[fid] ");
            }

            if Some(i) == saved_module_id {
                print!("[mid] ");
            }

            if Some(i) == saved_ip {
                print!("[ip] ");
            }

            if Some(i) == saved_fp {
                print!("[fp] ");
            }

            if i > stop {
                println!("[{}]", i);
                break;
            }


            Self::dump_value(value, 0);
        }
        println!("-------- END DUMP ----------");
    }

    fn read_functions(&mut self, bytecode: &Bytecode, module_id: usize) {
        assert!(module_id < self.functions.len());

        let func_map_start = bytecode.read_u16(bytecode::POS_FUNC_MAP_START) as usize;
        let func_count = bytecode.read_u8(bytecode::POS_FUNC_COUNT) as usize;

        for i in 0..func_count {
            let base = func_map_start + i * 8;
            let stack_size = bytecode.read_u8(base + bytecode::FUNC_OFFSET_STACK_SIZE) as usize;
            let param_size = bytecode.read_u8(base + bytecode::FUNC_OFFSET_PARAM_SIZE ) as usize;
            let pos = bytecode.read_u16(base + bytecode::FUNC_OFFSET_POS) as usize;
            let ref_start = bytecode.read_u16(base + bytecode::FUNC_OFFSET_REF_START) as usize;

            self.functions[module_id].push(Function {
                stack_size,
                param_size,
                pos,
                ref_start,
            });
        }
    }

    fn read_modules(&mut self, bytecode: &Bytecode, all_global_ids: &FxHashMap<String, usize>) -> Vec<usize> {
        let module_map_start = bytecode.read_u16(bytecode::POS_MODULE_MAP_START) as usize;
        let module_count = bytecode.read_u8(bytecode::POS_MODULE_COUNT) as usize;

        let mut modules = Vec::with_capacity(module_count);

        for i in 0..module_count {
            let loc = bytecode.read_u16(module_map_start + i * 2) as usize;
            let len = bytecode.read_u16(loc) as usize;

            let mut buf = Vec::with_capacity(len);
            buf.resize(len, 0);
            bytecode.read_bytes(loc + 2, &mut buf[..]);
            let name = str::from_utf8(&buf[..]).expect("invalid module name");

            let global_id = all_global_ids[name];
            modules.push(global_id);
        }

        modules
    }

    fn current_func(&self) -> &Function {
        &self.functions[self.current_module][self.current_func]
    }

    fn main_func(&self) -> &Function {
        &self.functions[0][0]
    }

    fn call(&mut self, global_module_id: usize, func_id: usize) {
        let func = &self.functions[global_module_id][func_id];

        push!(self, Value::new_u64(self.current_module as u64));
        push!(self, Value::new_u64(self.current_func as u64));
        self.current_func = func_id;
        self.current_module = global_module_id;

        push!(self, Value::new_u64(self.ip as u64));
        push!(self, Value::new_u64(self.fp as u64));

        self.ip = func.pos;

        // Allocate stack frame
        self.fp = self.sp + 1;
        self.sp += func.stack_size as usize;
    }

    fn panic(&self, message: &str) -> ! {
        eprintln!("PANIC AT RUNTIME: {}", message);
        std::process::exit(10)
    }

    #[inline]
    fn next_inst(&mut self, bytecode: &Bytecode) -> [u8; 2] {
        let mut buf = [0u8; 2];
        bytecode.read_bytes(self.ip, &mut buf);

        self.ip += 2;

        buf
    }

    #[inline]
    fn get_ref_value_i64(&mut self, bytecode: &Bytecode, ref_id: u8) -> i64 {
        let ref_start = self.current_func().ref_start;
        bytecode.read_i64(ref_start + ref_id as usize * 8)
    }

    #[inline]
    fn get_ref_value_u64(&mut self, bytecode: &Bytecode, ref_id: u8) -> u64 {
        let ref_start = self.current_func().ref_start;
        bytecode.read_u64(ref_start + ref_id as usize * 8)
    }
    
    pub fn run(&mut self, module_bodies: Vec<(String, ModuleBody)>, enable_trace: bool, enable_measure: bool) {
        #[inline]
        fn ip_after_jump_to(ip: usize, loc: u8) -> usize {
            let loc = i8::from_le_bytes([loc]) as isize;
            (ip as isize - 2 + loc * 2) as usize
        }

        // global id -> module
        let mut all_modules = Vec::new();
        for (_, body) in &module_bodies {
            let module = match body {
                ModuleBody::Normal(_) => Module::Normal,
                ModuleBody::Native(module) => module.clone(),
            };
            all_modules.push(module);
        }

        // module name -> global id
        let mut module_global_ids = FxHashMap::default();

        {
            let mut next_id = 0;
            for (name, _) in &module_bodies {
                module_global_ids.insert(name.clone(), next_id);
                next_id += 1;
            }
        }

        // global id -> bytecode
        let bytecodes: Vec<Option<Bytecode>> = module_bodies
            .into_iter()
            .map(|(_, body)| match body {
                ModuleBody::Normal(bc) => Some(bc),
                ModuleBody::Native(_) => None,
            })
            .collect();

        // string map start per module
        let mut string_map_start = Vec::with_capacity(all_modules.len());
        for bytecode in &bytecodes {
            if let Some(bytecode) = bytecode {
                let sms = bytecode.read_u16(bytecode::POS_STRING_MAP_START) as usize;
                string_map_start.push(Some(sms));
            } else {
                string_map_start.push(None);
            }
        }

        self.functions.resize(all_modules.len(), Vec::new());

        // module id -> local id -> global id
        let mut modules = Vec::with_capacity(all_modules.len());
        for bytecode in &bytecodes {
            if let Some(bytecode) = bytecode {
                let module_map = self.read_modules(bytecode, &module_global_ids);
                modules.push(Some(module_map));
            } else {
                modules.push(None);
            }
        }

        assert_eq!(all_modules.len(), module_global_ids.len());
        assert_eq!(all_modules.len(), bytecodes.len());
        assert_eq!(all_modules.len(), modules.len());

        // Function
        for (global_module_id, bytecode) in bytecodes.iter().enumerate() {
            if let Some(bytecode) = bytecode {
                self.read_functions(bytecode, global_module_id);
            }
        }

        if self.functions.is_empty() {
            panic!("the bytecode need an entrypoint");
        }

        self.current_func = 0;

        let func = self.main_func().clone();
        self.ip = func.pos;
        self.fp = 0;
        if func.stack_size > 0 {
            self.sp = func.stack_size as usize - 1;
        }

        let mut current_bytecode = bytecodes[0].as_ref().unwrap();

        loop {
            let [opcode, arg] = self.next_inst(current_bytecode);
            if opcode == opcode::END {
                break;
            }

            if cfg!(debug_assertions) && enable_trace {
                let func = self.current_func();
                print!("{}  ", self.ip);
                current_bytecode.dump_inst(
                    opcode,
                    arg,
                    self.ip - 2,
                    func.ref_start,
                    string_map_start[self.current_module].unwrap()
                );
            }

            if cfg!(debug_assertions) && enable_measure {
                self.performance.new_inst(opcode);
            }

            match opcode {
                opcode::NOP => {},
                opcode::ZERO => {
                    let count = arg as usize;

                    unsafe {
                        let dst = &mut self.stack[self.sp + 1] as *mut _;
                        ptr::write_bytes(dst, 0, count);
                    }

                    self.sp += count;
                },
                opcode::INT => {
                    let value = self.get_ref_value_i64(current_bytecode, arg);
                    push!(self, Value::new_i64(value));
                },
                opcode::TINY_INT => {
                    let n = i8::from_le_bytes([arg]);
                    push!(self, Value::new_i64(n as i64));
                },
                opcode::STRING => {
                    let string_map_start = string_map_start[self.current_module].unwrap();
                    let loc = current_bytecode.read_u16(string_map_start + arg as usize * 2) as usize;

                    // Read the string length
                    let len = current_bytecode.read_u64(loc) as usize;

                    let size = len + size_of::<u64>();
                    let mut region = self.gc.alloc::<u8>(size, false, &mut self.stack[..=self.sp]);

                    // Read the string bytes
                    unsafe {
                        let region = region.as_mut();

                        // Write the string length
                        let len_ptr = region.as_mut_ptr::<u64>();
                        *len_ptr = len as u64;

                        // Write the string bytes
                        let bytes_ptr = region.as_mut_ptr::<u8>().add(size_of::<u64>());
                        let mut bytes = slice::from_raw_parts_mut(bytes_ptr, len);
                        current_bytecode.read_bytes(loc + size_of::<u64>() as usize, &mut bytes);
                    }

                    unsafe {
                        push!(self, Value::new_ptr_to_heap(region.as_mut().as_ptr::<Value>()));
                    }
                },
                opcode::TRUE => {
                    push!(self, Value::new_true());
                },
                opcode::FALSE => {
                    push!(self, Value::new_false());
                },
                opcode::NULL => {
                    let nullptr = ptr::null_mut::<Value>();
                    push!(self, Value::new_ptr(nullptr));
                },
                opcode::POINTER => {
                    let value = pop!(self);
                    push!(self, Value::new_ptr(value.as_ptr::<Value>()));
                },
                opcode::DEREFERENCE => {
                    // Does nothing
                },
                opcode::NEGATIVE => {
                    let tos = &mut self.stack[self.sp];
                    *tos = Value::new_i64(-tos.as_i64());
                },
                opcode::COPY => {
                    let size = arg as usize;
                    if self.sp + size >= STACK_SIZE {
                        self.panic("stack overflow");
                    }

                    let value_ref = pop!(self);
                    // TODO: check pointer

                    unsafe {
                        let value_ref = value_ref.as_ptr();
                        let dst = &mut self.stack[self.sp + 1];
                        ptr::copy_nonoverlapping(value_ref, dst, size);
                    }

                    self.sp += size;
                },
                opcode::OFFSET => {
                    let offset = pop!(self).as_i64();
                    if offset < 0 {
                        self.panic("negative offset");
                    }

                    let ptr = self.stack[self.sp].as_ptr::<Value>();
                    let new_ptr = unsafe { ptr.add(offset as usize) };
                    self.stack[self.sp] = Value::new_ptr(new_ptr);
                },
                opcode::CONST_OFFSET => {
                    let offset = arg as usize;

                    let ptr = self.stack[self.sp].as_ptr::<Value>();
                    let new_ptr = unsafe { ptr.add(offset as usize) };
                    self.stack[self.sp] = Value::new_ptr(new_ptr);
                },
                opcode::DUPLICATE => {
                    let value = self.get_ref_value_u64(current_bytecode, arg);
                    let size = (value >> 32) as usize; // upper 32 bits
                    let count = (value as u32) as usize; // lower 32 bits

                    let ptr = &self.stack[self.sp - (size - 1)] as *const Value;

                    for i in 1..=count {
                        unsafe {
                            let dest = ptr.add(i * size) as *mut _;
                            ptr.copy_to_nonoverlapping(dest, size);
                        }
                    }

                    self.sp += size * count;
                },
                opcode::LOAD_REF => {
                    let loc = (self.fp as isize + i8::from_le_bytes([arg]) as isize) as usize;
                    if loc >= STACK_SIZE {
                        self.panic("out of bounds");
                    }

                    let value = &mut self.stack[loc];
                    push!(self, Value::new_ptr(value as *mut Value));
                },
                opcode::LOAD_COPY => {
                    let loc = i8::from_le_bytes([arg & 0b11111000]) >> 3;
                    let size = (arg & 0b00000111) as usize;

                    let loc = (self.fp as isize + loc as isize) as usize;
                    if loc >= STACK_SIZE {
                        self.panic("out of bounds");
                    }

                    unsafe {
                        let src = self.stack.as_ptr().add(loc);
                        let dst = &mut self.stack[self.sp + 1];
                        ptr::copy_nonoverlapping(src, dst, size);
                    }

                    self.sp += size;
                },
                opcode::STORE => {
                    let size = arg as usize;

                    let dst = pop!(self);
                    // TODO: check pointer

                    unsafe {
                        let dst = dst.as_ptr();
                        let src = &self.stack[self.sp - size + 1] as *const _;
                        ptr::copy_nonoverlapping(src, dst, size);
                    }

                    self.sp -= size;
                },
                opcode::WRAP => {
                    let size = arg as usize;
                    if size != 1 {
                        let mut region = self.gc.alloc::<Value>(size, true, &mut self.stack[..=self.sp]);

                        unsafe {
                            let dst = region.as_mut().as_mut_ptr::<Value>();
                            let src = &self.stack[self.sp - size + 1] as *const _;
                            ptr::copy_nonoverlapping(src, dst, size);
                        }

                        self.sp -= size;

                        let region = unsafe { region.as_mut() };
                        push!(self, Value::new_ptr_to_heap::<Value>(region.as_ptr()));
                    }
                },
                opcode::UNWRAP => {
                    let size = arg as usize;
                    if size != 1 {
                        unsafe {
                            let dst = &mut self.stack[self.sp] as *mut _;
                            let src = self.stack[self.sp].as_ptr::<Value>();
                            ptr::copy_nonoverlapping(src, dst, size);
                        }

                        self.sp += size - 1;
                    }
                },
                opcode::BINOP_ADD..=opcode::BINOP_NEQ => {
                    let result = unsafe {
                        let rhs = pop!(self).raw_i64();
                        let lhs = pop!(self).raw_i64();

                        match opcode {
                            opcode::BINOP_ADD => Value::from_raw_i64(lhs + rhs),
                            opcode::BINOP_SUB => Value::from_raw_i64(lhs - rhs),
                            opcode::BINOP_MUL => {
                                let lhs = lhs >> 1;
                                let rhs = rhs >> 1;
                                Value::from_raw_i64((lhs * rhs) << 1)
                            },
                            opcode::BINOP_DIV => {
                                Value::new_i64(lhs / rhs)
                            },
                            opcode::BINOP_MOD => Value::from_raw_i64(lhs % rhs),
                            opcode::BINOP_LT => Value::new_bool(lhs < rhs),
                            opcode::BINOP_LE => Value::new_bool(lhs <= rhs),
                            opcode::BINOP_GT => Value::new_bool(lhs > rhs),
                            opcode::BINOP_GE => Value::new_bool(lhs >= rhs),
                            opcode::BINOP_EQ => Value::new_bool(lhs == rhs),
                            opcode::BINOP_NEQ => Value::new_bool(lhs != rhs),
                            _ => panic!("binop bug"),
                        }
                    };

                    push!(self, result);
                },
                opcode::POP => {
                    self.sp -= 1;
                },
                opcode::ALLOC => {
                    let size = arg as usize;

                    let mut region = self.gc.alloc::<Value>(size, true, &mut self.stack[..=self.sp]);

                    unsafe {
                        let dst = region.as_mut().as_mut_ptr::<Value>();
                        let src = &self.stack[self.sp - size + 1] as *const _;
                        ptr::copy_nonoverlapping(src, dst, size);
                    }

                    self.sp -= size;

                    let region = unsafe { region.as_mut() };
                    push!(self, Value::new_ptr_to_heap::<Value>(region.as_ptr()));
                },
                opcode::CALL => {
                    self.call(self.current_module, arg as usize);
                },
                opcode::CALL_POS => {
                    let pos = pop!(self).as_u64();

                    let module_local_id = (pos >> 32) as usize;
                    let func_id = pos as u32 as usize;

                    if module_local_id == SELF_MODULE_ID {
                        self.call(self.current_module, func_id);
                    } else {
                        let module_global_id = modules[self.current_module].as_ref().unwrap()[module_local_id];
                        let module = &mut all_modules[module_global_id];

                        match module {
                            Module::Normal => {
                                current_bytecode = &bytecodes[module_global_id].as_ref().unwrap();
                                self.call(module_global_id, func_id);
                            },
                            Module::Native(funcs) => {
                                let (param_size, func) = &funcs[func_id];

                                let fp = self.fp;
                                self.fp = self.sp + 1;

                                func.0(self);

                                self.fp = fp;
                                self.sp -= param_size;
                            },
                        }
                    }
                },
                opcode::CALL_EXTERN => {
                    let module_local_id = ((arg & 0b11110000) >> 4) as usize;
                    let func_id = (arg & 0b00001111) as usize;

                    let module_global_id = modules[self.current_module].as_ref().unwrap()[module_local_id];
                    let module = &mut all_modules[module_global_id];

                    match module {
                        Module::Normal => {
                            current_bytecode = &bytecodes[module_global_id].as_ref().unwrap();
                            self.call(module_global_id, func_id);
                        },
                        Module::Native(funcs) => {
                            let (param_size, func) = &funcs[func_id];

                            let fp = self.fp;
                            self.fp = self.sp + 1;

                            func.0(self);

                            self.fp = fp;
                            self.sp -= param_size;
                        },
                    }
                },
                opcode::RETURN => {
                    // Restore stack frame
                    self.sp = self.fp - 1;
                    self.fp = pop!(self).as_u64() as usize;

                    self.ip = pop!(self).as_u64() as usize;
                    let prev_func = pop!(self).as_u64() as usize;
                    let prev_module =  pop!(self).as_u64() as usize;

                    // Pop arguments
                    self.sp -= self.current_func().param_size as usize;

                    self.current_func = prev_func;
                    self.current_module = prev_module;
                    current_bytecode = &bytecodes[self.current_module].as_ref().unwrap();
                },
                opcode::CALL_NATIVE => {
                    unimplemented!();
                },
                opcode::JUMP => {
                    self.ip = ip_after_jump_to(self.ip, arg);
                },
                opcode::JUMP_IF_FALSE => {
                    let cond = pop!(self);
                    if cond.is_false() {
                        self.ip = ip_after_jump_to(self.ip, arg);
                    }
                },
                opcode::JUMP_IF_TRUE => {
                    let cond = pop!(self);
                    if cond.is_true() {
                        self.ip = ip_after_jump_to(self.ip, arg);
                    }
                },
                _ => {
                    panic!("Unknown opcode (0x{:x})", opcode);
                },
            }

            if cfg!(debug_assertions) && enable_measure {
                self.performance.end_inst();
            }
        }

        if cfg!(debug_assertions) {
            let main_func_stack_size = self.main_func().stack_size as usize;
            if (main_func_stack_size != 0 || self.sp != 0) && self.sp != main_func_stack_size - 1 {
                self.dump_stack(self.sp);
                eprintln!("warning: expected sp {}, but sp is {}.", main_func_stack_size - 1, self.sp);
            }
        }

        if enable_measure {
            let total = self.performance.total;
            // Print as CSV
            for (opcode, p) in &self.performance.insts {
                let average = p.average.floor();
                eprintln!(
                    "{},{},{},{}",
                    opcode_name(*opcode),
                    p.count,
                    average,
                    p.total / total * 100.0,
                );
            }
        }
    }
}
