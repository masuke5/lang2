use std::collections::HashMap;
use std::time::Instant;
use std::ptr::NonNull;
use std::mem;
use std::mem::size_of;
use std::slice;
use std::str;
use std::ptr;

use crate::bytecode;
use crate::bytecode::{Bytecode, opcode, opcode_name};
use crate::value::{FromValue, Value, Lang2String};
use crate::gc::Gc;
use crate::module::Module;

const STACK_SIZE: usize = 10000;

macro_rules! pop {
    ($self:ident, $ty:ty) => {
        {
            let v: $ty = FromValue::from_value(mem::replace(&mut $self.stack[$self.sp], Value::Unintialized));
            $self.sp -= 1;
            v
        }
    };
    ($self:ident) => {
        {
            let v = FromValue::from_value(mem::replace(&mut $self.stack[$self.sp], Value::Unintialized));
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

#[derive(Debug)]
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
    insts: HashMap<u8, InstPerformance>,
    current_opcode: u8,
    started_time: Instant,
    total: f32,
}

impl Performance {
    pub fn new() -> Self {
        Performance {
            insts: HashMap::new(),
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

    functions: Vec<Function>,
    current_func: usize,
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
        }
    }

    pub fn arg_loc(&self, n: usize, args_size: usize) -> usize {
        self.fp - args_size - n
    }

    pub fn get_value<V: FromValue>(&self, loc: usize) -> V {
        let value = self.stack[loc].clone();
        FromValue::from_value(value)
    }

    #[allow(dead_code)]
    pub fn get_value_ref<V: FromValue>(&self, loc: usize) -> &V {
        let value = &self.stack[loc];
        FromValue::from_value_ref(value)
    }

    pub fn get_string(&self, loc: usize) -> &Lang2String {
        let value = &self.stack[loc];
        let ptr = match value {
            Value::PointerToHeap(ptr) => ptr,
            _ => panic!(),
        };

        unsafe { &*ptr.as_ref().as_ptr::<Lang2String>() }
    }

    fn dump_value(value: &Value, depth: usize) {
        print!("{}", "  ".repeat(depth));
        match value {
            Value::Int(n) => println!("int {}", n),
            Value::Bool(true) => println!("bool true"),
            Value::Bool(false) => println!("bool false"),
            Value::Ref(ptr) => {
                println!("ref");
                let value = unsafe { ptr.as_ref() };
                Self::dump_value(value, depth + 1);
            },
            Value::PointerToStack(ptr) => println!("ptr {:p}", ptr.as_ptr()),
            Value::PointerToHeap(ptr) => unsafe { println!("ptr {:p}", ptr.as_ref().as_ptr::<Value>()) },
            Value::Unintialized => println!("uninitialized"),
        }
    }

    fn dump_stack(&self, stop: usize) {
        let current_is_main = self.current_func == 0;
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

    fn read_functions(&mut self, bytecode: &Bytecode) {
        let func_map_start = bytecode.read_u16(bytecode::POS_FUNC_MAP_START as usize) as usize;
        let func_count = bytecode.read_u8(bytecode::POS_FUNC_COUNT as usize) as usize;

        for i in 0..func_count {
            let base = func_map_start + i * 8;
            let stack_size = bytecode.read_u8(base + bytecode::FUNC_OFFSET_STACK_SIZE as usize) as usize;
            let param_size = bytecode.read_u8(base + bytecode::FUNC_OFFSET_PARAM_SIZE as usize) as usize;
            let pos = bytecode.read_u16(base + bytecode::FUNC_OFFSET_POS as usize) as usize;
            let ref_start = bytecode.read_u16(base + bytecode::FUNC_OFFSET_REF_START as usize) as usize;

            self.functions.push(Function {
                stack_size,
                param_size,
                pos,
                ref_start,
            });
        }
    }

    fn read_modules(&mut self, bytecode: &Bytecode, all_module_id: &HashMap<String, usize>) -> Vec<usize> {
        let module_map_start = bytecode.read_u16(bytecode::POS_MODULE_MAP_START as usize) as usize;
        let module_count = bytecode.read_u8(bytecode::POS_MODULE_COUNT as usize) as usize;

        let mut modules = Vec::with_capacity(module_count);

        for i in 0..module_count {
            let loc = bytecode.read_u16(module_map_start + i * 2) as usize;
            let len = bytecode.read_u16(loc) as usize;

            let mut buf = Vec::with_capacity(len);
            buf.resize(len, 0);
            bytecode.read_bytes(loc + 2, &mut buf[..]);
            let name = str::from_utf8(&buf[..]).unwrap(); // TODO: Avoid unwrap

            let module_id = all_module_id[name];
            modules.push(module_id);
        }

        modules
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
        let ref_start = self.functions[self.current_func].ref_start;
        bytecode.read_i64(ref_start + ref_id as usize * 8)
    }
    
    pub fn run(&mut self, bytecode: Bytecode, std_module: Module, enable_trace: bool, enable_measure: bool) {
        #[inline]
        fn ip_after_jump_to(ip: usize, loc: u8) -> usize {
            let loc = i8::from_le_bytes([loc]) as isize;
            (ip as isize - 2 + loc * 2) as usize
        }

        // Module
        let mut all_modules = vec![std_module];

        let mut all_module_id = HashMap::new();
        all_module_id.insert(String::from("$std"), 0);

        let modules = self.read_modules(&bytecode, &all_module_id);

        // Function
        self.read_functions(&bytecode);
        if self.functions.is_empty() {
            panic!("the bytecode need an entrypoint");
        }

        let string_map_start = bytecode.read_u16(bytecode::POS_STRING_MAP_START as usize) as usize;

        self.current_func = 0;

        let func = &self.functions[0];
        self.ip = func.pos;
        self.fp = 0;
        self.sp = func.stack_size as usize - 1;

        loop {
            let [opcode, arg] = self.next_inst(&bytecode);
            if opcode == opcode::END {
                break;
            }

            if cfg!(debug_assertions) && enable_trace {
                let func = &self.functions[self.current_func];
                bytecode.dump_inst(opcode, arg, self.ip - 2, func.ref_start, string_map_start);
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
                    let value = self.get_ref_value_i64(&bytecode, arg);
                    push!(self, Value::Int(value));
                },
                opcode::STRING => {
                    let loc = bytecode.read_u16(string_map_start + arg as usize * 2) as usize;

                    // Read the string length
                    let len = bytecode.read_u64(loc) as usize;

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
                        bytecode.read_bytes(loc + size_of::<u64>() as usize, &mut bytes);
                    }
                    
                    push!(self, Value::PointerToHeap(region));
                },
                opcode::TRUE => {
                    push!(self, Value::Bool(true));
                },
                opcode::FALSE => {
                    push!(self, Value::Bool(false));
                },
                opcode::NULL => {
                    let nullptr = unsafe { NonNull::new_unchecked(ptr::null_mut()) };
                    push!(self, Value::PointerToStack(nullptr));
                },
                opcode::POINTER => {
                    let value = pop!(self, Value).expect_ref();
                    push!(self, Value::PointerToStack(value));
                },
                opcode::DEREFERENCE => {
                    let ptr = pop!(self, Value).expect_ptr();
                    push!(self, Value::Ref(ptr));
                },
                opcode::NEGATIVE => {
                    match self.stack[self.sp] {
                        Value::Int(ref mut n) => *n = -*n,
                        _ => panic!("expected int"),
                    };
                },
                opcode::COPY => {
                    let size = arg as usize;
                    if self.sp + size >= STACK_SIZE {
                        panic!("stack overflow");
                    }

                    let value_ref = pop!(self, Value).expect_ref();

                    unsafe {
                        let value_ref = value_ref.as_ptr();
                        let dst = &mut self.stack[self.sp + 1];
                        ptr::copy_nonoverlapping(value_ref, dst, size);
                    }

                    self.sp += size;
                },
                opcode::OFFSET => {
                    let offset: i64 = pop!(self);
                    if offset < 0 {
                        panic!("negative offset");
                    }

                    let ptr = match self.stack[self.sp] {
                        Value::Ref(ref mut ptr) => ptr,
                        _ => panic!("expected ref"),
                    };

                    let new_ptr = unsafe { ptr.as_ptr().add(offset as usize) };
                    *ptr = NonNull::new(new_ptr).unwrap();
                },
                // opcode::DUPLICATE => {}
                opcode::LOAD_REF => {
                    let loc = (self.fp as isize + i8::from_le_bytes([arg]) as isize) as usize;
                    if loc >= STACK_SIZE {
                        panic!("out of bounds");
                    }

                    let value = &mut self.stack[loc];
                    let ptr = unsafe { NonNull::new_unchecked(value as *mut _) };
                    push!(self, Value::Ref(ptr));
                },
                opcode::LOAD_COPY => {
                    let loc = i8::from_le_bytes([arg & 0b11111000]) >> 3;
                    let size = (arg & 0b00000111) as usize;

                    let loc = (self.fp as isize + loc as isize) as usize;
                    if loc >= STACK_SIZE {
                        panic!("out of bounds");
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

                    unsafe {
                        let dst = pop!(self, Value).expect_ref().as_ptr();
                        let src = &self.stack[self.sp - size + 1] as *const _;
                        ptr::copy_nonoverlapping(src, dst, size);
                    }

                    self.sp -= size;
                },
                opcode::BINOP_ADD..=opcode::BINOP_NEQ => {
                    match pop!(self, Value) {
                        Value::Int(rhs) => {
                            let lhs: i64 = pop!(self);

                            let result = match opcode {
                                opcode::BINOP_ADD => Value::Int(lhs + rhs),
                                opcode::BINOP_SUB => Value::Int(lhs - rhs),
                                opcode::BINOP_MUL => Value::Int(lhs * rhs),
                                opcode::BINOP_DIV => Value::Int(lhs / rhs),
                                opcode::BINOP_MOD => Value::Int(lhs % rhs),
                                opcode::BINOP_LT => Value::Bool(lhs < rhs),
                                opcode::BINOP_LE => Value::Bool(lhs <= rhs),
                                opcode::BINOP_GT => Value::Bool(lhs > rhs),
                                opcode::BINOP_GE => Value::Bool(lhs >= rhs),
                                opcode::BINOP_EQ => Value::Bool(lhs == rhs),
                                opcode::BINOP_NEQ => Value::Bool(lhs != rhs),
                                _ => panic!("bug"),
                            };

                            push!(self, result);
                        },
                        lhs => {
                            let lhs = lhs.expect_ptr();
                            let rhs = pop!(self, Value).expect_ptr();

                            let result = match opcode {
                                opcode::BINOP_EQ => Value::Bool(lhs == rhs),
                                opcode::BINOP_NEQ => Value::Bool(lhs != rhs),
                                _ => panic!("unexpected binary operator"),
                            };

                            push!(self, result);
                        },
                    }
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

                    push!(self, Value::PointerToHeap(region));
                },
                opcode::CALL => {
                    let func = &self.functions[arg as usize];

                    push!(self, Value::Int(self.current_func as i64));
                    self.current_func = arg as usize;

                    push!(self, Value::Int(self.ip as i64));
                    push!(self, Value::Int(self.fp as i64));

                    self.ip = func.pos;

                    // Allocate stack frame
                    self.fp = self.sp + 1;
                    self.sp += func.stack_size as usize;
                },
                opcode::CALL_EXTERN => {
                    let module_id = ((arg & 0b11110000) >> 4) as usize;
                    let func_id = (arg & 0b00001111) as usize;

                    let actual_module_id = modules[module_id];
                    let module = &mut all_modules[actual_module_id];

                    match module {
                        Module::Normal => unimplemented!(),
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
                    self.fp = pop!(self, i64) as usize;

                    self.ip = pop!(self, i64) as usize;
                    let prev_func = pop!(self, i64) as usize;

                    // Pop arguments
                    self.sp -= self.functions[self.current_func].param_size as usize;

                    self.current_func = prev_func;
                },
                opcode::CALL_NATIVE => {
                    unimplemented!();
                },
                opcode::JUMP => {
                    self.ip = ip_after_jump_to(self.ip, arg);
                },
                opcode::JUMP_IF_FALSE => {
                    let cond: bool = pop!(self);
                    if !cond {
                        self.ip = ip_after_jump_to(self.ip, arg);
                    }
                },
                opcode::JUMP_IF_TRUE => {
                    let cond: bool = pop!(self);
                    if cond {
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
            let main_func_stack_size = self.functions[0].stack_size as usize;
            if (main_func_stack_size == 0 && self.sp == 0) || self.sp != main_func_stack_size - 1 {
                self.dump_stack(self.sp);
                eprintln!("warning: expected stack size {}, but sp is {}.", main_func_stack_size, self.sp);
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
