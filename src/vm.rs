use std::ptr::NonNull;
use std::mem;
use std::mem::size_of;
use std::slice;
use std::ptr;
use std::io::{Read, Seek};

use crate::bytecode;
use crate::bytecode::{BytecodeStream, opcode};
use crate::value::{FromValue, Value};
use crate::gc::Gc;

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

macro_rules! fn_read {
    ($ty:ty, $name:ident) => {
        #[allow(dead_code)]
        fn $name(&self, pos: usize) -> $ty {
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

    fn_read!(u8, read_u8);
    fn_read!(i8, read_i8);
    fn_read!(u16, read_u16);
    fn_read!(i16, read_i16);
    fn_read!(u32, read_u32);
    fn_read!(i32, read_i32);
    fn_read!(u64, read_u64);
    fn_read!(i64, read_i64);
    fn_read!(u128, read_u128);
    fn_read!(i128, read_i128);

    fn read_bytes(&self, pos: usize, bytes: &mut [u8]) {
        unsafe {
            let src = self.bytes.as_ptr().add(pos);
            let dst = bytes.as_mut_ptr();
            ptr::copy_nonoverlapping(src, dst, bytes.len());
        }
    }
}

#[derive(Debug)]
struct Function {
    stack_size: usize,
    param_size: usize,
    pos: usize,
    ref_start: usize,
}

pub struct VM {
    bytecode: Bytecode,
    
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
    pub fn new(bytecode: Bytecode) -> Self {
        Self {
            bytecode,
            gc: Gc::new(),
            ip: 0,
            fp: 0,
            sp: 0,
            stack: unsafe { mem::zeroed() },
            functions: Vec::new(),
            current_func: 0,
        }
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

    #[allow(dead_code)]
    fn dump_stack(&self, stop: usize) {
        println!("-------- STACK DUMP --------");
        for (i, value) in self.stack.iter().enumerate() {
            if i > stop {
                break;
            }

            if i == self.fp as usize {
                print!("(fp) ");
            }

            Self::dump_value(value, 0);
        }
        println!("-------- END DUMP ----------");
    }

    fn read_functions(&mut self) {
        let func_map_start = self.bytecode.read_u8(bytecode::POS_FUNC_MAP_START as usize) as usize;
        let func_count = self.bytecode.read_u8(bytecode::POS_FUNC_COUNT as usize) as usize;

        for i in 0..func_count {
            let base = func_map_start + i * 8;
            let stack_size = self.bytecode.read_u8(base + bytecode::FUNC_OFFSET_STACK_SIZE as usize) as usize;
            let param_size = self.bytecode.read_u8(base + bytecode::FUNC_OFFSET_PARAM_SIZE as usize) as usize;
            let pos = self.bytecode.read_u16(base + bytecode::FUNC_OFFSET_POS as usize) as usize;
            let ref_start = self.bytecode.read_u16(base + bytecode::FUNC_OFFSET_REF_START as usize) as usize;

            self.functions.push(Function {
                stack_size,
                param_size,
                pos,
                ref_start,
            });
        }
    }

    fn next_inst(&mut self) -> [u8; 2] {
        let mut buf = [0u8; 2];
        self.bytecode.read_bytes(self.ip, &mut buf);

        self.ip += 2;

        buf
    }

    fn get_ref_value_i64(&mut self, ref_id: u8) -> i64 {
        let ref_start = self.functions[self.current_func].ref_start;
        let mut bytes = [0u8; 8];
        self.bytecode.read_bytes(ref_start + ref_id as usize * 8, &mut bytes);
        i64::from_le_bytes(bytes)
    }
    
    pub fn run(&mut self, enable_trace: bool) {
        self.read_functions();

        let string_map_start = self.bytecode.read_u16(bytecode::POS_STRING_MAP_START as usize) as usize;

        if self.functions.is_empty() {
            panic!("the bytecode need an entrypoint");
        }

        self.current_func = 0;

        let func = &self.functions[0];
        self.ip = func.pos;
        self.sp = func.stack_size as usize;

        loop {
            let [opcode, arg] = self.next_inst();
            if opcode == opcode::END {
                break;
            }

            if enable_trace {
                println!("TRACE 0x{:x}", opcode);
            }

            match opcode {
                opcode::NOP => {},
                opcode::ZERO => {
                    push!(self, Value::Int(0));
                },
                opcode::INT => {
                    let value = self.get_ref_value_i64(arg);
                    push!(self, Value::Int(value));
                },
                opcode::STRING => {
                    let loc = self.bytecode.read_u16(string_map_start + arg as usize * 2) as usize;

                    // Read the string length
                    let len = self.bytecode.read_u64(loc) as usize;

                    let size = len + size_of::<u64>();
                    let mut region = self.gc.alloc::<u8>(size, &mut self.stack[..=self.sp]);

                    // Read the string bytes
                    unsafe {
                        let bytes_ptr = region.as_mut().as_mut_ptr::<u8>().add(size_of::<u64>());
                        let mut bytes = slice::from_raw_parts_mut(bytes_ptr, len);
                        self.bytecode.read_bytes(loc + size_of::<u64>() as usize, &mut bytes);
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

                    let mut region = self.gc.alloc::<Value>(size, &mut self.stack[..=self.sp]);

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
                    self.current_func = arg as usize;

                    push!(self, Value::Int(self.current_func as i64));
                    push!(self, Value::Int(self.ip as i64));
                    push!(self, Value::Int(self.fp as i64));

                    self.ip = func.pos;

                    // Allocate stack frame
                    self.fp = self.sp + 1;
                    self.sp += func.stack_size as usize;
                },
                opcode::RETURN => {
                    // Restore stack frame
                    self.sp = self.fp - 1;
                    self.fp = pop!(self, i64) as usize;

                    self.ip = pop!(self, i64) as usize;
                    self.current_func = pop!(self, i64) as usize;

                    // Pop arguments
                    self.sp -= self.functions[self.current_func].param_size as usize;
                },
                opcode::CALL_NATIVE => {
                    unimplemented!();
                },
                opcode::JUMP => {
                    let func = &self.functions[self.current_func];
                    self.ip = func.pos + arg as usize;
                },
                opcode::JUMP_IF_FALSE => {
                    let cond: bool = pop!(self);
                    if !cond {
                        let func = &self.functions[self.current_func];
                        self.ip = func.pos + arg as usize;
                    }
                },
                opcode::JUMP_IF_TRUE => {
                    let cond: bool = pop!(self);
                    if cond {
                        let func = &self.functions[self.current_func];
                        self.ip = func.pos + arg as usize;
                    }
                },
                _ => {
                    panic!("Unknown opcode (0x{:x})", opcode);
                },
            }
        }

        self.dump_stack(self.sp);
        assert_eq!(self.sp, self.functions[0].stack_size as usize);
        self.sp = self.fp;
    }
}
