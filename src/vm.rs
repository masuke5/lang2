use std::collections::HashMap;
use std::ptr;
use std::ptr::NonNull;
use std::mem;
use std::slice;

use crate::id::{Id, IdMap};
use crate::inst::{Inst, BinOp, Function};
use crate::value::{FromValue, Value, Pointer, Lang2String};
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

pub struct Context {
    stack: NonNull<Value>,
    #[allow(dead_code)] // remove later
    gc: NonNull<Gc>,
    current_param: usize,
    param_size: usize,
    sp: usize,
}

impl Context {
    pub fn next_param<T: FromValue>(&mut self) -> T {
        let stack = unsafe { slice::from_raw_parts_mut(self.stack.as_mut(), STACK_SIZE) };
        let v = mem::replace(&mut stack[self.sp - self.param_size + 1 + self.current_param], Value::Unintialized);
        self.current_param += 1;
        FromValue::from_value(v)
    }

    pub fn next_string_ptr(&mut self) -> &mut Lang2String {
        let stack = unsafe { slice::from_raw_parts_mut(self.stack.as_mut(), STACK_SIZE) };
        let value = mem::replace(&mut stack[self.sp - self.param_size + 1 + self.current_param], Value::Unintialized);
        self.current_param += 1;

        unsafe {
            let ptr = match value {
                Value::Pointer(ptr) => ptr.expect_to_heap::<Lang2String>(),
                _ => panic!(),
            };

            &mut *ptr
        }
    }
}

struct Bytecode {
    bytecode: Vec<u8>,
    func_count: usize,
    string_count: usize,
    func_map_start: usize,
    string_map_start: usize,
}

impl Bytecode {
    fn new(bytecode: Vec<u8>) -> Result<Self, &'static str> {
        if !bytecode.starts_with(&[0x4c, 0x32, 0x42, 0x43]) { // "L2BC"
            return Err("invalid header");
        }

        let mut slf = Self {
            bytecode,
            func_count: 0,
            string_count: 0,
            func_map_start: 0,
            string_map_start: 0,
        };
        slf.init();

        Ok(slf)
    }

    fn init(&mut self) {
        self.func_count = self.bytecode[4] as usize;
        self.string_count = self.bytecode[5] as usize;
        self.func_map_start = self.read::<u16>(6) as usize;
        self.string_map_start = self.read::<u16>(8) as usize;
    }

    fn read<T: Copy>(&self, pos: usize) -> T {
        if pos + mem::size_of::<T>() >= self.bytecode.len() {
            panic!("out of bounds");
        }

        unsafe { *(self.bytecode.as_ptr().add(pos) as *const T) }
    }

    fn get_ptr<T>(&self, pos: usize) -> *const T {
        if pos + mem::size_of::<T>() >= self.bytecode.len() {
            panic!("out of bounds");
        }

        unsafe { self.bytecode.as_ptr().add(pos) as *const T }
    }
}

pub struct VM<'a> {
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
    insts_stack: Vec<&'a Vec<Inst>>,
}

impl<'a> VM<'a> {
    pub fn new(bytecode: Bytecode) -> Self {
        Self {
            bytecode,
            gc: Gc::new(),
            ip: 0,
            fp: 0,
            sp: 0,
            stack: unsafe { mem::zeroed() },
            insts_stack: Vec::new(),
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
            Value::Pointer(ptr) => println!("ptr {:p}", ptr.as_non_null()),
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

            if i == self.fp {
                print!("(fp) ");
            }

            Self::dump_value(value, 0);
        }
        println!("-------- END DUMP ----------");
    }

    pub fn run(&'a mut self, enable_trace: bool) {
        unimplemented!();
    }
}
