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

pub struct VM<'a> {
    functions: HashMap<Id, Function>,
    
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
    pub fn new(functions: HashMap<Id, Function>) -> Self {
        Self {
            functions,
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
        let main_id = IdMap::get("$main").unwrap();
        if !self.functions.contains_key(&main_id) {
            return;
        }

        let main_func = &self.functions[&main_id];
        // extend stack for local variables
        self.sp += main_func.stack_size;

        let mut insts = &main_func.insts;

        loop {
            if self.ip >= insts.len() {
                break;
            }

            let inst = &insts[self.ip];

            if enable_trace {
                Self::trace(inst);
            }

            match inst {
                Inst::Int(n) => {
                    push!(self, Value::Int(*n));
                },
                Inst::String(s) => {
                    let size = mem::size_of::<usize>() + s.len();
                    let mut region = self.gc.alloc::<u8>(size, &mut self.stack);

                    unsafe {
                        let new_str = region.as_mut().as_mut_ptr::<Lang2String>();
                        (*new_str).write_string(s);
                    }

                    push!(self, Value::Pointer(Pointer::ToHeap(region)));
                },
                Inst::True => {
                    push!(self, Value::Bool(true));
                },
                Inst::False => {
                    push!(self, Value::Bool(false));
                },
                Inst::Null => {
                    let nullptr = unsafe { NonNull::new_unchecked(ptr::null_mut()) };
                    push!(self, Value::Pointer(Pointer::ToStack(nullptr)));
                },
                Inst::Load(loc) => {
                    let loc = (self.fp as isize + *loc) as usize;
                    if loc >= STACK_SIZE {
                        panic!("out of bounds");
                    }

                    let value = &mut self.stack[loc];
                    let ptr = unsafe { NonNull::new_unchecked(value as *mut _) };
                    push!(self, Value::Ref(ptr));
                },
                Inst::LoadCopy(loc, size) => {
                    let base = (self.fp as isize + *loc) as usize;
                    if base + size >= STACK_SIZE || self.sp + size >= STACK_SIZE {
                        panic!("out of bounds");
                    }

                    unsafe {
                        let src = &self.stack[base] as *const _;
                        let dst = &mut self.stack[self.sp + 1] as *mut _;
                        ptr::copy_nonoverlapping(src, dst, *size);
                    }

                    self.sp += size;
                },
                Inst::Pointer => {
                    let value_ptr = pop!(self, Value).expect_ref();

                    push!(self, Value::Pointer(Pointer::ToStack(value_ptr)));
                },
                Inst::Dereference => {
                    let ptr = pop!(self, Value).expect_ptr();
                    push!(self, Value::Ref(ptr));
                },
                Inst::Negative => {
                    let n = match self.stack[self.sp] {
                        Value::Int(ref mut n) => n,
                        _ => panic!("expected int"),
                    };

                    *n = -(*n);
                },
                Inst::Copy(size) => {
                    // Copy if TOS is a reference
                    match &self.stack[self.sp] {
                        Value::Ref(ptr) if *size == 1 => {
                            let value = unsafe { ptr.as_ref() };
                            self.stack[self.sp] = value.clone();
                        },
                        Value::Ref(ptr) => {
                            self.sp -= 1; // pop

                            unsafe {
                                let src = ptr.as_ptr();
                                let dst = &mut self.stack[self.sp + 1] as *mut _;
                                ptr::copy_nonoverlapping(src, dst, *size);
                            }

                            self.sp += size;
                        },
                        _ => {},
                    };
                },
                Inst::Duplicate(size, count) => {
                    let ptr = &self.stack[self.sp - (size - 1)] as *const Value;

                    for i in 1..=*count {
                        unsafe {
                            let dest = ptr.add(i * size) as *mut _;
                            ptr.copy_to_nonoverlapping(dest, *size);
                        }
                    }

                    self.sp += size * count;
                },
                Inst::Offset => {
                    let offset: i64 = pop!(self);

                    let ptr = match &mut self.stack[self.sp] {
                        Value::Ref(ptr) => ptr,
                        _ => panic!("expected ref"),
                    };

                    let new_ptr = unsafe { ptr.as_ptr().add(offset as usize) };
                    *ptr = NonNull::new(new_ptr).unwrap();
                },
                Inst::BinOp(binop) => {
                    match pop!(self, Value) {
                        Value::Int(rhs) => {
                            let lhs: i64 = pop!(self);

                            let result = match binop {
                                BinOp::Add => Value::Int(lhs + rhs),
                                BinOp::Sub => Value::Int(lhs - rhs),
                                BinOp::Mul => Value::Int(lhs * rhs),
                                BinOp::Div => Value::Int(lhs / rhs),
                                BinOp::Mod => Value::Int(lhs % rhs),
                                BinOp::LessThan => Value::Bool(lhs < rhs),
                                BinOp::LessThanOrEqual => Value::Bool(lhs <= rhs),
                                BinOp::GreaterThan => Value::Bool(lhs > rhs),
                                BinOp::GreaterThanOrEqual => Value::Bool(lhs >= rhs),
                                BinOp::Equal => Value::Bool(lhs == rhs),
                                BinOp::NotEqual => Value::Bool(lhs != rhs),
                            };

                            push!(self, result);
                        },
                        Value::Pointer(lhs) => {
                            let lhs = lhs.as_non_null();
                            let rhs = pop!(self, Value).expect_ptr();

                            let result = match binop {
                                BinOp::Equal => Value::Bool(lhs == rhs),
                                BinOp::NotEqual => Value::Bool(lhs != rhs),
                                _ => panic!("unexpected binary operator"),
                            };

                            push!(self, result);
                        },
                        _ => panic!("expected int or pointer"),
                    }
                },
                Inst::StoreWithSize(size) => {
                    unsafe {
                        let dst = pop!(self, Value).expect_ref().as_ptr();
                        let src = &self.stack[self.sp - *size + 1] as *const _;
                        ptr::copy_nonoverlapping(src, dst, *size);
                    }

                    self.sp -= size;
                },
                Inst::Alloc(size) => {
                    let mut ptr_to_region = self.gc.alloc::<Value>(*size, &mut self.stack);

                    unsafe {
                        let dst = ptr_to_region.as_mut().as_mut_ptr::<Value>();
                        let src = &self.stack[self.sp - *size + 1] as *const _;
                        ptr::copy_nonoverlapping(src, dst, *size);
                    }

                    self.sp -= size;

                    push!(self, Value::Pointer(Pointer::ToHeap(ptr_to_region)));
                },
                Inst::Call(name) => {
                    let func = &self.functions[name];

                    self.stack[self.sp + 1] = Value::Int(func.param_size as i64);
                    self.stack[self.sp + 2] = Value::Int(self.ip as i64);
                    self.stack[self.sp + 3] = Value::Int(self.fp as i64);
                    self.sp += 3;
                    self.insts_stack.push(insts);

                    // Allocate stack frame
                    self.ip = 0;
                    self.fp = self.sp + 1;
                    self.sp += func.stack_size;

                    insts = &func.insts;

                    continue;
                },
                Inst::CallNative(_, func, param_size) => {
                    let mut ctx = Context {
                        stack: unsafe { NonNull::new_unchecked(&mut self.stack[0] as *mut _) },
                        gc: unsafe { NonNull::new_unchecked(&mut self.gc as *mut _) },
                        current_param: 0,
                        param_size: *param_size,
                        sp: self.sp,
                    };

                    let return_values = func.0(&mut ctx);

                    // Pop arguments
                    self.sp -= param_size;

                    let rv_size = return_values.len();

                    unsafe {
                        let src = return_values.as_ptr();
                        let dst = &mut self.stack[self.sp + 1] as *mut _;
                        ptr::copy_nonoverlapping(src, dst, rv_size);
                    }
                    
                    self.sp += rv_size;
                },
                Inst::Pop => {
                    pop!(self, Value);
                },
                Inst::Return => {
                    // Restore stack frame
                    self.sp = self.fp - 1;
                    self.fp = pop!(self, i64) as usize;
                    self.ip = pop!(self, i64) as usize;
                    insts = self.insts_stack.pop().unwrap();

                    // Pop arguments
                    let param_size = pop!(self, i64) as usize;
                    self.sp -= param_size;
                }
                Inst::Jump(loc) => {
                    self.ip = *loc;
                    continue;
                },
                Inst::JumpIfZero(loc) => {
                    let cond: bool = pop!(self);
                    if !cond {
                        self.ip = *loc;
                        continue;
                    }
                },
                Inst::JumpIfNonZero(loc) => {
                    let cond: bool = pop!(self);
                    if cond {
                        self.ip = *loc;
                        continue;
                    }
                },
            }

            self.ip += 1;
        }

        assert!(self.insts_stack.is_empty());
        assert_eq!(self.sp, main_func.stack_size);
        self.sp = self.fp;
    }

    fn trace(inst: &Inst) {
        use std::io::{Write, stderr};
        eprintln!("{}", inst);
        stderr().flush().unwrap();
    }
}
