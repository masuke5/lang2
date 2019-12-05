use std::collections::HashMap;
use std::ptr;
use std::ptr::NonNull;
use std::mem;

use crate::id::{Id, IdMap};
use crate::inst::{Inst, BinOp, Function};
use crate::value::{FromValue, Value, Pointer};
use crate::utils;
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
            Value::String(s) => println!("str \"{}\"", utils::escape_string(s)),
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
                    push!(self, Value::String(s.clone()));
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
                    let value = &mut self.stack[(self.fp as isize + *loc) as usize];
                    let ptr = NonNull::new(value as *mut _).unwrap();
                    push!(self, Value::Ref(ptr));
                },
                Inst::LoadCopy(loc, size) => {
                    for i in 0..*size {
                        let value = self.stack[(self.fp as isize + *loc) as usize + i].clone();
                        push!(self, value);
                    }
                },
                Inst::Pointer => {
                    let value_ref: Value = pop!(self);
                    let value_ptr = value_ref.expect_ref();

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
                            let src = ptr.as_ptr();

                            self.sp -= 1; // pop

                            for i in 0..*size {
                                unsafe {
                                    let src_ptr = src.add(i);
                                    push!(self, (*src_ptr).clone());
                                };
                            }
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
                    let ptr = match pop!(self, Value) {
                        Value::Ref(ptr) => ptr,
                        _ => panic!("expected reference"),
                    };

                    for i in (0..*size).rev() {
                        let value: Value = pop!(self);

                        unsafe {
                            let ptr = ptr.as_ptr().add(i);
                            *ptr = value;
                        }
                    }
                },
                Inst::Alloc(size) => {
                    let ptr_to_region = self.gc.alloc(*size, &mut self.stack);

                    unsafe {
                        let base_ptr = ptr_to_region.as_ref().base.as_ptr();
                        for i in (0..*size).rev() {
                            let value: Value = pop!(self);
                            ptr::write(base_ptr.add(i), value);
                        }
                    }

                    push!(self, Value::Pointer(Pointer::ToHeap(ptr_to_region)));
                },
                Inst::Call(name) => {
                    let func = &self.functions[name];

                    push!(self, Value::Int(func.param_size as i64));
                    push!(self, Value::Int(self.ip as i64));
                    push!(self, Value::Int(self.fp as i64));
                    self.insts_stack.push(insts);

                    // Allocate stack frame
                    self.ip = 0;
                    self.fp = self.sp + 1;
                    self.sp += func.stack_size;

                    insts = &func.insts;

                    continue;
                },
                Inst::CallNative(_, func, param_size) => {
                    let return_value = func.0(&self.stack[self.sp - param_size + 1..=self.sp]);

                    // Pop arguments
                    self.sp -= param_size;

                    push!(self, return_value);
                },
                Inst::Pop => {
                    pop!(self, Value);
                },
                Inst::Return(size) => {
                    // Save a return value
                    let mut values = Vec::new();
                    for _ in 0..*size {
                        let value: Value = pop!(self);
                        values.push(value);
                    }

                    // Restore stack frame
                    self.sp = self.fp - 1;
                    self.fp = pop!(self, i64) as usize;
                    self.ip = pop!(self, i64) as usize;
                    insts = self.insts_stack.pop().unwrap();

                    // Pop arguments
                    let param_size = pop!(self, i64) as usize;
                    self.sp -= param_size;

                    // Push the return value
                    for value in values.into_iter().rev() {
                        push!(self, value);
                    }
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
        eprintln!("{}", inst);
    }
}
