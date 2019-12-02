use std::collections::HashMap;
use std::ptr::NonNull;
use std::mem;

use crate::id::{Id, IdMap};
use crate::inst::{Inst, BinOp, Function};
use crate::value::{FromValue, Value};
use crate::utils;

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
            Value::Record(values) => {
                for value in values {
                    Self::dump_value(value, depth + 1);
                }
            },
            Value::Ref(ptr) => {
                println!("ref");
                let value = unsafe { ptr.as_ref() };
                Self::dump_value(value, depth + 1);
            },
            Value::Pointer(ptr) => println!("ptr {:p}", ptr),
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

    pub fn run(&'a mut self) {
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
                assert!(self.insts_stack.is_empty());
                assert_eq!(self.sp, main_func.stack_size);
                self.sp = self.fp;
                break;
            }

            match &insts[self.ip] {
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
                Inst::Load(loc) => {
                    let value = &mut self.stack[(self.fp as isize + *loc) as usize];
                    let ptr = NonNull::new(value as *mut _).unwrap();
                    push!(self, Value::Ref(ptr));
                },
                Inst::Record(size) => {
                    let mut values = Vec::with_capacity(*size);
                    values.resize(*size, Value::Unintialized);

                    for i in (0..*size).rev() {
                        let v: Value = pop!(self);
                        values[i] = v;
                    }

                    push!(self, Value::Record(values));
                },
                Inst::Pointer => {
                    let value_ref: Value = pop!(self);
                    match value_ref {
                        Value::Ref(ptr) => {
                            push!(self, Value::Pointer(ptr));
                        },
                        _ => panic!("expected ref"),
                    }
                },
                Inst::Dereference => {
                    let ptr: NonNull<Value> = pop!(self);
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
                        Value::Ref(mut ptr) if *size == 1 => {
                            let value = unsafe { ptr.as_mut() };
                            self.stack[self.sp] = value.clone();
                        },
                        Value::Ref(ptr) => {
                            let src = ptr.as_ptr();

                            self.sp -= 1; // pop

                            for i in 0..*size {
                                unsafe {
                                    let src_ptr = src.wrapping_add(i);
                                    push!(self, (*src_ptr).clone());
                                };
                            }
                        },
                        _ => {},
                    }
                },
                Inst::Offset(offset) => {
                    let ptr = match &mut self.stack[self.sp] {
                        Value::Ref(ptr) => ptr,
                        _ => panic!("expected ref"),
                    };

                    let new_ptr = ptr.as_ptr().wrapping_add(*offset);
                    *ptr = NonNull::new(new_ptr).unwrap();
                },
                Inst::Field(i) => {
                    fn field(value: &mut Value, i: usize, needs_ref: bool) -> Value {
                        match value {
                            Value::Ref(ptr) => {
                                let mut value = unsafe { ptr.as_mut() };
                                field(&mut value, i, true)
                            },
                            Value::Record(ref mut values) => {
                                let value = &mut values[i];

                                if needs_ref {
                                    Value::Ref(NonNull::new(value as *mut _).unwrap())
                                } else {
                                    value.clone()
                                }
                            },
                            _ => panic!(),
                        }
                    }

                    let mut record: Value = pop!(self);
                    let value = field(&mut record, *i, false);

                    push!(self, value);
                },
                Inst::BinOp(binop) => {
                    let rhs: i64 = pop!(self);
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
                Inst::StoreWithSize(size) => {
                    let ptr = match pop!(self, Value) {
                        Value::Ref(ptr) => ptr,
                        _ => panic!("expected reference"),
                    };

                    for i in (0..*size).rev() {
                        let value: Value = pop!(self);

                        unsafe {
                            let ptr = ptr.as_ptr().wrapping_add(i);
                            *ptr = value;
                        }
                    }
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
                #[cfg(debug_assertions)]
                Inst::CallNative(_, func, param_size) => {
                    let return_value = func.0(&self.stack[self.sp - param_size + 1..=self.sp]);

                    // Pop arguments
                    self.sp -= param_size;

                    push!(self, return_value);
                },
                #[cfg(not(debug_assertions))]
                Inst::CallNative(func, param_size) => {
                    let return_value = func.0(&self.stack[self.sp - param_size + 1..self.sp + 1]);

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
    }
}
