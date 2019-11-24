use std::collections::HashMap;
use std::ptr::NonNull;
use std::mem;

use crate::id::{Id, IdMap};
use crate::inst::{Inst, BinOp, Function};
use crate::value::{FromValue, Value};

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

    #[allow(dead_code)]
    fn dump_stack(&self, stop: usize) {
        fn dump_value(value: &Value, depth: usize) {
            print!("{}", "  ".repeat(depth));
            match value {
                Value::Int(n) => println!("int {}", n),
                Value::String(s) => println!("str \"{}\"", s),
                Value::Bool(true) => println!("bool true"),
                Value::Bool(false) => println!("bool false"),
                Value::Record(values) => {
                    for value in values {
                        dump_value(value, depth + 1);
                    }
                },
                Value::Ref(ptr) => {
                    let value = unsafe { ptr.as_ref() };
                    dump_value(value, depth + 1);
                },
                Value::Unintialized => println!("uninitialized"),
            }
        }

        println!("-------- STACK DUMP --------");
        for (i, value) in self.stack.iter().enumerate() {
            if i > stop {
                break;
            }

            if i == self.fp {
                print!("(fp) ");
            }

            dump_value(value, 0);
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
                Inst::Load(loc, offset) => {
                    let value = &mut self.stack[(self.fp as isize + *loc) as usize + offset];
                    if value.should_clone() {
                        push!(self, value.clone());
                    } else {
                        let ptr = NonNull::new(value as *mut _).unwrap();
                        push!(self, Value::Ref(ptr));
                    }
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
                Inst::Field(i) => {
                    let mut values: Vec<Value> = pop!(self);
                    let value = values.drain(*i..*i + 1).next().unwrap();
                    push!(self, value);
                },
                Inst::BinOp(binop) => {
                    match binop {
                        BinOp::And | BinOp::Or => {
                            let rhs: bool = pop!(self);
                            let lhs: bool = pop!(self);

                            let result = match binop {
                                BinOp::And => Value::Bool(lhs && rhs),
                                BinOp::Or => Value::Bool(lhs || rhs),
                                _ => panic!(),
                            };

                            push!(self, result);
                        },
                        binop => {
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
                                _ => panic!(),
                            };

                            push!(self, result);
                        },
                    }
                },
                Inst::Save(loc, offset) => {
                    let value: Value = pop!(self);
                    let loc = (self.fp as isize + loc) as usize + offset;

                    self.stack[loc] = value;
                },
                Inst::Call(name) => {
                    let func = &self.functions[name];

                    let arg_size = func.params.iter().fold(0, |acc, ty| acc + ty.size() as i64);
                    push!(self, Value::Int(arg_size));
                    push!(self, Value::Int(self.ip as i64));
                    push!(self, Value::Int(self.fp as i64));
                    self.insts_stack.push(insts);
                    
                    // Allocate stack frame
                    self.ip = 0;
                    self.fp = self.sp;
                    self.sp += func.stack_size;

                    insts = &func.insts;

                    continue;
                },
                #[cfg(debug_assertions)]
                Inst::CallNative(_, func, param_count) => {
                    let return_value = func.0(&self.stack[self.sp - param_count + 1..self.sp + 1]);

                    // Pop arguments
                    self.sp -= param_count;

                    push!(self, return_value);
                },
                #[cfg(not(debug_assertions))]
                Inst::CallNative(func, param_count) => {
                    let return_value = func.0(&self.stack[self.sp - param_count + 1..self.sp + 1]);

                    // Pop arguments
                    self.sp -= param_count;

                    push!(self, return_value);
                },
                Inst::Pop => {
                    pop!(self, Value);
                },
                Inst::Return(_) => {
                    // Save a return value
                    let value: Value = pop!(self);

                    // Restore
                    self.sp = self.fp;
                    self.fp = pop!(self, i64) as usize;
                    self.ip = pop!(self, i64) as usize;
                    insts = self.insts_stack.pop().unwrap();

                    // Pop arguments
                    let arg_size = pop!(self, i64) as usize;
                    self.sp -= arg_size;

                    // Push the return value
                    push!(self, value);
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
