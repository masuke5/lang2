use std::collections::HashMap;
use std::iter;

use crate::id::{Id, IdMap};
use crate::inst::{Inst, BinOp, Function};
use crate::value::{FromValue, Value};

macro_rules! pop {
    ($self:ident, $ty:ty) => {
        {
            let v: $ty = FromValue::from_value(&$self.stack.pop().unwrap());
            v
        }
    };
    ($self:ident) => {
        FromValue::from_value(&$self.stack.pop().unwrap())
    };
}

#[derive(Debug)]
pub struct VM<'a> {
    functions: HashMap<Id, Function>,

    ip: usize,
    fp: usize,
    stack: Vec<Value>,
    insts_stack: Vec<&'a Vec<Inst>>,
    return_value: Vec<Value>,
}

impl<'a> VM<'a> {
    pub fn new(functions: HashMap<Id, Function>) -> Self {
        Self {
            functions,
            ip: 0,
            fp: 0,
            stack: Vec::new(),
            insts_stack: Vec::new(),
            return_value: Vec::with_capacity(10),
        }
    }

    #[allow(dead_code)]
    fn dump_stack(&self) {
        fn dump_value(value: &Value) {
            match value {
                Value::Int(n) => println!("int {}", n),
                Value::String(s) => println!("str \"{}\"", s),
                Value::Bool(true) => println!("bool true"),
                Value::Bool(false) => println!("bool false"),
                Value::Unintialized => println!("uninitialized"),
            }
        }

        println!("-------- STACK DUMP --------");
        for (i, value) in self.stack.iter().enumerate() {
            if i == self.fp {
                print!("(fp) ");
            }

            dump_value(value);
        }
        println!("-------- END DUMP ----------");
    }

    pub fn run(&'a mut self) {
        let main_id = IdMap::get("$main").unwrap();
        if !self.functions.contains_key(&main_id) {
            return;
        }

        let main_func = &self.functions[&main_id];
        self.stack.extend(iter::repeat(Value::Unintialized).take(main_func.stack_size));
        let mut insts = &main_func.insts;

        loop {
            if self.ip >= insts.len() {
                self.stack.truncate(self.fp);
                break;
            }

            match &insts[self.ip] {
                Inst::Int(n) => {
                    self.stack.push(Value::Int(*n));
                },
                Inst::String(s) => {
                    self.stack.push(Value::String(s.clone()));
                },
                Inst::True => {
                    self.stack.push(Value::Bool(true));
                },
                Inst::False => {
                    self.stack.push(Value::Bool(false));
                },
                Inst::Load(loc, offset) => {
                    let value = self.stack[(self.fp as isize + *loc) as usize + offset].clone();
                    self.stack.push(value);
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

                            self.stack.push(result);
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

                            self.stack.push(result);
                        },
                    }
                },
                Inst::Save(loc, offset) => {
                    let value: Value = pop!(self);
                    let loc = (self.fp as isize + loc) as usize + offset;

                    self.stack[loc] = value.clone();
                },
                Inst::Call(name) => {
                    let func = &self.functions[name];

                    let arg_size = func.params.iter().fold(0, |acc, ty| acc + ty.size() as i64);
                    self.stack.push(Value::Int(arg_size));
                    self.stack.push(Value::Int(self.ip as i64));
                    self.stack.push(Value::Int(self.fp as i64));
                    self.insts_stack.push(insts);
                    
                    self.ip = 0;
                    self.fp = self.stack.len();

                    self.stack.extend(iter::repeat(Value::Unintialized).take(func.stack_size));
                    insts = &func.insts;

                    continue;
                },
                #[cfg(debug_assertions)]
                Inst::CallNative(_, func, param_count) => {
                    let start = self.stack.len() - param_count;
                    let end = start + param_count;
                    let return_value = func.0(&self.stack[start..end]);

                    self.stack.truncate(start);

                    self.stack.push(return_value);
                },
                #[cfg(not(debug_assertions))]
                Inst::CallNative(func, param_count) => {
                    let start = self.stack.len() - param_count;
                    let end = start + param_count;
                    let return_value = func.0(&self.stack[start..end]);

                    self.stack.truncate(start);

                    self.stack.push(return_value);
                },
                Inst::Pop => {
                    self.stack.pop().unwrap();
                },
                Inst::Return(size) => {
                    // Save a return value
                    self.return_value.resize(*size, Value::Unintialized);
                    for i in 0..*size {
                        self.return_value[size - i - 1] = pop!(self);
                    }

                    // Restore
                    self.stack.truncate(self.fp);
                    self.fp = pop!(self, i64) as usize;
                    self.ip = pop!(self, i64) as usize;
                    insts = self.insts_stack.pop().unwrap();

                    let arg_size = pop!(self, i64) as usize;
                    self.stack.truncate(self.stack.len() - arg_size);

                    // Push the return value
                    for value in self.return_value.drain(..) {
                        self.stack.push(value);
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
