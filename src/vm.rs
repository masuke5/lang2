use std::collections::HashMap;
use crate::id::{Id, IdMap};
use crate::inst::{Inst, BinOp, Function};
use crate::value::{FromValue, Value};

fn get_id(id_map: &IdMap, s: &str) -> Id {
    id_map.get(s).unwrap()
}

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
    id_map: IdMap,

    ip: usize,
    fp: usize,
    stack: Vec<Value>,
    insts_stack: Vec<&'a Vec<Inst>>,
}

impl<'a> VM<'a> {
    pub fn new(functions: HashMap<Id, Function>, id_map: IdMap) -> Self {
        Self {
            functions,
            id_map,
            ip: 0,
            fp: 0,
            stack: Vec::new(),
            insts_stack: Vec::new(),
        }
    }

    pub fn run(&'a mut self) {
        if !self.functions.contains_key(&get_id(&self.id_map, "$main")) {
            return;
        }

        let mut insts = &self.functions[&get_id(&self.id_map, "$main")].insts;

        loop {
            if self.ip >= insts.len() {
                break;
            }

            match &insts[self.ip] {
                Inst::Int(n) => {
                    self.stack.push(Value::Int(*n));
                },
                Inst::True => {
                    self.stack.push(Value::Bool(true));
                },
                Inst::False => {
                    self.stack.push(Value::Bool(false));
                },
                Inst::Load(loc, _) => {
                    let value = self.stack[(self.fp as isize + *loc) as usize].clone();
                    self.stack.push(value);
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

                    self.stack.push(result);
                },
                Inst::Save(loc, _) => {
                    let value = self.stack.pop().unwrap();
                    let loc = (self.fp as isize + loc) as usize;

                    self.stack[loc] = value.clone();
                },
                Inst::Call(name) => {
                    let func = &self.functions[name];

                    self.stack.push(Value::Int(func.params.len() as i64));
                    self.stack.push(Value::Int(self.ip as i64));
                    self.stack.push(Value::Int(self.fp as i64));
                    self.insts_stack.push(insts);
                    
                    self.ip = 0;
                    self.fp = self.stack.len();

                    insts = &func.insts;

                    continue;
                },
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
                Inst::Return => {
                    let return_value: Value = pop!(self);

                    // Restore
                    self.stack.truncate(self.fp);
                    self.fp = pop!(self, i64) as usize;
                    self.ip = pop!(self, i64) as usize;
                    insts = self.insts_stack.pop().unwrap();

                    let param_count = pop!(self, i64) as usize;
                    for _ in 0..param_count {
                        self.stack.pop().unwrap();
                    }

                    self.stack.push(return_value);
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
