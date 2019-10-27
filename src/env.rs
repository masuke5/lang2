use crate::ast::Stmt;
use crate::id::Id;

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Int(i64),
    Bool(bool),
}

impl Value {
    pub fn int(&self) -> i64 {
        match self {
            Value::Int(n) => *n,
            _ => panic!(),
        }
    }

    pub fn bool(&self) -> bool {
        match self {
            Value::Bool(v) => *v,
            _ => panic!(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    pub params: Vec<Id>,
    pub stmt: Stmt,
}
