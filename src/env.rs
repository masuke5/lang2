use crate::ast::Stmt;

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
pub struct Function<'a> {
    pub params: Vec<&'a str>,
    pub stmt: Stmt<'a>,
}
