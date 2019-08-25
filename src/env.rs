use crate::ast::Stmt;

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Int(i64),
}

impl Value {
    pub fn int(&self) -> i64 {
        #[allow(unreachable_patterns)]
        match self {
            Value::Int(n) => *n,
            _ => panic!(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Function<'a> {
    pub params: Vec<&'a str>,
    pub stmt: Stmt<'a>,
}
