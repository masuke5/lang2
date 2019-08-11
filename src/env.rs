#[derive(Debug, PartialEq)]
pub enum Value {
    Int(i64),
}

impl Value {
    pub fn int(&self) -> i64 {
        match self {
            Value::Int(n) => *n,
            _ => panic!(),
        }
    }
}
