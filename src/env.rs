#[derive(Debug, PartialEq)]
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
