pub trait FromValue {
    fn from_value(value: &Value) -> Self;
}

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Bool(bool),
    String(String),
    Tuple(Vec<Value>),
    Unintialized,
}

impl FromValue for i64 {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Int(n) => *n,
            _ => panic!("expected int"),
        }
    }
}

impl FromValue for bool {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Bool(b) => *b,
            _ => panic!("expected bool"),
        }
    }
}

impl FromValue for String {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::String(s) => s.clone(),
            _ => panic!("expected string"),
        }
    }
}

impl FromValue for Value {
    fn from_value(value: &Value) -> Value {
        value.clone()
    }
}

