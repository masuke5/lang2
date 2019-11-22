pub trait FromValue {
    fn from_value_ref(value: &Value) -> &Self;
    fn from_value(value: Value) -> Self;
}

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Bool(bool),
    String(String),
    Record(Vec<Value>),
    Unintialized,
}

macro_rules! impl_from_value {
    ($ty:ty, $($pat:pat => $expr:expr),* $(,)*) => {
        impl FromValue for $ty {
            #[allow(unreachable_patterns)]
            fn from_value_ref(value: &Value) -> &Self {
                match value {
                    $($pat => $expr,)*
                    _ => panic!(),
                }
            }

            #[allow(unreachable_patterns)]
            fn from_value(value: Value) -> Self {
                match value {
                    $($pat => $expr,)*
                    _ => panic!(),
                }
            }
        }
    }
}

impl_from_value! {i64,
    Value::Int(n) => n,
}

impl_from_value! {bool,
    Value::Bool(b) => b,
}

impl_from_value! {String,
    Value::String(s) => s,
}

impl_from_value! {Value,
    value => value,
}
