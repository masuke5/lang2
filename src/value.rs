pub trait FromValue {
    fn from_value_ref(value: &Value) -> &Self;
    fn from_value(value: Value) -> Self;
}

#[derive(Debug, Clone)]
pub enum Value {
    Unintialized,
    Int(i64),
    Bool(bool),
    String(String),
    Record(Vec<Value>),
}

macro_rules! impl_from_value {
    ($ty:ty, $mess:tt, $($pat:pat => $expr:expr),* $(,)*) => {
        impl FromValue for $ty {
            #[allow(unreachable_patterns)]
            fn from_value_ref(value: &Value) -> &Self {
                match value {
                    $($pat => $expr,)*
                    _ => panic!($mess),
                }
            }

            #[allow(unreachable_patterns)]
            fn from_value(value: Value) -> Self {
                match value {
                    $($pat => $expr,)*
                    _ => panic!($mess),
                }
            }
        }
    }
}

impl_from_value! {i64, "expected int",
    Value::Int(n) => n,
}

impl_from_value! {bool, "expected bool",
    Value::Bool(b) => b,
}

impl_from_value! {String, "expected string",
    Value::String(s) => s,
}

impl_from_value! {Value, "",
    value => value,
}
