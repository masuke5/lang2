use std::ptr::NonNull;
use crate::gc::GcRegion;

pub trait FromValue {
    fn from_value_ref(value: &Value) -> &Self;
    fn from_value(value: Value) -> Self;
}

#[derive(Debug, Clone)]
pub enum Pointer {
    ToStack(NonNull<Value>),
    ToHeap(NonNull<GcRegion>),
}

impl Pointer {
    pub fn as_non_null(&self) -> NonNull<Value> {
        match self {
            Pointer::ToStack(ptr) => *ptr,
            Pointer::ToHeap(mut ptr) => {
                let region = unsafe { ptr.as_mut() };
                region.base
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Unintialized,
    Int(i64),
    Bool(bool),
    String(String),
    Ref(NonNull<Value>),
    Pointer(Pointer),
}

impl Value {
    #[allow(dead_code)]
    pub fn expect_ptr_ref(&self) -> NonNull<Value> {
        match self {
            Value::Pointer(ptr) => ptr.as_non_null(),
            _ => panic!("expected pointer"),
        }
    }

    pub fn expect_ptr(self) -> NonNull<Value> {
        match self {
            Value::Pointer(ptr) => ptr.as_non_null(),
            _ => panic!("expected pointer"),
        }
    }

    // reference to Value::Ref
    #[allow(dead_code)]
    pub fn expect_ref_ref(&self) -> NonNull<Value> {
        match self {
            Value::Ref(ptr) => *ptr,
            _ => panic!("expected ref"),
        }
    }

    pub fn expect_ref(self) -> NonNull<Value> {
        match self {
            Value::Ref(ptr) => ptr,
            _ => panic!("expected ref"),
        }
    }
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

impl_from_value! {Pointer, "expected pointer",
    Value::Pointer(ptr) => ptr,
}

impl_from_value! {Value, "",
    value => value,
}
