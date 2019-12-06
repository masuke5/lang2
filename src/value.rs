use std::str;
use std::fmt;
use std::slice;
use std::ptr;
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
                region.as_non_null()
            },
        }
    }

    pub unsafe fn expect_to_heap<T>(&self) -> *mut T {
        match self {
            Pointer::ToHeap(ptr) => {
                (*ptr.as_ptr()).as_mut_ptr::<T>()
            },
            _ => panic!("expected to heap"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Unintialized,
    Int(i64),
    Bool(bool),
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

impl_from_value! {Pointer, "expected pointer",
    Value::Pointer(ptr) => ptr,
}

impl_from_value! {Value, "",
    value => value,
}

#[repr(C)]
pub struct Lang2String {
    pub len: usize,
    pub bytes: [u8; 0],
}

impl Lang2String {
    pub unsafe fn write_string(&mut self, s: &String) {
        self.len = s.len();
        ptr::copy_nonoverlapping(s.as_bytes().as_ptr(), self.bytes.as_mut_ptr(), s.len());
    }
}

impl fmt::Display for Lang2String {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = unsafe {
            let bytes = slice::from_raw_parts(self.bytes.as_ptr(), self.len);
            str::from_utf8_unchecked(bytes)
        };

        f.write_str(s)
    }
}
