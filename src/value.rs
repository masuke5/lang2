use std::fmt;
use std::mem;
use std::ptr;
use std::slice;
use std::str;

use crate::ty::{Type, TypeCon};

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub enum Value {
    Int(u64),
    Float(f64),
    PointerToHeap(*mut Value),
    Pointer(*mut Value),
}

impl Value {
    #[inline(always)]
    pub fn is_heap_ptr(&self) -> bool {
        match self {
            Self::PointerToHeap(..) => true,
            _ => false,
        }
    }

    #[inline(always)]
    pub fn is_true(&self) -> bool {
        !self.is_false()
    }

    #[inline(always)]
    pub fn is_false(&self) -> bool {
        match self {
            Self::Int(n) if *n == 0 => true,
            _ => false,
        }
    }

    #[inline(always)]
    pub fn as_u64(&self) -> u64 {
        match self {
            Self::Int(n) => *n,
            _ => panic!("non-integer value"),
        }
    }

    #[inline(always)]
    pub fn as_i64(&self) -> i64 {
        match self {
            Self::Int(n) => unsafe { mem::transmute(*n) },
            _ => panic!("non-integer value"),
        }
    }

    #[inline(always)]
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Float(n) => *n,
            _ => panic!("non-floating point value"),
        }
    }

    #[inline(always)]
    pub fn as_ptr<T>(&self) -> *mut T {
        match self {
            Self::Pointer(ptr) | Self::PointerToHeap(ptr) => *ptr as *mut _,
            _ => panic!("non-pointer value"),
        }
    }

    #[allow(dead_code)]
    pub const fn zero() -> Self {
        Self::Int(0)
    }

    #[inline(always)]
    pub fn new_u64(value: u64) -> Self {
        Self::Int(value)
    }

    #[inline(always)]
    pub fn new_i64(value: i64) -> Self {
        Self::Int(unsafe { mem::transmute(value) })
    }

    #[inline(always)]
    pub fn new_f64(value: f64) -> Self {
        Self::Float(value)
    }

    #[inline(always)]
    pub fn new_ptr<T>(ptr: *const T) -> Self {
        Self::Pointer(ptr as *mut _)
    }

    #[inline(always)]
    pub fn new_ptr_to_heap<T>(ptr: *const T) -> Self {
        Self::PointerToHeap(ptr as *mut _)
    }

    #[inline(always)]
    pub fn new_bool(b: bool) -> Self {
        if b {
            Self::new_true()
        } else {
            Self::new_false()
        }
    }

    #[inline(always)]
    pub fn new_true() -> Self {
        Self::Int(1)
    }

    #[inline(always)]
    pub fn new_false() -> Self {
        Self::Int(0)
    }
}

pub trait FromValue {
    fn from_value(value: Value) -> Self;
}

impl FromValue for Value {
    fn from_value(value: Value) -> Self {
        value
    }
}

impl<V: FromValue> FromValue for *mut V {
    fn from_value(value: Value) -> Self {
        value.as_ptr()
    }
}

impl<V: FromValue> FromValue for *const V {
    fn from_value(value: Value) -> Self {
        value.as_ptr()
    }
}

impl FromValue for i64 {
    fn from_value(value: Value) -> Self {
        value.as_i64()
    }
}

impl FromValue for Lang2String {
    fn from_value(_: Value) -> Self {
        panic!();
    }
}

impl FromValue for Slice {
    fn from_value(_: Value) -> Self {
        panic!();
    }
}

pub trait ToType {
    fn to_type() -> Type;
}

impl ToType for i64 {
    fn to_type() -> Type {
        Type::Int
    }
}

impl<T: ToType> ToType for *const T {
    fn to_type() -> Type {
        Type::App(TypeCon::Pointer(false), vec![<T as ToType>::to_type()])
    }
}

impl<T: ToType> ToType for *mut T {
    fn to_type() -> Type {
        Type::App(TypeCon::Pointer(true), vec![<T as ToType>::to_type()])
    }
}

impl ToType for Slice {
    fn to_type() -> Type {
        Type::App(TypeCon::Slice(false), vec![Type::Int])
    }
}

impl ToType for Lang2String {
    fn to_type() -> Type {
        Type::String
    }
}

pub struct Lang2Str {
    len: u64,
    bytes: *const u8,
}

impl Lang2Str {
    pub unsafe fn from_bytes_ptr(ptr: *const u8) -> Self {
        // Check if ptr is aligned
        if (ptr as usize) % 8 != 0 {
            panic!("the pointer is not aligned");
        }

        #[allow(clippy::cast_ptr_alignment)]
        let len = *(ptr as *const u64);

        // Check if is valid UTF-8 string
        let bytes_ptr = ptr.add(mem::size_of::<u64>());
        let bytes = slice::from_raw_parts(bytes_ptr, len as usize);
        str::from_utf8(bytes).unwrap();

        Self {
            len,
            bytes: bytes_ptr,
        }
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn as_str(&self) -> &str {
        unsafe {
            let bytes = slice::from_raw_parts(self.bytes, self.len as usize);
            str::from_utf8_unchecked(bytes)
        }
    }
}

#[repr(C)]
pub struct Lang2String {
    pub len: usize,
    pub bytes: [u8; 0],
}

impl Lang2String {
    #[allow(dead_code)]
    pub unsafe fn write_string(&mut self, s: &str) {
        self.len = s.len();
        ptr::copy_nonoverlapping(s.as_bytes().as_ptr(), self.bytes.as_mut_ptr(), s.len());
    }
}

impl fmt::Display for Lang2String {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = unsafe {
            let bytes = slice::from_raw_parts(self.bytes.as_ptr(), self.len);
            str::from_utf8(bytes).unwrap()
        };

        f.write_str(s)
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Slice {
    pub values: *mut Value,
    pub start: Value,
    pub end: Value,
}
