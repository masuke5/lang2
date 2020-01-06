use std::str;
use std::fmt;
use std::slice;
use std::ptr;
use std::mem;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Value(u64);

impl Value {
    pub fn is_heap_ptr(self) -> bool {
        (self.0 & 1) != 0
    }

    pub fn as_u64(self) -> u64 {
        self.0 >> 1
    }

    pub fn as_i64(self) -> i64 {
        let value: i64 = unsafe { mem::transmute(self.0) };
        value >> 1
    }

    pub fn as_ptr<T>(self) -> *mut T {
        let ptr = self.0;
        let ptr = ptr & !1;
        ptr as _
    }

    pub fn new_u64(value: u64) -> Self {
        Self(value << 1)
    }

    pub fn new_i64(value: i64) -> Self {
        let value = value << 1;
        unsafe { mem::transmute(value) }
    }

    pub fn new_ptr<T>(ptr: *const T) -> Self {
        unsafe { Self::from_raw(ptr as u64) }
    }

    pub fn new_ptr_to_heap<T>(ptr: *const T) -> Self {
        let value = ptr as u64;
        let value = value | 1;
        unsafe { Self::from_raw(value) }
    }

    pub unsafe fn from_raw(value: u64) -> Self {
        Self(value)
    }

    pub unsafe fn from_raw_i64(value: i64) -> Self {
        Self(mem::transmute(value))
    }

    pub unsafe fn raw(self) -> u64 {
        self.0
    }

    pub unsafe fn raw_i64(self) -> i64 {
        mem::transmute(self.0)
    }
}

#[repr(C)]
pub struct Lang2String {
    pub len: usize,
    pub bytes: [u8; 0],
}

impl Lang2String {
    #[allow(dead_code)]
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
