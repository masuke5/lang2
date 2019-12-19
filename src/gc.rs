use std::collections::{HashMap, LinkedList};
use std::collections::hash_map::Entry;
use std::mem;
use std::ptr::NonNull;
use std::ffi::c_void;
use libc;
use crate::value::{Value};

#[derive(Debug)]
#[repr(C)]
pub struct GcRegion {
    // first bit: this region is marked or not
    // second bit: this region consists of Value
    bits: u8,
    pub size: usize,
    data: [c_void; 0],
}

impl PartialEq for GcRegion {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size &&
            self.is_marked() == other.is_marked() &&
            self.data.as_ptr() == other.data.as_ptr()
    }
}

impl GcRegion {
    fn new(size: usize, consists_of_value: bool) -> NonNull<Self> {
        let ptr = unsafe {
            let ptr = libc::malloc(mem::size_of::<Self>() + size) as *mut Self;
            (*ptr).size = size;
            (*ptr).bits = if consists_of_value {
                0b01000000
            } else {
                0
            };

            NonNull::new(ptr).unwrap()
        };

        ptr
    }

    fn mark(&mut self) {
        self.bits |= 0b10000000;
    }

    fn unmark(&mut self) {
        self.bits &= !0b10000000;
    }

    fn is_marked(&self) -> bool {
        (self.bits & 0b10000000) != 0
    }

    fn consists_of_value(&self) -> bool {
        (self.bits & 0b01000000) != 0
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.data.as_ptr() as *const c_void as *const T
    }

    pub fn as_mut_ptr<T>(&mut self) -> *mut T {
        self.data.as_ptr() as *mut T
    }

    pub fn as_non_null<T>(&mut self) -> NonNull<T> {
        unsafe { NonNull::new_unchecked(self.data.as_ptr() as *mut _) }
    }
}

pub struct Gc {
    values: LinkedList<NonNull<GcRegion>>,
    freelist_map: HashMap<usize, LinkedList<NonNull<GcRegion>>>,
}

impl Gc {
    pub fn new() -> Self {
        Self {
            values: LinkedList::new(),
            freelist_map: HashMap::with_capacity(16),
        }
    }

    fn pop_region(&mut self, size: usize) -> Option<NonNull<GcRegion>> {
        match self.freelist_map.get_mut(&size) {
            Some(freelist) if !freelist.is_empty() => {
                let region = freelist.pop_front().unwrap();
                Some(region)
            },
            Some(_) | None => None,
        }
    }

    pub fn alloc<T>(&mut self, count: usize, consists_of_value: bool, stack: &mut [Value]) -> NonNull<GcRegion> {
        let size = count * mem::size_of::<T>();

        // Search from freelist
        let region = match self.pop_region(size) {
            Some(region) => region,
            None => {
                self.gc(stack);

                match self.pop_region(size) {
                    Some(region) => region,
                    None => GcRegion::new(size, consists_of_value),
                }
            },
        };

        self.values.push_front(region);

        self.values.front_mut().unwrap().clone()
    }

    fn mark(&mut self, stack: &mut [Value]) {
        fn mark(value: &mut Value) {
            match value {
                Value::PointerToHeap(ptr) => {
                    let region = unsafe { ptr.as_mut() };
                    region.mark();

                    // Regions can consist of string and more
                    if region.consists_of_value() {
                        unsafe {
                            let base = region.as_mut_ptr::<Value>();
                            for i in 0..region.size {
                                mark(&mut *base.add(i));
                            }
                        }
                    }
                },
                _ => {},
            }
        }

        for value in stack {
            mark(value);
        }
    }

    fn sweep(&mut self) {
        let free_regions = self.values.drain_filter(|value| {
            let value = unsafe { value.as_mut() };
            if value.is_marked() {
                value.unmark();
                false
            } else {
                true
            }
        });

        for region_ptr in free_regions {
            let region = unsafe { region_ptr.as_ref() };
            let freelist = match self.freelist_map.entry(region.size) {
                Entry::Occupied(l) => l.into_mut(),
                Entry::Vacant(v) => v.insert(LinkedList::new()),
            };

            freelist.push_front(region_ptr.clone());
        }
    }

    fn gc(&mut self, stack: &mut [Value]) {
        self.mark(stack);
        self.sweep();
    }
}

impl Drop for Gc {
    fn drop(&mut self) {
        fn free(region: &NonNull<GcRegion>){ 
            unsafe { libc::free(region.as_ptr() as *mut _) };
        }

        for region in &self.values {
            free(region);
        }

        for list in self.freelist_map.values() {
            for region in list {
                free(region);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(mut region: NonNull<GcRegion>, values: &[Value]) {
        let region = unsafe { region.as_mut() };
        if values.len() != region.size {
            panic!("value size != region size");
        }
        
        unsafe {
            let ptr = region.as_mut_ptr::<Value>();
            for i in 0..region.size {
                ptr.add(i).write(values[i].clone());
            }
        }
    }

    #[test]
    fn test_gc() {
        let mut gc = Gc::new();

        let region1 = gc.alloc::<Value>(2, true, &mut []);
        write(region1, &[Value::Int(1), Value::Int(2)]);

        let region2 = gc.alloc::<Value>(2, true, &mut [Value::PointerToHeap(region1)]);
        write(region2, &[Value::Int(3), Value::PointerToHeap(region1)]);

        let region3 = gc.alloc::<Value>(1, true, &mut [Value::PointerToHeap(region1), Value::PointerToHeap(region2)]);
        write(region3, &[Value::Int(190419041)]);

        let mut stack = vec![
            Value::Int(5),
            Value::Int(6),
            Value::PointerToHeap(region2),
        ];

        gc.gc(&mut stack);

        unsafe {
            assert_eq!(region3.as_ref(), gc.freelist_map.get_mut(&1).unwrap().pop_front().unwrap().as_ref());
        }
        assert_eq!(None, gc.freelist_map.get(&2));
    }
}
