use std::collections::{HashMap, LinkedList};
use std::collections::hash_map::Entry;
use std::ptr::NonNull;
use std::mem;
use std::ffi::c_void;
use libc;
use crate::value::{Value, Pointer};

#[derive(Debug, PartialEq)]
pub struct GcRegion {
    pub base: NonNull<Value>,
    pub size: usize,
    is_marked: bool,
}

impl Drop for GcRegion {
    fn drop(&mut self) {
        unsafe {
            libc::free(self.base.as_ptr() as *mut c_void);
        }
    }
}

impl GcRegion {
    fn new(size: usize) -> Self {
        let ptr = unsafe {
            let ptr = libc::malloc(size * mem::size_of::<Value>());
            NonNull::new(ptr as *mut Value).unwrap()
        };

        Self {
            base: ptr,
            size,
            is_marked: false,
        }
    }
}

pub struct Gc {
    values: LinkedList<GcRegion>,
    freelist_map: HashMap<usize, LinkedList<GcRegion>>,
}

impl Gc {
    pub fn new() -> Self {
        Self {
            values: LinkedList::new(),
            freelist_map: HashMap::with_capacity(16),
        }
    }

    fn pop_region(&mut self, size: usize) -> Option<GcRegion> {
        match self.freelist_map.get_mut(&size) {
            Some(freelist) if !freelist.is_empty() => {
                let region = freelist.pop_front().unwrap();
                Some(region)
            },
            Some(_) | None => None,
        }
    }

    pub fn alloc(&mut self, size: usize, stack: &mut [Value]) -> NonNull<GcRegion> {
        // Search from freelist
        let region = match self.pop_region(size) {
            Some(region) => region,
            None => {
                self.gc(stack);

                match self.pop_region(size) {
                    Some(region) => region,
                    None => GcRegion::new(size),
                }
            },
        };

        self.values.push_front(region);

        let ptr = self.values.front_mut().unwrap() as *mut _;
        unsafe { NonNull::new_unchecked(ptr) }
    }

    fn mark(&mut self, stack: &mut [Value]) {
        fn mark(value: &mut Value) {
            match value {
                Value::Pointer(Pointer::ToHeap(ptr)) => {
                    let region = unsafe { ptr.as_mut() };
                    region.is_marked = true;

                    unsafe {
                        let base = region.base.as_ptr();
                        for i in 0..region.size {
                            mark(&mut *base.add(i));
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
            if value.is_marked {
                value.is_marked = false;
                false
            } else {
                true
            }
        });

        for region in free_regions {
            let freelist = match self.freelist_map.entry(region.size) {
                Entry::Occupied(l) => l.into_mut(),
                Entry::Vacant(v) => v.insert(LinkedList::new()),
            };

            freelist.push_front(region);
        }
    }

    fn gc(&mut self, stack: &mut [Value]) {
        self.mark(stack);
        self.sweep();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(region: NonNull<GcRegion>, values: &[Value]) {
        let region = unsafe { region.as_ref() };
        if values.len() != region.size {
            panic!("value size != region size");
        }
        
        unsafe {
            let ptr = region.base.as_ptr();
            for i in 0..region.size {
                ptr.add(i).write(values[i].clone());
            }
        }
    }

    #[test]
    fn test_gc() {
        let mut gc = Gc::new();

        let region1 = gc.alloc(2, &mut []);
        write(region1, &[Value::Int(1), Value::Int(2)]);

        let region2 = gc.alloc(2, &mut [Value::Pointer(Pointer::ToHeap(region1))]);
        write(region2, &[Value::Int(3), Value::Pointer(Pointer::ToHeap(region1))]);

        let region3 = gc.alloc(1, &mut [Value::Pointer(Pointer::ToHeap(region1)), Value::Pointer(Pointer::ToHeap(region2))]);
        write(region3, &[Value::Int(190419041)]);

        let mut stack = vec![
            Value::Int(5),
            Value::Int(6),
            Value::Pointer(Pointer::ToHeap(region2)),
        ];

        gc.gc(&mut stack);

        assert_eq!(unsafe { region3.as_ref() }, &gc.freelist_map.get_mut(&1).unwrap().pop_front().unwrap());
        assert_eq!(None, gc.freelist_map.get(&2));
    }
}
