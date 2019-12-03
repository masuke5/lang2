use std::collections::{HashMap, LinkedList};
use std::collections::hash_map::Entry;
use std::ptr::NonNull;
use std::mem;
use std::ffi::c_void;
use libc;
use crate::value::{Value, Pointer};

#[derive(Debug)]
pub struct GcRegion {
    pub base: *mut Value, // TODO: NonNull
    pub size: usize,
    is_marked: bool,
}

impl Drop for GcRegion {
    fn drop(&mut self) {
        unsafe {
            libc::free(self.base as *mut c_void);
        }
    }
}

impl GcRegion {
    fn new(size: usize) -> Self {
        let ptr = unsafe {
            libc::malloc(size * mem::size_of::<Value>())
        };

        Self {
            base: ptr as *mut Value,
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
                    let value = unsafe { ptr.as_mut() };
                    value.is_marked = true;
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
