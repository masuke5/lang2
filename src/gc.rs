use crate::value::Value;
use std::collections::{hash_map::Entry, HashMap, LinkedList};
use std::ffi::c_void;
use std::fmt;
use std::mem;
use std::ptr::NonNull;

#[repr(C)]
pub struct GcRegion {
    bits: u64,
    pub size: usize,
    data: [c_void; 0],
}

impl fmt::Debug for GcRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GcRegion {{ is_marked: {}, consists_of_value: {}, size: {}, data ({:p}) }}",
            self.is_marked(),
            self.consists_of_value(),
            self.size,
            self.data.as_ptr(),
        )
    }
}

impl PartialEq for GcRegion {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
            && self.is_marked() == other.is_marked()
            && self.data.as_ptr() == other.data.as_ptr()
    }
}

impl Drop for GcRegion {
    fn drop(&mut self) {
        unsafe {
            let self_ptr: *mut _ = self;
            libc::free(self_ptr as *mut c_void);
        }
    }
}

impl GcRegion {
    const IS_MARKED: u64 = 1;
    const CONSISTS_OF_VALUE: u64 = 1 << 1;

    fn new(size: usize, consists_of_value: bool) -> NonNull<Self> {
        unsafe {
            let ptr = libc::calloc(1, mem::size_of::<Self>() + size) as *mut Self;
            (*ptr).size = size;
            (*ptr).bits = if consists_of_value {
                Self::CONSISTS_OF_VALUE
            } else {
                0
            };

            NonNull::new(ptr).unwrap()
        }
    }

    fn mark(&mut self) {
        self.bits |= Self::IS_MARKED;
    }

    fn unmark(&mut self) {
        self.bits &= !Self::IS_MARKED;
    }

    fn is_marked(&self) -> bool {
        (self.bits & Self::IS_MARKED) != 0
    }

    fn consists_of_value(&self) -> bool {
        (self.bits & Self::CONSISTS_OF_VALUE) != 0
    }

    #[allow(dead_code)]
    pub fn as_ptr<T>(&self) -> *const T {
        let data_ptr: *const _ = self.data.as_ptr();
        data_ptr as *const T
    }

    pub fn as_mut_ptr<T>(&mut self) -> *mut T {
        self.data.as_ptr() as *mut T
    }

    #[allow(dead_code)]
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
            }
            Some(_) | None => None,
        }
    }

    pub fn alloc<T>(
        &mut self,
        count: usize,
        consists_of_value: bool,
        stack: &mut [Value],
    ) -> NonNull<GcRegion> {
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
            }
        };

        // Push the region and return a reference to it
        self.values.push_front(region);
        *self.values.front_mut().unwrap()
    }

    fn mark(&mut self, stack: &mut [Value]) {
        fn mark(value: &mut Value) {
            if value.is_heap_ptr() {
                let ptr = value.as_ptr::<GcRegion>();
                let region = unsafe { &mut *ptr.sub(1) };

                if !region.is_marked() {
                    region.mark();

                    // Regions can consist of string and more
                    if region.consists_of_value() {
                        unsafe {
                            let base = region.as_mut_ptr::<Value>();
                            let field_count = region.size / mem::size_of::<Value>();
                            for i in 0..field_count {
                                mark(&mut *base.add(i));
                            }
                        }
                    }
                }
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

        for mut region_ptr in free_regions {
            let region = unsafe { region_ptr.as_mut() };

            #[cfg(feature = "vmdebug")]
            {
                // Zero clear before add to freelist
                unsafe {
                    let p = region.as_mut_ptr::<u8>();
                    p.write_bytes(0, region.size);
                }
            }

            let freelist = match self.freelist_map.entry(region.size) {
                Entry::Occupied(l) => l.into_mut(),
                Entry::Vacant(v) => v.insert(LinkedList::new()),
            };

            freelist.push_front(region_ptr);
        }
    }

    fn gc(&mut self, stack: &mut [Value]) {
        self.mark(stack);
        self.sweep();
    }
}

impl Drop for Gc {
    fn drop(&mut self) {
        fn free(region: NonNull<GcRegion>) {
            unsafe { libc::free(region.as_ptr() as *mut _) };
        }

        for region in &self.values {
            free(*region);
        }

        for list in self.freelist_map.values() {
            for region in list {
                free(*region);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    fn write(mut region: NonNull<GcRegion>, values: &[Value]) {
        let region = unsafe { region.as_mut() };
        if size_of::<Value>() * values.len() != region.size {
            panic!("value size != region size");
        }

        unsafe {
            let ptr = region.as_mut_ptr::<Value>();
            for i in 0..values.len() {
                ptr.add(i).write(values[i].clone());
            }
        }
    }

    #[test]
    fn test_gc() {
        unsafe {
            let mut gc = Gc::new();

            let mut region1 = gc.alloc::<Value>(2, true, &mut []);
            write(region1, &[Value::new_i64(1), Value::new_i64(2)]);

            let mut region2 =
                gc.alloc::<Value>(2, true, &mut [Value::new_ptr_to_heap(region1.as_mut())]);
            write(
                region2,
                &[Value::new_i64(3), Value::new_ptr_to_heap(region1.as_mut())],
            );

            let region3 = gc.alloc::<Value>(
                1,
                true,
                &mut [
                    Value::new_ptr_to_heap(region1.as_mut()),
                    Value::new_ptr_to_heap(region2.as_mut()),
                ],
            );
            write(region3, &[Value::new_i64(190419041)]);

            let mut stack = vec![
                Value::new_i64(5),
                Value::new_i64(6),
                Value::new_ptr_to_heap(region2.as_mut()),
            ];

            gc.gc(&mut stack);

            assert_eq!(
                region3.as_ref(),
                gc.freelist_map
                    .get_mut(&8)
                    .unwrap()
                    .pop_front()
                    .unwrap()
                    .as_ref()
            );
            assert_eq!(None, gc.freelist_map.get(&2));
        }
    }
}
