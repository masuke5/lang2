use crate::ast::SymbolPath;
use crate::id::{reserved_id, IdMap};
use crate::module::{ImplementationBuilder, ModuleBuilder, ModuleWithChild};
use crate::ty::{type_size_nocheck, Type, TypeCon, TypeVar};
use crate::value::{FromValue, Lang2String, Slice, ToType, Value};
use crate::vm::VM;

fn printnln(vm: &mut VM) {
    get_args!(vm, n: i64);

    println!("{}", n);
}

fn printn(vm: &mut VM) {
    get_args!(vm, n: i64);

    print!("{}", n);
}

fn print(vm: &mut VM) {
    get_args!(vm, s: *const Lang2String);

    print!("{}", unsafe { &*s });
}

fn println(vm: &mut VM) {
    get_args!(vm, s: *const Lang2String);

    println!("{}", unsafe { &*s });
}

fn len(vm: &mut VM) {
    get_args!(vm, slice: *const Slice);

    let len = unsafe {
        let start = (*slice).start.as_i64();
        let end = (*slice).end.as_i64();
        end - start
    };

    vm.return_values(&[Value::new_i64(len)], 1);
}

fn string_len(vm: &mut VM) {
    get_args!(vm, s: *const Lang2String);

    let len = unsafe { (*s).len };
    vm.return_values(&[Value::new_i64(len as i64)], 1);
}

fn string_index(vm: &mut VM) {
    get_args!(vm, index: i64, s: *const Lang2String);

    // Check bounds
    let len = unsafe { (*s).len };
    if index < 0 || index >= len as i64 {
        vm.panic("out of bounds");
    }

    // Return character as new string
    let ch = unsafe { *(*s).bytes.as_ptr().add(index as usize) as char };
    let s = vm.alloc_str(&ch.to_string());

    vm.return_values(&[Value::new_ptr_to_heap(s)], 2);
}

fn string_concat(vm: &mut VM) {
    get_args!(vm, a: *const Lang2String, b: *const Lang2String);

    let a = unsafe { format!("{}", (*a)) };
    let b = unsafe { format!("{}", (*b)) };
    let c = a + &b;

    let s = vm.alloc_str(&c);

    vm.return_values(&[Value::new_ptr_to_heap(s)], 2);
}

fn string_sub(vm: &mut VM) {
    get_args!(vm, start: i64, len: i64, s: *const Lang2String);

    // Check bounds
    let s_len = unsafe { (*s).len };
    if start < 0 || start >= s_len as i64 || start + len < 0 || start + len >= s_len as i64 {
        vm.panic("out of bounds");
    }

    let start = start as usize;
    let len = len as usize;

    let s = unsafe { format!("{}", *s) };
    let s = &s[start..start + len];
    let s = vm.alloc_str(s);

    vm.return_values(&[Value::new_ptr_to_heap(s)], 3);
}

fn string_first_byte(vm: &mut VM) {
    get_args!(vm, s: *const Lang2String);

    // Check length
    let len = unsafe { (*s).len };
    if len < 1 {
        vm.panic("empty string");
    }

    let byte = unsafe { *(*s).bytes.as_ptr() };
    vm.return_values(&[Value::new_i64(byte as i64)], 1);
}

pub fn module() -> ModuleWithChild {
    let var = TypeVar::new();

    ModuleBuilder::new()
        .define_func("printn", vec![ltype!(int)], ltype!(unit), printn)
        .define_func("printnln", vec![ltype!(int)], ltype!(unit), printnln)
        .define_func("print", vec![ltype!(*string)], ltype!(unit), print)
        .define_func("println", vec![ltype!(*string)], ltype!(unit), println)
        .define_func_poly(
            "len",
            vec![(IdMap::new_id("T"), var)],
            vec![Type::App(TypeCon::Slice(false), vec![Type::Var(var)])],
            Type::Int,
            len,
        )
        .implmentation(
            ImplementationBuilder::new(SymbolPath::new().append_str("std").append_str("String"))
                .define_func("len", vec![ltype!(*string)], ltype!(int), string_len)
                .define_func(
                    "index",
                    vec![ltype!(int), ltype!(*string)],
                    ltype!(*string),
                    string_index,
                )
                .define_func(
                    "concat",
                    vec![ltype!(*string), ltype!(*string)],
                    ltype!(*string),
                    string_concat,
                )
                .define_func(
                    "sub",
                    vec![ltype!(int), ltype!(int), ltype!(*string)],
                    ltype!(*string),
                    string_sub,
                )
                .define_func(
                    "first_byte",
                    vec![ltype!(*string)],
                    ltype!(int),
                    string_first_byte,
                ),
        )
        .build(SymbolPath::new().append_id(*reserved_id::STD_MODULE))
}
