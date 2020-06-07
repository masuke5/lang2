use crate::ast::SymbolPath;
use crate::id::{reserved_id, IdMap};
use crate::module::{ModuleBuilder, ModuleWithChild};
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
        .define_func("string_len", vec![ltype!(*string)], ltype!(int), string_len)
        .build(SymbolPath::new().append_id(*reserved_id::STD_MODULE))
}
