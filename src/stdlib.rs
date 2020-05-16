use crate::ast::SymbolPath;
use crate::id::{reserved_id, IdMap};
use crate::module::{ModuleBuilder, ModuleWithChild};
use crate::ty::{Type, TypeCon, TypeVar};
use crate::value::{Slice, Value};
use crate::vm::VM;

fn printnln(vm: &mut VM) {
    let n = vm.get_value(vm.arg_loc(0, 1)).as_i64();

    println!("{}", n);
}

fn printn(vm: &mut VM) {
    let n = vm.get_value(vm.arg_loc(0, 1)).as_i64();

    print!("{}", n);
}

fn print(vm: &mut VM) {
    let s = vm.get_string(vm.arg_loc(0, 1));

    print!("{}", s);
}

fn println(vm: &mut VM) {
    let s = vm.get_string(vm.arg_loc(0, 1));

    println!("{}", s);
}

fn len(vm: &mut VM) {
    let slice: *const Slice = vm.get_value(vm.arg_loc(0, 1)).as_ptr();
    let len = unsafe {
        let start = (*slice).start.as_i64();
        let end = (*slice).end.as_i64();
        end - start
    };

    vm.write_return_value(&[Value::new_i64(len)], 1);
}

pub fn module() -> ModuleWithChild {
    let mut b = ModuleBuilder::new();

    b.define_func("printn", vec![ltype!(int)], ltype!(unit), printn);
    b.define_func("printnln", vec![ltype!(int)], ltype!(unit), printnln);
    b.define_func("print", vec![ltype!(*string)], ltype!(unit), print);
    b.define_func("println", vec![ltype!(*string)], ltype!(unit), println);

    {
        let var = TypeVar::new();
        b.define_func_poly(
            "len",
            vec![(IdMap::new_id("T"), var)],
            vec![Type::App(TypeCon::Slice(false), vec![Type::Var(var)])],
            Type::Int,
            len,
        );
    }

    b.build(SymbolPath::new().append_id(*reserved_id::STD_MODULE))
}
