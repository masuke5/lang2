use crate::ast::SymbolPath;
use crate::id::reserved_id;
use crate::module::{ModuleBuilder, ModuleWithChild};
use crate::ty::{Type, TypeCon};
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

pub fn module() -> ModuleWithChild {
    let mut b = ModuleBuilder::new();

    b.define_func("printn", vec![ltype!(int)], ltype!(unit), printn);
    b.define_func("printnln", vec![ltype!(int)], ltype!(unit), printnln);
    b.define_func("print", vec![ltype!(*string)], ltype!(unit), print);
    b.define_func("println", vec![ltype!(*string)], ltype!(unit), println);

    b.build(SymbolPath::new().append_id(*reserved_id::STD_MODULE))
}
