use rustc_hash::FxHashMap;

use crate::id::{Id, IdMap, reserved_id};
use crate::ty::{Type, TypeCon, TypeVar};
use crate::vm::VM;
use crate::module::{Module, ModuleHeader, FunctionHeader, NativeFunctionBody as Body};
use crate::ast::{SymbolPath, SymbolPathSegment};

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

fn func(
    funcs: &mut (&mut Vec<(usize, Body)>, &mut FxHashMap<Id, (u16, FunctionHeader)>),
    name: &'static str,
    ty_params: Vec<(Id, TypeVar)>,
    param_size: usize,
    params: Vec<Type>,
    return_ty: Type,
    body: Body
) {
    let id = IdMap::new_id(name);
    let header = FunctionHeader {
        params,
        return_ty,
        ty_params,
    };
    
    funcs.0.push((param_size, body));
    funcs.1.insert(id, (funcs.0.len() as u16 - 1, header));
}

pub fn module() -> (Module, ModuleHeader) {
    let mut funcs = FxHashMap::default();
    let mut bodies = Vec::new();
    let mut f = (&mut bodies, &mut funcs);

    func(&mut f, "printn", vec![], 1, vec![Type::Int], Type::Unit, Body(printn));
    func(&mut f, "printnln", vec![], 1, vec![Type::Int], Type::Unit, Body(printnln));
    func(&mut f, "print", vec![], 1, vec![Type::App(TypeCon::Pointer(false), vec![Type::String])], Type::Unit, Body(print));
    func(&mut f, "println", vec![], 1, vec![Type::App(TypeCon::Pointer(false), vec![Type::String])], Type::Unit, Body(println));

    let module = Module::Native(bodies);
    let header = ModuleHeader {
        id: SymbolPath::new().append(SymbolPathSegment::new(*reserved_id::STD_MODULE)),
        functions: funcs,
    };

    (module, header)
}
