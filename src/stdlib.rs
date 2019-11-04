use std::collections::HashMap;

use crate::ty::Type;
use crate::inst::{NativeFunction, NativeFunctionBody as FuncBody};
use crate::value::{Value, FromValue};

pub type NativeFuncMap = HashMap<&'static str, NativeFunction>;

fn printi(args: &[Value]) -> Value {
    let n: i64 = FromValue::from_value(&args[0]);

    println!("{}", n);

    Value::Int(0)
}

fn func(params: Vec<Type>, return_ty: Type, body: FuncBody) -> NativeFunction {
    NativeFunction {
        params,
        return_ty,
        body,
    }
}

pub fn functions() -> NativeFuncMap {
    let mut funcs = HashMap::new();

    funcs.insert("printi", func(vec![Type::Int], Type::Int, FuncBody(printi)));

    funcs
}
