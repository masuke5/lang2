use std::collections::HashMap;
use std::vec::Drain;

use crate::ty::Type;
use crate::inst::{NativeFunction, NativeFunctionBody as FuncBody};
use crate::value::{Value, FromValue};

pub type NativeFuncMap = HashMap<&'static str, NativeFunction>;

fn param<T: FromValue>(args: &mut Drain<Value>) -> T {
    FromValue::from_value(args.next().unwrap())
}

fn printnln(mut args: Drain<Value>) -> Value {
    let n: i64 = param(&mut args);

    println!("{}", n);

    Value::Int(0)
}

fn printn(mut args: Drain<Value>) -> Value {
    let n: i64 = param(&mut args);

    print!("{}", n);

    Value::Int(0)
}

fn print(mut args: Drain<Value>) -> Value {
    let s: String = param(&mut args);

    print!("{}", s);

    Value::Int(0)
}


fn println(mut args: Drain<Value>) -> Value {
    let s: String = param(&mut args);

    println!("{}", s);

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

    funcs.insert("printn", func(vec![Type::Int], Type::Int, FuncBody(printn)));
    funcs.insert("printnln", func(vec![Type::Int], Type::Int, FuncBody(printnln)));
    funcs.insert("print", func(vec![Type::String], Type::Int, FuncBody(print)));
    funcs.insert("println", func(vec![Type::String], Type::Int, FuncBody(println)));

    funcs
}
