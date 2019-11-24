use std::collections::HashMap;

use crate::ty::Type;
use crate::inst::{NativeFunction, NativeFunctionBody as FuncBody};
use crate::value::{Value, FromValue};

pub type NativeFuncMap = HashMap<&'static str, NativeFunction>;

fn param<T: FromValue>(arg: &Value) -> T {
    FromValue::from_value(arg.clone())
}

fn printnln(args: &[Value]) -> Value {
    let n: i64 = param(&args[0]);

    println!("{}", n);

    Value::Int(0)
}

fn printn(args: &[Value]) -> Value {
    let n: i64 = param(&args[0]);

    print!("{}", n);

    Value::Int(0)
}

fn print(args: &[Value]) -> Value {
    let s: String = param(&args[0]);

    print!("{}", s);

    Value::Int(0)
}


fn println(args: &[Value]) -> Value {
    let s: String = param(&args[0]);

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
