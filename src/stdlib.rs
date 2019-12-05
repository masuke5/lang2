use std::collections::HashMap;

use crate::ty::Type;
use crate::inst::{NativeFunction, NativeFunctionBody as FuncBody};
use crate::value::Value;
use crate::vm::Context;

pub type NativeFuncMap = HashMap<&'static str, NativeFunction>;

fn printnln(ctx: &mut Context) -> Vec<Value> {
    let n: i64 = ctx.next_param();

    println!("{}", n);

    vec![Value::Int(0)]
}

fn printn(ctx: &mut Context) -> Vec<Value> {
    let n: i64 = ctx.next_param();

    print!("{}", n);

    vec![Value::Int(0)]
}

fn print(ctx: &mut Context) -> Vec<Value> {
    let s: String = ctx.next_param();

    print!("{}", s);

    vec![Value::Int(0)]
}


fn println(ctx: &mut Context) -> Vec<Value> {
    let s: String = ctx.next_param();

    println!("{}", s);

    vec![Value::Int(0)]
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

    funcs.insert("printn", func(vec![Type::Int], Type::Unit, FuncBody(printn)));
    funcs.insert("printnln", func(vec![Type::Int], Type::Unit, FuncBody(printnln)));
    funcs.insert("print", func(vec![Type::String], Type::Unit, FuncBody(print)));
    funcs.insert("println", func(vec![Type::String], Type::Unit, FuncBody(println)));

    funcs
}
