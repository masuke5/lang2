use std::fmt;
use std::collections::HashMap;

use crate::vm::VM;
use crate::ty::Type;
use crate::id::Id;

pub struct NativeFunctionBody(pub fn(&mut VM));

impl fmt::Debug for NativeFunctionBody {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[function pointer]")
    }
}

#[derive(Debug)]
pub enum Module {
    Normal,
    Native(Vec<(usize, NativeFunctionBody)>), // parameter size, function pointer
}

#[derive(Debug)]
pub struct FunctionHeader {
    pub params: Vec<Type>,
    pub return_ty: Type,
}

#[derive(Debug)]
pub struct ModuleHeader {
    pub id: Id,
    pub functions: HashMap<Id, (u16, FunctionHeader)>,
}

impl ModuleHeader {
    pub fn find_func(&self, id: Id) -> Option<&(u16, FunctionHeader)> {
        self.functions.get(&id)
    }
}
