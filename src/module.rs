use std::fmt;
use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;

use crate::vm::VM;
use crate::ty::{Type, TypeVar};
use crate::id::Id;

pub const MODULE_EXTENSION: &str = "lang2";
pub const ROOT_MODULE_FILE: &str = "mod.lang2";

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
    pub ty_params: Vec<(Id, TypeVar)>,
}

#[derive(Debug)]
pub struct ModuleHeader {
    pub id: Id,
    pub functions: FxHashMap<Id, (u16, FunctionHeader)>,
}

impl ModuleHeader {
    pub fn find_func(&self, id: Id) -> Option<&(u16, FunctionHeader)> {
        self.functions.get(&id)
    }
}

pub fn find_module_file(current_module: &Path, module_name: &str) -> Option<PathBuf> {
    let current_dir = current_module.parent().unwrap();

    let path = current_dir.join(&format!("{}.{}", module_name, MODULE_EXTENSION));
    if path.exists() {
        return Some(path);
    }

    let path = current_dir.join(module_name).join(ROOT_MODULE_FILE);
    if path.exists() {
        return Some(path);
    }

    None
}
