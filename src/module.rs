use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;

use crate::ast::SymbolPath;
use crate::id::{Id, IdMap};
use crate::ty::{Type, TypeCon, TypeVar};

pub const MODULE_EXTENSION: &str = "lang2";
pub const ROOT_MODULE_FILE: &str = "mod.lang2";

#[derive(Debug, Clone)]
pub struct FunctionHeader {
    pub params: Vec<Type>,
    pub return_ty: Type,
    pub ty_params: Vec<(Id, TypeVar)>,
}

#[derive(Debug, Clone)]
pub struct Implementation {
    pub functions: FxHashMap<Id, (usize, FunctionHeader)>,
}

impl Implementation {
    pub fn new() -> Self {
        Self {
            functions: FxHashMap::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModuleHeader {
    pub path: SymbolPath,
    pub functions: FxHashMap<Id, (usize, FunctionHeader)>,
    pub types: FxHashMap<Id, Option<TypeCon>>,
    pub impls: FxHashMap<SymbolPath, Implementation>,
}

impl ModuleHeader {
    pub fn new(path: &SymbolPath) -> Self {
        Self {
            path: path.clone(),
            functions: FxHashMap::default(),
            types: FxHashMap::default(),
            impls: FxHashMap::default(),
        }
    }
}

pub fn find_module_file(root_path: &Path, module_path: &SymbolPath) -> Option<PathBuf> {
    // example: std::collections::vec -> name: vec, dir: std/collection
    let mut module_dir = PathBuf::from(root_path);
    for segment in &module_path.segments {
        module_dir = module_dir.join(&IdMap::name(segment.id));
    }

    let module_name = module_dir
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let module_dir = module_dir.parent().unwrap();

    let path = module_dir.join(&format!("{}.{}", module_name, MODULE_EXTENSION));
    if path.exists() {
        return Some(path);
    }

    let path = module_dir.join(&module_name).join(ROOT_MODULE_FILE);
    if path.exists() {
        return Some(path);
    }

    None
}
