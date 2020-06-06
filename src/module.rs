use std::fmt;
use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;

use crate::ast::SymbolPath;
use crate::id::{Id, IdMap};
use crate::ty::{type_size, Type, TypeCon, TypeVar};
use crate::vm::VM;

pub const MODULE_EXTENSION: &str = "lang2";
pub const ROOT_MODULE_FILE: &str = "mod.lang2";

#[derive(Clone)]
pub struct NativeFunctionBody(pub fn(&mut VM));

impl fmt::Debug for NativeFunctionBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[function pointer]")
    }
}

#[derive(Debug, Clone)]
pub enum Module {
    Normal,
    Native(Vec<(usize, NativeFunctionBody)>), // parameter size, function pointer
}

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
    pub impls: FxHashMap<Id, Implementation>,
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

#[derive(Debug)]
pub struct ModuleWithChild {
    pub module: Module,
    pub header: ModuleHeader,
    pub child_modules: FxHashMap<Id, ModuleWithChild>,
}

#[derive(Debug)]
pub struct ModuleContainer {
    modules: FxHashMap<Id, ModuleWithChild>,
}

impl ModuleContainer {
    pub fn new() -> Self {
        Self {
            modules: FxHashMap::default(),
        }
    }

    pub fn add(&mut self, id: Id, module: ModuleWithChild) {
        self.modules.insert(id, module);
    }

    pub fn contains(&self, path: &SymbolPath) -> bool {
        self.get(path).is_some()
    }

    pub fn get(&self, path: &SymbolPath) -> Option<&ModuleWithChild> {
        let mut iter = path.segments.iter();

        let mut module = self.modules.get(&iter.next()?.id)?;
        for segment in iter {
            module = module.child_modules.get(&segment.id)?;
        }

        Some(&module)
    }
}

#[derive(Debug)]
pub struct ImplementationBuilder {
    func_bodies: Vec<(usize, NativeFunctionBody)>,
    func_headers: FxHashMap<Id, (usize, FunctionHeader)>,
}

#[derive(Debug)]
pub struct ModuleBuilder {
    func_bodies: Vec<(usize, NativeFunctionBody)>,
    func_headers: FxHashMap<Id, (usize, FunctionHeader)>,
    impls: Vec<Implementation>,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self {
            func_bodies: Vec::new(),
            func_headers: FxHashMap::default(),
            impls: Vec::new(),
        }
    }

    pub fn define_func(
        &mut self,
        name: &str,
        params: Vec<Type>,
        return_ty: Type,
        body: fn(&mut VM),
    ) {
        self.define_func_poly(name, Vec::new(), params, return_ty, body);
    }

    pub fn define_func_poly(
        &mut self,
        name: &str,
        ty_params: Vec<(Id, TypeVar)>,
        params: Vec<Type>,
        return_ty: Type,
        body: fn(&mut VM),
    ) {
        let name = IdMap::new_id(name);
        let param_size = params.iter().fold(0, |size, ty| {
            size + type_size(ty).expect("Param size couldn't be calculated")
        });

        self.func_bodies
            .push((param_size, NativeFunctionBody(body)));
        self.func_headers.insert(
            name,
            (
                self.func_headers.len(),
                FunctionHeader {
                    params,
                    return_ty,
                    ty_params,
                },
            ),
        );
    }

    pub fn build(self, path: SymbolPath) -> ModuleWithChild {
        let module = Module::Native(self.func_bodies);
        let header = ModuleHeader {
            path,
            functions: self.func_headers,
            types: FxHashMap::default(),
            impls: FxHashMap::default(),
        };

        ModuleWithChild {
            module,
            header,
            child_modules: FxHashMap::default(),
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
