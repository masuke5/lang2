use std::fmt;
use std::mem;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

use lazy_static::lazy_static;
use rustc_hash::FxHashMap;

use crate::error::Error;
use crate::id::{Id, IdMap};
use crate::span::Span;
use crate::utils::{format_bool, HashMapWithScope};

macro_rules! _ltype_arrow {
    ($arg:expr, -> $($ret:tt)*) => {
        Type::App(TypeCon::Arrow, vec![$arg, ltype!($($ret)*)])
    };
    ($arg:expr,) => { $arg };
}

macro_rules! _ltype_term {
    (int $($rest:tt)*) => { _ltype_arrow!(Type::Int, $($rest)*) };
    (unit $($rest:tt)*) => { _ltype_arrow!(Type::Unit, $($rest)*) };
    (bool $($rest:tt)*) => { _ltype_arrow!(Type::Bool, $($rest)*) };
    (string $($rest:tt)*) => { _ltype_arrow!(Type::String, $($rest)*) };
    () => { compile_error!("invalid type name") }
}

macro_rules! _ltype_pointer {
    (*mut $($rest:tt)*) => { Type::App(TypeCon::Pointer(true), vec![_ltype_pointer!($($rest)*)]) };
    (*$($rest:tt)*) => { Type::App(TypeCon::Pointer(false), vec![_ltype_pointer!($($rest)*)]) };
    ($($ty:tt)*) => { _ltype_term!($($ty)*) }
}

macro_rules! ltype {
    ($($ty:tt)*) => {
        _ltype_pointer!($($ty)*)
    };
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct TypeVar(u32);

lazy_static! {
    static ref VAR_ID_MAP: RwLock<FxHashMap<TypeVar, Id>> = { RwLock::new(FxHashMap::default()) };
}

static NEXT_VAR: AtomicU32 = AtomicU32::new(0);

impl TypeVar {
    pub fn new() -> Self {
        let var = Self(NEXT_VAR.load(Ordering::Acquire));
        NEXT_VAR.fetch_add(1, Ordering::Acquire);
        var
    }

    pub fn with_id(id: Id) -> Self {
        let var = Self(NEXT_VAR.load(Ordering::Acquire));
        NEXT_VAR.fetch_add(1, Ordering::Acquire);

        let mut var_id_map = VAR_ID_MAP.write().expect("STR_MAP poisoned");
        var_id_map.insert(var, id);

        var
    }
}

impl fmt::Display for TypeVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let var_id_map = VAR_ID_MAP.read().expect("STR_MAP poisoned");
        let id = var_id_map.get(self);

        match id {
            Some(id) if cfg!(debug_assertions) => write!(f, "{}{{{}}}", IdMap::name(*id), self.0),
            Some(id) if !cfg!(debug_assertions) => write!(f, "{}", IdMap::name(*id)),
            _ => write!(f, "'{}", self.0),
        }
    }
}

#[derive(PartialEq, Clone)]
pub enum Type {
    Int,
    Bool,
    String,
    Unit,
    Null,
    App(TypeCon, Vec<Type>),
    Var(TypeVar),
    Poly(Vec<TypeVar>, Box<Type>),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int => write!(f, "int"),
            Self::Bool => write!(f, "bool"),
            Self::String => write!(f, "string"),
            Self::Unit => write!(f, "()"),
            Self::Null => write!(f, "null"),
            Self::App(TypeCon::Pointer(is_mutable), types) => {
                write!(f, "*{}{}", format_bool(*is_mutable, "mut "), types[0])
            }
            Self::App(TypeCon::Tuple, types) => {
                write!(f, "(")?;
                write_iter!(f, types.iter())?;
                write!(f, ")")
            }
            Self::App(TypeCon::Struct(fields), types) => {
                write!(f, "{{")?;
                write_iter!(
                    f,
                    fields.iter().zip(types.iter()).map(|(name, ty)| format!(
                        "{}: {}",
                        IdMap::name(*name),
                        ty
                    ))
                )?;
                write!(f, "}}")
            }
            Self::App(TypeCon::Arrow, types) => write!(f, "{} -> {}", types[0], types[1]),
            Self::App(TypeCon::Array(size), types) => write!(f, "[{}; {}]", types[0], size),
            #[cfg(debug_assertions)]
            Self::App(TypeCon::Unique(tycon, uniq), types) => {
                write!(f, "{} u{}", Type::App(*tycon.clone(), types.clone()), uniq)
            }
            #[cfg(not(debug_assertions))]
            Self::App(TypeCon::Unique(tycon, _), types) => {
                write!(f, "{}", Type::App(*tycon.clone(), types.clone()))
            }
            Self::App(TypeCon::Wrapped, types) => write!(f, "'{}", types[0]),
            #[cfg(not(debug_assertions))]
            Self::App(TypeCon::Named(name, _), types)
            | Self::App(TypeCon::UnsizedNamed(name), types) => {
                write!(f, "{}", IdMap::name(*name))?;
                if !types.is_empty() {
                    write!(f, "<")?;
                    write_iter!(f, types.iter())?;
                    write!(f, ">")?;
                }

                Ok(())
            }
            Self::App(tycon, tys) => {
                write!(f, "{}<", tycon)?;
                write_iter!(f, tys.iter())?;
                write!(f, ">")
            }
            Self::Var(var) => write!(f, "{}", var),
            Self::Poly(vars, ty) => {
                if !cfg!(debug_assertions) {
                    write!(f, "{}", ty)
                } else {
                    write!(f, "(")?;
                    write_iter!(f, vars.iter())?;
                    write!(f, "): {}", ty)
                }
            }
        }
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Type({})", self)
    }
}

#[derive(PartialEq, Clone)]
pub enum TypeCon {
    Pointer(bool),
    Tuple,
    Arrow,
    Struct(Vec<Id>),
    Array(usize),
    Fun(Vec<TypeVar>, Box<Type>),
    Unique(Box<TypeCon>, u32),
    Named(Id, usize),
    UnsizedNamed(Id),
    Wrapped,
}

impl fmt::Display for TypeCon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pointer(is_mutable) => write!(f, "{}pointer", format_bool(*is_mutable, "mut ")),
            Self::Tuple => write!(f, "tuple"),
            Self::Arrow => write!(f, "arrow"),
            Self::Struct(fields) => {
                write!(f, "{{")?;
                write_iter!(f, fields.iter().map(|id| IdMap::name(*id)))?;
                write!(f, "}}")
            }
            Self::Array(size) => write!(f, "array({})", size),
            Self::Fun(params, body) => {
                write!(f, "fun(")?;
                write_iter!(f, params.iter().map(|a| format!("{}", a)))?;
                write!(f, ") = {}", body)
            }
            Self::Unique(tycon, uniq) => write!(f, "unique({}){{{}}}", tycon, uniq),
            Self::Named(name, size) => write!(f, "{}{{size={}}}", IdMap::name(*name), size),
            Self::UnsizedNamed(name) => write!(f, "{}{{size=?}} ", IdMap::name(*name)),
            Self::Wrapped => write!(f, "wrapped"),
        }
    }
}

impl fmt::Debug for TypeCon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TypeCon({})", self)
    }
}

pub fn subst(ty: Type, map: &FxHashMap<TypeVar, Type>) -> Type {
    match ty {
        Type::Int => Type::Int,
        Type::Bool => Type::Bool,
        Type::String => Type::String,
        Type::Unit => Type::Unit,
        Type::Null => Type::Null,
        Type::Var(var) => match map.iter().find(|(v, _)| var == **v) {
            Some((_, ty)) => ty.clone(),
            None => Type::Var(var),
        },
        Type::App(TypeCon::Fun(params, body), tys) => {
            let mut map_in_func = FxHashMap::default();
            for (param, ty) in params.into_iter().zip(tys.into_iter()) {
                map_in_func.insert(param, ty);
            }

            let body = subst(*body, &map_in_func);
            subst(body, map)
        }
        Type::App(tycon, tys) => {
            let mut new_tys = Vec::with_capacity(tys.len());
            for ty in tys {
                new_tys.push(subst(ty, &map));
            }

            Type::App(tycon, new_tys)
        }
        Type::Poly(vars, ty) => {
            let mut new_map = FxHashMap::default();
            let mut new_vars = Vec::with_capacity(vars.len());
            for var in vars {
                new_vars.push(TypeVar::new());
                new_map.insert(var, Type::Var(*new_vars.last().unwrap()));
            }

            let ty = subst(*ty, &new_map);
            let ty = subst(ty, map);
            Type::Poly(new_vars, Box::new(ty))
        }
    }
}

pub fn unify(errors: &mut Vec<Error>, span: &Span, a: &Type, b: &Type) -> Option<()> {
    if let (Type::App(a_tycon, a_tys), Type::App(b_tycon, b_tys)) = (a, b) {
        let ok = match (a_tycon, b_tycon) {
            (TypeCon::Pointer(false), TypeCon::Pointer(false)) => true,
            (TypeCon::Pointer(true), TypeCon::Pointer(false)) => false,
            (TypeCon::Pointer(false), TypeCon::Pointer(true)) => true,
            (TypeCon::Pointer(true), TypeCon::Pointer(true)) => true,
            (TypeCon::Named(a, _), TypeCon::UnsizedNamed(b)) if a == b => true,
            (TypeCon::UnsizedNamed(a), TypeCon::Named(b, _)) if a == b => true,
            (a, b) => a == b,
        };

        if ok {
            for (a_ty, b_ty) in a_tys.iter().zip(b_tys.iter()) {
                unify(errors, span, a_ty, b_ty)?;
            }

            return Some(());
        }
    };

    match (a, b) {
        (Type::App(TypeCon::Fun(params, body), tys), b)
        | (b, Type::App(TypeCon::Fun(params, body), tys)) => {
            let mut map = FxHashMap::default();
            for (param, ty) in params.iter().zip(tys.iter()) {
                map.insert(param.clone(), ty.clone());
            }

            unify(errors, span, &subst(*body.clone(), &map), b)?;
            Some(())
        }
        (
            Type::App(TypeCon::Unique(_, uniq1), tys1),
            Type::App(TypeCon::Unique(_, uniq2), tys2),
        ) => {
            if uniq1 != uniq2 {
                return None;
            }

            for (ty1, ty2) in tys1.iter().zip(tys2.iter()) {
                unify(errors, span, ty1, ty2)?;
            }

            Some(())
        }
        (Type::Poly(vars1, ty1), Type::Poly(vars2, ty2)) => {
            let mut map = FxHashMap::default();
            for (var1, var2) in vars1.iter().zip(vars2.iter()) {
                map.insert(var2.clone(), Type::Var(*var1));
            }

            unify(errors, span, ty1, &subst(*ty2.clone(), &map))?;
            Some(())
        }
        (Type::Var(v1), Type::Var(v2)) if v1 == v2 => Some(()),
        (Type::Int, Type::Int) => Some(()),
        (Type::Bool, Type::Bool) => Some(()),
        (Type::String, Type::String) => Some(()),
        (Type::Unit, Type::Unit) => Some(()),
        (Type::App(TypeCon::Pointer(_), _), Type::Null) => Some(()),
        (Type::Null, Type::App(TypeCon::Pointer(_), _)) => Some(()),
        (a, b) => {
            errors.push(Error::new(
                &format!("`{}` and `{}` are not equivalent", a, b),
                span.clone(),
            ));
            None
        }
    }
}

// Returns size of a specified type. if a specified type size coludn't be calculated, returns None.
#[inline]
pub fn type_size(ty: &Type) -> Option<usize> {
    match ty {
        Type::App(TypeCon::Fun(params, body), tys) => {
            let mut map = FxHashMap::default();
            for (param, ty) in params.iter().zip(tys.iter()) {
                map.insert(param.clone(), ty.clone());
            }

            let body = subst(*body.clone(), &map);
            type_size(&body)
        }
        Type::App(TypeCon::Pointer(_), _) => Some(1),
        Type::App(TypeCon::Wrapped, _) => Some(1),
        Type::App(TypeCon::Array(size), types) => {
            let elem_size = type_size(&types[0])?;
            Some(elem_size * size)
        }
        Type::App(TypeCon::Named(_, size), _) => Some(*size),
        Type::App(TypeCon::UnsizedNamed(..), _) => None,
        Type::App(TypeCon::Arrow, _) => Some(2),
        ty @ Type::App(TypeCon::Unique(_, _), _) => {
            let ty = expand_unique(ty.clone());
            type_size(&ty)
        }
        Type::App(_, tys) => {
            let mut size = 0;
            for ty in tys {
                size += type_size(ty)?;
            }

            Some(size)
        }
        Type::Poly(_, _) | Type::Var(_) => None,
        Type::String => None,
        Type::Unit => Some(0),
        Type::Int | Type::Null | Type::Bool => Some(1),
    }
}

#[inline]
pub fn type_size_nocheck(ty: &Type) -> usize {
    type_size(ty).unwrap_or(0)
}

pub fn expand_unique(ty: Type) -> Type {
    match ty {
        Type::App(TypeCon::Fun(params, body), args) => {
            // { params_i -> args_i }
            let map: FxHashMap<TypeVar, Type> = params.into_iter().zip(args.into_iter()).collect();
            expand_unique(subst(*body, &map))
        }
        Type::App(TypeCon::Unique(tycon, _), tys) => expand_unique(Type::App(*tycon, tys)),
        ty => ty,
    }
}

pub fn expand_wrap(ty: Type) -> Type {
    match ty {
        Type::App(TypeCon::Fun(params, body), args) => {
            // { params_i -> args_i }
            let map: FxHashMap<TypeVar, Type> = params.into_iter().zip(args.into_iter()).collect();
            expand_wrap(subst(*body, &map))
        }
        Type::App(TypeCon::Wrapped, types) => expand_wrap(types[0].clone()),
        ty => ty,
    }
}

pub fn wrap_typevar(ty: &mut Type) {
    match ty {
        Type::App(TypeCon::Fun(_, _), _) => panic!(),
        Type::App(tycon, types) => {
            match tycon {
                TypeCon::Tuple | TypeCon::Array(_) | TypeCon::Struct(_) => {
                    for ty in types {
                        wrap_typevar(ty);
                    }
                }
                _ => {}
            };
        }
        vty @ Type::Var(_) => {
            let tyvar = mem::replace(vty, Type::Int);
            *vty = Type::App(TypeCon::Wrapped, vec![tyvar]);
        }
        _ => {}
    }
}

pub fn generate_func_type(params: &[Type], return_ty: &Type, ty_params: &[(Id, TypeVar)]) -> Type {
    assert!(!params.is_empty());

    // Generate type
    let mut stack = Vec::with_capacity(params.len());
    for param in params {
        stack.push(param);
    }

    let mut result_ty = Type::App(
        TypeCon::Arrow,
        vec![stack.pop().unwrap().clone(), return_ty.clone()],
    );
    while let Some(ty) = stack.pop() {
        result_ty = Type::App(TypeCon::Arrow, vec![ty.clone(), result_ty]);
    }

    if ty_params.is_empty() {
        result_ty
    } else {
        let tyvars = ty_params.iter().map(|(_, var)| *var).collect();
        Type::Poly(tyvars, Box::new(result_ty))
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeBody<'a> {
    Resolved(&'a TypeCon),
    Unresolved(&'a TypeCon),
}

impl TypeBody<'_> {
    pub fn tycon(&self) -> &TypeCon {
        match self {
            Self::Resolved(tycon) | Self::Unresolved(tycon) => tycon,
        }
    }
}

#[derive(Debug)]
pub struct TypeDefinitions {
    tycons: HashMapWithScope<Id, Option<TypeCon>>,
    sizes: HashMapWithScope<Id, usize>,
    is_resolved: bool,
}

impl TypeDefinitions {
    pub fn new() -> Self {
        Self {
            tycons: HashMapWithScope::new(),
            sizes: HashMapWithScope::new(),
            is_resolved: false,
        }
    }

    pub fn push_scope(&mut self) {
        self.tycons.push_scope();
        self.sizes.push_scope();
    }

    pub fn pop_scope(&mut self) {
        self.tycons.pop_scope();
        self.sizes.pop_scope();
    }

    pub fn insert(&mut self, name: Id) {
        self.tycons.insert(name, None);

        self.is_resolved = false;
    }

    pub fn set_body(&mut self, name: Id, tycon_: TypeCon) {
        let tycon = self.tycons.get_mut(&name).unwrap();
        *tycon = Some(tycon_);

        self.is_resolved = false;
    }

    pub fn get(&self, name: Id) -> Option<TypeBody<'_>> {
        let tycon = self.tycons.get(&name)?.as_ref()?;
        if self.sizes.contains_key(&name) {
            Some(TypeBody::Resolved(tycon))
        } else {
            Some(TypeBody::Unresolved(tycon))
        }
    }

    pub fn get_size(&self, name: Id) -> Option<usize> {
        self.sizes.get(&name).copied()
    }

    pub fn contains(&self, name: Id) -> bool {
        self.tycons.contains_key(&name)
    }

    fn resolve_in_type(&mut self, ty: &mut Type) -> Option<()> {
        match ty {
            Type::App(tycon, types) => {
                match &tycon {
                    TypeCon::Pointer(_) | TypeCon::Arrow | TypeCon::Wrapped => return Some(()),
                    _ => {}
                }

                self.resolve_in_tycon(tycon)?;
                for ty in types {
                    self.resolve_in_type(ty)?;
                }
            }
            Type::Poly(_, ty) => self.resolve_in_type(ty)?,
            _ => {}
        };

        Some(())
    }

    fn resolve_in_tycon(&mut self, tycon: &mut TypeCon) -> Option<()> {
        match tycon {
            TypeCon::UnsizedNamed(name) => match self.sizes.get(name) {
                Some(size) => *tycon = TypeCon::Named(*name, *size),
                None => {
                    // Calculate the type size after resolve it
                    let (mut tycon_to_calc, level) = self
                        .tycons
                        .get_with_level(name)
                        .map(|(tycon, l)| (tycon.as_ref().unwrap().clone(), l))
                        .unwrap();
                    self.resolve_in_tycon(&mut tycon_to_calc)?;

                    let size = type_size(&Type::App(tycon_to_calc, vec![]))?;
                    self.sizes.insert_with_level(level, *name, size);

                    *tycon = TypeCon::Named(*name, size);
                }
            },
            TypeCon::Fun(_, ty) => self.resolve_in_type(ty)?,
            TypeCon::Unique(tycon, _) => self.resolve_in_tycon(tycon)?,
            _ => {}
        };

        Some(())
    }

    pub fn resolve(&mut self) -> Result<(), Vec<Id>> {
        let mut names_not_calculated = Vec::new();

        let mut tycons = Vec::new();
        for (level, name, tycon) in &mut self.tycons {
            let tycon = match tycon.as_ref() {
                Some(tycon) => tycon,
                None => {
                    names_not_calculated.push(*name);
                    continue;
                }
            };

            tycons.push((level, *name, tycon.clone()));
        }

        for (level, name, tycon) in &mut tycons {
            if self.resolve_in_tycon(tycon).is_none() {
                names_not_calculated.push(*name);
                continue;
            }

            let size = match type_size(&Type::App(tycon.clone(), vec![])) {
                Some(size) => size,
                None => {
                    names_not_calculated.push(*name);
                    continue;
                }
            };

            self.sizes.insert_with_level(*level, *name, size);
        }

        for (level, name, tycon) in tycons {
            self.tycons.insert_with_level(level, name, Some(tycon));
        }

        if names_not_calculated.is_empty() {
            self.is_resolved = true;
            Ok(())
        } else {
            Err(names_not_calculated)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltype() {
        assert_eq!(ltype!(int), Type::Int);
        assert_eq!(ltype!(string), Type::String);
        assert_eq!(ltype!(unit), Type::Unit);
        assert_eq!(ltype!(bool), Type::Bool);
        assert_eq!(
            ltype!(*int),
            Type::App(TypeCon::Pointer(false), vec![Type::Int])
        );
        assert_eq!(
            ltype!(*mut int),
            Type::App(TypeCon::Pointer(true), vec![Type::Int])
        );
        assert_eq!(
            ltype!(**int),
            Type::App(
                TypeCon::Pointer(false),
                vec![Type::App(TypeCon::Pointer(false), vec![Type::Int])]
            )
        );
        assert_eq!(
            ltype!(int -> *int),
            Type::App(
                TypeCon::Arrow,
                vec![
                    Type::Int,
                    Type::App(TypeCon::Pointer(false), vec![Type::Int]),
                ],
            )
        );
    }

    #[test]
    fn resolve_type() {
        let id = IdMap::new_id;

        let mut types = TypeDefinitions::new();
        types.push_scope();

        types.insert(id("def"));

        // def = abc
        types.set_body(
            id("def"),
            TypeCon::Fun(
                vec![],
                Box::new(Type::App(TypeCon::UnsizedNamed(id("abc")), vec![])),
            ),
        );

        types.push_scope();

        types.insert(id("ghi"));
        types.insert(id("abc"));

        // ghi = (abc, def)
        types.set_body(
            id("ghi"),
            TypeCon::Fun(
                vec![],
                Box::new(Type::App(
                    TypeCon::Tuple,
                    vec![
                        Type::App(TypeCon::UnsizedNamed(id("abc")), vec![]),
                        Type::App(TypeCon::UnsizedNamed(id("def")), vec![]),
                    ],
                )),
            ),
        );

        // abc = int
        types.set_body(id("abc"), TypeCon::Fun(vec![], Box::new(Type::Int)));

        types.resolve().unwrap();

        assert_eq!(
            types.get(id("abc")).unwrap(),
            TypeBody::Resolved(&TypeCon::Fun(vec![], Box::new(Type::Int)))
        );
        assert_eq!(
            types.get(id("ghi")).unwrap(),
            TypeBody::Resolved(&TypeCon::Fun(
                vec![],
                Box::new(Type::App(
                    TypeCon::Tuple,
                    vec![
                        Type::App(TypeCon::Named(id("abc"), 1), vec![]),
                        Type::App(TypeCon::Named(id("def"), 1), vec![]),
                    ],
                )),
            )),
        );

        types.pop_scope();

        assert_eq!(
            types.get(id("def")).unwrap(),
            TypeBody::Resolved(&TypeCon::Fun(
                vec![],
                Box::new(Type::App(TypeCon::Named(id("abc"), 1), vec![])),
            )),
        );

        types.pop_scope();
    }
}
