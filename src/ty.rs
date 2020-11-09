use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

use lazy_static::lazy_static;
use rustc_hash::FxHashMap;

use crate::ast::SymbolPath;
use crate::error::{Error, ErrorList};
use crate::id::{Id, IdMap};
use crate::span::Span;
use crate::utils::format_bool;

lazy_static! {
    pub static ref MODULE_STD_PATH: SymbolPath = SymbolPath::new().append_str("std");
}

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
    static ref VAR_ID_MAP: RwLock<FxHashMap<TypeVar, Id>> = RwLock::new(FxHashMap::default());
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

static NEXT_UNIQUE: AtomicU32 = AtomicU32::new(0);

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct Unique(u32);

impl Unique {
    pub fn new() -> Self {
        let var = Self(NEXT_UNIQUE.load(Ordering::Acquire));
        NEXT_UNIQUE.fetch_add(1, Ordering::Acquire);
        var
    }
}

impl fmt::Display for Unique {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unique({})", self.0)
    }
}

impl fmt::Debug for Unique {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(PartialEq, Clone)]
pub enum Type {
    UInt,
    Int,
    Bool,
    Char,
    Float,
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
            Self::UInt => write!(f, "uint"),
            Self::Float => write!(f, "float"),
            Self::Bool => write!(f, "bool"),
            Self::Char => write!(f, "char"),
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
            Self::App(TypeCon::Slice(is_mutable), types) => {
                write!(f, "&{}[{}]", format_bool(*is_mutable, "mut "), types[0])
            }
            #[cfg(debug_assertions)]
            Self::App(TypeCon::Unique(tycon, uniq), types) => {
                write!(f, "{} u{}", Type::App(*tycon.clone(), types.clone()), uniq)
            }
            #[cfg(not(debug_assertions))]
            Self::App(TypeCon::Unique(tycon, _), types) => {
                write!(f, "{}", Type::App(*tycon.clone(), types.clone()))
            }
            #[cfg(not(debug_assertions))]
            Self::App(TypeCon::Named(name), types) => {
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
    Slice(bool),
    Fun(Vec<TypeVar>, Box<Type>),
    Unique(Box<TypeCon>, Unique),
    Named(SymbolPath),
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
            Self::Slice(is_mutable) => write!(f, "{}slice", format_bool(*is_mutable, "mut ")),
            Self::Fun(params, body) => {
                write!(f, "fun(")?;
                write_iter!(f, params.iter().map(|a| format!("{}", a)))?;
                write!(f, ") = {}", body)
            }
            Self::Unique(tycon, uniq) => write!(f, "unique({}){{{}}}", tycon, uniq),
            Self::Named(name) => write!(f, "{}", name),
        }
    }
}

impl fmt::Debug for TypeCon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TypeCon({})", self)
    }
}

pub fn subst(ty: Type, map: &FxHashMap<TypeVar, Type>) -> Type {
    type T = Type;
    type TC = TypeCon;

    match ty {
        T::Int | T::UInt | T::Float | T::Char | T::Bool | T::String | T::Unit | T::Null => ty,
        T::Var(var) => match map.iter().find(|(v, _)| var == **v) {
            Some((_, ty)) => ty.clone(),
            None => T::Var(var),
        },
        T::App(TC::Fun(params, body), tys) => {
            let map_in_func: FxHashMap<TypeVar, T> =
                params.into_iter().zip(tys.into_iter()).collect();

            let body = subst(*body, &map_in_func);
            subst(body, map)
        }
        T::App(tycon, tys) => {
            let new_tys: Vec<T> = tys.into_iter().map(|ty| subst(ty, &map)).collect();
            T::App(tycon, new_tys)
        }
        T::Poly(vars, ty) => {
            let mut new_map = FxHashMap::default();
            let mut new_vars = Vec::with_capacity(vars.len());
            for var in vars {
                new_vars.push(TypeVar::new());
                new_map.insert(var, T::Var(*new_vars.last().unwrap()));
            }

            let ty = subst(*ty, &new_map);
            let ty = subst(ty, map);
            T::Poly(new_vars, Box::new(ty))
        }
    }
}

pub fn unify(span: &Span, a: &Type, b: &Type) -> Option<()> {
    let result = unify_inner(span, a, b);
    if result.is_none() {
        error!(&span.clone(), "`{}` is not equivalent to `{}`", a, b);
    }

    result
}

pub fn unify_inner(span: &Span, a: &Type, b: &Type) -> Option<()> {
    if let (Type::App(a_tycon, a_tys), Type::App(b_tycon, b_tys)) = (a, b) {
        if a == b {
            for (a_ty, b_ty) in a_tys.iter().zip(b_tys.iter()) {
                unify_inner(span, a_ty, b_ty)?;
            }

            return Some(());
        }
    };

    match (a, b) {
        (Type::App(TypeCon::Fun(params, body), tys), b)
        | (b, Type::App(TypeCon::Fun(params, body), tys)) => {
            let mut map = FxHashMap::default();
            for (param, ty) in params.iter().zip(tys.iter()) {
                map.insert(*param, ty.clone());
            }

            unify_inner(span, &subst(*body.clone(), &map), b)?;
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
                unify_inner(span, ty1, ty2)?;
            }

            Some(())
        }
        (Type::Poly(vars1, ty1), Type::Poly(vars2, ty2)) => {
            let map: FxHashMap<TypeVar, Type> = vars2
                .iter()
                .copied()
                .zip(vars1.iter().map(|v| Type::Var(*v)))
                .collect();

            unify_inner(span, ty1, &subst(*ty2.clone(), &map))?;
            Some(())
        }
        (Type::Var(v1), Type::Var(v2)) if v1 == v2 => Some(()),
        (Type::Int, Type::Int) => Some(()),
        (Type::UInt, Type::UInt) => Some(()),
        (Type::Float, Type::Float) => Some(()),
        (Type::Bool, Type::Bool) => Some(()),
        (Type::String, Type::String) => Some(()),
        (Type::Char, Type::Char) => Some(()),
        (Type::Unit, Type::Unit) => Some(()),
        (Type::App(TypeCon::Pointer(_), _), Type::Null) => Some(()),
        (Type::Null, Type::App(TypeCon::Pointer(_), _)) => Some(()),
        _ => None,
    }
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
    fn test_subst() {
        assert_eq!(Type::Int, subst(Type::Int, &FxHashMap::default()));
    }

    #[test]
    fn test_subst_with_tyfun() {
        let a = TypeVar::new();
        let b = TypeVar::new();
        assert_eq!(
            Type::App(TypeCon::Pointer(false), vec![Type::Bool]),
            subst(
                // (fn a => Pointer(a))(b)
                Type::App(
                    TypeCon::Fun(
                        vec![a],
                        Box::new(Type::App(TypeCon::Pointer(false), vec![Type::Var(a)]))
                    ),
                    vec![Type::Var(b)]
                ),
                // b = Bool
                &[(b, Type::Bool)].iter().cloned().collect(),
            )
        );
    }

    #[test]
    fn test_subst_with_app() {
        let a = TypeVar::new();
        assert_eq!(
            Type::App(TypeCon::Tuple, vec![Type::Unit, Type::Unit]),
            subst(
                Type::App(TypeCon::Tuple, vec![Type::Var(a), Type::Var(a)]),
                &[(a, Type::Unit)].iter().cloned().collect(),
            ),
        );
    }

    #[test]
    fn test_subst_with_poly() {
        let a = TypeVar::new();
        let ty = subst(
            Type::Poly(
                vec![a],
                Box::new(Type::App(TypeCon::Pointer(true), vec![Type::Var(a)])),
            ),
            &[(a, Type::Bool)].iter().cloned().collect(),
        );

        assert!(match ty {
            Type::Poly(params, box Type::App(TypeCon::Pointer(true), types)) => match &types[0] {
                Type::Var(var) if params[0] == *var && *var != a => true,
                _ => false,
            },
            _ => false,
        });
    }

    #[test]
    fn test_unify() {
        let span = Span::zero(*crate::id::reserved_id::TEST);
        assert!(unify(&span, &Type::Int, &Type::Int).is_some());
    }

    #[test]
    fn test_unify_with_pointer_and_null() {
        let span = Span::zero(*crate::id::reserved_id::TEST);
        assert!(unify(
            &span,
            &Type::App(TypeCon::Pointer(false), vec![Type::Int]),
            &Type::Null,
        )
        .is_some());
    }

    #[test]
    fn test_unify_with_tyfun() {
        let span = Span::zero(*crate::id::reserved_id::TEST);
        let a = TypeVar::new();
        let b = TypeVar::new();
        let c = TypeVar::new();
        assert!(unify(
            &span,
            // (fn a => Pointer(a))(b)
            &Type::App(
                TypeCon::Fun(
                    vec![a],
                    Box::new(Type::App(TypeCon::Pointer(false), vec![Type::Var(a)]))
                ),
                vec![Type::Var(b)]
            ),
            // (fn c => Pointer(c))(b)
            &Type::App(
                TypeCon::Fun(
                    vec![c],
                    Box::new(Type::App(TypeCon::Pointer(false), vec![Type::Var(c)]))
                ),
                vec![Type::Var(b)]
            ),
        )
        .is_some());
    }

    #[test]
    fn test_unify_with_unique() {
        let span = Span::zero(*crate::id::reserved_id::TEST);

        let u1 = Unique::new();
        let u2 = Unique::new();

        // The same unique but the tycon is different
        assert!(unify_inner(
            &span,
            &Type::App(TypeCon::Unique(Box::new(TypeCon::Tuple), u1), vec![]),
            &Type::App(TypeCon::Unique(Box::new(TypeCon::Arrow), u1), vec![]),
        )
        .is_some());

        // Different unique
        assert!(unify_inner(
            &span,
            &Type::App(TypeCon::Unique(Box::new(TypeCon::Tuple), u1), vec![]),
            &Type::App(TypeCon::Unique(Box::new(TypeCon::Tuple), u2), vec![]),
        )
        .is_none());

        // The same unique and the same tycon
        assert!(unify_inner(
            &span,
            &Type::App(TypeCon::Unique(Box::new(TypeCon::Tuple), u2), vec![]),
            &Type::App(TypeCon::Unique(Box::new(TypeCon::Tuple), u2), vec![]),
        )
        .is_some());
    }

    #[test]
    fn test_unify_with_poly() {
        let span = Span::zero(*crate::id::reserved_id::TEST);

        let a = TypeVar::new();
        let b = TypeVar::new();
        assert!(unify(
            &span,
            &Type::Poly(
                vec![a],
                Box::new(Type::App(TypeCon::Pointer(true), vec![Type::Var(a)]))
            ),
            &Type::Poly(
                vec![b],
                Box::new(Type::App(TypeCon::Pointer(true), vec![Type::Var(b)]))
            ),
        )
        .is_some());
    }
}
