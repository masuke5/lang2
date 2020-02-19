use std::fmt;
use std::mem;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

use lazy_static::lazy_static;
use rustc_hash::FxHashMap;

use crate::id::{Id, IdMap};
use crate::utils::{HashMapWithScope, format_bool};
use crate::error::Error;
use crate::span::Span;

macro_rules! ltype {
    (int) => { Type::Int };
    (unit) => { Type::Unit };
    (bool) => { Type::Bool };
    (string) => { Type::String };
    (*$($rest:tt)*) => { Type::App(TypeCon::Pointer(false), vec![ltype!($($rest)*)]) };
    () => { compile_error!("expected type or invalid type name") }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct TypeVar(u32);

lazy_static! {
    static ref VAR_ID_MAP: RwLock<FxHashMap<TypeVar, Id>> = {
        RwLock::new(FxHashMap::default())
    };
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let var_id_map = VAR_ID_MAP.read().expect("STR_MAP poisoned");
        let id = var_id_map.get(self);

        match id {
            Some(id) if cfg!(debug_assertions) => write!(f, "{}{{{}}}", IdMap::name(*id), self.0),
            Some(id) if !cfg!(debug_assertions) => write!(f, "{}", IdMap::name(*id)),
            _ => write!(f, "'{}", self.0),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Int => write!(f, "int"),
            Self::Bool => write!(f, "bool"),
            Self::String => write!(f, "string"),
            Self::Unit => write!(f, "()"),
            Self::Null => write!(f, "null"),
            Self::App(TypeCon::Pointer(is_mutable), types) => write!(f, "*{}{}", format_bool(*is_mutable, "mut "), types[0]),
            Self::App(TypeCon::Tuple, types) => {
                write!(f, "(")?;
                write_iter!(f, types.iter())?;
                write!(f, ")")
            },
            Self::App(TypeCon::Struct(fields), types) => {
                write!(f, "{{")?;
                write_iter!(f, fields
                    .iter()
                    .zip(types.iter())
                    .map(|(name, ty)| format!("{}: {}", IdMap::name(*name), ty))
                )?;
                write!(f, "}}")
            },
            Self::App(TypeCon::Arrow, types) => write!(f, "{} -> {}", types[0], types[1]),
            Self::App(TypeCon::Array(size), types) => write!(f, "[{}; {}]", types[0], size),
            Self::App(TypeCon::Unique(tycon, uniq), types) => write!(f, "{} u{}", Type::App(*tycon.clone(), types.clone()), uniq),
            Self::App(TypeCon::Wrapped, types) => write!(f, "'{}", types[0]),
            Self::App(tycon, tys) => {
                write!(f, "{}(", tycon)?;
                write_iter!(f, tys.iter())?;
                write!(f, ")")
            },
            Self::Var(var) => write!(f, "{}", var),
            Self::Poly(vars, ty) => {
                if !cfg!(debug_assertions) {
                    write!(f, "{}", ty)
                } else {
                    write!(f, "(")?;
                    write_iter!(f, vars.iter())?;
                    write!(f, "): {}", ty)
                }
            },
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeCon {
    Pointer(bool),
    Tuple,
    Arrow,
    Struct(Vec<Id>),
    Array(usize),
    Fun(Vec<TypeVar>, Box<Type>),
    Unique(Box<TypeCon>, u32),
    Named(Id, usize),
    Wrapped,
}

impl fmt::Display for TypeCon {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Pointer(is_mutable) => write!(f, "{}pointer", format_bool(*is_mutable, "mut ")),
            Self::Tuple => write!(f, "tuple"),
            Self::Arrow => write!(f, "arrow"),
            Self::Struct(fields) => {
                write!(f, "{{")?;
                write_iter!(f, fields.iter().map(|id| IdMap::name(*id)))?;
                write!(f, "}}")
            },
            Self::Array(size) => write!(f, "array({})", size),
            Self::Fun(params, body) => {
                write!(f, "fun(")?;
                write_iter!(f, params.iter().map(|a| format!("{}", a)))?;
                write!(f, ") {}", body)
            },
            Self::Unique(tycon, uniq) => write!(f, "unique({}){{{}}}", tycon, uniq),
            Self::Named(name, size) => write!(f, "{}{{size={}}}", IdMap::name(*name), size),
            Self::Wrapped => write!(f, "wrapped"),
        }
    }
}

pub fn subst(ty: Type, map: &FxHashMap<TypeVar, Type>) -> Type {
    match ty {
        Type::Int => Type::Int,
        Type::Bool => Type::Bool,
        Type::String => Type::String,
        Type::Unit => Type::Unit,
        Type::Null => Type::Null,
        Type::Var(var) => {
            match map.iter().find(|(v, _)| var == **v) {
                Some((_, ty)) => ty.clone(),
                None => Type::Var(var),
            }
        },
        Type::App(TypeCon::Fun(params, body), tys) => {
            let mut map_in_func = FxHashMap::default();
            for (param, ty) in params.into_iter().zip(tys.into_iter()) {
                map_in_func.insert(param, ty);
            }

            let body = subst(*body, &map_in_func);
            subst(body, map)
        },
        Type::App(tycon, tys) => {
            let mut new_tys = Vec::with_capacity(tys.len());
            for ty in tys {
                new_tys.push(subst(ty, &map));
            }

            Type::App(tycon, new_tys)
        },
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
        },
    }
}

pub fn unify(errors: &mut Vec<Error>, span: &Span, a: &Type, b: &Type) -> Option<()> {
    match (a, b) {
        (Type::App(a_tycon, a_tys), Type::App(b_tycon, b_tys)) => {
            let ok = match (a_tycon, b_tycon) {
                (TypeCon::Pointer(false), TypeCon::Pointer(false)) => true,
                (TypeCon::Pointer(true), TypeCon::Pointer(false)) => false,
                (TypeCon::Pointer(false), TypeCon::Pointer(true)) => true,
                (TypeCon::Pointer(true), TypeCon::Pointer(true)) => true,
                (a, b) if a == b => true,
                _ => false,
            };

            if ok {
                for (a_ty, b_ty) in a_tys.iter().zip(b_tys.iter()) {
                    unify(errors, span, a_ty, b_ty)?;
                }

                return Some(());
            }

        },
        _ => {},
    };

    match (a, b) {
        (Type::App(TypeCon::Fun(params, body), tys), b) | (b, Type::App(TypeCon::Fun(params, body), tys)) => {
            let mut map = FxHashMap::default();
            for (param, ty) in params.iter().zip(tys.iter()) {
                map.insert(param.clone(), ty.clone());
            }

            unify(errors, span, &subst(*body.clone(), &map), b)?;
            Some(())
        },
        (Type::App(TypeCon::Unique(_, uniq1), tys1), Type::App(TypeCon::Unique(_, uniq2), tys2)) => {
            if uniq1 != uniq2 {
                return None;
            }

            for (ty1, ty2) in tys1.iter().zip(tys2.iter()) {
                unify(errors, span, ty1, ty2)?;
            }

            Some(())
        },
        (Type::Poly(vars1, ty1), Type::Poly(vars2, ty2)) => {
            let mut map = FxHashMap::default();
            for (var1, var2) in vars1.iter().zip(vars2.iter()) {
                map.insert(var2.clone(), Type::Var(*var1));
            }

            unify(errors, span, ty1, &subst(*ty2.clone(), &map))?;
            Some(())
        },
        (Type::Var(v1), Type::Var(v2)) if v1 == v2 => Some(()),
        (Type::Int, Type::Int) => Some(()),
        (Type::Bool, Type::Bool) => Some(()),
        (Type::String, Type::String) => Some(()),
        (Type::Unit, Type::Unit) => Some(()),
        (Type::App(TypeCon::Pointer(_), _), Type::Null) => Some(()),
        (Type::Null, Type::App(TypeCon::Pointer(_), _)) => Some(()),
        (a, b) => {
            errors.push(Error::new(&format!("`{}` and `{}` are not equivalent", a, b), span.clone()));
            None
        },
    }
}

// Returns size of a specified type. if a specified type size coludn't be calculated, returns None.
pub fn type_size(ty: &Type) -> Option<usize> {
    match ty {
        Type::App(TypeCon::Fun(params, body), tys) => {
            let mut map = FxHashMap::default();
            for (param, ty) in params.iter().zip(tys.iter()) {
                map.insert(param.clone(), ty.clone());
            }

            let body = subst(*body.clone(), &map);
            type_size(&body)
        },
        Type::App(TypeCon::Pointer(_), _) => Some(1),
        Type::App(TypeCon::Wrapped, _) => Some(1),
        Type::App(TypeCon::Array(size), types) => {
            let elem_size = type_size(&types[0])?;
            Some(elem_size * size)
        },
        Type::App(TypeCon::Named(_, size), _) => {
            Some(*size)
        },
        Type::App(TypeCon::Arrow, _) => Some(2),
        ty @ Type::App(TypeCon::Unique(_, _), _) => {
            let ty = expand_unique(ty.clone());
            type_size(&ty)
        },
        Type::App(_, tys) => {
            let mut size = 0;
            for ty in tys {
                size += type_size(ty)?;
            }

            Some(size)
        }
        Type::Poly(_, _) | Type::Var(_) => None,
        Type::Unit => Some(0),
        _ => Some(1),
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
        },
        Type::App(TypeCon::Unique(tycon, _), tys) => {
            expand_unique(Type::App(*tycon, tys))
        },
        ty => ty,
    }
}

pub fn expand_wrap(ty: Type) -> Type {
    match ty {
        Type::App(TypeCon::Fun(params, body), args) => {
            // { params_i -> args_i }
            let map: FxHashMap<TypeVar, Type> = params.into_iter().zip(args.into_iter()).collect();
            expand_wrap(subst(*body, &map))
        },
        Type::App(TypeCon::Wrapped, types) => {
            expand_wrap(types[0].clone())
        },
        ty => ty,
    }
}

pub fn resolve_type_sizes_in_tycon(type_sizes: &HashMapWithScope<Id, usize>, tycon: TypeCon) -> TypeCon {
    match tycon {
        TypeCon::Named(name, _) => {
            TypeCon::Named(name, *type_sizes.get(&name).unwrap())
        },
        TypeCon::Fun(params, ty) => {
            TypeCon::Fun(params, Box::new(resolve_type_sizes(type_sizes, *ty)))
        },
        TypeCon::Unique(tycon, uniq) => {
            TypeCon::Unique(Box::new(resolve_type_sizes_in_tycon(type_sizes, *tycon)), uniq)
        },
        tycon => tycon,
    }
}

pub fn resolve_type_sizes(type_sizes: &HashMapWithScope<Id, usize>, ty: Type) -> Type {
    match ty {
        Type::App(tycon, types) => {
            let tycon = resolve_type_sizes_in_tycon(type_sizes, tycon);
            let types = types.into_iter().map(|ty| resolve_type_sizes(type_sizes, ty)).collect();
            Type::App(tycon, types)
        },
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
                },
                _ => {},
            };
        },
        vty @ Type::Var(_) => {
            let tyvar = mem::replace(vty, Type::Int);
            *vty = Type::App(TypeCon::Wrapped, vec![tyvar]);
        },
        _ => {},
    }
}

pub fn generate_func_type(params: &Vec<Type>, return_ty: &Type, ty_params: &Vec<(Id, TypeVar)>) -> Type {
    assert!(!params.is_empty());

    // Generate type
    let mut stack = Vec::with_capacity(params.len());
    for param in params {
        stack.push(param);
    }

    let mut result_ty = Type::App(TypeCon::Arrow, vec![stack.pop().unwrap().clone(), return_ty.clone()]);
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
