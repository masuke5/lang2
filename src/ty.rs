use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

use lazy_static::lazy_static;
use rustc_hash::FxHashMap;

use crate::id::{Id, IdMap};

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
            Some(id) if cfg!(debug_assertions) => write!(f, "{}'{}", IdMap::name(*id), self.0),
            Some(id) if !cfg!(debug_assertions) => write!(f, "{}", IdMap::name(*id)),
            _ => write!(f, "'{}", self.0),
        }
    }
}

macro_rules! write_iter {
    ($f:expr, $iter:expr) => {
        write_iter!($f, $iter, |a| a);
    };
    ($f:expr, $iter:expr, $filter:expr) => {
        let filter = $filter;
        let mut iter = $iter;

        if let Some(first) = iter.next() {
            write!($f, "{}", filter(first))?;
            for value in iter {
                write!($f, ", {}", filter(value))?;
            }
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
            Self::Unit => write!(f, "unit"),
            Self::Null => write!(f, "null"),
            Self::App(TypeCon::Pointer(is_mutable), types) => write!(f, "*{}{}", if *is_mutable { "mut " } else { "" }, types[0]),
            Self::App(TypeCon::Tuple, types) => {
                write!(f, "(")?;
                write_iter!(f, types.iter());
                write!(f, ")")
            },
            Self::App(TypeCon::Struct(fields), types) => {
                write!(f, "{{")?;
                write_iter!(f, fields.iter().zip(types.iter()), |(name, ty): (&Id, &Type)| format!("{}: {}", IdMap::name(*name), ty));
                write!(f, "}}")
            },
            Self::App(TypeCon::Array(size), types) => write!(f, "[{}; {}]", types[0], size),
            Self::App(TypeCon::Unique(tycon, uniq), types) => write!(f, "{} u{}", Type::App(*tycon.clone(), types.clone()), uniq),
            Self::App(tycon, tys) => {
                write!(f, "{}(", tycon)?;
                write_iter!(f, tys.iter());
                write!(f, ")")
            },
            Self::Var(var) => write!(f, "{}", var),
            Self::Poly(vars, ty) => {
                write!(f, "{}<", ty)?;
                write_iter!(f, vars.iter());
                write!(f, ">")
            },
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeCon {
    Pointer(bool),
    Tuple,
    Struct(Vec<Id>),
    Array(usize),
    Fun(Vec<TypeVar>, Box<Type>),
    Unique(Box<TypeCon>, u32),
    Named(Id),
}

impl fmt::Display for TypeCon {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Pointer(is_mutable) => write!(f, "{}pointer", if *is_mutable { "mut " } else { "" }),
            Self::Tuple => write!(f, "tuple"),
            Self::Struct(fields) => {
                write!(f, "{{")?;
                write_iter!(f, fields.iter(), |id: &Id| IdMap::name(*id));
                write!(f, "}}")
            },
            Self::Array(size) => write!(f, "array({})", size),
            Self::Fun(params, body) => {
                write!(f, "fun(")?;
                write_iter!(f, params.iter(), |a| format!("{}", a));
                write!(f, ") {}", body)
            },
            Self::Unique(tycon, uniq) => write!(f, "unique({}){{{}}}", tycon, uniq),
            Self::Named(name) => write!(f, "{}", IdMap::name(*name)),
        }
    }
}
