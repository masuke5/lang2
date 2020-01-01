use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use crate::id::{Id, IdMap};

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct TypeVar(u32);

// const LETTERS: [char; 26] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'n', 'm', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'];

impl fmt::Display for TypeVar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

static NEXT_VAR: AtomicU32 = AtomicU32::new(0);

impl TypeVar {
    pub fn new() -> Self {
        let var = Self(NEXT_VAR.load(Ordering::Acquire));
        NEXT_VAR.fetch_add(1, Ordering::Acquire);
        var
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
            Self::App(tycon, tys) => {
                write!(f, "{}[", tycon)?;

                if !tys.is_empty() {
                    let mut iter = tys.iter();
                    write!(f, "{}", *iter.next().unwrap())?;
                    for ty in iter {
                        write!(f, ", {}", ty)?;
                    }
                }

                write!(f, "]")
            },
            Self::Var(var) => write!(f, "var({})", var),
            Self::Poly(vars, ty) => {
                write!(f, "{}<", ty)?;

                if !vars.is_empty() {
                    let mut iter = vars.iter();
                    write!(f, "{}", *iter.next().unwrap())?;
                    for var in vars {
                        write!(f, ", {}", var)?;
                    }
                }

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
}

impl fmt::Display for TypeCon {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Pointer(is_mutable) => write!(f, "{}pointer", if *is_mutable { "mut " } else { "" }),
            Self::Tuple => write!(f, "tuple"),
            Self::Struct(fields) => {
                write!(f, "{{")?;

                if !fields.is_empty() {
                    let mut iter = fields.iter();
                    write!(f, "{}", IdMap::name(*iter.next().unwrap()))?;
                    for field in iter {
                        write!(f, ", {}", IdMap::name(*field))?;
                    }
                }

                write!(f, "}}")
            },
            Self::Array(size) => write!(f, "array({})", size),
            Self::Fun(params, body) => {
                write!(f, "fun(")?;

                if !params.is_empty() {
                    let mut iter = params.iter();
                    write!(f, "{}", iter.next().unwrap())?;
                    for param in iter {
                        write!(f, ", {}", param)?;
                    }
                }

                write!(f, ") = {}", body)
            },
            Self::Unique(tycon, uniq) => write!(f, "unique({}){{{}}}", tycon, uniq),
        }
    }
}
