use std::fmt;
use crate::id::{Id, IdMap};

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int,
    Bool,
    String,
    Unit,
    Null,
    Pointer(Box<Type>),
    Tuple(Vec<Type>),
    Struct(Vec<(Id, Type)>),
    Array(Box<Type>, usize),
    Named(Id),
    Invalid,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Bool => write!(f, "bool"),
            Type::String => write!(f, "string"),
            Type::Unit => write!(f, "()"),
            Type::Null => write!(f, "null"),
            Type::Pointer(ty) => write!(f, "*{}", ty),
            Type::Tuple(inner) => {
                write!(f, "(")?;

                if !inner.is_empty() {
                    let mut iter = inner.iter();
                    write!(f, "{}", iter.next().unwrap())?;   
                    for ty in iter {
                        write!(f, ", {}", ty)?;
                    }
                }

                write!(f, ")")
            },
            Type::Struct(fields) => {
                write!(f, "struct {{")?;

                let mut iter = fields.iter();

                if let Some((id, ty)) = iter.next() {
                    write!(f, " {}: {}", IdMap::name(*id), ty)?;
                    for (id, ty) in iter {
                        write!(f, ", {}: {}", IdMap::name(*id), ty)?;
                    }
                }

                write!(f, " }}")
            },
            Type::Array(ty, size) => write!(f, "[{}; {}]", ty, size),
            Type::Named(name) => {
                write!(f, "{}", IdMap::name(*name))
            },
            Type::Invalid => write!(f, "invalid"),
        }
    }
}
