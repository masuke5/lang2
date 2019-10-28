use std::fmt;

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int,
    Bool,
    Invalid,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Bool => write!(f, "bool"),
            Type::Invalid => write!(f, "invalid"),
        }
    }
}

impl Type {
    // in bytes
    pub fn size(&self) -> usize {
        match self {
            Type::Int => 8,
            Type::Bool => 8,
            Type::Invalid => 0,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Typed<T> {
    ty: Type,
    kind: T,
}
