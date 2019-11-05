use std::fmt;

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int,
    Bool,
    String,
    Invalid,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Bool => write!(f, "bool"),
            Type::String => write!(f, "string"),
            Type::Invalid => write!(f, "invalid"),
        }
    }
}
