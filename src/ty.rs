use std::fmt;

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int,
    Bool,
    String,
    Tuple(Vec<Type>),
    Invalid,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Bool => write!(f, "bool"),
            Type::String => write!(f, "string"),
            Type::Tuple(inner) => {
                write!(f, "(")?;

                if inner.len() > 0 {
                    let mut iter = inner.iter();
                    write!(f, "{}", iter.next().unwrap())?;   
                    for ty in iter {
                        write!(f, ", {}", ty)?;
                    }
                }

                write!(f, ")")
            }
            Type::Invalid => write!(f, "invalid"),
        }
    }
}

impl Type {
    pub fn size(&self) -> usize {
        match self {
            Type::Tuple(inner) => inner.len(),
            Type::Invalid => 0,
            _ => 1,
        }
    }
}
