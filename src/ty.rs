#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int,
    Bool,
    Invalid,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Typed<T> {
    ty: Type,
    kind: T,
}
