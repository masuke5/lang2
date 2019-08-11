use crate::span::Span;

#[derive(Debug, PartialEq)]
pub struct Error {
    pub msg: String,
    pub span: Span,
}

impl Error {
    pub fn new(msg: &str, span: Span) -> Self {
        Self {
            msg: msg.to_string(),
            span,
        }
    }
}
