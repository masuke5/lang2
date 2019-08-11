use crate::span::Span;

#[derive(Debug, PartialEq)]
pub struct Error {
    msg: String,
    span: Span,
}

impl Error {
    pub fn new(msg: &str, span: Span) -> Self {
        Self {
            msg: msg.to_string(),
            span,
        }
    }
}
