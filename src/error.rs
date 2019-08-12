use std::error;
use std::fmt;
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

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}:{}-{}:{}", self.msg,
               self.span.start_line,
               self.span.start_col,
               self.span.end_line,
               self.span.end_col)
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        "Parse error"
    }
}
