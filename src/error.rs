use crate::span::Span;
use std::fmt;

#[derive(Debug, PartialEq)]
pub enum Level {
    Error,
    Warning,
}

#[derive(Debug, PartialEq)]
pub struct Error {
    pub level: Level,
    pub msg: String,
    pub span: Span,
}

impl Error {
    pub fn new(msg: &str, span: Span) -> Self {
        Self {
            level: Level::Error,
            msg: msg.to_string(),
            span,
        }
    }

    pub fn new_warning(msg: &str, span: Span) -> Self {
        Self {
            level: Level::Warning,
            msg: msg.to_string(),
            span,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at {}:{}-{}:{}",
            self.msg,
            self.span.start_line,
            self.span.start_col,
            self.span.end_line,
            self.span.end_col
        )
    }
}

macro_rules! error {
    ($self:ident, $span:expr, $fmt: tt $(,$arg:expr)*) => {
        $self.errors.push(Error::new(&format!($fmt $(,$arg)*), $span));
    };
}

macro_rules! warn {
    ($self:ident, $span:expr, $fmt: tt $(,$arg:expr)*) => {
        $self.errors.push(Error::new_warning(&format!($fmt $(,$arg)*), $span));
    };
}

#[derive(Debug)]
pub struct ErrorList {
    errors: Vec<Error>,
    has_error: bool,
}

impl ErrorList {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            has_error: false,
        }
    }

    pub fn push(&mut self, error: Error) {
        if let Level::Error = &error.level {
            self.has_error = true;
        }

        self.errors.push(error);
    }

    pub fn append(&mut self, mut other: ErrorList) {
        if other.has_error {
            self.has_error = true;
        }

        self.errors.append(&mut other.errors);
    }

    pub fn has_error(&self) -> bool {
        self.has_error
    }

    pub fn errors(self) -> Vec<Error> {
        self.errors
    }
}
