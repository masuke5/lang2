use std::fmt;
use std::fs;

use rustc_hash::FxHashMap;

use crate::id::{Id, IdMap};
use crate::span::Span;

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
    file_cache: FxHashMap<Id, Vec<String>>,
    has_error: bool,
}

impl ErrorList {
    pub fn new() -> Self {
        Self {
            file_cache: FxHashMap::default(),
            has_error: false,
        }
    }

    fn print_error(&mut self, error: Error) {
        let es = error.span;

        let input = match self.file_cache.get(&es.file) {
            Some(input) => input,
            None => {
                let contents = fs::read_to_string(&IdMap::name(es.file)).unwrap();
                self.file_cache.insert(
                    es.file,
                    contents.split('\n').map(|c| c.to_string()).collect(),
                );
                &self.file_cache[&es.file]
            }
        };

        let (color, label) = match error.level {
            Level::Error => ("\x1b[91m", "error"),     // bright red
            Level::Warning => ("\x1b[93m", "warning"), // bright yellow
        };

        // Print the error position and message
        println!(
            "{}{}\x1b[0m: {}:{}:{}-{}:{}: \x1b[97m{}\x1b[0m",
            color,
            label,
            IdMap::name(es.file),
            es.start_line + 1,
            es.start_col,
            es.end_line + 1,
            es.end_col,
            error.msg
        );

        // Print the lines
        let line_count = es.end_line - es.start_line + 1;
        for i in 0..line_count {
            let line = (es.start_line + i) as usize;
            let line_len = if line >= input.len() {
                0
            } else {
                input[line].len() as u32
            };
            println!(
                "{}",
                if line >= input.len() {
                    ""
                } else {
                    &input[line]
                }
            );

            let indent = input[line]
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .fold(0, |indent, c| {
                    indent
                        + match c {
                            ' ' => 1,
                            '\t' => 4,
                            _ => unreachable!(),
                        }
                }) as u32;

            // Print the error span
            let (start, length) = if line_count == 1 {
                (es.start_col, es.end_col - es.start_col)
            } else if i == 0 {
                (es.start_col, line_len - es.start_col)
            } else if i == line_count - 1 {
                (indent, (line_len - es.end_col).saturating_sub(1))
            } else {
                (indent, line_len - indent)
            };

            let (start, length) = (start as usize, length as usize);

            println!(
                "{}{}{}\x1b[0m",
                " ".repeat(start),
                color,
                "^".repeat(length)
            );
        }
    }

    pub fn push(&mut self, error: Error) {
        if let Level::Error = &error.level {
            self.has_error = true;
        }

        self.print_error(error);
    }

    pub fn append(&mut self, other: ErrorList) {
        if other.has_error {
            self.has_error = true;
        }
    }

    pub fn has_error(&self) -> bool {
        self.has_error
    }

    pub fn errors(self) -> Vec<Error> {
        Vec::new()
    }
}
