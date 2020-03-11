use std::fmt;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

use lazy_static::lazy_static;
use rustc_hash::FxHashMap;

use crate::id::{Id, IdMap};
use crate::span::Span;

#[derive(Debug, PartialEq)]
pub enum Level {
    Error,
    Warning,
}

#[derive(Debug, PartialEq)]
pub struct Error<'a> {
    pub level: Level,
    pub msg: String,
    pub span: &'a Span,
}

impl<'a> Error<'a> {
    pub fn new(msg: &str, span: &'a Span) -> Self {
        Self {
            level: Level::Error,
            msg: msg.to_string(),
            span,
        }
    }

    pub fn new_warning(msg: &str, span: &'a Span) -> Self {
        Self {
            level: Level::Warning,
            msg: msg.to_string(),
            span,
        }
    }
}

impl<'a> fmt::Display for Error<'a> {
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
    ($span:expr, $fmt: tt $(,$arg:expr)*) => {
        ErrorList::push(Error::new(&format!($fmt $(,$arg)*), $span));
    };
}

macro_rules! warn {
    ($span:expr, $fmt: tt $(,$arg:expr)*) => {
        ErrorList::push(Error::new_warning(&format!($fmt $(,$arg)*), $span));
    };
}

#[derive(Debug)]
pub struct ErrorList {}

static HAS_ERRORS: AtomicBool = AtomicBool::new(false);

lazy_static! {
    static ref FILE_CACHE: RwLock<FxHashMap<Id, Vec<String>>> =
        { RwLock::new(FxHashMap::default()) };
}

impl ErrorList {
    fn print_error(error: Error<'_>) {
        let mut file_cache = FILE_CACHE.write().expect("FILE_CACHE poisoned");
        let es = error.span;

        let input = match file_cache.get(&es.file) {
            Some(input) => input,
            None => {
                let contents = fs::read_to_string(&IdMap::name(es.file)).unwrap();
                file_cache.insert(
                    es.file,
                    contents.split('\n').map(|c| c.to_string()).collect(),
                );
                &file_cache[&es.file]
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

    pub fn push(error: Error<'_>) {
        if let Level::Error = &error.level {
            HAS_ERRORS.store(true, Ordering::Relaxed);
        }

        Self::print_error(error);
    }

    pub fn has_error() -> bool {
        HAS_ERRORS.load(Ordering::Acquire)
    }
}
