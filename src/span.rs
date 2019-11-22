use crate::id::Id;

#[derive(Debug, PartialEq, Clone)]
pub struct Span {
    pub file: Id,
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

impl Span {
    pub fn merge(left: &Span, right: &Span) -> Span {
        Span {
            file: left.file,
            start_line: left.start_line,
            start_col: left.start_col,
            end_line: right.end_line,
            end_col: right.end_col,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Spanned<T> {
    pub kind: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(kind: T, span: Span) -> Spanned<T> {
        Self {
            kind,
            span,
        }
    }
}
