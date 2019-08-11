use std::str::Chars;
use std::iter::Peekable;
use crate::error::Error;
use crate::span::{Span, Spanned};
use crate::token::*;

pub struct Lexer<'a> {
    input: Peekable<Chars<'a>>,
    start_line: u32,
    start_col: u32,
    line: u32,
    col: u32,
}

impl<'a> Lexer<'a> {
    pub fn new(s: &'a str) -> Lexer<'a> {
        Self {
            input: s.chars().peekable(),
            start_line: 0,
            start_col: 0,
            line: 0,
            col: 0,
        }
    }

    fn read_char(&mut self) -> char {
        match self.input.next() {
            Some('\n') => {
                self.line = 0;
                self.col = 0;
                '\n'
            },
            Some(ch) => {
                self.col += 1;
                ch
            },
            None => '\0',
        }
    }

    fn peek(&mut self) -> char {
        match self.input.peek() {
            Some(ch) => *ch,
            None => '\0',
        }
    }

    fn skip_whitespace(&mut self) {
        loop {
            let c = self.peek();
            if !c.is_whitespace() || c == '\0' {
                if c == '\n' {
                    self.line = 0;
                    self.col = 0;
                }
                break;
            }

            self.read_char();
        }
    }

    fn error(&mut self, msg: &str) -> Error {
        Error::new(msg, Span {
            start_line: self.start_line,
            start_col: self.start_col,
            end_line: self.line,
            end_col: self.col,
        })
    }

    fn next_token(&mut self) -> Result<Spanned<Token>, Error> {
        match self.read_char() {
            c => Err(self.error(&format!("Invalid character `{}`", c))),
        }
    }

    pub fn lex(mut self) -> Result<Vec<Spanned<Token>>, Vec<Error>> {
        let mut tokens = Vec::new();
        let mut errors = Vec::new();

        while self.peek() != '\0' {
            self.skip_whitespace();

            self.start_line = self.line;
            self.start_col = self.col;

            match self.next_token() {
                Ok(token) => tokens.push(token),
                Err(err) => errors.push(err),
            };
        }

        if errors.len() > 0 {
            Err(errors)
        } else {
            Ok(tokens)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_character() {
        let lexer = Lexer::new("あ あ");
        let errors = lexer.lex().unwrap_err();
        let expected = vec![
            Error::new("Invalid character `あ`", Span {
                start_line: 0,
                start_col: 0,
                end_line: 0,
                end_col: 1,
            }),
            Error::new("Invalid character `あ`", Span {
                start_line: 0,
                start_col: 2,
                end_line: 0,
                end_col: 3,
            }),
        ];

        for (error, expected) in errors.into_iter().zip(expected.into_iter()) {
            assert_eq!(error, expected);
        }
    }
}
