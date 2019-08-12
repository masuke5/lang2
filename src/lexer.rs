use std::str::Chars;
use std::iter::Peekable;
use crate::error::Error;
use crate::span::{Span, Spanned};
use crate::token::*;

fn is_identifier_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

pub struct Lexer<'a> {
    raw: &'a str,
    input: Peekable<Chars<'a>>,
    start_line: u32,
    start_col: u32,
    line: u32,
    col: u32,
    // For make slice for identifier
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(s: &'a str) -> Lexer<'a> {
        Self {
            raw: s,
            input: s.chars().peekable(),
            start_line: 0,
            start_col: 0,
            line: 0,
            col: 0,
            pos: 0,
        }
    }

    fn read_char(&mut self) -> char {
        let c = match self.input.next() {
            Some('\n') => {
                self.line += 1;
                self.col = 0;
                '\n'
            },
            Some(ch) => {
                self.col += 1;
                ch
            },
            None => '\0',
        };

        self.pos += c.len_utf8();
        c
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

    fn lex_number(&mut self, start_char: char) -> Token<'a> {
        let mut n = start_char.to_digit(10).unwrap() as i64;
        loop {
            match self.peek() {
                c if c.is_digit(10) => {
                    n = n * 10 + c.to_digit(10).unwrap() as i64;
                    self.read_char();
                },
                _ => {
                    break;
                }
            }
        }

        Token::Number(n)
    }

    fn lex_identifier(&mut self, c: char) -> Token<'a> {
        let start_pos = self.pos - c.len_utf8();

        loop {
            if is_identifier_char(self.peek()) {
                self.read_char();
            } else {
                break;
            }
        }

        match &self.raw[start_pos..self.pos] {
            "let" => Token::Let,
            s => Token::Identifier(s),
        }
    }

    fn next_token(&mut self) -> Result<Token<'a>, Error> {
        match self.read_char() {
            c if c.is_digit(10) => Ok(self.lex_number(c)),
            c if is_identifier_char(c) => Ok(self.lex_identifier(c)),
            '+' => Ok(Token::Add),
            '-' => Ok(Token::Sub),
            '*' => Ok(Token::Asterisk),
            '/' => Ok(Token::Div),
            '(' => Ok(Token::Lparen),
            ')' => Ok(Token::Rparen),
            '=' => Ok(Token::Assign),
            ';' => Ok(Token::Semicolon),
            c => Err(self.error(&format!("Invalid character `{}`", c))),
        }
    }

    pub fn lex(mut self) -> Result<Vec<Spanned<Token<'a>>>, Vec<Error>> {
        let mut tokens = Vec::new();
        let mut errors = Vec::new();

        while self.peek() != '\0' {
            self.skip_whitespace();

            self.start_line = self.line;
            self.start_col = self.col;

            match self.next_token() {
                Ok(token) => tokens.push(Spanned::<Token>::new(token, Span {
                    start_line: self.start_line,
                    end_line: self.line,
                    start_col: self.start_col,
                    end_col: self.col,
                })),
                Err(err) => errors.push(err),
            };
        }

        tokens.push(Spanned::<Token>::new(Token::EOF, Span {
            start_line: 0,
            end_line: 0,
            start_col: 0,
            end_col: 0,
        }));

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
    use pretty_assertions::assert_eq;

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

    #[test]
    fn lex() {
        fn new(kind: Token, start_line: u32, start_col: u32, end_line: u32, end_col: u32) -> Spanned<Token> {
            Spanned::<Token>::new(kind, Span {
                start_line,
                start_col,
                end_line,
                end_col,
            })
        }

        let lexer = Lexer::new("let b = 1 + 2\n678 * (345 - 10005) /123 + abc");
        let tokens = lexer.lex().unwrap();
        let expected = vec![
            new(Token::Let,                 0,  0, 0,  3),
            new(Token::Identifier("b"),     0,  4, 0,  5),
            new(Token::Assign,              0,  6, 0,  7),
            new(Token::Number(1),           0,  8, 0,  9),
            new(Token::Add,                 0, 10, 0, 11),
            new(Token::Number(2),           0, 12, 0, 13),
            new(Token::Number(678),         1,  0, 1,  3),
            new(Token::Asterisk,            1,  4, 1,  5),
            new(Token::Lparen,              1,  6, 1,  7),
            new(Token::Number(345),         1,  7, 1, 10),
            new(Token::Sub,                 1, 11, 1, 12),
            new(Token::Number(10005),       1, 13, 1, 18),
            new(Token::Rparen,              1, 18, 1, 19),
            new(Token::Div,                 1, 20, 1, 21),
            new(Token::Number(123),         1, 21, 1, 24),
            new(Token::Add,                 1, 25, 1, 26),
            new(Token::Identifier("abc"),   1, 27, 1, 30),
            new(Token::EOF,                 0, 0, 0, 0),
        ];

        for (token, expected) in tokens.into_iter().zip(expected.into_iter()) {
            assert_eq!(token, expected);
        }
    }
}
