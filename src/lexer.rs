use std::str::Chars;
use std::iter::Peekable;
use crate::error::Error;
use crate::span::{Span, Spanned};
use crate::token::*;
use crate::id::{Id, IdMap};

fn is_identifier_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

pub struct Lexer<'a> {
    file: Id,
    raw: &'a str,
    input: Peekable<Chars<'a>>,
    errors: Vec<Error>,
    start_line: u32,
    start_col: u32,
    line: u32,
    col: u32,
    // To make slice for identifier
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(s: &'a str, file: Id) -> Lexer<'a> {
        Self {
            file,
            raw: s,
            input: s.chars().peekable(),
            errors: Vec::new(),
            start_line: 0,
            start_col: 0,
            line: 0,
            col: 0,
            pos: 0,
        }
    }

    fn error(&mut self, msg: &str) {
        let error = Error::new(msg, Span {
            file: self.file,
            start_line: self.start_line,
            start_col: self.start_col,
            end_line: self.line,
            end_col: self.col,
        });
        self.errors.push(error);
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

    fn next_is(&mut self, ch: char) -> bool {
        match self.input.peek() {
            Some(c) => ch == *c,
            None => false,
        }
    }

    fn two_char(&mut self, token: Token) -> Option<Token> {
        self.read_char();
        Some(token)
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

    fn lex_number(&mut self, start_char: char) -> Token {
        let mut n = i64::from(start_char.to_digit(10).unwrap());
        loop {
            match self.peek() {
                c if c.is_digit(10) => {
                    n = n * 10 + i64::from(c.to_digit(10).unwrap());
                    self.read_char();
                },
                _ => {
                    break;
                }
            }
        }

        Token::Number(n)
    }

    fn lex_identifier(&mut self, c: char) -> Token {
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
            "fn" => Token::Fn,
            "int" => Token::Int,
            "bool" => Token::Bool,
            "string" => Token::StringType,
            "return" => Token::Return,
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "true" => Token::True,
            "false" => Token::False,
            s => {
                let id = IdMap::new_id(s);
                Token::Identifier(id)
            }
        }
    }

    fn lex_string(&mut self) -> Option<Token> {
        let mut s = String::new();
        while self.peek() != '"' && self.peek() != '\0' {
            if self.peek() == '\\' {
                self.read_char();
                match self.read_char(){
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    // ASCII 7-bit character code
                    'x' => {
                        let mut n = 0u32;
                        let mut count = 0;
                        while self.peek().is_digit(16) && count < 2 {
                            n = n * 16 + self.peek().to_digit(16).unwrap();
                            count += 1;
                            self.read_char();
                        }

                        if count == 2 && n <= 0x7f {
                            s.push(std::char::from_u32(n as u32).unwrap());
                        } else {
                            self.error(&format!("invalid character code '{:x}'", n));
                        }
                    },
                    ch => {
                        self.error(&format!("unknown escape sequence '\\{}'", ch));
                        return None;
                    },
                };
            } else {
                s.push(self.peek());
                self.read_char();
            }
        }

        if self.peek() == '\0' {
            self.error("unexpected EOF");
            None
        } else {
            self.read_char();
            Some(Token::String(s))
        }
    }

    fn skip_comment(&mut self) {
        if self.peek() == '#' {
            // multi-line comment
            self.read_char();
            while self.read_char() != '#' || self.read_char() != '#' {
                if self.peek() == '\0' {
                    break;
                }
            }
        } else {
            // single-line comment
            while self.read_char() != '\n' {
                if self.peek() == '\0' {
                    break;
                }
            }
        }
    }

    fn next_token(&mut self) -> Option<Token> {
        match self.read_char() {
            c if c.is_digit(10) => Some(self.lex_number(c)),
            c if is_identifier_char(c) => Some(self.lex_identifier(c)),
            '"' => self.lex_string(),
            '+' => Some(Token::Add),
            '-' => Some(Token::Sub),
            '*' => Some(Token::Asterisk),
            '/' => Some(Token::Div),
            '(' => Some(Token::Lparen),
            ')' => Some(Token::Rparen),
            '{' => Some(Token::Lbrace),
            '}' => Some(Token::Rbrace),
            '=' => Some(Token::Equal),
            ';' => Some(Token::Semicolon),
            ',' => Some(Token::Comma),
            ':' if self.next_is('=') => self.two_char(Token::Assign),
            ':' => Some(Token::Colon),
            '<' if self.next_is('=') => self.two_char(Token::LessThanOrEqual),
            '<' => Some(Token::LessThan),
            '>' if self.next_is('=') => self.two_char(Token::GreaterThanOrEqual),
            '>' => Some(Token::GreaterThan),
            '&' if self.next_is('&') => self.two_char(Token::And),
            '&' => Some(Token::Ampersand),
            '|' if self.next_is('|') => self.two_char(Token::Or),
            '!' if self.next_is('=') => self.two_char(Token::NotEqual),
            '.' => Some(Token::Dot),
            '#' => {
                self.skip_comment();
                None
            },
            c => {
                self.error(&format!("Invalid character `{}`", c));
                None
            },
        }
    }

    pub fn lex(mut self) -> Result<Vec<Spanned<Token>>, Vec<Error>> {
        let mut tokens = Vec::new();

        self.skip_whitespace();

        while self.peek() != '\0' {
            self.start_line = self.line;
            self.start_col = self.col;

            if let Some(token) = self.next_token() {
                tokens.push(Spanned::<Token>::new(token, Span {
                    file: self.file,
                    start_line: self.start_line,
                    end_line: self.line,
                    start_col: self.start_col,
                    end_col: self.col,
                }));
            }

            self.skip_whitespace();
        }

        tokens.push(Spanned::<Token>::new(Token::EOF, Span {
            file: self.file,
            start_line: 0,
            end_line: 0,
            start_col: 0,
            end_col: 0,
        }));

        if !self.errors.is_empty() {
            Err(self.errors)
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
        let file = IdMap::new_id("test.lang2");
        let lexer = Lexer::new("あ あ", file);
        let errors = lexer.lex().unwrap_err();
        let expected = vec![
            Error::new("Invalid character `あ`", Span {
                file,
                start_line: 0,
                start_col: 0,
                end_line: 0,
                end_col: 1,
            }),
            Error::new("Invalid character `あ`", Span {
                file,
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
                file: IdMap::new_id("test.lang2"),
                start_line,
                start_col,
                end_line,
                end_col,
            })
        }

        let lexer = Lexer::new(r#"let b = 1 + 2
678 * (345 - 10005) /123 + abc
"abcあいうえお\n\t""#, IdMap::new_id("test.lang2"));
        let tokens = lexer.lex().unwrap();
        let expected = vec![
            new(Token::Let,                 0,  0, 0,  3),
            new(Token::Identifier(IdMap::new_id("b")), 0,  4, 0,  5),
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
            new(Token::Identifier(IdMap::new_id("abc")), 1, 27, 1, 30),
            new(Token::String(String::from("abcあいうえお\n\t")), 2, 0, 2, 14),
            new(Token::EOF,                 0, 0, 0, 0),
        ];

        for (token, expected) in tokens.into_iter().zip(expected.into_iter()) {
            assert_eq!(token, expected);
        }
    }
}
