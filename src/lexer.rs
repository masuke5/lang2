use crate::error::{Error, ErrorList};
use crate::id::{Id, IdMap};
use crate::span::{Span, Spanned};
use crate::token::*;
use std::iter::Peekable;
use std::str::Chars;

fn is_identifier_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

pub struct Lexer<'a> {
    file: Id,
    raw: &'a str,
    input: Peekable<Chars<'a>>,
    start_line: u32,
    start_col: u32,
    line: u32,
    col: u32,
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(s: &'a str, file: Id) -> Lexer<'a> {
        Self {
            file,
            raw: s,
            input: s.chars().peekable(),
            start_line: 0,
            start_col: 0,
            line: 0,
            col: 0,
            pos: 0,
        }
    }

    fn span(&self) -> Span {
        Span {
            file: self.file,
            start_line: self.start_line,
            start_col: self.start_col,
            end_line: self.line,
            end_col: self.col,
        }
    }

    fn read_char(&mut self) -> char {
        let c = match self.input.next() {
            Some('\n') => {
                self.line += 1;
                self.col = 0;
                '\n'
            }
            Some(ch) => {
                self.col += 1;
                ch
            }
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

    fn lex_number_with_radix(&mut self, initial: i64, radix: u32) -> Token {
        let mut n = initial;
        loop {
            match self.peek() {
                c if c.is_digit(radix) => {
                    n = n * radix as i64 + i64::from(c.to_digit(radix).unwrap());
                    self.read_char();
                }
                _ => {
                    break;
                }
            }
        }

        if self.peek() == 'u' {
            self.read_char();
            Token::UnsignedNumber(n as u64)
        } else {
            Token::Number(n)
        }
    }

    fn lex_number(&mut self, start_char: char) -> Token {
        match start_char {
            '0' if self.next_is('x') => {
                self.read_char();
                self.lex_number_with_radix(0, 16)
            }
            '0' if self.next_is('b') => {
                self.read_char();
                self.lex_number_with_radix(0, 2)
            }
            '0' => self.lex_number_with_radix(0, 8),
            ch => self.lex_number_with_radix(ch.to_digit(10).unwrap() as i64, 10),
        }
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

        let s = &self.raw[start_pos..self.pos];
        Keyword::from_str(s).map_or_else(
            || {
                let id = IdMap::new_id(s);
                Token::Identifier(id)
            },
            Token::Keyword,
        )
    }

    fn lex_escape_sequence(&mut self) -> Option<char> {
        match self.read_char() {
            '"' => Some('"'),
            '\\' => Some('\\'),
            'n' => Some('\n'),
            'r' => Some('\r'),
            't' => Some('\t'),
            // ASCII 7-bit character code
            'x' => {
                // Read hex number
                let mut n = 0u32;
                let mut count = 0;
                while self.peek().is_digit(16) && count < 2 {
                    n = n * 16 + self.peek().to_digit(16).unwrap();
                    count += 1;
                    self.read_char();
                }

                if count == 2 && n <= 0x7f {
                    Some(std::char::from_u32(n).unwrap())
                } else if count <= 1 {
                    error!(&self.span(), "character code is too short");
                    None
                } else {
                    error!(&self.span(), "invalid character code '{:x}'", n);
                    None
                }
            }
            ch => {
                error!(&self.span(), "unknown escape sequence '\\{}'", ch);
                None
            }
        }
    }

    fn lex_string(&mut self) -> Option<Token> {
        let mut s = String::new();
        while self.peek() != '"' && self.peek() != '\0' {
            if self.peek() == '\\' {
                self.read_char();

                if let Some(ch) = self.lex_escape_sequence() {
                    s.push(ch);
                }
            } else {
                s.push(self.peek());
                self.read_char();
            }
        }

        if self.peek() == '\0' {
            error!(&self.span(), "unexpected EOF");
            None
        } else {
            self.read_char();
            Some(Token::String(s))
        }
    }

    fn lex_char(&mut self) -> Option<Token> {
        let ch = self.read_char();
        let ch = match ch {
            '\'' => {
                error!(&self.span(), "empty character is not allowed");
                return Some(Token::Char('\0'));
            }
            '\\' => match self.lex_escape_sequence() {
                Some(ch) => ch,
                None => {
                    self.read_char();
                    return Some(Token::Char('\0'));
                }
            },
            ch => ch,
        };

        if self.peek() == '\'' {
            self.read_char();
            Some(Token::Char(ch))
        } else {
            error!(&self.span(), "only one character is allowed");
            return Some(Token::Char('\0'));
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
            '\'' => self.lex_char(),
            '+' if self.next_is('=') => self.two_char(Token::AddAssign),
            '-' if self.next_is('=') => self.two_char(Token::SubAssign),
            '*' if self.next_is('=') => self.two_char(Token::MulAssign),
            '/' if self.next_is('=') => self.two_char(Token::DivAssign),
            '%' if self.next_is('=') => self.two_char(Token::ModAssign),
            '+' => Some(Token::Add),
            '-' if self.next_is('>') => self.two_char(Token::Arrow),
            '-' => Some(Token::Sub),
            '*' => Some(Token::Asterisk),
            '/' => Some(Token::Div),
            '%' => Some(Token::Percent),
            '(' => Some(Token::Lparen),
            ')' => Some(Token::Rparen),
            '{' => Some(Token::Lbrace),
            '}' => Some(Token::Rbrace),
            '[' => Some(Token::Lbracket),
            ']' => Some(Token::Rbracket),
            '=' => Some(Token::Equal),
            ';' => Some(Token::Semicolon),
            ',' => Some(Token::Comma),
            ':' if self.next_is('=') => self.two_char(Token::Assign),
            ':' if self.next_is(':') => self.two_char(Token::Scope),
            ':' => Some(Token::Colon),
            '<' if self.next_is('>') => self.two_char(Token::NotEqual),
            '<' if self.next_is('=') => self.two_char(Token::LessThanOrEqual),
            '<' if self.next_is('<') => self.two_char(Token::LShift),
            '<' => Some(Token::LessThan),
            '>' if self.next_is('=') => self.two_char(Token::GreaterThanOrEqual),
            '>' if self.next_is('>') => self.two_char(Token::RShift),
            '>' => Some(Token::GreaterThan),
            '&' if self.next_is('&') => self.two_char(Token::And),
            '&' => Some(Token::Ampersand),
            '|' if self.next_is('|') => self.two_char(Token::Or),
            '|' => Some(Token::VerticalBar),
            '^' => Some(Token::Xor),
            '.' if self.next_is('<') => self.two_char(Token::LTypeArgs),
            '.' if self.next_is('.') => self.two_char(Token::DoubleDot),
            '.' => Some(Token::Dot),
            '#' => {
                self.skip_comment();
                None
            }
            c => {
                error!(&self.span(), "unknown character `{}`", c);
                None
            }
        }
    }

    pub fn lex(mut self) -> Vec<Spanned<Token>> {
        let mut tokens = Vec::new();

        self.skip_whitespace();

        while self.peek() != '\0' {
            self.start_line = self.line;
            self.start_col = self.col;

            if let Some(token) = self.next_token() {
                tokens.push(Spanned::new(token, self.span()));
            }

            self.skip_whitespace();
        }

        tokens.push(Spanned::<Token>::new(Token::EOF, Span::zero(self.file)));

        tokens
    }
}
