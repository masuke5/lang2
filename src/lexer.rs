use crate::error::{Error, ErrorList};
use crate::id::{Id, IdMap};
use crate::span::{Span, Spanned};
use crate::token::*;
use std::iter::Peekable;
use std::str::Chars;

fn is_identifier_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn make_compound_assignment_operator(c: char) -> Option<Token> {
    match c {
        '+' => Some(Token::AddAssign),
        '-' => Some(Token::SubAssign),
        '*' => Some(Token::MulAssign),
        '/' => Some(Token::DivAssign),
        _ => None,
    }
}

pub struct Lexer<'a> {
    file: Id,
    raw: &'a str,
    input: Peekable<Chars<'a>>,
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
            start_line: 0,
            start_col: 0,
            line: 0,
            col: 0,
            pos: 0,
        }
    }

    fn error(&mut self, msg: &str) {
        let span = Span {
            file: self.file,
            start_line: self.start_line,
            start_col: self.start_col,
            end_line: self.line,
            end_col: self.col,
        };
        let error = Error::new(msg, &span);
        ErrorList::push(error);
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

        Token::Number(n)
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

        match &self.raw[start_pos..self.pos] {
            "let" => Token::Let,
            "mut" => Token::Mut,
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
            "type" => Token::Type,
            "struct" => Token::Struct,
            "import" => Token::Import,
            "as" => Token::As,
            "impl" => Token::Impl,
            "__null__" => Token::Null,
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
                match self.read_char() {
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
                            s.push(std::char::from_u32(n).unwrap());
                        } else {
                            self.error(&format!("invalid character code '{:x}'", n));
                        }
                    }
                    ch => {
                        self.error(&format!("unknown escape sequence '\\{}'", ch));
                        return None;
                    }
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
            c if self.next_is('=') && make_compound_assignment_operator(c).is_some() => {
                self.two_char(make_compound_assignment_operator(c).unwrap())
            }
            '"' => self.lex_string(),
            '+' => Some(Token::Add),
            '-' if self.next_is('>') => self.two_char(Token::Arrow),
            '-' => Some(Token::Sub),
            '*' => Some(Token::Asterisk),
            '/' => Some(Token::Div),
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
            '<' => Some(Token::LessThan),
            '>' if self.next_is('=') => self.two_char(Token::GreaterThanOrEqual),
            '>' => Some(Token::GreaterThan),
            '&' if self.next_is('&') => self.two_char(Token::And),
            '&' => Some(Token::Ampersand),
            '|' if self.next_is('|') => self.two_char(Token::Or),
            '.' if self.next_is('<') => self.two_char(Token::LTypeArgs),
            '.' if self.next_is('.') => self.two_char(Token::DoubleDot),
            '.' => Some(Token::Dot),
            '#' => {
                self.skip_comment();
                None
            }
            c => {
                self.error(&format!("Invalid character `{}`", c));
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
                tokens.push(Spanned::<Token>::new(
                    token,
                    Span {
                        file: self.file,
                        start_line: self.start_line,
                        end_line: self.line,
                        start_col: self.start_col,
                        end_col: self.col,
                    },
                ));
            }

            self.skip_whitespace();
        }

        tokens.push(Spanned::<Token>::new(Token::EOF, Span::zero(self.file)));

        tokens
    }
}
