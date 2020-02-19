use std::fmt;
use crate::id::{Id, IdMap};
use crate::span::Spanned;

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Number(i64),
    String(String),
    Identifier(Id),
    True,
    False,
    Null,
    Let,
    Mut,
    Fn,
    Int,
    Bool,
    StringType,
    Return,
    If,
    Else,
    While,
    Type,
    Struct,
    New,
    Import,
    As,
    Add,
    Sub,
    Asterisk,
    Div,
    EOF,
    Lparen,
    Rparen,
    Lbrace,
    Rbrace,
    Lbracket,
    Rbracket,
    LTypeArgs,
    Semicolon,
    Comma,
    Colon,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    And,
    Or,
    Dot,
    Assign,
    Ampersand,
    Scope,
    Arrow,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Number(_) => write!(f, "number"),
            Token::String(_) => write!(f, "string"),
            Token::Identifier(_) => write!(f, "identifier"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Null => write!(f, "__null__"),
            Token::Let => write!(f, "let"),
            Token::Mut => write!(f, "mut"),
            Token::Fn => write!(f, "fn"),
            Token::Int => write!(f, "int"),
            Token::Bool => write!(f, "bool"),
            Token::StringType => write!(f, "string"),
            Token::Return => write!(f, "return"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::While => write!(f, "while"),
            Token::Type => write!(f, "type"),
            Token::Struct => write!(f, "struct"),
            Token::New => write!(f, "new"),
            Token::Import => write!(f, "import"),
            Token::As => write!(f, "as"),
            Token::Add => write!(f, "+"),
            Token::Sub => write!(f, "-"),
            Token::Asterisk => write!(f, "*"),
            Token::Div => write!(f, "/"),
            Token::EOF => write!(f, "EOF"),
            Token::Lparen => write!(f, "("),
            Token::Rparen => write!(f, ")"),
            Token::Lbrace => write!(f, "{{"),
            Token::Rbrace => write!(f, "}}"),
            Token::Lbracket => write!(f, "["),
            Token::Rbracket => write!(f, "]"),
            Token::LTypeArgs => write!(f, ".<"),
            Token::Semicolon => write!(f, ";"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Equal => write!(f, "="),
            Token::NotEqual => write!(f, "<>"),
            Token::LessThan => write!(f, "<"),
            Token::LessThanOrEqual => write!(f, "<="),
            Token::GreaterThan => write!(f, ">"),
            Token::GreaterThanOrEqual => write!(f, ">="),
            Token::And => write!(f, "&&"),
            Token::Or => write!(f, "||"),
            Token::Dot => write!(f, "."),
            Token::Assign => write!(f, ":="),
            Token::Ampersand => write!(f, "&"),
            Token::Scope => write!(f, "::"),
            Token::Arrow => write!(f, "->"),
        }
    }
}

impl Token {
    fn detail(&self) -> String {
        match self {
            Token::Number(n) => format!("{}", n),
            Token::Identifier(id) => format!("`{}`", IdMap::name(*id)),
            Token::String(s) => format!("\"{}\"", s),
            token => format!("{}", token),
        }
    }
}

pub fn dump_token(tokens: Vec<Spanned<Token>>) {
    for token in tokens {
        println!("{} {}:{}-{}:{}", token.kind.detail(), token.span.start_line, token.span.start_col, token.span.end_line, token.span.end_col);
    }
}

