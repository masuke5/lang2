use std::fmt;
use crate::id::Id;

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Number(i64),
    String(String),
    Identifier(Id),
    True,
    False,
    Let,
    Fn,
    Int,
    Bool,
    Return,
    If,
    Else,
    While,
    Add,
    Sub,
    Asterisk,
    Div,
    EOF,
    Lparen,
    Rparen,
    Lbrace,
    Rbrace,
    Assign,
    Semicolon,
    Comma,
    Colon,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Number(_) => write!(f, "number"),
            Token::String(_) => write!(f, "string"),
            Token::Identifier(_) => write!(f, "identifier"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Let => write!(f, "let"),
            Token::Fn => write!(f, "fn"),
            Token::Int => write!(f, "int"),
            Token::Bool => write!(f, "bool"),
            Token::Return => write!(f, "return"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::While => write!(f, "while"),
            Token::Add => write!(f, "+"),
            Token::Sub => write!(f, "-"),
            Token::Asterisk => write!(f, "*"),
            Token::Div => write!(f, "/"),
            Token::EOF => write!(f, "EOF"),
            Token::Lparen => write!(f, "("),
            Token::Rparen => write!(f, ")"),
            Token::Lbrace => write!(f, "{{"),
            Token::Rbrace => write!(f, "}}"),
            Token::Assign => write!(f, "="),
            Token::Semicolon => write!(f, ";"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Equal => write!(f, "=="),
            Token::NotEqual => write!(f, "!="),
            Token::LessThan => write!(f, "<"),
            Token::LessThanOrEqual => write!(f, "<="),
            Token::GreaterThan => write!(f, ">"),
            Token::GreaterThanOrEqual => write!(f, ">="),
        }
    }
}
