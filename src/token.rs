use std::fmt;

#[derive(Debug, PartialEq, Clone)]
pub enum Token<'a> {
    Number(i64),
    Identifier(&'a str),
    Let,
    Fn,
    Int,
    Return,
    If,
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

impl<'a> fmt::Display for Token<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Number(n) => write!(f, "{}", n),
            Token::Identifier(ident) => write!(f, "{}", ident),
            Token::Let => f.write_str("let"),
            Token::Fn => f.write_str("fn"),
            Token::Int => f.write_str("int"),
            Token::Return => f.write_str("return"),
            Token::If => f.write_str("if"),
            Token::Add => f.write_str("+"),
            Token::Sub => f.write_str("-"),
            Token::Asterisk => f.write_str("*"),
            Token::Div => f.write_str("/"),
            Token::EOF => f.write_str("EOF"),
            Token::Lparen => f.write_str("("),
            Token::Rparen => f.write_str(")"),
            Token::Lbrace => f.write_str("{"),
            Token::Rbrace => f.write_str("}"),
            Token::Assign => f.write_str("="),
            Token::Semicolon => f.write_str(";"),
            Token::Comma => f.write_str(","),
            Token::Colon => f.write_str(":"),
            Token::Equal => f.write_str("=="),
            Token::NotEqual => f.write_str("!="),
            Token::LessThan => f.write_str("<"),
            Token::LessThanOrEqual => f.write_str("<="),
            Token::GreaterThan => f.write_str(">"),
            Token::GreaterThanOrEqual => f.write_str(">="),
        }
    }
}
