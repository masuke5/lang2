use std::fmt;

#[derive(Debug, PartialEq)]
pub enum Token<'a> {
    Number(i64),
    Identifier(&'a str),
    Add,
    Sub,
    Asterisk,
    Div,
    EOF,
    Lparen,
    Rparen,
}

impl<'a> fmt::Display for Token<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Number(n) => write!(f, "{}", n),
            Token::Identifier(ident) => write!(f, "{}", ident),
            Token::Add => f.write_str("+"),
            Token::Sub => f.write_str("-"),
            Token::Asterisk => f.write_str("*"),
            Token::Div => f.write_str("/"),
            Token::EOF => f.write_str("EOF"),
            Token::Lparen => f.write_str("("),
            Token::Rparen => f.write_str(")"),
        }
    }
}
