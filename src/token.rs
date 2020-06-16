use crate::id::{Id, IdMap};
use crate::span::Spanned;
use std::fmt;

#[derive(Debug, PartialEq, Clone)]
pub enum Keyword {
    True,
    False,
    Null,
    Let,
    Mut,
    Fn,
    Int,
    Bool,
    StringType,
    Char,
    Return,
    If,
    Else,
    While,
    Type,
    Struct,
    Import,
    Impl,
    As,
    Not,
}

impl Keyword {
    pub fn to_str(&self) -> &'static str {
        match self {
            Self::True => "true",
            Self::False => "false",
            Self::Null => "__null__",
            Self::Let => "let",
            Self::Mut => "mut",
            Self::Fn => "fn",
            Self::Int => "int",
            Self::Bool => "bool",
            Self::StringType => "string",
            Self::Char => "char",
            Self::Return => "return",
            Self::If => "if",
            Self::Else => "else",
            Self::While => "while",
            Self::Type => "type",
            Self::Struct => "struct",
            Self::Import => "import",
            Self::Impl => "impl",
            Self::As => "as",
            Self::Not => "not",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "true" => Self::True,
            "false" => Self::False,
            "__null__" => Self::Null,
            "let" => Self::Let,
            "mut" => Self::Mut,
            "fn" => Self::Fn,
            "int" => Self::Int,
            "bool" => Self::Bool,
            "string" => Self::StringType,
            "char" => Self::Char,
            "return" => Self::Return,
            "if" => Self::If,
            "else" => Self::Else,
            "while" => Self::While,
            "type" => Self::Type,
            "struct" => Self::Struct,
            "import" => Self::Import,
            "impl" => Self::Impl,
            "as" => Self::As,
            "not" => Self::Not,
            _ => return None,
        })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Number(i64),
    String(String),
    Char(char),
    Identifier(Id),
    Keyword(Keyword),
    Add,
    Sub,
    Asterisk,
    Div,
    Percent,
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
    DoubleDot,
    Assign,
    Ampersand,
    VerticalBar,
    Xor,
    LShift,
    RShift,
    Scope,
    Arrow,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Number(_) => write!(f, "number"),
            Token::String(_) => write!(f, "string"),
            Token::Char(_) => write!(f, "char"),
            Token::Identifier(_) => write!(f, "identifier"),
            Token::Keyword(kw) => write!(f, "{}", kw.to_str()),
            Token::Add => write!(f, "+"),
            Token::Sub => write!(f, "-"),
            Token::Asterisk => write!(f, "*"),
            Token::Div => write!(f, "/"),
            Token::Percent => write!(f, "%"),
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
            Token::DoubleDot => write!(f, ".."),
            Token::Assign => write!(f, ":="),
            Token::Ampersand => write!(f, "&"),
            Token::VerticalBar => write!(f, "|"),
            Token::Xor => write!(f, "^"),
            Token::LShift => write!(f, "<<"),
            Token::RShift => write!(f, ">>"),
            Token::Scope => write!(f, "::"),
            Token::Arrow => write!(f, "->"),
            Token::AddAssign => write!(f, "+="),
            Token::SubAssign => write!(f, "-="),
            Token::MulAssign => write!(f, "*="),
            Token::DivAssign => write!(f, "/="),
            Token::ModAssign => write!(f, "%="),
        }
    }
}

impl Token {
    #[allow(dead_code)]
    pub fn is_open_parenthese(&self) -> bool {
        match self {
            Token::Lparen | Token::Lbracket | Token::Lbrace | Token::LTypeArgs => true,
            _ => false,
        }
    }

    pub fn is_close_parenthese(&self) -> bool {
        match self {
            Token::Rparen | Token::Rbracket | Token::Rbrace | Token::GreaterThan => true,
            _ => false,
        }
    }

    pub fn matching_parenthese(&self) -> Option<Token> {
        match self {
            Token::Lparen => Some(Token::Rparen),
            Token::Lbracket => Some(Token::Rbracket),
            Token::Lbrace => Some(Token::Rbrace),
            Token::LTypeArgs => Some(Token::GreaterThan),
            Token::Rparen => Some(Token::Lparen),
            Token::Rbracket => Some(Token::Lbracket),
            Token::Rbrace => Some(Token::Rbrace),
            Token::GreaterThan => Some(Token::LTypeArgs),
            _ => None,
        }
    }

    fn detail(&self) -> String {
        match self {
            Token::Number(n) => format!("{}", n),
            Token::Identifier(id) => format!("`{}`", IdMap::name(*id)),
            Token::String(s) => format!("\"{}\"", s),
            Token::Char(ch) => format!("'{}'", ch),
            token => format!("{}", token),
        }
    }
}

pub fn dump_token(tokens: Vec<Spanned<Token>>) {
    for token in tokens {
        println!(
            "{} {}:{}-{}:{}",
            token.kind.detail(),
            token.span.start_line,
            token.span.start_col,
            token.span.end_line,
            token.span.end_col
        );
    }
}
