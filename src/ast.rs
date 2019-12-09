use crate::span::Spanned;
use crate::ty::Type;
use crate::id::{Id, IdMap};
use crate::utils::{escape_string, span_to_string};

#[derive(Debug, PartialEq, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    And,
    Or,
}

impl BinOp {
    pub fn to_symbol(&self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Equal => "==",
            BinOp::NotEqual => "!=",
            BinOp::LessThan => "<",
            BinOp::LessThanOrEqual => "<=",
            BinOp::GreaterThan => ">",
            BinOp::GreaterThanOrEqual => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Number(i64),
    String(usize),
    Unit,
    True,
    False,
    Null,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Field {
    Number(usize),
    Id(Id),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Literal(Literal),
    Tuple(Vec<Spanned<Expr>>),
    Struct(Id, Vec<(Spanned<Id>, Spanned<Expr>)>),
    Array(Box<Spanned<Expr>>, usize),
    Field(Box<Spanned<Expr>>, Field),
    Subscript(Box<Spanned<Expr>>, Box<Spanned<Expr>>),
    BinOp(BinOp, Box<Spanned<Expr>>, Box<Spanned<Expr>>),
    Variable(Id),
    Call(Id, Vec<Spanned<Expr>>),
    Dereference(Box<Spanned<Expr>>),
    Address(Box<Spanned<Expr>>, bool),
    Negative(Box<Spanned<Expr>>),
    Alloc(Box<Spanned<Expr>>, bool),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Stmt {
    Bind(Id, Spanned<Expr>, bool),
    Expr(Spanned<Expr>),
    Block(Vec<Spanned<Stmt>>),
    Return(Option<Spanned<Expr>>),
    If(Spanned<Expr>, Box<Spanned<Stmt>>, Option<Box<Spanned<Stmt>>>),
    While(Spanned<Expr>, Box<Spanned<Stmt>>),
    Assign(Spanned<Expr>, Spanned<Expr>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum TopLevel {
    Stmt(Spanned<Stmt>),
    Function(Id, Vec<(Id, Type, bool)>, Type, Spanned<Stmt>),
    Type(Id, Type),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub top: Vec<Spanned<TopLevel>>,
    pub strings: Vec<String>,
}

pub fn dump_expr(expr: &Spanned<Expr>, strings: &[String], depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    match &expr.kind {
        Expr::Literal(Literal::Number(n)) => println!("{} {}", n, span_to_string(&expr.span)),
        Expr::Literal(Literal::String(i)) => println!("\"{}\" {}", escape_string(&strings[*i]), span_to_string(&expr.span)),
        Expr::Literal(Literal::Unit) => println!("() {}", span_to_string(&expr.span)),
        Expr::Literal(Literal::True) => println!("true {}", span_to_string(&expr.span)),
        Expr::Literal(Literal::False) => println!("false {}", span_to_string(&expr.span)),
        Expr::Literal(Literal::Null) => println!("__null__ {}", span_to_string(&expr.span)),
        Expr::Tuple(exprs) => {
            println!("tuple {}", span_to_string(&expr.span));
            for expr in exprs {
                dump_expr(&expr, strings, depth + 1);
            }
        },
        Expr::Struct(id, fields) => {
            println!("struct {} {}", IdMap::name(*id), span_to_string(&expr.span));
            for (name, expr) in fields {
                print!("{}", "  ".repeat(depth + 1));
                println!("{}: {}", IdMap::name(name.kind), span_to_string(&name.span));
                dump_expr(expr, strings, depth + 2);
            }
        },
        Expr::Array(init_expr, size) => {
            println!("[{}] {}", size, span_to_string(&expr.span));
            dump_expr(init_expr, strings, depth + 1);
        },
        Expr::Field(expr, field) => {
            match field {
                Field::Number(i) => println!(".{} {}", i, span_to_string(&expr.span)),
                Field::Id(id) => println!(".{} {}", IdMap::name(*id), span_to_string(&expr.span)),
            };

            dump_expr(&expr, strings, depth + 1);
        },
        Expr::Subscript(expr, subscript) => {
            println!("subscript {}", span_to_string(&expr.span));
            dump_expr(&expr, strings, depth + 1);
            dump_expr(&subscript, strings, depth + 1);
        },
        Expr::Variable(name) => println!("{} {}", IdMap::name(*name), span_to_string(&expr.span)),
        Expr::BinOp(binop, lhs, rhs) => {
            println!("{} {}", binop.to_symbol(), span_to_string(&expr.span));
            dump_expr(&lhs, strings, depth + 1);
            dump_expr(&rhs, strings, depth + 1);
        },
        Expr::Call(name, args) => {
            println!("{} {}", IdMap::name(*name), span_to_string(&expr.span));
            for arg in args {
                dump_expr(&arg, strings, depth + 1);
            }
        },
        Expr::Address(expr_, is_mutable) => {
            println!("&{} {}", if *is_mutable { "mut" } else { "" }, span_to_string(&expr.span));
            dump_expr(&expr_, strings, depth + 1);
        },
        Expr::Dereference(expr_) => {
            println!("* {}", span_to_string(&expr.span));
            dump_expr(&expr_, strings, depth + 1);
        },
        Expr::Negative(expr_) => {
            println!("neg {}", span_to_string(&expr.span));
            dump_expr(&expr_, strings, depth + 1);
        },
        Expr::Alloc(expr_, is_mutable) => {
            println!("alloc{} {}", if *is_mutable { " mut" } else { "" }, span_to_string(&expr.span));
            dump_expr(&expr_, strings, depth + 1);
        },
    }
}

pub fn dump_stmt(stmt: &Spanned<Stmt>, strings: &[String], depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    match &stmt.kind {
        Stmt::Bind(name, expr, is_mutable) => {
            println!("let {}{} =", if *is_mutable { " mut " } else { "" }, IdMap::name(*name));
            dump_expr(&expr, strings, depth + 1);
        },
        Stmt::Assign(lhs, rhs) => {
            println!(":= {}", span_to_string(&stmt.span));
            dump_expr(&lhs, strings, depth + 1);
            dump_expr(&rhs, strings, depth + 1);
        },
        Stmt::Expr(expr) => {
            dump_expr(&expr, strings, depth);
        },
        Stmt::Block(stmts) => {
            println!("block {}", span_to_string(&stmt.span));
            for stmt in stmts {
                dump_stmt(&stmt, strings, depth + 1);
            }
        },
        Stmt::Return(expr) => {
            println!("return {}", span_to_string(&stmt.span));
            if let Some(expr) = expr {
                dump_expr(&expr, strings, depth + 1);
            }
        },
        Stmt::If(cond, body, else_stmt) => {
            println!("if {}", span_to_string(&stmt.span));
            dump_expr(&cond, strings, depth + 1);
            dump_stmt(&body, strings, depth + 1);
            if let Some(else_stmt) = else_stmt {
                dump_stmt(&else_stmt, strings, depth + 1);
            }
        },
        Stmt::While(cond, body) => {
            println!("while {}", span_to_string(&stmt.span));
            dump_expr(&cond, strings, depth + 1);
            dump_stmt(&body, strings, depth + 1);
        },
    }
}

pub fn dump_toplevel(toplevel: &Spanned<TopLevel>, strings: &[String]) {
    match &toplevel.kind {
        TopLevel::Stmt(stmt) => dump_stmt(&stmt, strings, 0),
        TopLevel::Function(name, params, return_ty, body) => {
            println!("fn {}({}): {:?} {}", IdMap::name(*name), params.len(), return_ty, span_to_string(&toplevel.span));
            dump_stmt(&body, strings, 1);
        },
        TopLevel::Type(name, ty) => {
            println!(
                "type {} {} {}",
                IdMap::name(*name),
                ty,
                span_to_string(&toplevel.span)
            );
        },
    }
}

pub fn dump_ast(program: &Program) {
    for toplevel in &program.top {
        dump_toplevel(&toplevel, &program.strings);
    }
}
