mod span;
mod error;
mod token;
mod lexer;
mod ast;
mod parser;

use std::env;
use std::process::exit;
use lexer::Lexer;
use span::{Span, Spanned};
use token::Token;
use error::Error;
use parser::Parser;
use ast::*;

fn print_usage() {
    println!("usage: lang2 <input>");
}

fn span_to_string(span: &Span) -> String {
    format!("\x1b[33m{}:{}-{}:{}\x1b[0m", span.start_line, span.start_col, span.end_line, span.end_col)
}

fn dump_token(tokens: Vec<Spanned<Token>>) {
    for token in tokens {
        println!("{} {}:{}-{}:{}", token.kind, token.span.start_line, token.span.start_col, token.span.end_line, token.span.end_col);
    }
}

fn dump_expr(expr: Spanned<Expr>, depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    match expr.kind {
        Expr::Literal(Literal::Number(n)) => println!("{} {}", n, span_to_string(&expr.span)),
        Expr::BinOp(binop, lhs, rhs) => {
            println!("{} {}", binop.to_symbol(), span_to_string(&expr.span));
            dump_expr(*lhs, depth + 1);
            dump_expr(*rhs, depth + 1);
        },
    }
}

fn dump_ast(program: Program) {
    dump_expr(program.expr, 0);
}

fn print_errors(errors: Vec<Error>) {
    for error in errors {
        println!("{:?}", error);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        exit(1);
    }

    let lexer = Lexer::new(&args[1]);
    let tokens = match lexer.lex() {
        Ok(tokens) => tokens,
        Err(errors) => {
            print_errors(errors);
            exit(1);
        },
    };

    let parser = Parser::new(tokens);
    let program = match parser.parse() {
        Ok(program) => program,
        Err(errors) => {
            print_errors(errors);
            exit(1);
        },
    };

    dump_ast(program);
}
