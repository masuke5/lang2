mod span;
mod error;
mod token;
mod lexer;
mod ast;
mod parser;
mod env;
mod executor;

use std::process::exit;
use lexer::Lexer;
use span::{Span, Spanned};
use token::Token;
use error::Error;
use parser::Parser;
use ast::*;
use executor::Executor;

use clap::{Arg, App, ArgMatches};

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

fn dump_stmt(stmt: Spanned<Stmt>, depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    match stmt.kind {
        Stmt::Bind(name, expr) => {
            println!("let {} =", name);
            dump_expr(expr, depth + 1);
        },
        Stmt::Expr(expr) => {
            dump_expr(expr, depth);
        },
        _ => {
            unimplemented!();
        },
    }
}

fn dump_ast(program: Program) {
    for stmt in program.stmt {
        dump_stmt(stmt, 0);
    }
}

fn print_errors(input: String, errors: Vec<Error>) {
    let input: Vec<&str> = input.lines().collect();
    // Line number string length (Example: 123 = 3, 123456 = 6)
    let line_num_len = format!("{}", input.len()).len();

    for error in errors {
        // Print the line number
        print!("\x1b[96m{:<width$} | \x1b[0m", error.span.start_line, width = line_num_len);
        // Print the line
        println!("{}", input[error.span.start_line as usize]);
        // Print the error span
        print!("\x1b[96m{} | \x1b[91m{}", " ".repeat(line_num_len), " ".repeat(error.span.start_col as usize));
        print!("{} ", "~".repeat((error.span.end_col - error.span.start_col) as usize));
        // Print the error message
        println!("{}\x1b[0m", error.msg);
    }
}

fn execute(matches: ArgMatches, input: &str) -> Result<(), Vec<Error>> {
    let lexer = Lexer::new(input);
    let tokens = lexer.lex()?;
    if matches.is_present("dump-token") {
        dump_token(tokens);
        exit(1);
    }

    let parser = Parser::new(tokens);
    let program = parser.parse()?;
    if matches.is_present("dump-ast") {
        dump_ast(program);
        exit(1);
    }

    let mut executor = Executor::new();
    let result = executor.exec(program);

    println!("{}", result);

    Ok(())
}

fn main() {
    let matches = App::new("lang2")
        .version("0.0")
        .author("masuke5 <s.zerogoichi@gmail.com>")
        .about("lang2 interpreter")
        .arg(Arg::with_name("cmd")
             .short("c")
             .long("cmd")
             .help("Runs string")
             .takes_value(true))
        .arg(Arg::with_name("dump-token")
             .long("dump-token")
             .help("Dumps tokens"))
        .arg(Arg::with_name("dump-ast")
             .long("dump-ast")
             .help("Dumps AST"))
        .get_matches();

    let cmd = matches.value_of("cmd").unwrap().to_string();
    if let Err(errors) = execute(matches, &cmd) {
        print_errors(cmd, errors);
        exit(1);
    }
}
