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

fn dump_ast(program: Program) {
    dump_expr(program.expr, 0);
}

fn print_errors(errors: Vec<Error>) {
    for error in errors {
        println!("{:?}", error);
    }
}

fn execute(matches: ArgMatches) -> Result<(), Vec<Error>> {
    let cmd = matches.value_of("cmd").unwrap();

    let lexer = Lexer::new(&cmd);
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

    if let Err(errors) = execute(matches) {
        print_errors(errors);
        exit(1);
    }
}
