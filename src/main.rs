#![feature(bind_by_move_pattern_guards)]

mod span;
mod error;
mod token;
mod lexer;
mod ast;
mod parser;
mod env;
mod executor;
mod ty;
mod sema;

use std::process::exit;
use std::fs::File;
use std::io::Read;
use std::borrow::Cow;

use lexer::Lexer;
use span::{Span, Spanned};
use token::Token;
use error::Error;
use parser::Parser;
use ast::*;
use executor::Executor;
use sema::Analyzer;

use clap::{Arg, App, ArgMatches};

fn span_to_string(span: &Span) -> String {
    format!("\x1b[33m{}:{}-{}:{}\x1b[0m", span.start_line, span.start_col, span.end_line, span.end_col)
}

fn dump_token(tokens: Vec<Spanned<Token>>) {
    for token in tokens {
        println!("{} {}:{}-{}:{}", token.kind, token.span.start_line, token.span.start_col, token.span.end_line, token.span.end_col);
    }
}

fn dump_expr(expr: SpannedTyped<Expr>, depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    match expr.kind {
        Expr::Literal(Literal::Number(n)) => println!("{} {}", n, span_to_string(&expr.span)),
        Expr::Literal(Literal::True) => println!("true {}", span_to_string(&expr.span)),
        Expr::Literal(Literal::False) => println!("false {}", span_to_string(&expr.span)),
        Expr::Variable(name) => println!("{} {}", name, span_to_string(&expr.span)),
        Expr::BinOp(binop, lhs, rhs) => {
            println!("{} {}", binop.to_symbol(), span_to_string(&expr.span));
            dump_expr(*lhs, depth + 1);
            dump_expr(*rhs, depth + 1);
        },
        Expr::Call(name, args) => {
            println!("{} {}", name, span_to_string(&expr.span));
            for arg in args {
                dump_expr(arg, depth + 1);
            }
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
        Stmt::Block(stmts) => {
            for stmt in stmts {
                dump_stmt(stmt, depth);
            }
        },
        Stmt::Return(expr) => {
            println!("return {}", span_to_string(&stmt.span));
            dump_expr(expr, depth + 1);
        },
        Stmt::If(cond, body) => {
            println!("if {}", span_to_string(&stmt.span));
            dump_expr(cond, depth + 1);
            dump_stmt(*body, depth + 1);
        },
    }
}

fn dump_toplevel(toplevel: Spanned<TopLevel>) {
    match toplevel.kind {
        TopLevel::Stmt(stmt) => dump_stmt(stmt, 0),
        TopLevel::Function(name, params, return_ty, body) => {
            println!("fn {}({}): {:?} {}", name, params.len(), return_ty, span_to_string(&toplevel.span));
            dump_stmt(body, 1);
        },
    }
}

fn dump_ast(program: Program) {
    for toplevel in program.top {
        dump_toplevel(toplevel);
    }
}

fn print_errors(input: &str, errors: Vec<Error>) {
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

fn execute(matches: &ArgMatches, input: &str) -> Result<(), Vec<Error>> {
    let lexer = Lexer::new(input);
    let tokens = lexer.lex()?;
    if matches.is_present("dump-token") {
        dump_token(tokens);
        exit(1);
    }

    let parser = Parser::new(tokens);
    let mut program = parser.parse()?;
    if matches.is_present("dump-ast") {
        dump_ast(program);
        exit(1);
    }

    let analyzer = Analyzer::new();
    analyzer.analyze(&mut program)?;

    let mut executor = Executor::new();
    executor.exec(program);

    Ok(())
}

fn get_input<'a>(matches: &'a ArgMatches) -> Result<Cow<'a, str>, String> {
    if let Some(filepath) = matches.value_of("file") {
        let mut file = File::open(filepath).map_err(|err| format!("{}", err))?;
        let mut input = String::new();
        file.read_to_string(&mut input).map_err(|err| format!("{}", err))?;

        Ok(input.into())
    } else if let Some(input) = matches.value_of("cmd") {
        Ok(input.into())
    } else {
        Err(String::from("Not specified file or cmd"))
    }
}

fn main() {
    let matches = App::new("lang2")
        .version("0.0")
        .author("masuke5 <s.zerogoichi@gmail.com>")
        .about("lang2 interpreter")
        .arg(Arg::with_name("file")
            .help("Runs file")
            .index(1))
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

    let input = match get_input(&matches) {
        Ok(input) => input,
        Err(err) => {
            eprintln!("Unable to load input: {}", err);
            exit(1);
        },
    };

    if let Err(errors) = execute(&matches, &input) {
        print_errors(&input, errors);
        exit(1);
    }
}
