mod span;
mod error;
mod token;
mod lexer;
mod ast;
mod parser;
mod ty;
mod sema;
mod id;
mod inst;
mod vm;
mod value;
mod stdlib;

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
// use executor::Executor;
use sema::Analyzer;
use id::IdMap;
use vm::VM;

use clap::{Arg, App, ArgMatches};

fn span_to_string(span: &Span) -> String {
    format!("\x1b[33m{}:{}-{}:{}\x1b[0m", span.start_line, span.start_col, span.end_line, span.end_col)
}

fn dump_token(tokens: Vec<Spanned<Token>>) {
    for token in tokens {
        println!("{} {}:{}-{}:{}", token.kind, token.span.start_line, token.span.start_col, token.span.end_line, token.span.end_col);
    }
}

fn dump_expr(id_map: &IdMap, expr: Spanned<Expr>, depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    match expr.kind {
        Expr::Literal(Literal::Number(n)) => println!("{} {}", n, span_to_string(&expr.span)),
        Expr::Literal(Literal::String(s)) => println!("\"{}\" {}", s, span_to_string(&expr.span)),
        Expr::Literal(Literal::True) => println!("true {}", span_to_string(&expr.span)),
        Expr::Literal(Literal::False) => println!("false {}", span_to_string(&expr.span)),
        Expr::Tuple(exprs) => {
            println!("tuple {}", span_to_string(&expr.span));
            for expr in exprs {
                dump_expr(id_map, expr, depth + 1);
            }
        },
        Expr::Field(expr, field) => {
            match field {
                Field::Number(i) => println!(".{} {}", i, span_to_string(&expr.span)),
            };

            dump_expr(id_map, *expr, depth + 1);
        },
        Expr::Variable(name) => println!("{} {}", id_map.name(&name), span_to_string(&expr.span)),
        Expr::BinOp(binop, lhs, rhs) => {
            println!("{} {}", binop.to_symbol(), span_to_string(&expr.span));
            dump_expr(id_map, *lhs, depth + 1);
            dump_expr(id_map, *rhs, depth + 1);
        },
        Expr::Call(name, args) => {
            println!("{} {}", id_map.name(&name), span_to_string(&expr.span));
            for arg in args {
                dump_expr(id_map, arg, depth + 1);
            }
        },
    }
}

fn dump_stmt(id_map: &IdMap, stmt: Spanned<Stmt>, depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    match stmt.kind {
        Stmt::Bind(name, expr) => {
            println!("let {} =", id_map.name(&name));
            dump_expr(id_map, expr, depth + 1);
        },
        Stmt::Expr(expr) => {
            dump_expr(id_map, expr, depth);
        },
        Stmt::Block(stmts) => {
            println!("block {}", span_to_string(&stmt.span));
            for stmt in stmts {
                dump_stmt(id_map, stmt, depth + 1);
            }
        },
        Stmt::Return(expr) => {
            println!("return {}", span_to_string(&stmt.span));
            dump_expr(id_map, expr, depth + 1);
        },
        Stmt::If(cond, body, else_stmt) => {
            println!("if {}", span_to_string(&stmt.span));
            dump_expr(id_map, cond, depth + 1);
            dump_stmt(id_map, *body, depth + 1);
            if let Some(else_stmt) = else_stmt {
                dump_stmt(id_map, *else_stmt, depth + 1);
            }
        },
        Stmt::While(cond, body) => {
            println!("while {}", span_to_string(&stmt.span));
            dump_expr(id_map, cond, depth + 1);
            dump_stmt(id_map, *body, depth + 1);
        },
    }
}

fn dump_toplevel(id_map: &IdMap, toplevel: Spanned<TopLevel>) {
    match toplevel.kind {
        TopLevel::Stmt(stmt) => dump_stmt(id_map, stmt, 0),
        TopLevel::Function(name, params, return_ty, body) => {
            println!("fn {}({}): {:?} {}", id_map.name(&name), params.len(), return_ty, span_to_string(&toplevel.span));
            dump_stmt(id_map, body, 1);
        },
    }
}

fn dump_ast(id_map: &IdMap, program: Program) {
    for toplevel in program.top {
        dump_toplevel(id_map, toplevel);
    }
}

fn print_errors(input: &str, errors: Vec<Error>) {
    let input: Vec<&str> = input.lines().collect();

    for error in errors {
        // Print the error position and message
        let es = error.span;
        println!("\x1b[91merror\x1b[0m:{}:{}-{}:{}: \x1b[97m{}\x1b[0m", es.start_line, es.start_col, es.end_line, es.end_col, error.msg);

        // Print the lines
        let line_count = es.end_line - es.start_line + 1;
        for i in 0..line_count {
            let line = (es.start_line + i) as usize;
            let line_len = if line >= input.len() as usize { 0 } else { input[line].len() as u32 };
            println!("{}", if line >= input.len() as usize { "" } else { input[line] });

            // Print the error span
            let (start, length) = if line_count == 1 {
                (es.start_col, es.end_col - es.start_col)
            } else if i == 0 {
                (es.start_col, line_len - es.start_col)
            } else if i == line_count - 1 {
                (0, line_len - es.end_col)
            } else {
                (0, line_len)
            };

            let (start, length) = (start as usize, length as usize);

            println!("{}\x1b[91m{}\x1b[0m", " ".repeat(start), "~".repeat(length));
        }
    }
}

fn execute(matches: &ArgMatches, input: &str) -> Result<(), Vec<Error>> {
    let mut id_map = IdMap::new();
    let lexer = Lexer::new(input, &mut id_map);
    let tokens = lexer.lex()?;
    if matches.is_present("dump-token") {
        dump_token(tokens);
        exit(0);
    }

    let parser = Parser::new(tokens);
    let program = parser.parse()?;
    if matches.is_present("dump-ast") {
        dump_ast(&id_map, program);
        exit(0);
    }

    let stdlib_funcs = stdlib::functions();

    let analyzer = Analyzer::new(&stdlib_funcs, &mut id_map);
    let functions = analyzer.analyze(program)?;

    if matches.is_present("dump-insts") {
        for (name, func) in functions {
            println!("{}:", id_map.name(&name));
            inst::dump_insts(&func.insts, &id_map);
        }
        exit(0);
    }

    let mut vm = VM::new(functions, id_map);
    vm.run();

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
        .arg(Arg::with_name("dump-insts")
             .long("dump-insts")
             .help("Dumps instructions"))
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
