mod span;
mod error;
mod token;
mod lexer;

use std::env;
use std::process::exit;
use lexer::Lexer;

fn print_usage() {
    println!("usage: lang2 <input>");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        exit(1);
    }

    let lexer = Lexer::new(&args[1]);
    match lexer.lex() {
        Ok(tokens) => {
            for token in tokens {
                println!("{} {}:{}-{}:{}", token.kind, token.span.start_line, token.span.start_col, token.span.end_line, token.span.end_col);
            }
        },
        Err(errors) => {
            for error in errors {
                println!("{:?}", error);
            }
        },
    };
}
