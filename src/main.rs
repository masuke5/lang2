#![feature(box_patterns, slice_concat_trait, drain_filter, seek_convenience)]

mod span;
mod error;
mod token;
mod lexer;
mod ast;
mod parser;
mod ty;
mod sema;
mod id;
mod bytecode;
mod vm;
mod value;
mod stdlib;
mod utils;
mod gc;
mod module;
mod translate;

use std::process::exit;
use std::fs::File;
use std::io::Read;
use std::borrow::Cow;
use std::path::PathBuf;
use std::env;

use lexer::Lexer;
use token::dump_token;
use error::Error;
use parser::Parser;
use ast::*;
use sema::Analyzer;
use id::{Id, IdMap};
use vm::VM;

use clap::{Arg, App, ArgMatches};
use rustc_hash::FxHashMap;

fn print_errors(input: &str, errors: Vec<Error>) {
    let input: Vec<&str> = input.lines().collect();

    for error in errors {
        // Print the error position and message
        let es = error.span;
        println!(
            "\x1b[91merror\x1b[0m: {}:{}:{}-{}:{}: \x1b[97m{}\x1b[0m",
            IdMap::name(es.file),
            es.start_line + 1,
            es.start_col,
            es.end_line + 1,
            es.end_col,
            error.msg);

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

            println!("{}\x1b[91m{}\x1b[0m", " ".repeat(start), "^".repeat(length));
        }
    }
}

fn execute(matches: &ArgMatches, input: &str, file: Id, file_path: Option<PathBuf>) -> Result<(), Vec<Error>> {
    fn ok_if_empty(errors: Vec<Error>) -> Result<(), Vec<Error>> {
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    let enable_trace = matches.is_present("trace");
    let enable_measure = matches.is_present("measure");
    if !cfg!(debug_assertions) && (enable_measure || enable_trace) {
        eprintln!("warning: \"--trace\" and \"--measure\" are enabled only when the interpreter is built on debug mode");
    }

    let (std_module, std_module_header) = stdlib::module();

    let pwd = env::current_dir().expect("Unable to get current directory");
    let file_path = file_path.unwrap_or(PathBuf::from(pwd).join("cmd"));

    // Lex
    let lexer = Lexer::new(input, file);
    let (tokens, mut errors) = lexer.lex();
    if matches.is_present("dump-token") {
        dump_token(tokens);
        return ok_if_empty(errors);
    }

    // Parse
    let parser = Parser::new(&file_path, tokens);
    let program = match parser.parse() {
        Ok(p) => p,
        Err(mut perrors) => {
            errors.append(&mut perrors);
            return Err(errors);
        },
    };

    if matches.is_present("dump-ast") {
        dump_ast(&program);
        return ok_if_empty(errors);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // let stdlib_funcs = stdlib::functions();

    // Analyze semantics and translate to a bytecode

    let analyzer = Analyzer::new();
    let mut module_bytecodes: FxHashMap<String, sema::AnalyzerResult> = FxHashMap::default();
    let (bytecode, _) = analyzer.analyze(program, std_module_header, &mut module_bytecodes)?;

    if matches.is_present("dump-insts") {
        println!("--- MAIN");
        bytecode.dump();

        for (name, (bytecode, _)) in module_bytecodes {
            println!("--- {}", name);
            bytecode.dump();
        }

        return Ok(());
    }

    let module_bytecodes: Vec<(String, bytecode::Bytecode)> = module_bytecodes.into_iter().map(|(n, (b, _))| (n, b)).collect();

    // Execute the bytecode
    let mut vm = VM::new();
    vm.run(bytecode, std_module, module_bytecodes, enable_trace, enable_measure);

    Ok(())
}

fn get_input<'a>(matches: &'a ArgMatches) -> Result<(Id, Cow<'a, str>, Option<PathBuf>), String> {
    if let Some(filepath_str) = matches.value_of("file") {
        let mut file = File::open(filepath_str).map_err(|err| format!("{}", err))?;
        let mut input = String::new();
        file.read_to_string(&mut input).map_err(|err| format!("{}", err))?;

        let filepath = IdMap::new_id(&filepath_str);

        Ok((filepath, input.into(), Some(PathBuf::from(filepath_str))))
    } else if let Some(input) = matches.value_of("cmd") {
        Ok((IdMap::new_id("$cmd"), input.into(), None))
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
            .index(1)
            .required(false))
        .arg(Arg::with_name("cmd")
             .short("c")
             .long("cmd")
             .help("Runs string")
             .takes_value(true)
             .required(false))
        .arg(Arg::with_name("dump-token")
             .long("dump-token")
             .help("Dumps tokens"))
        .arg(Arg::with_name("dump-ast")
             .long("dump-ast")
             .help("Dumps AST"))
        .arg(Arg::with_name("dump-insts")
             .long("dump-insts")
             .help("Dumps instructions"))
        .arg(Arg::with_name("trace")
             .long("trace")
             .help("Traces instructions"))
        .arg(Arg::with_name("measure")
             .long("measure")
             .help("Measures the performance"))
        .get_matches();

    let (file, input, file_path) = match get_input(&matches) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("Unable to load input: {}", err);
            exit(1);
        },
    };

    if let Err(errors) = execute(&matches, &input, file, file_path) {
        print_errors(&input, errors);
        exit(1);
    }
}
