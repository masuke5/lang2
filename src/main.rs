#![feature(box_patterns, slice_concat_trait, drain_filter, seek_convenience)]

#[macro_use]
mod utils;
#[macro_use]
mod ty;

mod span;
mod error;
mod token;
mod lexer;
mod ast;
mod parser;
mod sema;
mod id;
mod bytecode;
mod vm;
mod value;
mod stdlib;
mod gc;
mod module;
mod translate;
mod escape;

use std::process::exit;
use std::fs;
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
use id::{Id, IdMap, reserved_id};
use vm::VM;
use module::ModuleContainer;
use sema::ModuleBody;

use clap::{Arg, App, ArgMatches};
use rustc_hash::FxHashMap;

fn print_errors(errors: Vec<Error>) {
    let mut file_cache: FxHashMap<Id, Vec<String>> = FxHashMap::default();

    for error in errors {
        let es = error.span;

        let input = match file_cache.get(&es.file) {
            Some(input) => input,
            None => {
                let contents = fs::read_to_string(&IdMap::name(es.file)).unwrap();
                file_cache.insert(es.file, contents.split("\n").map(|c| c.to_string()).collect());
                &file_cache[&es.file]
            },
        };

        // Print the error position and message
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
            println!("{}", if line >= input.len() as usize { "" } else { &input[line] });

            let indent = input[line]
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .fold(0, |indent, c| indent + match c {
                    ' ' => 1,
                    '\t' => 4,
                    _ => unreachable!(),
                }) as u32;

            // Print the error span
            let (start, length) = if line_count == 1 {
                (es.start_col, es.end_col - es.start_col)
            } else if i == 0 {
                (es.start_col, line_len - es.start_col)
            } else if i == line_count - 1 {
                (indent, (line_len - es.end_col).checked_sub(1).unwrap_or(0))
            } else {
                (indent, line_len - indent)
            };

            let (start, length) = (start as usize, length as usize);

            println!("{}\x1b[91m{}\x1b[0m", " ".repeat(start), "^".repeat(length));
        }
    }
}

fn execute(matches: &ArgMatches, input: &str, file: Id, main_module_name: Id, file_path: Option<PathBuf>) -> Result<(), Vec<Error>> {
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

    let mut container = ModuleContainer::new();
    container.add(*reserved_id::STD_MODULE, stdlib::module());

    let pwd = env::current_dir().expect("Unable to get current directory");
    let file_path = file_path.unwrap_or(PathBuf::from(&pwd).join("cmd"));
    let tmp = pwd.join(&file_path);
    let root_path = tmp.parent().unwrap();

    // Lex
    let lexer = Lexer::new(input, file);
    let (tokens, mut errors) = lexer.lex();
    if matches.is_present("dump-token") {
        dump_token(tokens);
        return ok_if_empty(errors);
    }

    // Parse
    let parser = Parser::new(&root_path, tokens, rustc_hash::FxHashSet::default(), &container);
    let main_module_path = SymbolPath::from_path(&root_path, &file_path);
    let mut module_buffers = match parser.parse(&main_module_path) {
        Ok(p) => p,
        Err(mut perrors) => {
            errors.append(&mut perrors);
            return Err(errors);
        },
    };

    for (_, program) in &mut module_buffers {
        escape::find(program);
    }

    if matches.is_present("dump-ast") {
        for (name, program) in module_buffers {
            println!("--- {}", name);
            dump_ast(&program);
        }

        return ok_if_empty(errors);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Analyze semantics and translate to a bytecode

    let mut module_bodies = sema::analyze_semantics(module_buffers, &container)?;

    if matches.is_present("dump-insts") {
        for (name, body) in module_bodies {
            if let ModuleBody::Normal(bytecode) = body {
                println!("--- {}", name);
                bytecode.dump();
            }
        }

        return Ok(());
    }

    let mut new_module_bodies = Vec::with_capacity(module_bodies.len());
    let main_body = module_bodies.remove(&IdMap::name(main_module_name)).unwrap();

    // After push the main bytecode, push the other bytecodes
    new_module_bodies.push((IdMap::name(main_module_name), main_body));
    for (name, body) in module_bodies {
        new_module_bodies.push((name, body));
    }

    // Execute the bytecode
    let mut vm = VM::new();
    vm.run(new_module_bodies, enable_trace, enable_measure);

    Ok(())
}

fn get_input<'a>(matches: &'a ArgMatches) -> Result<(Id, Cow<'a, str>, Id, Option<PathBuf>), String> {
    if let Some(filepath_str) = matches.value_of("file") {
        let mut file = File::open(filepath_str).map_err(|err| format!("{}", err))?;
        let mut input = String::new();
        file.read_to_string(&mut input).map_err(|err| format!("{}", err))?;

        let filepath_id = IdMap::new_id(&filepath_str);
        let filepath = PathBuf::from(filepath_str);
        let module_name = filepath.file_stem().unwrap();
        let module_name = IdMap::new_id(&format!("::{}", &module_name.to_string_lossy()));

        Ok((filepath_id, input.into(), module_name, Some(filepath)))
    } else if let Some(input) = matches.value_of("cmd") {
        let filepath_id = *reserved_id::CMD;
        let module_name = filepath_id;
        Ok((filepath_id, input.into(), module_name, None))
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

    let (file, input, module_name, file_path) = match get_input(&matches) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("Unable to load input: {}", err);
            exit(1);
        },
    };

    if let Err(errors) = execute(&matches, &input, file, module_name, file_path) {
        print_errors(errors);
        exit(1);
    }
}
