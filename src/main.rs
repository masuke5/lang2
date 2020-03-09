#![feature(box_patterns, drain_filter)]
#![warn(rust_2018_idioms, unused_import_braces)]
#![deny(trivial_casts, trivial_numeric_casts, elided_lifetimes_in_paths)]

#[macro_use]
extern crate pretty_assertions;

#[macro_use]
mod utils;
#[macro_use]
mod error;
#[macro_use]
mod ty;
mod ast;
mod bytecode;
mod escape;
mod gc;
mod id;
mod lexer;
mod module;
mod parser;
mod sema;
mod span;
mod stdlib;
mod token;
mod translate;
mod value;
mod vm;

use std::borrow::Cow;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::process::exit;

use ast::*;
use error::ErrorList;
use id::{reserved_id, Id, IdMap};
use lexer::Lexer;
use module::ModuleContainer;
use parser::Parser;
use sema::ModuleBody;
use token::dump_token;
use vm::VM;

use clap::{App, Arg, ArgMatches};

fn execute(
    matches: &ArgMatches<'_>,
    input: &str,
    file: Id,
    main_module_name: Id,
    file_path: Option<PathBuf>,
) {
    let enable_trace = matches.is_present("trace");
    let enable_measure = matches.is_present("measure");
    if !cfg!(debug_assertions) && (enable_measure || enable_trace) {
        eprintln!("warning: \"--trace\" and \"--measure\" are enabled only when the interpreter is built on debug mode");
    }

    let mut container = ModuleContainer::new();
    container.add(*reserved_id::STD_MODULE, stdlib::module());

    let pwd = env::current_dir().expect("Unable to get current directory");
    let file_path = file_path.unwrap_or_else(|| PathBuf::from(&pwd).join("cmd"));
    let tmp = pwd.join(&file_path);
    let root_path = tmp.parent().unwrap();

    // Lex
    let lexer = Lexer::new(input, file);
    let tokens = lexer.lex();
    if matches.is_present("dump-token") {
        dump_token(tokens);
        return;
    }

    // Parse
    let parser = Parser::new(
        &root_path,
        tokens,
        rustc_hash::FxHashSet::default(),
        &container,
    );
    let main_module_path = SymbolPath::from_path(&root_path, &file_path);
    let mut module_buffers = parser.parse(&main_module_path);

    for program in module_buffers.values_mut() {
        escape::find(program);
    }

    if matches.is_present("dump-ast") {
        for (name, program) in module_buffers {
            println!("--- {}", name);
            dump_ast(&program);
        }

        return;
    }

    // Analyze semantics and translate to a bytecode

    let mut module_bodies = sema::do_semantics_analysis(module_buffers, &container);

    if matches.is_present("dump-insts") {
        for (name, body) in module_bodies {
            if let ModuleBody::Normal(bytecode) = body {
                println!("--- {}", name);
                bytecode.dump();
            }
        }

        return;
    }

    if ErrorList::has_error() {
        return;
    }

    let mut new_module_bodies = Vec::with_capacity(module_bodies.len());
    let main_body = module_bodies
        .remove(&IdMap::name(main_module_name))
        .unwrap();

    // After push the main bytecode, push the other bytecodes
    new_module_bodies.push((IdMap::name(main_module_name), main_body));
    for (name, body) in module_bodies {
        new_module_bodies.push((name, body));
    }

    // Execute the bytecode
    let mut vm = VM::new();
    vm.run(new_module_bodies, enable_trace, enable_measure);
}

fn get_input<'a>(
    matches: &'a ArgMatches<'a>,
) -> Result<(Id, Cow<'a, str>, Id, Option<PathBuf>), String> {
    if let Some(filepath_str) = matches.value_of("file") {
        let mut file = File::open(filepath_str).map_err(|err| format!("{}", err))?;
        let mut input = String::new();
        file.read_to_string(&mut input)
            .map_err(|err| format!("{}", err))?;

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
        .arg(
            Arg::with_name("file")
                .help("Runs file")
                .index(1)
                .required(false),
        )
        .arg(
            Arg::with_name("cmd")
                .short("c")
                .long("cmd")
                .help("Runs string")
                .takes_value(true)
                .required(false),
        )
        .arg(
            Arg::with_name("dump-token")
                .long("dump-token")
                .help("Dumps tokens"),
        )
        .arg(
            Arg::with_name("dump-ast")
                .long("dump-ast")
                .help("Dumps AST"),
        )
        .arg(
            Arg::with_name("dump-insts")
                .long("dump-insts")
                .help("Dumps instructions"),
        )
        .arg(
            Arg::with_name("trace")
                .long("trace")
                .help("Traces instructions"),
        )
        .arg(
            Arg::with_name("measure")
                .long("measure")
                .help("Measures the performance"),
        )
        .get_matches();

    let (file, input, module_name, file_path) = match get_input(&matches) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("Unable to load input: {}", err);
            exit(1);
        }
    };

    execute(&matches, &input, file, module_name, file_path);
    if ErrorList::has_error() {
        exit(1);
    }
}
