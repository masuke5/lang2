#![feature(box_patterns, drain_filter)]
#![warn(rust_2018_idioms, unused_import_braces)]
#![deny(trivial_casts, trivial_numeric_casts, elided_lifetimes_in_paths)]

#[macro_use]
extern crate pretty_assertions;

#[macro_use]
mod utils;
#[macro_use]
pub mod error;
#[macro_use]
mod ty;
mod ast;
mod bytecode;
mod escape;
mod gc;
mod heapvar;
pub mod id;
mod ir;
mod lexer;
mod module;
mod parser;
mod sema;
pub mod span;
mod stdlib;
mod token;
mod translate;
mod value;
mod vm;

use std::env;
use std::path::{Path, PathBuf};

use ast::*;
use error::ErrorList;
use id::{reserved_id, Id, IdMap};
use lexer::Lexer;
use module::ModuleContainer;
use parser::Parser;
use sema::ModuleBody;
use token::dump_token;
use vm::VM;

#[derive(Debug, PartialEq, Eq)]
pub enum ExecuteMode {
    DumpToken,
    DumpAST,
    DumpInstruction,
    Normal,
}

impl Default for ExecuteMode {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Debug)]
pub struct ExecuteOption {
    input: String,
    file: Id,
    main_module_name: Id,
    file_path: Option<PathBuf>,
    mode: ExecuteMode,
    enable_trace: bool,
    enable_measure: bool,
}

impl ExecuteOption {
    pub fn new(input: String, file: Id, main_module_name: Id) -> Self {
        Self {
            input,
            file,
            main_module_name,
            file_path: None,
            mode: ExecuteMode::Normal,
            enable_trace: false,
            enable_measure: false,
        }
    }

    pub fn enable_trace(mut self, enable: bool) -> Self {
        self.enable_trace = enable;
        self
    }

    pub fn enable_measure(mut self, enable: bool) -> Self {
        self.enable_measure = enable;
        self
    }

    pub fn file_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.file_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn mode(mut self, mode: ExecuteMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn execute(self) {
        if !cfg!(debug_assertions) && (self.enable_measure || self.enable_trace) {
            eprintln!("warning: \"--trace\" and \"--measure\" are enabled only when the interpreter is built on debug mode");
        }

        let mut container = ModuleContainer::new();
        container.add(*reserved_id::STD_MODULE, stdlib::module());

        let pwd = env::current_dir().expect("Unable to get current directory");
        let file_path = self
            .file_path
            .unwrap_or_else(|| PathBuf::from(&pwd).join("cmd"));
        let tmp = pwd.join(&file_path);
        let root_path = tmp.parent().unwrap();

        // Lex
        let lexer = Lexer::new(&self.input, self.file);
        let tokens = lexer.lex();
        if self.mode == ExecuteMode::DumpToken {
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
            heapvar::find(program);
        }

        if self.mode == ExecuteMode::DumpAST {
            for (name, program) in module_buffers {
                println!("--- {}", name);
                dump_ast(&program);
            }

            return;
        }

        // Analyze semantics and translate to a bytecode

        let mut module_bodies = sema::do_semantics_analysis(module_buffers, &container);

        if self.mode == ExecuteMode::DumpInstruction {
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
            .remove(&IdMap::name(self.main_module_name))
            .unwrap();

        // After push the main bytecode, push the other bytecodes
        new_module_bodies.push((IdMap::name(self.main_module_name), main_body));
        for (name, body) in module_bodies {
            new_module_bodies.push((name, body));
        }

        // Execute the bytecode
        let mut vm = VM::new();
        vm.run(new_module_bodies, self.enable_trace, self.enable_measure);
    }
}
