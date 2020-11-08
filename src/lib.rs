#![feature(box_syntax, box_patterns, drain_filter)]
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
mod lexer;
mod load_module;
mod module;
mod parser;
pub mod span;
mod token;
mod value;

use std::env;
use std::path::{Path, PathBuf};

use ast::*;
use id::Id;
use lexer::Lexer;
use parser::Parser;
use token::dump_token;

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

    pub fn generate_bytecodes(&self) -> Option<()> {
        if !cfg!(debug_assertions) && (self.enable_measure || self.enable_trace) {
            eprintln!("warning: \"--trace\" and \"--measure\" are enabled only when the interpreter is built on debug mode");
        }

        let pwd = env::current_dir().expect("Unable to get current directory");
        let file_path = self
            .file_path
            .clone()
            .unwrap_or_else(|| PathBuf::from(&pwd).join("cmd"));
        let tmp = pwd.join(&file_path);
        let root_path = tmp.parent().unwrap();

        // Lex
        let lexer = Lexer::new(&self.input, self.file);
        let tokens = lexer.lex();
        if self.mode == ExecuteMode::DumpToken {
            dump_token(tokens);
            return None;
        }

        // Parse
        let parser = Parser::new(&root_path, tokens, rustc_hash::FxHashSet::default());
        let main_module_path = SymbolPath::from_path(&root_path, &file_path);
        let program = parser.parse(&main_module_path);

        // Load dependent modules
        let mut modules = load_module::load_dependent_modules(root_path, &program);
        modules.insert(main_module_path.clone(), program);

        for program in modules.values_mut() {
            escape::find(program);
            heapvar::find(program);
        }

        if self.mode == ExecuteMode::DumpAST {
            for (path, program) in &modules {
                println!("Module {}", path);
                dump_ast(program);
            }
            return None;
        }

        Some(())
    }

    pub fn execute(self) {
        let vm_module_bodies = match self.generate_bytecodes() {
            Some(vmb) => vmb,
            None => return,
        };
    }
}
