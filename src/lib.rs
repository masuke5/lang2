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
mod codegen;
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
use codegen::codegen;
use error::ErrorList;
use id::{reserved_id, Id, IdMap};
use lexer::Lexer;
use module::ModuleContainer;
use parser::Parser;
use sema::ModuleBody;
use token::dump_token;
use vm::{ModuleBody as VMModuleBody, VM};

#[derive(Debug, PartialEq, Eq)]
pub enum ExecuteMode {
    DumpToken,
    DumpAST,
    DumpIR,
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

        // Do semantics analysis and translate to IR

        let module_bodies = sema::do_semantics_analysis(module_buffers, &container);
        if self.mode == ExecuteMode::DumpIR {
            for (name, body) in module_bodies {
                if let ModuleBody::Normal(module) = body {
                    println!("--- {}", name);
                    for module in module.imported_modules {
                        println!("MODULE {}", module);
                    }

                    for (id, (name, func)) in module.functions.into_iter().enumerate() {
                        println!("FUNC {} ({}):", IdMap::name(name), id);
                        ir::dump_expr(&func.body);
                    }
                }
            }

            return;
        }

        if ErrorList::has_error() {
            return;
        }

        // Generate bytecode

        let mut vm_module_bodies = vec![(
            String::new(),
            VMModuleBody::Normal(bytecode::Bytecode::new()),
        )];

        for (name, body) in module_bodies {
            let body = match body {
                ModuleBody::Normal(module) => {
                    let bytecode = codegen(module);
                    VMModuleBody::Normal(bytecode)
                }
                ModuleBody::Native(module) => VMModuleBody::Native(module),
            };

            if name == format!("{}", main_module_path) {
                vm_module_bodies[0] = (name, body);
            } else {
                vm_module_bodies.push((name, body));
            }
        }

        if self.mode == ExecuteMode::DumpInstruction {
            for (id, (name, body)) in vm_module_bodies.into_iter().enumerate() {
                if let VMModuleBody::Normal(bytecode) = body {
                    println!("--- {} ({})", name, id);
                    bytecode.dump();
                }
            }

            return;
        }

        // Execute the bytecode

        let mut vm = VM::new();
        vm.run(vm_module_bodies, self.enable_trace, self.enable_measure);
    }
}
