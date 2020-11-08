use crate::ast::{Block, Expr, ImportRange, ImportRangePath, Program, Stmt, SymbolPath};
use crate::error::{Error, ErrorList};
use crate::id::IdMap;
use crate::module::{MODULE_EXTENSION, ROOT_MODULE_FILE};
use crate::span::{Span, Spanned};
use rustc_hash::FxHashMap;
use std::fs;
use std::path::{Path, PathBuf};

struct Loader {
    loaded_modules: FxHashMap<SymbolPath, Program>,
    root_path: PathBuf,
}

impl Loader {
    fn new(root_path: &Path) -> Self {
        Self {
            loaded_modules: FxHashMap::default(),
            root_path: root_path.to_path_buf(),
        }
    }

    fn generate_module_path(&self, path: &SymbolPath) -> [PathBuf; 2] {
        let mut module_path = self.root_path.clone();
        for segment in &path.segments {
            module_path = module_path.join(IdMap::name(segment.id));
        }

        [
            module_path.with_extension(MODULE_EXTENSION),
            module_path.join(ROOT_MODULE_FILE),
        ]
    }

    fn parse_module(
        &mut self,
        module_path: &SymbolPath,
        path: &Path,
        span: &Span,
    ) -> Option<Program> {
        use crate::lexer::Lexer;
        use crate::parser::Parser;

        let code = match fs::read_to_string(path) {
            Ok(code) => code,
            Err(err) => {
                error!(span, "Unable to load file `{}`: {}", path.display(), err);
                return None;
            }
        };

        let file = IdMap::new_id(&format!("{}", path.display()));
        let lexer = Lexer::new(&code, file);
        let tokens = lexer.lex();
        let parser = Parser::new(&self.root_path, tokens, rustc_hash::FxHashSet::default());
        let program = parser.parse(module_path);
        Some(program)
    }

    fn load_module(&mut self, path: &SymbolPath, span: &Span, may_not_be_module: bool) {
        // Don't load if the module is already loaded
        if self.loaded_modules.contains_key(path) {
            return;
        }

        let module_paths = self.generate_module_path(&path);
        for module_path in &module_paths {
            if module_path.exists() {
                if let Some(program) = self.parse_module(path, module_path, span) {
                    self.loaded_modules.insert(path.clone(), program);
                }
                return;
            }
        }

        // `path` may be path to a function or a type in a module
        if may_not_be_module {
            if let Some(parent) = path.parent() {
                self.load_module(&parent, span, false);
                return;
            }
        }

        error!(span, "Module not found `{}`", path);
    }

    fn load(&mut self, range: &ImportRange, span: &Span) {
        for path in range.to_paths() {
            let symbol_path = path.as_path();
            let may_not_be_module = match path {
                ImportRangePath::All(..) => false,
                ImportRangePath::Renamed(..) | ImportRangePath::Path(..) => true,
            };
            self.load_module(symbol_path, span, may_not_be_module);
        }
    }

    fn load_in_expr(&mut self, expr: &Spanned<Expr>) {
        match &expr.kind {
            Expr::Tuple(exprs) => {
                for expr in exprs {
                    self.load_in_expr(expr);
                }
            }
            Expr::Struct(_, fields) => {
                for (_, expr) in fields {
                    self.load_in_expr(expr);
                }
            }
            Expr::Array(expr, _) => self.load_in_expr(expr),
            Expr::Subscript(array_expr, index_expr) => {
                self.load_in_expr(array_expr);
                self.load_in_expr(index_expr);
            }
            Expr::Range(start, end) => {
                if let Some(start) = start {
                    self.load_in_expr(start);
                }
                if let Some(end) = end {
                    self.load_in_expr(end);
                }
            }
            Expr::BinOp(_, lhs, rhs) => {
                self.load_in_expr(lhs);
                self.load_in_expr(rhs);
            }
            Expr::Call(func_expr, arg_expr) => {
                self.load_in_expr(func_expr);
                self.load_in_expr(arg_expr);
            }
            Expr::Dereference(expr)
            | Expr::Address(expr, _)
            | Expr::Negative(expr)
            | Expr::App(expr, _)
            | Expr::Not(expr) => self.load_in_expr(expr),
            Expr::Block(block) => {
                self.load_in_block(block);
            }
            Expr::If(cond, then, els) => {
                self.load_in_expr(cond);
                self.load_in_expr(then);
                if let Some(els) = els {
                    self.load_in_expr(els);
                }
            }
            _ => {}
        }
    }

    fn load_in_stmt(&mut self, stmt: &Spanned<Stmt>) {
        match &stmt.kind {
            Stmt::Bind(_, _, expr, _, _, _) => self.load_in_expr(expr),
            Stmt::Expr(expr) | Stmt::Return(Some(expr)) => self.load_in_expr(expr),
            Stmt::While(cond, body) => {
                self.load_in_expr(cond);
                self.load_in_stmt(body);
            }
            Stmt::Assign(lhs, rhs) => {
                self.load_in_expr(lhs);
                self.load_in_expr(rhs);
            }
            Stmt::Import(range) => {
                self.load(range, &stmt.span);
            }
            _ => {}
        }
    }

    fn load_in_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.load_in_stmt(stmt);
        }
    }
}

pub fn load_dependent_modules(
    root_path: &Path,
    program: &Program,
) -> FxHashMap<SymbolPath, Program> {
    let mut loader = Loader::new(root_path);
    loader.load_in_block(&program.main);
    loader.loaded_modules
}
