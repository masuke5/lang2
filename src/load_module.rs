use crate::ast::{
    AstFunction as AstFunction_, Block as Block_, Empty, Expr as Expr_, ImportRange,
    ImportRangePath, Program as Program_, Stmt as Stmt_, SymbolPath, Typed,
};
use crate::error::{Error, ErrorList};
use crate::id::IdMap;
use crate::module::{MODULE_EXTENSION, ROOT_MODULE_FILE};
use crate::span::{Span, Spanned};
use rustc_hash::FxHashMap;
use std::fs;
use std::path::{Path, PathBuf};

type Expr = Expr_<Empty>;
type UntypedExpr = Typed<Expr, Empty>;
type Stmt = Stmt_<Empty>;
type Block = Block_<Empty>;
type Program = Program_<Empty>;
type AstFunction = AstFunction_<Empty>;

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

    fn generate_module_filepath(&self, path: &SymbolPath) -> [PathBuf; 2] {
        assert!(path.is_absolute);

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
        assert!(module_path.is_absolute);

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
        assert!(path.is_absolute);

        // Don't load if the module is already loaded
        if self.loaded_modules.contains_key(path) {
            return;
        }

        let module_filepaths = self.generate_module_filepath(&path);
        for module_filepath in &module_filepaths {
            // Generate absolute module path
            let absolute_symbol_path = SymbolPath::from_path(&self.root_path, module_filepath);

            if module_filepath.exists() {
                if let Some(program) =
                    self.parse_module(&absolute_symbol_path, module_filepath, span)
                {
                    self.load_in_program(&program);
                    self.loaded_modules.insert(absolute_symbol_path, program);
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

    fn load(&mut self, base: &SymbolPath, range: &ImportRange, span: &Span) {
        for path in range.to_paths() {
            let mut symbol_path = path.as_path().clone();
            if !symbol_path.is_absolute {
                symbol_path = base.clone().join(&symbol_path);
            }

            let may_not_be_module = match path {
                ImportRangePath::All(..) => false,
                ImportRangePath::Renamed(..) | ImportRangePath::Path(..) => true,
            };
            self.load_module(&symbol_path, span, may_not_be_module);
        }
    }

    // `mpath` is the path of module that includes `expr`
    fn load_in_expr(&mut self, mpath: &SymbolPath, expr: &UntypedExpr) {
        match &expr.kind {
            Expr::Tuple(exprs) => {
                for expr in exprs {
                    self.load_in_expr(mpath, expr);
                }
            }
            Expr::Struct(_, fields) => {
                for (_, expr) in fields {
                    self.load_in_expr(mpath, expr);
                }
            }
            Expr::Array(expr, _) => self.load_in_expr(mpath, expr),
            Expr::Subscript(array_expr, index_expr) => {
                self.load_in_expr(mpath, array_expr);
                self.load_in_expr(mpath, index_expr);
            }
            Expr::Range(start, end) => {
                if let Some(start) = start {
                    self.load_in_expr(mpath, start);
                }
                if let Some(end) = end {
                    self.load_in_expr(mpath, end);
                }
            }
            Expr::BinOp(_, lhs, rhs) => {
                self.load_in_expr(mpath, lhs);
                self.load_in_expr(mpath, rhs);
            }
            Expr::Call(func_expr, arg_expr) => {
                self.load_in_expr(mpath, func_expr);
                self.load_in_expr(mpath, arg_expr);
            }
            Expr::Dereference(expr)
            | Expr::Address(expr, _)
            | Expr::Negative(expr)
            | Expr::App(expr, _)
            | Expr::Not(expr) => self.load_in_expr(mpath, expr),
            Expr::Block(block) => {
                self.load_in_block(mpath, block);
            }
            Expr::If(cond, then, els) => {
                self.load_in_expr(mpath, cond);
                self.load_in_expr(mpath, then);
                if let Some(els) = els {
                    self.load_in_expr(mpath, els);
                }
            }
            _ => {}
        }
    }

    fn load_in_stmt(&mut self, mpath: &SymbolPath, stmt: &Spanned<Stmt>) {
        match &stmt.kind {
            Stmt::Bind(_, _, expr, _, _, _) => self.load_in_expr(mpath, expr),
            Stmt::Expr(expr) | Stmt::Return(Some(expr)) => self.load_in_expr(mpath, expr),
            Stmt::While(cond, body) => {
                self.load_in_expr(mpath, cond);
                self.load_in_stmt(mpath, body);
            }
            Stmt::Assign(lhs, rhs) => {
                self.load_in_expr(mpath, lhs);
                self.load_in_expr(mpath, rhs);
            }
            Stmt::Import(range) => {
                self.load(&mpath.parent().unwrap(), range, &stmt.span);
            }
            _ => {}
        }
    }

    fn load_in_block(&mut self, mpath: &SymbolPath, block: &Block) {
        for stmt in &block.stmts {
            self.load_in_stmt(mpath, stmt);
        }
        for func in &block.functions {
            self.load_in_func(mpath, func);
        }
    }

    fn load_in_func(&mut self, mpath: &SymbolPath, func: &AstFunction) {
        self.load_in_expr(mpath, &func.body);
    }

    fn load_in_program(&mut self, program: &Program) {
        assert!(program.module_path.is_absolute);
        self.load_in_block(&program.module_path, &program.main);
        for imp in &program.impls {
            for func in &imp.functions {
                self.load_in_func(&program.module_path, func);
            }
        }
    }
}

pub fn load_dependent_modules(
    root_path: &Path,
    program: &Program,
) -> FxHashMap<SymbolPath, Program> {
    let mut loader = Loader::new(root_path);
    loader.load_in_program(program);
    loader.loaded_modules
}
