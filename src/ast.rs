use std::fmt;

use crate::id::{Id, IdMap};
use crate::span::Spanned;
use crate::utils::{escape_string, format_bool, format_iter, span_to_string};

#[derive(Debug, PartialEq, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    And,
    Or,
}

impl BinOp {
    pub fn to_symbol(&self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Equal => "==",
            BinOp::NotEqual => "!=",
            BinOp::LessThan => "<",
            BinOp::LessThanOrEqual => "<=",
            BinOp::GreaterThan => ">",
            BinOp::GreaterThanOrEqual => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Number(i64),
    String(usize),
    Unit,
    True,
    False,
    Null,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Field {
    Number(usize),
    Id(Id),
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SymbolPath {
    pub segments: Vec<SymbolPathSegment>,
}

impl fmt::Display for SymbolPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.segments.iter().fold(String::new(), |acc, seg| format!(
                "{}::{}",
                acc,
                IdMap::name(seg.id)
            ))
        )
    }
}

impl fmt::Debug for SymbolPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl SymbolPath {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn from_path(root: &std::path::Path, path: &std::path::Path) -> Self {
        // Convert paths for comparison
        let root = match root.canonicalize() {
            Ok(root) => root,
            Err(_) => return Self::new(),
        };
        let root = root.as_path();

        let path = match path.canonicalize() {
            Ok(path) => path,
            Err(_) => return Self::new(),
        };
        let mut path = path.as_path();

        let mut spath = Self {
            segments: Vec::new(),
        };
        if root == path {
            return spath;
        }

        let mut paths = vec![path.file_stem().unwrap().to_string_lossy()];
        path = path.parent().unwrap();

        while root != path {
            paths.push(path.file_name().unwrap().to_string_lossy());
            path = path.parent().unwrap();
        }

        spath.segments = paths
            .into_iter()
            .map(|s| SymbolPathSegment::new(IdMap::new_id(&s)))
            .rev()
            .collect();

        spath
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn parent(&self) -> Option<SymbolPath> {
        let mut path = self.clone();
        path.segments.pop()?;
        Some(path)
    }

    pub fn tail(&self) -> Option<&SymbolPathSegment> {
        self.segments.last()
    }

    #[allow(dead_code)]
    pub fn append_str(self, s: &str) -> SymbolPath {
        self.append_id(IdMap::new_id(s))
    }

    pub fn append_id(self, id: Id) -> SymbolPath {
        self.append(SymbolPathSegment::new(id))
    }

    pub fn append(mut self, segment: SymbolPathSegment) -> SymbolPath {
        self.segments.push(segment);
        self
    }
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub struct SymbolPathSegment {
    pub id: Id,
}

impl SymbolPathSegment {
    pub fn new(id: Id) -> Self {
        Self { id }
    }
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub enum ImportRange {
    Symbol(Id),
    Renamed(Id, Id),
    All,
    Multiple(Vec<ImportRange>),
    Scope(Id, Box<ImportRange>),
}

impl fmt::Display for ImportRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Symbol(id) => write!(f, "{}", IdMap::name(*id)),
            Self::Renamed(id, renamed) => {
                write!(f, "{} as {}", IdMap::name(*id), IdMap::name(*renamed))
            }
            Self::All => write!(f, "*"),
            Self::Multiple(symbols) => {
                write!(f, "{{")?;
                write_iter!(f, symbols.iter())?;
                write!(f, "}}")
            }
            Self::Scope(module, rest) => write!(f, "{}::{}", IdMap::name(*module), rest),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImportRangePath {
    All(SymbolPath),
    Renamed(SymbolPath, Id),
    Path(SymbolPath),
}

impl ImportRangePath {
    pub fn as_path(&self) -> &SymbolPath {
        match self {
            Self::All(path) | Self::Renamed(path, _) | Self::Path(path) => path,
        }
    }
}

impl ImportRange {
    pub fn to_paths(&self) -> Vec<ImportRangePath> {
        let mut result = Vec::new();
        let mut path_stack = vec![SymbolPath::new()];
        let mut range_stack: Vec<&ImportRange> = vec![self];

        while let Some(range) = range_stack.pop() {
            let path = path_stack.pop().unwrap();

            match range {
                ImportRange::Symbol(id) => {
                    let path = path.clone().append_id(*id);
                    result.push(ImportRangePath::Path(path));
                }
                ImportRange::Renamed(id, renamed) => {
                    let path = path.clone().append_id(*id);
                    result.push(ImportRangePath::Renamed(path, *renamed));
                }
                ImportRange::All => {
                    result.push(ImportRangePath::All(path.clone()));
                }
                ImportRange::Multiple(ranges) => {
                    path_stack.push(path.clone());

                    for range in ranges {
                        path_stack.push(path.clone());
                        range_stack.push(range);
                    }
                }
                ImportRange::Scope(id, rest) => {
                    path_stack.push(path.append_id(*id));
                    range_stack.push(rest);
                }
            }
        }

        result.reverse();
        result
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Literal(Literal),
    Tuple(Vec<Spanned<Expr>>),
    Struct(Spanned<AstType>, Vec<(Spanned<Id>, Spanned<Expr>)>),
    Array(Box<Spanned<Expr>>, usize),
    Field(Box<Spanned<Expr>>, Field),
    Subscript(Box<Spanned<Expr>>, Box<Spanned<Expr>>),
    BinOp(BinOp, Box<Spanned<Expr>>, Box<Spanned<Expr>>),
    Variable(Id, bool),
    Path(SymbolPath),
    Call(Box<Spanned<Expr>>, Box<Spanned<Expr>>),
    Dereference(Box<Spanned<Expr>>),
    Address(Box<Spanned<Expr>>, bool),
    Negative(Box<Spanned<Expr>>),
    Block(Block),
    If(
        Box<Spanned<Expr>>,
        Box<Spanned<Expr>>,
        Option<Box<Spanned<Expr>>>,
    ),
    App(Box<Spanned<Expr>>, Vec<Spanned<AstType>>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Stmt {
    // name, type, initial expression, is mutable, is escaped, should store in heap
    Bind(
        Id,
        Option<Spanned<AstType>>,
        Box<Spanned<Expr>>,
        bool,
        bool,
        bool,
    ),
    Expr(Spanned<Expr>),
    Return(Option<Spanned<Expr>>),
    While(Spanned<Expr>, Box<Spanned<Stmt>>),
    Assign(Spanned<Expr>, Box<Spanned<Expr>>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Param {
    pub name: Id,
    pub ty: Spanned<AstType>,
    pub is_mutable: bool,
    pub is_escaped: bool,
    pub is_in_heap: bool,
}

#[derive(Debug, PartialEq, Clone)]
pub struct AstFunction {
    pub name: Spanned<Id>,
    pub params: Vec<Param>,
    pub return_ty: Spanned<AstType>,
    pub body: Spanned<Expr>,
    pub ty_params: Vec<Spanned<Id>>,
    pub has_escaped_variables: bool,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Block {
    pub types: Vec<AstTypeDef>,
    pub functions: Vec<AstFunction>,
    pub stmts: Vec<Spanned<Stmt>>,
    pub imports: Vec<Spanned<ImportRange>>,
    pub result_expr: Box<Spanned<Expr>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct AstTypeDef {
    pub name: Id,
    pub ty: Spanned<AstType>,
    pub var_ids: Vec<Spanned<Id>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub main: Block,
    pub strings: Vec<String>,
    pub imported_modules: Vec<SymbolPath>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum AstType {
    Int,
    String,
    Bool,
    Unit,
    Named(Id),
    Pointer(Box<Spanned<AstType>>, bool),
    Array(Box<Spanned<AstType>>, usize),
    Tuple(Vec<Spanned<AstType>>),
    Struct(Vec<(Spanned<Id>, Spanned<AstType>)>),
    App(Spanned<Id>, Vec<Spanned<AstType>>),
    Arrow(Box<Spanned<AstType>>, Box<Spanned<AstType>>),
}

impl fmt::Display for AstType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AstType::Int => write!(f, "int"),
            AstType::String => write!(f, "string"),
            AstType::Bool => write!(f, "bool"),
            AstType::Unit => write!(f, "unit"),
            AstType::Named(id) => write!(f, "{}", IdMap::name(*id)),
            AstType::Pointer(ty, is_mutable) => {
                write!(f, "*{}{}", format_bool(*is_mutable, "mut "), ty.kind)
            }
            AstType::Array(ty, size) => write!(f, "[{}; {}]", ty.kind, size),
            AstType::Tuple(types) => {
                write!(f, "(")?;
                write_iter!(f, types.iter().map(|t| &t.kind))?;
                write!(f, ")")
            }
            AstType::Struct(fields) => {
                write!(f, "struct {{ ")?;
                write_iter!(
                    f,
                    fields
                        .iter()
                        .map(|(id, ty)| format!("{}: {}", IdMap::name(id.kind), ty.kind))
                )?;
                write!(f, " }}")
            }
            AstType::App(name, types) => {
                write!(f, "{}<", IdMap::name(name.kind))?;
                write_iter!(f, types.iter().map(|t| &t.kind))?;
                write!(f, ">")
            }
            AstType::Arrow(arg, ret) => write!(f, "{} -> {}", arg.kind, ret.kind),
        }
    }
}

pub fn dump_block(block: &Block, strings: &[String], depth: usize) {
    for range in &block.imports {
        println!("import {} {}", range.kind, span_to_string(&range.span));
    }

    for ty in &block.types {
        println!(
            "type {}<{}> {}",
            IdMap::name(ty.name),
            format_iter(ty.var_ids.iter().map(|id| IdMap::name(id.kind))),
            ty.ty.kind,
        );
    }

    for func in &block.functions {
        print!("fn {}", IdMap::name(func.name.kind));

        if func.has_escaped_variables {
            print!(" \x1b[32mhas escaped vars\x1b[0m");
        }

        if !func.ty_params.is_empty() {
            print!("<");
            print!(
                "{}",
                format_iter(func.ty_params.iter().map(|id| IdMap::name(id.kind)))
            );
            print!(">");
        }

        println!(
            "({}): {}",
            format_iter(func.params.iter().map(|p| format!(
                "{}{}{}: {}",
                format_bool(p.is_mutable, "mut "),
                IdMap::name(p.name),
                format_bool(p.is_escaped, " \x1b[32mescaped\x1b[0m"),
                p.ty.kind
            ))),
            func.return_ty.kind,
        );

        dump_expr(&func.body, strings, depth + 1);
    }

    for stmt in &block.stmts {
        dump_stmt(stmt, strings, depth);
    }

    dump_expr(&block.result_expr, strings, depth);
}

pub fn dump_expr(expr: &Spanned<Expr>, strings: &[String], depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    match &expr.kind {
        Expr::Literal(Literal::Number(n)) => println!("{} {}", n, span_to_string(&expr.span)),
        Expr::Literal(Literal::String(i)) => println!(
            "\"{}\" {}",
            escape_string(&strings[*i]),
            span_to_string(&expr.span)
        ),
        Expr::Literal(Literal::Unit) => println!("() {}", span_to_string(&expr.span)),
        Expr::Literal(Literal::True) => println!("true {}", span_to_string(&expr.span)),
        Expr::Literal(Literal::False) => println!("false {}", span_to_string(&expr.span)),
        Expr::Literal(Literal::Null) => println!("__null__ {}", span_to_string(&expr.span)),
        Expr::Tuple(exprs) => {
            println!("tuple {}", span_to_string(&expr.span));
            for expr in exprs {
                dump_expr(&expr, strings, depth + 1);
            }
        }
        Expr::Struct(ty, fields) => {
            println!("struct {} {}", ty.kind, span_to_string(&expr.span));
            for (name, expr) in fields {
                print!("{}", "  ".repeat(depth + 1));
                println!("{}: {}", IdMap::name(name.kind), span_to_string(&name.span));
                dump_expr(expr, strings, depth + 2);
            }
        }
        Expr::Array(init_expr, size) => {
            println!("[{}] {}", size, span_to_string(&expr.span));
            dump_expr(init_expr, strings, depth + 1);
        }
        Expr::Field(expr, field) => {
            match field {
                Field::Number(i) => println!(".{} {}", i, span_to_string(&expr.span)),
                Field::Id(id) => println!(".{} {}", IdMap::name(*id), span_to_string(&expr.span)),
            };

            dump_expr(&expr, strings, depth + 1);
        }
        Expr::Subscript(expr, subscript) => {
            println!("subscript {}", span_to_string(&expr.span));
            dump_expr(&expr, strings, depth + 1);
            dump_expr(&subscript, strings, depth + 1);
        }
        Expr::Variable(name, is_escaped) => {
            println!(
                "{}{} {}",
                IdMap::name(*name),
                format_bool(*is_escaped, " \x1b[32mescaped\x1b[0m"),
                span_to_string(&expr.span)
            );
        }
        Expr::BinOp(binop, lhs, rhs) => {
            println!("{} {}", binop.to_symbol(), span_to_string(&expr.span));
            dump_expr(&lhs, strings, depth + 1);
            dump_expr(&rhs, strings, depth + 1);
        }
        Expr::Call(func_expr, arg) => {
            println!("call {}", span_to_string(&expr.span));

            dump_expr(func_expr, strings, depth + 1);
            dump_expr(&arg, strings, depth + 1);
        }
        Expr::Path(path) => {
            println!("path {} {}", path, span_to_string(&expr.span));
        }
        Expr::Address(expr_, is_mutable) => {
            println!(
                "&{} {}",
                format_bool(*is_mutable, "mut"),
                span_to_string(&expr.span)
            );
            dump_expr(&expr_, strings, depth + 1);
        }
        Expr::Dereference(expr_) => {
            println!("* {}", span_to_string(&expr.span));
            dump_expr(&expr_, strings, depth + 1);
        }
        Expr::Negative(expr_) => {
            println!("neg {}", span_to_string(&expr.span));
            dump_expr(&expr_, strings, depth + 1);
        }
        Expr::Block(block) => {
            dump_block(block, strings, depth);
        }
        Expr::If(cond, body, else_expr) => {
            println!("if {}", span_to_string(&expr.span));
            dump_expr(&cond, strings, depth + 1);
            dump_expr(&body, strings, depth + 1);
            if let Some(else_stmt) = else_expr {
                dump_expr(&else_stmt, strings, depth + 1);
            }
        }
        Expr::App(expr, tyargs) => {
            println!("app <{}>", format_iter(tyargs.iter().map(|a| &a.kind)));
            dump_expr(expr, strings, depth + 1);
        }
    }
}

pub fn dump_stmt(stmt: &Spanned<Stmt>, strings: &[String], depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    match &stmt.kind {
        Stmt::Bind(name, ty, expr, is_mutable, is_escaped, is_heapvar) => {
            print!(
                "let{}{}{} ",
                format_bool(*is_mutable, " mut"),
                format_bool(*is_escaped, " \x1b[32mescaped\x1b[0m"),
                format_bool(*is_heapvar, " \x1b[32min heap\x1b[0m")
            );
            print!("{}", IdMap::name(*name));
            if let Some(ty) = ty {
                print!(": {}", ty.kind);
            }

            println!(" =");
            dump_expr(&expr, strings, depth + 1);
        }
        Stmt::Assign(lhs, rhs) => {
            println!(":= {}", span_to_string(&stmt.span));
            dump_expr(&lhs, strings, depth + 1);
            dump_expr(&rhs, strings, depth + 1);
        }
        Stmt::Expr(expr) => {
            print!("\r");
            dump_expr(&expr, strings, depth);
        }
        Stmt::Return(expr) => {
            println!("return {}", span_to_string(&stmt.span));
            if let Some(expr) = expr {
                dump_expr(&expr, strings, depth + 1);
            }
        }
        Stmt::While(cond, body) => {
            println!("while {}", span_to_string(&stmt.span));
            dump_expr(&cond, strings, depth + 1);
            dump_stmt(&body, strings, depth + 1);
        }
    }
}

pub fn dump_program(program: &Program, depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    for module_path in &program.imported_modules {
        println!("module {}", module_path);
    }

    dump_block(&program.main, &program.strings, 0);
}

pub fn dump_ast(program: &Program) {
    dump_program(program, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;

    #[test]
    fn from_path() {
        let root = env::current_dir().unwrap();
        let path = PathBuf::from("test2/test.lang2");
        let actual = SymbolPath::from_path(&root, &path);

        let expected = SymbolPath::new().append_str("test2").append_str("test");
        assert_eq!(expected, actual);
    }

    #[test]
    fn to_paths() {
        type IR = ImportRange;
        let id = IdMap::new_id;

        // m1::m2::{m3, m4 as m5, m6::m7, m8::*};
        let range = IR::Scope(
            id("m1"),
            Box::new(IR::Scope(
                id("m2"),
                Box::new(IR::Multiple(vec![
                    IR::Symbol(id("m3")),
                    IR::Renamed(id("m4"), id("m5")),
                    IR::Scope(id("m6"), Box::new(IR::Symbol(id("m7")))),
                    IR::Scope(id("m8"), Box::new(IR::All)),
                ])),
            )),
        );

        let paths = range.to_paths();

        let base_path = SymbolPath::new().append_str("m1").append_str("m2");
        let expected = vec![
            // m1::m2::m3
            ImportRangePath::Path(base_path.clone().append_str("m3")),
            // m1::m2::m4
            ImportRangePath::Renamed(base_path.clone().append_str("m4"), id("m5")),
            // m1::m2::m6::m7
            ImportRangePath::Path(base_path.clone().append_str("m6").append_str("m7")),
            // m1::m2::m8::*
            ImportRangePath::All(base_path.clone().append_str("m8")),
        ];

        assert_eq!(expected, paths);
    }
}
