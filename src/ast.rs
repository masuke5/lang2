use std::fmt;

use crate::id::{Id, IdMap};
use crate::span::{Span, Spanned};
use crate::utils::{escape_character, escape_string, format_bool, format_iter, span_to_string};

#[derive(Debug, PartialEq, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    LShift,
    RShift,
    BitAnd,
    BitOr,
    BitXor,
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
            BinOp::Mod => "%",
            BinOp::Equal => "==",
            BinOp::NotEqual => "!=",
            BinOp::LessThan => "<",
            BinOp::LessThanOrEqual => "<=",
            BinOp::GreaterThan => ">",
            BinOp::GreaterThanOrEqual => ">=",
            BinOp::LShift => "<<",
            BinOp::RShift => ">>",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Number(i64),
    UnsignedNumber(u64),
    Float(f64),
    String(String),
    Char(char),
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
    pub is_absolute: bool,
    pub segments: Vec<SymbolPathSegment>,
}

impl fmt::Display for SymbolPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_absolute {
            write!(f, "::")?;
        }

        let mut iter = self.segments.iter();
        if let Some(segment) = iter.next() {
            write!(f, "{}", IdMap::name(segment.id))?;
        }

        for segment in iter {
            write!(f, "::{}", IdMap::name(segment.id))?;
        }

        Ok(())
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
            is_absolute: false,
        }
    }

    pub fn new_absolute() -> Self {
        Self {
            segments: Vec::new(),
            is_absolute: true,
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

        let mut spath = Self::new_absolute();
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

    pub fn pop_head(&self) -> Option<SymbolPath> {
        if self.is_empty() {
            return None;
        }

        let mut path = self.clone();
        path.segments.remove(0);
        path.is_absolute = false;
        Some(path)
    }

    pub fn tail(&self) -> Option<&SymbolPathSegment> {
        self.segments.last()
    }

    pub fn head(&self) -> Option<&SymbolPathSegment> {
        self.segments.first()
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

    pub fn join(mut self, other: &SymbolPath) -> SymbolPath {
        if other.is_absolute {
            other.clone()
        } else {
            self.segments.extend(other.segments.clone());
            self
        }
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
    Root(Box<ImportRange>),
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
            Self::Root(inner) => write!(f, "::{}", inner),
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
                ImportRange::Root(inner) => {
                    path_stack.push(SymbolPath::new_absolute());
                    range_stack.push(inner);
                }
            }
        }

        result.reverse();
        result
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Empty;

impl fmt::Display for Empty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Typed<K, T> {
    pub kind: K,
    pub span: Span,
    pub ty: T,
    pub is_mutable: bool, // when is_lvalue is true
    pub is_lvalue: bool,
    pub is_in_heap: bool,
    pub converted_from: Option<T>,
}

impl<K, T> Typed<K, T> {
    pub fn new(kind: K, span: Span, ty: T) -> Self {
        Self {
            kind,
            span,
            ty,
            is_mutable: false,
            is_lvalue: false,
            is_in_heap: false,
            converted_from: None,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ConcreteSymbol {
    Function(SymbolPath, Id), // module path, function name
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr<T> {
    Literal(Literal),
    Tuple(Vec<Typed<Expr<T>, T>>),
    Struct(Spanned<AstType>, Vec<(Spanned<Id>, Typed<Expr<T>, T>)>),
    Array(Box<Typed<Expr<T>, T>>, usize),
    Field(Box<Typed<Expr<T>, T>>, Field),
    Subscript(Box<Typed<Expr<T>, T>>, Box<Typed<Expr<T>, T>>),
    Range(
        Option<Box<Typed<Expr<T>, T>>>,
        Option<Box<Typed<Expr<T>, T>>>,
    ),
    BinOp(BinOp, Box<Typed<Expr<T>, T>>, Box<Typed<Expr<T>, T>>),
    Variable(Id, bool),
    Path(SymbolPath, Option<ConcreteSymbol>),
    Call(Box<Typed<Expr<T>, T>>, Box<Typed<Expr<T>, T>>),
    Dereference(Box<Typed<Expr<T>, T>>),
    Address(Box<Typed<Expr<T>, T>>, bool),
    Negative(Box<Typed<Expr<T>, T>>),
    Not(Box<Typed<Expr<T>, T>>),
    Block(Block<T>),
    If(
        Box<Typed<Expr<T>, T>>,
        Box<Typed<Expr<T>, T>>,
        Option<Box<Typed<Expr<T>, T>>>,
    ),
    App(Box<Typed<Expr<T>, T>>, Vec<Spanned<AstType>>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Stmt<T> {
    // name, type, initial expression, is mutable, is escaped, should store in heap
    Bind(
        Id,
        Option<Spanned<AstType>>,
        Box<Typed<Expr<T>, T>>,
        bool,
        bool,
        bool,
    ),
    Expr(Typed<Expr<T>, T>),
    Return(Option<Typed<Expr<T>, T>>),
    While(Typed<Expr<T>, T>, Box<Spanned<Stmt<T>>>),
    Assign(Typed<Expr<T>, T>, Box<Typed<Expr<T>, T>>),
    Import(ImportRange),
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
pub struct AstFunction<T> {
    pub name: Spanned<Id>,
    pub params: Vec<Param>,
    pub return_ty: Spanned<AstType>,
    pub body: Typed<Expr<T>, T>,
    pub ty_params: Vec<Spanned<Id>>,
    pub has_escaped_variables: bool,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Block<T> {
    pub types: Vec<AstTypeDef>,
    pub functions: Vec<AstFunction<T>>,
    pub function_names: Vec<Id>,      // for gen_bc
    pub function_unique_ids: Vec<Id>, // for gen_bc
    pub stmts: Vec<Spanned<Stmt<T>>>,
    pub result_expr: Box<Typed<Expr<T>, T>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct AstTypeDef {
    pub name: Id,
    pub ty: Spanned<AstType>,
    pub var_ids: Vec<Spanned<Id>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Impl<T> {
    pub target: Spanned<Id>,
    pub functions: Vec<AstFunction<T>>,
}

impl<T> Impl<T> {
    pub fn new(target: Spanned<Id>) -> Self {
        Self {
            target,
            functions: Vec::new(),
        }
    }

    pub fn add_function(&mut self, func: AstFunction<T>) {
        self.functions.push(func);
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program<T> {
    pub module_path: SymbolPath,
    pub main: Block<T>,
    pub imported_modules: Vec<SymbolPath>,
    pub impls: Vec<Impl<T>>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum AstType {
    Int,
    UInt,
    Float,
    String,
    Char,
    Bool,
    Unit,
    Named(Id),
    Pointer(Box<Spanned<AstType>>, bool),
    Array(Box<Spanned<AstType>>, usize),
    Tuple(Vec<Spanned<AstType>>),
    Struct(Vec<(Spanned<Id>, Spanned<AstType>)>),
    App(Spanned<Id>, Vec<Spanned<AstType>>),
    Arrow(Box<Spanned<AstType>>, Box<Spanned<AstType>>),
    Slice(Box<Spanned<AstType>>, bool),
}

impl fmt::Display for AstType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AstType::Int => write!(f, "int"),
            AstType::UInt => write!(f, "uint"),
            AstType::Float => write!(f, "float"),
            AstType::String => write!(f, "string"),
            AstType::Char => write!(f, "char"),
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
            AstType::Slice(inner, is_mutable) => {
                write!(f, "&{}[{}]", format_bool(*is_mutable, "mut "), inner.kind)
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

pub fn dump_func<T: fmt::Display>(func: &AstFunction<T>, depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

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

    dump_expr(&func.body, depth + 1);
}

pub fn dump_block<T: fmt::Display>(block: &Block<T>, depth: usize) {
    for ty in &block.types {
        print!("{}", "  ".repeat(depth));
        println!(
            "type {}<{}> {}",
            IdMap::name(ty.name),
            format_iter(ty.var_ids.iter().map(|id| IdMap::name(id.kind))),
            ty.ty.kind,
        );
    }

    for func in &block.functions {
        dump_func(func, depth);
    }

    if !block.function_unique_ids.is_empty() {
        print!("{}", "  ".repeat(depth));
        println!(
            "functions: {}",
            format_iter(block.function_unique_ids.iter().map(|id| IdMap::name(*id)))
        );
    }

    for stmt in &block.stmts {
        dump_stmt(stmt, depth);
    }

    dump_expr(&block.result_expr, depth);
}

pub fn dump_expr<T: fmt::Display>(expr: &Typed<Expr<T>, T>, depth: usize) {
    fn meta<T: fmt::Display>(expr: &Typed<Expr<T>, T>) -> String {
        let mut result = String::new();

        result += &span_to_string(&expr.span);

        let ty = format!("{}", expr.ty);
        if !ty.is_empty() {
            result += &format!(" \x1b[35m[{}]\x1b[0m", ty);
        }

        result
    }

    // Print indent
    print!("{}", "  ".repeat(depth));

    match &expr.kind {
        Expr::Literal(Literal::Number(n)) => println!("{} {}", n, meta(expr)),
        Expr::Literal(Literal::UnsignedNumber(n)) => println!("{}u {}", n, meta(expr)),
        Expr::Literal(Literal::Float(n)) => println!("{} {}", n, meta(expr)),
        Expr::Literal(Literal::String(s)) => println!("\"{}\" {}", escape_string(s), meta(expr)),
        Expr::Literal(Literal::Char(ch)) => {
            println!("\"{}\" {}", escape_character(*ch), meta(expr))
        }
        Expr::Literal(Literal::Unit) => println!("() {}", meta(expr)),
        Expr::Literal(Literal::True) => println!("true {}", meta(expr)),
        Expr::Literal(Literal::False) => println!("false {}", meta(expr)),
        Expr::Literal(Literal::Null) => println!("__null__ {}", meta(expr)),
        Expr::Tuple(exprs) => {
            println!("tuple {}", meta(expr));
            for expr in exprs {
                dump_expr(&expr, depth + 1);
            }
        }
        Expr::Struct(ty, fields) => {
            println!("struct {} {}", ty.kind, meta(expr));
            for (name, expr) in fields {
                print!("{}", "  ".repeat(depth + 1));
                println!("{}: {}", IdMap::name(name.kind), span_to_string(&name.span));
                dump_expr(expr, depth + 2);
            }
        }
        Expr::Array(init_expr, size) => {
            println!("[{}] {}", size, meta(expr));
            dump_expr(init_expr, depth + 1);
        }
        Expr::Field(inner_expr, field) => {
            match field {
                Field::Number(i) => println!(".{} {}", i, meta(expr)),
                Field::Id(id) => println!(".{} {}", IdMap::name(*id), meta(expr)),
            };

            dump_expr(&inner_expr, depth + 1);
        }
        Expr::Subscript(expr, subscript) => {
            println!("subscript {}", meta(expr));
            dump_expr(&expr, depth + 1);
            dump_expr(&subscript, depth + 1);
        }
        Expr::Range(start, end) => {
            println!("range {}", meta(expr));
            if let Some(start) = start {
                dump_expr(&start, depth + 1);
            }
            if let Some(end) = end {
                dump_expr(&end, depth + 1);
            }
        }
        Expr::Variable(name, is_escaped) => {
            println!(
                "{}{} {}",
                IdMap::name(*name),
                format_bool(*is_escaped, " \x1b[32mescaped\x1b[0m"),
                meta(expr)
            );
        }
        Expr::BinOp(binop, lhs, rhs) => {
            println!("{} {}", binop.to_symbol(), meta(expr));
            dump_expr(&lhs, depth + 1);
            dump_expr(&rhs, depth + 1);
        }
        Expr::Call(func_expr, arg) => {
            println!("call {}", meta(expr));

            dump_expr(func_expr, depth + 1);
            dump_expr(&arg, depth + 1);
        }
        Expr::Path(path, _) => {
            println!("path {} {}", path, meta(expr));
        }
        Expr::Address(expr_, is_mutable) => {
            println!("&{} {}", format_bool(*is_mutable, "mut"), meta(expr));
            dump_expr(&expr_, depth + 1);
        }
        Expr::Dereference(expr_) => {
            println!("* {}", meta(expr));
            dump_expr(&expr_, depth + 1);
        }
        Expr::Negative(expr_) => {
            println!("neg {}", meta(expr));
            dump_expr(&expr_, depth + 1);
        }
        Expr::Not(expr_) => {
            println!("not {}", meta(expr));
            dump_expr(&expr_, depth + 1);
        }
        Expr::Block(block) => {
            print!("\r");
            dump_block(block, depth);
        }
        Expr::If(cond, body, else_expr) => {
            println!("if {}", meta(expr));
            dump_expr(&cond, depth + 1);
            dump_expr(&body, depth + 1);
            if let Some(else_stmt) = else_expr {
                dump_expr(&else_stmt, depth + 1);
            }
        }
        Expr::App(expr, tyargs) => {
            println!("app <{}>", format_iter(tyargs.iter().map(|a| &a.kind)));
            dump_expr(expr, depth + 1);
        }
    }
}

pub fn dump_stmt<T: fmt::Display>(stmt: &Spanned<Stmt<T>>, depth: usize) {
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
            dump_expr(&expr, depth + 1);
        }
        Stmt::Assign(lhs, rhs) => {
            println!(":= {}", span_to_string(&stmt.span));
            dump_expr(&lhs, depth + 1);
            dump_expr(&rhs, depth + 1);
        }
        Stmt::Expr(expr) => {
            print!("\r");
            dump_expr(&expr, depth);
        }
        Stmt::Return(expr) => {
            println!("return {}", span_to_string(&stmt.span));
            if let Some(expr) = expr {
                dump_expr(&expr, depth + 1);
            }
        }
        Stmt::While(cond, body) => {
            println!("while {}", span_to_string(&stmt.span));
            dump_expr(&cond, depth + 1);
            dump_stmt(&body, depth + 1);
        }
        Stmt::Import(range) => {
            println!("import {} {}", range, span_to_string(&stmt.span));
        }
    }
}

pub fn dump_program<T: fmt::Display>(program: &Program<T>, depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    for module_path in &program.imported_modules {
        println!("module {}", module_path);
    }

    for impl_ in &program.impls {
        println!("impl {}", IdMap::name(impl_.target.kind));
        for func in &impl_.functions {
            dump_func(func, 1);
        }
    }

    dump_block(&program.main, 0);
}

pub fn dump_ast<T: fmt::Display>(program: &Program<T>) {
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

        // m1::m2::{m3, m4 as m5, m6::m7, m8::*, ::m9::m10};
        let range = IR::Scope(
            id("m1"),
            Box::new(IR::Scope(
                id("m2"),
                Box::new(IR::Multiple(vec![
                    IR::Symbol(id("m3")),
                    IR::Renamed(id("m4"), id("m5")),
                    IR::Scope(id("m6"), Box::new(IR::Symbol(id("m7")))),
                    IR::Scope(id("m8"), Box::new(IR::All)),
                    IR::Root(Box::new(IR::Scope(
                        id("m9"),
                        Box::new(IR::Symbol(id("m10"))),
                    ))),
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
            // ::m9::m10
            ImportRangePath::Path(
                SymbolPath::new_absolute()
                    .append_str("m9")
                    .append_str("m10"),
            ),
        ];

        assert_eq!(expected, paths);
    }
}
