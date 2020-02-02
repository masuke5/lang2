use std::fmt;

use crate::span::Spanned;
use crate::id::{Id, IdMap};
use crate::utils::{escape_string, span_to_string, format_iter, format_bool};

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolPath {
    pub segments: Vec<SymbolPathSegment>,
}

impl fmt::Display for SymbolPath {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.segments
            .iter()
            .fold(String::new(), |acc, seg| format!("{}::{}", acc, IdMap::name(seg.id)))
        )
    }
}

impl SymbolPath {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn from_path(root: &std::path::Path, path: &std::path::Path) -> Self {
        // TODO: better conversion
        // TODO: Avoid unwrap()
        // Convert paths for comparison
        let root = root.canonicalize().unwrap();
        let root = root.as_path();
        let path = path.canonicalize().unwrap();
        let mut path = path.as_path();

        let mut spath = Self { segments: Vec::new() };
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
        self.segments.get(self.segments.len() - 1)
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
        Self {
            id,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ImportRange {
     pub module_path: SymbolPath,
     pub symbols: ImportSymbolList,
}

impl fmt::Display for ImportRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}::{}", self.module_path, self.symbols)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ImportSymbolList {
    Single(ImportSymbol),
    Multiple(Vec<ImportSymbol>),
    All,
}

impl fmt::Display for ImportSymbolList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Single(symbol) => write!(f, "{}", symbol),
            Self::Multiple(symbols) => {
                write!(f, "{{")?;
                write_iter!(f, symbols.iter())?;
                write!(f, "}}")
            },
            Self::All => write!(f, "*"),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ImportSymbol {
    Id(Id),
    As(Id, Id),
}

impl fmt::Display for ImportSymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Id(id) => write!(f, "{}", IdMap::name(*id)),
            Self::As(original, renamed) => write!(f, "{} as {}", IdMap::name(*original), IdMap::name(*renamed)),
        }
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
    Variable(Id),
    Call(Spanned<SymbolPath>, Vec<Spanned<Expr>>, Vec<Spanned<AstType>>),
    Dereference(Box<Spanned<Expr>>),
    Address(Box<Spanned<Expr>>, bool),
    Negative(Box<Spanned<Expr>>),
    Alloc(Box<Spanned<Expr>>, bool),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Stmt {
    Bind(Id, Spanned<Expr>, bool),
    Expr(Spanned<Expr>),
    Block(Vec<Spanned<Stmt>>),
    Return(Option<Spanned<Expr>>),
    If(Spanned<Expr>, Box<Spanned<Stmt>>, Option<Box<Spanned<Stmt>>>),
    While(Spanned<Expr>, Box<Spanned<Stmt>>),
    Assign(Spanned<Expr>, Spanned<Expr>),
    Import(Spanned<ImportRange>),
    FnDef(Box<AstFunction>),
    TypeDef(AstTypeDef),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Param {
    pub name: Id,
    pub ty: Spanned<AstType>,
    pub is_mutable: bool,
}

#[derive(Debug, PartialEq, Clone)]
pub struct AstFunction {
    pub name: Id,
    pub params: Vec<Param>,
    pub return_ty: Spanned<AstType>,
    pub body: Spanned<Stmt>,
    pub ty_params: Vec<Spanned<Id>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct AstTypeDef {
    pub name: Id,
    pub ty: Spanned<AstType>,
    pub var_ids: Vec<Spanned<Id>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub main_stmts: Vec<Spanned<Stmt>>,
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
}

impl fmt::Display for AstType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AstType::Int => write!(f, "int"),
            AstType::String => write!(f, "string"),
            AstType::Bool => write!(f, "bool"),
            AstType::Unit => write!(f, "unit"),
            AstType::Named(id) => write!(f, "{}", IdMap::name(*id)),
            AstType::Pointer(ty, is_mutable) => write!(f, "*{}{}", format_bool(*is_mutable, "mut "), ty.kind),
            AstType::Array(ty, size) => write!(f, "[{}; {}]", ty.kind, size),
            AstType::Tuple(types) => {
                write!(f, "(")?;
                write_iter!(f, types.iter().map(|t| &t.kind))?;
                write!(f, ")")
            },
            AstType::Struct(fields) => {
                write!(f, "struct {{ ")?;
                write_iter!(f, fields.iter().map(|(id, ty)| format!("{}: {}", IdMap::name(id.kind), ty.kind)))?;
                write!(f, " }}")
            }
            AstType::App(name, types) => {
                write!(f, "{}<", IdMap::name(name.kind))?;
                write_iter!(f, types.iter().map(|t| &t.kind))?;
                write!(f, ">")
            },
        }
    }
}

pub fn dump_expr(expr: &Spanned<Expr>, strings: &[String], depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    match &expr.kind {
        Expr::Literal(Literal::Number(n)) => println!("{} {}", n, span_to_string(&expr.span)),
        Expr::Literal(Literal::String(i)) => println!("\"{}\" {}", escape_string(&strings[*i]), span_to_string(&expr.span)),
        Expr::Literal(Literal::Unit) => println!("() {}", span_to_string(&expr.span)),
        Expr::Literal(Literal::True) => println!("true {}", span_to_string(&expr.span)),
        Expr::Literal(Literal::False) => println!("false {}", span_to_string(&expr.span)),
        Expr::Literal(Literal::Null) => println!("__null__ {}", span_to_string(&expr.span)),
        Expr::Tuple(exprs) => {
            println!("tuple {}", span_to_string(&expr.span));
            for expr in exprs {
                dump_expr(&expr, strings, depth + 1);
            }
        },
        Expr::Struct(ty, fields) => {
            println!("struct {} {}", ty.kind, span_to_string(&expr.span));
            for (name, expr) in fields {
                print!("{}", "  ".repeat(depth + 1));
                println!("{}: {}", IdMap::name(name.kind), span_to_string(&name.span));
                dump_expr(expr, strings, depth + 2);
            }
        },
        Expr::Array(init_expr, size) => {
            println!("[{}] {}", size, span_to_string(&expr.span));
            dump_expr(init_expr, strings, depth + 1);
        },
        Expr::Field(expr, field) => {
            match field {
                Field::Number(i) => println!(".{} {}", i, span_to_string(&expr.span)),
                Field::Id(id) => println!(".{} {}", IdMap::name(*id), span_to_string(&expr.span)),
            };

            dump_expr(&expr, strings, depth + 1);
        },
        Expr::Subscript(expr, subscript) => {
            println!("subscript {}", span_to_string(&expr.span));
            dump_expr(&expr, strings, depth + 1);
            dump_expr(&subscript, strings, depth + 1);
        },
        Expr::Variable(name) => println!("{} {}", IdMap::name(*name), span_to_string(&expr.span)),
        Expr::BinOp(binop, lhs, rhs) => {
            println!("{} {}", binop.to_symbol(), span_to_string(&expr.span));
            dump_expr(&lhs, strings, depth + 1);
            dump_expr(&rhs, strings, depth + 1);
        },
        Expr::Call(name, args, tyargs) => {
            print!("call {}", name.kind);

            if !tyargs.is_empty() {
                print!(".<{}>", format_iter(tyargs.iter().map(|t| &t.kind)));
            }

            println!(" {}", span_to_string(&expr.span));
            for arg in args {
                dump_expr(&arg, strings, depth + 1);
            }
        },
        Expr::Address(expr_, is_mutable) => {
            println!("&{} {}", format_bool(*is_mutable, "mut"), span_to_string(&expr.span));
            dump_expr(&expr_, strings, depth + 1);
        },
        Expr::Dereference(expr_) => {
            println!("* {}", span_to_string(&expr.span));
            dump_expr(&expr_, strings, depth + 1);
        },
        Expr::Negative(expr_) => {
            println!("neg {}", span_to_string(&expr.span));
            dump_expr(&expr_, strings, depth + 1);
        },
        Expr::Alloc(expr_, is_mutable) => {
            println!("new{} {}", format_bool(*is_mutable, " mut"), span_to_string(&expr.span));
            dump_expr(&expr_, strings, depth + 1);
        },
    }
}

pub fn dump_stmt(stmt: &Spanned<Stmt>, strings: &[String], depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    match &stmt.kind {
        Stmt::Bind(name, expr, is_mutable) => {
            println!("let {}{} =", format_bool(*is_mutable, "mut "), IdMap::name(*name));
            dump_expr(&expr, strings, depth + 1);
        },
        Stmt::Assign(lhs, rhs) => {
            println!(":= {}", span_to_string(&stmt.span));
            dump_expr(&lhs, strings, depth + 1);
            dump_expr(&rhs, strings, depth + 1);
        },
        Stmt::Expr(expr) => {
            print!("\r");
            dump_expr(&expr, strings, depth);
        },
        Stmt::Block(stmts) => {
            println!("block {}", span_to_string(&stmt.span));
            for stmt in stmts {
                dump_stmt(&stmt, strings, depth + 1);
            }
        },
        Stmt::Return(expr) => {
            println!("return {}", span_to_string(&stmt.span));
            if let Some(expr) = expr {
                dump_expr(&expr, strings, depth + 1);
            }
        },
        Stmt::If(cond, body, else_stmt) => {
            println!("if {}", span_to_string(&stmt.span));
            dump_expr(&cond, strings, depth + 1);
            dump_stmt(&body, strings, depth + 1);
            if let Some(else_stmt) = else_stmt {
                dump_stmt(&else_stmt, strings, depth + 1);
            }
        },
        Stmt::While(cond, body) => {
            println!("while {}", span_to_string(&stmt.span));
            dump_expr(&cond, strings, depth + 1);
            dump_stmt(&body, strings, depth + 1);
        },
        Stmt::FnDef(func) => {
            print!("fn {}", IdMap::name(func.name));

            if !func.ty_params.is_empty() {
                print!("<");
                print!("{}", format_iter(func.ty_params.iter().map(|id| IdMap::name(id.kind))));
                print!(">");
            }

            println!("({}): {}",
                format_iter(func.params
                    .iter()
                    .map(|p| format!("{}{}: {}", format_bool(p.is_mutable, "mut "), IdMap::name(p.name), p.ty.kind))
                ),
                func.return_ty.kind,
            );
            dump_stmt(&func.body, strings, 1);
        },
        Stmt::TypeDef(ty) => {
            println!("type {}<{}> {}",
                IdMap::name(ty.name), 
                format_iter(ty.var_ids
                    .iter()
                    .map(|id| IdMap::name(id.kind))
                ),
                ty.ty.kind,
            );
        },
        Stmt::Import(name) => {
            println!("import {} {}", name.kind, span_to_string(&stmt.span));
        },
    }
}

pub fn dump_program(program: &Program, depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    for module_path in &program.imported_modules {
        println!("module {}", module_path);
    }

    for stmt in &program.main_stmts {
        dump_stmt(&stmt, &program.strings, 0);
    }
}

pub fn dump_ast(program: &Program) {
    dump_program(program, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::env;

    #[test]
    fn test_symbol_path_from_path() {
        let root = env::current_dir().unwrap();
        let path = PathBuf::from("test2/test.lang2");
        let actual = SymbolPath::from_path(&root, &path);

        let expected = SymbolPath::new()
            .append_str("test2")
            .append_str("test");
        assert_eq!(expected, actual);
    }
}
