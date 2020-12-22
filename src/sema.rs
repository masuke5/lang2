use crate::ast::{
    AstFunction as AstFunction_, Block as Block_, Expr as Expr_, Impl as Impl_,
    Program as Program_, Stmt as Stmt_, *,
};
use crate::error::{Error, ErrorList};
use crate::id::{reserved_id, Id, IdMap};
use crate::span::{Span, Spanned};
use crate::ty::*;
use crate::utils::{format_bool, format_iter};
use rustc_hash::{FxHashMap, FxHashSet};

type UExpr = Expr_<Empty>;
type UntypedExpr = Typed<UExpr, Empty>;
type UntypedStmt = Stmt_<Empty>;
type UntypedBlock = Block_<Empty>;
type UntypedImpl = Impl_<Empty>;
type UntypedAstFunction = AstFunction_<Empty>;
type UntypedProgram = Program_<Empty>;

type TExpr = Expr_<Type>;
type TypedExpr = Typed<TExpr, Type>;
type TypedStmt = Stmt_<Type>;
type TypedBlock = Block_<Type>;

#[derive(Debug, PartialEq, Clone)]
pub struct TypedParam {
    pub name: Id,
    pub ty: Type,
    pub is_mutable: bool,
    pub is_escaped: bool,
    pub is_in_heap: bool,
}

#[derive(Debug, Clone)]
pub struct TypedFunction {
    pub name: Id,
    pub params: Vec<TypedParam>,
    pub return_ty: Type,
    pub body: TypedExpr,
    pub ty_params: Vec<TypeVar>,
    pub has_escaped_variables: bool,
}

#[derive(Debug, Clone)]
pub struct TypedProgram {
    pub module_path: SymbolPath,
    // The function that a key is reserved_id::MAIN_FUNC is main function
    pub functions: FxHashMap<Id, TypedFunction>,
}

pub fn dump_typed_func(func: &TypedFunction, depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    print!("{}", IdMap::name(func.name));

    if func.has_escaped_variables {
        print!(" \x1b[32mhas escaped vars\x1b[0m");
    }

    if !func.ty_params.is_empty() {
        print!("<");
        print!(
            "{}",
            format_iter(func.ty_params.iter().map(|var| format!("{}", var)))
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
            p.ty
        ))),
        func.return_ty,
    );

    dump_expr(&func.body, depth + 1);
}

pub fn dump_typed_program(program: &TypedProgram, depth: usize) {
    // Print indent
    print!("{}", "  ".repeat(depth));

    for func in program.functions.values() {
        dump_typed_func(func, depth);
    }
}

macro_rules! try_some {
    () => {};
    (,$($rest:tt)*) => {
        try_some!($($rest)*);
    };
    (mut $var:ident $($rest:tt)*) => {
        let mut $var = $var?;
        try_some!($($rest)*);
    };
    ($var:ident $($rest:tt)*) => {
        let $var = $var?;
        try_some!($($rest)*);
    };
}

// Environment {{{

#[derive(Debug, Clone)]
struct Variable {
    is_mutable: bool,
    is_escaped: bool,
    is_in_heap: bool,
    ty: Type,
}

#[derive(Debug, Clone)]
struct Function {
    name: Id,
    params: Vec<Type>,
    return_ty: Type,
    module: SymbolPath,
    ty_params: Vec<(Id, TypeVar)>,
}

impl Function {
    pub fn generate_arrow_type(&self) -> Type {
        generate_func_type(&self.params, &self.return_ty, &self.ty_params)
    }
}

#[derive(Debug, Clone)]
enum Entry {
    Variable(Variable),
    Function(Function),
}

#[derive(Debug, Clone)]
struct TypeDef {
    name: Id,
    body: Option<TypeCon>,
    module: SymbolPath,
}

impl TypeDef {
    fn new_header(name: Id, module: SymbolPath) -> Self {
        Self {
            name,
            body: None,
            module,
        }
    }

    fn path(&self) -> SymbolPath {
        self.module.clone().append_id(self.name)
    }
}

#[derive(Debug, Clone)]
enum ScopeType {
    Def(TypeDef),
    Var(TypeVar),
}

#[derive(Debug, Clone)]
struct Scope {
    entries: FxHashMap<Id, Entry>,
    types: FxHashMap<Id, ScopeType>,
    modules: FxHashMap<Id, SymbolPath>,
    impl_funcs: FxHashMap<Id, FxHashMap<Id, Function>>,
}

impl Scope {
    fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
            types: FxHashMap::default(),
            modules: FxHashMap::default(),
            impl_funcs: FxHashMap::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct Environment {
    scopes: Vec<Scope>,
}

impl Environment {
    fn new() -> Self {
        Self {
            scopes: vec![Scope::new()],
        }
    }

    fn last_scope(&mut self) -> &mut Scope {
        if self.scopes.is_empty() {
            panic!("there is no scopes");
        }

        self.scopes.last_mut().unwrap()
    }

    fn find_type(&self, name: Id) -> Option<&ScopeType> {
        for scope in self.scopes.iter().rev() {
            if let Some(def) = scope.types.get(&name) {
                return Some(def);
            }
        }
        None
    }

    fn find_type_mut(&mut self, name: Id) -> Option<&mut ScopeType> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(def) = scope.types.get_mut(&name) {
                return Some(def);
            }
        }
        None
    }

    fn define_type(&mut self, name: Id, tydef: ScopeType) {
        let last_scope = self.last_scope();
        last_scope.types.insert(name, tydef);
    }

    fn types(&self) -> FxHashMap<Id, TypeDef> {
        let mut types = FxHashMap::default();
        for scope in self.scopes.iter() {
            for (name, ty) in &scope.types {
                if let ScopeType::Def(tydef) = ty {
                    types.insert(*name, tydef.clone());
                }
            }
        }
        types
    }

    fn find_entry(&self, name: Id) -> Option<&Entry> {
        for scope in self.scopes.iter().rev() {
            if let Some(entry) = scope.entries.get(&name) {
                return Some(entry);
            }
        }
        None
    }

    fn define_entry(&mut self, name: Id, entry: Entry) {
        let last_scope = self.last_scope();
        last_scope.entries.insert(name, entry);
    }

    fn entries(&self) -> FxHashMap<Id, Entry> {
        let mut entries = FxHashMap::default();
        for scope in self.scopes.iter() {
            for (name, entry) in &scope.entries {
                entries.insert(*name, entry.clone());
            }
        }
        entries
    }

    fn import_module(&mut self, name: Id, path: SymbolPath) {
        assert!(path.is_absolute);
        let last_scope = self.last_scope();
        last_scope.modules.insert(name, path);
    }

    fn find_module(&self, name: Id) -> Option<&SymbolPath> {
        for scope in self.scopes.iter().rev() {
            if let Some(module) = scope.modules.get(&name) {
                return Some(module);
            }
        }
        None
    }

    fn define_impl_func(&mut self, target: Id, func: Function) {
        assert!(self.find_type(target).is_some());
        let last_scope = self.last_scope();
        last_scope
            .impl_funcs
            .entry(target)
            .or_insert(FxHashMap::default())
            .insert(func.name, func);
    }

    fn find_impl_func(&self, target: Id, name: Id) -> Option<&Function> {
        for scope in self.scopes.iter().rev() {
            if let Some(funcs) = scope.impl_funcs.get(&target) {
                return funcs.get(&name);
            }
        }
        None
    }

    fn push_env(&mut self, mut other: Environment) {
        self.scopes.append(&mut other.scopes);
    }

    unsafe fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    unsafe fn pop_scope(&mut self) {
        self.scopes.pop().unwrap();
    }
}

macro_rules! pushed_scope {
    // FIXME: $blockの中のreturnやbreakに対応
    ($env:expr, $block:expr) => {{
        unsafe { $env.push_scope() };
        let result = $block;
        unsafe { $env.pop_scope() };
        result
    }};
}

// }}}

struct Analyzer<'a> {
    module_envs: &'a FxHashMap<SymbolPath, Environment>,
    module_path: SymbolPath,
    env: Environment,
    functions: FxHashMap<Id, TypedFunction>,
    uniq: usize,
}

impl<'a> Analyzer<'a> {
    fn new(module_envs: &'a FxHashMap<SymbolPath, Environment>, module_path: SymbolPath) -> Self {
        Self {
            env: module_envs[&module_path].clone(),
            module_envs,
            module_path,
            functions: FxHashMap::default(),
            uniq: 0,
        }
    }

    fn generate_inner_function_name(&mut self, name: Id) -> Id {
        let name = IdMap::name(name);
        let name = format!("$inner.{}.{}", self.uniq, name);
        self.uniq += 1;
        IdMap::new_id(&name)
    }

    fn generate_impl_function_name(target: Id, name: Id) -> Id {
        let name = format!("{}.{}", IdMap::name(target), IdMap::name(name));
        IdMap::new_id(&name)
    }

    fn expand_name(&self, ty: Type) -> Type {
        match ty {
            Type::App(TypeCon::Named(path), args) => {
                let ty = if path.parent().unwrap() == self.module_path {
                    self.env.find_type(path.tail().unwrap().id).cloned().expect(
                        "the type is created as Named despite the fact that it is undefined",
                    )
                } else {
                    let env = &self.module_envs[&path.parent().unwrap()];
                    env.find_type(path.tail().unwrap().id).cloned().expect(
                        "the type is created as Named despite the fact that it is undefined",
                    )
                };

                match ty {
                    ScopeType::Def(tydef) => {
                        let tycon = tydef.body.expect("type body should be set");
                        Type::App(tycon, args)
                    }
                    ScopeType::Var(var) => Type::Var(var),
                }
            }
            ty => ty,
        }
    }

    // TODO: Make it simple
    fn cast_for_binop(binop: &BinOp, lhs: &Type, rhs: &Type) -> Option<Type> {
        type B = BinOp;

        const NUMERIC_TYPES: &[Type] = &[Type::Int, Type::UInt, Type::Float];

        let (cast, allow_pointer, allow_float) = match binop {
            B::Add | B::Sub | B::Mul | B::Div => (true, false, true),
            B::Mod => (true, false, false),
            B::LShift | B::RShift | B::BitAnd | B::BitOr | B::BitXor => (false, false, false),
            B::Equal | B::NotEqual => (true, true, true),
            B::LessThan | B::LessThanOrEqual | B::GreaterThan | B::GreaterThanOrEqual => {
                (true, false, true)
            }
            B::And | B::Or => {
                return if *lhs == Type::Bool && *rhs == Type::Bool {
                    Some(Type::Bool)
                } else {
                    None
                };
            }
        };

        if allow_pointer {
            match (lhs, rhs) {
                (Type::App(TypeCon::Pointer(lm), a), Type::App(TypeCon::Pointer(rm), b))
                    if a[0] == b[0] =>
                {
                    return Some(Type::App(TypeCon::Pointer(*lm || *rm), vec![a[0].clone()]));
                }
                _ => {}
            }
        }

        if !NUMERIC_TYPES.contains(lhs) || !NUMERIC_TYPES.contains(rhs) {
            return None;
        }

        if !cast {
            if lhs != rhs {
                return None;
            }

            return Some(lhs.clone());
        }

        if !allow_float && (*lhs == Type::Float || *rhs == Type::Float) && lhs == rhs {
            return None;
        }

        let mut ty = lhs.clone();

        if allow_float && ty == Type::Float {
            if *rhs != Type::Float {
                return None;
            }
        }

        if *rhs == Type::Int {
            ty = Type::Int;
        }

        Some(ty)
    }

    fn get_type_to_path(&self, span: &Span, path: &SymbolPath) -> Option<Type> {
        assert!(path.is_absolute);

        // First consider path is a module
        if self.module_envs.contains_key(&path) {
            error!(&span, "path `{}` is a module", path);
            return None;
        }

        // Consider path is a function in impl
        if let Some(module_path) = path.parent().unwrap().parent() {
            let type_name = path.parent().unwrap().tail().unwrap().id;
            let func_name = path.tail().unwrap().id;
            if let Some(module_env) = self.module_envs.get(&module_path) {
                if let Some(func) = module_env.find_impl_func(type_name, func_name) {
                    return Some(func.generate_arrow_type());
                }
            }
        }

        // Consider path is a function or type named item_name in a module
        // named module_path
        let module_path = path.parent().unwrap();
        let item_name = path.tail().unwrap();

        if let Some(module_env) = self.module_envs.get(&module_path) {
            if module_env.find_type(item_name.id).is_some() {
                error!(&span, "path `{}` is a type", path);
                return None;
            }

            match module_env.find_entry(item_name.id) {
                Some(Entry::Function(func)) => Some(func.generate_arrow_type()),
                Some(Entry::Variable(..)) => {
                    error!(&span, "external module variable");
                    None
                }
                None => {
                    error!(
                        &span,
                        "module `{}` has no function or type named `{}`",
                        module_path,
                        IdMap::name(item_name.id)
                    );
                    None
                }
            }
        } else {
            error!(&span, "undefined symbol `{}`", path);
            None
        }
    }

    fn get_path_type(&self, span: &Span, path: &SymbolPath) -> Option<Type> {
        // This is safe because the path has 2 segments at least
        let head = path.head().unwrap();
        let rest = path.pop_head().unwrap();

        // Consider that the path is an implementation function in a module
        if rest.segments.len() == 1 {
            let tail = rest.tail().unwrap();

            // In self module
            if let Some(func) = self.env.find_impl_func(head.id, tail.id) {
                return Some(func.generate_arrow_type());
            }

            // In external module
            if let Some(ScopeType::Def(tydef)) = self.env.find_type(head.id) {
                if let Some(func) = self.module_envs[&tydef.module].find_impl_func(head.id, tail.id)
                {
                    return Some(func.generate_arrow_type());
                }
            }
        }

        match self.env.find_module(head.id) {
            Some(module_path) if !path.is_absolute => {
                let absolute_path = module_path.clone().join(&rest);
                self.get_type_to_path(&span, &absolute_path)
            }
            _ => {
                // Consider the path is absolute
                let mut path = path.clone();
                path.is_absolute = true;
                self.get_type_to_path(&span, &path)
            }
        }
    }

    fn analyze_expr(&mut self, func: &Function, expr: UntypedExpr) -> Option<TypedExpr> {
        let (kind, ty) = match expr.kind {
            UExpr::Literal(Literal::Number(n)) => (TExpr::Literal(Literal::Number(n)), Type::Int),
            UExpr::Literal(Literal::UnsignedNumber(n)) => {
                (TExpr::Literal(Literal::UnsignedNumber(n)), Type::UInt)
            }
            UExpr::Literal(Literal::Float(n)) => (TExpr::Literal(Literal::Float(n)), Type::Float),
            UExpr::Literal(Literal::String(s)) => {
                (TExpr::Literal(Literal::String(s)), Type::String)
            }
            UExpr::Literal(Literal::Char(c)) => (TExpr::Literal(Literal::Char(c)), Type::Char),
            UExpr::Literal(Literal::Unit) => (TExpr::Literal(Literal::Unit), Type::Unit),
            UExpr::Literal(Literal::True) => (TExpr::Literal(Literal::True), Type::Bool),
            UExpr::Literal(Literal::False) => (TExpr::Literal(Literal::False), Type::Bool),
            UExpr::Literal(Literal::Null) => (TExpr::Literal(Literal::Null), Type::Null),
            UExpr::Tuple(exprs) => {
                let mut new_exprs = Vec::with_capacity(exprs.len());
                let mut types = Vec::with_capacity(exprs.len());
                for expr in exprs {
                    let expr = self.analyze_expr(func, expr);
                    if let Some(expr) = expr {
                        types.push(expr.ty.clone());
                        new_exprs.push(expr);
                    }
                }
                (TExpr::Tuple(new_exprs), Type::App(TypeCon::Tuple, types))
            }
            UExpr::Struct(ast_ty, fields) => {
                // The parser only accept a named type for `ty` like App(Named(NAME), [TYARGS])

                fn get_field_types(struct_ty: &Type) -> Option<FxHashMap<Id, Type>> {
                    match struct_ty {
                        Type::App(TypeCon::Struct(names), types) => {
                            Some(names.iter().copied().zip(types.clone()).collect())
                        }
                        _ => None,
                    }
                }

                let ty = analyze_type(&self.env, &ast_ty)?;
                let result_ty = ty.clone();

                let ty = self.expand_name(ty);
                let ty = expand_unique(ty);
                let mut ty = subst(ty, &FxHashMap::default());

                // Get the struct field types
                let mut field_types = match get_field_types(&ty) {
                    Some(ft) => ft,
                    _ => {
                        error!(&ast_ty.span, "type `{}` is not structure", result_ty);
                        return None;
                    }
                };

                // Analyze the fields
                let mut new_fields = Vec::with_capacity(fields.len());
                let mut analyzed_fields = FxHashSet::default();
                let mut var_map = FxHashMap::default();

                for (name, expr) in fields {
                    if analyzed_fields.contains(&name.kind) {
                        error!(&name.span, "duplicated field");
                    }

                    match field_types.get(&name.kind) {
                        Some(field_ty) => {
                            analyzed_fields.insert(name.kind);
                            if let Some(expr) = self.analyze_expr_with_conv(func, expr, field_ty) {
                                if let Type::Var(var) = field_ty {
                                    var_map.insert(*var, expr.ty.clone());
                                    ty = subst(ty, &var_map);
                                    field_types = get_field_types(&ty).unwrap();
                                } else {
                                    unify(&expr.span, &expr.ty, field_ty);
                                }

                                new_fields.push((name, expr));
                            }
                        }
                        _ => {
                            error!(
                                &name.span,
                                "type `{}` has no field named `{}`",
                                result_ty,
                                IdMap::name(name.kind)
                            );
                            continue;
                        }
                    }
                }

                let ty_func = self.expand_name(result_ty.clone());
                let mut args = Vec::new();
                match ty_func {
                    Type::App(TypeCon::Fun(params, ..), ..) => {
                        args.reserve(params.len());
                        for param in params {
                            let arg_ty = var_map
                                .get(&param)
                                .cloned()
                                .unwrap_or_else(|| Type::Var(param));
                            args.push(arg_ty);
                        }
                    }
                    _ => panic!(),
                }

                let result_ty = if args.is_empty() {
                    result_ty
                } else {
                    match result_ty {
                        Type::App(tycon, ..) => Type::App(tycon, args),
                        _ => panic!(),
                    }
                };

                (TExpr::Struct(ast_ty, new_fields), result_ty)
            }
            UExpr::Array(elem_expr, size) => {
                let elem_expr = self.analyze_expr(func, *elem_expr)?;
                let elem_ty = elem_expr.ty.clone();
                (
                    TExpr::Array(box elem_expr, size),
                    Type::App(TypeCon::Array(size), vec![elem_ty]),
                )
            }
            UExpr::Field(comp_expr, field) => {
                let comp_expr = self.analyze_expr(func, *comp_expr)?;

                // Check the compound expression type and get the field type
                let ty = match &comp_expr.ty {
                    Type::App(TypeCon::Pointer(..), types) => types[0].clone(),
                    ty => ty.clone(),
                };
                let ty = self.expand_name(ty);
                let ty = expand_unique(ty);
                let ty = subst(ty, &FxHashMap::default());

                let is_mutable = match &comp_expr.ty {
                    Type::App(TypeCon::Pointer(is_mutable), ..) => *is_mutable,
                    _ => comp_expr.is_mutable,
                };

                let result_ty = match field {
                    Field::Number(n) => match ty {
                        Type::App(TypeCon::Tuple, types) => {
                            // Check the field existence
                            if n < types.len() {
                                types[n].clone()
                            } else {
                                error!(
                                    &expr.span,
                                    "tuple `{}` has no {} elements", comp_expr.ty, n
                                );
                                return None;
                            }
                        }
                        _ => {
                            error!(&comp_expr.span, "type `{}` is not tuple", comp_expr.ty);
                            return None;
                        }
                    },
                    Field::Id(name) => match ty {
                        Type::App(TypeCon::Struct(names), types) => {
                            // Check the field existence
                            let defined_field = names
                                .into_iter()
                                .zip(types)
                                .find(|(fname, _)| *fname == name);
                            if let Some((_, field_ty)) = defined_field {
                                field_ty
                            } else {
                                error!(
                                    &comp_expr.span,
                                    "type `{}` has no field named `{}`",
                                    comp_expr.ty,
                                    IdMap::name(name)
                                );
                                return None;
                            }
                        }
                        _ => {
                            error!(&comp_expr.span, "type `{}` is not structure", comp_expr.ty);
                            return None;
                        }
                    },
                };

                return Some(TypedExpr {
                    is_mutable,
                    is_lvalue: true,
                    is_in_heap: comp_expr.is_in_heap,
                    kind: TExpr::Field(box comp_expr, field),
                    span: expr.span,
                    ty: result_ty,
                    converted_from: None,
                });
            }
            UExpr::Subscript(array_expr, subscript_expr) => {
                let array_expr = self.analyze_expr(func, *array_expr);
                let subscript_expr = self.analyze_expr_with_conv(func, *subscript_expr, &Type::Int);
                try_some!(array_expr, subscript_expr);

                unify(&subscript_expr.span, &subscript_expr.ty, &Type::Int);

                // Check if the array_expr type is array or slice
                let array_ty = match array_expr.ty.clone() {
                    Type::App(TypeCon::Pointer(..), types) => types[0].clone(),
                    ty => ty,
                };
                let array_ty = self.expand_name(array_ty);
                let array_ty = expand_unique(array_ty);
                let array_ty = subst(array_ty, &FxHashMap::default());

                // Check and get the element type
                let element_ty = match array_ty {
                    Type::App(TypeCon::Array(..), types) => types[0].clone(),
                    Type::App(TypeCon::Slice(is_mutable), types) => {
                        Type::App(TypeCon::Pointer(is_mutable), types)
                    }
                    _ => {
                        error!(
                            &array_expr.span,
                            "type `{}` is not array or slice", array_expr.ty
                        );
                        return None;
                    }
                };

                let is_mutable = match &array_expr.ty {
                    Type::App(TypeCon::Pointer(is_mutable), ..) => *is_mutable,
                    _ => array_expr.is_mutable,
                };
                let is_lvalue = array_expr.is_lvalue;

                return Some(TypedExpr {
                    kind: TExpr::Subscript(box array_expr, box subscript_expr),
                    span: expr.span,
                    ty: element_ty,
                    is_mutable,
                    is_lvalue,
                    is_in_heap: false,
                    converted_from: None,
                });
            }
            UExpr::Range(..) => {
                error!(
                    &expr.span,
                    "range expressions outside a subscript expression are unsupported currently"
                );
                return None;
            }
            UExpr::BinOp(binop, lhs, rhs) => {
                let lhs = self.analyze_expr(func, *lhs);
                let rhs = self.analyze_expr(func, *rhs);
                try_some!(lhs, rhs);

                let ty = match Self::cast_for_binop(&binop, &lhs.ty, &rhs.ty) {
                    Some(ty) => ty,
                    None => {
                        error!(&expr.span, "{} {} {}", lhs.ty, binop.to_symbol(), rhs.ty);
                        return None;
                    }
                };

                let lhs = self.convert_analyzed_expr(lhs, &ty);
                let rhs = self.convert_analyzed_expr(rhs, &ty);
                try_some!(lhs, rhs);
                unify(&lhs.span, &lhs.ty, &ty);
                unify(&rhs.span, &rhs.ty, &ty);

                let result_ty = match &binop {
                    BinOp::Equal
                    | BinOp::NotEqual
                    | BinOp::LessThan
                    | BinOp::LessThanOrEqual
                    | BinOp::GreaterThan
                    | BinOp::GreaterThanOrEqual => Type::Bool,
                    _ => ty,
                };

                (TExpr::BinOp(binop, box lhs, box rhs), result_ty)
            }
            UExpr::Variable(name, is_mutable) => {
                let entry = match self.env.find_entry(name) {
                    Some(entry) => entry,
                    None => {
                        error!(
                            &expr.span,
                            "variable or function `{}` is undefined",
                            IdMap::name(name)
                        );
                        return None;
                    }
                };

                match entry {
                    Entry::Variable(var) => {
                        return Some(TypedExpr {
                            kind: TExpr::Variable(name, is_mutable),
                            span: expr.span,
                            ty: var.ty.clone(),
                            is_mutable: var.is_mutable,
                            is_lvalue: true,
                            is_in_heap: var.is_in_heap,
                            converted_from: None,
                        });
                    }
                    Entry::Function(func) => {
                        let ty = func.generate_arrow_type();
                        (TExpr::Variable(name, is_mutable), ty)
                    }
                }
            }
            UExpr::Path(path) => {
                let ty = self.get_path_type(&expr.span, &path)?;
                (TExpr::Path(path), ty)
            }
            UExpr::Call(func_expr, arg_expr) => {
                let func_expr = self.analyze_expr(func, *func_expr);
                let arg_expr = self.analyze_expr(func, *arg_expr);
                try_some!(func_expr, arg_expr);

                // Get the return type and check the argument type
                let func_ty = self.expand_name(func_expr.ty.clone());
                let func_ty = expand_unique(func_ty);
                let func_ty = subst(func_ty, &FxHashMap::default());

                let return_ty = match func_ty {
                    Type::Poly(params, box Type::App(TypeCon::Arrow, types)) => {
                        if let Type::Var(var) = &types[0] {
                            let mut map = FxHashMap::default();
                            map.insert(*var, arg_expr.ty.clone());
                            subst(types[1].clone(), &map)
                        } else {
                            unify(&arg_expr.span, &arg_expr.ty, &types[0]);
                            Type::Poly(params, box types[1].clone())
                        }
                    }
                    Type::App(TypeCon::Arrow, types) => {
                        unify(&arg_expr.span, &arg_expr.ty, &types[0]);
                        types[1].clone()
                    }
                    ty => {
                        error!(&func_expr.span, "type `{}` is not function", ty);
                        return None;
                    }
                };

                (TExpr::Call(box func_expr, box arg_expr), return_ty)
            }
            UExpr::Dereference(ptr_expr) => {
                let ptr_expr = self.analyze_expr(func, *ptr_expr)?;

                let ty = self.expand_name(ptr_expr.ty.clone());
                let ty = expand_unique(ty);
                let ty = subst(ty, &FxHashMap::default());
                let (ty, is_mutable) = match &ty {
                    Type::App(TypeCon::Pointer(is_mutable), types) => {
                        (types[0].clone(), *is_mutable)
                    }
                    _ => {
                        error!(&ptr_expr.span, "type `{}` is not pointer", ty);
                        return None;
                    }
                };

                return Some(TypedExpr {
                    kind: TExpr::Dereference(box ptr_expr),
                    ty,
                    span: expr.span,
                    is_lvalue: true,
                    is_mutable,
                    is_in_heap: false,
                    converted_from: None,
                });
            }
            UExpr::Address(inner_expr, is_mutable) => match inner_expr.kind {
                UExpr::Subscript(
                    list_expr,
                    box UntypedExpr {
                        kind: UExpr::Range(start_expr, end_expr),
                        span: subscript_span @ _,
                        ..
                    },
                ) => {
                    let list_expr = self.analyze_expr(func, *list_expr);
                    let start_expr = if let Some(start_expr) = start_expr {
                        self.analyze_expr_with_conv(func, *start_expr, &Type::Int)
                            .map(|expr| Some(expr))
                    } else {
                        Some(None)
                    };
                    let end_expr = if let Some(end_expr) = end_expr {
                        self.analyze_expr_with_conv(func, *end_expr, &Type::Int)
                            .map(|expr| Some(expr))
                    } else {
                        Some(None)
                    };
                    try_some!(list_expr, start_expr, end_expr);

                    if let Some(start_expr) = &start_expr {
                        unify(&start_expr.span, &start_expr.ty, &Type::Int);
                    }
                    if let Some(end_expr) = &end_expr {
                        unify(&end_expr.span, &end_expr.ty, &Type::Int);
                    }

                    if list_expr.is_lvalue && is_mutable && !list_expr.is_mutable {
                        error!(&list_expr.span, "this expression is immutable");
                    }

                    let list_ty = self.expand_name(list_expr.ty.clone());
                    let list_ty = expand_unique(list_ty);
                    let list_ty = subst(list_ty, &FxHashMap::default());
                    let element_ty = match list_ty {
                        Type::App(TypeCon::Array(..), types)
                        | Type::App(TypeCon::Slice(..), types) => types[0].clone(),
                        _ => {
                            error!(
                                &list_expr.span,
                                "type `{}` is not array or slice", list_expr.ty
                            );
                            return None;
                        }
                    };

                    let result_ty = Type::App(TypeCon::Slice(is_mutable), vec![element_ty]);
                    (
                        TExpr::Address(
                            box TypedExpr::new(
                                TExpr::Subscript(
                                    box list_expr,
                                    box TypedExpr::new(
                                        TExpr::Range(
                                            start_expr.map(Box::new),
                                            end_expr.map(Box::new),
                                        ),
                                        subscript_span,
                                        Type::Null,
                                    ),
                                ),
                                inner_expr.span,
                                Type::Null,
                            ),
                            is_mutable,
                        ),
                        result_ty,
                    )
                }
                _ => {
                    let inner_expr = self.analyze_expr(func, *inner_expr)?;
                    if inner_expr.is_lvalue && is_mutable && !inner_expr.is_mutable {
                        error!(&inner_expr.span, "this expression is immutable");
                    }

                    let result_ty =
                        Type::App(TypeCon::Pointer(is_mutable), vec![inner_expr.ty.clone()]);
                    (TExpr::Address(box inner_expr, is_mutable), result_ty)
                }
            },
            UExpr::Negative(inner_expr) => {
                let inner_expr = self.analyze_expr(func, *inner_expr)?;
                unify(&inner_expr.span, &inner_expr.ty, &Type::Int);
                (TExpr::Negative(box inner_expr), Type::Int)
            }
            UExpr::Not(inner_expr) => {
                let inner_expr = self.analyze_expr(func, *inner_expr)?;
                unify(&inner_expr.span, &inner_expr.ty, &Type::Bool);
                (TExpr::Not(box inner_expr), Type::Bool)
            }
            UExpr::Block(block) => {
                let block = pushed_scope!(self.env, { self.analyze_block(func, block, false) })?;
                let ty = block.result_expr.ty.clone();
                (TExpr::Block(block), ty)
            }
            UExpr::If(cond_expr, then_expr, else_expr) => {
                let cond_expr = self.analyze_expr_with_conv(func, *cond_expr, &Type::Bool);
                let then_expr = self.analyze_expr(func, *then_expr);
                try_some!(cond_expr, then_expr);

                unify(&cond_expr.span, &cond_expr.ty, &Type::Bool);

                if let Some(else_expr) = else_expr {
                    let else_expr = self.analyze_expr_with_conv(func, *else_expr, &then_expr.ty)?;
                    unify(&else_expr.span, &else_expr.ty, &then_expr.ty);
                    let ty = then_expr.ty.clone();
                    (
                        TExpr::If(box cond_expr, box then_expr, Some(box else_expr)),
                        ty,
                    )
                } else {
                    let ty = then_expr.ty.clone();
                    (TExpr::If(box cond_expr, box then_expr, None), ty)
                }
            }
            UExpr::App(inner_expr, ty_args) => {
                let inner_expr = self.analyze_expr(func, *inner_expr)?;

                // Analyze the type arguments
                let mut new_ty_args = Vec::with_capacity(ty_args.len());
                for ty_arg in &ty_args {
                    if let Some(ty) = analyze_type(&self.env, ty_arg) {
                        new_ty_args.push(ty);
                    }
                }

                if new_ty_args.len() != ty_args.len() {
                    return None;
                }

                let ty = match &inner_expr.ty {
                    Type::Poly(params, ty) => {
                        let map: FxHashMap<TypeVar, Type> =
                            params.iter().copied().zip(new_ty_args).collect();
                        subst(*ty.clone(), &map)
                    }
                    ty => {
                        error!(&inner_expr.span, "type `{}` is not polymorphic type", ty);
                        return None;
                    }
                };

                (TExpr::App(box inner_expr, ty_args), ty)
            }
        };

        Some(TypedExpr::new(kind, expr.span, ty))
    }

    fn analyze_expr_with_conv(
        &mut self,
        func: &Function,
        expr: UntypedExpr,
        ty: &Type,
    ) -> Option<TypedExpr> {
        let expr = self.analyze_expr(func, expr)?;
        self.convert_analyzed_expr(expr, ty)
    }

    fn convert_analyzed_expr(&self, mut expr: TypedExpr, ty: &Type) -> Option<TypedExpr> {
        if let TExpr::Tuple(exprs) = &mut expr.kind {
            if let Type::App(TypeCon::Tuple, types) = ty {
                let mut converted_types = Vec::with_capacity(types.len());
                for (expr, formal_ty) in exprs.iter_mut().zip(types) {
                    if let Some(converted_expr) =
                        self.convert_analyzed_expr(expr.clone(), formal_ty)
                    {
                        converted_types.push(converted_expr.ty.clone());
                        *expr = converted_expr;
                    }
                }

                expr.ty = Type::App(TypeCon::Tuple, converted_types);
                return Some(expr);
            }
        }

        match (&expr.ty, ty) {
            (l, r) if l == r => {}
            (Type::Int, Type::UInt)
            | (Type::UInt, Type::Int)
            // Pointer and null
            | (Type::App(TypeCon::Pointer(..), ..), Type::Null)
            | (Type::Null, Type::App(TypeCon::Pointer(..), ..)) => {}
            // Pointer weakening
            (Type::App(TypeCon::Pointer(true), ltypes), Type::App(TypeCon::Pointer(false), rtypes))
            | (Type::App(TypeCon::Slice(true), ltypes), Type::App(TypeCon::Slice(false), rtypes))
                if ltypes[0] == rtypes[0] => {},
            _ =>  return Some(expr),
        }

        if expr.ty != *ty {
            expr.converted_from = Some(expr.ty);
            expr.ty = ty.clone();
        }

        Some(expr)
    }

    fn analyze_func(
        &mut self,
        name: Id,
        func: UntypedAstFunction,
        header: &Function,
    ) -> Option<TypedFunction> {
        // Analyze the function body
        let body = pushed_scope!(self.env, {
            // Define the type parameters
            for (name, var) in &header.ty_params {
                self.env.define_type(*name, ScopeType::Var(*var));
            }

            // Define the parameters as variables
            for (ty, param) in header.params.iter().zip(&func.params) {
                self.env.define_entry(
                    param.name,
                    Entry::Variable(Variable {
                        ty: ty.clone(),
                        is_mutable: param.is_mutable,
                        is_escaped: param.is_escaped,
                        is_in_heap: param.is_in_heap,
                    }),
                );
            }

            self.analyze_expr(&header, func.body)
        })?;

        // Generate the function metadata
        let params = func
            .params
            .iter()
            .zip(&header.params)
            .map(|(param, ty)| TypedParam {
                name: param.name,
                ty: ty.clone(),
                is_mutable: param.is_mutable,
                is_escaped: param.is_escaped,
                is_in_heap: param.is_in_heap,
            })
            .collect();

        Some(TypedFunction {
            name,
            params,
            body,
            return_ty: header.return_ty.clone(),
            ty_params: header.ty_params.iter().map(|(_, var)| *var).collect(),
            has_escaped_variables: func.has_escaped_variables,
        })
    }

    fn analyze_block(
        &mut self,
        func: &Function,
        block: UntypedBlock,
        is_top_level: bool,
    ) -> Option<TypedBlock> {
        if !is_top_level {
            generate_type_headers(&block, &self.module_path, &mut self.env);
            analyze_typedef(&block, &mut self.env);
            generate_function_headers(&block, &self.module_path, &mut self.env);
        }

        let mut stmts = Vec::with_capacity(block.stmts.len());
        for stmt in block.stmts {
            if let Some(stmt) = self.analyze_stmt(func, stmt) {
                stmts.push(stmt);
            }
        }

        let result_expr = self.analyze_expr(func, *block.result_expr)?;

        // Analyze functions
        for block_func in block.functions {
            let header = match self.env.find_entry(block_func.name.kind) {
                Some(Entry::Function(func)) => func.clone(),
                _ => return None,
            };

            // Generate the function name
            let func_name = if is_top_level {
                block_func.name.kind
            } else {
                self.generate_inner_function_name(block_func.name.kind)
            };

            if let Some(func) = self.analyze_func(func_name, block_func, &header) {
                self.functions.insert(func.name, func);
            }
        }

        Some(TypedBlock {
            types: Vec::new(),
            functions: Vec::new(),
            stmts,
            result_expr: box result_expr,
        })
    }

    fn import_by_range(&mut self, span: &Span, path: &SymbolPath, range: &ImportRange) {
        match range {
            ImportRange::Symbol(..) | ImportRange::Renamed(..) => {
                let (name, renamed) = match range {
                    ImportRange::Symbol(name) => (*name, *name),
                    ImportRange::Renamed(name, renamed) => (*name, *renamed),
                    _ => unreachable!(),
                };

                let joined_path = path.clone().append_id(name);
                if self.module_envs.contains_key(&joined_path) {
                    self.env.import_module(renamed, joined_path);
                    return;
                }

                // Consider the module named `path` includes the function or type named `name`
                if let Some(module_env) = self.module_envs.get(path) {
                    // Cannot import a function or type in self module
                    if *path != self.module_path {
                        if module_env.find_type(name).is_some() {
                            self.env.define_type(
                                renamed,
                                ScopeType::Def(TypeDef::new_header(name, path.clone())),
                            );
                            return;
                        }
                        if let Some(entry @ Entry::Function(_)) = module_env.find_entry(name) {
                            self.env.define_entry(renamed, entry.clone());
                            return;
                        }
                        if let Some(Entry::Variable(_)) = module_env.find_entry(name) {
                            error!(span, "cannot import variables in other modules");
                            return;
                        }
                    } else {
                        if module_env.find_type(name).is_some() {
                            error!(span, "cannot import a function or type in self module");
                        }
                        if let Some(Entry::Function(_)) = module_env.find_entry(name) {
                            error!(span, "cannot import a function or type in self module");
                        }
                    }
                }

                error!(span, "unresolved import");
            }
            ImportRange::All => {
                let module_env = match self.module_envs.get(path) {
                    Some(module_env) => module_env,
                    None => {
                        error!(span, "module `{}` not found", path);
                        return;
                    }
                };

                for (name, tydef) in module_env.types() {
                    self.env.define_type(name, ScopeType::Def(tydef));
                }
                for (name, entry) in module_env.entries() {
                    self.env.define_entry(name, entry);
                }
            }
            ImportRange::Scope(name, rest) => {
                self.import_by_range(span, &path.clone().append_id(*name), rest);
            }
            ImportRange::Multiple(ranges) => {
                for range in ranges {
                    self.import_by_range(span, path, range);
                }
            }
            ImportRange::Root(inner) => {
                self.import_by_range(span, &SymbolPath::new_absolute(), inner);
            }
        }
    }

    // `func` is the function that includes `stmt`
    fn analyze_stmt(
        &mut self,
        func: &Function,
        stmt: Spanned<UntypedStmt>,
    ) -> Option<Spanned<TypedStmt>> {
        let new_stmt = match stmt.kind {
            UntypedStmt::Bind(name, ty, init_expr, is_mutable, is_escaped, is_in_heap) => {
                let (ty, init_expr) = match ty {
                    Some(ty) => {
                        let ty = analyze_type(&self.env, &ty)?;
                        let init_expr = self.analyze_expr_with_conv(func, *init_expr, &ty)?;
                        unify(&init_expr.span, &init_expr.ty, &ty);
                        (ty, init_expr)
                    }
                    None => {
                        let init_expr = self.analyze_expr(func, *init_expr)?;
                        (init_expr.ty.clone(), init_expr)
                    }
                };

                self.env.define_entry(
                    name,
                    Entry::Variable(Variable {
                        is_mutable,
                        is_escaped,
                        is_in_heap,
                        ty: ty.clone(),
                    }),
                );

                TypedStmt::Bind(
                    name,
                    None,
                    box init_expr,
                    is_mutable,
                    is_escaped,
                    is_in_heap,
                )
            }
            UntypedStmt::Expr(expr) => {
                let expr = self.analyze_expr(func, expr)?;
                TypedStmt::Expr(expr)
            }
            UntypedStmt::Return(expr) => {
                let expr = match expr {
                    Some(expr) => {
                        let expr = self.analyze_expr_with_conv(func, expr, &func.return_ty)?;
                        unify(&expr.span, &expr.ty, &func.return_ty);
                        Some(expr)
                    }
                    None => None,
                };
                TypedStmt::Return(expr)
            }
            UntypedStmt::While(cond, body) => {
                let cond = self.analyze_expr_with_conv(func, cond, &Type::Bool);
                let body = self.analyze_stmt(func, *body);
                try_some!(cond, body);

                unify(&cond.span, &cond.ty, &Type::Bool);

                TypedStmt::While(cond, box body)
            }
            UntypedStmt::Assign(lhs, rhs) => {
                let lhs = self.analyze_expr(func, lhs)?;
                let rhs = self.analyze_expr_with_conv(func, *rhs, &lhs.ty)?;
                unify(&rhs.span, &lhs.ty, &rhs.ty);

                if !lhs.is_lvalue {
                    error!(&lhs.span, "this expression is not lvalue");
                } else if !lhs.is_mutable {
                    error!(&lhs.span, "this expression is not mutable");
                }

                TypedStmt::Assign(lhs, box rhs)
            }
            UntypedStmt::Import(range) => {
                let module_path = self.module_path.clone();
                self.import_by_range(&stmt.span, &module_path.parent().unwrap(), &range);
                TypedStmt::Import(range)
            }
        };

        Some(Spanned::new(new_stmt, stmt.span))
    }

    fn analyze(mut self, program: UntypedProgram) -> Option<TypedProgram> {
        // Create a function as the main block
        let main_func = Function {
            name: *reserved_id::MAIN_FUNC,
            params: Vec::new(),
            return_ty: Type::Unit,
            module: self.module_path.clone(),
            ty_params: Vec::new(),
        };

        let block = pushed_scope!(self.env, {
            let block = match self.analyze_block(&main_func, program.main, true) {
                Some(block) => block,
                None => return None,
            };

            // Analyze implement function bodies
            for imp in program.impls {
                for func in imp.functions {
                    let env = &self.module_envs[&program.module_path];
                    let func_name =
                        Self::generate_impl_function_name(imp.target.kind, func.name.kind);
                    let func_header = env.find_impl_func(imp.target.kind, func.name.kind).unwrap();
                    if let Some(func) = self.analyze_func(func_name, func, func_header) {
                        self.functions.insert(func_name, func);
                    }
                }
            }

            block
        });

        self.functions.insert(
            main_func.name,
            TypedFunction {
                name: main_func.name,
                params: Vec::new(),
                return_ty: Type::Unit,
                body: Typed::new(
                    TExpr::Block(block.clone()),
                    Span::zero(*reserved_id::MAIN_FUNC),
                    Type::Unit,
                ),
                ty_params: Vec::new(),
                has_escaped_variables: false,
            },
        );

        Some(TypedProgram {
            module_path: self.module_path,
            functions: self.functions,
        })
    }
}

fn import_only_type_by_range(
    env: &mut Environment,
    module_envs: &FxHashMap<SymbolPath, Environment>,
    span: &Span,
    path: &SymbolPath,
    range: &ImportRange,
) {
    match range {
        ImportRange::Symbol(..) | ImportRange::Renamed(..) => {
            let (name, renamed) = match range {
                ImportRange::Symbol(name) => (*name, *name),
                ImportRange::Renamed(name, renamed) => (*name, *renamed),
                _ => unreachable!(),
            };

            let joined_path = path.clone().append_id(name);
            if module_envs.contains_key(&joined_path) {
                env.import_module(renamed, joined_path);
                return;
            }

            // Consider the module named `path` includes the function or type named `name`
            if let Some(module_env) = module_envs.get(path) {
                if module_env.find_type(name).is_some() {
                    env.define_type(
                        renamed,
                        ScopeType::Def(TypeDef::new_header(name, path.clone())),
                    );
                }
            }
        }
        ImportRange::All => {
            let module_env = match module_envs.get(path) {
                Some(module_env) => module_env,
                None => return,
            };

            for (name, tydef) in module_env.types() {
                env.define_type(name, ScopeType::Def(tydef));
            }
        }
        ImportRange::Scope(name, rest) => {
            import_only_type_by_range(env, module_envs, span, &path.clone().append_id(*name), rest);
        }
        ImportRange::Multiple(ranges) => {
            for range in ranges {
                import_only_type_by_range(env, module_envs, span, path, range);
            }
        }
        ImportRange::Root(inner) => {
            import_only_type_by_range(env, module_envs, span, &SymbolPath::new_absolute(), inner);
        }
    }
}

fn generate_type_headers(block: &UntypedBlock, module_path: &SymbolPath, env: &mut Environment) {
    // Get type on top level
    for tydef in &block.types {
        env.define_type(
            tydef.name,
            ScopeType::Def(TypeDef::new_header(tydef.name, module_path.clone())),
        );
    }
}

fn analyze_type(env: &Environment, ty: &Spanned<AstType>) -> Option<Type> {
    match &ty.kind {
        AstType::Int => Some(Type::Int),
        AstType::UInt => Some(Type::UInt),
        AstType::Float => Some(Type::Float),
        AstType::Bool => Some(Type::Bool),
        AstType::Char => Some(Type::Char),
        AstType::Unit => Some(Type::Unit),
        AstType::String => Some(Type::String),
        AstType::Named(name) => match env.find_type(*name) {
            Some(ScopeType::Def(def)) => Some(Type::App(TypeCon::Named(def.path()), Vec::new())),
            Some(ScopeType::Var(var)) => Some(Type::Var(*var)),
            None => {
                error!(&ty.span, "type `{}` not found", IdMap::name(*name));
                None
            }
        },
        AstType::Pointer(ty, is_mutable) => Some(Type::App(
            TypeCon::Pointer(*is_mutable),
            vec![analyze_type(env, ty)?],
        )),
        AstType::Array(elem_ty, size) => Some(Type::App(
            TypeCon::Array(*size),
            vec![analyze_type(env, elem_ty)?],
        )),
        AstType::Slice(elem_ty, is_mutable) => Some(Type::App(
            TypeCon::Slice(*is_mutable),
            vec![analyze_type(env, elem_ty)?],
        )),
        AstType::Tuple(types) => Some(Type::App(
            TypeCon::Tuple,
            types
                .iter()
                .map(|ty| analyze_type(env, ty))
                .collect::<Option<Vec<Type>>>()?,
        )),
        AstType::Struct(fields) => Some(Type::App(
            TypeCon::Struct(fields.iter().map(|(name, _)| name.kind).collect()),
            fields
                .iter()
                .map(|(_, ty)| analyze_type(env, ty))
                .collect::<Option<Vec<Type>>>()?,
        )),
        AstType::App(name, types) => {
            let tycon = match env.find_type(name.kind) {
                Some(ScopeType::Def(def)) => TypeCon::Named(def.path()),
                Some(ScopeType::Var(var)) => {
                    error!(&name.span, "cannot instantiate type variable `{}`", var);
                    return None;
                }
                None => {
                    error!(&name.span, "type `{}` not found", IdMap::name(name.kind));
                    return None;
                }
            };

            Some(Type::App(
                tycon,
                types
                    .iter()
                    .map(|ty| analyze_type(env, ty))
                    .collect::<Option<Vec<Type>>>()?,
            ))
        }
        AstType::Arrow(arg, ret) => {
            let arg = analyze_type(env, arg);
            let ret = analyze_type(env, ret);
            try_some!(arg, ret);
            Some(Type::App(TypeCon::Arrow, vec![arg, ret]))
        }
    }
}

fn analyze_func_type(
    env: &mut Environment,
    module_path: &SymbolPath,
    func: &UntypedAstFunction,
) -> Option<Entry> {
    pushed_scope!(env, {
        // Define type parameters
        let mut ty_params = Vec::with_capacity(func.ty_params.len());
        for param in &func.ty_params {
            let var = TypeVar::with_id(param.kind);
            env.define_type(param.kind, ScopeType::Var(var));
            ty_params.push((param.kind, var));
        }

        // Analyze parameter types
        let params: Option<Vec<Type>> = func
            .params
            .iter()
            .map(|param| analyze_type(env, &param.ty))
            .collect();

        let params = match params {
            Some(params) => params,
            None => return None,
        };

        // Analyze return type
        let return_ty = match analyze_type(env, &func.return_ty) {
            Some(return_ty) => return_ty,
            None => return None,
        };

        Some(Entry::Function(Function {
            name: func.name.kind,
            params,
            return_ty,
            ty_params,
            module: module_path.clone(),
        }))
    })
}

fn analyze_typedef(block: &UntypedBlock, env: &mut Environment) {
    for tydef in &block.types {
        let mut param_vars = Vec::with_capacity(tydef.var_ids.len());
        let ty = pushed_scope!(env, {
            for var in &tydef.var_ids {
                param_vars.push(TypeVar::with_id(var.kind));
                env.define_type(var.kind, ScopeType::Var(*param_vars.last().unwrap()));
            }
            analyze_type(env, &tydef.ty)
        });

        if let Some(ty) = ty {
            let tydef = match env.find_type_mut(tydef.name) {
                Some(ScopeType::Def(tydef)) => tydef,
                _ => panic!(),
            };
            tydef.body = Some(TypeCon::Fun(param_vars, box ty));
        }
    }
}

fn generate_function_headers(
    block: &UntypedBlock,
    module_path: &SymbolPath,
    env: &mut Environment,
) {
    for func in &block.functions {
        if let Some(entry) = analyze_func_type(env, module_path, func) {
            env.define_entry(func.name.kind, entry);
        }
    }
}

fn generate_impl_headers(impls: &[UntypedImpl], module_path: &SymbolPath, env: &mut Environment) {
    for imp in impls {
        // Check target existence
        if env.find_type(imp.target.kind).is_none() {
            error!(
                &imp.target.span,
                "type `{}` is undefined",
                IdMap::name(imp.target.kind)
            );
        }

        for func in &imp.functions {
            if let Some(Entry::Function(func)) = analyze_func_type(env, module_path, func) {
                env.define_impl_func(imp.target.kind, func);
            }
        }
    }
}

pub fn do_semantic_analysis(
    modules: FxHashMap<SymbolPath, UntypedProgram>,
) -> Option<FxHashMap<SymbolPath, TypedProgram>> {
    // Generate environment each all modules
    let mut envs = FxHashMap::default();
    for path in modules.keys() {
        envs.insert(path.clone(), Environment::new());
    }

    // Generate type headers in all modules
    for (path, program) in &modules {
        let env = envs.get_mut(path).unwrap();
        generate_type_headers(&program.main, &program.module_path, env);
    }

    // Analyze type definitions in all modules
    for (path, program) in &modules {
        // Import only types
        let mut import_env = Environment::new();
        for stmt in &program.main.stmts {
            if let UntypedStmt::Import(range) = &stmt.kind {
                import_only_type_by_range(
                    &mut import_env,
                    &envs,
                    &stmt.span,
                    &program.module_path.parent().unwrap(),
                    range,
                );
            }
        }

        let env = envs.get_mut(path).unwrap();
        env.push_env(import_env);
        analyze_typedef(&program.main, env);
    }

    // Generate function headers in all modules
    for (path, program) in &modules {
        let env = envs.get_mut(path).unwrap();
        generate_function_headers(&program.main, &program.module_path, env);
    }

    // Generate implement headers in all modules
    for (path, program) in &modules {
        let env = envs.get_mut(path).unwrap();
        generate_impl_headers(&program.impls, &program.module_path, env);
    }

    let mut programs = FxHashMap::default();
    for (path, program) in modules {
        let analyzer = Analyzer::new(&envs, path.clone());
        if let Some(program) = analyzer.analyze(program) {
            programs.insert(path.clone(), program);
        }
    }

    if programs.len() == envs.len() {
        Some(programs)
    } else {
        None
    }
}
