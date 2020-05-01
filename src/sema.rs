use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::{Param as AstParam, *};
use crate::error::{Error, ErrorList};
use crate::id::{reserved_id, Id, IdMap};
use crate::ir::{
    CodeBuf, Expr as IRExpr, Function as IRFunction, Module as IRModule, Stmt as IRStmt,
};
use crate::module::{FunctionHeader, Implementation, Module, ModuleContainer, ModuleHeader};
use crate::span::{Span, Spanned};
use crate::translate;
use crate::translate::RelativeVariableLoc;
use crate::ty::*;
use crate::utils::HashMapWithScope;

macro_rules! try_some {
    ($($var:ident),*) => {
        $(let $var = $var?;)*
    };
}

macro_rules! fn_to_expect {
    ($fn_name:ident, $type_name:tt, $ty:ty, $pat:pat => $expr:expr,) => {
        fn $fn_name<'a>(ty: &'a Type, span: &Span) -> Option<&'a $ty> {
            match ty {
                $pat => $expr,
                _ => {
                    let msg = format!(
                        concat!("expected type `", $type_name, "` but got type `{}`"),
                        ty
                    );
                    ErrorList::push(Error::new(&msg, &span));
                    None
                }
            }
        }
    };
}

fn_to_expect! {
    expect_tuple, "tuple", Vec<Type>,
    Type::App(TypeCon::Tuple, types) => Some(types),
}

#[derive(Debug)]
pub enum ImportedEntry {
    Function(Id, FunctionHeaderWithId),
    Type(Id, TypeCon),
}

#[derive(Debug)]
pub struct ExprInfo {
    pub ty: Type,
    pub span: Span,
    pub ir: IRExpr,
    pub is_lvalue: bool,
    pub is_mutable: bool,
}

impl ExprInfo {
    fn new(ir: IRExpr, ty: Type, span: Span) -> Self {
        Self {
            ty,
            span,
            ir,
            is_lvalue: false,
            is_mutable: false,
        }
    }

    fn new_lvalue(ir: IRExpr, ty: Type, span: Span, is_mutable: bool) -> Self {
        Self {
            ty,
            span,
            ir,
            is_lvalue: true,
            is_mutable,
        }
    }
}

#[derive(Debug)]
struct Param {
    name: Id,
    ty: Type,
    is_mutable: bool,
    is_escaped: bool,
}

#[derive(Debug, Clone)]
pub enum VariableLoc {
    Stack(isize),
    StackInHeap(usize, usize),
}

#[derive(Debug)]
struct Variable {
    ty: Type,
    is_mutable: bool,
    loc: VariableLoc,
}

impl Variable {
    fn new(ty: Type, is_mutable: bool, loc: VariableLoc) -> Self {
        Self {
            ty,
            is_mutable,
            loc,
        }
    }
}

#[derive(Debug)]
enum Entry {
    Variable(Variable),
    Function(FunctionHeaderWithId),
}

#[derive(Debug, Clone)]
pub struct FunctionHeaderWithId {
    module_id: Option<usize>,
    original_name: Option<Id>,
    func_id: usize,
    header: FunctionHeader,
}

impl FunctionHeaderWithId {
    fn new(module_id: usize, func_id: usize, header: FunctionHeader) -> Self {
        Self {
            module_id: Some(module_id),
            original_name: None,
            func_id,
            header,
        }
    }

    fn new_renamed(
        module_id: usize,
        func_id: usize,
        header: FunctionHeader,
        original_name: Id,
    ) -> Self {
        Self {
            module_id: Some(module_id),
            original_name: Some(original_name),
            func_id,
            header,
        }
    }

    fn new_self(header: FunctionHeader, func_id: usize) -> Self {
        Self {
            module_id: None,
            original_name: None,
            func_id,
            header,
        }
    }
}

#[derive(Debug)]
pub struct Analyzer<'a> {
    visible_modules: FxHashSet<SymbolPath>,
    module_headers: FxHashMap<SymbolPath, (usize, ModuleHeader)>,
    next_module_id: usize,

    types: HashMapWithScope<Id, Type>,
    tycons: TypeDefinitions,
    tycon_spans: HashMapWithScope<Id, Span>,
    impls: FxHashMap<Id, Implementation>,

    variables: HashMapWithScope<Id, Entry>,
    next_temp_num: u32,
    var_level: usize,

    current_func: Id,
    current_func_index: usize,
    ir_funcs: Vec<(Id, IRFunction)>,

    _phantom: &'a std::marker::PhantomData<Self>,
}

impl<'a> Analyzer<'a> {
    pub fn new() -> Self {
        let mut slf = Self {
            visible_modules: FxHashSet::default(),
            module_headers: FxHashMap::default(),
            next_module_id: 0,
            variables: HashMapWithScope::new(),
            types: HashMapWithScope::new(),
            tycons: TypeDefinitions::new(),
            tycon_spans: HashMapWithScope::new(),
            impls: FxHashMap::default(),
            current_func_index: 0,
            current_func: *reserved_id::MAIN_FUNC,
            ir_funcs: Vec::new(),
            next_temp_num: 0,
            var_level: 0,
            _phantom: &std::marker::PhantomData,
        };
        slf.push_type_scope();
        slf.push_scope();

        slf
    }

    #[inline]
    fn push_scope(&mut self) {
        self.variables.push_scope();
    }

    #[inline]
    fn pop_scope(&mut self) {
        self.variables.pop_scope();
    }

    #[inline]
    fn push_type_scope(&mut self) {
        self.types.push_scope();
        self.tycons.push_scope();
        self.tycon_spans.push_scope();
    }

    #[inline]
    fn pop_type_scope(&mut self) {
        self.tycons.pop_scope();
        self.types.pop_scope();
        self.tycon_spans.pop_scope();
    }

    // =====================================
    // Type
    // =====================================

    fn expand_name(&self, ty: Type) -> Option<Type> {
        match ty {
            Type::App(TypeCon::Fun(params, body), types) => {
                let map = params.into_iter().zip(types.into_iter()).collect();
                self.expand_name(subst(*body, &map))
            }
            Type::App(TypeCon::Named(name, _), types)
            | Type::App(TypeCon::UnsizedNamed(name), types) => match self.tycons.get(name) {
                Some(TypeBody::Resolved(tycon)) | Some(TypeBody::Unresolved(tycon)) => {
                    Some(Type::App(tycon.clone(), types))
                }
                None => None,
            },
            Type::App(tycon, types) => {
                let mut new_types = Vec::with_capacity(types.len());
                for ty in types {
                    let ty = self.expand_name(ty)?;
                    new_types.push(ty);
                }

                Some(Type::App(tycon, new_types))
            }
            ty => Some(ty),
        }
    }

    #[inline]
    fn type_size_err(span: Span, ty: &Type) -> usize {
        match type_size(ty) {
            Some(size) => size,
            None => {
                error!(&span, "the size of type `{}` cannot be calculated", ty);
                0
            }
        }
    }

    fn ir_func(&mut self, index: usize) -> &mut IRFunction {
        &mut self.ir_funcs[index].1
    }

    // ====================================
    //  Variable
    // ====================================

    fn new_var(
        &mut self,
        current_func: usize,
        id: Id,
        ty: Type,
        is_mutable: bool,
        is_escaped: bool,
    ) -> VariableLoc {
        let new_var_size = type_size_nocheck(&ty);
        let var_level = self.var_level;
        let current_func = self.ir_func(current_func);

        let loc = if is_escaped {
            let loc = VariableLoc::StackInHeap(current_func.stack_in_heap_size + 1, var_level);
            current_func.stack_in_heap_size += new_var_size;
            loc
        } else {
            let loc = VariableLoc::Stack(current_func.stack_size as isize);
            current_func.stack_size += new_var_size;
            loc
        };

        self.variables.insert(
            id,
            Entry::Variable(Variable::new(ty, is_mutable, loc.clone())),
        );

        loc
    }

    #[inline]
    fn new_var_in_current_func(
        &mut self,
        id: Id,
        ty: Type,
        is_mutable: bool,
        is_escaped: bool,
    ) -> VariableLoc {
        self.new_var(self.current_func_index, id, ty, is_mutable, is_escaped)
    }

    fn gen_temp_id(&mut self) -> Id {
        let id = IdMap::new_id(&format!("$comp{}", self.next_temp_num));
        self.next_temp_num += 1;
        id
    }

    // Convert from VariableLoc to RelativeVariableLoc
    fn relative_loc(&self, loc: &VariableLoc) -> RelativeVariableLoc {
        match loc {
            VariableLoc::Stack(loc) => RelativeVariableLoc::Stack(*loc),
            VariableLoc::StackInHeap(loc, level) => {
                RelativeVariableLoc::StackInHeap(*loc, self.var_level - *level)
            }
        }
    }

    fn find_var(&self, id: Id) -> Option<&Entry> {
        self.variables.get(&id)
    }

    fn find_external_entry(&self, path: &SymbolPath) -> Option<ImportedEntry> {
        // header, code id, module id
        assert!(!path.is_empty());

        // m1::m2::Type::func
        // m1::func

        let module_path = path.parent().unwrap();
        let entry_name = path.tail().unwrap().id;

        if module_path.is_empty() {
            return None;
        }

        match self.module_headers.get(&module_path) {
            // Find a function or a type of `entry_id`
            Some((module_id, module)) => {
                // Not imported modules are invisible
                if !self.visible_modules.contains(&module_path) {
                    return None;
                }

                if let Some(tycon) = module.types.get(&entry_name) {
                    let tycon = tycon.as_ref().unwrap();
                    return Some(ImportedEntry::Type(entry_name, tycon.clone()));
                }

                let (func_id, header) = module.functions.get(&entry_name)?;
                let header = FunctionHeaderWithId::new(*module_id, *func_id, header.clone());

                Some(ImportedEntry::Function(entry_name, header))
            }
            // `path` may be a method
            None => {
                let type_name = module_path.tail()?.id;
                let module_path = module_path.parent()?;
                let func_name = entry_name;

                if module_path.is_empty() {
                    let (func_id, header) =
                        self.impls.get(&type_name)?.functions.get(&func_name)?;
                    let header = FunctionHeaderWithId::new_self(header.clone(), *func_id);

                    Some(ImportedEntry::Function(func_name, header))
                } else {
                    // Not imported modules are invisible
                    if !self.visible_modules.contains(&module_path) {
                        return None;
                    }

                    let (module_id, module) = self.module_headers.get(&module_path)?;
                    let (func_id, header) =
                        module.impls.get(&type_name)?.functions.get(&func_name)?;
                    let header = FunctionHeaderWithId::new(*module_id, *func_id, header.clone());

                    Some(ImportedEntry::Function(func_name, header))
                }
            }
        }
    }

    // Pointer
    //   Struct => ok
    //   _ => error
    // Struct => ok
    // _ => error
    fn get_struct_fields<'b>(
        &'b mut self,
        ty: &'b Type,
        span: &Span,
        is_mutable: bool,
    ) -> Option<(Vec<(Id, Type)>, bool)> {
        let ty = expand_unique(ty.clone());
        match ty {
            Type::App(TypeCon::Struct(fields), tys) => Some((
                fields.into_iter().zip(tys.into_iter()).collect(),
                is_mutable,
            )),
            Type::App(TypeCon::Pointer(is_mutable), tys) => {
                let ty = expand_unique(tys[0].clone());
                match ty {
                    Type::App(TypeCon::Struct(fields), tys) => Some((
                        fields.into_iter().zip(tys.into_iter()).collect(),
                        is_mutable,
                    )),
                    ty => {
                        error!(
                            &span.clone(),
                            "expected type `struct` or `*struct` but got type `{}`", ty
                        );
                        None
                    }
                }
            }
            ty => {
                error!(
                    &span.clone(),
                    "expected type `struct` or `*struct` but got type `{}`", ty
                );
                None
            }
        }
    }

    // ====================================
    //  Expression
    // ====================================

    // Return true if `walk_expr` passed `expr` may push multiple values
    fn expr_push_multiple_values(expr: &Expr) -> bool {
        match expr {
            // always
            Expr::Tuple(_) | Expr::Struct(_, _) | Expr::Array(_, _) => true,
            // only if the return value is a compound value
            Expr::Call(..) => true,
            _ => false,
        }
    }

    fn insert_type_param(param_ty: &Type, ty: &Type, map: &mut FxHashMap<TypeVar, Type>) {
        match (param_ty, ty) {
            (Type::App(ptycon, ptypes), Type::App(atycon, atypes)) if ptycon == atycon => {
                for (pty, aty) in ptypes.iter().zip(atypes.iter()) {
                    Self::insert_type_param(pty, aty, map);
                }
            }
            (Type::Var(var), ty) if !map.contains_key(var) => {
                map.insert(*var, ty.clone());
            }
            _ => {}
        }
    }

    // Walk an expression and convert it to a specified type if possible
    fn walk_expr_with_conversion(
        &mut self,
        expr: Spanned<Expr>,
        dest_ty: &Type,
    ) -> Option<ExprInfo> {
        match dest_ty {
            Type::App(TypeCon::Wrapped, _) => {
                let expr = self.walk_expr(expr)?;

                // Wrap if necessary
                if let Type::App(TypeCon::Wrapped, _) = &expr.ty {
                    // Don't need wrap again
                    Some(expr)
                } else {
                    let mut expr = expr;
                    expr.ir = translate::wrap(expr.ir, &expr.ty);
                    expr.ty = Type::App(TypeCon::Wrapped, vec![expr.ty]);

                    Some(expr)
                }
            }
            Type::App(TypeCon::Pointer(false), _) => {
                let mut expr = self.walk_expr(expr)?;

                // Weaken a pointer if necessary
                if let Type::App(TypeCon::Pointer(true), types) = expr.ty {
                    expr.ty = Type::App(TypeCon::Pointer(false), types);
                }

                Some(expr)
            }
            _ => match expr.kind {
                Expr::Tuple(exprs) => {
                    // Extract inner types
                    let dest_types = match dest_ty {
                        Type::App(TypeCon::Tuple, types) => types,
                        _ => {
                            // If `dest_ty` is not a tuple, walk the expression without conversion
                            let mut tys = Vec::new();

                            for expr in exprs.into_iter() {
                                let expr = self.walk_expr(expr);
                                if let Some(expr) = expr {
                                    tys.push(expr.ty.clone());
                                }
                            }

                            return Some(ExprInfo::new(
                                IRExpr::Unit,
                                Type::App(TypeCon::Tuple, tys),
                                expr.span,
                            ));
                        }
                    };

                    // Check size of the tuples
                    if dest_types.len() != exprs.len() {
                        error!(&
                            expr.span,
                            "expected that tuple size will be `{}` but the number of expressions `{}`",
                            dest_types.len(),
                            exprs.len()
                        );
                        return None;
                    }

                    // Walk each expression
                    let mut ir_exprs = Vec::with_capacity(exprs.len());
                    let mut tys = Vec::new();

                    for (expr, dest_ty) in exprs.into_iter().zip(dest_types.iter()) {
                        let expr = self.walk_expr_with_conversion(expr, dest_ty);
                        if let Some(expr) = expr {
                            tys.push(expr.ty.clone());
                            ir_exprs.push(translate::literal_tuple(expr));
                        }
                    }

                    Some(ExprInfo::new(
                        IRExpr::Record(ir_exprs),
                        Type::App(TypeCon::Tuple, tys),
                        expr.span,
                    ))
                }
                _ => {
                    let mut expr = self.walk_expr(expr)?;

                    // Convert to InHeap if necessary
                    if !dest_ty.is_in_heap() {
                        if let Type::App(TypeCon::InHeap, types) = &expr.ty {
                            expr.ir = translate::copy_in_heap(expr.ir, &expr.ty, &types[0]);
                            expr.ty = types[0].clone();
                        }
                    }

                    // Unwrap if necessary
                    if !dest_ty.is_wrapped() {
                        if let Type::App(TypeCon::Wrapped, types) = &expr.ty {
                            expr.ir = translate::unwrap(expr.ir, &expr.ty);
                            expr.ty = types[0].clone();
                        }
                    }

                    Some(expr)
                }
            },
        }
    }

    // Unwrap and dereference an expression if possible
    fn walk_expr_with_unwrap_and_deref(&mut self, expr: Spanned<Expr>) -> Option<ExprInfo> {
        let mut expr = self.walk_expr(expr)?;

        if let Type::App(TypeCon::InHeap, types) = &expr.ty {
            expr.ir = translate::copy_in_heap(expr.ir, &expr.ty, &types[0]);
            expr.ty = types[0].clone();
        }

        if let Type::App(TypeCon::Wrapped, types) = &expr.ty {
            expr.ir = translate::unwrap(expr.ir, &expr.ty);
            expr.ty = types[0].clone();
        }

        Some(expr)
    }

    // Unwrap an expression if possible
    fn walk_expr_with_unwrap(&mut self, expr: Spanned<Expr>) -> Option<ExprInfo> {
        let mut expr = self.walk_expr(expr)?;

        if let Type::App(TypeCon::Wrapped, types) = &expr.ty {
            expr.ir = translate::unwrap(expr.ir, &expr.ty);
            expr.ty = types[0].clone();
        }

        Some(expr)
    }

    #[allow(clippy::cognitive_complexity)]
    fn walk_expr(&mut self, expr: Spanned<Expr>) -> Option<ExprInfo> {
        let (ir, ty) = match expr.kind {
            Expr::Literal(Literal::Number(n)) => (translate::literal_int(n), Type::Int),
            Expr::Literal(Literal::String(s)) => {
                let ty = Type::App(TypeCon::Pointer(false), vec![Type::String]);
                (translate::literal_str(s), ty)
            }
            Expr::Literal(Literal::Unit) => (IRExpr::Unit, Type::Unit),
            Expr::Literal(Literal::True) => (translate::literal_true(), Type::Bool),
            Expr::Literal(Literal::False) => (translate::literal_false(), Type::Bool),
            Expr::Literal(Literal::Null) => (translate::literal_null(), Type::Null),
            Expr::Tuple(exprs) => {
                let mut ir_exprs = Vec::with_capacity(exprs.len());
                let mut types = Vec::new();
                for expr in exprs {
                    let expr = self.walk_expr(expr);
                    if let Some(expr) = expr {
                        types.push(expr.ty.clone());
                        ir_exprs.push(translate::literal_tuple(expr));
                    }
                }

                (IRExpr::Record(ir_exprs), Type::App(TypeCon::Tuple, types))
            }
            Expr::Struct(ty, field_exprs) => {
                // The parser only allow a named type for `ty`
                // The type that expended it is like App(Unique(Fun(params, STRUCT), 123), [TYARGS])

                let ty = self.walk_type(ty)?;
                // Clone the not expanded type for returning ExprInfo
                let mut expr_ty = ty.clone();

                let ty = self.expand_name(ty)?;

                // Get type parameters
                let ty_params = match ty.clone() {
                    Type::App(TypeCon::Unique(box TypeCon::Fun(params, _), _), _) => params,
                    _ => panic!(),
                };

                let ty = expand_unique(ty);
                let ty = expand_wrap(ty);
                let ty = subst(ty, &FxHashMap::default());

                // Convert struct fields to a vector of tuples of field name and type
                let mut fields: Vec<(Id, Type)> = match ty {
                    Type::App(TypeCon::Struct(fields), types) => {
                        fields.into_iter().zip(types.into_iter()).collect()
                    }
                    ty => {
                        error!(&expr.span.clone(), "not struct type `{}`", ty);
                        return None;
                    }
                };

                let mut ir_exprs = Vec::with_capacity(fields.len());
                let mut map = FxHashMap::default();

                // Walk expressions in field declaration order
                for i in 0..fields.len() {
                    let field_expr = field_exprs
                        .iter()
                        .find(|(id, _)| id.kind == fields[i].0)
                        .map(|(_, expr)| expr);

                    if let Some(expr) = field_expr {
                        if let Some(expr) =
                            self.walk_expr_with_conversion(expr.clone(), &fields[i].1)
                        {
                            Self::insert_type_param(&fields[i].1, &expr.ty, &mut map);
                            for (_, ty) in &mut fields {
                                *ty = subst(ty.clone(), &map);
                            }

                            unify(&expr.span, &fields[i].1, &expr.ty);
                            ir_exprs.push(translate::literal_struct_field(expr));
                        }
                    } else {
                        error!(
                            &expr.span.clone(),
                            "missing field `{}`",
                            IdMap::name(fields[i].0)
                        );
                    }
                }

                // Substutute type arguments and types inferred for `expr_ty`
                let args: Vec<Type> = ty_params
                    .into_iter()
                    .map(|var| map.get(&var).cloned().unwrap_or(Type::Var(var)))
                    .collect();
                match &mut expr_ty {
                    Type::App(_, types) => {
                        assert!(types.len() <= args.len());

                        if types.len() < args.len() {
                            for arg in args.into_iter().skip(types.len()) {
                                types.push(arg);
                            }
                        }
                    }
                    _ => panic!(),
                }

                (IRExpr::Record(ir_exprs), expr_ty)
            }
            Expr::Array(expr, size) => {
                let expr = self.walk_expr(*expr)?;
                let ty = expr.ty.clone();

                (
                    translate::literal_array(expr, size),
                    Type::App(TypeCon::Array(size), vec![ty]),
                )
            }
            Expr::Field(comp_expr, field) => {
                let should_store = Self::expr_push_multiple_values(&comp_expr.kind);
                let comp_expr = self.walk_expr(*comp_expr)?;
                let mut is_mutable = comp_expr.is_mutable;

                let loc = if should_store {
                    let id = self.gen_temp_id();
                    Some(self.new_var_in_current_func(id, comp_expr.ty.clone(), false, false))
                } else {
                    None
                };

                let ty = self.expand_name(comp_expr.ty.clone())?;
                let mut ty = expand_wrap(ty);

                if let Type::App(TypeCon::InHeap, types) = ty {
                    ty = types[0].clone();
                }

                // Get the field type and offset
                let (field_ty, offset) = match field {
                    Field::Number(i) => {
                        let types = match &ty {
                            Type::App(TypeCon::Pointer(is_mutable_), tys) => {
                                is_mutable = *is_mutable_;
                                expect_tuple(&tys[0], &comp_expr.span.clone())?
                            }
                            ty => expect_tuple(ty, &comp_expr.span.clone())?,
                        };

                        match types.get(i) {
                            Some(ty) => {
                                let offset = types
                                    .iter()
                                    .take(i)
                                    .fold(0, |acc, ty| acc + type_size_nocheck(ty));
                                (ty.clone(), offset)
                            }
                            None => {
                                error!(&expr.span, "error");
                                return None;
                            }
                        }
                    }
                    Field::Id(name) => {
                        let (fields, is_mutable_) =
                            self.get_struct_fields(&ty, &comp_expr.span, is_mutable)?;
                        is_mutable = is_mutable_;

                        let i = match fields.iter().position(|(id, _)| *id == name) {
                            Some(i) => i,
                            None => {
                                error!(
                                    &expr.span,
                                    "no field in `{}`: `{}`",
                                    comp_expr.ty,
                                    IdMap::name(name)
                                );
                                return None;
                            }
                        };

                        let offset = fields
                            .iter()
                            .take(i)
                            .fold(0, |acc, (_, ty)| acc + type_size_nocheck(ty));
                        (fields[i].1.clone(), offset)
                    }
                };

                let ir =
                    translate::field(loc.map(|loc| self.relative_loc(&loc)), comp_expr, offset);
                return Some(ExprInfo::new_lvalue(ir, field_ty, expr.span, is_mutable));
            }
            Expr::Subscript(expr, subscript_expr) => {
                let should_store = Self::expr_push_multiple_values(&expr.kind);

                let expr = self.walk_expr_with_unwrap(*expr);
                let subscript_expr = self.walk_expr_with_conversion(*subscript_expr, &Type::Int);
                try_some!(expr, subscript_expr);

                let mut expr = expr;

                let loc = if should_store {
                    let id = self.gen_temp_id();
                    Some(self.new_var_in_current_func(id, expr.ty.clone(), false, false))
                } else {
                    None
                };

                let (ty, is_in_heap) = match &expr.ty {
                    Type::App(TypeCon::InHeap, types) => (types[0].clone(), true),
                    _ => (expr.ty.clone(), false),
                };

                let (ty, is_pointer) = match ty {
                    Type::App(TypeCon::Array(_), tys) => (tys[0].clone(), false),
                    Type::App(TypeCon::Pointer(is_mutable), tys) => {
                        expr.is_mutable = is_mutable;

                        match &tys[0] {
                            Type::App(TypeCon::Array(_), tys) => (tys[0].clone(), true),
                            ty => {
                                error!(&expr.span.clone(), "expected array but got type `{}`", ty);
                                return None;
                            }
                        }
                    }
                    ty => {
                        error!(&expr.span.clone(), "expected array but got type `{}`", ty);
                        return None;
                    }
                };

                unify(&subscript_expr.span, &subscript_expr.ty, &Type::Int);

                let span = expr.span.clone();
                let is_lvalue = expr.is_lvalue;
                let is_mutable = expr.is_mutable;
                let ir = translate::subscript(
                    loc.map(|loc| self.relative_loc(&loc)),
                    is_in_heap,
                    is_pointer,
                    expr,
                    subscript_expr,
                    &ty,
                );

                return Some(ExprInfo {
                    ty,
                    ir,
                    span,
                    is_lvalue,
                    is_mutable,
                });
            }
            Expr::Variable(name, _) => {
                let entry = match self.find_var(name) {
                    Some(v) => v,
                    None => {
                        error!(&expr.span.clone(), "undefined variable or function");
                        return None;
                    }
                };

                let (insts, ty, is_mutable) = match entry {
                    Entry::Variable(var) => (
                        translate::variable(&self.relative_loc(&var.loc)),
                        var.ty.clone(),
                        var.is_mutable,
                    ),
                    Entry::Function(fh) => {
                        let ir = translate::func_pos(fh.module_id, fh.func_id);
                        let ty = generate_func_type(
                            &fh.header.params,
                            &fh.header.return_ty,
                            &fh.header.ty_params,
                        );
                        (ir, ty, false)
                    }
                };
                return Some(ExprInfo::new_lvalue(insts, ty, expr.span, is_mutable));
            }
            Expr::Path(path) => {
                let FunctionHeaderWithId {
                    module_id,
                    func_id,
                    header,
                    ..
                } = match self.find_external_entry(&path) {
                    Some(ImportedEntry::Function(_, h)) => h,
                    Some(ImportedEntry::Type(_, _)) => {
                        error!(&expr.span, "`{}` is a type", path);
                        return None;
                    }
                    None => {
                        error!(&expr.span, "undefined function `{}`", path);
                        return None;
                    }
                };

                let mut ty = header.return_ty.clone();
                for param_ty in &header.params {
                    ty = Type::App(TypeCon::Arrow, vec![param_ty.clone(), ty]);
                }

                let ir = translate::func_pos(module_id, func_id);
                (ir, ty)
            }
            Expr::BinOp(BinOp::And, lhs, rhs) => {
                let lhs = self.walk_expr_with_conversion(*lhs, &Type::Int);
                let rhs = self.walk_expr_with_conversion(*rhs, &Type::Int);
                try_some!(lhs, rhs);

                // Type check
                match (&lhs.ty, &rhs.ty) {
                    (Type::Bool, Type::Bool) => {}
                    (lty, rty) => {
                        error!(&expr.span.clone(), "{} && {}", lty, rty);
                    }
                }

                (translate::binop_and(lhs, rhs), Type::Bool)
            }
            Expr::BinOp(BinOp::Or, lhs, rhs) => {
                let lhs = self.walk_expr_with_conversion(*lhs, &Type::Int);
                let rhs = self.walk_expr_with_conversion(*rhs, &Type::Int);
                try_some!(lhs, rhs);

                // Type check
                match (&lhs.ty, &rhs.ty) {
                    (Type::Bool, Type::Bool) => {}
                    (lty, rty) => {
                        error!(&expr.span.clone(), "{} || {}", lty, rty);
                    }
                }

                (translate::binop_or(lhs, rhs), Type::Bool)
            }
            Expr::BinOp(binop, lhs, rhs) => {
                let lhs = self.walk_expr_with_unwrap_and_deref(*lhs);
                let rhs = self.walk_expr_with_unwrap_and_deref(*rhs);
                try_some!(lhs, rhs);

                let binop_symbol = binop.to_symbol();
                let ty = match (&binop, &lhs.ty, &rhs.ty) {
                    (BinOp::Add, Type::Int, Type::Int) => Type::Int,
                    (BinOp::Sub, Type::Int, Type::Int) => Type::Int,
                    (BinOp::Mul, Type::Int, Type::Int) => Type::Int,
                    (BinOp::Div, Type::Int, Type::Int) => Type::Int,
                    (BinOp::Equal, Type::Int, Type::Int) => Type::Bool,
                    (BinOp::NotEqual, Type::Int, Type::Int) => Type::Bool,
                    (BinOp::LessThan, Type::Int, Type::Int) => Type::Bool,
                    (BinOp::LessThanOrEqual, Type::Int, Type::Int) => Type::Bool,
                    (BinOp::GreaterThan, Type::Int, Type::Int) => Type::Bool,
                    (BinOp::GreaterThanOrEqual, Type::Int, Type::Int) => Type::Bool,

                    (
                        BinOp::Equal,
                        Type::App(TypeCon::Pointer(_), _),
                        Type::App(TypeCon::Pointer(_), _),
                    ) => Type::Bool,
                    (BinOp::Equal, Type::Null, Type::App(TypeCon::Pointer(_), _)) => Type::Bool,
                    (BinOp::Equal, Type::App(TypeCon::Pointer(_), _), Type::Null) => Type::Bool,
                    (
                        BinOp::NotEqual,
                        Type::App(TypeCon::Pointer(_), _),
                        Type::App(TypeCon::Pointer(_), _),
                    ) => Type::Bool,
                    (BinOp::NotEqual, Type::Null, Type::App(TypeCon::Pointer(_), _)) => Type::Bool,
                    (BinOp::NotEqual, Type::App(TypeCon::Pointer(_), _), Type::Null) => Type::Bool,
                    _ => {
                        error!(
                            &expr.span.clone(),
                            "`{} {} {}`", lhs.ty, binop_symbol, rhs.ty
                        );
                        return None;
                    }
                };

                (translate::binop(binop, lhs, rhs), ty)
            }
            Expr::Call(func_expr, arg_expr) => {
                let func_expr = self.walk_expr(*func_expr)?;

                // Get the argument type and return type
                let (mut arg_ty, mut return_ty) = match &func_expr.ty {
                    Type::Poly(_, box Type::App(TypeCon::Arrow, types)) => {
                        (types[0].clone(), types[1].clone())
                    }
                    Type::App(TypeCon::Arrow, types) => (types[0].clone(), types[1].clone()),
                    ty => {
                        error!(&func_expr.span, "expected function but got type `{}`", ty);
                        return None;
                    }
                };

                let arg_expr = self.walk_expr_with_conversion(*arg_expr, &arg_ty)?;

                // Type interface
                let mut map = FxHashMap::default();
                Self::insert_type_param(&arg_ty, &arg_expr.ty, &mut map);
                arg_ty = subst(arg_ty, &map);
                return_ty = subst(return_ty, &map);

                unify(&arg_expr.span, &arg_ty, &arg_expr.ty);

                let ir = translate::call(&return_ty, func_expr, arg_expr);
                (ir, return_ty)
            }
            Expr::Address(expr, is_mutable) => {
                let expr = self.walk_expr_with_unwrap(*expr)?;

                let inner_type = expand_inheap(expr.ty.clone());
                let ty = Type::App(TypeCon::Pointer(is_mutable), vec![inner_type]);

                if !expr.is_lvalue {
                    // Store `expr` and return the pointer to it if `expr` is not lvalue
                    let temp = self.gen_temp_id();
                    let loc = self.new_var_in_current_func(
                        temp,
                        Type::App(TypeCon::InHeap, vec![expr.ty.clone()]),
                        is_mutable,
                        false,
                    );

                    let ir = translate::address_no_lvalue(expr, &self.relative_loc(&loc));
                    (ir, ty)
                } else {
                    if is_mutable && !expr.is_mutable {
                        error!(&expr.span, "this expression is immutable");
                        return None;
                    }

                    let ir = translate::address(expr);
                    (ir, ty)
                }
            }
            Expr::Dereference(expr_) => {
                let expr_ = self.walk_expr_with_unwrap(*expr_)?;

                match expr_.ty.clone() {
                    Type::App(TypeCon::Pointer(is_mutable), tys) => {
                        let ir = translate::dereference(expr_);
                        return Some(ExprInfo::new_lvalue(
                            ir,
                            tys[0].clone(),
                            expr.span,
                            is_mutable,
                        ));
                    }
                    ty => {
                        error!(&expr.span, "expected type `pointer` but got type `{}`", ty);
                        return None;
                    }
                }
            }
            Expr::Negative(expr) => {
                let expr = self.walk_expr_with_conversion(*expr, &Type::Int)?;

                match expr.ty.clone() {
                    ty @ Type::Int /* | Type::Float */ => {
                        (translate::negative(expr), ty)
                    },
                    ty => {
                        error!(&expr.span, "expected type `int` or `float` but got type `{}`", ty);
                        return None;
                    },
                }
            }
            Expr::Block(block) => {
                self.push_scope();

                let (ty, ir) = self.walk_block(block)?;

                self.pop_scope();

                (ir, ty)
            }
            Expr::If(cond, then_expr, None) => {
                let cond = self.walk_expr_with_conversion(*cond, &Type::Bool);
                let then_expr = self.walk_expr(*then_expr);
                try_some!(cond, then_expr);

                unify(&cond.span, &Type::Bool, &cond.ty);

                (translate::if_expr(cond, then_expr), Type::Unit)
            }
            Expr::If(cond, then_expr, Some(else_expr)) => {
                let cond = self.walk_expr_with_conversion(*cond, &Type::Bool);
                let then = self.walk_expr(*then_expr);
                try_some!(cond, then);

                let els = self.walk_expr_with_conversion(*else_expr, &then.ty)?;

                unify(&cond.span, &Type::Bool, &cond.ty);
                unify(&els.span, &then.ty, &els.ty);

                let ty = then.ty.clone();
                (translate::if_else_expr(cond, then, els), ty)
            }
            Expr::App(expr, old_tyargs) => {
                let expr = self.walk_expr(*expr);

                let mut tyargs = Vec::with_capacity(old_tyargs.len());
                let mut failed = false;
                for arg in old_tyargs {
                    if let Some(arg) = self.walk_type(arg) {
                        tyargs.push(arg);
                    } else {
                        failed = true;
                    }
                }

                let mut expr = expr?;
                if failed {
                    return None;
                }

                expr.ty = match expr.ty {
                    Type::Poly(vars, ty) => {
                        if vars.len() != tyargs.len() {
                            error!(
                                &expr.span,
                                "the expression takes {} parameters but got {} arguments",
                                vars.len(),
                                tyargs.len()
                            );
                            return None;
                        }

                        let map: FxHashMap<TypeVar, Type> =
                            vars.into_iter().zip(tyargs.into_iter()).collect();
                        subst(*ty, &map)
                    }
                    ty => {
                        error!(
                            &expr.span,
                            "The expression with type `{}` cannot be apply", ty
                        );
                        return None;
                    }
                };

                return Some(expr);
            }
        };

        Some(ExprInfo::new(ir, ty, expr.span))
    }

    // =====================================
    // Statement
    // =====================================

    fn expr_has_side_effects(expr: &Expr) -> bool {
        use Expr::*;

        match expr {
            Literal(_) | Path(_) | Variable(_, _) => false,
            Call(_, _) | Block(_) => true,
            Tuple(exprs) => exprs
                .iter()
                .map(|e| &e.kind)
                .any(Self::expr_has_side_effects),
            Struct(_, fields) => fields
                .iter()
                .map(|(_, e)| &e.kind)
                .any(Self::expr_has_side_effects),
            Subscript(e1, e2) | BinOp(_, e1, e2) => {
                Self::expr_has_side_effects(&e1.kind) || Self::expr_has_side_effects(&e2.kind)
            }
            Array(expr, _)
            | Field(expr, _)
            | Dereference(expr)
            | Address(expr, _)
            | Negative(expr)
            | App(expr, _) => Self::expr_has_side_effects(&expr.kind),
            If(cond, then, els) => {
                Self::expr_has_side_effects(&cond.kind)
                    || Self::expr_has_side_effects(&then.kind)
                    || els
                        .as_ref()
                        .map(|e| Self::expr_has_side_effects(&e.kind))
                        .unwrap_or(false)
            }
        }
    }

    fn walk_stmt(&mut self, stmt: Spanned<Stmt>) -> Option<CodeBuf> {
        let stmts = match stmt.kind {
            Stmt::Expr(expr) => {
                // An expression that doesn't have side effects is unnecessary
                if !Self::expr_has_side_effects(&expr.kind) {
                    warn!(&expr.span, "Unnecessary expression");
                }

                let expr = self.walk_expr(expr)?;
                translate::expr_stmt(expr)
            }
            Stmt::While(cond, stmt) => {
                let cond = self.walk_expr_with_conversion(cond, &Type::Bool);
                let body = self.walk_stmt(*stmt);
                try_some!(cond, body);

                unify(&cond.span, &Type::Bool, &cond.ty);

                translate::while_stmt(cond, body)
            }
            Stmt::Bind(name, ty, expr, is_mutable, is_escaped, is_in_heap) => {
                let mut expr = match ty {
                    Some(ty) => {
                        let ty = self.walk_type(ty)?;
                        let mut expr = self.walk_expr_with_conversion(*expr, &ty)?;

                        unify(&expr.span, &ty, &expr.ty)?;
                        expr.ty = ty;

                        expr
                    }
                    None => self.walk_expr(*expr)?,
                };

                if is_in_heap {
                    expr.ty = Type::App(TypeCon::InHeap, vec![expr.ty]);
                }

                let loc =
                    self.new_var_in_current_func(name, expr.ty.clone(), is_mutable, is_escaped);

                translate::bind_stmt(&self.relative_loc(&loc), expr)
            }
            Stmt::Assign(lhs, rhs) => {
                let mut lhs = self.walk_expr(lhs)?;
                let is_in_heap = if let Type::App(TypeCon::InHeap, types) = &mut lhs.ty {
                    lhs.ty = types[0].clone();
                    true
                } else {
                    false
                };

                let rhs = self.walk_expr_with_conversion(*rhs, &lhs.ty)?;

                if !lhs.is_lvalue {
                    error!(&lhs.span, "unassignable expression");
                    return None;
                }

                if !lhs.is_mutable {
                    error!(&lhs.span, "immutable expression");
                    return None;
                }

                unify(&rhs.span, &lhs.ty, &rhs.ty)?;

                translate::assign_stmt(lhs, rhs, is_in_heap)
            }
            Stmt::Return(expr) => {
                // Check if is outside function
                if self.current_func_index == 0 {
                    error!(&stmt.span, "return statement outside function");
                    return None;
                }

                let return_var_ty = match self.variables.get(&self.current_func).unwrap() {
                    Entry::Function(header) => header.header.return_ty.clone(),
                    _ => panic!("{} is not function", IdMap::name(self.current_func)),
                };

                let expr = match expr {
                    Some(expr) => Some(self.walk_expr_with_conversion(expr, &return_var_ty)?),
                    None => None,
                };

                // Check type
                let return_ty = expr.as_ref().map_or(&Type::Unit, |expr| &expr.ty);
                unify(&stmt.span, &return_var_ty, return_ty);

                translate::return_stmt(expr, &return_var_ty)
            }
        };

        Some(stmts)
    }

    // =====================================
    // Importing
    // =====================================

    fn generate_from_range_path<F>(
        &mut self,
        range_span: &Span,
        path: &ImportRangePath,
        mut insert: F,
    ) where
        F: FnMut(usize, Id, Id),
    {
        match path {
            ImportRangePath::All(spath) => {
                // Import all functions if module `spath` exists
                match self.module_headers.get(&spath) {
                    Some((module_id, module)) => {
                        // Insert functions
                        for func_name in module.functions.keys() {
                            insert(*module_id, *func_name, *func_name);
                        }

                        // Insert types
                        for name in module.types.keys() {
                            insert(*module_id, *name, *name);
                        }
                    }
                    _ => error!(&range_span.clone(), "undefined module `{}`", spath),
                }
            }
            ImportRangePath::Path(spath) | ImportRangePath::Renamed(spath, _)
                if self.module_headers.contains_key(spath) =>
            {
                // if path is module
                self.visible_modules.insert(path.as_path().clone());
            }
            ImportRangePath::Path(spath) | ImportRangePath::Renamed(spath, _) => {
                // If `spath` doesn't exists as a module

                // Get the module path from `spath`
                let module_path = spath.parent();
                if module_path.is_none() {
                    error!(&range_span.clone(), "undefined module `{}`", spath);
                    return;
                }

                let module_path = module_path.unwrap();
                let entry = spath.tail().unwrap();

                // Find the module
                let (module_id, _) = match self.module_headers.get(&module_path) {
                    Some(t) => t,
                    None => {
                        error!(&range_span.clone(), "undefined module `{}`", spath);
                        return;
                    }
                };

                let name_to_insert = match path {
                    ImportRangePath::Renamed(_, name) => *name,
                    _ => entry.id,
                };

                insert(*module_id, entry.id, name_to_insert);
            }
        }
    }

    fn insert_header_from_imported_entry(&mut self, entry: ImportedEntry) {
        match entry {
            ImportedEntry::Function(name, header) => {
                self.variables.insert(name, Entry::Function(header));
            }
            ImportedEntry::Type(name, tycon) => {
                self.tycons.insert(name);
                self.tycons.set_body(name, tycon);
            }
        }
    }

    fn generate_imported_entry(
        &mut self,
        module_id: usize,
        original_name: Id,
        renamed_name: Id,
        span: &Span,
    ) -> Option<ImportedEntry> {
        // Find the module by ID
        let (_, header) = self
            .module_headers
            .values()
            .find(|(id, _)| *id == module_id)
            .unwrap();

        match header.types.get(&original_name) {
            Some(Some(body)) => Some(ImportedEntry::Type(renamed_name, body.clone())),
            _ => match header.functions.get(&original_name) {
                Some((id, func)) => {
                    let h = if original_name == renamed_name {
                        FunctionHeaderWithId::new(module_id, *id, func.clone())
                    } else {
                        FunctionHeaderWithId::new_renamed(
                            module_id,
                            *id,
                            func.clone(),
                            renamed_name,
                        )
                    };
                    Some(ImportedEntry::Function(renamed_name, h))
                }
                None => {
                    error!(
                        &span.clone(),
                        "undefined function or type `{}`",
                        IdMap::name(original_name)
                    );
                    None
                }
            },
        }
    }

    fn insert_func_headers_by_range(&mut self, range: &Spanned<ImportRange>) {
        let paths = range.kind.to_paths();
        for path in paths {
            let mut list = Vec::new();
            self.generate_from_range_path(&range.span, &path, |m, o, r| {
                list.push((m, o, r));
            });

            for (m, o, r) in list {
                if let Some(entry) = self.generate_imported_entry(m, o, r, &range.span) {
                    self.insert_header_from_imported_entry(entry);
                }
            }
        }
    }

    // ===================================
    // Definition
    // ===================================

    fn insert_extern_module_headers(&mut self, imports: &[Spanned<ImportRange>]) {
        for range in imports {
            self.insert_func_headers_by_range(&range);
        }
    }

    fn insert_type_headers_in_stmts(&mut self, types: &[AstTypeDef]) {
        // Insert type headers
        for tydef in types {
            self.insert_type_header(tydef);
        }
    }

    fn walk_type_def_in_stmts(&mut self, types: Vec<AstTypeDef>) {
        // Walk the type definitions
        for tydef in types {
            self.walk_type_def(tydef.clone());
        }
    }

    fn insert_func_headers_in_stmts(&mut self, funcs: &[AstFunction]) {
        // Insert function headers
        for func in funcs {
            if self.variables.contains_key(&func.name.kind) {
                error!(
                    &func.name.span.clone(),
                    "A function or variable with the same name exists"
                );
                continue;
            }

            if let Some((header, func_id)) = self.generate_function_header(&func) {
                let fh = FunctionHeaderWithId::new_self(header, func_id);
                self.variables.insert(func.name.kind, Entry::Function(fh));
            }
        }
    }

    fn walk_block(&mut self, block: Block) -> Option<(Type, IRExpr)> {
        self.insert_extern_module_headers(&block.imports);
        self.insert_type_headers_in_stmts(&block.types);
        self.walk_type_def_in_stmts(block.types);

        let _ = self.tycons.resolve();

        self.insert_func_headers_in_stmts(&block.functions);

        // Walk the statements
        let mut stmts: Vec<IRStmt> = Vec::new();
        for stmt in block.stmts {
            let stmt_irs = self.walk_stmt(stmt);
            if let Some(stmt_irs) = stmt_irs {
                for stmt_ir in stmt_irs.iter() {
                    stmts.push(stmt_ir.clone());
                }
            }
        }

        let result_expr = self.walk_expr(*block.result_expr);

        // Walk the function bodies in the statements
        for func in block.functions {
            if let Some(Entry::Function(header)) = self.variables.get(&func.name.kind) {
                let header = header.header.clone();
                self.walk_function(func, &header);
            }
        }

        let result_expr = result_expr?;
        let result_expr_ir = IRExpr::Copy(box result_expr.ir, type_size_nocheck(&result_expr.ty));

        // Wrap in Seq if necessary
        let block_expr = if stmts.is_empty() {
            result_expr_ir
        } else {
            IRExpr::Seq(stmts, box result_expr_ir)
        };

        Some((result_expr.ty, block_expr))
    }

    // =======================================
    // Type Definition
    // =======================================

    fn walk_type(&mut self, ty: Spanned<AstType>) -> Option<Type> {
        match ty.kind {
            AstType::Int => Some(Type::Int),
            AstType::Bool => Some(Type::Bool),
            AstType::Unit => Some(Type::Unit),
            AstType::String => Some(Type::String),
            AstType::Named(name) => {
                if let Some(ty) = self.types.get(&name) {
                    Some(ty.clone())
                } else if let Some(size) = self.tycons.get_size(name) {
                    Some(Type::App(TypeCon::Named(name, size), Vec::new()))
                } else if self.tycons.contains(name) {
                    Some(Type::App(TypeCon::UnsizedNamed(name), Vec::new()))
                } else {
                    error!(&ty.span, "undefined type `{}`", IdMap::name(name));
                    None
                }
            }
            AstType::Pointer(ty, is_mutable) => Some(Type::App(
                TypeCon::Pointer(is_mutable),
                vec![self.walk_type(*ty)?],
            )),
            AstType::Array(ty, size) => {
                Some(Type::App(TypeCon::Array(size), vec![self.walk_type(*ty)?]))
            }
            AstType::Tuple(types) => {
                let mut new_types = Vec::new();
                for ty in types {
                    new_types.push(self.walk_type(ty)?);
                }

                Some(Type::App(TypeCon::Tuple, new_types))
            }
            AstType::Struct(fields) => {
                let mut field_names = Vec::new();
                let mut types = Vec::new();
                for (name, ty) in fields {
                    field_names.push(name.kind);
                    types.push(self.walk_type(ty)?);
                }

                Some(Type::App(TypeCon::Struct(field_names), types))
            }
            AstType::App(name, types) => {
                self.push_type_scope();

                let tycon = match self.tycons.get_size(name.kind) {
                    Some(size) => TypeCon::Named(name.kind, size),
                    None if self.tycons.contains(name.kind) => TypeCon::UnsizedNamed(name.kind),
                    None => {
                        error!(&name.span, "undefined type `{}`", IdMap::name(name.kind));
                        return None;
                    }
                };

                let mut new_types = Vec::with_capacity(types.len());
                for ty in types {
                    let ty = self.walk_type(ty)?;
                    new_types.push(ty);
                }

                self.pop_type_scope();

                Some(Type::App(tycon, new_types))
            }
            AstType::Arrow(arg, ret) => {
                let arg = self.walk_type(*arg);
                let ret = self.walk_type(*ret);
                try_some!(arg, ret);

                Some(Type::App(TypeCon::Arrow, vec![arg, ret]))
            }
        }
    }

    fn insert_type_header(&mut self, tydef: &AstTypeDef) {
        self.tycons.insert(tydef.name);
        self.tycon_spans.insert(tydef.name, tydef.ty.span.clone());
    }

    fn walk_type_def(&mut self, tydef: AstTypeDef) {
        self.push_type_scope();

        let mut vars = Vec::with_capacity(tydef.var_ids.len());
        for var in &tydef.var_ids {
            vars.push(TypeVar::with_id(var.kind));
            self.types
                .insert(var.kind, Type::Var(*vars.last().unwrap()));
        }

        let mut ty = match self.walk_type(tydef.ty.clone()) {
            Some(ty) => ty,
            None => return,
        };
        wrap_typevar(&mut ty);

        let tycon = TypeCon::Fun(vars, Box::new(ty));

        self.pop_type_scope();

        self.tycons
            .set_body(tydef.name, TypeCon::Unique(Box::new(tycon), Unique::new()));
    }

    // =================================
    //  Function Definition
    // =================================

    // Insert parameters and return value as variables and return instructions that copy the
    // parameters to heap
    fn insert_params_as_variables(
        &mut self,
        params: Vec<Param>,
    ) -> Option<(usize, usize, Vec<IRStmt>)> {
        let mut stmts = Vec::new();

        let mut stack_in_heap_size = 0;
        let mut param_size = 0;
        let mut loc = 0isize;

        for param in params.into_iter().rev() {
            let size = type_size_nocheck(&param.ty);
            loc -= size as isize;
            param_size += size;

            let var_loc = if param.is_escaped {
                // Copy the parameter to heap if escaped
                let heap_loc = stack_in_heap_size + 1;

                for stmt in translate::escaped_param(&param.ty, loc, heap_loc).into_iter() {
                    stmts.push(stmt);
                }

                stack_in_heap_size += type_size_nocheck(&param.ty);

                VariableLoc::StackInHeap(heap_loc, self.var_level)
            } else {
                VariableLoc::Stack(loc)
            };

            // Insert the parameter as a variable to the current scope
            let var = Entry::Variable(Variable::new(param.ty.clone(), param.is_mutable, var_loc));
            self.variables.insert(param.name, var);
        }

        Some((param_size, stack_in_heap_size, stmts))
    }

    fn generate_function_header(&mut self, func: &AstFunction) -> Option<(FunctionHeader, usize)> {
        self.push_type_scope();

        let mut vars = Vec::new();
        for var_id in &func.ty_params {
            vars.push((var_id.kind, TypeVar::with_id(var_id.kind)));
            self.types
                .insert(var_id.kind, Type::Var(vars.last().unwrap().1));
        }

        let mut param_types = Vec::new();
        for AstParam { ty, .. } in &func.params {
            let ty_span = ty.span.clone();
            let mut ty = match self.walk_type(ty.clone()) {
                Some(ty) => ty,
                None => return None,
            };
            wrap_typevar(&mut ty);

            Self::type_size_err(ty_span, &ty);
            param_types.push(ty);
        }

        let return_ty_span = func.return_ty.span.clone();
        let mut return_ty = match self.walk_type(func.return_ty.clone()) {
            Some(ty) => ty,
            None => return None,
        };

        wrap_typevar(&mut return_ty);

        self.pop_type_scope();

        Self::type_size_err(return_ty_span, &return_ty);

        // Insert a header of the function
        let header = FunctionHeader {
            params: param_types,
            return_ty,
            ty_params: vars,
        };

        let ir_func = IRFunction {
            stack_in_heap_size: 0,
            stack_size: 0,
            param_size: 0,
            body: IRExpr::Unit,
        };
        self.ir_funcs.push((func.name.kind, ir_func));

        Some((header, self.ir_funcs.len() - 1))
    }

    fn walk_function(&mut self, func: AstFunction, header: &FunctionHeader) {
        let func_name = func.name.kind;
        let ir_func_index = self
            .ir_funcs
            .iter_mut()
            .position(|(name, _)| *name == func_name)
            .unwrap();

        self.current_func = func.name.kind;
        self.current_func_index = ir_func_index;

        self.push_scope();
        self.push_type_scope();

        if func.has_escaped_variables {
            self.var_level += 1;
        }

        let return_ty = header.return_ty.clone();

        for (id, var) in &header.ty_params {
            self.types.insert(*id, Type::Var(*var));
        }

        // Convert AstParam to Param
        let params: Vec<Param> = func
            .params
            .iter()
            .zip(header.params.iter())
            .map(|(ap, ty)| Param {
                name: ap.name,
                ty: ty.clone(),
                is_mutable: ap.is_mutable,
                is_escaped: ap.is_escaped,
            })
            .collect();

        let (param_size, heap_size, param_stmts) = match self.insert_params_as_variables(params) {
            Some(t) => t,
            None => return,
        };

        // Add stack_in_heap_size before walk the function body
        let (_, ir_func) = &mut self.ir_funcs[ir_func_index];
        ir_func.param_size = param_size;
        ir_func.stack_in_heap_size = heap_size;

        let body = match self.walk_expr(func.body) {
            Some(e) => e,
            None => return,
        };

        // Check if a specified return type is the same as the expression type
        unify(&body.span, &return_ty, &body.ty);

        // Set function body
        let (_, ir_func) = &mut self.ir_funcs[ir_func_index];

        let body_expr = if param_stmts.is_empty() {
            body.ir
        } else {
            IRExpr::Seq(param_stmts, box body.ir)
        };
        ir_func.body = IRExpr::Copy(box body_expr, type_size_nocheck(&body.ty));

        // Finalize
        if func.has_escaped_variables {
            self.var_level -= 1;
        }

        self.pop_type_scope();
        self.pop_scope();
    }

    // =========================================
    // Implementation
    // =========================================

    fn insert_impl_headers(&mut self, implementation: &Impl) {
        let mut new_impl = Implementation::new();

        for func in &implementation.functions {
            if let Some((header, func_id)) = self.generate_function_header(&func) {
                new_impl.functions.insert(func.name.kind, (func_id, header));
            }
        }

        self.impls.insert(implementation.target.kind, new_impl);
    }

    fn walk_impl(&mut self, implementation: Impl) {
        let impl_header = match self.impls.get(&implementation.target.kind) {
            Some(imp) => imp,
            None => return,
        };

        assert_eq!(impl_header.functions.len(), implementation.functions.len());

        let iter = impl_header
            .functions
            .clone()
            .into_iter()
            .map(|(_, (_, h))| h)
            .zip(implementation.functions.into_iter());

        for (header, func) in iter {
            self.walk_function(func, &header);
        }
    }

    // =========================================
    // Module
    // =========================================

    pub fn load_modules(
        &mut self,

        imported_modules: &[SymbolPath],
        module_headers: &FxHashMap<SymbolPath, ModuleHeader>,
    ) {
        assert!(self.module_headers.is_empty());

        let mut headers = FxHashMap::default();
        for path in imported_modules {
            let module_header = module_headers[&path].clone();

            let module_id = self.next_module_id;
            self.next_module_id += 1;
            headers.insert(path.clone(), (module_id, module_header));
        }

        self.module_headers = headers;
    }

    pub fn update_modules(&mut self, module_headers: &FxHashMap<SymbolPath, ModuleHeader>) {
        for (path, (_, header)) in &mut self.module_headers {
            *header = module_headers[path].clone();
        }
    }

    pub fn insert_public_type_headers(
        &mut self,
        program: &Program,
        module_header: &mut ModuleHeader,
    ) {
        // Insert type headers
        for tydef in &program.main.types {
            module_header.types.insert(tydef.name, None);
            self.tycons.insert(tydef.name);
        }
    }

    pub fn walk_public_type(&mut self, program: &Program, module_header: &mut ModuleHeader) {
        // Import extern module types
        for range in &program.main.imports {
            for path in range.kind.to_paths() {
                let mut list = Vec::new();
                self.generate_from_range_path(&range.span, &path, |m, o, r| {
                    list.push((m, o, r));
                });

                for (module_id, original, renamed) in list {
                    let (_, header) = self
                        .module_headers
                        .values()
                        .find(|(id, _)| *id == module_id)
                        .unwrap();

                    if header.types.get(&original).is_some() {
                        self.tycons.insert(renamed);
                    }
                }
            }
        }

        // Import type headers only
        self.walk_type_def_in_stmts(program.main.types.clone());

        for tydef in &program.main.types {
            let body = match self.tycons.get(tydef.name) {
                Some(body) => body,
                None => continue,
            };

            let tycon = body.tycon().clone();
            module_header.types.insert(tydef.name, Some(tycon));
        }
    }

    pub fn resolve_type(&mut self, program: &Program) {
        // Import extern module types
        for range in &program.main.imports {
            for path in range.kind.to_paths() {
                let mut list = Vec::new();
                self.generate_from_range_path(&range.span, &path, |m, o, r| {
                    list.push((m, o, r));
                });

                for (module_id, original, renamed) in list {
                    let (_, header) = self
                        .module_headers
                        .values()
                        .find(|(id, _)| *id == module_id)
                        .unwrap();

                    if let Some(Some(tycon)) = header.types.get(&original) {
                        self.tycons.set_body(renamed, tycon.clone());
                    }
                }
            }
        }

        let _ = self.tycons.resolve();
    }

    pub fn insert_public_functions(&mut self, program: &Program, module_header: &mut ModuleHeader) {
        // Insert type headers
        for tydef in &program.main.types {
            self.insert_type_header(tydef);
        }

        // Insert main function header
        let header = FunctionHeader {
            params: Vec::new(),
            return_ty: Type::Unit,
            ty_params: Vec::new(),
        };
        module_header
            .functions
            .insert(*reserved_id::MAIN_FUNC, (0, header));

        for implementation in &program.impls {
            if self.tycons.contains(implementation.target.kind) {
                self.insert_impl_headers(implementation);
            }
        }

        // Walk the type definitions
        for tydef in &program.main.types {
            self.walk_type_def(tydef.clone());
        }

        if let Err(ids) = self.tycons.resolve() {
            for id in ids {
                let span = self.tycon_spans.get(&id).unwrap();
                error!(
                    span,
                    "The type `{}` could not be calculated",
                    IdMap::name(id)
                )
            }
        }

        // main
        self.ir_funcs.push((
            *reserved_id::MAIN_FUNC,
            IRFunction {
                stack_size: 0,
                stack_in_heap_size: 0,
                param_size: 0,
                body: IRExpr::Unit,
            },
        ));

        for func in &program.main.functions {
            if let Some((header, func_id)) = self.generate_function_header(&func) {
                module_header
                    .functions
                    .insert(func.name.kind, (func_id, header));
            }
        }

        self.pop_type_scope();
    }

    pub fn analyze(mut self, program: Program) -> IRModule {
        // Clear headers inserted by Self::insert_public_functions
        self.ir_funcs.clear();

        self.push_scope();
        self.push_type_scope();

        let range = ImportRange::Scope(*reserved_id::STD_MODULE, Box::new(ImportRange::All));
        self.insert_func_headers_by_range(&Spanned {
            kind: range,
            span: program.main.result_expr.span.clone(),
        });

        // The headers are already inserted in Self::insert_public_functions
        for implementation in program.impls {
            self.walk_impl(implementation);
        }

        let main_func = IRFunction {
            stack_size: 0,
            stack_in_heap_size: 0,
            param_size: 0,
            body: IRExpr::Unit,
        };
        self.ir_funcs.push((*reserved_id::MAIN_FUNC, main_func));

        // Main statements
        let body = self.walk_block(program.main);
        if let Some((_, body)) = body {
            self.ir_funcs[0].1.body = body;
        }

        self.pop_scope();
        self.pop_type_scope();

        IRModule {
            functions: self.ir_funcs,
            imported_modules: program
                .imported_modules
                .into_iter()
                .map(|m| format!("{}", m))
                .collect(),
        }
    }
}

pub enum ModuleBody {
    Normal(IRModule),
    Native(Module),
}

pub fn do_semantics_analysis(
    module_buffers: FxHashMap<SymbolPath, Program>,
    native_modules: &ModuleContainer,
) -> FxHashMap<String, ModuleBody> {
    struct Module<'a> {
        analyzer: Analyzer<'a>,
    }

    impl<'a> Module<'a> {
        fn new() -> Self {
            Self {
                analyzer: Analyzer::new(),
            }
        }
    }

    let mut modules: FxHashMap<SymbolPath, Module<'_>> = FxHashMap::default();
    let mut module_headers: FxHashMap<SymbolPath, ModuleHeader> = FxHashMap::default();
    let mut bodies: FxHashMap<String, ModuleBody> = FxHashMap::default();

    let mut imported_native_modules: FxHashSet<SymbolPath> = FxHashSet::default();

    // Initialize and insert type headers
    for (path, program) in &module_buffers {
        modules.insert(path.clone(), Module::new());
        module_headers.insert(path.clone(), ModuleHeader::new(path));

        let module = modules.get_mut(path).unwrap();
        let module_header = module_headers.get_mut(path).unwrap();
        module
            .analyzer
            .insert_public_type_headers(program, module_header);

        // Native modules
        for path in &program.imported_modules {
            if let Some(module) = native_modules.get(path) {
                module_headers.insert(path.clone(), module.header.clone());
                imported_native_modules.insert(path.clone());
            }
        }
    }

    // Walk types
    for (path, program) in &module_buffers {
        let module = modules.get_mut(path).unwrap();

        // Load modules
        module
            .analyzer
            .load_modules(&program.imported_modules, &module_headers);

        let module_header = module_headers.get_mut(path).unwrap();
        module.analyzer.walk_public_type(program, module_header);
    }

    for (path, program) in &module_buffers {
        let module = modules.get_mut(path).unwrap();
        module.analyzer.update_modules(&module_headers);
        module.analyzer.resolve_type(program);
    }

    // Insert function headers
    for (path, program) in &module_buffers {
        let module = modules.get_mut(path).unwrap();
        let module_header = module_headers.get_mut(path).unwrap();

        // Insert function headers
        module
            .analyzer
            .insert_public_functions(program, module_header);
    }

    for (name, program) in module_buffers {
        let Module { mut analyzer } = modules.remove(&name).unwrap();

        analyzer.update_modules(&module_headers);

        let bytecode = analyzer.analyze(program);
        bodies.insert(format!("{}", name), ModuleBody::Normal(bytecode));
    }

    for path in &imported_native_modules {
        let module = native_modules.get(path).unwrap().module.clone();
        bodies.insert(format!("{}", path), ModuleBody::Native(module));
    }

    bodies
}
