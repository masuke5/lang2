use std::collections::LinkedList;
use std::mem;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::{Param as AstParam, *};
use crate::bytecode::{Bytecode, BytecodeBuilder, Function, InstList};
use crate::error::Error;
use crate::id::{reserved_id, Id, IdMap};
use crate::module::{FunctionHeader, Module, ModuleContainer, ModuleHeader};
use crate::span::{Span, Spanned};
use crate::translate;
use crate::translate::RelativeVariableLoc;
use crate::ty::*;
use crate::utils::HashMapWithScope;

macro_rules! error {
    ($self:ident, $span:expr, $fmt: tt $(,$arg:expr)*) => {
        $self.errors.push(Error::new(&format!($fmt $(,$arg)*), $span));
    };
}

macro_rules! try_some {
    ($($var:ident),*) => {
        $(let $var = $var?;)*
    };
}

macro_rules! fn_to_expect {
    ($fn_name:ident, $type_name:tt, $ty:ty, $pat:pat => $expr:expr,) => {
        fn $fn_name<'a>(errors: &mut Vec<Error>, ty: &'a Type, span: Span) -> Option<&'a $ty> {
            match ty {
                $pat => $expr,
                _ => {
                    let msg = format!(
                        concat!("expected type `", $type_name, "` but got type `{}`"),
                        ty
                    );
                    let error = Error::new(&msg, span);
                    errors.push(error);
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
    // Type
}

#[derive(Debug)]
pub struct ExprInfo {
    pub ty: Type,
    pub span: Span,
    pub insts: InstList,
    pub is_lvalue: bool,
    pub is_mutable: bool,
}

impl ExprInfo {
    fn new(insts: InstList, ty: Type, span: Span) -> Self {
        Self {
            ty,
            span,
            insts,
            is_lvalue: false,
            is_mutable: false,
        }
    }

    fn new_lvalue(insts: InstList, ty: Type, span: Span, is_mutable: bool) -> Self {
        Self {
            ty,
            span,
            insts,
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
    module_id: Option<u16>,
    original_name: Option<Id>,
    func_id: u16,
    header: FunctionHeader,
}

impl FunctionHeaderWithId {
    fn new(module_id: u16, func_id: u16, header: FunctionHeader) -> Self {
        Self {
            module_id: Some(module_id),
            original_name: None,
            func_id,
            header,
        }
    }

    fn new_renamed(
        module_id: u16,
        func_id: u16,
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

    fn new_self(header: FunctionHeader, func_id: u16) -> Self {
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
    module_headers: FxHashMap<SymbolPath, (u16, ModuleHeader)>,
    types: HashMapWithScope<Id, Type>,
    tycons: HashMapWithScope<Id, Option<TypeCon>>,
    type_sizes: HashMapWithScope<Id, usize>,
    variables: HashMapWithScope<Id, Entry>,
    errors: Vec<Error>,
    current_func: Id,
    next_temp_num: u32,
    next_unique: u32,
    function_insts: LinkedList<(Id, InstList)>,
    var_level: usize,
    _phantom: &'a std::marker::PhantomData<Self>,
}

impl<'a> Analyzer<'a> {
    pub fn new() -> Self {
        Self {
            visible_modules: FxHashSet::default(),
            module_headers: FxHashMap::default(),
            variables: HashMapWithScope::new(),
            types: HashMapWithScope::new(),
            tycons: HashMapWithScope::new(),
            type_sizes: HashMapWithScope::new(),
            errors: Vec::new(),
            current_func: *reserved_id::MAIN_FUNC,
            next_temp_num: 0,
            next_unique: 0,
            function_insts: LinkedList::new(),
            var_level: 0,
            _phantom: &std::marker::PhantomData,
        }
    }

    fn add_error(&mut self, msg: &str, span: Span) {
        self.errors.push(Error::new(msg, span));
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
        self.type_sizes.push_scope();
    }

    #[inline]
    fn pop_type_scope(&mut self) {
        self.type_sizes.pop_scope();
        self.tycons.pop_scope();
        self.types.pop_scope();
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
            Type::App(TypeCon::Named(name, _), types) => match self.tycons.get(&name) {
                Some(tycon) => Some(Type::App(tycon.clone().unwrap(), types)),
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
    fn type_size_err(&mut self, span: Span, ty: &Type) -> usize {
        match type_size(ty) {
            Some(size) => size,
            None => {
                self.errors.push(Error::new(
                    &format!("the size of type `{}` cannot be calculated", ty),
                    span,
                ));
                0
            }
        }
    }

    // ====================================
    //  Variable
    // ====================================

    fn new_var(
        &mut self,
        current_func: &mut Function,
        id: Id,
        ty: Type,
        is_mutable: bool,
        is_escaped: bool,
    ) -> VariableLoc {
        let new_var_size = type_size_nocheck(&ty);

        let loc = if is_escaped {
            let loc = VariableLoc::StackInHeap(
                current_func.stack_in_heap_size as usize + 1,
                self.var_level,
            );
            current_func.stack_in_heap_size += new_var_size as u8;
            loc
        } else {
            let loc = VariableLoc::Stack(current_func.stack_size as isize);
            current_func.stack_size += new_var_size as u8;
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
        code: &mut BytecodeBuilder,
        id: Id,
        ty: Type,
        is_mutable: bool,
        is_escaped: bool,
    ) -> VariableLoc {
        self.new_var(
            code.get_function_mut(self.current_func).unwrap(),
            id,
            ty,
            is_mutable,
            is_escaped,
        )
    }

    fn gen_temp_id(&mut self) -> Id {
        let id = IdMap::new_id(&format!("$comp{}", self.next_temp_num));
        self.next_temp_num += 1;
        id
    }

    fn get_return_var(&self) -> &Variable {
        match self.find_var(*reserved_id::RETURN_VALUE).unwrap() {
            Entry::Variable(var) => var,
            Entry::Function(..) => panic!("the return variable should not be a function entry"),
        }
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

    fn find_external_func(&self, path: &SymbolPath) -> Option<(&FunctionHeader, u16, Option<u16>)> {
        // header, code id, module id
        assert!(!path.is_empty());

        let module_path = path.parent().unwrap();
        let func_id = path.tail().unwrap().id;

        if module_path.is_empty() {
            None
        } else {
            if !self.visible_modules.contains(&module_path) {
                return None;
            }

            // Get the function from the module
            let (module_id, module) = self.module_headers.get(&module_path)?;
            module
                .functions
                .get(&func_id)
                .map(|(func_id, fh)| (fh, *func_id, Some(*module_id)))
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
                            self,
                            span.clone(),
                            "expected type `struct` or `*struct` but got type `{}`",
                            ty
                        );
                        None
                    }
                }
            }
            ty => {
                error!(
                    self,
                    span.clone(),
                    "expected type `struct` or `*struct` but got type `{}`",
                    ty
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

    fn walk_call(
        &mut self,
        code: &mut BytecodeBuilder,
        func_expr: Spanned<Expr>,
        arg_expr: Spanned<Expr>,
        map: &mut FxHashMap<TypeVar, Type>,
    ) -> Option<(Type, Type, InstList, InstList)> {
        let func_span = func_expr.span.clone();
        let (ty, func_ty, mut insts, func_insts) = match func_expr.kind {
            Expr::Call(func_expr, arg_expr) => self.walk_call(code, *func_expr, *arg_expr, map)?,
            Expr::Variable(name, _) => match self.variables.get(&name) {
                Some(Entry::Function(fh)) => {
                    let ty = generate_func_type(
                        &fh.header.params,
                        &fh.header.return_ty,
                        &fh.header.ty_params,
                    );

                    (
                        ty.clone(),
                        ty,
                        InstList::new(),
                        translate::call_func_id(fh.module_id, fh.func_id),
                    )
                }
                _ => {
                    let expr = self.walk_expr(code, func_expr)?;
                    (
                        expr.ty.clone(),
                        expr.ty.clone(),
                        InstList::new(),
                        translate::call_expr(expr),
                    )
                }
            },
            _ => {
                let expr = self.walk_expr(code, func_expr)?;
                (
                    expr.ty.clone(),
                    expr.ty.clone(),
                    InstList::new(),
                    translate::call_expr(expr),
                )
            }
        };

        // Unwrap Poly
        let ty = match ty {
            Type::Poly(_, ty) => *ty,
            ty => ty,
        };

        let (mut arg_ty, mut return_ty) = match &ty {
            Type::App(TypeCon::Arrow, types) => (types[0].clone(), types[1].clone()),
            ty => {
                error!(self, func_span, "expected type `function` but got `{}`", ty);
                return None;
            }
        };

        let arg_expr = self.walk_expr_with_conversion(code, arg_expr, &arg_ty)?;
        Self::insert_type_param(&arg_ty, &arg_expr.ty, map);
        arg_ty = subst(arg_ty, &map);
        return_ty = subst(return_ty, &map);

        unify(&mut self.errors, &arg_expr.span, &arg_ty, &arg_expr.ty)?;

        translate::arg(&mut insts, arg_expr);

        Some((return_ty, func_ty, insts, func_insts))
    }

    fn walk_expr_with_conversion(
        &mut self,
        code: &mut BytecodeBuilder,
        expr: Spanned<Expr>,
        ty: &Type,
    ) -> Option<ExprInfo> {
        match ty {
            Type::App(TypeCon::Wrapped, _) => {
                let expr = self.walk_expr(code, expr)?;

                if let Type::App(TypeCon::Wrapped, _) = &expr.ty {
                    Some(expr)
                } else {
                    let mut expr = expr;
                    expr.insts = translate::wrap(expr.insts, &expr.ty);
                    expr.ty = Type::App(TypeCon::Wrapped, vec![expr.ty]);

                    Some(expr)
                }
            }
            _ => match expr.kind {
                Expr::Tuple(exprs) => {
                    let types = match ty {
                        Type::App(TypeCon::Tuple, types) => types,
                        _ => return None,
                    };

                    // Check size of the tuples
                    if types.len() != exprs.len() {
                        error!(
                            self,
                            expr.span,
                            "expected that tuple size will be `{}` but the number of expressions `{}`",
                            types.len(),
                            exprs.len()
                        );
                        return None;
                    }

                    // Walk each expression and join all instructions
                    let mut insts = InstList::new();
                    let mut tys = Vec::new();

                    for (expr, ty) in exprs.into_iter().zip(types.iter()) {
                        let expr = self.walk_expr_with_conversion(code, expr, ty);
                        if let Some(expr) = expr {
                            tys.push(expr.ty.clone());
                            insts.append(translate::literal_tuple(expr));
                        }
                    }

                    Some(ExprInfo::new(
                        insts,
                        Type::App(TypeCon::Tuple, tys),
                        expr.span,
                    ))
                }
                _ => {
                    let mut expr = self.walk_expr(code, expr)?;

                    // if `ty` is not wrapped and `expr.ty` is wrapped
                    if let Type::App(TypeCon::Wrapped, _) = ty {
                        // Does nothing
                    } else if let Type::App(TypeCon::Wrapped, types) = &expr.ty {
                        expr.insts = translate::unwrap(expr.insts, &expr.ty);
                        expr.ty = types[0].clone();
                    }

                    Some(expr)
                }
            },
        }
    }

    fn walk_expr_with_unwrap(
        &mut self,
        code: &mut BytecodeBuilder,
        expr: Spanned<Expr>,
    ) -> Option<ExprInfo> {
        let mut expr = self.walk_expr(code, expr)?;

        if let Type::App(TypeCon::Wrapped, types) = &expr.ty {
            expr.insts = translate::unwrap(expr.insts, &expr.ty);
            expr.ty = types[0].clone();
        }

        Some(expr)
    }

    #[allow(clippy::cognitive_complexity)]
    fn walk_expr(&mut self, code: &mut BytecodeBuilder, expr: Spanned<Expr>) -> Option<ExprInfo> {
        let (insts, ty) = match expr.kind {
            Expr::Literal(Literal::Number(n)) => (translate::literal_int(n), Type::Int),
            Expr::Literal(Literal::String(i)) => {
                let ty = Type::App(TypeCon::Pointer(false), vec![Type::String]);
                (translate::literal_str(i), ty)
            }
            Expr::Literal(Literal::Unit) => (InstList::new(), Type::Unit),
            Expr::Literal(Literal::True) => (translate::literal_true(), Type::Bool),
            Expr::Literal(Literal::False) => (translate::literal_false(), Type::Bool),
            Expr::Literal(Literal::Null) => (translate::literal_null(), Type::Null),
            Expr::Tuple(exprs) => {
                let mut insts = InstList::new();
                let mut types = Vec::new();
                for expr in exprs {
                    let expr = self.walk_expr(code, expr);
                    if let Some(expr) = expr {
                        types.push(expr.ty.clone());
                        insts.append(translate::literal_tuple(expr));
                    }
                }

                (insts, Type::App(TypeCon::Tuple, types))
            }
            Expr::Struct(ty, field_exprs) => {
                let ty = self.walk_type(ty)?;
                let mut expr_ty = ty.clone();

                let ty = self.expand_name(ty)?;

                let ty_params = match ty.clone() {
                    Type::App(TypeCon::Unique(box TypeCon::Fun(params, _), _), _) => params,
                    _ => panic!(),
                };

                let ty = expand_unique(ty);
                let ty = expand_wrap(ty);
                let ty = subst(ty, &FxHashMap::default());

                let mut fields: Vec<(Id, Type)> = match ty {
                    Type::App(TypeCon::Struct(fields), tys) => {
                        fields.into_iter().zip(tys.into_iter()).collect()
                    }
                    ty => {
                        error!(self, expr.span.clone(), "type `{}` is not struct", ty);
                        return None;
                    }
                };

                let mut insts = InstList::new();
                let mut map = FxHashMap::default();

                // Push instructions to `insts` in order
                for i in 0..fields.len() {
                    let field_expr = field_exprs
                        .iter()
                        .find(|(id, _)| id.kind == fields[i].0)
                        .map(|(_, expr)| expr);
                    if let Some(expr) = field_expr {
                        if let Some(expr) =
                            self.walk_expr_with_conversion(code, expr.clone(), &fields[i].1)
                        {
                            Self::insert_type_param(&fields[i].1, &expr.ty, &mut map);
                            for (_, ty) in &mut fields {
                                *ty = subst(ty.clone(), &map);
                            }

                            unify(&mut self.errors, &expr.span, &fields[i].1, &expr.ty);
                            insts.append(translate::literal_struct_field(expr));
                        }
                    } else {
                        error!(
                            self,
                            expr.span.clone(),
                            "missing field `{}`",
                            IdMap::name(fields[i].0)
                        );
                    }
                }

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

                (insts, expr_ty)
            }
            Expr::Array(expr, size) => {
                let expr = self.walk_expr(code, *expr)?;
                let ty = expr.ty.clone();

                (
                    translate::literal_array(expr, size),
                    Type::App(TypeCon::Array(size), vec![ty]),
                )
            }
            Expr::Field(comp_expr, field) => {
                let should_store = Self::expr_push_multiple_values(&comp_expr.kind);
                let comp_expr = self.walk_expr(code, *comp_expr)?;
                let mut is_mutable = comp_expr.is_mutable;

                let loc = if should_store {
                    let id = self.gen_temp_id();
                    Some(self.new_var_in_current_func(code, id, comp_expr.ty.clone(), false, false))
                } else {
                    None
                };

                let ty = self.expand_name(comp_expr.ty.clone())?;
                let ty = expand_wrap(ty);

                let should_deref = match &ty {
                    Type::App(TypeCon::Pointer(_), _) => true,
                    _ => false,
                };

                // Get the field type and offset
                let (field_ty, offset) = match field {
                    Field::Number(i) => {
                        let types = match &ty {
                            Type::App(TypeCon::Pointer(is_mutable_), tys) => {
                                is_mutable = *is_mutable_;
                                expect_tuple(&mut self.errors, &tys[0], comp_expr.span.clone())?
                            }
                            ty => expect_tuple(&mut self.errors, ty, comp_expr.span.clone())?,
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
                                error!(self, expr.span, "error");
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
                                    self,
                                    expr.span,
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

                let insts = translate::field(
                    loc.map(|loc| self.relative_loc(&loc)),
                    should_deref,
                    comp_expr,
                    offset,
                );
                return Some(ExprInfo::new_lvalue(insts, field_ty, expr.span, is_mutable));
            }
            Expr::Subscript(expr, subscript_expr) => {
                let should_store = Self::expr_push_multiple_values(&expr.kind);

                let expr = self.walk_expr_with_unwrap(code, *expr);
                let subscript_expr =
                    self.walk_expr_with_conversion(code, *subscript_expr, &Type::Int);
                try_some!(expr, subscript_expr);

                let mut expr = expr;

                let loc = if should_store {
                    let id = self.gen_temp_id();
                    Some(self.new_var_in_current_func(code, id, expr.ty.clone(), false, false))
                } else {
                    None
                };

                let (ty, should_deref) = match expr.ty.clone() {
                    Type::App(TypeCon::Array(_), tys) => (tys[0].clone(), false),
                    Type::App(TypeCon::Pointer(is_mutable), tys) => {
                        expr.is_mutable = is_mutable;

                        match &tys[0] {
                            Type::App(TypeCon::Array(_), tys) => (tys[0].clone(), true),
                            ty => {
                                error!(
                                    self,
                                    expr.span.clone(),
                                    "expected array but got type `{}`",
                                    ty
                                );
                                return None;
                            }
                        }
                    }
                    ty => {
                        error!(
                            self,
                            expr.span.clone(),
                            "expected array but got type `{}`",
                            ty
                        );
                        return None;
                    }
                };

                unify(
                    &mut self.errors,
                    &subscript_expr.span,
                    &subscript_expr.ty,
                    &Type::Int,
                );

                let span = expr.span.clone();
                let is_lvalue = expr.is_lvalue;
                let is_mutable = expr.is_mutable;
                let insts = translate::subscript(
                    loc.map(|loc| self.relative_loc(&loc)),
                    should_deref,
                    expr,
                    subscript_expr,
                );

                return Some(ExprInfo {
                    ty,
                    insts,
                    span,
                    is_lvalue,
                    is_mutable,
                });
            }
            Expr::Variable(name, _) => {
                let entry = match self.find_var(name) {
                    Some(v) => v,
                    None => {
                        self.add_error("undefined variable or function", expr.span.clone());
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
                        let insts = translate::func_pos(fh.module_id, fh.func_id);
                        let ty = generate_func_type(
                            &fh.header.params,
                            &fh.header.return_ty,
                            &fh.header.ty_params,
                        );
                        (insts, ty, false)
                    }
                };
                return Some(ExprInfo::new_lvalue(insts, ty, expr.span, is_mutable));
            }
            Expr::Path(path) => {
                let (header, func_id, module_id) = match self.find_external_func(&path) {
                    Some(t) => t,
                    None => {
                        error!(self, expr.span, "undefined function `{}`", path);
                        return None;
                    }
                };

                let mut ty = header.return_ty.clone();
                for param_ty in &header.params {
                    ty = Type::App(TypeCon::Arrow, vec![param_ty.clone(), ty]);
                }

                let insts = translate::func_pos(module_id, func_id);
                (insts, ty)
            }
            Expr::BinOp(BinOp::And, lhs, rhs) => {
                let lhs = self.walk_expr_with_conversion(code, *lhs, &Type::Int);
                let rhs = self.walk_expr_with_conversion(code, *rhs, &Type::Int);
                try_some!(lhs, rhs);

                // Type check
                match (&lhs.ty, &rhs.ty) {
                    (Type::Bool, Type::Bool) => {}
                    (lty, rty) => {
                        error!(self, expr.span.clone(), "{} && {}", lty, rty);
                    }
                }

                (translate::binop_and(lhs, rhs), Type::Bool)
            }
            Expr::BinOp(BinOp::Or, lhs, rhs) => {
                let lhs = self.walk_expr_with_conversion(code, *lhs, &Type::Int);
                let rhs = self.walk_expr_with_conversion(code, *rhs, &Type::Int);
                try_some!(lhs, rhs);

                // Type check
                match (&lhs.ty, &rhs.ty) {
                    (Type::Bool, Type::Bool) => {}
                    (lty, rty) => {
                        error!(self, expr.span.clone(), "{} || {}", lty, rty);
                    }
                }

                (translate::binop_or(lhs, rhs), Type::Bool)
            }
            Expr::BinOp(binop, lhs, rhs) => {
                let lhs = self.walk_expr_with_unwrap(code, *lhs);
                let rhs = self.walk_expr_with_unwrap(code, *rhs);
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
                        self.add_error(
                            &format!("`{} {} {}`", lhs.ty, binop_symbol, rhs.ty),
                            expr.span.clone(),
                        );
                        return None;
                    }
                };

                (translate::binop(binop, lhs, rhs), ty)
            }
            Expr::Call(func_expr, arg_expr) => {
                let mut map = FxHashMap::default();
                let (ty, func_ty, args_insts, call_insts) =
                    self.walk_call(code, *func_expr, *arg_expr, &mut map)?;

                let func_ty = subst(func_ty, &map);

                let insts = translate::call(&ty, &func_ty, args_insts, call_insts);
                (insts, ty)
            }
            Expr::Address(expr, is_mutable) => {
                let expr = self.walk_expr_with_unwrap(code, *expr)?;

                if is_mutable && !expr.is_mutable {
                    error!(self, expr.span, "this expression is immutable");
                    return None;
                }

                let ty = Type::App(TypeCon::Pointer(is_mutable), vec![expr.ty.clone()]);

                if !expr.is_lvalue {
                    let temp = self.gen_temp_id();
                    let loc = self.new_var_in_current_func(
                        code,
                        temp,
                        expr.ty.clone(),
                        is_mutable,
                        false,
                    );

                    let insts = translate::address_no_lvalue(expr, &self.relative_loc(&loc));
                    (insts, ty)
                } else {
                    let insts = translate::address(expr);
                    (insts, ty)
                }
            }
            Expr::Dereference(expr_) => {
                let expr_ = self.walk_expr_with_unwrap(code, *expr_)?;

                match expr_.ty.clone() {
                    Type::App(TypeCon::Pointer(is_mutable), tys) => {
                        let insts = translate::dereference(expr_);
                        return Some(ExprInfo::new_lvalue(
                            insts,
                            tys[0].clone(),
                            expr.span,
                            is_mutable,
                        ));
                    }
                    ty => {
                        error!(
                            self,
                            expr.span, "expected type `pointer` but got type `{}`", ty
                        );
                        return None;
                    }
                }
            }
            Expr::Negative(expr) => {
                let expr = self.walk_expr_with_conversion(code, *expr, &Type::Int)?;

                match expr.ty.clone() {
                    ty @ Type::Int /* | Type::Float */ => {
                        (translate::negative(expr), ty)
                    },
                    ty => {
                        error!(self, expr.span, "expected type `int` or `float` but got type `{}`", ty);
                        return None;
                    },
                }
            }
            Expr::Alloc(expr, is_mutable) => {
                let expr = self.walk_expr_with_unwrap(code, *expr)?;

                let ty = Type::App(TypeCon::Pointer(is_mutable), vec![expr.ty.clone()]);
                let insts = translate::alloc(expr);
                (insts, ty)
            }
            Expr::Block(block) => {
                self.push_scope();

                let (ty, insts) = self.walk_block(code, block)?;

                self.pop_scope();

                (insts, ty)
            }
            Expr::If(cond, then_expr, None) => {
                let cond = self.walk_expr_with_conversion(code, *cond, &Type::Bool);
                let then_expr = self.walk_expr(code, *then_expr);
                try_some!(cond, then_expr);

                unify(&mut self.errors, &cond.span, &Type::Bool, &cond.ty);

                (
                    translate::if_expr(cond, then_expr.insts, &then_expr.ty),
                    Type::Unit,
                )
            }
            Expr::If(cond, then_expr, Some(else_expr)) => {
                let cond = self.walk_expr_with_conversion(code, *cond, &Type::Bool);
                let then = self.walk_expr(code, *then_expr);
                try_some!(cond, then);

                let els = self.walk_expr_with_conversion(code, *else_expr, &then.ty)?;

                unify(&mut self.errors, &cond.span, &Type::Bool, &cond.ty);
                unify(&mut self.errors, &els.span, &then.ty, &els.ty);

                (
                    translate::if_else_expr(cond, then.insts, els.insts, &then.ty),
                    then.ty.clone(),
                )
            }
            Expr::App(expr, old_tyargs) => {
                let expr = self.walk_expr(code, *expr);

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
                                self,
                                expr.span,
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
                            self,
                            expr.span, "The expression with type `{}` cannot be apply", ty
                        );
                        return None;
                    }
                };

                return Some(expr);
            }
        };

        Some(ExprInfo::new(insts, ty, expr.span))
    }

    // =====================================
    // Statement
    // =====================================

    fn walk_stmt(&mut self, code: &mut BytecodeBuilder, stmt: Spanned<Stmt>) -> Option<InstList> {
        let insts = match stmt.kind {
            Stmt::Expr(expr) => {
                let expr = self.walk_expr(code, expr)?;

                translate::expr_stmt(expr)
            }
            Stmt::While(cond, stmt) => {
                let cond = self.walk_expr_with_conversion(code, cond, &Type::Bool);
                let body = self.walk_stmt(code, *stmt);
                try_some!(cond, body);

                unify(&mut self.errors, &cond.span, &Type::Bool, &cond.ty);

                translate::while_stmt(cond, body)
            }
            Stmt::Bind(name, ty, expr, is_mutable, is_escaped) => {
                let expr = match ty {
                    Some(ty) => {
                        let ty = self.walk_type(ty)?;
                        let mut expr = self.walk_expr_with_conversion(code, *expr, &ty)?;

                        unify(&mut self.errors, &expr.span, &ty, &expr.ty)?;
                        expr.ty = ty;

                        expr
                    }
                    None => self.walk_expr(code, *expr)?,
                };

                let loc = self.new_var_in_current_func(
                    code,
                    name,
                    expr.ty.clone(),
                    is_mutable,
                    is_escaped,
                );

                translate::bind_stmt(&self.relative_loc(&loc), expr)
            }
            Stmt::Assign(lhs, rhs) => {
                let lhs = self.walk_expr(code, lhs)?;
                let rhs = self.walk_expr_with_conversion(code, *rhs, &lhs.ty)?;

                if !lhs.is_lvalue {
                    error!(self, lhs.span, "unassignable expression");
                    return None;
                }

                if !lhs.is_mutable {
                    error!(self, lhs.span, "immutable expression");
                    return None;
                }

                unify(&mut self.errors, &rhs.span, &lhs.ty, &rhs.ty)?;

                translate::assign_stmt(lhs, rhs)
            }
            Stmt::Return(expr) => {
                let func_name = code.get_function(self.current_func).unwrap().name;

                // Check if is outside function
                if func_name == *reserved_id::MAIN_FUNC {
                    error!(self, stmt.span, "return statement outside function");
                    return None;
                }

                let return_var_ty = self.get_return_var().ty.clone();

                let expr = match expr {
                    Some(expr) => {
                        Some(self.walk_expr_with_conversion(code, expr, &return_var_ty)?)
                    }
                    None => None,
                };

                // Check type
                let return_ty = expr.as_ref().map_or(&Type::Unit, |expr| &expr.ty);
                unify(&mut self.errors, &stmt.span, &return_var_ty, return_ty);

                let return_var = self.get_return_var();
                translate::return_stmt(&self.relative_loc(&return_var.loc), expr, &return_var.ty)
            }
        };

        Some(insts)
    }

    // =====================================
    // Importing
    // =====================================

    fn generate_from_range_path(
        &mut self,
        range_span: &Span,
        path: &ImportRangePath,
    ) -> Option<Vec<ImportedEntry>> {
        if let ImportRangePath::All(spath) = path {
            // Import all functions if module `spath` exists
            match self.module_headers.get(&spath) {
                Some((module_id, module)) => {
                    let mut entries = Vec::with_capacity(module.functions.len());
                    for (func_name, (func_id, func)) in &module.functions {
                        let header = FunctionHeaderWithId::new(*module_id, *func_id, func.clone());
                        entries.push(ImportedEntry::Function(*func_name, header));
                    }

                    return Some(entries);
                }
                _ => {
                    error!(self, range_span.clone(), "undefined module `{}`", spath);
                    return None;
                }
            }
        }

        let spath = path.as_path();
        if !self.module_headers.contains_key(spath) {
            // If module `spath` doesn't exists

            // Get the module path from `spath`
            let parent = spath.parent();
            if parent.is_none() {
                error!(self, range_span.clone(), "undefined module `{}`", spath);
                return None;
            }

            let parent = parent.unwrap();
            let symbol = spath.tail().unwrap();

            // Find the module
            match self.module_headers.get(&parent) {
                Some((module_id, module)) => match module.functions.get(&symbol.id) {
                    Some((func_id, func)) => {
                        let (name, header) = match &path {
                            ImportRangePath::Path(..) => (
                                symbol.id,
                                FunctionHeaderWithId::new(*module_id, *func_id, func.clone()),
                            ),
                            ImportRangePath::Renamed(_, renamed) => (
                                *renamed,
                                FunctionHeaderWithId::new_renamed(
                                    *module_id,
                                    *func_id,
                                    func.clone(),
                                    symbol.id,
                                ),
                            ),
                            ImportRangePath::All(..) => unreachable!(),
                        };

                        Some(vec![ImportedEntry::Function(name, header)])
                    }
                    None => {
                        // If the module exists but the function doesn't exist
                        error!(
                            self,
                            range_span.clone(),
                            "undefined function `{}` in `{}`",
                            IdMap::name(symbol.id),
                            parent
                        );
                        None
                    }
                },
                None => {
                    error!(self, range_span.clone(), "undefined module `{}`", spath);
                    None
                }
            }
        } else {
            // if path is module
            self.visible_modules.insert(path.as_path().clone());
            None
        }
    }

    fn insert_header_from_imported_entry(&mut self, entry: ImportedEntry) {
        match entry {
            ImportedEntry::Function(name, header) => {
                self.variables.insert(name, Entry::Function(header));
            }
        }
    }

    fn insert_func_headers_by_range(&mut self, range: &Spanned<ImportRange>) {
        let paths = range.kind.to_paths();
        for path in paths {
            if let Some(entries) = self.generate_from_range_path(&range.span, &path) {
                for entry in entries {
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

    fn insert_func_headers_in_stmts(&mut self, code: &mut BytecodeBuilder, funcs: &[AstFunction]) {
        // Insert function headers
        for func in funcs {
            if self.variables.contains_key(&func.name.kind) {
                error!(
                    self,
                    func.name.span.clone(),
                    "A function or variable with the same name exists"
                );
                continue;
            }

            if let Some((header, func)) = self.generate_function_header(&func) {
                let func_name = func.name;
                code.insert_function_header(func);

                let func_id = code.get_function(func_name).unwrap().code_id;
                let fh = FunctionHeaderWithId::new_self(header, func_id);
                self.variables.insert(func_name, Entry::Function(fh));
            }
        }
    }

    fn walk_block(&mut self, code: &mut BytecodeBuilder, block: Block) -> Option<(Type, InstList)> {
        self.insert_extern_module_headers(&block.imports);
        self.insert_type_headers_in_stmts(&block.types);
        self.walk_type_def_in_stmts(block.types);

        self.resolve_type_sizes();

        self.insert_func_headers_in_stmts(code, &block.functions);

        // Walk the statements
        let mut insts = InstList::new();
        for stmt in block.stmts {
            let stmt_insts = self.walk_stmt(code, stmt);
            if let Some(stmt_insts) = stmt_insts {
                insts.append(stmt_insts);
            }
        }

        let result_expr = self.walk_expr(code, *block.result_expr);

        // Walk the function bodies in the statements
        for func in block.functions {
            self.walk_function(code, func);
        }

        let result_expr = result_expr?;
        insts.append(result_expr.insts);

        Some((result_expr.ty, insts))
    }

    // =======================================
    // Type Definition
    // =======================================

    fn resolve_type_sizes(&mut self) {
        for map in self.types.maps.iter_mut() {
            for ty in map.values_mut() {
                let oty = mem::replace(ty, Type::Int);
                *ty = resolve_type_sizes(&self.type_sizes, oty);
            }
        }

        for map in self.tycons.maps.iter_mut() {
            for tycon in map.values_mut() {
                if let Some(tycon) = tycon {
                    let otc = mem::replace(tycon, TypeCon::Tuple);
                    *tycon = resolve_type_sizes_in_tycon(&self.type_sizes, otc);
                }
            }
        }
    }

    fn walk_type(&mut self, ty: Spanned<AstType>) -> Option<Type> {
        match ty.kind {
            AstType::Int => Some(Type::Int),
            AstType::Bool => Some(Type::Bool),
            AstType::Unit => Some(Type::Unit),
            AstType::String => Some(Type::String),
            AstType::Named(name) => {
                if let Some(ty) = self.types.get(&name) {
                    Some(ty.clone())
                } else if self.tycons.contains_key(&name) {
                    Some(Type::App(
                        TypeCon::Named(name, *self.type_sizes.get(&name).unwrap_or(&246)),
                        Vec::new(),
                    ))
                } else {
                    error!(self, ty.span, "undefined type `{}`", IdMap::name(name));
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

                if !self.tycons.contains_key(&name.kind) {
                    error!(
                        self,
                        name.span,
                        "undefined type `{}`",
                        IdMap::name(name.kind)
                    );
                    return None;
                }

                let mut new_types = Vec::with_capacity(types.len());
                for ty in types {
                    let ty = self.walk_type(ty)?;
                    new_types.push(ty);
                }

                self.pop_type_scope();

                Some(Type::App(
                    TypeCon::Named(name.kind, *self.type_sizes.get(&name.kind).unwrap_or(&245)),
                    new_types,
                ))
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
        self.tycons.insert(tydef.name, None);
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

        let size = self.type_size_err(tydef.ty.span, &ty);

        let tycon = TypeCon::Fun(vars, Box::new(ty));
        let uniq = self.next_unique;
        self.next_unique += 1;

        self.pop_type_scope();

        self.tycons
            .insert(tydef.name, Some(TypeCon::Unique(Box::new(tycon), uniq)));
        self.type_sizes.insert(tydef.name, size);
    }

    // =================================
    //  Function Definition
    // =================================

    // Insert parameters and return value as variables and return instructions that copy the
    // parameters to heap
    fn insert_params_as_variables(
        &mut self,
        func: &mut Function,
        params: Vec<Param>,
        return_ty: &Type,
    ) -> Option<InstList> {
        let mut insts = InstList::new();

        let mut loc = -5isize; // fp, ip, mid, fid, ep
        for param in params.into_iter().rev() {
            loc -= type_size_nocheck(&param.ty) as isize;

            let loc = if param.is_escaped {
                // Copy the parameter to heap if escaped
                let heap_loc = func.stack_in_heap_size as usize + 1;

                insts.append(translate::escaped_param(&param.ty, loc, heap_loc));
                func.stack_in_heap_size += type_size_nocheck(&param.ty) as u8;

                VariableLoc::StackInHeap(heap_loc, self.var_level)
            } else {
                VariableLoc::Stack(loc)
            };

            // Insert the parameter as a variable to the current scope
            let var = Entry::Variable(Variable::new(param.ty.clone(), param.is_mutable, loc));
            self.variables.insert(param.name, var);
        }

        // Insert a variable that will be stored a return value
        loc -= type_size_nocheck(return_ty) as isize;

        self.variables.insert(
            *reserved_id::RETURN_VALUE,
            Entry::Variable(Variable::new(
                return_ty.clone(),
                false,
                VariableLoc::Stack(loc),
            )),
        );

        Some(insts)
    }

    fn generate_function_header(
        &mut self,
        func: &AstFunction,
    ) -> Option<(FunctionHeader, Function)> {
        self.push_type_scope();

        let mut vars = Vec::new();
        for var_id in &func.ty_params {
            vars.push((var_id.kind, TypeVar::with_id(var_id.kind)));
            self.types
                .insert(var_id.kind, Type::Var(vars.last().unwrap().1));
        }

        let mut param_types = Vec::new();
        let mut param_size = 0;
        for AstParam { ty, .. } in &func.params {
            let ty_span = ty.span.clone();
            let mut ty = match self.walk_type(ty.clone()) {
                Some(ty) => ty,
                None => return None,
            };
            wrap_typevar(&mut ty);

            param_size += self.type_size_err(ty_span, &ty);
            param_types.push(ty);
        }

        let return_ty_span = func.return_ty.span.clone();
        let mut return_ty = match self.walk_type(func.return_ty.clone()) {
            Some(ty) => ty,
            None => return None,
        };

        wrap_typevar(&mut return_ty);

        self.pop_type_scope();

        self.type_size_err(return_ty_span, &return_ty);

        // Insert a header of the function
        let header = FunctionHeader {
            params: param_types,
            return_ty,
            ty_params: vars,
        };

        // Create a function for the bytecode
        let bc_func = Function::new(func.name.kind, param_size);

        Some((header, bc_func))
    }

    fn walk_function(&mut self, code: &mut BytecodeBuilder, func: AstFunction) {
        self.current_func = func.name.kind;

        self.push_scope();
        self.push_type_scope();
        self.var_level += 1;

        let header = match self.variables.get(&func.name.kind) {
            Some(Entry::Function(h)) => &h.header,
            // To reach there means the function header may be not inserted by errors
            _ => return,
        };

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

        let code_func = code.get_function_mut(func.name.kind).unwrap();
        let param_insts = match self.insert_params_as_variables(code_func, params, &return_ty) {
            Some(insts) => insts,
            None => return,
        };

        let body = match self.walk_expr(code, func.body) {
            Some(e) => e,
            None => return,
        };

        unify(&mut self.errors, &body.span, &return_ty, &body.ty);

        let return_var = self.get_return_var();
        let body_insts =
            translate::return_stmt(&self.relative_loc(&return_var.loc), Some(body), &return_ty);

        let mut insts = param_insts;
        insts.append(body_insts);

        self.function_insts.push_back((func.name.kind, insts));

        self.var_level -= 1;
        self.pop_type_scope();
        self.pop_scope();
    }

    // =========================================
    // Module
    // =========================================

    pub fn load_modules(
        &mut self,
        code: &mut BytecodeBuilder,
        imported_modules: Vec<SymbolPath>,
        func_headers: &FxHashMap<SymbolPath, FxHashMap<Id, (u16, FunctionHeader)>>,
    ) -> FxHashMap<SymbolPath, (u16, ModuleHeader)> {
        let mut headers = FxHashMap::default();
        for path in imported_modules {
            let func_headers = func_headers[&path].clone();

            let module_header = ModuleHeader {
                path: path.clone(),
                functions: func_headers,
            };

            let module_id = code.push_module(&format!("{}", path));
            headers.insert(path, (module_id, module_header));
        }

        headers
    }

    pub fn get_public_function_headers(
        &mut self,
        code: &mut BytecodeBuilder,
        program: &Program,
    ) -> FxHashMap<Id, (u16, FunctionHeader)> {
        let mut function_headers = FxHashMap::default();

        self.push_type_scope();

        // Insert type headers
        for tydef in &program.main.types {
            self.insert_type_header(tydef);
        }

        // Walk the type definitions
        for tydef in &program.main.types {
            self.walk_type_def(tydef.clone());
        }

        self.resolve_type_sizes();

        // Insert main function header
        let header = FunctionHeader {
            params: Vec::new(),
            return_ty: Type::Unit,
            ty_params: Vec::new(),
        };
        code.insert_function_header(Function::new(*reserved_id::MAIN_FUNC, 0));
        function_headers.insert(
            *reserved_id::MAIN_FUNC,
            (
                code.get_function(*reserved_id::MAIN_FUNC).unwrap().code_id,
                header,
            ),
        );

        for func in &program.main.functions {
            if let Some((header, func)) = self.generate_function_header(&func) {
                let func_name = func.name;

                code.insert_function_header(func);
                function_headers.insert(
                    func_name,
                    (code.get_function(func_name).unwrap().code_id, header),
                );
            }
        }

        self.pop_type_scope();

        function_headers
    }

    pub fn analyze(
        mut self,
        mut code: BytecodeBuilder,
        program: Program,
        func_headers: &FxHashMap<SymbolPath, FxHashMap<Id, (u16, FunctionHeader)>>,
    ) -> Result<Bytecode, Vec<Error>> {
        assert!(self.module_headers.is_empty());

        self.push_scope();
        self.push_type_scope();

        self.module_headers = self.load_modules(&mut code, program.imported_modules, func_headers);

        let range = ImportRange::Scope(*reserved_id::STD_MODULE, Box::new(ImportRange::All));
        self.insert_func_headers_by_range(&Spanned {
            kind: range,
            span: program.main.result_expr.span.clone(),
        });

        // Main statements
        let insts = self.walk_block(&mut code, program.main);
        if let Some((_, insts)) = insts {
            self.function_insts
                .push_front((*reserved_id::MAIN_FUNC, insts));
        }

        self.pop_scope();
        self.pop_type_scope();

        for (name, insts) in self.function_insts {
            code.push_function_body(name, insts);
        }

        if !self.errors.is_empty() {
            Err(self.errors)
        } else {
            Ok(code.build(&program.strings))
        }
    }
}

pub enum ModuleBody {
    Normal(Bytecode),
    Native(Module),
}

pub fn analyze_semantics(
    module_buffers: FxHashMap<SymbolPath, Program>,
    native_modules: &ModuleContainer,
) -> Result<FxHashMap<String, ModuleBody>, Vec<Error>> {
    type FunctionHeaders = FxHashMap<Id, (u16, FunctionHeader)>;
    let mut func_headers: FxHashMap<SymbolPath, FunctionHeaders> = FxHashMap::default();
    let mut bytecode_builders: FxHashMap<SymbolPath, BytecodeBuilder> = FxHashMap::default();
    let mut analyzers: FxHashMap<SymbolPath, Analyzer> = FxHashMap::default();
    let mut bodies: FxHashMap<String, ModuleBody> = FxHashMap::default();

    let mut imported_native_modules: FxHashSet<SymbolPath> = FxHashSet::default();

    // Insert all function headers
    for (name, program) in &module_buffers {
        analyzers.insert(name.clone(), Analyzer::new());
        bytecode_builders.insert(name.clone(), BytecodeBuilder::new());
        let headers = analyzers
            .get_mut(name)
            .unwrap()
            .get_public_function_headers(bytecode_builders.get_mut(name).unwrap(), program);
        func_headers.insert(name.clone(), headers);

        // Native modules
        for path in &program.imported_modules {
            if let Some(module) = native_modules.get(path) {
                let headers = &module.header.functions;
                func_headers.insert(path.clone(), headers.clone());
                imported_native_modules.insert(path.clone());
            }
        }
    }

    let mut errors = Vec::new();

    for (name, program) in module_buffers {
        let analyzer = analyzers.remove(&name).unwrap();
        let builder = bytecode_builders.remove(&name).unwrap();
        let bytecode = match analyzer.analyze(builder, program, &func_headers) {
            Ok(b) => b,
            Err(mut new_errors) => {
                errors.append(&mut new_errors);
                continue;
            }
        };
        bodies.insert(format!("{}", name), ModuleBody::Normal(bytecode));
    }

    for path in &imported_native_modules {
        let module = native_modules.get(path).unwrap().module.clone();
        bodies.insert(format!("{}", path), ModuleBody::Native(module));
    }

    if !errors.is_empty() {
        Err(errors)
    } else {
        Ok(bodies)
    }
}
