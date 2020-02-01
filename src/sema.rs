use std::collections::LinkedList;
use std::mem;

use rustc_hash::FxHashMap;

use crate::ty::*;
use crate::ast::{*, Param as AstParam};
use crate::error::Error;
use crate::span::{Span, Spanned};
use crate::id::{Id, IdMap, reserved_id};
use crate::bytecode::{Bytecode, Function, opcode, BytecodeBuilder, InstList};
use crate::module::{FunctionHeader, ModuleHeader};
use crate::translate;
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
                    let msg = format!(concat!("expected type `", $type_name, "` but got type `{}`"), ty);
                    let error = Error::new(&msg, span);
                    errors.push(error);
                    None
                },
            }
        }
    };
}

fn_to_expect! {
    expect_tuple, "tuple", Vec<Type>,
    Type::App(TypeCon::Tuple, types) => Some(types),
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
}

#[derive(Debug)]
struct Variable {
    ty: Type,
    is_mutable: bool,
    loc: isize,
}

impl Variable {
    fn new(ty: Type, is_mutable: bool, loc: isize) -> Self {
        Self {
            ty,
            is_mutable,
            loc,
        }
    }
}

#[derive(Debug)]
pub struct Analyzer<'a> {
    function_headers: FxHashMap<Id, FunctionHeader>,
    module_headers: FxHashMap<SymbolPath, (u16, ModuleHeader)>,
    types: HashMapWithScope<Id, Type>,
    tycons: HashMapWithScope<Id, Option<TypeCon>>,
    type_sizes: HashMapWithScope<Id, usize>,
    variables: HashMapWithScope<Id, Variable>,
    errors: Vec<Error>,
    current_func: Id,
    next_temp_num: u32,
    next_unique: u32,
    function_insts: LinkedList<(Id, InstList)>,
    _phantom: &'a std::marker::PhantomData<Self>,
}

impl<'a> Analyzer<'a> {
    pub fn new() -> Self {
        Self {
            function_headers: FxHashMap::default(),
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

    // Insert parameters and return value as variables to `self.variables`
    fn insert_params(&mut self, params: Vec<Param>, return_ty: &Type) -> Option<()> {
        let mut loc = -4isize; // fp, ip
        for Param { name, ty, is_mutable } in params.into_iter().rev() {
            loc -= type_size_nocheck(&ty) as isize;

            // Insert the parameter as a variable to the current scope
            self.variables.insert(name, Variable::new(ty.clone(), is_mutable, loc));
        }

        loc -= type_size_nocheck(return_ty) as isize;

        self.variables.insert(*reserved_id::RETURN_VALUE, Variable::new(return_ty.clone(), false, loc));

        Some(())
    }

    fn get_return_var(&self) -> &Variable {
        self.find_var(*reserved_id::RETURN_VALUE).unwrap()
    }

    fn expand_name(&self, ty: Type) -> Option<Type> {
        match ty {
            Type::App(TypeCon::Fun(params, body), types) => {
                let map = params.into_iter().zip(types.into_iter()).collect();
                self.expand_name(subst(*body, &map))
            },
            Type::App(TypeCon::Named(name, _), types) => {
                match self.tycons.get(&name) {
                    Some(tycon) => Some(Type::App(tycon.clone().unwrap(), types)),
                    None => None,
                }
            },
            Type::App(tycon, types) => {
                let mut new_types = Vec::with_capacity(types.len());
                for ty in types {
                    let ty = self.expand_name(ty)?;
                    new_types.push(ty);
                }

                Some(Type::App(tycon, new_types))
            },
            ty => Some(ty),
        }
    }

    #[inline]
    fn type_size_err(&mut self, span: Span, ty: &Type) -> usize {
        match type_size(ty) {
            Some(size) => size,
            None => {
                self.errors.push(Error::new(&format!("the size of type `{}` cannot be calculated", ty), span));
                0
            },
        }
    }

    // ====================================
    //  Variable
    // ====================================

    fn new_var(&mut self, current_func: &mut Function, id: Id, ty: Type, is_mutable: bool) -> isize {
        let new_var_size = type_size_nocheck(&ty);

        let last_map = self.variables.last_scope().unwrap();
        let loc = match last_map.get(&id) {
            // If the same scope contains the same size variable, use the variable location
            Some(var) if new_var_size == type_size_nocheck(&var.ty) => {
                var.loc
            },
            _ => {
                let loc = current_func.stack_size as isize;
                current_func.stack_size += new_var_size as u8;
                loc
            },
        };

        self.variables.insert(id, Variable::new(ty.clone(), is_mutable, loc));

        loc
    }

    #[inline]
    fn new_var_in_current_func(&mut self, code: &mut BytecodeBuilder, id: Id, ty: Type, is_mutable: bool) -> isize {
        self.new_var(code.get_function_mut(self.current_func).unwrap(), id, ty, is_mutable)
    }

    fn gen_temp_id(&mut self) -> Id {
        let id = IdMap::new_id(&format!("$comp{}", self.next_temp_num));
        self.next_temp_num += 1;
        id
    }

    fn find_var(&self, id: Id) -> Option<&Variable> {
        self.variables.get(&id)
    }

    fn find_func(&self, code: &BytecodeBuilder, path: &SymbolPath) -> Option<(&FunctionHeader, u16, Option<u16>)> { // header, code id, module id
        assert!(!path.is_empty());

        let func_id = path.tail().unwrap().id;
        if let Some(fh) = self.function_headers.get(&func_id) {
            return Some((fh, code.get_function(func_id).unwrap().code_id, None));
        }

        let module_path = path.parent()?;
        let (module_id, module) = self.module_headers.get(&module_path)?;

        if let Some((func_id, fh)) = module.functions.get(&func_id) {
            return Some((fh, *func_id, Some(*module_id)));
        }

        None
    }

    // Pointer
    //   Struct => ok
    //   _ => error
    // Struct => ok
    // _ => error
    fn get_struct_fields<'b>(&'b mut self, ty: &'b Type, span: &Span, is_mutable: bool)
        -> Option<(Vec<(Id, Type)>, bool)>
    {
        let ty = expand_unique(ty.clone());
        match ty {
            Type::App(TypeCon::Struct(fields), tys) => Some((fields.into_iter().zip(tys.into_iter()).collect(), is_mutable)),
            Type::App(TypeCon::Pointer(is_mutable), tys) => {
                let ty = expand_unique(tys[0].clone());
                match ty {
                    Type::App(TypeCon::Struct(fields), tys) => {
                        Some((fields.into_iter().zip(tys.into_iter()).collect(), is_mutable))
                    },
                    ty => {
                        error!(self, span.clone(), "expected type `struct` or `*struct` but got type `{}`", ty);
                        None
                    },
                }
            },
            ty => {
                error!(self, span.clone(), "expected type `struct` or `*struct` but got type `{}`", ty);
                None
            },
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
            Expr::Call(_, _, _) => true,
            _ => false,
        }
    }

    fn insert_type_param(param_ty: &Type, ty: &Type, map: &mut FxHashMap<TypeVar, Type>) {
        match (param_ty, ty) {
            (Type::App(ptycon, ptypes), Type::App(atycon, atypes)) if ptycon == atycon => {
                for (pty, aty) in ptypes.iter().zip(atypes.iter()) {
                    Self::insert_type_param(pty, aty, map);
                }
            },
            (Type::Var(var), ty) if !map.contains_key(var) => {
                map.insert(*var, ty.clone());
            },
            _ => {},
        }
    }

    fn walk_expr_with_conversion(&mut self, code: &mut BytecodeBuilder, expr: Spanned<Expr>, ty: &Type) -> Option<ExprInfo> {
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
            },
            _ => match expr.kind {
                Expr::Tuple(exprs) => {
                    let types = match ty {
                        Type::App(TypeCon::Tuple, types) => types,
                        _ => return None,
                    };

                    if types.len() != exprs.len() {
                        error!(self, expr.span, "error 1");
                        return None;
                    }

                    let mut insts = InstList::new();
                    let mut tys = Vec::new();
                    for (expr, ty) in exprs.into_iter().zip(types.iter()) {
                        let expr = self.walk_expr_with_conversion(code, expr, ty);
                        if let Some(expr) = expr {
                            tys.push(expr.ty.clone());
                            insts.append(translate::literal_tuple(expr));
                        }
                    }

                    Some(ExprInfo::new(insts, Type::App(TypeCon::Tuple, tys), expr.span))
                },
                _ => {
                    let mut expr = self.walk_expr(code, expr)?;

                    // if `ty` is not wrapped and `expr.ty` is wrapped
                    if let Type::App(TypeCon::Wrapped, _) = ty {
                    } else {
                        match &expr.ty {
                            Type::App(TypeCon::Wrapped, types) => {
                                expr.insts = translate::unwrap(expr.insts, &expr.ty);
                                expr.ty = types[0].clone();
                            },
                            _ => {},
                        };
                    }

                    Some(expr)
                }
            },
        }
    }

    fn walk_expr_with_unwrap(&mut self, code: &mut BytecodeBuilder, expr: Spanned<Expr>) -> Option<ExprInfo> {
        let mut expr = self.walk_expr(code, expr)?;

        match &expr.ty {
            Type::App(TypeCon::Wrapped, types) => {
                expr.insts = translate::unwrap(expr.insts, &expr.ty);
                expr.ty = types[0].clone();
            },
            _ => {},
        };

        Some(expr)
    }

    fn walk_expr(&mut self, code: &mut BytecodeBuilder, expr: Spanned<Expr>) -> Option<ExprInfo> {
        let (insts, ty) = match expr.kind {
            Expr::Literal(Literal::Number(n)) => {
                (translate::literal_int(n), Type::Int)
            },
            Expr::Literal(Literal::String(i)) => {
                let ty = Type::App(TypeCon::Pointer(false), vec![Type::String]);
                (translate::literal_str(i), ty)
            },
            Expr::Literal(Literal::Unit) => {
                (translate::literal_unit(), Type::Unit)
            },
            Expr::Literal(Literal::True) => {
                (translate::literal_true(), Type::Bool)
            },
            Expr::Literal(Literal::False) => {
                (translate::literal_false(), Type::Bool)
            },
            Expr::Literal(Literal::Null) => {
                (translate::literal_null(), Type::Null)
            },
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
            },
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
                    Type::App(TypeCon::Struct(fields), tys) => fields.into_iter().zip(tys.into_iter()).collect(),
                    ty => {
                        error!(self, expr.span.clone(), "type `{}` is not struct", ty);
                        return None;
                    },
                };

                let mut insts = InstList::new();
                let mut map = FxHashMap::default();

                // Push instructions to `insts` in order
                for i in 0..fields.len() {
                    let field_expr = field_exprs.iter().find(|(id, _)| id.kind == fields[i].0).map(|(_, expr)| expr);
                    if let Some(expr) = field_expr {
                        if let Some(expr) = self.walk_expr_with_conversion(code, expr.clone(), &fields[i].1) {
                            Self::insert_type_param(&fields[i].1, &expr.ty, &mut map);
                            for (_, ty) in &mut fields {
                                *ty = subst(ty.clone(), &map);
                            }

                            unify(&mut self.errors, &expr.span, &fields[i].1, &expr.ty);
                            insts.append(translate::literal_struct_field(expr));
                        }
                    } else {
                        error!(self, expr.span.clone(), "missing field `{}`", IdMap::name(fields[i].0));
                    }
                }

                let args: Vec<Type> = ty_params.into_iter()
                    .map(|var| map.get(&var).cloned().unwrap_or(Type::Var(var)))
                    .collect();
                match &mut expr_ty {
                    Type::App(_, types) => {
                        if types.len() < args.len() {
                            for arg in args.into_iter().skip(types.len()) {
                                types.push(arg);
                            }
                        } else if types.len() > args.len() {
                            panic!();
                        }
                    },
                    _ => panic!(),
                }

                (insts, expr_ty)
            },
            Expr::Array(expr, size) => {
                let expr = self.walk_expr(code, *expr)?;
                let ty = expr.ty.clone();

                (
                    translate::literal_array(expr, size),
                    Type::App(TypeCon::Array(size), vec![ty]),
                )
            },
            Expr::Field(comp_expr, field) => {
                let should_store = Self::expr_push_multiple_values(&comp_expr.kind);
                let comp_expr = self.walk_expr(code, *comp_expr)?;
                let mut is_mutable = comp_expr.is_mutable;

                let loc = if should_store {
                    let id = self.gen_temp_id();
                    Some(self.new_var_in_current_func(code, id, comp_expr.ty.clone(), false))
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
                            },
                            ty => expect_tuple(&mut self.errors, ty, comp_expr.span.clone())?,
                        };
                        
                        match types.get(i) {
                            Some(ty) => {
                                let offset = types.iter().take(i).fold(0, |acc, ty| acc + type_size_nocheck(ty));
                                (ty.clone(), offset)
                            },
                            None => {
                                error!(self, expr.span, "error");
                                return None;
                            },
                        }
                    },
                    Field::Id(name) => {
                        let (fields, is_mutable_) = self.get_struct_fields(&ty, &comp_expr.span, is_mutable)?;
                        is_mutable = is_mutable_;

                        let i = match fields.iter().position(|(id, _)| *id == name) {
                            Some(i) => i,
                            None => {
                                error!(self, expr.span, "no field in `{}`: `{}`", comp_expr.ty, IdMap::name(name));
                                return None;
                            },
                        };

                        let offset = fields.iter().take(i).fold(0, |acc, (_, ty)| acc + type_size_nocheck(ty));
                        (fields[i].1.clone(), offset)
                    }
                };

                let insts = translate::field(loc, should_deref, comp_expr, offset);
                let ty = field_ty.clone();
                return Some(ExprInfo::new_lvalue(insts, ty, expr.span, is_mutable));
            },
            Expr::Subscript(expr, subscript_expr) => {
                let should_store = Self::expr_push_multiple_values(&expr.kind);

                let expr = self.walk_expr_with_unwrap(code, *expr);
                let subscript_expr = self.walk_expr_with_conversion(code, *subscript_expr, &Type::Int);
                try_some!(expr, subscript_expr);

                let mut expr = expr;

                let loc = if should_store {
                    let id = self.gen_temp_id();
                    Some(self.new_var_in_current_func(code, id, expr.ty.clone(), false))
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
                                error!(self, expr.span.clone(), "expected array but got type `{}`", ty);
                                return None;
                            },
                        }
                    }
                    ty => {
                        error!(self, expr.span.clone(), "expected array but got type `{}`", ty);
                        return None;
                    },
                };

                unify(&mut self.errors, &subscript_expr.span, &subscript_expr.ty, &Type::Int);

                let span = expr.span.clone();
                let is_lvalue = expr.is_lvalue;
                let is_mutable = expr.is_mutable;
                let insts = translate::subscript(
                    loc,
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
            },
            Expr::Variable(name) => {
                let var = match self.find_var(name) {
                    Some(v) => v,
                    None => {
                        self.add_error("undefined variable", expr.span.clone());
                        return None;
                    },
                };

                let insts = translate::variable(var.loc);
                return Some(ExprInfo::new_lvalue(insts, var.ty.clone(), expr.span, var.is_mutable));
            },
            Expr::BinOp(BinOp::And, lhs, rhs) => {
                let lhs = self.walk_expr_with_conversion(code, *lhs, &Type::Int);
                let rhs = self.walk_expr_with_conversion(code, *rhs, &Type::Int);
                try_some!(lhs, rhs);

                // Type check
                match (&lhs.ty, &rhs.ty) {
                    (Type::Bool, Type::Bool) => {},
                    (lty, rty) => {
                        error!(self, expr.span.clone(), "{} && {}", lty, rty);
                    },
                }

                (translate::binop_and(lhs, rhs), Type::Bool)
            },
            Expr::BinOp(BinOp::Or, lhs, rhs) => {
                let lhs = self.walk_expr_with_conversion(code, *lhs, &Type::Int);
                let rhs = self.walk_expr_with_conversion(code, *rhs, &Type::Int);
                try_some!(lhs, rhs);

                // Type check
                match (&lhs.ty, &rhs.ty) {
                    (Type::Bool, Type::Bool) => {},
                    (lty, rty) => {
                        error!(self, expr.span.clone(), "{} || {}", lty, rty);
                    },
                }

                (translate::binop_or(lhs, rhs), Type::Bool)
            },
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

                    (BinOp::Equal, Type::App(TypeCon::Pointer(_), _), Type::App(TypeCon::Pointer(_), _)) => Type::Bool,
                    (BinOp::Equal, Type::Null, Type::App(TypeCon::Pointer(_), _)) => Type::Bool,
                    (BinOp::Equal, Type::App(TypeCon::Pointer(_), _), Type::Null) => Type::Bool,
                    (BinOp::NotEqual, Type::App(TypeCon::Pointer(_), _), Type::App(TypeCon::Pointer(_), _)) => Type::Bool,
                    (BinOp::NotEqual, Type::Null, Type::App(TypeCon::Pointer(_), _)) => Type::Bool,
                    (BinOp::NotEqual, Type::App(TypeCon::Pointer(_), _), Type::Null) => Type::Bool,
                    _ => {
                        self.add_error(&format!("`{} {} {}`", lhs.ty, binop_symbol, rhs.ty), expr.span.clone());
                        return None;
                    }
                };

                (translate::binop(binop, lhs, rhs), ty)
            },
            Expr::Call(path, args, ty_args) => {
                let (callee_func, code_id, module_id) = match self.find_func(code, &path.kind) {
                    Some(func) => func,
                    None => {
                        error!(self, expr.span.clone(), "undefined function");
                        return None;
                    },
                };

                let ty_params = callee_func.ty_params.clone();
                let mut return_ty = callee_func.return_ty.clone();
                let mut params = callee_func.params.clone();

                // map <- { ty_params_i -> ty_args_i }
                let iter = ty_params
                    .iter()
                    .map(|(_, var)| var)
                    .copied()
                    .zip(ty_args.iter().cloned());
                let mut map = FxHashMap::default();

                for (var, ty) in iter {
                    let ty = self.walk_type(ty)?;
                    map.insert(var, ty);
                }

                fn subst_param(params: &mut Vec<Type>, return_ty: &mut Type, map: &FxHashMap<TypeVar, Type>) {
                    // Substitute type arguments for return types
                    *return_ty = subst(return_ty.clone(), &map);

                    // Substitute type arguments for parameter types
                    for param in params {
                        *param = subst(param.clone(), &map);
                    }
                }

                subst_param(&mut params, &mut return_ty, &map);

                // Check parameter length
                if args.len() != params.len() {
                    error!(self, expr.span.clone(),
                        "the function takes {} parameters. but got {} arguments",
                        params.len(),
                        args.len());
                    return None;
                }

                // Check argument types
                let mut insts = Vec::new();
                for (i, arg) in args.into_iter().enumerate() {
                    let arg = self.walk_expr_with_conversion(code, arg, &params[i]);
                    if let Some(arg) = arg {
                        Self::insert_type_param(&params[i], &arg.ty, &mut map);
                        subst_param(&mut params, &mut return_ty, &map);
                        unify(&mut self.errors, &arg.span, &arg.ty, &params[i]);

                        insts.push(arg);
                    }
                }

                let insts = translate::call(code_id, module_id, insts, &return_ty);
                (insts, return_ty)
            },
            Expr::Address(expr, is_mutable) => {
                let expr = self.walk_expr_with_unwrap(code, *expr)?;

                if is_mutable && !expr.is_mutable {
                    error!(self, expr.span, "this expression is immutable");
                    return None;
                } 

                let ty = Type::App(TypeCon::Pointer(is_mutable), vec![expr.ty.clone()]);

                if !expr.is_lvalue {
                    let temp = self.gen_temp_id();
                    let loc = self.new_var_in_current_func(code, temp, expr.ty.clone(), is_mutable);

                    let insts = translate::address_no_lvalue(expr, loc);
                    (insts, ty)
                } else {
                    let insts = translate::address(expr);
                    (insts, ty)
                }
            },
            Expr::Dereference(expr_) => {
                let expr_ = self.walk_expr_with_unwrap(code, *expr_)?;

                match expr_.ty.clone() {
                    Type::App(TypeCon::Pointer(is_mutable), tys) => {
                        let insts = translate::dereference(expr_);
                        return Some(ExprInfo::new_lvalue(insts, tys[0].clone(), expr.span, is_mutable));
                    }
                    ty => {
                        error!(self, expr.span, "expected type `pointer` but got type `{}`", ty);
                        return None;
                    }
                }
            },
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
            },
            Expr::Alloc(expr, is_mutable) => {
                let expr = self.walk_expr_with_unwrap(code, *expr)?;

                let ty = Type::App(TypeCon::Pointer(is_mutable), vec![expr.ty.clone()]);
                let insts = translate::alloc(expr);
                (insts, ty)
            },
        };

        Some(ExprInfo::new(insts, ty, expr.span))
    }

    fn walk_stmt(&mut self, code: &mut BytecodeBuilder, stmt: Spanned<Stmt>) -> Option<InstList> {
        let insts = match stmt.kind {
            Stmt::Expr(expr) => {
                let expr = self.walk_expr(code, expr)?;

                translate::expr_stmt(expr)
            },
            Stmt::If(cond, stmt, None) => {
                let cond = self.walk_expr_with_conversion(code, cond, &Type::Bool);
                let then_insts = self.walk_stmt(code, *stmt);
                try_some!(cond, then_insts);

                unify(&mut self.errors, &cond.span, &Type::Bool, &cond.ty);

                translate::if_stmt(cond, then_insts)
            },
            Stmt::If(cond, then_stmt, Some(else_stmt)) => {
                let cond = self.walk_expr_with_conversion(code, cond, &Type::Bool);
                let then = self.walk_stmt(code, *then_stmt);
                let els = self.walk_stmt(code, *else_stmt);
                try_some!(cond, then, els);

                unify(&mut self.errors, &cond.span, &Type::Bool, &cond.ty);

                translate::if_else_stmt(cond, then, els)
            },
            Stmt::While(cond, stmt) => {
                let cond = self.walk_expr_with_conversion(code, cond, &Type::Bool);
                let body = self.walk_stmt(code, *stmt);
                try_some!(cond, body);

                unify(&mut self.errors, &cond.span, &Type::Bool, &cond.ty);

                translate::while_stmt(cond, body)
            },
            Stmt::Block(stmts) => {
                self.push_scope();
                let insts = self.walk_stmts_including_def(code, stmts);
                self.pop_scope();

                insts
            },
            Stmt::Bind(name, expr, is_mutable) => {
                if self.function_headers.contains_key(&name) {
                    error!(self, stmt.span, "the same function as this variable name exists");
                    return None;
                }

                let expr = self.walk_expr(code, expr)?;
                let loc = self.new_var_in_current_func(code, name, expr.ty.clone(), is_mutable);

                translate::bind_stmt(loc, expr)
            },
            Stmt::Assign(lhs, rhs) => {
                let lhs = self.walk_expr(code, lhs)?;
                let rhs = self.walk_expr_with_conversion(code, rhs, &lhs.ty)?;

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
            },
            Stmt::Return(expr) => {
                let func_name = code.get_function(self.current_func).unwrap().name;

                // Check if is outside function
                if func_name == *reserved_id::MAIN_FUNC {
                    error!(self, stmt.span, "return statement outside function");
                    return None;
                }

                let return_var = self.get_return_var();
                let ty = return_var.ty.clone();
                let loc = return_var.loc;

                let expr = match expr {
                    Some(expr) => Some(self.walk_expr_with_conversion(code, expr, &ty)?),
                    None => None,
                };

                // Check type
                let return_ty = expr.as_ref().map_or(&Type::Unit, |expr| &expr.ty);
                unify(&mut self.errors, &stmt.span, &ty, return_ty);

                translate::return_stmt(loc, expr, &ty)
            },
            Stmt::Import(_) => InstList::new(),
            Stmt::FnDef(_) => InstList::new(),
            Stmt::TypeDef(_) => InstList::new(),
        };

        Some(insts)
    }

    fn walk_stmts_including_def(&mut self, code: &mut BytecodeBuilder, stmts: Vec<Spanned<Stmt>>) -> InstList {
        // Insert type headers
        for stmt in &stmts {
            if let Stmt::TypeDef(tydef) = &stmt.kind {
                self.insert_type_header(tydef);
            }
        }

        // Walk the type definitions
        for stmt in &stmts {
            if let Stmt::TypeDef(tydef) = &stmt.kind {
                self.walk_type_def(tydef.clone()); // TODO: Avoid clone()
            }
        }

        self.resolve_type_sizes();

        // Insert function headers
        'l: for stmt in &stmts {
            if let Stmt::FnDef(func) = &stmt.kind {
                if self.function_headers.contains_key(&func.name) {
                    error!(self, stmt.span.clone(), "this function name exists already");
                    continue;
                }

                if let Some((header, func)) = self.generate_function_header(&func) {
                    self.function_headers.insert(func.name, header);
                    code.insert_function_header(func);
                }
            }
        }

        // Walk the statements
        let mut insts = InstList::new();
        for stmt in &stmts {
            let stmt_insts = self.walk_stmt(code, stmt.clone()); // TODO: Avoid clone()
            if let Some(stmt_insts) = stmt_insts {
                insts.append(stmt_insts);
            }
        }

        // Walk the function bodies in the statements
        for stmt in stmts {
            if let Stmt::FnDef(func) = stmt.kind {
                self.walk_function(code, *func);
            }
        }

        insts
    }

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
                    Some(Type::App(TypeCon::Named(name, *self.type_sizes.get(&name).unwrap_or(&246)), Vec::new()))
                } else {
                    error!(self, ty.span, "undefined type `{}`", IdMap::name(name));
                    None
                }
            },
            AstType::Pointer(ty, is_mutable) => Some(Type::App(TypeCon::Pointer(is_mutable), vec![self.walk_type(*ty)?])),
            AstType::Array(ty, size) => Some(Type::App(TypeCon::Array(size), vec![self.walk_type(*ty)?])),
            AstType::Tuple(types) => {
                let mut new_types = Vec::new();
                for ty in types {
                    new_types.push(self.walk_type(ty)?);
                }

                Some(Type::App(TypeCon::Tuple, new_types))
            },
            AstType::Struct(fields) => {
                let mut field_names = Vec::new();
                let mut types = Vec::new();
                for (name, ty) in fields {
                    field_names.push(name.kind);
                    types.push(self.walk_type(ty)?);
                }

                Some(Type::App(TypeCon::Struct(field_names), types))
            },
            AstType::App(name, types) => {
                self.push_type_scope();

                if !self.tycons.contains_key(&name.kind) {
                    error!(self, name.span, "undefined type `{}`", IdMap::name(name.kind));
                    return None;
                }

                let mut new_types = Vec::with_capacity(types.len());
                for ty in types {
                    let ty = self.walk_type(ty)?;
                    new_types.push(ty);
                }

                self.pop_type_scope();

                Some(Type::App(TypeCon::Named(name.kind, *self.type_sizes.get(&name.kind).unwrap_or(&245)), new_types))
            },
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
            self.types.insert(var.kind, Type::Var(*vars.last().unwrap()));
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

        self.tycons.insert(tydef.name, Some(TypeCon::Unique(Box::new(tycon), uniq)));
        self.type_sizes.insert(tydef.name, size);
    }

    fn walk_function(&mut self, code: &mut BytecodeBuilder, func: AstFunction) {
        self.current_func = func.name;

        self.push_scope();
        self.push_type_scope();

        let header = match self.function_headers.get(&func.name) {
            Some(fh) => fh,
            None => return, // the function headers may be not inserted by errors
        };

        let return_ty = header.return_ty.clone();

        for (id, var) in &header.ty_params {
            self.types.insert(*id, Type::Var(*var));
        }

        // params
        let params: Vec<Param> = func.params
            .iter()
            .zip(header.params.iter())
            .map(|(ap, ty)| Param {
                name: ap.name,
                ty: ty.clone(),
                is_mutable: ap.is_mutable,
            })
            .collect();
        if self.insert_params(params, &return_ty).is_none() {
            return;
        }

        // `None` is not returned because `func.body` is always a block statement
        let mut insts = self.walk_stmt(code, func.body).unwrap();

        // Push a return instruction if the return value type is unit
        let return_var = self.get_return_var();
        if let Type::Unit = return_var.ty {
            insts.push_inst_noarg(opcode::RETURN);
        }

        self.function_insts.push_back((func.name, insts));

        self.pop_type_scope();
        self.pop_scope();
    }

    // =================================
    //  Header
    // =================================

    fn generate_function_header(&mut self, func: &AstFunction) -> Option<(FunctionHeader, Function)> {
        self.push_type_scope();

        let mut vars = Vec::new();
        for var_id in &func.ty_params {
            vars.push((var_id.kind, TypeVar::with_id(var_id.kind)));
            self.types.insert(var_id.kind, Type::Var(vars.last().unwrap().1));
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
        let bc_func = Function::new(func.name, param_size);

        Some((header, bc_func))
    }

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
                id: path.clone(),
                functions: func_headers,
            };

            let module_id = code.push_module(&format!("{}", path));
            headers.insert(path, (module_id, module_header));
        }

        headers
    }

    pub fn get_public_function_headers(&mut self, code: &mut BytecodeBuilder, program: &Program) -> FxHashMap<Id, (u16, FunctionHeader)> {
        let mut function_headers = FxHashMap::default();

        self.push_type_scope();
        
        // Insert type headers
        for stmt in &program.main_stmts {
            if let Stmt::TypeDef(tydef) = &stmt.kind {
                self.insert_type_header(tydef);
            }
        }

        // Walk the type definitions
        for stmt in &program.main_stmts {
            if let Stmt::TypeDef(tydef) = &stmt.kind {
                self.walk_type_def(tydef.clone()); // TODO: Avoid clone()
            }
        }

        self.resolve_type_sizes();

        // Insert main function header
        let header = FunctionHeader {
            params: Vec::new(),
            return_ty: Type::Unit,
            ty_params: Vec::new(),
        };
        code.insert_function_header(Function::new(*reserved_id::MAIN_FUNC, 0));
        function_headers.insert(*reserved_id::MAIN_FUNC, (code.get_function(*reserved_id::MAIN_FUNC).unwrap().code_id, header));

        for stmt in &program.main_stmts {
            match &stmt.kind {
                Stmt::FnDef(func) => {
                    if let Some((header, func)) = self.generate_function_header(&func) {
                        let func_name = func.name;
                        code.insert_function_header(func);
                        function_headers.insert(func_name, (code.get_function(func_name).unwrap().code_id, header));
                    }
                },
                Stmt::Import(_) => {},
                _ => {}, // TODO: Implement it
            }
        }

        self.pop_type_scope();

        function_headers
    }

    pub fn analyze(
        mut self,
        mut code: BytecodeBuilder,
        program: Program,
        std_module: ModuleHeader,
        func_headers: &FxHashMap<SymbolPath, FxHashMap<Id, (u16, FunctionHeader)>>,
    ) -> Result<Bytecode, Vec<Error>> {
        assert!(self.module_headers.is_empty());
        assert!(self.function_headers.is_empty());

        // Insert main function header
        let header = FunctionHeader {
            params: Vec::new(),
            return_ty: Type::Unit,
            ty_params: Vec::new(),
        };
        self.function_headers.insert(*reserved_id::MAIN_FUNC, header);

        self.push_scope();
        self.push_type_scope();

        let std_module_id = code.push_module(&format!("{}", std_module.id));
        self.module_headers = self.load_modules(&mut code, program.imported_modules, func_headers);
        self.module_headers.insert(std_module.id.clone(), (std_module_id.clone(), std_module));

        // Main statements
        let insts = self.walk_stmts_including_def(&mut code, program.main_stmts);
        self.function_insts.push_front((*reserved_id::MAIN_FUNC, insts));

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

pub fn analyze_semantics(
    module_buffers: FxHashMap<SymbolPath, Program>,
    std_module: ModuleHeader
) -> Result<FxHashMap<String, Bytecode>, Vec<Error>> {
    type FunctionHeaders = FxHashMap<Id, (u16, FunctionHeader)>;
    let mut func_headers: FxHashMap<SymbolPath, FunctionHeaders> = FxHashMap::default();
    let mut bytecode_builders: FxHashMap<SymbolPath, BytecodeBuilder> = FxHashMap::default();
    let mut analyzers: FxHashMap<SymbolPath, Analyzer> = FxHashMap::default();
    let mut bytecodes: FxHashMap<String, Bytecode> = FxHashMap::default();

    for (name, program) in &module_buffers {
        analyzers.insert(name.clone(), Analyzer::new());
        bytecode_builders.insert(name.clone(), BytecodeBuilder::new());
        let headers = analyzers.get_mut(name).unwrap()
            .get_public_function_headers(bytecode_builders.get_mut(name).unwrap(), program);
        func_headers.insert(name.clone(), headers);
    }

    let mut errors = Vec::new();

    for (name, program) in module_buffers {
        let analyzer = analyzers.remove(&name).unwrap();
        let builder = bytecode_builders.remove(&name).unwrap();
        let bytecode = match analyzer.analyze(builder, program, std_module.clone(), &func_headers) {
            Ok(b) => b,
            Err(mut new_errors) => {
                errors.append(&mut new_errors);
                continue;
            },
        };
        bytecodes.insert(format!("{}", name), bytecode);
    }

    if !errors.is_empty() {
        Err(errors)
    } else {
        Ok(bytecodes)
    }
}
