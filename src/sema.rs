use std::io::{Read, Write, Seek};

use rustc_hash::FxHashMap;

use crate::ty::{Type, TypeCon, TypeVar};
use crate::ast::*;
use crate::error::Error;
use crate::span::{Span, Spanned};
use crate::id::{Id, IdMap};
use crate::bytecode::{Function, opcode, BytecodeBuilder, BytecodeStream, InstList};
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

fn subst(ty: Type, map: &FxHashMap<TypeVar, Type>) -> Type {
    match ty {
        Type::Int => Type::Int,
        Type::Bool => Type::Bool,
        Type::String => Type::String,
        Type::Unit => Type::Unit,
        Type::Null => Type::Null,
        Type::Var(var) => {
            match map.iter().find(|(v, _)| var == **v) {
                Some((_, ty)) => ty.clone(),
                None => Type::Var(var),
            }
        },
        Type::App(TypeCon::Fun(params, body), tys) => {
            let mut map_in_func = FxHashMap::default();
            for (param, ty) in params.into_iter().zip(tys.into_iter()) {
                map_in_func.insert(param, ty);
            }

            let body = subst(*body, &map_in_func);
            subst(body, map)
        },
        Type::App(tycon, tys) => {
            let mut new_tys = Vec::with_capacity(tys.len());
            for ty in tys {
                new_tys.push(subst(ty, &map));
            }

            Type::App(tycon, new_tys)
        },
        Type::Poly(vars, ty) => {
            let mut new_map = FxHashMap::default();
            let mut new_vars = Vec::with_capacity(vars.len());
            for var in vars {
                new_vars.push(TypeVar::new());
                new_map.insert(var, Type::Var(*new_vars.last().unwrap()));
            }

            let ty = subst(*ty, &new_map);
            let ty = subst(ty, map);
            Type::Poly(new_vars, Box::new(ty))
        },
    }
}

fn unify(errors: &mut Vec<Error>, span: &Span, a: &Type, b: &Type) -> Option<()> {
    match (a, b) {
        (Type::App(TypeCon::Struct(a_fields), a_tys), Type::App(TypeCon::Struct(b_fields), b_tys))
            if a_fields.len() == b_fields.len() &&
               a_fields.iter().zip(b_fields).all(|(a, b)| a == b) =>
        {
            for (a_ty, b_ty) in a_tys.iter().zip(b_tys.iter()) {
                unify(errors, span, a_ty, b_ty)?;
            }

            return Some(());
        },
        (Type::App(TypeCon::Array(a_size), a_tys), Type::App(TypeCon::Array(b_size), b_tys)) if a_size == b_size => {
            for (a_ty, b_ty) in a_tys.iter().zip(b_tys.iter()) {
                unify(errors, span, a_ty, b_ty)?;
            }

            return Some(());
        },
        (Type::App(TypeCon::Pointer(a_mut), a_tys), Type::App(TypeCon::Pointer(b_mut), b_tys)) if a_mut == b_mut => {
            for (a_ty, b_ty) in a_tys.iter().zip(b_tys.iter()) {
                unify(errors, span, a_ty, b_ty)?;
            }

            return Some(());
        },
        (Type::App(TypeCon::Named(a_name), a_tys), Type::App(TypeCon::Named(b_name), b_tys)) if a_name == b_name => {
            for (a_ty, b_ty) in a_tys.iter().zip(b_tys.iter()) {
                unify(errors, span, a_ty, b_ty)?;
            }

            return Some(());
        },
        (Type::App(a_tycon, a_tys), Type::App(b_tycon, b_tys)) if a_tycon == b_tycon => {
            match a_tycon {
                TypeCon::Tuple => {
                    for (a_ty, b_ty) in a_tys.iter().zip(b_tys.iter()) {
                        unify(errors, span, a_ty, b_ty)?;
                    }

                    return Some(());
                },
                _ => {},
            }
        },
        _ => {},
    };

    match (a, b) {
        (Type::App(TypeCon::Fun(params, body), tys), b) | (b, Type::App(TypeCon::Fun(params, body), tys)) => {
            let mut map = FxHashMap::default();
            for (param, ty) in params.iter().zip(tys.iter()) {
                map.insert(param.clone(), ty.clone());
            }

            unify(errors, span, &subst(*body.clone(), &map), b)?;
            Some(())
        },
        (Type::App(TypeCon::Unique(_, uniq1), tys1), Type::App(TypeCon::Unique(_, uniq2), tys2)) => {
            if uniq1 != uniq2 {
                return None;
            }

            for (ty1, ty2) in tys1.iter().zip(tys2.iter()) {
                unify(errors, span, ty1, ty2)?;
            }

            Some(())
        },
        (Type::Poly(vars1, ty1), Type::Poly(vars2, ty2)) => {
            let mut map = FxHashMap::default();
            for (var1, var2) in vars1.iter().zip(vars2.iter()) {
                map.insert(var2.clone(), Type::Var(*var1));
            }

            unify(errors, span, ty1, &subst(*ty2.clone(), &map))?;
            Some(())
        },
        (Type::Var(v1), Type::Var(v2)) if v1 == v2 => Some(()),
        (Type::Int, Type::Int) => Some(()),
        (Type::Bool, Type::Bool) => Some(()),
        (Type::String, Type::String) => Some(()),
        (Type::Unit, Type::Unit) => Some(()),
        (Type::App(TypeCon::Pointer(_), _), Type::Null) => Some(()),
        (Type::Null, Type::App(TypeCon::Pointer(_), _)) => Some(()),
        (a, b) => {
            errors.push(Error::new(&format!("`{}` and `{}` are not equivalent", a, b), span.clone()));
            None
        },
    }
}

fn expand_unique(ty: Type) -> Type {
    match ty {
        Type::App(TypeCon::Fun(params, body), args) => {
            // { params_i -> args_i }
            let map: FxHashMap<TypeVar, Type> = params.into_iter().zip(args.into_iter()).collect();
            expand_unique(subst(*body, &map))
        },
        Type::App(TypeCon::Unique(tycon, _), tys) => {
            expand_unique(Type::App(*tycon, tys))
        },
        ty => ty,
    }
}

fn_to_expect! {
    expect_tuple, "tuple", Vec<Type>,
    Type::App(TypeCon::Tuple, types) => Some(types),
}

#[derive(Debug)]
struct ExprInfo {
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
    types: HashMapWithScope<Id, Type>,
    tycons: HashMapWithScope<Id, Option<TypeCon>>,
    variables: Vec<FxHashMap<Id, Variable>>,
    errors: Vec<Error>,
    main_func_id: Id,
    return_value_id: Id,
    current_func: Id,
    next_temp_num: u32,
    next_unique: u32,
    std_module: ModuleHeader,
    _phantom: &'a std::marker::PhantomData<Self>,
}

impl<'a> Analyzer<'a> {
    pub fn new(std_module: ModuleHeader) -> Self {
        let main_func_id = IdMap::new_id("$main");
        let return_value_id = IdMap::new_id("$rv");

        Self {
            function_headers: FxHashMap::default(),
            variables: Vec::with_capacity(5),
            types: HashMapWithScope::new(),
            tycons: HashMapWithScope::new(),
            errors: Vec::new(),
            main_func_id,
            return_value_id,
            current_func: main_func_id, 
            next_temp_num: 0,
            next_unique: 0,
            std_module,
            _phantom: &std::marker::PhantomData,
        }
    }

    fn add_error(&mut self, msg: &str, span: Span) {
        self.errors.push(Error::new(msg, span));
    }

    #[inline]
    fn push_scope(&mut self) {
        self.variables.push(FxHashMap::default());
    }

    #[inline]
    fn pop_scope(&mut self) {
        self.variables.pop().unwrap();
    }

    // Insert parameters and return value as variables to `self.variables`
    fn insert_params(&mut self, params: Vec<Param>, return_ty: &Type) -> Option<()> {
        let mut loc = -3isize; // fp, ip
        for Param { name, ty, is_mutable } in params.into_iter().rev() {
            let ty = self.walk_type(ty)?;
            loc -= self.type_size_nocheck(&ty) as isize;

            // Insert the parameter as a variable to the current scope
            let last_map = self.variables.last_mut().unwrap();
            last_map.insert(name, Variable::new(ty.clone(), is_mutable, loc));
        }

        loc -= self.type_size_nocheck(return_ty) as isize;

        let last_map = self.variables.last_mut().unwrap();
        last_map.insert(self.return_value_id, Variable::new(return_ty.clone(), false, loc));

        Some(())
    }

    fn get_return_var(&self) -> &Variable {
        self.find_var(self.return_value_id).unwrap()
    }

    fn expand_name(&self, ty: Type) -> Option<Type> {
        match ty {
            Type::App(TypeCon::Fun(params, body), types) => {
                let map = params.into_iter().zip(types.into_iter()).collect();
                self.expand_name(subst(*body, &map))
            },
            Type::App(TypeCon::Named(name), types) => {
                match self.tycons.find(&name) {
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

    // Returns size of a specified type. if a specified type size coludn't be calculated, returns None.
    fn type_size(&self, ty: &Type) -> Option<usize> {
        match ty {
            Type::App(TypeCon::Fun(params, body), tys) => {
                let mut map = FxHashMap::default();
                for (param, ty) in params.iter().zip(tys.iter()) {
                    map.insert(param.clone(), ty.clone());
                }

                let body = subst(*body.clone(), &map);
                self.type_size(&body)
            },
            Type::App(TypeCon::Pointer(_), _) => Some(1),
            Type::App(TypeCon::Array(size), types) => {
                let elem_size = self.type_size(&types[0])?;
                Some(elem_size * size)
            },
            ty @ Type::App(TypeCon::Named(_), _) => {
                let ty = self.expand_name(ty.clone())?;
                self.type_size(&ty)
            },
            ty @ Type::App(TypeCon::Unique(_, _), _) => {
                let ty = expand_unique(ty.clone());
                self.type_size(&ty)
            },
            Type::App(_, tys) => {
                let mut size = 0;
                for ty in tys {
                    size += self.type_size(ty)?;
                }

                Some(size)
            }
            Type::Poly(_, _) | Type::Var(_) => None,
            _ => Some(1),
        }
    }

    #[inline]
    fn type_size_err(&mut self, span: Span, ty: &Type) -> usize {
        match self.type_size(ty) {
            Some(size) => size,
            None => {
                self.errors.push(Error::new(&format!("the size of type `{}` cannot be calculated", ty), span));
                0
            },
        }
    }

    #[inline]
    fn type_size_nocheck(&self, ty: &Type) -> usize {
        self.type_size(ty).unwrap_or(0)
    }

    // ====================================
    //  Variable
    // ====================================

    fn new_var(&mut self, current_func: &mut Function, id: Id, ty: Type, is_mutable: bool) -> isize {
        let new_var_size = self.type_size_nocheck(&ty);

        let last_map = self.variables.last().unwrap();
        let loc = match last_map.get(&id) {
            // If the same scope contains the same size variable, use the variable location
            Some(var) if new_var_size == self.type_size_nocheck(&var.ty) => {
                var.loc
            },
            _ => {
                let loc = current_func.stack_size as isize;
                current_func.stack_size += new_var_size as u8;
                loc
            },
        };

        let last_map = self.variables.last_mut().unwrap();
        last_map.insert(id, Variable::new(ty.clone(), is_mutable, loc));

        loc
    }

    fn gen_temp_id(&mut self) -> Id {
        let id = IdMap::new_id(&format!("$comp{}", self.next_temp_num));
        self.next_temp_num += 1;
        id
    }

    #[allow(dead_code)]
    fn dump_variables(&self) {
        let mut depth = 0;
        for variables in self.variables.iter() {
            if variables.is_empty() {
                println!("{}EMPTY", "  ".repeat(depth));
            }

            for (id, var) in variables {
                print!("{}", "  ".repeat(depth));
                if var.is_mutable {
                    print!("mut ");
                }
                println!("{}: {} ({})", IdMap::name(*id), var.ty, var.loc);
            }

            depth += 1;
        }
    }

    fn find_var(&self, id: Id) -> Option<&Variable> {
        for variables in self.variables.iter().rev() {
            if let Some(var) = variables.get(&id) {
                return Some(var);
            }
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

    fn walk_expr<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, expr: Spanned<Expr>) -> Option<ExprInfo> {
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
                        types.push(expr.ty);
                        insts.append(expr.insts);
                    }
                }

                (insts, Type::App(TypeCon::Tuple, types))
            },
            Expr::Struct(ty, field_exprs) => {
                let ty = self.walk_type(ty)?;
                let expr_ty = ty.clone();

                let ty = self.expand_name(ty)?;
                let ty = expand_unique(ty);
                let ty = subst(ty, &FxHashMap::default());

                let fields = match ty {
                    Type::App(TypeCon::Struct(fields), tys) => fields.into_iter().zip(tys.into_iter()),
                    ty => {
                        error!(self, expr.span.clone(), "expected struct but got type `{}`", ty);
                        return None;
                    },
                };

                let mut insts = InstList::new();

                // Push instructions to `insts` in order
                for (field_id, field_ty) in fields {
                    let field_expr = field_exprs.iter().find(|(id, _)| id.kind == field_id);
                    match field_expr {
                        Some((_, expr)) => {
                            // TODO: Avoid clone()
                            if let Some(expr) = self.walk_expr(code, expr.clone()) {
                                insts.append(expr.insts);
                                unify(&mut self.errors, &expr.span, &field_ty, &expr.ty);
                            }
                        },
                        None => {
                            error!(self, expr.span.clone(), "missing field `{}`", IdMap::name(field_id));
                        },
                    }
                }

                (insts, expr_ty)
            },
            Expr::Array(expr, size) => {
                let expr = self.walk_expr(code, *expr)?;

                (
                    translate::literal_array(expr.insts, self.type_size_nocheck(&expr.ty), size),
                    Type::App(TypeCon::Array(size), vec![expr.ty]),
                )
            },
            Expr::Field(comp_expr, field) => {
                let should_store = Self::expr_push_multiple_values(&comp_expr.kind);
                let comp_expr = self.walk_expr(code, *comp_expr)?;
                let comp_expr_size = self.type_size_nocheck(&comp_expr.ty);
                let mut is_mutable = comp_expr.is_mutable;

                let loc = if should_store {
                    let id = self.gen_temp_id();
                    Some(self.new_var(code.current_func_mut(), id, comp_expr.ty.clone(), false))
                } else {
                    None
                };

                let ty = self.expand_name(comp_expr.ty.clone())?;

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
                                let offset = types.iter().take(i).fold(0, |acc, ty| acc + self.type_size_nocheck(ty));
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

                        let offset = fields.iter().take(i).fold(0, |acc, (_, ty)| acc + self.type_size_nocheck(ty));
                        (fields[i].1.clone(), offset)
                    }
                };

                let insts = translate::field(loc, should_deref, comp_expr.insts, comp_expr_size, offset);
                let ty = field_ty.clone();
                return Some(ExprInfo::new_lvalue(insts, ty, expr.span, is_mutable));
            },
            Expr::Subscript(expr, subscript_expr) => {
                let should_store = Self::expr_push_multiple_values(&expr.kind);

                let expr = self.walk_expr(code, *expr);
                let subscript_expr = self.walk_expr(code, *subscript_expr);
                try_some!(expr, subscript_expr);

                let mut expr = expr;

                let loc = if should_store {
                    let id = self.gen_temp_id();
                    Some(self.new_var(code.current_func_mut(), id, expr.ty.clone(), false))
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

                expr.insts = translate::subscript(
                    loc,
                    should_deref,
                    expr.insts,
                    self.type_size_nocheck(&expr.ty),
                    subscript_expr.insts,
                    self.type_size_nocheck(&subscript_expr.ty),
                );
                expr.ty = ty;
                return Some(expr);
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
                let lhs = self.walk_expr(code, *lhs);
                let rhs = self.walk_expr(code, *rhs);
                try_some!(lhs, rhs);

                // Type check
                match (&lhs.ty, &rhs.ty) {
                    (Type::Bool, Type::Bool) => {},
                    (lty, rty) => {
                        error!(self, expr.span.clone(), "{} && {}", lty, rty);
                    },
                }

                let lhs_size = self.type_size_nocheck(&lhs.ty);
                let rhs_size = self.type_size_nocheck(&rhs.ty);
                (translate::binop_and(lhs.insts, lhs_size, rhs.insts, rhs_size), Type::Bool)
            },
            Expr::BinOp(BinOp::Or, lhs, rhs) => {
                let lhs = self.walk_expr(code, *lhs);
                let rhs = self.walk_expr(code, *rhs);
                try_some!(lhs, rhs);

                // Type check
                match (&lhs.ty, &rhs.ty) {
                    (Type::Bool, Type::Bool) => {},
                    (lty, rty) => {
                        error!(self, expr.span.clone(), "{} || {}", lty, rty);
                    },
                }

                let lhs_size = self.type_size_nocheck(&lhs.ty);
                let rhs_size = self.type_size_nocheck(&rhs.ty);
                (translate::binop_or(lhs.insts, lhs_size, rhs.insts, rhs_size), Type::Bool)
            },
            Expr::BinOp(binop, lhs, rhs) => {
                let lhs = self.walk_expr(code, *lhs);
                let rhs = self.walk_expr(code, *rhs);
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

                let lhs_size = self.type_size_nocheck(&lhs.ty);
                let rhs_size = self.type_size_nocheck(&rhs.ty);
                (translate::binop(binop, lhs.insts, lhs_size, rhs.insts, rhs_size), ty)
            },
            Expr::Call(name, args, ty_args) => {
                let (callee_func, code_id, module_id) = match self.function_headers.get(&name) {
                    Some(func) => (func, code.get_function(name).unwrap().code_id, None),
                    None => {
                        if let Some((id, func)) = self.std_module.find_func(name) {
                            (func, *id, Some(0))
                        } else {
                            error!(self, expr.span.clone(), "undefined function");
                            return None;
                        }
                    },
                };

                let ty_params = callee_func.ty_params.clone();
                let return_ty = callee_func.return_ty.clone();
                let old_params = callee_func.params.clone();

                // Check type parameter length
                if ty_params.len() != ty_args.len() {
                    error!(self, expr.span.clone(),
                        "the function takes {} type parameters. but got {} type arguments",
                        ty_params.len(),
                        ty_args.len());
                    return None;
                }

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

                // Substitute type arguments for return types
                let return_ty = subst(return_ty, &map);

                // Substitute type arguments for parameter types
                let mut params = Vec::with_capacity(old_params.len());
                for param in old_params {
                    let ty = subst(param, &map);
                    params.push(ty);
                }

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
                for (arg, param_ty) in args.into_iter().zip(params.iter()) {
                    let arg = self.walk_expr(code, arg);
                    if let Some(arg) = arg {
                        insts.push((arg.insts, self.type_size_nocheck(&arg.ty)));
                        unify(&mut self.errors, &arg.span, &arg.ty, &param_ty);
                    }
                }

                let insts = translate::call(code_id, module_id, insts, self.type_size_nocheck(&return_ty));
                (insts, return_ty)
            },
            Expr::Address(expr, is_mutable) => {
                let expr = self.walk_expr(code, *expr)?;

                if !expr.is_lvalue {
                    error!(self, expr.span, "this expression is not lvalue");
                    return None;
                } else if is_mutable && !expr.is_mutable {
                    error!(self, expr.span, "this expression is immutable");
                    return None;
                } else {
                    let insts = translate::address(expr.insts);
                    (insts, Type::App(TypeCon::Pointer(is_mutable), vec![expr.ty]))
                }
            },
            Expr::Dereference(expr) => {
                let expr = self.walk_expr(code, *expr)?;
                let expr_size = self.type_size_nocheck(&expr.ty);

                match expr.ty {
                    Type::App(TypeCon::Pointer(is_mutable), tys) => {
                        let insts = translate::dereference(expr.insts, expr_size);
                        return Some(ExprInfo::new_lvalue(insts, tys[0].clone(), expr.span, is_mutable));
                    }
                    ty => {
                        error!(self, expr.span, "expected type `pointer` but got type `{}`", ty);
                        return None;
                    }
                }
            },
            Expr::Negative(expr) => {
                let expr = self.walk_expr(code, *expr)?;
                let expr_size = self.type_size_nocheck(&expr.ty);

                match expr.ty {
                    ty @ Type::Int /* | Type::Float */ => {
                        (translate::negative(expr.insts, expr_size), ty)
                    },
                    ty => {
                        error!(self, expr.span, "expected type `int` or `float` but got type `{}`", ty);
                        return None;
                    },
                }
            },
            Expr::Alloc(expr, is_mutable) => {
                let expr = self.walk_expr(code, *expr)?;

                let insts = translate::alloc(expr.insts, self.type_size_nocheck(&expr.ty));
                (insts, Type::App(TypeCon::Pointer(is_mutable), vec![expr.ty]))
            },
        };

        Some(ExprInfo::new(insts, ty, expr.span))
    }

    fn walk_stmt<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, stmt: Spanned<Stmt>) -> Option<InstList> {
        let insts = match stmt.kind {
            Stmt::Expr(expr) => {
                let expr = self.walk_expr(code, expr)?;

                translate::expr_stmt(expr.insts, self.type_size_nocheck(&expr.ty))
            },
            Stmt::If(cond, stmt, None) => {
                let cond = self.walk_expr(code, cond);
                let then_insts = self.walk_stmt(code, *stmt);
                try_some!(cond, then_insts);

                unify(&mut self.errors, &cond.span, &Type::Bool, &cond.ty);

                translate::if_stmt(cond.insts, self.type_size_nocheck(&cond.ty), then_insts)
            },
            Stmt::If(cond, then_stmt, Some(else_stmt)) => {
                let cond = self.walk_expr(code, cond);
                let then = self.walk_stmt(code, *then_stmt);
                let els = self.walk_stmt(code, *else_stmt);
                try_some!(cond, then, els);

                unify(&mut self.errors, &cond.span, &Type::Bool, &cond.ty);

                translate::if_else_stmt(cond.insts, self.type_size_nocheck(&cond.ty), then, els)
            },
            Stmt::While(cond, stmt) => {
                let cond = self.walk_expr(code, cond);
                let body = self.walk_stmt(code, *stmt);
                try_some!(cond, body);

                unify(&mut self.errors, &cond.span, &Type::Bool, &cond.ty);

                translate::while_stmt(cond.insts, self.type_size_nocheck(&cond.ty), body)
            },
            Stmt::Block(stmts) => {
                self.push_scope();
                let mut insts = InstList::new();
                for stmt in stmts {
                    if let Some(t) = self.walk_stmt(code, stmt) {
                        insts.append(t);
                    }
                }
                self.pop_scope();

                insts
            },
            Stmt::Bind(name, expr, is_mutable) => {
                let expr = self.walk_expr(code, expr)?;
                let loc = self.new_var(code.current_func_mut(), name, expr.ty.clone(), is_mutable);

                let size = self.type_size_err(expr.span, &expr.ty);
                translate::bind_stmt(loc, expr.insts, size)
            },
            Stmt::Assign(lhs, rhs) => {
                let lhs = self.walk_expr(code, lhs);
                let rhs = self.walk_expr(code, rhs);
                try_some!(lhs, rhs);

                if !lhs.is_lvalue {
                    error!(self, lhs.span, "unassignable expression");
                    return None;
                }

                if !lhs.is_mutable {
                    error!(self, lhs.span, "immutable expression");
                    return None;
                }

                unify(&mut self.errors, &rhs.span, &lhs.ty, &rhs.ty)?;

                translate::assign_stmt(lhs.insts, rhs.insts, self.type_size_nocheck(&rhs.ty))
            },
            Stmt::Return(expr) => {
                let func_name = code.current_func().name;

                // Check if is outside function
                if func_name == self.main_func_id {
                    error!(self, stmt.span, "return statement outside function");
                    return None;
                }

                let expr = match expr {
                    Some(expr) => Some(self.walk_expr(code, expr)?),
                    None => None,
                };

                // Check type
                let return_var = self.find_var(self.return_value_id).unwrap();
                let ty = return_var.ty.clone();
                let loc = return_var.loc;

                let return_ty = expr.as_ref().map_or(&Type::Unit, |expr| &expr.ty);
                unify(&mut self.errors, &stmt.span, &ty, return_ty);

                translate::return_stmt(loc, expr.map(|expr| (expr.insts, self.type_size_nocheck(&expr.ty))))
            },
        };

        Some(insts)
    }

    fn walk_type(&mut self, ty: Spanned<AstType>) -> Option<Type> {
        match ty.kind {
            AstType::Int => Some(Type::Int),
            AstType::Bool => Some(Type::Bool),
            AstType::Unit => Some(Type::Unit),
            AstType::String => Some(Type::String),
            AstType::Named(name) => {
                if let Some(ty) = self.types.find(&name) {
                    Some(ty.clone())
                } else if self.tycons.contains_key(&name) {
                    Some(Type::App(TypeCon::Named(name), Vec::new()))
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
                self.types.push_scope();
                self.tycons.push_scope();

                if !self.tycons.contains_key(&name.kind) {
                    error!(self, name.span, "undefined type `{}`", IdMap::name(name.kind));
                    return None;
                }

                let mut new_types = Vec::with_capacity(types.len());
                for ty in types {
                    let ty = self.walk_type(ty)?;
                    new_types.push(ty);
                }

                self.types.pop_scope();
                self.tycons.pop_scope();

                Some(Type::App(TypeCon::Named(name.kind), new_types))
            },
        }
    }
    
    fn insert_type_header(&mut self, tydef: &AstTypeDef) {
        self.tycons.insert(tydef.name, None);
    }

    fn walk_type_def(&mut self, tydef: AstTypeDef) {
        self.types.push_scope();
        self.tycons.push_scope();

        let mut vars = Vec::with_capacity(tydef.var_ids.len());
        for var in &tydef.var_ids {
            vars.push(TypeVar::with_id(var.kind));
            self.types.insert(var.kind, Type::Var(*vars.last().unwrap()));
        }

        let ty = match self.walk_type(tydef.ty.clone()) {
            Some(ty) => ty,
            None => return,
        };

        let tycon = TypeCon::Fun(vars, Box::new(ty));
        let uniq = self.next_unique;
        self.next_unique += 1;

        self.tycons.pop_scope();
        self.types.pop_scope();

        self.tycons.insert(tydef.name, Some(TypeCon::Unique(Box::new(tycon), uniq)));
    }

    fn walk_function<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, func: AstFunction) {
        self.current_func = func.name;

        self.push_scope();
        self.types.push_scope();
        self.tycons.push_scope();

        let header = &self.function_headers[&func.name];
        let return_ty = header.return_ty.clone();

        for (id, var) in &header.ty_params {
            self.types.insert(*id, Type::Var(*var));
        }

        // params
        if self.insert_params(func.params, &return_ty).is_none() {
            return;
        }

        code.begin_function(func.name);
        // `None` is not returned because `func.body` is always a block statement
        let mut insts = self.walk_stmt(code, func.body).unwrap();

        // Push a return instruction if the return value type is unit
        let return_var = self.get_return_var();
        if let Type::Unit = return_var.ty {
            insts.push_inst_noarg(opcode::RETURN);
        }

        code.end_function(func.name, insts);

        self.tycons.pop_scope();
        self.types.pop_scope();
        self.pop_scope();
    }

    fn walk_main_stmt<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, stmt: Spanned<Stmt>) -> Option<InstList> {
        self.walk_stmt(code, stmt)
    }

    // =================================
    //  Header
    // =================================

    fn insert_function_header<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, func: &AstFunction) {
        self.types.push_scope();
        self.tycons.push_scope();

        let mut vars = Vec::new();
        for var_id in &func.ty_params {
            vars.push((var_id.kind, TypeVar::with_id(var_id.kind)));
            self.types.insert(var_id.kind, Type::Var(vars.last().unwrap().1));
        }

        let mut param_types = Vec::new();
        let mut param_size = 0;
        for Param { ty, .. } in &func.params {
            let ty_span = ty.span.clone();
            let ty = match self.walk_type(ty.clone()) {
                Some(ty) => ty,
                None => return,
            };

            param_size += self.type_size_err(ty_span, &ty);
            param_types.push(ty);
        }

        let return_ty_span = func.return_ty.span.clone();
        let return_ty = match self.walk_type(func.return_ty.clone()) {
            Some(ty) => ty,
            None => return,
        };

        self.tycons.pop_scope();
        self.types.pop_scope();

        self.type_size_err(return_ty_span, &return_ty);

        // Insert a header of the function
        let header = FunctionHeader {
            params: param_types,
            return_ty,
            ty_params: vars,
        };
        self.function_headers.insert(func.name, header);

        // Insert function
        let func = Function::new(func.name, param_size);
        code.new_function(func);
    }

    pub fn analyze<W: Read + Write + Seek>(mut self, code: W, program: Program) -> Result<BytecodeStream<W>, Vec<Error>> {
        let mut code = BytecodeBuilder::new(BytecodeStream::new(code), &program.strings, &[&self.std_module]);

        // Insert main function header
        let header = FunctionHeader {
            params: Vec::new(),
            return_ty: Type::Unit,
            ty_params: Vec::new(),
        };
        self.function_headers.insert(self.main_func_id, header);

        // Insert main function
        let func = Function::new(self.main_func_id, 0);
        code.new_function(func);

        self.types.push_scope();
        self.tycons.push_scope();
        self.push_scope();

        // Type definition
        for tydef in &program.types {
            self.insert_type_header(tydef);
        }

        for tydef in program.types {
            self.walk_type_def(tydef);
        }

        // Insert function headers
        for func in &program.functions {
            self.insert_function_header(&mut code, func);
        }
        code.end_new_function();

        // Function body
        for func in program.functions {
            self.walk_function(&mut code, func);
        }

        // Main statements
        code.begin_function(self.main_func_id);
        let mut insts = InstList::new();
        for stmt in program.main_stmts {
            if let Some(stmt_insts) = self.walk_main_stmt(&mut code, stmt) {
                insts.append(stmt_insts);
            }
        }
        code.end_function(self.main_func_id, insts);

        self.pop_scope();
        self.tycons.pop_scope();
        self.types.pop_scope();

        if !self.errors.is_empty() {
            Err(self.errors)
        } else {
            Ok(code.build())
        }
    }
}
