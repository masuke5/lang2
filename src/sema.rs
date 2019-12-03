use std::collections::HashMap;

use crate::ty::Type;
use crate::ast::*;
use crate::error::Error;
use crate::span::{Span, Spanned};
use crate::id::{Id, IdMap};
use crate::inst::{Inst, Function, NativeFunctionBody, BinOp as IBinOp};
use crate::stdlib::NativeFuncMap;

macro_rules! error {
    ($self:ident, $span:expr, $fmt: tt $(,$arg:expr)*) => {
        $self.errors.push(Error::new(&format!($fmt $(,$arg)*), $span));
    };
}

macro_rules! check_type {
    ($self:ident, $ty1:expr, $ty2:expr, $format:tt, $span:expr) => {
        {
            if $ty1 != Type::Invalid && $ty2 != Type::Invalid {
                if $ty1 != $ty2 {
                    $self.errors.push(Error::new(&format!($format, expected = $ty1, actual = $ty2), $span));
                    false
                } else {
                    true
                }
            } else {
                false
            }
        }
    };
}

// Return size of specified type.
// The reason why I create this as macro is because doesn't borrow `self.types` but `self`
macro_rules! type_size {
    ($self:ident, $ty:expr) => {
        match $ty {
            Type::Named(id) => {
                $self.types.get(&id)
                    .map(|ty| ty.size())
                    .unwrap_or(1) // don't have to add an error because do it already
            },
            ty => ty.size(),
        }
    }
}


#[derive(Debug)]
struct FunctionHeader {
    pub params: Vec<Type>,
    pub return_ty: Type,
}

#[derive(Debug)]
struct ExprInfo {
    pub ty: Type,
    pub span: Span,
}

impl ExprInfo {
    fn new(ty: Type, span: Span) -> Self {
        Self {
            ty,
            span,
        }
    }

    fn invalid(span: Span) -> Self {
        Self {
            ty: Type::Invalid,
            span,
        }
    }
}

#[derive(Debug)]
pub struct Analyzer<'a> {
    stdlib_funcs: &'a NativeFuncMap,
    functions: HashMap<Id, Function>,
    function_headers: HashMap<Id, FunctionHeader>,
    types: HashMap<Id, Type>,
    variables: Vec<HashMap<Id, (isize, Type, bool)>>,
    errors: Vec<Error>,
    main_func_id: Id,
    current_func: Id,
    next_temp_num: u32,
}

impl<'a> Analyzer<'a> {
    pub fn new(stdlib_funcs: &'a NativeFuncMap) -> Self {
        let main_func_id = IdMap::new_id("$main");

        Self {
            stdlib_funcs,
            functions: HashMap::new(),
            function_headers: HashMap::new(),
            variables: Vec::with_capacity(5),
            types: HashMap::new(),
            errors: Vec::new(),
            main_func_id,
            current_func: main_func_id, 
            next_temp_num: 0,
        }
    }

    fn add_error(&mut self, msg: &str, span: Span) {
        self.errors.push(Error::new(msg, span));
    }

    fn push_scope(&mut self) {
        self.variables.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.variables.pop().unwrap();
    }

    fn insert_params(&mut self, params: Vec<(Id, Type, bool)>) {
        let last_map = self.variables.last_mut().unwrap();
        let mut loc = -3isize; // fp, ip
        for (id, ty, is_mutable) in params.iter().rev() {
            loc -= type_size!(self, ty) as isize;
            last_map.insert(*id, (loc, ty.clone(), *is_mutable));
        }
    }

    // Insert a copy instruction if necessary
    fn insert_copy_inst(&self, insts: &mut Vec<Inst>, ty: &Type) {
        match insts.last() {
            Some(inst) => match inst {
                Inst::Load(_) | Inst::Dereference | Inst::Offset(_) => {
                    let size = type_size!(self, ty);
                    insts.push(Inst::Copy(size));
                },
                _ => {},
            },
            _ => {},
        }
    }

    // ====================================
    //  Variable
    // ====================================

    fn new_var(&mut self, id: Id, ty: Type, is_mutable: bool) -> isize {
        let last_map = self.variables.last_mut().unwrap();
        let current_func = self.functions.get_mut(&self.current_func).unwrap();

        let loc = current_func.stack_size as isize;
        last_map.insert(id, (loc, ty.clone(), is_mutable));

        current_func.stack_size += type_size!(self, &ty);

        loc
    }

    fn gen_temp_id(&mut self) -> Id {
        let id = IdMap::new_id(&format!("$comp{}", self.next_temp_num));
        self.next_temp_num += 1;
        id
    }

    fn find_var(&self, id: Id) -> Option<&(isize, Type, bool)> {
        for variables in self.variables.iter().rev() {
            if let Some(var) = variables.get(&id) {
                return Some(var);
            }
        }

        None
    }

    #[allow(unused_variables)]
    fn call_native(name: Id, body: NativeFunctionBody, params: usize) -> Inst {
        #[cfg(debug_assertions)]
        { Inst::CallNative(name, body, params) }
        #[cfg(not(debug_assertions))]
        { Inst::CallNative(body, params) }
    }

    fn should_store(ty: &Type) -> bool {
        match ty {
            Type::Tuple(_) | Type::Struct(_) => true,
            _ => false,
        }
    }

    // ====================================
    //  Lvalue
    // ====================================

    fn expr_is_lvalue(expr: &Expr) -> bool {
        match expr {
            Expr::Variable(_) | Expr::Dereference(_) => true,
            Expr::Field(expr, _) => Self::expr_is_lvalue(&expr.kind),
            _ => false,
        }
    }

    // return true if expr specified is mutable. this function doesn't check if a variable exists
    fn expr_is_mutable(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Variable(name) => {
                match self.find_var(*name) {
                    Some((_, _, is_mutable)) => *is_mutable,
                    None => false,
                }
            },
            Expr::Dereference(_) => {
                // TODO: return true if the expr type is `mut pointer`
                true
            },
            Expr::Field(expr, _) => {
                self.expr_is_mutable(&expr.kind)
            },
            _ => false, // expr is not lvalue
        }
    }

    // ====================================
    //  Tuple
    // ====================================

    fn walk_tuple(&mut self, insts: &mut Vec<Inst>, exprs: Vec<Spanned<Expr>>) -> (Type, usize) {
        let mut types = Vec::new();
        let mut size = 0;

        for expr in exprs {
            match expr.kind {
                Expr::Tuple(exprs) => {
                    let (ty, tuple_size) = self.walk_tuple(insts, exprs);
                    size += tuple_size;
                    types.push(ty);
                },
                Expr::Struct(name, fields) => {
                    let (ty, tsize) = self.walk_struct(insts, name, fields, expr.span);
                    size += tsize;
                    types.push(ty);
                },
                _ => {
                    let expr = self.walk_expr(insts, expr);
                    self.insert_copy_inst(insts, &expr.ty);

                    size += type_size!(self, &expr.ty);
                    types.push(expr.ty);
                },
            }
        }

        (Type::Tuple(types), size)
    }

    fn walk_struct(&mut self, insts: &mut Vec<Inst>, name: Id, exprs: Vec<(Spanned<Id>, Spanned<Expr>)>, span: Span) -> (Type, usize) {
        let ty = match self.types.get(&name) {
            Some(ty) => ty,
            None => {
                error!(self, span.clone(), "undefined type `{}`", IdMap::name(name));
                return (Type::Invalid, 0);
            },
        };

        // Get fields
        let ty_fields = match ty.clone() {
            Type::Struct(fields) => fields,
            ty => {
                error!(self, span.clone(), "expected struct but got `{}`", ty);
                return (Type::Invalid, 0);
            },
        };

        let mut fields = Vec::new();
        let mut size = 0;
        let mut not_enough_fields = Vec::new();

        for (name, ty) in ty_fields {
            match exprs.iter().find(|(field_name, _)| field_name.kind == name) {
                Some((_, expr)) => {
                    let expr = expr.clone();
                    let ty = match expr.kind {
                        Expr::Tuple(exprs) => {
                            let (ty, tsize) = self.walk_tuple(insts, exprs);
                            size += tsize;
                            ty
                        },
                        Expr::Struct(name, fields) => {
                            let (ty, tsize) = self.walk_struct(insts, name, fields, expr.span);
                            size += tsize;
                            ty
                        },
                        _ => {
                            // TODO: remove clone() if possible
                            let expr = self.walk_expr(insts, expr.clone());
                            check_type!(self, ty, expr.ty, "expected type `{expected}` but got `{actual}`", expr.span);

                            self.insert_copy_inst(insts, &expr.ty);

                            size += type_size!(self, &expr.ty);
                            expr.ty
                        },
                    };

                    fields.push((name, ty));
                },
                None => {
                    not_enough_fields.push(name);
                },
            }
        }

        // Add an error if there are not enough fields
        if !not_enough_fields.is_empty() {
            // Convert Id of not enough fields to string and join
            let mut fields = not_enough_fields
                .into_iter()
                .map(|id| IdMap::name(id))
                .fold(String::new(), |acc, s| acc + &s + ", ");
            // Remove trailing comma
            fields.truncate(fields.len() - 2);
            error!(self, span.clone(), "not enough fields: {}", fields);
        }

        (Type::Named(name), size)
    }

    fn walk_array(&mut self, insts: &mut Vec<Inst>, init_expr: Spanned<Expr>, size: usize) -> (Type, usize) {
        let init_expr = self.walk_expr(insts, init_expr);
        let expr_size = type_size!(self, &init_expr.ty);

        self.insert_copy_inst(insts, &init_expr.ty);
        insts.push(Inst::Duplicate(expr_size, size - 1));

        (Type::Array(Box::new(init_expr.ty), size), expr_size * size)
    }

    fn store_comp_literal(
        &mut self,
        insts: &mut Vec<Inst>,
        id: Id,
        expr: Spanned<Expr>,
        force_create: bool,
        is_mutable: bool
    ) -> (Type, isize) {
        let (ty, size) = match expr.kind {
            Expr::Tuple(exprs) => self.walk_tuple(insts, exprs),
            Expr::Struct(name, fields) => self.walk_struct(insts, name, fields, expr.span),
            Expr::Array(init_expr, size) => self.walk_array(insts, *init_expr, size),
            _ => panic!("the expression is not a compound literal"),
        };

        // Create a variable if variable `id` does not exists or `force_create` is true
        let loc = match self.find_var(id) {
            Some((loc, _, _)) if !force_create => *loc,
            _ => self.new_var(id, ty.clone(), is_mutable),
        };

        insts.push(Inst::Load(loc));
        insts.push(Inst::StoreWithSize(size));

        (ty, loc)
    }

    // ====================================
    //  Field
    // ====================================

    // Named =>
    //   Struct => ok
    //   _ => error
    // Pointer
    //   Struct => ok
    //   Named =>
    //     Struct => ok
    //     _ => error
    //   _ => error
    // Struct => ok
    // _ => error
    fn get_struct_fields<'b>(&'b mut self, ty: &'b Type, span: &Span) -> Option<&'b Vec<(Id, Type)>> {
        match ty {
            Type::Struct(fields) => Some(fields),
            Type::Named(name) => {
                let ty = match self.types.get(name) {
                    Some(ty) => ty,
                    None => {
                        error!(self, span.clone(), "undefined type");
                        return None;
                    },
                };

                match ty {
                    Type::Struct(fields) => Some(fields),
                    ty => {
                        error!(self, span.clone(), "expected struct but got `{}`", ty);
                        None
                    },
                }
            },
            Type::Pointer(ty) => match ty {
                box Type::Struct(fields) => Some(fields),
                box Type::Named(name) => {
                    let ty = match self.types.get(name) {
                        Some(ty) => ty,
                        None => {
                            error!(self, span.clone(), "undefined type");
                            return None;
                        },
                    };

                    match ty {
                        Type::Struct(fields) => Some(fields),
                        ty => {
                            error!(self, span.clone(), "expected struct but got `{}`", ty);
                            None
                        },
                    }
                },
                ty => {
                    error!(self, span.clone(), "expected struct but got `{}`", ty);
                    None
                },
            },
            ty => {
                error!(self, span.clone(), "expected struct but got `{}`", ty);
                None
            },
        }
    }

    fn walk_field(&mut self, insts: &mut Vec<Inst>, field: Field, expr: Spanned<Expr>) -> Option<(ExprInfo, usize)> {
        let (expr, offset) = match expr.kind {
            Expr::Field(expr, field) => {
                self.walk_field(insts, field, *expr)?
            },
            _ => {
                let expr = self.walk_expr(insts, expr);
                (expr, 0)
            },
        };

        match &expr.ty {
            Type::Pointer(_) => insts.push(Inst::Dereference),
            _ => {}
        }

        let (field_ty, types, i) = match field {
            Field::Number(i) => {
                // Return if tuple_expr type is not tuple
                let types = match &expr.ty {
                    Type::Tuple(types) => types,
                    Type::Pointer(ty) => {
                        match ty {
                            box Type::Tuple(types) => types,
                            ty => {
                                error!(self, expr.span.clone(), "expected tuple but got `{}`", ty);
                                return None;
                            },
                        }
                    },
                    ty => {
                        error!(self, expr.span.clone(), "expected tuple but got `{}`", ty);
                        return None;
                    },
                };

                // Get the field type
                let field_ty = match types.get(i) {
                    Some(ty) => ty,
                    None => {
                        error!(self, expr.span, "error");
                        return None;
                    },
                };

                (field_ty.clone(), types.clone(), i)
            },
            Field::Id(id) => {
                let fields = self.get_struct_fields(&expr.ty, &expr.span)?;

                // Get the field index
                let i = match fields.iter().position(|(name, _)| *name == id) {
                    Some(i) => i,
                    None => {
                        error!(self, expr.span, "undefined field `{}`", IdMap::name(id));
                        return None;
                    },
                };
                let (_, field_ty) = &fields[i];

                let types: Vec<Type> = fields.iter().map(|(_, ty)| ty.clone()).collect();
                (field_ty.clone(), types, i)
            },
        };

        let offset_add = types.iter()
            .take(i)
            .fold(0, |acc, ty| acc + type_size!(self, &ty));

        let info = ExprInfo::new(field_ty.clone(), expr.span);
        Some((info, offset + offset_add))
    }

    // ====================================
    //  Expression
    // ====================================

    // 複数の値を返す可能性のある式はstoreしなければならない
    fn walk_expr(&mut self, insts: &mut Vec<Inst>, expr: Spanned<Expr>) -> ExprInfo {
        let ty = match expr.kind {
            Expr::Literal(Literal::Number(n)) => {
                insts.push(Inst::Int(n));
                Type::Int
            },
            Expr::Literal(Literal::String(s)) => {
                insts.push(Inst::String(s));
                Type::String
            },
            Expr::Literal(Literal::Unit) => {
                insts.push(Inst::Int(0));
                Type::Unit
            },
            Expr::Literal(Literal::True) => {
                insts.push(Inst::True);
                Type::Bool
            },
            Expr::Literal(Literal::False) => {
                insts.push(Inst::False);
                Type::Bool
            },
            Expr::Tuple(_) | Expr::Struct(_, _) | Expr::Array(_, _) => {
                let id = self.gen_temp_id();
                let span = expr.span.clone();
                let (ty, loc) = self.store_comp_literal(insts, id, expr, true, false);

                insts.push(Inst::Load(loc));

                return ExprInfo::new(ty, span);
            },
            Expr::Field(tuple_expr, field) => {
                let (field_expr, offset) = match self.walk_field(insts, field, *tuple_expr) {
                    Some(t) => t,
                    None => return ExprInfo::invalid(expr.span),
                };

                insts.push(Inst::Offset(offset));

                field_expr.ty
            },
            Expr::Variable(name) => {
                let (loc, ty, _) = match self.find_var(name) {
                    Some(r) => r,
                    None => {
                        self.add_error("undefined variable", expr.span.clone());
                        return ExprInfo::invalid(expr.span);
                    },
                };

                insts.push(Inst::Load(*loc));

                ty.clone()
            },
            //   lhs
            //   jump_if_zero B
            //   rhs
            //   jump_if_zero B
            // A:
            //   true
            //   jump END
            // B:
            //   false
            // END:
            Expr::BinOp(BinOp::And, lhs, rhs) => {
                // Jump to `B` if `lhs` is false
                let lhs = self.walk_expr(insts, *lhs);
                let jump1 = insts.len();
                insts.push(Inst::Int(0));

                // Jump to `B` if `rhs` is false
                let rhs = self.walk_expr(insts, *rhs);
                let jump2 = insts.len();
                insts.push(Inst::Int(0));

                // A: Push true
                insts.push(Inst::True);
                let jump_to_end = insts.len();
                insts.push(Inst::Int(0));

                // B: Push false
                insts[jump1] = Inst::JumpIfZero(insts.len());
                insts[jump2] = Inst::JumpIfZero(insts.len());
                insts.push(Inst::False);

                insts[jump_to_end] = Inst::Jump(insts.len());

                // Type check
                match (lhs.ty, rhs.ty) {
                    (Type::Bool, Type::Bool) => {},
                    (Type::Invalid, _) | (_, Type::Invalid) => {},
                    (lty, rty) => {
                        error!(self, expr.span.clone(), "{} && {}", lty, rty);
                    },
                }

                Type::Bool
            },
            //   lhs
            //   jump_non_zero B
            //   rhs
            //   jump_non_zero B
            // A:
            //   false
            //   jump END
            // B:
            //   true
            // END:
            Expr::BinOp(BinOp::Or, lhs, rhs) => {
                // Jump to `B` if `lhs` is true
                let lhs = self.walk_expr(insts, *lhs);
                let jump1 = insts.len();
                insts.push(Inst::Int(0));

                // Jump to `B` if `rhs` is true
                let rhs = self.walk_expr(insts, *rhs);
                let jump2 = insts.len();
                insts.push(Inst::Int(0));

                // A: Push false
                insts.push(Inst::False);
                let jump_to_end = insts.len();
                insts.push(Inst::Int(0));

                // B: Push true
                insts[jump1] = Inst::JumpIfNonZero(insts.len());
                insts[jump2] = Inst::JumpIfNonZero(insts.len());
                insts.push(Inst::True);

                insts[jump_to_end] = Inst::Jump(insts.len());

                // Type check
                match (lhs.ty, rhs.ty) {
                    (Type::Bool, Type::Bool) => {},
                    (Type::Invalid, _) | (_, Type::Invalid) => {},
                    (lty, rty) => {
                        error!(self, expr.span.clone(), "{} || {}", lty, rty);
                    },
                }

                Type::Bool
            },
            Expr::BinOp(binop, lhs, rhs) => {
                let lhs = self.walk_expr(insts, *lhs);
                let rhs = self.walk_expr(insts, *rhs);

                // Insert an instruction
                let ibinop = match binop {
                    BinOp::Add => IBinOp::Add,
                    BinOp::Sub => IBinOp::Sub,
                    BinOp::Mul => IBinOp::Mul,
                    BinOp::Div => IBinOp::Div,
                    BinOp::LessThan => IBinOp::LessThan,
                    BinOp::LessThanOrEqual => IBinOp::LessThanOrEqual,
                    BinOp::GreaterThan => IBinOp::GreaterThan,
                    BinOp::GreaterThanOrEqual => IBinOp::GreaterThanOrEqual,
                    BinOp::Equal => IBinOp::Equal,
                    BinOp::NotEqual => IBinOp::NotEqual,
                    _ => panic!(),
                };
                insts.push(Inst::BinOp(ibinop));

                // Type check
                if !check_type!(self, lhs.ty, rhs.ty, "different types `{expected}` and `{actual}`", expr.span.clone()) {
                    return ExprInfo::new(lhs.ty, expr.span);
                }

                let binop_symbol = binop.to_symbol();
                match (binop, &lhs.ty) {
                    (BinOp::Add, Type::Int) => Type::Int,
                    (BinOp::Sub, Type::Int) => Type::Int,
                    (BinOp::Mul, Type::Int) => Type::Int,
                    (BinOp::Div, Type::Int) => Type::Int,
                    (BinOp::Equal, Type::Int) => Type::Bool,
                    (BinOp::NotEqual, Type::Int) => Type::Bool,
                    (BinOp::LessThan, Type::Int) => Type::Bool,
                    (BinOp::LessThanOrEqual, Type::Int) => Type::Bool,
                    (BinOp::GreaterThan, Type::Int) => Type::Bool,
                    (BinOp::GreaterThanOrEqual, Type::Int) => Type::Bool,
                    _ => {
                        self.add_error(&format!("`{} {} {}` is not possible", lhs.ty, binop_symbol, rhs.ty), expr.span.clone());
                        Type::Invalid
                    }
                }
            },
            Expr::Call(name, args) => {
                let name_str = IdMap::name(name);

                let (return_ty, params, inst) = match self.stdlib_funcs.get(&*name_str) {
                    Some(func) => {
                        (func.return_ty.clone(), func.params.clone(), Self::call_native(name, func.body.clone(), func.params.len()))
                    },
                    None => {
                        // Get the callee function
                        let callee_func = match self.function_headers.get(&name) {
                            Some(func) => func,
                            None => {
                                error!(self, expr.span.clone(), "undefined function");
                                return ExprInfo::invalid(expr.span);
                            },
                        };

                        (callee_func.return_ty.clone(), callee_func.params.clone(), Inst::Call(name))
                    },
                };

                // Check parameter length
                if args.len() != params.len() {
                    error!(self, expr.span.clone(),
                        "the function takes {} parameters. but got {} arguments",
                        params.len(),
                        args.len());
                    return ExprInfo::new(return_ty, expr.span);
                }

                // Check parameter types
                for (arg, param_ty) in args.into_iter().zip(params.iter()) {
                    let arg = self.walk_expr(insts, arg);
                    self.insert_copy_inst(insts, &arg.ty);

                    check_type!(self, *param_ty, arg.ty, "the parameter type is `{expected}`. but got `{actual}` type", arg.span.clone()); 
                }

                // Insert an instruction
                insts.push(inst);

                // Store if the return value is compound data
                if Self::should_store(&return_ty) {
                    let id = self.gen_temp_id();
                    let loc = self.new_var(id, return_ty.clone(), false);
                    insts.push(Inst::Load(loc));
                    insts.push(Inst::StoreWithSize(type_size!(self, &return_ty)));
                    insts.push(Inst::Load(loc));
                }

                return_ty
            },
            Expr::Address(expr) => {
                if !Self::expr_is_lvalue(&expr.kind) {
                    error!(self, expr.span, "this expression is not lvalue");
                    Type::Invalid
                } else {
                    let expr = self.walk_expr(insts, *expr);
                    insts.push(Inst::Pointer);

                    Type::Pointer(Box::new(expr.ty))
                }
            },
            Expr::Dereference(expr) => {
                let expr = self.walk_expr(insts, *expr);
                match expr.ty {
                    Type::Pointer(ty) => {
                        insts.push(Inst::Dereference);
                        *ty
                    }
                    Type::Invalid => Type::Invalid,
                    ty => {
                        error!(self, expr.span, "expected type `pointer` but got type `{}`", ty);
                        Type::Invalid
                    }
                }
            },
            Expr::Negative(expr) => {
                let expr = self.walk_expr(insts, *expr);
                match expr.ty {
                    ty @ Type::Int /* | Type::Float */ => {
                        insts.push(Inst::Negative);
                        ty
                    },
                    ty => {
                        error!(self, expr.span, "expected type `int` or `float` but got type `{}`", ty);
                        Type::Invalid
                    },
                }
            },
        };

        ExprInfo::new(ty, expr.span)
    }

    fn walk_stmt(&mut self, insts: &mut Vec<Inst>, stmt: Spanned<Stmt>) {
        match stmt.kind {
            Stmt::Expr(expr) => {
                let expr = self.walk_expr(insts, expr);

                let pop_count = type_size!(self, &expr.ty);
                for _ in 0..pop_count {
                    insts.push(Inst::Pop);
                }
            },
            Stmt::If(cond, stmt, else_stmt) => {
                // Condition
                let expr = self.walk_expr(insts, cond);
                check_type!(self, Type::Bool, expr.ty, "expected type `{expected}` but got type `{actual}`", expr.span);

                // Insert dummy instruction to jump to else-clause or end
                let jump_to_else = insts.len();
                insts.push(Inst::Int(0));

                // Then-clause
                self.walk_stmt(insts, *stmt);

                if let Some(else_stmt) = else_stmt {
                    // Insert dummy instruction to jump to end
                    let jump_to_end = insts.len();
                    insts.push(Inst::Int(0));

                    insts[jump_to_else] = Inst::JumpIfZero(insts.len());

                    // Insert else-clause instructions
                    self.walk_stmt(insts, *else_stmt);

                    insts[jump_to_end] = Inst::Jump(insts.len());
                } else {
                    // Insert instruction to jump to end
                    insts[jump_to_else] = Inst::JumpIfZero(insts.len());
                }
            },
            Stmt::While(cond, stmt) => {
                let begin = insts.len();

                // Insert condition expression instruction
                let cond = self.walk_expr(insts, cond);
                check_type!(self, Type::Bool, cond.ty, "expected type `{expected}` but got type `{actual}`", cond.span);

                // Insert dummy instruction to jump to end
                let jump_to_end = insts.len();
                insts.push(Inst::Int(0));

                // Insert body statement instruction
                self.walk_stmt(insts, *stmt);

                // Jump to begin
                insts.push(Inst::Jump(begin));

                // Insert instruction to jump to end
                insts[jump_to_end] = Inst::JumpIfZero(insts.len());
            },
            Stmt::Block(stmts) => {
                self.push_scope();
                for stmt in stmts {
                    self.walk_stmt(insts, stmt);
                }
                self.pop_scope();
            },
            Stmt::Bind(name, expr, is_mutable) => {
                match expr.kind {
                    Expr::Tuple(_) | Expr::Struct(_, _) | Expr::Array(_, _) => {
                        self.store_comp_literal(insts, name, expr, true, is_mutable);
                    },
                    _ => {
                        let expr = self.walk_expr(insts, expr);
                        self.insert_copy_inst(insts, &expr.ty);

                        let loc = self.new_var(name, expr.ty.clone(), is_mutable);
                        insts.push(Inst::Load(loc as isize));
                        insts.push(Inst::StoreWithSize(type_size!(self, &expr.ty)));
                    }
                }
            },
            Stmt::Assign(lhs, rhs) => {
                if !Self::expr_is_lvalue(&lhs.kind) {
                    error!(self, lhs.span, "unassignable expression");
                    return;
                }

                if !self.expr_is_mutable(&lhs.kind) {
                    error!(self, lhs.span, "immutable expression");
                    return;
                }

                let rhs = self.walk_expr(insts, rhs);
                self.insert_copy_inst(insts, &rhs.ty);
                let lhs = self.walk_expr(insts, lhs);

                check_type!(self, lhs.ty, rhs.ty, "expected type `{expected}` but got type `{actual}`", rhs.span);

                insts.push(Inst::StoreWithSize(type_size!(self, &lhs.ty)));
            },
            Stmt::Return(expr) => {
                let func_name = self.functions[&self.current_func].name;

                // Check if is outside function
                if func_name == self.main_func_id {
                    error!(self, stmt.span, "return statement outside function");
                    return;
                }

                let expr = match expr {
                    Some(expr) => self.walk_expr(insts, expr),
                    None => {
                        insts.push(Inst::Int(0));
                        ExprInfo::new(Type::Unit, stmt.span)
                    }
                };
                self.insert_copy_inst(insts, &expr.ty);

                // Check type
                let return_ty = &self.function_headers[&self.current_func].return_ty;
                check_type!(self, *return_ty, expr.ty, "expected `{expected}` type, but got `{actual}` type", expr.span);

                insts.push(Inst::Return(type_size!(self, return_ty)));
            },
        }
    }

    fn walk_type(&mut self, ty: &Type, span: &Span) {
        match ty {
            Type::Named(name) => {
                if !self.types.contains_key(name) {
                    error!(self, span.clone(), "undefined type `{}`", IdMap::name(*name));
                }
            },
            Type::Struct(fields) => {
                for (_, ty) in fields {
                    self.walk_type(ty, span);
                }
            },
            Type::Tuple(types) => {
                for ty in types {
                    self.walk_type(ty, span);
                }
            },
            Type::Array(ty, _) => {
                self.walk_type(ty, span);
            },
            Type::Pointer(ty) => self.walk_type(ty, span),
            Type::Int | Type::Bool | Type::String | Type::Unit | Type::Invalid => {},
        }
    }

    fn walk_toplevel(&mut self, main_insts: &mut Vec<Inst>, toplevel: Spanned<TopLevel>) {
        match toplevel.kind {
            TopLevel::Stmt(stmt) => {
                self.current_func = self.main_func_id;
                self.walk_stmt(main_insts, stmt);
            },
            TopLevel::Function(name, params, _, stmt) => {
                self.current_func = name;

                self.push_scope();

                // params
                self.insert_params(params);

                // body
                let mut insts = Vec::new();
                self.walk_stmt(&mut insts, stmt);

                // insert a return instruction if the return value type is unit
                if let Type::Unit = &self.function_headers[&self.current_func].return_ty {
                    insts.push(Inst::Int(0));
                    insts.push(Inst::Return(1));
                }

                self.functions.get_mut(&name).unwrap().insts = insts;

                self.pop_scope();
            },
            TopLevel::Type(_, ty) => {
                self.walk_type(&ty, &toplevel.span);
            },
        }
    }

    fn insert_function_header(&mut self, toplevel: &TopLevel) {
        match toplevel {
            TopLevel::Function(name, params, return_ty, _) => {
                let param_types: Vec<Type> = params.iter().map(|(_, ty, _)| ty.clone()).collect();
                let param_size = param_types.iter().fold(0, |acc, ty| acc + type_size!(self, ty));

                // Insert a header of the function
                let header = FunctionHeader {
                    params: param_types,
                    return_ty: return_ty.clone(),
                };
                self.function_headers.insert(*name, header);

                // Insert function
                let func = Function::new(*name, param_size);
                self.functions.insert(*name, func);
            },
            TopLevel::Type(name, ty) => {
                self.types.insert(*name, ty.clone());
            },
            _ => {},
        }
    }

    pub fn analyze(mut self, program: Program) -> Result<HashMap<Id, Function>, Vec<Error>> {
        // Insert main function header
        let header = FunctionHeader {
            params: Vec::new(),
            return_ty: Type::Unit,
        };
        self.function_headers.insert(self.main_func_id, header);

        // Insert main function
        let func = Function::new(self.main_func_id, 0);
        self.functions.insert(self.main_func_id, func);

        // Insert function headers
        for toplevel in program.top.iter() {
            self.insert_function_header(&toplevel.kind);
        }

        self.push_scope();

        let mut main_insts = Vec::new();
        for toplevel in program.top {
            self.walk_toplevel(&mut main_insts, toplevel);
        }

        self.pop_scope();

        self.functions.get_mut(&self.main_func_id).unwrap().insts = main_insts;

        if !self.errors.is_empty() {
            Err(self.errors)
        } else {
            Ok(self.functions)
        }
    }
}
