use std::io::{Read, Write, Seek};
use std::collections::HashMap;

use crate::ty::Type;
use crate::ast::*;
use crate::error::Error;
use crate::span::{Span, Spanned};
use crate::id::{Id, IdMap};
use crate::bytecode::{Function, opcode, BytecodeBuilder, BytecodeStream};

macro_rules! error {
    ($self:ident, $span:expr, $fmt: tt $(,$arg:expr)*) => {
        $self.errors.push(Error::new(&format!($fmt $(,$arg)*), $span));
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

fn check_type(errors: &mut Vec<Error>, expected: &Type, actual: &Type, span: Span) -> bool {
    // Don't add an error if type of either `lhs` and `rhs` is invalid
    if *expected == Type::Invalid || *actual == Type::Invalid {
        return false;
    }

    // A null can assign to a pointer
    if let Type::Pointer(_, _) = expected {
        if *actual == Type::Null {
            return true;
        }
    }

    // A mutable pointer can assign to a immutable pointer
    if let Type::Pointer(expected, false) = expected {
        if let Type::Pointer(actual, _) = actual {
            if *expected == *actual {
                return true;
            }
        }
    }

    if expected != actual {
        let error = Error::new(&format!("expected type `{}` but got type `{}`", expected, actual), span);
        errors.push(error);
        false
    } else {
        true
    }
}

fn_to_expect! {
    expect_tuple, "tuple", Vec<Type>,
    Type::Tuple(types) => Some(types),
}

fn_to_expect! {
    expect_struct, "struct", Vec<(Id, Type)>,
    Type::Struct(fields) => Some(fields),
}

// Return size of specified type.
fn type_size(types: &HashMap<Id, Type>, ty: &Type) -> usize {
    match ty {
        Type::Named(id) => {
            types.get(id)
                .map(|ty| type_size(types, ty))
                .unwrap_or(1)
        },
        Type::Tuple(tys) => tys.iter().fold(0, |acc, ty| acc + type_size(types, ty)),
        Type::Struct(fields) => fields.iter().fold(0, |acc, (_, ty)| acc + type_size(types, ty)),
        Type::Array(ty, size) => type_size(types, ty) * size,
        _ => 1,
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
    pub is_lvalue: bool,
    pub is_mutable: bool,
}

impl ExprInfo {
    fn new(ty: Type, span: Span) -> Self {
        Self {
            ty,
            span,
            is_lvalue: false,
            is_mutable: false,
        }
    }

    fn new_lvalue(ty: Type, span: Span, is_mutable: bool) -> Self {
        Self {
            ty,
            span,
            is_lvalue: true,
            is_mutable,
        }
    }

    fn invalid(span: Span) -> Self {
        Self::new(Type::Invalid, span)
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
    function_headers: HashMap<Id, FunctionHeader>,
    types: HashMap<Id, Type>,
    variables: Vec<HashMap<Id, Variable>>,
    errors: Vec<Error>,
    main_func_id: Id,
    return_value_id: Id,
    current_func: Id,
    next_temp_num: u32,
    _phantom: &'a std::marker::PhantomData<Self>,
}

impl<'a> Analyzer<'a> {
    pub fn new() -> Self {
        let main_func_id = IdMap::new_id("$main");
        let return_value_id = IdMap::new_id("$rv");

        Self {
            function_headers: HashMap::new(),
            variables: Vec::with_capacity(5),
            types: HashMap::new(),
            errors: Vec::new(),
            main_func_id,
            return_value_id,
            current_func: main_func_id, 
            next_temp_num: 0,
            _phantom: &std::marker::PhantomData,
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

    // Insert parameters and return value as variables to `self.variables`
    fn insert_params(&mut self, params: Vec<(Id, Type, bool)>, return_ty: &Type) {
        let last_map = self.variables.last_mut().unwrap();
        let mut loc = -3isize; // fp, ip
        for (id, ty, is_mutable) in params.iter().rev() {
            loc -= type_size(&self.types, ty) as isize;
            last_map.insert(*id, Variable::new(ty.clone(), *is_mutable, loc));
        }

        loc -= type_size(&self.types, return_ty) as isize;
        last_map.insert(self.return_value_id, Variable::new(return_ty.clone(), false, loc));
    }

    // Insert a copy instruction if necessary
    fn insert_copy_inst<W: Read + Write + Seek>(&self, bytecode: &mut BytecodeBuilder<W>, ty: &Type) {
        if bytecode.code.len() < 2 {
            return;
        }

        match bytecode.prev_opcode() {
            // opcode::LOAD_REF => { }, TODO: Insert LOAD_COPY
            opcode::LOAD_REF | opcode::DEREFERENCE | opcode::OFFSET => {
                let size = type_size(&self.types, ty);
                bytecode.insert_inst(opcode::COPY, size as u8);
            },
            opcode::CALL | opcode::CALL_NATIVE if Self::should_store(ty) => {
                let size = type_size(&self.types, ty);
                bytecode.insert_inst(opcode::COPY, size as u8);
            },
            _ => {},
        }
    }

    fn get_return_var(&self) -> &Variable {
        self.find_var(self.return_value_id).unwrap()
    }

    // ====================================
    //  Variable
    // ====================================

    fn new_var(&mut self, current_func: &mut Function, id: Id, ty: Type, is_mutable: bool) -> isize {
        let last_map = self.variables.last_mut().unwrap();
        let new_var_size = type_size(&self.types, &ty);

        let loc = match last_map.get(&id) {
            // If the same scope contains the same size variable, use the variable location
            Some(var) if new_var_size == type_size(&self.types, &var.ty) => {
                var.loc
            },
            _ => {
                let loc = current_func.stack_size as isize;
                current_func.stack_size += new_var_size as u8;
                loc
            },
        };

        last_map.insert(id, Variable::new(ty.clone(), is_mutable, loc));

        loc
    }

    fn gen_temp_id(&mut self) -> Id {
        let id = IdMap::new_id(&format!("$comp{}", self.next_temp_num));
        self.next_temp_num += 1;
        id
    }

    fn find_var(&self, id: Id) -> Option<&Variable> {
        for variables in self.variables.iter().rev() {
            if let Some(var) = variables.get(&id) {
                return Some(var);
            }
        }

        None
    }

    fn should_store(ty: &Type) -> bool {
        match ty {
            Type::Tuple(_) | Type::Struct(_) | Type::Array(_, _) => true,
            _ => false,
        }
    }

    // ====================================
    //  Tuple
    // ====================================

    fn walk_tuple<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, exprs: Vec<Spanned<Expr>>) -> (Type, usize) {
        let mut types = Vec::new();
        let mut size = 0;

        for expr in exprs {
            match expr.kind {
                Expr::Tuple(exprs) => {
                    let (ty, tuple_size) = self.walk_tuple(code, exprs);
                    size += tuple_size;
                    types.push(ty);
                },
                Expr::Struct(name, fields) => {
                    let (ty, tsize) = self.walk_struct(code, name, fields, expr.span);
                    size += tsize;
                    types.push(ty);
                },
                _ => {
                    let expr = self.walk_expr(code, expr);
                    self.insert_copy_inst(code, &expr.ty);

                    size += type_size(&self.types, &expr.ty);
                    types.push(expr.ty);
                },
            }
        }

        (Type::Tuple(types), size)
    }

    fn walk_struct<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, name: Id, exprs: Vec<(Spanned<Id>, Spanned<Expr>)>, span: Span) -> (Type, usize) {
        let ty = match self.types.get(&name) {
            Some(ty) => ty,
            None => {
                error!(self, span.clone(), "undefined type `{}`", IdMap::name(name));
                return (Type::Invalid, 0);
            },
        };

        // Get fields
        let ty_fields = match expect_struct(&mut self.errors, ty, span.clone()) {
            Some(fields) => fields,
            None => return (Type::Invalid, 0),
        };
        let ty_fields = ty_fields.clone();

        let mut fields = Vec::new();
        let mut size = 0;
        let mut not_enough_fields = Vec::new();

        for (name, ty) in ty_fields {
            match exprs.iter().find(|(field_name, _)| field_name.kind == name) {
                Some((_, expr)) => {
                    let expr = expr.clone();
                    let ty = match expr.kind {
                        Expr::Tuple(exprs) => {
                            let (ty, tsize) = self.walk_tuple(code, exprs);
                            size += tsize;
                            ty
                        },
                        Expr::Struct(name, fields) => {
                            let (ty, tsize) = self.walk_struct(code, name, fields, expr.span);
                            size += tsize;
                            ty
                        },
                        _ => {
                            let expr = self.walk_expr(code, expr);
                            check_type(&mut self.errors, &ty, &expr.ty, expr.span);

                            self.insert_copy_inst(code, &expr.ty);

                            size += type_size(&self.types, &expr.ty);
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

    fn walk_array<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, init_expr: Spanned<Expr>, size: usize) -> (Type, usize) {
        let init_expr = self.walk_expr(code, init_expr);
        let expr_size = type_size(&self.types, &init_expr.ty);

        self.insert_copy_inst(code, &init_expr.ty);
        code.insert_inst(opcode::DUPLICATE, (size - 1) as u8);

        (Type::Array(Box::new(init_expr.ty), size), expr_size * size)
    }

    fn store_comp_literal<W: Read + Write + Seek>(
        &mut self,
        code: &mut BytecodeBuilder<W>,
        id: Id,
        expr: Spanned<Expr>,
        force_create: bool,
        is_mutable: bool
    ) -> (Type, isize) {
        let (ty, size) = match expr.kind {
            Expr::Tuple(exprs) => self.walk_tuple(code, exprs),
            Expr::Struct(name, fields) => self.walk_struct(code, name, fields, expr.span),
            Expr::Array(init_expr, size) => self.walk_array(code, *init_expr, size),
            _ => panic!("the expression is not a compound literal"),
        };

        // Create a variable if variable `id` does not exists or `force_create` is true
        let loc = match self.find_var(id) {
            Some(var) if !force_create => var.loc,
            _ => self.new_var(code.current_func_mut(), id, ty.clone(), is_mutable),
        };

        code.insert_inst(opcode::LOAD_REF, loc as u8);
        code.insert_inst(opcode::STORE, size as u8);

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

                expect_struct(&mut self.errors, ty, span.clone())
            },
            Type::Pointer(ty, _) => match ty {
                box Type::Struct(fields) => Some(fields),
                box Type::Named(name) => {
                    let ty = match self.types.get(name) {
                        Some(ty) => ty,
                        None => {
                            error!(self, span.clone(), "undefined type");
                            return None;
                        },
                    };

                    expect_struct(&mut self.errors, ty, span.clone())
                },
                ty => {
                    error!(self, span.clone(), "expected type `struct` or `*struct` but got type `{}`", ty);
                    None
                },
            },
            ty => {
                error!(self, span.clone(), "expected type `struct` or `*struct` but got type `{}`", ty);
                None
            },
        }
    }

    fn walk_field<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, field: Field, expr: Spanned<Expr>) -> Option<ExprInfo> {
        let mut expr = match expr.kind {
            Expr::Field(expr, field) => {
                self.walk_field(code, field, *expr)?
            },
            _ => {
                let expr = self.walk_expr(code, expr);
                expr
            },
        };

        match &expr.ty {
            Type::Pointer(_, is_mutable) => {
                self.insert_copy_inst(code, &expr.ty);
                code.insert_inst_noarg(opcode::DEREFERENCE);
                expr.is_mutable = *is_mutable;
            },
            _ => {}
        }

        let (field_ty, types, i) = match field {
            Field::Number(i) => {
                // Return if tuple_expr type is not tuple
                let types = match &expr.ty {
                    Type::Tuple(types) => types,
                    Type::Pointer(ty, _) => expect_tuple(&mut self.errors, ty, expr.span.clone())?,
                    ty => {
                        error!(self, expr.span.clone(), "expected `tuple` or `*tuple` but got `{}`", ty);
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

        let offset = types.iter()
            .take(i)
            .fold(0, |acc, ty| acc + type_size(&self.types, &ty));

        if offset != 0 {
            code.insert_inst_ref(opcode::INT, offset);
            code.insert_inst_noarg(opcode::OFFSET);
        }

        expr.ty = field_ty.clone();
        Some(expr)
    }

    // ====================================
    //  Expression
    // ====================================

    // 複数の値を返す可能性のある式はstoreしなければならない
    fn walk_expr<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, expr: Spanned<Expr>) -> ExprInfo {
        let ty = match expr.kind {
            Expr::Literal(Literal::Number(n)) => {
                code.insert_inst_ref(opcode::INT, n);
                Type::Int
            },
            Expr::Literal(Literal::String(i)) => {
                code.insert_inst(opcode::STRING, i as u8);
                Type::Pointer(Box::new(Type::String), false)
            },
            Expr::Literal(Literal::Unit) => {
                code.insert_inst(opcode::ZERO, 1);
                Type::Unit
            },
            Expr::Literal(Literal::True) => {
                code.insert_inst_noarg(opcode::TRUE);
                Type::Bool
            },
            Expr::Literal(Literal::False) => {
                code.insert_inst_noarg(opcode::FALSE);
                Type::Bool
            },
            Expr::Literal(Literal::Null) => {
                code.insert_inst_noarg(opcode::NULL);
                Type::Null
            },
            Expr::Tuple(_) | Expr::Struct(_, _) | Expr::Array(_, _) => {
                let id = self.gen_temp_id();
                let span = expr.span.clone();
                let (ty, loc) = self.store_comp_literal(code, id, expr, true, false);

                code.insert_inst(opcode::LOAD_REF, loc as u8);

                return ExprInfo::new(ty, span);
            },
            Expr::Field(tuple_expr, field) => {
                let field_expr = match self.walk_field(code, field, *tuple_expr) {
                    Some(t) => t,
                    None => return ExprInfo::invalid(expr.span),
                };

                return field_expr;
            },
            Expr::Subscript(expr, subscript_expr) => {
                let mut expr = self.walk_expr(code, *expr);

                let ty = match expr.ty {
                    Type::Array(ty, _) => *ty,
                    Type::Pointer(ty, is_mutable) => {
                        expr.is_mutable = is_mutable;
                        self.insert_copy_inst(code, &Type::Pointer(ty.clone(), is_mutable));
                        code.insert_inst_noarg(opcode::DEREFERENCE);

                        match *ty {
                            Type::Array(ty, _) => *ty,
                            ty => {
                                error!(self, expr.span.clone(), "expected array but got type `{}`", ty);
                                return ExprInfo::invalid(expr.span);
                            },
                        }
                    }
                    ty => {
                        error!(self, expr.span.clone(), "expected array but got type `{}`", ty);
                        return ExprInfo::invalid(expr.span);
                    },
                };

                let subscript_expr = self.walk_expr(code, *subscript_expr);
                self.insert_copy_inst(code, &subscript_expr.ty);

                check_type(&mut self.errors, &Type::Int, &subscript_expr.ty, subscript_expr.span);

                code.insert_inst_noarg(opcode::OFFSET);

                expr.ty = ty;
                return expr;
            },
            Expr::Variable(name) => {
                let var = match self.find_var(name) {
                    Some(v) => v,
                    None => {
                        self.add_error("undefined variable", expr.span.clone());
                        return ExprInfo::invalid(expr.span);
                    },
                };

                code.insert_inst(opcode::LOAD_REF, var.loc as u8);

                return ExprInfo::new_lvalue(var.ty.clone(), expr.span, var.is_mutable);
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
                let lhs = self.walk_expr(code, *lhs);
                self.insert_copy_inst(code, &lhs.ty);
                let jump1 = code.jump();

                // Jump to `B` if `rhs` is false
                let rhs = self.walk_expr(code, *rhs);
                self.insert_copy_inst(code, &lhs.ty);
                let jump2 = code.jump();

                // A: Push true
                code.insert_inst_noarg(opcode::TRUE);
                let jump_to_end = code.jump();

                // B: Push false
                code.insert_jump_if_false_inst(jump1);
                code.insert_jump_if_false_inst(jump2);
                code.insert_inst_noarg(opcode::FALSE);

                code.insert_jump_inst(jump_to_end);

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
                let lhs = self.walk_expr(code, *lhs);
                self.insert_copy_inst(code, &lhs.ty);
                let jump1 = code.jump();

                // Jump to `B` if `rhs` is true
                let rhs = self.walk_expr(code, *rhs);
                self.insert_copy_inst(code, &lhs.ty);
                let jump2 = code.jump();

                // A: Push false
                code.insert_inst_noarg(opcode::FALSE);
                let jump_to_end = code.jump();

                // B: Push true
                code.insert_jump_if_true_inst(jump1);
                code.insert_jump_if_true_inst(jump2);
                code.insert_inst_noarg(opcode::TRUE);

                code.insert_jump_inst(jump_to_end);

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
                let lhs = self.walk_expr(code, *lhs);
                self.insert_copy_inst(code, &lhs.ty);
                let rhs = self.walk_expr(code, *rhs);
                self.insert_copy_inst(code, &lhs.ty);

                // Insert an instruction
                let opcode = match binop {
                    BinOp::Add => opcode::BINOP_ADD,
                    BinOp::Sub => opcode::BINOP_SUB,
                    BinOp::Mul => opcode::BINOP_MUL,
                    BinOp::Div => opcode::BINOP_DIV,
                    BinOp::LessThan => opcode::BINOP_LT,
                    BinOp::LessThanOrEqual => opcode::BINOP_LE,
                    BinOp::GreaterThan => opcode::BINOP_GT,
                    BinOp::GreaterThanOrEqual => opcode::BINOP_GE,
                    BinOp::Equal => opcode::BINOP_EQ,
                    BinOp::NotEqual => opcode::BINOP_NEQ,
                    _ => panic!(),
                };
                code.insert_inst_noarg(opcode);

                let binop_symbol = binop.to_symbol();
                match (binop, &lhs.ty, &rhs.ty) {
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

                    (BinOp::Equal, Type::Pointer(_, _), Type::Pointer(_, _)) => Type::Bool,
                    (BinOp::Equal, Type::Null, Type::Pointer(_, _)) => Type::Bool,
                    (BinOp::Equal, Type::Pointer(_, _), Type::Null) => Type::Bool,
                    (BinOp::NotEqual, Type::Pointer(_, _), Type::Pointer(_, _)) => Type::Bool,
                    (BinOp::NotEqual, Type::Null, Type::Pointer(_, _)) => Type::Bool,
                    (BinOp::NotEqual, Type::Pointer(_, _), Type::Null) => Type::Bool,
                    _ => {
                        self.add_error(&format!("`{} {} {}`", lhs.ty, binop_symbol, rhs.ty), expr.span.clone());
                        Type::Invalid
                    }
                }
            },
            Expr::Call(name, args) => {
                let (return_ty, params, code_id) = {
                    // Get the callee function
                    let callee_func = match self.function_headers.get(&name) {
                        Some(func) => func,
                        None => {
                            error!(self, expr.span.clone(), "undefined function");
                            return ExprInfo::invalid(expr.span);
                        },
                    };

                    let return_value_size = type_size(&self.types, &callee_func.return_ty);
                    if return_value_size > std::u8::MAX as usize {
                        panic!("too large return value");
                    }

                    code.insert_inst(opcode::ZERO, return_value_size as u8);

                    (callee_func.return_ty.clone(), callee_func.params.clone(), code.get_function(name).unwrap().code_id)
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
                    let arg = self.walk_expr(code, arg);
                    self.insert_copy_inst(code, &arg.ty);
                    check_type(&mut self.errors, &param_ty, &arg.ty, arg.span.clone());
                }

                // TODO: Insert an instruction
                code.insert_inst(opcode::CALL, code_id as u8);

                // Store if the return value is compound data
                if Self::should_store(&return_ty) {
                    let id = self.gen_temp_id();
                    let loc = self.new_var(code.current_func_mut(), id, return_ty.clone(), false);
                    code.insert_inst(opcode::LOAD_REF, loc as u8);
                    code.insert_inst(opcode::STORE, type_size(&self.types, &return_ty) as u8);
                    code.insert_inst(opcode::LOAD_REF, loc as u8);
                }

                return_ty
            },
            Expr::Address(expr, is_mutable) => {
                let expr = self.walk_expr(code, *expr);

                if !expr.is_lvalue {
                    error!(self, expr.span, "this expression is not lvalue");
                    Type::Invalid
                } else if is_mutable && !expr.is_mutable {
                    error!(self, expr.span, "this expression is immutable");
                    Type::Invalid
                } else {
                    code.insert_inst_noarg(opcode::POINTER);
                    Type::Pointer(Box::new(expr.ty), is_mutable)
                }
            },
            Expr::Dereference(expr) => {
                let expr = self.walk_expr(code, *expr);
                self.insert_copy_inst(code, &expr.ty);

                match expr.ty {
                    Type::Pointer(ty, is_mutable) => {
                        code.insert_inst_noarg(opcode::DEREFERENCE);
                        return ExprInfo::new_lvalue(*ty, expr.span, is_mutable); // TODO:
                    }
                    Type::Invalid => Type::Invalid,
                    ty => {
                        error!(self, expr.span, "expected type `pointer` but got type `{}`", ty);
                        Type::Invalid
                    }
                }
            },
            Expr::Negative(expr) => {
                let expr = self.walk_expr(code, *expr);
                self.insert_copy_inst(code, &expr.ty);

                match expr.ty {
                    ty @ Type::Int /* | Type::Float */ => {
                        code.insert_inst_noarg(opcode::NEGATIVE);
                        ty
                    },
                    ty => {
                        error!(self, expr.span, "expected type `int` or `float` but got type `{}`", ty);
                        Type::Invalid
                    },
                }
            },
            Expr::Alloc(expr, is_mutable) => {
                let expr = self.walk_expr(code, *expr);
                self.insert_copy_inst(code, &expr.ty);
                code.insert_inst(opcode::ALLOC, type_size(&self.types, &expr.ty) as u8);

                Type::Pointer(Box::new(expr.ty), is_mutable)
            },
        };

        ExprInfo::new(ty, expr.span)
    }

    fn walk_stmt<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, stmt: Spanned<Stmt>) {
        match stmt.kind {
            Stmt::Expr(expr) => {
                let expr = self.walk_expr(code, expr);

                let pop_count = type_size(&self.types, &expr.ty);
                for _ in 0..pop_count {
                    code.insert_inst_noarg(opcode::POP);
                }
            },
            Stmt::If(cond, stmt, else_stmt) => {
                // Condition
                let expr = self.walk_expr(code, cond);
                self.insert_copy_inst(code, &expr.ty);
                check_type(&mut self.errors, &Type::Bool, &expr.ty, expr.span);

                let jump_to_else = code.jump();

                // Then-clause
                self.walk_stmt(code, *stmt);

                if let Some(else_stmt) = else_stmt {
                    let jump_to_end = code.jump();

                    code.insert_jump_if_false_inst(jump_to_else);

                    // Insert else-clause instructions
                    self.walk_stmt(code, *else_stmt);

                    code.insert_jump_inst(jump_to_end);
                } else {
                    code.insert_jump_if_false_inst(jump_to_else);
                }
            },
            Stmt::While(cond, stmt) => {
                let begin = code.code.len();

                // Insert condition expression instruction
                let cond = self.walk_expr(code, cond);
                self.insert_copy_inst(code, &cond.ty);
                check_type(&mut self.errors, &Type::Bool, &cond.ty, cond.span);

                // Insert dummy instruction to jump to end
                let jump_to_end = code.jump();

                // Insert body statement instruction
                self.walk_stmt(code, *stmt);

                // Jump to begin
                code.insert_inst(opcode::JUMP, begin as u8);

                // Insert instruction to jump to end
                code.insert_jump_if_false_inst(jump_to_end);
            },
            Stmt::Block(stmts) => {
                self.push_scope();
                for stmt in stmts {
                    self.walk_stmt(code, stmt);
                }
                self.pop_scope();
            },
            Stmt::Bind(name, expr, is_mutable) => {
                match expr.kind {
                    Expr::Tuple(_) | Expr::Struct(_, _) | Expr::Array(_, _) => {
                        self.store_comp_literal(code, name, expr, true, is_mutable);
                    },
                    _ => {
                        let expr = self.walk_expr(code, expr);
                        self.insert_copy_inst(code, &expr.ty);

                        let loc = self.new_var(code.current_func_mut(), name, expr.ty.clone(), is_mutable);
                        code.insert_inst(opcode::LOAD_REF, loc as u8);
                        code.insert_inst(opcode::STORE, type_size(&self.types, &expr.ty) as u8);
                        // XXX: insts.push(Inst::Load(loc as isize));
                    }
                }
            },
            Stmt::Assign(lhs, rhs) => {
                let rhs = self.walk_expr(code, rhs);
                self.insert_copy_inst(code, &rhs.ty);
                let lhs = self.walk_expr(code, lhs);

                if !lhs.is_lvalue {
                    error!(self, lhs.span, "unassignable expression");
                    return;
                }

                if !lhs.is_mutable {
                    error!(self, lhs.span, "immutable expression");
                    return;
                }

                check_type(&mut self.errors, &lhs.ty, &rhs.ty, rhs.span);

                code.insert_inst(opcode::STORE, type_size(&self.types, &lhs.ty) as u8);
            },
            Stmt::Return(expr) => {
                let func_name = code.current_func().name;

                // Check if is outside function
                if func_name == self.main_func_id {
                    error!(self, stmt.span, "return statement outside function");
                    return;
                }

                let expr = match expr {
                    Some(expr) => self.walk_expr(code, expr),
                    None => {
                        code.insert_inst(opcode::ZERO, 1);
                        ExprInfo::new(Type::Unit, stmt.span)
                    }
                };
                self.insert_copy_inst(code, &expr.ty);

                // Check type
                let return_var = self.find_var(self.return_value_id).unwrap();
                let ty = return_var.ty.clone();
                let loc = return_var.loc;

                check_type(&mut self.errors, &ty, &expr.ty, expr.span);

                code.insert_inst(opcode::LOAD_REF, loc as u8);
                code.insert_inst(opcode::STORE, type_size(&self.types, &ty) as u8);
                code.insert_inst_noarg(opcode::RETURN);
            },
        }
    }

    // Check if specified type exists
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
            Type::Pointer(ty, _) => self.walk_type(ty, span),
            Type::Int | Type::Bool | Type::String | Type::Unit | Type::Invalid | Type::Null => {},
        }
    }

    fn walk_toplevel<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, toplevel: Spanned<TopLevel>) {
        match toplevel.kind {
            TopLevel::Function(name, params, return_ty, stmt) => {
                self.current_func = name;

                self.push_scope();

                // params
                self.insert_params(params, &return_ty);

                // body
                code.begin_function(name);
                self.walk_stmt(code, stmt);
                code.end_function(name);

                let return_var = self.get_return_var();

                // insert a return instruction if the return value type is unit
                if let Type::Unit = return_var.ty {
                    code.insert_inst_noarg(opcode::RETURN);
                }

                self.pop_scope();
            },
            TopLevel::Type(_, ty) => {
                self.walk_type(&ty, &toplevel.span);
            },
            _ => {},
        }
    }

    fn walk_main_stmt<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, toplevel: Spanned<TopLevel>) {
        if let TopLevel::Stmt(stmt) = toplevel.kind {
            self.walk_stmt(code, stmt);
        }
    }

    fn insert_function_header<W: Read + Write + Seek>(&mut self, code: &mut BytecodeBuilder<W>, toplevel: &TopLevel) {
        match toplevel {
            TopLevel::Function(name, params, return_ty, _) => {
                let param_types: Vec<Type> = params.iter().map(|(_, ty, _)| ty.clone()).collect();
                let param_size = param_types.iter().fold(0, |acc, ty| acc + type_size(&self.types, ty));

                // Insert a header of the function
                let header = FunctionHeader {
                    params: param_types,
                    return_ty: return_ty.clone(),
                };
                self.function_headers.insert(*name, header);

                // Insert function
                let func = Function::new(*name, param_size);
                code.new_function(func);
            },
            TopLevel::Type(name, ty) => {
                self.types.insert(*name, ty.clone());
            },
            _ => {},
        }
    }

    pub fn analyze<W: Read + Write + Seek>(mut self, code: W, mut program: Program) -> Result<BytecodeStream<W>, Vec<Error>> {
        let mut code = BytecodeBuilder::new(BytecodeStream::new(code), &program.strings);

        // Insert main function header
        let header = FunctionHeader {
            params: Vec::new(),
            return_ty: Type::Unit,
        };
        self.function_headers.insert(self.main_func_id, header);

        // Insert main function
        let func = Function::new(self.main_func_id, 0);
        code.new_function(func);

        // Insert function headers
        for toplevel in program.top.iter() {
            self.insert_function_header(&mut code, &toplevel.kind);
        }
        code.end_new_function();

        self.push_scope();

        for toplevel in program.top.drain_filter(|toplevel| match toplevel.kind { TopLevel::Stmt(_) => false, _ => true }) {
            self.walk_toplevel(&mut code, toplevel);
        }

        code.begin_function(self.main_func_id);
        for toplevel in program.top {
            self.walk_main_stmt(&mut code, toplevel);
        }
        code.end_function(self.main_func_id);

        self.pop_scope();

        if !self.errors.is_empty() {
            Err(self.errors)
        } else {
            Ok(code.build())
        }
    }
}
