use crate::ast::BinOp;
use crate::ir::{BinOp as IRBinOp, CodeBuf, Expr, Function, Label, Stmt, VariableLoc};
use crate::sema::{ExprInfo, VariableMap};
use crate::ty::{type_size_nocheck, Type, TypeCon};

#[derive(Debug)]
pub enum RelativeVariableLoc {
    Stack(isize),
    StackInHeap(usize, usize),
}

impl RelativeVariableLoc {
    fn as_ir_loc(&self) -> VariableLoc {
        match self {
            Self::Stack(loc) => VariableLoc::Local(*loc),
            Self::StackInHeap(loc, level) => VariableLoc::Heap(*loc, *level),
        }
    }
}

pub fn wrap(mut expr: Expr, ty: &Type) -> Expr {
    // Don't allow to wrap doubly
    assert!(if let Type::App(TypeCon::Wrapped, _) = ty {
        false
    } else {
        true
    });

    expr = Expr::Copy(box expr, type_size_nocheck(ty));

    let size = type_size_nocheck(ty);
    if size > 1 {
        expr = Expr::Wrap(box expr);
    }

    expr
}

pub fn unwrap(mut expr: Expr, ty: &Type) -> Expr {
    let inner_ty = if let Type::App(TypeCon::Wrapped, types) = ty {
        &types[0]
    } else {
        panic!("not wrapped type: {}", ty);
    };

    expr = Expr::Copy(box expr, type_size_nocheck(ty));

    let size = type_size_nocheck(inner_ty);
    if size > 1 {
        expr = Expr::Unwrap(box expr, size);
    }

    expr
}

pub fn escaped_param(ty: &Type, loc: isize, heap_loc: usize) -> CodeBuf {
    let mut stmts = CodeBuf::new();

    stmts.push(Stmt::Store(
        VariableLoc::Heap(heap_loc, 0),
        Expr::LoadCopy(VariableLoc::Local(loc), type_size_nocheck(ty)),
    ));

    stmts
}

pub fn copy(expr: Expr, ty: &Type) -> Expr {
    Expr::Copy(box expr, type_size_nocheck(ty))
}

// Expression

pub fn literal_int(n: i64) -> Expr {
    Expr::Int(n)
}

pub fn literal_str(s: String) -> Expr {
    Expr::String(s)
}

pub fn literal_true() -> Expr {
    Expr::True
}

pub fn literal_false() -> Expr {
    Expr::False
}

pub fn literal_null() -> Expr {
    Expr::Null
}

pub fn literal_array(expr: ExprInfo, arr_len: usize) -> Expr {
    Expr::Duplicate(box (expr.ir), arr_len)
}

pub fn literal_struct_field(expr: ExprInfo) -> Expr {
    Expr::Copy(box (expr.ir), type_size_nocheck(&expr.ty))
}

pub fn literal_tuple(expr: ExprInfo) -> Expr {
    Expr::Copy(box (expr.ir), type_size_nocheck(&expr.ty))
}

fn generate_pointer(
    ir_func: &mut Function,
    variables: &mut VariableMap,
    expr: ExprInfo,
) -> (Expr, Type) {
    let mut ir = expr.ir;
    let mut ty = expr.ty;

    if let Type::App(TypeCon::InHeap, types) = ty {
        assert!(expr.is_lvalue);
        ir = Expr::Dereference(box Expr::Copy(box ir, 1));
        ty = types[0].clone();
    }

    if !expr.is_lvalue && !ty.is_wrapped() {
        // Store it to temporary location and get a pointer to the location
        let loc = variables.create_temp(ir_func, ty.clone());
        let loc = variables.relative_loc(&loc);

        ir = Expr::Seq(
            vec![Stmt::Store(loc.as_ir_loc(), ir)],
            box (Expr::LoadRef(loc.as_ir_loc())),
        );
    }

    if expr.is_lvalue {
        if let Type::App(TypeCon::Wrapped, types) = ty {
            ir = Expr::Dereference(box Expr::Copy(box ir, 1));
            ty = types[0].clone();
        }
    }

    (ir, ty)
}

fn generate_pointer_with_deref(
    ir_func: &mut Function,
    variables: &mut VariableMap,
    expr: ExprInfo,
) -> (Expr, Type) {
    if let Type::App(TypeCon::Pointer(..), types) = &expr.ty {
        return (Expr::Copy(box expr.ir, 1), types[0].clone());
    }

    let (mut expr, mut ty) = generate_pointer(ir_func, variables, expr);
    if let Type::App(TypeCon::Pointer(..), types) = ty {
        expr = Expr::Dereference(box Expr::Copy(box expr, 1));
        ty = types[0].clone();
    }

    (expr, ty)
}

pub fn field(
    ir_func: &mut Function,
    variables: &mut VariableMap,
    comp_expr: ExprInfo,
    offset: usize,
) -> Expr {
    let (expr, _) = generate_pointer_with_deref(ir_func, variables, comp_expr);

    Expr::Offset(box expr, box Expr::Int(offset as i64))
}

pub fn subscript(
    ir_func: &mut Function,
    variables: &mut VariableMap,
    expr: ExprInfo,
    subscript_expr: ExprInfo,
    element_ty: &Type,
) -> Expr {
    let (ir, _) = generate_pointer_with_deref(ir_func, variables, expr);

    // ir[subscript_expr * type_size_nocheck(element_ty)]
    Expr::Offset(
        box ir,
        box Expr::BinOp(
            IRBinOp::Mul,
            box Expr::Copy(box subscript_expr.ir, type_size_nocheck(&subscript_expr.ty)),
            box Expr::Int(type_size_nocheck(element_ty) as i64),
        ),
    )
}

pub fn variable(loc: &RelativeVariableLoc) -> Expr {
    Expr::LoadRef(loc.as_ir_loc())
}

pub fn func_pos(module_id: Option<usize>, func_id: usize) -> Expr {
    Expr::Record(vec![Expr::FuncPos(module_id, func_id), Expr::EP])
}

//   lhs
//   JUMP_IF_FALSE a
//   rhs
//   JUMP_IF_FALSE a
//   TRUE
//   JUMP end
// a:
//   FALSE
// end:
pub fn binop_and(lhs: ExprInfo, rhs: ExprInfo) -> Expr {
    let a = Label::new();
    let end = Label::new();

    Expr::Seq(
        vec![
            Stmt::JumpIfFalse(a, copy(lhs.ir, &lhs.ty)),
            Stmt::JumpIfFalse(a, copy(rhs.ir, &rhs.ty)),
            Stmt::Push(Expr::True),
            Stmt::Jump(end),
            Stmt::Label(a),
            Stmt::Push(Expr::False),
            Stmt::Label(end),
        ],
        box Expr::TOS(type_size_nocheck(&lhs.ty)),
    )
}

//   lhs
//   JUMP_IF_TRUE a
//   rhs
//   JUMP_IF_TRUE a
//   FALSE
//   JUMP end
// a:
//   TRUE
// end:
pub fn binop_or(lhs: ExprInfo, rhs: ExprInfo) -> Expr {
    let a = Label::new();
    let end = Label::new();

    Expr::Seq(
        vec![
            Stmt::JumpIfTrue(a, copy(lhs.ir, &lhs.ty)),
            Stmt::JumpIfTrue(a, copy(rhs.ir, &rhs.ty)),
            Stmt::Push(Expr::False),
            Stmt::Jump(end),
            Stmt::Label(a),
            Stmt::Push(Expr::True),
            Stmt::Label(end),
        ],
        box Expr::TOS(type_size_nocheck(&lhs.ty)),
    )
}

pub fn binop(binop: BinOp, lhs: ExprInfo, rhs: ExprInfo) -> Expr {
    let binop = match binop {
        BinOp::Add => IRBinOp::Add,
        BinOp::Sub => IRBinOp::Sub,
        BinOp::Mul => IRBinOp::Mul,
        BinOp::Div => IRBinOp::Div,
        BinOp::LessThan => IRBinOp::LessThan,
        BinOp::LessThanOrEqual => IRBinOp::LessThanOrEqual,
        BinOp::GreaterThan => IRBinOp::GreaterThan,
        BinOp::GreaterThanOrEqual => IRBinOp::GreaterThanOrEqual,
        BinOp::Equal => IRBinOp::Equal,
        BinOp::NotEqual => IRBinOp::NotEqual,
        binop => panic!("unexpected binop: {:?}", binop),
    };

    Expr::BinOp(
        binop,
        box Expr::Copy(box lhs.ir, type_size_nocheck(&lhs.ty)),
        box Expr::Copy(box rhs.ir, type_size_nocheck(&rhs.ty)),
    )
}

pub fn call(return_ty: &Type, func: ExprInfo, arg: ExprInfo) -> Expr {
    Expr::Call(
        box copy(func.ir, &func.ty),
        box copy(arg.ir, &arg.ty),
        type_size_nocheck(return_ty),
    )
}

pub fn address(expr: ExprInfo) -> Expr {
    let mut ir = expr.ir;

    if let Type::App(TypeCon::InHeap, _) = &expr.ty {
        // Dereference if `expr.ty` is InHeap
        ir = Expr::Copy(box ir, type_size_nocheck(&expr.ty));
    }

    Expr::Pointer(box ir)
}

pub fn address_no_lvalue(expr: ExprInfo, loc: &RelativeVariableLoc) -> Expr {
    Expr::Seq(
        vec![Stmt::Store(
            loc.as_ir_loc(),
            Expr::Alloc(box copy(expr.ir, &expr.ty)),
        )],
        box Expr::Pointer(box Expr::LoadCopy(loc.as_ir_loc(), 1)),
    )
}

pub fn dereference(expr: ExprInfo) -> Expr {
    Expr::Dereference(box Expr::Copy(box expr.ir, type_size_nocheck(&expr.ty)))
}

pub fn negative(expr: ExprInfo) -> Expr {
    Expr::Negative(box Expr::Copy(box expr.ir, type_size_nocheck(&expr.ty)))
}

pub fn copy_in_heap(expr: Expr, ty: &Type, inner_ty: &Type) -> Expr {
    // **expr
    Expr::Copy(
        box Expr::Dereference(box Expr::Copy(box expr, type_size_nocheck(ty))),
        type_size_nocheck(inner_ty),
    )
}

pub fn if_expr(cond: ExprInfo, then: ExprInfo) -> Expr {
    let end = Label::new();

    Expr::Seq(
        vec![
            Stmt::JumpIfFalse(end, copy(cond.ir, &cond.ty)),
            Stmt::Discard(copy(then.ir, &then.ty)),
            Stmt::Label(end),
        ],
        box Expr::Unit,
    )
}

pub fn if_else_expr(cond: ExprInfo, then: ExprInfo, els: ExprInfo) -> Expr {
    let elsl = Label::new();
    let end = Label::new();

    Expr::Seq(
        vec![
            Stmt::JumpIfFalse(elsl, copy(cond.ir, &cond.ty)),
            Stmt::Push(copy(then.ir, &then.ty)),
            Stmt::Jump(end),
            Stmt::Label(elsl),
            Stmt::Push(copy(els.ir, &then.ty)),
            Stmt::Label(end),
        ],
        box Expr::TOS(type_size_nocheck(&then.ty)),
    )
}

pub fn array_to_slice(array: Expr, size: usize) -> Expr {
    // TODO: Add support for rvalue

    Expr::Alloc(box Expr::Record(vec![array, Expr::Int(size as i64)]))
}

pub fn slice(list: ExprInfo, start: ExprInfo, end: ExprInfo, elem_size: usize) -> Expr {
    // TODO: Add support for rvalue

    // alloc([
    //     &list + start * elem_size,
    //     end - start,
    // ])
    Expr::Alloc(box Expr::Record(vec![
        Expr::Offset(
            box list.ir,
            box Expr::BinOp(
                IRBinOp::Mul,
                box copy(start.ir.clone(), &start.ty),
                box Expr::Int(elem_size as i64),
            ),
        ),
        Expr::BinOp(
            IRBinOp::Sub,
            box copy(end.ir, &end.ty),
            box copy(start.ir, &start.ty),
        ),
    ]))
}

// Statement

pub fn expr_stmt(expr: ExprInfo) -> CodeBuf {
    let mut stmts = CodeBuf::new();
    stmts.push(Stmt::Discard(expr.ir));
    stmts
}

pub fn while_stmt(cond: ExprInfo, body: CodeBuf) -> CodeBuf {
    let begin = Label::new();
    let end = Label::new();

    let mut stmts = CodeBuf::new();
    stmts.push(Stmt::Label(begin));
    stmts.push(Stmt::JumpIfFalse(end, copy(cond.ir, &cond.ty)));
    stmts.append(body);
    stmts.push(Stmt::Jump(begin));
    stmts.push(Stmt::Label(end));

    stmts
}

pub fn bind_stmt(loc: &RelativeVariableLoc, expr: ExprInfo) -> CodeBuf {
    let size = if let Type::App(TypeCon::InHeap, types) = &expr.ty {
        type_size_nocheck(&types[0])
    } else {
        type_size_nocheck(&expr.ty)
    };

    let mut expr_ir = Expr::Copy(box expr.ir, size);
    if let Type::App(TypeCon::InHeap, _) = &expr.ty {
        expr_ir = Expr::Alloc(box expr_ir);
    }

    let mut stmts = CodeBuf::new();
    stmts.push(Stmt::Store(loc.as_ir_loc(), expr_ir));
    stmts
}

pub fn assign_stmt(lhs: ExprInfo, rhs: ExprInfo, is_in_heap: bool) -> CodeBuf {
    let lhs_ty = lhs.ty;
    let mut lhs = lhs.ir;
    if is_in_heap {
        // When is_in_heap is true, lhs is a pointer.
        lhs = copy(lhs, &lhs_ty);
    }

    let mut stmts = CodeBuf::new();
    stmts.push(Stmt::StoreFromRef(lhs, copy(rhs.ir, &rhs.ty)));
    stmts
}

pub fn return_stmt(expr: Option<ExprInfo>, return_ty: &Type) -> CodeBuf {
    let expr = expr.map(|expr| copy(expr.ir, return_ty));

    let mut stmts = CodeBuf::new();
    stmts.push(Stmt::Return(expr));
    stmts
}
