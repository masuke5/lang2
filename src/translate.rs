use crate::ast::BinOp;
use crate::ir::{BinOp as IRBinOp, CodeBuf, Expr, Label, Stmt, VariableLoc};
use crate::sema::ExprInfo;
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

pub fn field(
    loc: Option<RelativeVariableLoc>,
    is_in_heap: bool,
    is_pointer: bool,
    comp_expr: ExprInfo,
    offset: usize,
) -> Expr {
    let mut expr = comp_expr.ir;

    if let Some(loc) = loc {
        expr = Expr::Seq(
            vec![Stmt::Store(loc.as_ir_loc(), expr)],
            box (Expr::LoadRef(loc.as_ir_loc())),
        );
    }

    if is_in_heap {
        if is_pointer {
            expr = Expr::Copy(box (expr), 1);
        } else {
            expr = Expr::Copy(box (expr), type_size_nocheck(&comp_expr.ty));
        }
    }

    if is_pointer {
        expr = Expr::Copy(box (expr), type_size_nocheck(&comp_expr.ty));
    }

    if let Type::App(TypeCon::Wrapped, _) = &comp_expr.ty {
        expr = Expr::Copy(box (expr), type_size_nocheck(&comp_expr.ty));
    }

    Expr::Offset(box (expr), box (Expr::Int(offset as i64)))
}

pub fn subscript(
    loc: Option<RelativeVariableLoc>,
    is_in_heap: bool,
    is_pointer: bool,
    expr: ExprInfo,
    subscript_expr: ExprInfo,
    element_ty: &Type,
) -> Expr {
    let mut ir = expr.ir;

    if let Some(loc) = loc {
        ir = Expr::Seq(
            vec![Stmt::Store(loc.as_ir_loc(), ir)],
            box (Expr::LoadRef(loc.as_ir_loc())),
        );
    }

    if is_in_heap {
        if is_pointer {
            ir = Expr::Copy(box (ir), 1);
        } else {
            ir = Expr::Copy(box (ir), type_size_nocheck(&expr.ty));
        }
    }

    if is_pointer {
        ir = Expr::Copy(box ir, type_size_nocheck(&expr.ty));
    }

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

pub fn func_pos(module_id: Option<u16>, func_id: u16) -> Expr {
    Expr::Record(vec![
        Expr::FuncPos(module_id.map(|m| m as usize), func_id as usize),
        Expr::EP,
    ])
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
            Stmt::JumpIfFalse(a, lhs.ir),
            Stmt::JumpIfFalse(a, rhs.ir),
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
            Stmt::JumpIfTrue(a, lhs.ir),
            Stmt::JumpIfTrue(a, rhs.ir),
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

pub fn arg(exprs: &mut Vec<Expr>, arg_expr: ExprInfo) {
    exprs.push(Expr::Copy(box arg_expr.ir, type_size_nocheck(&arg_expr.ty)));
}

pub fn call(return_ty: &Type, func_expr: Expr, arg_expr: Expr) -> Expr {
    Expr::Call(box func_expr, box arg_expr, type_size_nocheck(return_ty))
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
        vec![Stmt::Store(loc.as_ir_loc(), Expr::Alloc(box expr.ir))],
        box Expr::Pointer(box Expr::LoadCopy(
            loc.as_ir_loc(),
            type_size_nocheck(&expr.ty),
        )),
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
    let mut expr_ir = expr.ir;
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
