use std::mem;

use rustc_hash::FxHashMap;

use crate::ir::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizeOption {
    pub constant_folding: bool,
}

fn try_convert_to_int(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Int(n) => Some(*n),
        Expr::BinOp(binop, lhs, rhs) => match (binop, lhs.as_ref(), rhs.as_ref()) {
            (BinOp::Mul, _, Expr::Int(0)) | (BinOp::Mul, Expr::Int(0), _) => Some(0),
            (BinOp::Add, _, _) => Some(try_convert_to_int(lhs)? + try_convert_to_int(rhs)?),
            (BinOp::Sub, _, _) => Some(try_convert_to_int(lhs)? - try_convert_to_int(rhs)?),
            (BinOp::Mul, _, _) => Some(try_convert_to_int(lhs)? * try_convert_to_int(rhs)?),
            (BinOp::Div, _, _) => Some(try_convert_to_int(lhs)? / try_convert_to_int(rhs)?),
            _ => None,
        },
        Expr::Negative(expr) => Some(-try_convert_to_int(expr)?),
        _ => None,
    }
}

fn try_convert_to_bool(expr: &Expr) -> Option<bool> {
    match expr {
        Expr::True => Some(true),
        Expr::False => Some(false),
        Expr::BinOp(binop, lhs, rhs) => match binop {
            BinOp::LessThan => Some(try_convert_to_int(lhs)? < try_convert_to_int(rhs)?),
            BinOp::LessThanOrEqual => Some(try_convert_to_int(lhs)? <= try_convert_to_int(rhs)?),
            BinOp::GreaterThan => Some(try_convert_to_int(lhs)? > try_convert_to_int(rhs)?),
            BinOp::GreaterThanOrEqual => Some(try_convert_to_int(lhs)? >= try_convert_to_int(rhs)?),
            BinOp::Equal => Some(try_convert_to_int(lhs)? == try_convert_to_int(rhs)?),
            BinOp::NotEqual => Some(try_convert_to_int(lhs)? != try_convert_to_int(rhs)?),
            _ => None,
        },
        _ => None,
    }
}

fn scan_expr(expr: &mut Expr, mut func: impl FnMut(&mut Expr)) {
    let mut stack: Vec<*mut Expr> = vec![expr];

    while let Some(expr) = stack.pop() {
        let expr = unsafe { &mut *expr };
        func(expr);

        match expr {
            Expr::Pointer(expr)
            | Expr::Dereference(expr)
            | Expr::Copy(expr, _)
            | Expr::Duplicate(expr, _)
            | Expr::Negative(expr)
            | Expr::Alloc(expr)
            | Expr::Wrap(expr)
            | Expr::Unwrap(expr, _)
            | Expr::SeqId(_, expr) => {
                stack.push(expr.as_mut());
            }
            Expr::Offset(expr1, expr2)
            | Expr::BinOp(_, expr1, expr2)
            | Expr::Call(expr1, expr2, _) => {
                stack.push(expr1.as_mut());
                stack.push(expr2.as_mut());
            }
            Expr::Record(exprs) => {
                for expr in exprs {
                    stack.push(expr);
                }
            }
            _ => {}
        }
    }
}

fn scan_expr_in_stmt(stmt: &mut Stmt, func: impl FnMut(&mut Expr) + Clone) {
    match stmt {
        Stmt::Discard(expr)
        | Stmt::Store(_, expr)
        | Stmt::Return(Some(expr))
        | Stmt::JumpIfFalse(_, expr)
        | Stmt::JumpIfTrue(_, expr)
        | Stmt::Push(expr) => scan_expr(expr, func),
        Stmt::StoreFromRef(expr1, expr2) => {
            scan_expr(expr1, func.clone());
            scan_expr(expr2, func);
        }
        _ => {}
    }
}

fn fold_constant(expr: &mut Expr) {
    scan_expr(expr, |expr| {
        if let Expr::Seq(..) = expr {
            panic!();
        }

        if let Some(n) = try_convert_to_int(expr) {
            *expr = Expr::Int(n);
        }

        if let Some(b) = try_convert_to_bool(expr) {
            let new_expr = if b { Expr::True } else { Expr::False };
            *expr = new_expr;
        }
    });
}

fn remove_redundant_expr(expr: &mut Expr) {
    #[allow(clippy::single_match)]
    scan_expr(expr, |expr| match expr {
        Expr::BinOp(binop, lhs, rhs) => match (binop, lhs.as_mut(), rhs.as_mut()) {
            (BinOp::Mul, term, Expr::Int(1)) | (BinOp::Mul, Expr::Int(1), term) => {
                *expr = mem::replace(term, Expr::Unit)
            }
            (BinOp::Add, term, Expr::Int(0)) | (BinOp::Add, Expr::Int(0), term) => {
                *expr = mem::replace(term, Expr::Unit)
            }
            _ => {}
        },
        _ => {}
    });
}

fn remove_redundant_copy(expr: &mut Expr) {
    scan_expr(expr, |expr| {
        if let Expr::Seq(..) = expr {
            panic!();
        }

        while let Expr::Copy(inner_expr, _) = expr {
            match inner_expr.as_mut() {
                Expr::LoadRef(..) | Expr::Offset(..) | Expr::Dereference(..) => break,
                _ => {
                    *expr = mem::replace(inner_expr.as_mut(), Expr::Unit);
                }
            }
        }
    });
}

fn expr_is_seq(expr: &Expr) -> bool {
    match expr {
        Expr::Seq(..) => true,
        _ => false,
    }
}

fn restore_seq(stmts: Vec<Stmt>) -> Vec<Stmt> {
    let mut new_stmts = Vec::new();
    let mut seq_stack = Vec::new();
    let mut seq_stmts = FxHashMap::default();

    // Insert statements to `seq_stmts` per ID
    for stmt in stmts {
        match stmt {
            Stmt::BeginSeq(id) => {
                seq_stack.push(id);
                seq_stmts.insert(id, Vec::new());
            }
            Stmt::EndSeq(_) => {
                seq_stack.pop().unwrap();
            }
            stmt => match seq_stack.last() {
                Some(seq_id) => seq_stmts.get_mut(seq_id).unwrap().push(stmt),
                None => new_stmts.push(stmt),
            },
        }
    }

    fn replace_seq_id(expr: &mut Expr, seq_stmts: &mut FxHashMap<SeqId, Vec<Stmt>>) {
        scan_expr(expr, |expr| {
            if let Expr::SeqId(id, inner_expr) = expr {
                let mut stmts = seq_stmts.remove(id).unwrap();
                for stmt in &mut stmts {
                    replace_seq_id_in_stmt(stmt, seq_stmts);
                }

                let mut inner_expr = mem::replace(inner_expr.as_mut(), Expr::Unit);
                replace_seq_id(&mut inner_expr, seq_stmts);

                *expr = Expr::Seq(stmts, box inner_expr);
            }
        });
    }

    fn replace_seq_id_in_stmt(stmt: &mut Stmt, seq_stmts: &mut FxHashMap<SeqId, Vec<Stmt>>) {
        match stmt {
            Stmt::Discard(expr)
            | Stmt::Store(_, expr)
            | Stmt::Return(Some(expr))
            | Stmt::JumpIfFalse(_, expr)
            | Stmt::JumpIfTrue(_, expr)
            | Stmt::Push(expr) => replace_seq_id(expr, seq_stmts),
            Stmt::StoreFromRef(expr1, expr2) => {
                replace_seq_id(expr1, seq_stmts);
                replace_seq_id(expr2, seq_stmts);
            }
            _ => {}
        }
    }

    for stmt in &mut new_stmts {
        replace_seq_id_in_stmt(stmt, &mut seq_stmts);
    }

    new_stmts
}

fn remove_seq(stmts: Vec<Stmt>) -> Vec<Stmt> {
    fn remove_seq(stmts: Vec<Stmt>) -> Vec<Stmt> {
        fn add_stmts(new_stmts: &mut Vec<Stmt>, stmts_to_add: &mut Vec<Stmt>, expr: Expr) -> Expr {
            let id = SeqId::new();
            new_stmts.push(Stmt::BeginSeq(id));
            new_stmts.append(stmts_to_add);
            new_stmts.push(Stmt::EndSeq(id));

            Expr::SeqId(id, box expr)
        }

        fn scan(mut expr: &mut Expr, new_stmts: &mut Vec<Stmt>) {
            loop {
                match expr {
                    Expr::Seq(stmts, inner_expr) => {
                        let inner_expr = mem::replace(inner_expr.as_mut(), Expr::Unit);
                        let new_expr = add_stmts(new_stmts, stmts, inner_expr);
                        *expr = new_expr;
                    }
                    Expr::SeqId(_, inner_expr) => {
                        expr = inner_expr;
                    }
                    _ => break,
                }
            }

            match expr {
                Expr::Pointer(expr)
                | Expr::Dereference(expr)
                | Expr::Copy(expr, _)
                | Expr::Duplicate(expr, _)
                | Expr::Negative(expr)
                | Expr::Alloc(expr)
                | Expr::Wrap(expr)
                | Expr::Unwrap(expr, _) => {
                    scan(expr, new_stmts);
                }
                Expr::Offset(expr1, expr2)
                | Expr::BinOp(_, expr1, expr2)
                | Expr::Call(expr1, expr2, _) => {
                    if !expr_is_seq(expr1) && expr_is_seq(expr2) {
                        let size = expr1.size();
                        let expr = mem::replace(expr1.as_mut(), Expr::Unit);
                        let new_expr =
                            add_stmts(new_stmts, &mut vec![Stmt::Push(expr)], Expr::TOS(size));
                        **expr1 = new_expr;
                    } else {
                        scan(expr1, new_stmts);
                    }

                    scan(expr2, new_stmts);
                }
                Expr::Record(exprs) => {
                    let has_seq = exprs.iter().any(expr_is_seq);
                    for expr in exprs {
                        if !expr_is_seq(expr) && has_seq {
                            let size = expr.size();
                            let expr2 = mem::replace(expr, Expr::Unit);
                            let new_expr =
                                add_stmts(new_stmts, &mut vec![Stmt::Push(expr2)], Expr::TOS(size));
                            *expr = new_expr;
                        } else {
                            scan(expr, new_stmts);
                        }
                    }
                }
                _ => {}
            }
        }

        let mut new_stmts: Vec<Stmt> = Vec::new();

        for mut stmt in stmts {
            match &mut stmt {
                Stmt::Discard(expr)
                | Stmt::Store(_, expr)
                | Stmt::Return(Some(expr))
                | Stmt::JumpIfFalse(_, expr)
                | Stmt::JumpIfTrue(_, expr)
                | Stmt::Push(expr) => scan(expr, &mut new_stmts),
                Stmt::StoreFromRef(expr1, expr2) => {
                    scan(expr1, &mut new_stmts);
                    scan(expr2, &mut new_stmts);
                }
                _ => {}
            };

            new_stmts.push(stmt);
        }

        new_stmts
    }

    let mut stmts = stmts;
    loop {
        let prev_len = stmts.len();
        stmts = remove_seq(stmts);

        if prev_len == stmts.len() {
            break;
        }
    }

    stmts
}

pub fn optimize(module: &mut Module, option: &OptimizeOption) {
    for (id, (_, func)) in module.functions.iter_mut().enumerate() {
        let func_body = mem::replace(&mut func.body, Expr::Unit);

        // Don't return in main function
        let stmt = if id == 0 {
            Stmt::Discard(func_body)
        } else {
            Stmt::Return(Some(func_body))
        };

        // Remove all Seq
        let mut stmts = remove_seq(vec![stmt]);

        // Remove redundant copies
        for stmt in &mut stmts {
            scan_expr_in_stmt(stmt, remove_redundant_copy);
        }

        // Constant folding
        if option.constant_folding {
            for stmt in &mut stmts {
                scan_expr_in_stmt(stmt, fold_constant);
            }
        }

        // Remove remove redundant expressions
        for stmt in &mut stmts {
            scan_expr_in_stmt(stmt, remove_redundant_expr);
        }

        // Remove redundant copies
        for stmt in &mut stmts {
            scan_expr_in_stmt(stmt, remove_redundant_copy);
        }

        // TODO: More optimization

        // Restore all Seq
        let stmts = restore_seq(stmts);

        func.body = Expr::Seq(stmts, box Expr::Unit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq_id(id: usize) -> SeqId {
        unsafe { SeqId::from_raw(id) }
    }

    fn get_seq_id(stmt: &Stmt) -> SeqId {
        match stmt {
            Stmt::BeginSeq(id) => *id,
            Stmt::EndSeq(id) => *id,
            _ => panic!("the statement `{}` is not BeginSeq and EndSeq", stmt),
        }
    }

    #[test]
    fn test_remove_redundant_copy() {
        let mut expr = Expr::Copy(box Expr::Int(10), 0);
        remove_redundant_copy(&mut expr);
        assert_eq!(expr, Expr::Int(10));

        let mut expr = Expr::Copy(box Expr::Copy(box Expr::Int(20), 0), 0);
        remove_redundant_copy(&mut expr);
        assert_eq!(expr, Expr::Int(20));

        let mut expr = Expr::Pointer(box Expr::Copy(box Expr::Int(10), 0));
        remove_redundant_copy(&mut expr);
        assert_eq!(expr, Expr::Pointer(box Expr::Int(10)));

        let mut expr = Expr::Pointer(box Expr::Copy(box Expr::Dereference(box Expr::Int(10)), 0));
        remove_redundant_copy(&mut expr);
        assert_eq!(
            expr,
            Expr::Pointer(box Expr::Copy(box Expr::Dereference(box Expr::Int(10)), 0))
        );
    }

    #[test]
    fn test_remove_seq() {
        let label = Label::new();
        let stmts = vec![
            Stmt::Discard(Expr::Int(1)),
            Stmt::Discard(Expr::Seq(
                vec![
                    Stmt::Push(Expr::Int(2)),
                    Stmt::Discard(Expr::Seq(vec![Stmt::Jump(label)], box Expr::Int(4))),
                ],
                box Expr::Seq(
                    vec![Stmt::Label(label), Stmt::Push(Expr::Int(3))],
                    box Expr::TOS(1),
                ),
            )),
        ];
        let stmts = remove_seq(stmts);
        let id0 = get_seq_id(&stmts[1]);
        let id1 = get_seq_id(&stmts[8]);
        let id2 = get_seq_id(&stmts[3]);

        assert_eq!(
            stmts,
            vec![
                Stmt::Discard(Expr::Int(1)),
                Stmt::BeginSeq(id0),
                Stmt::Push(Expr::Int(2)),
                Stmt::BeginSeq(id2),
                Stmt::Jump(label),
                Stmt::EndSeq(id2),
                Stmt::Discard(Expr::SeqId(id2, box Expr::Int(4))),
                Stmt::EndSeq(id0),
                Stmt::BeginSeq(id1),
                Stmt::Label(label),
                Stmt::Push(Expr::Int(3)),
                Stmt::EndSeq(id1),
                Stmt::Discard(Expr::SeqId(id0, box Expr::SeqId(id1, box Expr::TOS(1)))),
            ],
        );
    }

    #[test]
    fn test_remove_seq_with_binop() {
        let stmts = vec![Stmt::Discard(Expr::BinOp(
            BinOp::Add,
            box Expr::Int(5),
            box Expr::Seq(vec![Stmt::Push(Expr::Int(6))], box Expr::TOS(1)),
        ))];
        let stmts = remove_seq(stmts);
        let id0 = get_seq_id(&stmts[0]);
        let id1 = get_seq_id(&stmts[3]);

        assert_eq!(
            stmts,
            vec![
                Stmt::BeginSeq(id0),
                Stmt::Push(Expr::Int(5)),
                Stmt::EndSeq(id0),
                Stmt::BeginSeq(id1),
                Stmt::Push(Expr::Int(6)),
                Stmt::EndSeq(id1),
                Stmt::Discard(Expr::BinOp(
                    BinOp::Add,
                    box Expr::SeqId(id0, box Expr::TOS(1)),
                    box Expr::SeqId(id1, box Expr::TOS(1)),
                )),
            ]
        );

        let stmts = vec![Stmt::Discard(Expr::BinOp(
            BinOp::Add,
            box Expr::Seq(vec![Stmt::Push(Expr::Int(7))], box Expr::TOS(1)),
            box Expr::Seq(vec![Stmt::Push(Expr::Int(8))], box Expr::TOS(1)),
        ))];
        let stmts = remove_seq(stmts);
        let id0 = get_seq_id(&stmts[0]);
        let id1 = get_seq_id(&stmts[3]);

        assert_eq!(
            stmts,
            vec![
                Stmt::BeginSeq(id0),
                Stmt::Push(Expr::Int(7)),
                Stmt::EndSeq(id0),
                Stmt::BeginSeq(id1),
                Stmt::Push(Expr::Int(8)),
                Stmt::EndSeq(id1),
                Stmt::Discard(Expr::BinOp(
                    BinOp::Add,
                    box Expr::SeqId(id0, box Expr::TOS(1)),
                    box Expr::SeqId(id1, box Expr::TOS(1)),
                )),
            ]
        );
    }

    #[test]
    fn test_restore_seq() {
        let label = Label::new();
        let stmts = restore_seq(vec![
            Stmt::Discard(Expr::Int(1)),
            Stmt::BeginSeq(seq_id(0)),
            Stmt::Push(Expr::Int(2)),
            Stmt::BeginSeq(seq_id(2)),
            Stmt::Jump(label),
            Stmt::EndSeq(seq_id(2)),
            Stmt::Discard(Expr::SeqId(seq_id(2), box Expr::Int(4))),
            Stmt::EndSeq(seq_id(0)),
            Stmt::BeginSeq(seq_id(1)),
            Stmt::Label(label),
            Stmt::Push(Expr::Int(3)),
            Stmt::EndSeq(seq_id(1)),
            Stmt::Discard(Expr::SeqId(
                seq_id(0),
                box Expr::SeqId(seq_id(1), box Expr::TOS(1)),
            )),
        ]);

        assert_eq!(
            stmts,
            vec![
                Stmt::Discard(Expr::Int(1)),
                Stmt::Discard(Expr::Seq(
                    vec![
                        Stmt::Push(Expr::Int(2)),
                        Stmt::Discard(Expr::Seq(vec![Stmt::Jump(label)], box Expr::Int(4))),
                    ],
                    box Expr::Seq(
                        vec![Stmt::Label(label), Stmt::Push(Expr::Int(3))],
                        box Expr::TOS(1),
                    ),
                )),
            ]
        );
    }
}
