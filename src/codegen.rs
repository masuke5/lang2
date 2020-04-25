fn scan_expr(expr: &mut Expr, mut func: impl FnMut(&mut Expr)) {
    func(expr);

    match expr {
        Expr::Pointer(expr)
        | Expr::Dereference(expr)
        | Expr::Copy(expr, _)
        | Expr::Duplicate(expr, _)
        | Expr::Negative(expr)
        | Expr::Alloc(expr)
        | Expr::Wrap(expr)
        | Expr::Unwrap(expr, _) => {
            remove_redundant_copy(expr);
        }
        Expr::Offset(expr1, expr2) | Expr::BinOp(_, expr1, expr2) | Expr::Call(expr1, expr2, _) => {
            remove_redundant_copy(expr1);
            remove_redundant_copy(expr2);
        }
        Expr::Record(exprs) => {
            for expr in exprs {
                remove_redundant_copy(expr);
            }
        }
        _ => {}
    }
}

fn remove_redundant_copy(expr: &mut Expr) {
    scan_expr(expr, |expr| loop {
        if let Expr::Copy(inner_expr, _) = expr {
            match inner_expr.as_mut() {
                Expr::LoadRef(..) | Expr::Offset(..) | Expr::Dereference(..) => break,
                _ => {
                    *expr = mem::replace(inner_expr.as_mut(), Expr::Unit);
                }
            }
        } else {
            break;
        }
    });
}

fn remove_seq(stmts: Vec<Stmt>) -> Vec<Stmt> {
    fn scan(expr: &mut Expr, new_stmts: &mut Vec<Stmt>) {
        scan_expr(expr, |expr| loop {
            match expr {
                Expr::Seq(stmts, inner_expr) => {
                    new_stmts.append(stmts);
                    *expr = mem::replace(inner_expr.as_mut(), Expr::Unit);
                }
                _ => break,
            }
        })
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

#[cfg(test)]
mod tests {
    use super::*;

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
                vec![Stmt::Push(Expr::Int(2))],
                box Expr::Seq(
                    vec![Stmt::Label(label), Stmt::Push(Expr::Int(3))],
                    box Expr::TOS,
                ),
            )),
        ];
        let stmts = remove_seq(stmts);
        assert_eq!(
            stmts,
            vec![
                Stmt::Discard(Expr::Int(1)),
                Stmt::Push(Expr::Int(2)),
                Stmt::Label(label),
                Stmt::Push(Expr::Int(3)),
                Stmt::Discard(Expr::TOS),
            ],
        );
    }
}
