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
}
