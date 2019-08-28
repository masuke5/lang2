use std::collections::HashMap;

use crate::ast::*;
use crate::env::*;

pub struct Executor<'a> {
    var_map: HashMap<&'a str, Value>,
    functions: HashMap<&'a str, Function<'a>>,
    return_value: Option<Value>,
}

impl<'a> Executor<'a> {
    pub fn new() -> Self {
        Self {
            var_map: HashMap::new(),
            functions: HashMap::new(),
            return_value: None,
        }
    }

    fn run_binop(&mut self, binop: BinOp, lhs: Expr<'a>, rhs: Expr<'a>) -> Value {
        let left = self.run_expr(lhs).int();
        let right = self.run_expr(rhs).int();

        match binop {
            BinOp::Add => Value::Int(left + right),
            BinOp::Sub => Value::Int(left - right),
            BinOp::Mul => Value::Int(left * right),
            BinOp::Div => Value::Int(left / right),
            BinOp::Equal => Value::Bool(left == right),
            BinOp::NotEqual => Value::Bool(left != right),
            BinOp::LessThan => Value::Bool(left < right),
            BinOp::LessThanOrEqual => Value::Bool(left <= right),
            BinOp::GreaterThan => Value::Bool(left > right),
            BinOp::GreaterThanOrEqual => Value::Bool(left >= right),
        }
    }

    fn run_call(&mut self, name: &str, args: impl Iterator<Item = Expr<'a>>) -> Value {
        let func = self.functions[name].clone();

        for (i, arg) in args.enumerate() {
            let param_name = func.params[i];
            let value = self.run_expr(arg);
            self.var_map.insert(param_name, value);
        }
        
        match name {
            "printi" => stdlib::printi(self.var_map["n"].clone()),
            "printlf" => stdlib::printlf(),
            _ => { self.run_stmt(func.stmt); },
        }

        self.return_value.clone().unwrap()
    }

    fn run_expr(&mut self, expr: Expr<'a>) -> Value {
        #[allow(unreachable_patterns)]
        match expr {
            Expr::Literal(Literal::Number(n)) => Value::Int(n),
            Expr::Literal(Literal::True) => Value::Bool(true),
            Expr::Literal(Literal::False) => Value::Bool(false),
            Expr::BinOp(binop, lhs, rhs) => self.run_binop(binop, lhs.kind, rhs.kind),
            Expr::Variable(name) => self.var_map[name].clone(),
            Expr::Call(name, args) => self.run_call(name, args.into_iter().map(|expr| expr.kind)),
            _ => unimplemented!(),
        }
    }

    fn run_bind_stmt(&mut self, name: &'a str, expr: Expr<'a>) {
        let value = self.run_expr(expr);
        self.var_map.insert(name, value);
    }

    fn run_return(&mut self, expr: Expr<'a>) {
        let value = self.run_expr(expr);
        self.return_value = Some(value);
    }

    fn run_if(&mut self, cond: Expr<'a>, stmt: Stmt<'a>) -> bool {
        let cond = self.run_expr(cond);

        if cond.bool() {
            self.run_stmt(stmt)
        } else {
            false
        }
    }

    fn run_stmt(&mut self, stmt: Stmt<'a>) -> bool {
        #[allow(unreachable_patterns)]
        match stmt {
            Stmt::Bind(name, expr) => self.run_bind_stmt(name, expr.kind),
            Stmt::Expr(expr) => { self.run_expr(expr.kind); },
            Stmt::Block(stmts) => {
                for stmt in stmts {
                    if self.run_stmt(stmt.kind) {
                        return true;
                    }
                }
            },
            Stmt::Return(expr) => {
                self.run_return(expr.kind);
                return true;
            },
            Stmt::If(cond, stmt) => return self.run_if(cond.kind, stmt.kind),
            _ => unimplemented!(),
        };

        false
    }

    fn run_toplevel(&mut self, toplevel: TopLevel<'a>) -> bool {
        match toplevel {
            TopLevel::Stmt(stmt) => self.run_stmt(stmt.kind),
            TopLevel::Function(name, params, _, stmt) => {
                self.functions.insert(name, Function {
                    stmt: stmt.kind,
                    params: params.into_iter().map(|(name, _)| name).collect() 
                });
                false
            },
        }
    }

    pub fn exec(&mut self, program: Program<'a>) -> i64 {
        self.functions.insert("printi", Function {
            params: vec!["n"],
            stmt: Stmt::Block(Vec::new()),
        });
        self.functions.insert("printlf", Function {
            params: Vec::new(),
            stmt: Stmt::Block(Vec::new()),
        });

        for toplevel in program.top {
            if self.run_toplevel(toplevel.kind) {
                break;
            }
        }

        self.return_value.clone().unwrap_or(Value::Int(0)).int()
    }
}

mod stdlib {
    use crate::env::*;

    pub fn printi(n: Value) {
        print!("{}", n.int());
    }

    pub fn printlf() {
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    #[test]
    fn exec() {
        let lexer = Lexer::new(r#"
            fn add(a: int, b: int): int {
                return a + b;
            }

            let foo = 10 + 3 * 5 + 20;
            let bar = 30;
            let baz = add(foo, bar);
            return baz;"#);
        let tokens = lexer.lex().unwrap();
        let parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        let mut executor = Executor::new();
        let result = executor.exec(program);

        assert_eq!(result, 75);
    }
}
