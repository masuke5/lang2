use std::convert::TryInto;
use std::mem;

use rustc_hash::FxHashMap;

use crate::bytecode::{opcode, Bytecode, BytecodeBuilder, Function as BFunction, InstList};
use crate::ir::*;
use crate::vm::{CALL_STACK_SIZE, SELF_MODULE_ID};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizeOption {
    // Calculation at compile time
    pub calc_at_compile_time: bool,
}

struct StringList {
    strings: FxHashMap<String, usize>,
    next_id: usize,
}

impl StringList {
    fn new() -> Self {
        Self {
            strings: FxHashMap::default(),
            next_id: 0,
        }
    }

    fn get_id(&mut self, s: &str) -> usize {
        if let Some(id) = self.strings.get(s) {
            *id
        } else {
            let id = self.next_id;
            self.strings.insert(s.to_string(), id);
            self.next_id += 1;
            id
        }
    }

    fn all(self) -> Vec<String> {
        let mut strings = vec![String::new(); self.strings.len()];

        for (s, id) in self.strings {
            strings[id] = s;
        }

        strings
    }
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

struct Generator {
    builder: BytecodeBuilder,
    strings: StringList,
    param_size: usize,
}

impl Generator {
    fn new() -> Self {
        Self {
            builder: BytecodeBuilder::new(),
            strings: StringList::new(),
            param_size: 0,
        }
    }

    fn load_copy_if_possible(insts: &mut InstList, loc: &VariableLoc, size: usize) -> bool {
        if let VariableLoc::Local(mut loc) = *loc {
            if loc < 0 {
                loc -= CALL_STACK_SIZE as isize;
            }

            if loc >= -16 && loc <= 15 {
                if size <= 0b111 {
                    let arg = ((loc as i8) << 3) | size as i8;
                    insts.push(opcode::LOAD_COPY, u8::from_le_bytes(arg.to_le_bytes()));
                    return true;
                }
            }
        }

        false
    }

    fn gen_expr(&mut self, expr: &Expr) -> InstList {
        let mut insts = InstList::new();

        match expr {
            Expr::Int(n) => {
                if let Ok(n) = TryInto::<i8>::try_into(*n) {
                    insts.push(opcode::TINY_INT, n.to_le_bytes()[0]);
                } else {
                    let index = self.builder.new_ref_i64(*n);
                    insts.push(opcode::INT, index as u8);
                }
            }
            Expr::String(s) => {
                let id = self.strings.get_id(&s);
                insts.push(opcode::STRING, id as u8);
            }
            Expr::True => insts.push_noarg(opcode::TRUE),
            Expr::False => insts.push_noarg(opcode::FALSE),
            Expr::Null => insts.push_noarg(opcode::NULL),
            Expr::Unit => {}
            Expr::Pointer(expr) | Expr::Dereference(expr) => {
                let expr_insts = self.gen_expr(&expr);
                insts.append(expr_insts);
            }
            Expr::Copy(expr, size) => {
                // Insert instructions only if necessary
                match expr.as_ref() {
                    Expr::LoadRef(loc) if Self::load_copy_if_possible(&mut insts, &loc, *size) => {}
                    Expr::LoadRef(..) | Expr::Dereference(..) | Expr::Offset(..) => {
                        insts.append(self.gen_expr(&expr));
                        insts.push(opcode::COPY, *size as u8);
                    }
                    _ => panic!("not removed redundant copies: {:?}", expr),
                }
            }
            Expr::Offset(expr, offset) => {
                let expr_insts = self.gen_expr(&expr);
                insts.append(expr_insts);

                if let Expr::Int(n) = offset.as_ref() {
                    if *n > 0 {
                        if let Ok(n) = TryInto::<u8>::try_into(*n) {
                            insts.push(opcode::CONST_OFFSET, n);
                        } else {
                            let offset_insts = self.gen_expr(&offset);
                            insts.append(offset_insts);
                            insts.push_noarg(opcode::OFFSET);
                        }
                    }
                } else {
                    let offset_insts = self.gen_expr(&offset);
                    insts.append(offset_insts);
                    insts.push_noarg(opcode::OFFSET);
                }
            }
            Expr::Duplicate(expr, count) => {
                if *count > std::u32::MAX as usize {
                    panic!("too many count: {}", count);
                }

                insts.append(self.gen_expr(expr));

                let arg = ((expr.size() as u64) << 32) | *count as u64 - 1;
                let index = self.builder.new_ref_u64(arg);
                insts.push(opcode::DUPLICATE, index as u8);
            }
            Expr::LoadCopy(loc, size) => {
                if !Self::load_copy_if_possible(&mut insts, &loc, *size) {
                    insts.append(self.gen_expr(&Expr::Copy(box Expr::LoadRef(loc.clone()), *size)));
                }
            }
            Expr::LoadRef(loc) => match loc {
                VariableLoc::Local(loc) if *loc < 0 => {
                    let loc = loc - CALL_STACK_SIZE as isize;
                    insts.push(opcode::LOAD_REF, (loc as i8).to_le_bytes()[0])
                }
                VariableLoc::Local(loc) => {
                    insts.push(opcode::LOAD_REF, (*loc as i8).to_le_bytes()[0])
                }
                VariableLoc::Heap(loc, 0) => {
                    insts.push(opcode::LOAD_HEAP, *loc as u8);
                }
                VariableLoc::Heap(loc, level) if *level <= std::i8::MAX as usize => {
                    insts.push(opcode::TINY_INT, (*level as i8).to_le_bytes()[0]);
                    insts.push(opcode::LOAD_HEAP_TRACE, *loc as u8);
                }
                VariableLoc::Heap(loc, level) => {
                    let index = self.builder.new_ref_i64(*level as i64);
                    insts.push(opcode::INT, index as u8);
                    insts.push(opcode::LOAD_HEAP_TRACE, *loc as u8);
                }
            },
            Expr::BinOp(binop, lhs, rhs) => {
                let lhs = self.gen_expr(&lhs);
                let rhs = self.gen_expr(&rhs);

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
                };

                insts.append(lhs);
                insts.append(rhs);
                insts.push_noarg(opcode);
            }
            Expr::Negative(expr) => {
                insts.append(self.gen_expr(&expr));
                insts.push_noarg(opcode::NEGATIVE);
            }
            Expr::Alloc(expr) => {
                insts.append(self.gen_expr(&expr));

                let size = expr.size();
                insts.push(opcode::ALLOC, size as u8);
            }
            Expr::Record(exprs) => {
                for expr in exprs {
                    insts.append(self.gen_expr(&expr));
                }
            }
            Expr::Wrap(expr) => {
                insts.append(self.gen_expr(&expr));

                let size = expr.size();
                if size > 1 {
                    insts.push(opcode::WRAP, size as u8);
                }
            }
            Expr::Unwrap(expr, size) => {
                insts.append(self.gen_expr(&expr));
                if *size > 1 {
                    insts.push(opcode::UNWRAP, *size as u8);
                }
            }
            Expr::Call(func, arg, rv_size) => {
                // rv = return value
                let mut func = func;
                let mut args = vec![arg];

                while let Expr::Call(new_func, arg, _) = func.as_ref() {
                    args.push(arg);
                    func = new_func;
                }

                if *rv_size > 0 {
                    insts.push(opcode::ZERO, *rv_size as u8);
                }

                for arg in args.into_iter().rev() {
                    insts.append(self.gen_expr(arg));
                }

                match func.as_ref() {
                    Expr::Record(exprs) => match (&exprs[0], &exprs[1]) {
                        (Expr::FuncPos(Some(module_id @ 0..=15), func_id @ 0..=15), Expr::EP) => {
                            let arg = ((*module_id as u8) << 4) | *func_id as u8;
                            insts.push(opcode::CALL_EXTERN, arg);
                        }
                        (Expr::FuncPos(None, func_id), Expr::EP) => {
                            insts.push(opcode::CALL, *func_id as u8);
                        }
                        _ => {
                            insts.append(self.gen_expr(func));
                            insts.push_noarg(opcode::CALL_POS);
                        }
                    },
                    _ => {
                        insts.append(self.gen_expr(func));
                        insts.push_noarg(opcode::CALL_POS);
                    }
                }
            }
            Expr::FuncPos(module_id, func_id) => {
                let module_id = module_id.unwrap_or(SELF_MODULE_ID);
                let arg = ((module_id as u64) << 32) | *func_id as u64;
                let index = self.builder.new_ref_u64(arg);
                insts.push(opcode::INT, index as u8);
            }
            Expr::EP => insts.push_noarg(opcode::EP),
            Expr::Seq(stmts, expr) => {
                for stmt in stmts {
                    self.gen_stmt(&mut insts, stmt);
                }
                insts.append(self.gen_expr(expr));
            }
            Expr::TOS(..) => {}
            Expr::SeqId(..) => panic!(),
        }

        insts
    }

    fn gen_stmt(&mut self, insts: &mut InstList, stmt: &Stmt) {
        match stmt {
            Stmt::Discard(expr) => {
                insts.append(self.gen_expr(expr));
                for _ in 0..expr.size() {
                    insts.push_noarg(opcode::POP);
                }
            }
            Stmt::Store(loc, expr) => {
                insts.append(self.gen_expr(expr));
                insts.append(self.gen_expr(&Expr::LoadRef(loc.clone())));

                if expr.size() > 0 {
                    insts.push(opcode::STORE, expr.size() as u8);
                } else {
                    insts.push_noarg(opcode::POP);
                }
            }
            Stmt::StoreFromRef(ref_expr, expr) => {
                insts.append(self.gen_expr(expr));
                insts.append(self.gen_expr(ref_expr));

                if expr.size() > 0 {
                    insts.push(opcode::STORE, expr.size() as u8);
                } else {
                    insts.push_noarg(opcode::POP);
                }
            }
            Stmt::Return(expr) => {
                if let Some(expr) = expr {
                    let loc = 0 - self.param_size as isize - expr.size() as isize;
                    insts.append(self.gen_expr(expr));

                    if expr.size() > 0 {
                        insts.append(self.gen_expr(&Expr::LoadRef(VariableLoc::Local(loc))));
                        insts.push(opcode::STORE, expr.size() as u8);
                    }
                }

                insts.push_noarg(opcode::RETURN);
            }
            Stmt::Push(expr) => {
                insts.append(self.gen_expr(expr));
            }
            Stmt::Label(label) => {
                insts.add_label(label.as_usize());
            }
            Stmt::Jump(label) => {
                insts.push_jump(opcode::JUMP, label.as_usize());
            }
            Stmt::JumpIfFalse(label, expr) => {
                insts.append(self.gen_expr(expr));
                insts.push_jump(opcode::JUMP_IF_FALSE, label.as_usize());
            }
            Stmt::JumpIfTrue(label, expr) => {
                insts.append(self.gen_expr(expr));
                insts.push_jump(opcode::JUMP_IF_TRUE, label.as_usize());
            }
            Stmt::BeginSeq(..) | Stmt::EndSeq(..) => panic!(),
        }
    }

    fn resolve_labels(&mut self, insts: &mut InstList) {
        let mut jumps: Vec<(usize, usize)> = insts.jumps().iter().map(|(a, b)| (*a, *b)).collect();
        jumps.sort_by_key(|(index, _)| *index);

        let mut jumps = jumps.into_iter().peekable();

        for (i, [opcode, arg]) in insts.insts.iter_mut().enumerate() {
            if let Some((jump_index, label)) = jumps.peek() {
                if i == *jump_index {
                    assert!([opcode::JUMP, opcode::JUMP_IF_TRUE, opcode::JUMP_IF_FALSE]
                        .contains(opcode));

                    let label_loc = insts.labels[label];
                    let relative_loc = label_loc as isize - i as isize;
                    *arg = (relative_loc as i8).to_le_bytes()[0];

                    jumps.next();
                }
            } else {
                break;
            }
        }
    }

    fn generate(mut self, module: Module) -> Bytecode {
        for module in module.imported_modules {
            self.builder.push_module(&module);
        }

        for (id, (name, func)) in module.functions.into_iter().enumerate() {
            self.param_size = func.param_size;

            // Generate instructions
            let mut insts = self.gen_expr(&func.body);

            // Resolve labels
            self.resolve_labels(&mut insts);

            self.builder.insert_function_header(BFunction {
                name: Some(name),
                code_id: id as u16,
                stack_in_heap_size: func.stack_in_heap_size,
                stack_size: func.stack_size,
                param_size: func.param_size,
                pos: 0,
            });
            self.builder.push_function_body(name, insts);
        }

        self.builder.build(&self.strings.all())
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

fn calc_at_compile_time(expr: &mut Expr) {
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

pub fn codegen(mut module: Module, option: &OptimizeOption) -> Bytecode {
    let generator = Generator::new();

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
            match stmt {
                Stmt::Discard(expr)
                | Stmt::Store(_, expr)
                | Stmt::Return(Some(expr))
                | Stmt::JumpIfFalse(_, expr)
                | Stmt::JumpIfTrue(_, expr)
                | Stmt::Push(expr) => remove_redundant_copy(expr),
                Stmt::StoreFromRef(expr1, expr2) => {
                    remove_redundant_copy(expr1);
                    remove_redundant_copy(expr2);
                }
                _ => {}
            }
        }

        // Calculation at comple time
        if option.calc_at_compile_time {
            for stmt in &mut stmts {
                match stmt {
                    Stmt::Discard(expr)
                    | Stmt::Store(_, expr)
                    | Stmt::Return(Some(expr))
                    | Stmt::JumpIfFalse(_, expr)
                    | Stmt::JumpIfTrue(_, expr)
                    | Stmt::Push(expr) => calc_at_compile_time(expr),
                    Stmt::StoreFromRef(expr1, expr2) => {
                        calc_at_compile_time(expr1);
                        calc_at_compile_time(expr2);
                    }
                    _ => {}
                }
            }
        }

        // Remove remove redundant expressions
        for stmt in &mut stmts {
            match stmt {
                Stmt::Discard(expr)
                | Stmt::Store(_, expr)
                | Stmt::Return(Some(expr))
                | Stmt::JumpIfFalse(_, expr)
                | Stmt::JumpIfTrue(_, expr)
                | Stmt::Push(expr) => remove_redundant_expr(expr),
                Stmt::StoreFromRef(expr1, expr2) => {
                    remove_redundant_expr(expr1);
                    remove_redundant_expr(expr2);
                }
                _ => {}
            }
        }

        // Remove redundant copies
        for stmt in &mut stmts {
            match stmt {
                Stmt::Discard(expr)
                | Stmt::Store(_, expr)
                | Stmt::Return(Some(expr))
                | Stmt::JumpIfFalse(_, expr)
                | Stmt::JumpIfTrue(_, expr)
                | Stmt::Push(expr) => remove_redundant_copy(expr),
                Stmt::StoreFromRef(expr1, expr2) => {
                    remove_redundant_copy(expr1);
                    remove_redundant_copy(expr2);
                }
                _ => {}
            }
        }

        // TODO: More optimization

        // Restore all Seq
        let stmts = restore_seq(stmts);

        func.body = Expr::Seq(stmts, box Expr::Unit);
    }

    generator.generate(module)
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
