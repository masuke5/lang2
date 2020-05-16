use std::convert::TryInto;

use rustc_hash::FxHashMap;

use crate::bytecode::{opcode, Bytecode, BytecodeBuilder, Function as BFunction, InstList};
use crate::ir::*;
use crate::vm::{CALL_STACK_SIZE, SELF_MODULE_ID};

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

            if loc >= -16 && loc <= 15 && size <= 0b111 {
                let arg = ((loc as i8) << 3) | size as i8;
                insts.push(opcode::LOAD_COPY, u8::from_le_bytes(arg.to_le_bytes()));
                return true;
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
            Expr::OffsetSlice(slice, offset, size) => {
                insts.append(self.gen_expr(&slice));
                insts.append(self.gen_expr(&offset));
                insts.push(opcode::OFFSET_SLICE, *size as u8);
            }
            Expr::Duplicate(expr, count) => {
                if *count > std::u32::MAX as usize {
                    panic!("too many count: {}", count);
                }

                insts.append(self.gen_expr(expr));

                let arg = ((expr.size() as u64) << 32) | (*count as u64 - 1);
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

pub fn codegen(module: Module) -> Bytecode {
    let generator = Generator::new();
    generator.generate(module)
}
