use std::convert::TryInto;
use std::mem;

use crate::bytecode::{opcode, InstList};
use crate::ast::BinOp;

struct Label(u8, Option<usize>);

impl Label {
    fn id(&self) -> u8 {
        self.0
    }

    fn set_here(&mut self, insts: &InstList) {
        self.1 = Some(insts.len());
    }

    fn loc(&self) -> Option<usize> {
        self.1
    }
}

macro_rules! with_label {
    ([$($ident:ident),+], $insts:expr) => {
        {
            let_label!(0, label_locations, $($ident,)*);

            let mut insts = $insts;

            // Resolve jump destinations
            for (i, [opcode, arg]) in insts.insts.iter_mut().enumerate() {
                if (*opcode & 0b10000000) != 0 {
                    *opcode &= 0b01111111;
                    match *opcode {
                        opcode::JUMP | opcode::JUMP_IF_FALSE | opcode::JUMP_IF_TRUE => {
                            match label_locations[*arg as usize].loc() {
                                Some(loc) => {
                                    let relative_loc = loc as i32 - i as i32;
                                    *arg = i8::to_le_bytes(relative_loc as i8)[0];
                                },
                                None => panic!("label {} is not resolved", *arg),
                            }
                        },
                        _ => {},
                    }
                }
            }

            insts
        }
    }
}

macro_rules! let_label {
    ($id:expr, $locs:ident) => { let mut $locs: [Label; $id] = unsafe { mem::zeroed() }; };
    ($id:expr, $locs:ident,) => { let mut $locs: [Label; $id] = unsafe { mem::zeroed() }; };
    ($id:expr, $locs:ident, $name:ident, $($rest:ident,)*) => {
        let_label!($id + 1, $locs, $($rest,)*);
        $locs[$id] = Label($id, None);
        let $name = unsafe { &mut *$locs.as_mut_ptr().add($id) };
    };
}

fn push_copy_inst(insts: &mut InstList, size: usize) {
    if insts.len() < 1 {
        panic!();
    }

    let [opcode, arg] = insts.prev_inst();
    let loc = i8::from_le_bytes([arg]);

    match opcode {
        opcode::LOAD_REF if loc >= -16 && loc <= 15 && size <= 0b111 => {
            let arg = (loc << 3) | size as i8;
            insts.replace_last_inst_with(opcode::LOAD_COPY, u8::from_le_bytes(arg.to_le_bytes()));
        },
        opcode::LOAD_REF | opcode::DEREFERENCE | opcode::OFFSET => {
            insts.push_inst(opcode::COPY, size as u8);
        },
        _ => {},
    }
}

// Expression

pub fn literal_int(n: i64) -> InstList {
    let mut insts = InstList::new();

    if let Ok(n) = n.try_into() {
        let [n] = i8::to_le_bytes(n);
        insts.push_inst(opcode::TINY_INT, n);
    } else {
        insts.push_inst_ref(opcode::INT, n);
    }

    insts
}

pub fn literal_str(id: usize) -> InstList {
    let mut insts = InstList::new();
    insts.push_inst(opcode::STRING, id as u8);
    insts
}

pub fn literal_unit() -> InstList {
    let mut insts = InstList::new();
    insts.push_inst(opcode::ZERO, 1);
    insts
}

pub fn literal_true() -> InstList {
    let mut insts = InstList::new();
    insts.push_inst_noarg(opcode::TRUE);
    insts
}

pub fn literal_false() -> InstList {
    let mut insts = InstList::new();
    insts.push_inst_noarg(opcode::FALSE);
    insts
}

pub fn literal_null() -> InstList {
    let mut insts = InstList::new();
    insts.push_inst(opcode::ZERO, 1);
    insts
}

pub fn literal_array(expr: InstList, expr_size: usize, arr_size: usize) -> InstList {
    let mut insts = expr;
    push_copy_inst(&mut insts, expr_size);

    if arr_size >= 2 {
        let count = (arr_size - 1) as u64;
        let arg: u64 = ((expr_size as u64) << 32) | count;
        insts.push_inst_ref(opcode::DUPLICATE, arg);
    }

    insts
}

pub fn field(loc: Option<isize>, should_deref: bool, comp_expr: InstList, comp_expr_size: usize, offset: usize) -> InstList {
    let mut insts = comp_expr;

    if let Some(loc) = loc {
        let loc = i8::to_le_bytes(loc as i8)[0];
        insts.push_inst(opcode::LOAD_REF, loc);
        insts.push_inst(opcode::STORE, comp_expr_size as u8);
        insts.push_inst(opcode::LOAD_REF, loc);
    }

    if should_deref {
        push_copy_inst(&mut insts, 1);
        insts.push_inst_noarg(opcode::DEREFERENCE);
    }

    if let Ok(offset) = offset.try_into() {
        insts.push_inst(opcode::TINY_INT, offset);
    } else {
        insts.push_inst_ref(opcode::INT, offset as i64);
    }

    insts.push_inst_noarg(opcode::OFFSET);

    insts
}

pub fn subscript(loc: Option<isize>, should_deref: bool, expr: InstList, expr_size: usize, subscript_expr: InstList, subscript_expr_size: usize) -> InstList {
    let mut insts = expr;

    if let Some(loc) = loc {
        let loc = i8::to_le_bytes(loc as i8)[0];
        insts.push_inst(opcode::LOAD_REF, loc);
        insts.push_inst(opcode::STORE, expr_size as u8);
        insts.push_inst(opcode::LOAD_REF, loc);
    }

    if should_deref {
        push_copy_inst(&mut insts, 1);
        insts.push_inst_noarg(opcode::DEREFERENCE);
    }

    insts.append(subscript_expr);
    push_copy_inst(&mut insts, subscript_expr_size);
    // TODO: element size
    insts.push_inst_noarg(opcode::OFFSET);

    insts
}

pub fn variable(loc: isize) -> InstList {
    let mut insts = InstList::new();
    insts.push_inst(opcode::LOAD_REF, i8::to_le_bytes(loc as i8)[0]);
    insts
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
pub fn binop_and(lhs: InstList, lhs_size: usize, rhs: InstList, rhs_size: usize) -> InstList {
    with_label!([a, end], {
        let mut insts = lhs;
        push_copy_inst(&mut insts, lhs_size);
        insts.push_jump(opcode::JUMP_IF_FALSE, a.id());
        insts.append(rhs);
        push_copy_inst(&mut insts, rhs_size);
        insts.push_jump(opcode::JUMP_IF_FALSE, a.id());
        insts.push_inst_noarg(opcode::TRUE);
        insts.push_jump(opcode::JUMP, end.id());
        a.set_here(&insts);
        insts.push_inst_noarg(opcode::FALSE);
        end.set_here(&insts);
        insts
    })
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
pub fn binop_or(lhs: InstList, lhs_size: usize, rhs: InstList, rhs_size: usize) -> InstList {
    with_label!([a, end], {
        let mut insts = lhs;
        push_copy_inst(&mut insts, lhs_size);
        insts.push_jump(opcode::JUMP_IF_TRUE, a.id());
        insts.append(rhs);
        push_copy_inst(&mut insts, rhs_size);
        insts.push_jump(opcode::JUMP_IF_TRUE, a.id());
        insts.push_inst_noarg(opcode::FALSE);
        insts.push_jump(opcode::JUMP, end.id());
        a.set_here(&insts);
        insts.push_inst_noarg(opcode::TRUE);
        end.set_here(&insts);
        insts
    })
}

pub fn binop(binop: BinOp, lhs: InstList, lhs_size: usize, rhs: InstList, rhs_size: usize) -> InstList {
    let mut insts = lhs;
    push_copy_inst(&mut insts, lhs_size);
    insts.append(rhs);
    push_copy_inst(&mut insts, rhs_size);

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
        binop => panic!("unexpected binop `{}`", binop.to_symbol()),
    };

    insts.push_inst_noarg(opcode);
    insts
}

pub fn call(code_id: u16, module_id: Option<u16>, args: Vec<(InstList, usize)>, return_value_size: usize) -> InstList {
    let mut insts = InstList::new();
    
    // Push placeholder for the return value
    if return_value_size > 0 {
        insts.push_inst(opcode::ZERO, return_value_size as u8);
    }

    // Push arguments
    for (arg_insts, arg_size) in args {
        insts.append(arg_insts);
        push_copy_inst(&mut insts, arg_size);
    }

    if let Some(module_id) = module_id {
        let code_id = code_id as u8;
        let module_id = module_id as u8;
        let arg = (module_id << 4) | code_id;
        insts.push_inst(opcode::CALL_EXTERN, arg);
    } else {
        insts.push_inst(opcode::CALL, code_id as u8);
    }

    insts
}

pub fn address(expr: InstList) -> InstList {
    let mut insts = expr;
    insts.push_inst_noarg(opcode::POINTER);
    insts
}

pub fn address_no_lvalue(expr: InstList, loc: isize, expr_size: usize) -> InstList {
    let mut insts = expr;
    insts.push_inst(opcode::LOAD_REF, i8::to_le_bytes(loc as i8)[0]);
    insts.push_inst(opcode::STORE, expr_size as u8);
    insts.push_inst(opcode::LOAD_REF, i8::to_le_bytes(loc as i8)[0]);
    insts.push_inst_noarg(opcode::POINTER);
    insts
}

pub fn dereference(expr: InstList, expr_size: usize) -> InstList {
    let mut insts = expr;
    push_copy_inst(&mut insts, expr_size);
    insts.push_inst_noarg(opcode::DEREFERENCE);
    insts
}

pub fn negative(expr: InstList, expr_size: usize) -> InstList {
    let mut insts = expr;
    push_copy_inst(&mut insts, expr_size);
    insts.push_inst_noarg(opcode::NEGATIVE);
    insts
}

pub fn alloc(expr: InstList, expr_size: usize) -> InstList {
    let mut insts = expr;
    push_copy_inst(&mut insts, expr_size);
    insts.push_inst(opcode::ALLOC, expr_size as u8);
    insts
}

// Statement

pub fn expr_stmt(expr: InstList, expr_size: usize) -> InstList {
    let mut insts = expr;
    for _ in 0..expr_size {
        insts.push_inst_noarg(opcode::POP);
    }
    insts
}

pub fn if_stmt(cond: InstList, cond_size: usize, then: InstList) -> InstList {
    with_label!([end], {
        let mut insts = cond;
        push_copy_inst(&mut insts, cond_size);
        insts.push_jump(opcode::JUMP_IF_FALSE, end.id());
        insts.append(then);
        end.set_here(&insts);
        insts
    })
}

pub fn if_else_stmt(cond: InstList, cond_size: usize, then: InstList, els: InstList) -> InstList {
    with_label!([elsl, end], {
        let mut insts = cond;
        push_copy_inst(&mut insts, cond_size);
        insts.push_jump(opcode::JUMP_IF_FALSE, elsl.id());
        insts.append(then);
        insts.push_jump(opcode::JUMP, end.id());
        elsl.set_here(&insts);
        insts.append(els);
        end.set_here(&insts);
        insts
    })
}

pub fn while_stmt(cond: InstList, cond_size: usize, body: InstList) -> InstList {
    with_label!([begin, end], {
        let mut insts = InstList::new();
        begin.set_here(&insts);
        insts.append(cond);
        push_copy_inst(&mut insts, cond_size);
        insts.push_jump(opcode::JUMP_IF_FALSE, end.id());
        insts.append(body);
        insts.push_jump(opcode::JUMP, begin.id());
        end.set_here(&insts);
        insts
    })
}

pub fn bind_stmt(loc: isize, expr: InstList, expr_size: usize) -> InstList {
    let mut insts = expr;
    push_copy_inst(&mut insts, expr_size);
    insts.push_inst(opcode::LOAD_REF, i8::to_le_bytes(loc as i8)[0]);
    insts.push_inst(opcode::STORE, expr_size as u8);
    insts
}

pub fn assign_stmt(lhs: InstList, rhs: InstList, rhs_size: usize) -> InstList {
    let mut insts = rhs;
    push_copy_inst(&mut insts, rhs_size);
    insts.append(lhs);
    insts.push_inst(opcode::STORE, rhs_size as u8);
    insts
}

pub fn return_stmt(loc: isize, expr: Option<(InstList, usize)>) -> InstList {
    let mut insts = InstList::new();

    if let Some((expr_insts, expr_size)) = expr {
        insts.append(expr_insts);
        push_copy_inst(&mut insts, expr_size);
        insts.push_inst(opcode::LOAD_REF, i8::to_le_bytes(loc as i8)[0]);
        insts.push_inst(opcode::STORE, expr_size as u8);
    }

    insts.push_inst_noarg(opcode::RETURN);
    insts
}
