// bytecode container

use crate::utils;
use crate::vm::ModuleBody;

const CONTAINER_HEADER: &[u8] = b"LBCC";

const MODULE_TYPE_NORMAL: u8 = 0;
const MODULE_TYPE_NATIVE: u8 = 1;

pub struct BytecodeContainer {
    pub modules: Vec<(String, ModuleBody)>,
}

impl BytecodeContainer {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(CONTAINER_HEADER);

        fn push_name(bytes: &mut Vec<u8>, s: &str) {
            bytes.extend(&(s.len() as u64).to_le_bytes());
            bytes.extend(s.as_bytes());
        }

        fn align(bytes: &mut Vec<u8>, n: usize) {
            let new_len = utils::align(bytes.len(), n);
            if bytes.len() != new_len {
                // Write padding
                for _ in 0..new_len - bytes.len() {
                    bytes.push(0);
                }
            }
        }

        for (name, module) in &self.modules {
            align(&mut bytes, 8);

            match module {
                ModuleBody::Normal(bytecode) => {
                    bytes.push(MODULE_TYPE_NORMAL);
                    push_name(&mut bytes, name);
                    align(&mut bytes, 8);
                    bytes.extend(bytecode.as_bytes());
                }
                ModuleBody::Native(_) => {
                    bytes.push(MODULE_TYPE_NATIVE);
                    push_name(&mut bytes, name);
                }
            }
        }

        bytes
    }
}
