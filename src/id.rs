use lazy_static::lazy_static;
use rustc_hash::FxHashMap;
use std::fmt;
use std::sync::RwLock;

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub struct Id(u32);

impl fmt::Debug for Id {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", IdMap::name(*self))
    }
}

#[derive(Debug)]
pub struct IdMap {}

lazy_static! {
    static ref ID_MAP: RwLock<FxHashMap<Id, String>> = { RwLock::new(FxHashMap::default()) };
    static ref STR_MAP: RwLock<FxHashMap<String, Id>> = { RwLock::new(FxHashMap::default()) };
}

impl IdMap {
    pub fn new_id(id_str: &str) -> Id {
        let mut id_map = ID_MAP.write().expect("ID_MAP poisoned");
        let mut str_map = STR_MAP.write().expect("STR_MAP poisoned");

        match str_map.get(id_str) {
            Some(id) => *id,
            None => {
                let id = Id(id_map.len() as u32);
                id_map.insert(id, id_str.to_string());
                str_map.insert(id_str.to_string(), id);
                id
            }
        }
    }

    #[allow(dead_code)]
    pub fn get(id_str: &str) -> Option<Id> {
        let str_map = STR_MAP.read().expect("STR_MAP poisoned");
        str_map.get(id_str).copied()
    }

    pub fn name(id: Id) -> String {
        let id_map = ID_MAP.read().expect("ID_MAP poisoned");
        id_map[&id].clone()
    }
}

pub mod reserved_id {
    use super::{Id, IdMap};
    use lazy_static::lazy_static;

    // Basically use "$" that cannot use for identifiers
    lazy_static! {
        pub static ref MAIN_FUNC: Id = IdMap::new_id("$main");
        pub static ref STD_MODULE: Id = IdMap::new_id("std");
        pub static ref CMD: Id = IdMap::new_id("$cmd");
        pub static ref RETURN_VALUE: Id = IdMap::new_id("$rv");
        pub static ref DUMMY_PARAM: Id = IdMap::new_id("$dummy_param");
    }
}
