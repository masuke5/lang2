use std::collections::HashMap;
use std::sync::RwLock;
use lazy_static::lazy_static;

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct Id(u32);

#[derive(Debug)]
pub struct IdMap {}

lazy_static! {
    static ref ID_MAP: RwLock<HashMap<Id, String>> = {
        RwLock::new(HashMap::new())
    };

    static ref STR_MAP: RwLock<HashMap<String, Id>> = {
        RwLock::new(HashMap::new())
    };
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

    pub fn get(id_str: &str) -> Option<Id> {
        let str_map = STR_MAP.read().expect("STR_MAP poisoned");
        str_map.get(id_str).copied()
    }

    pub fn name(id: &Id) -> String {
        let id_map = ID_MAP.read().expect("ID_MAP poisoned");
        id_map[id].clone()
    }
}
