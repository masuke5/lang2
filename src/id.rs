use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct Id(u32);

#[derive(Debug)]
pub struct IdMap {
    map: HashMap<Id, String>,
    str_map: HashMap<String, Id>,
}

impl IdMap {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            str_map: HashMap::new(),
        }
    }

    pub fn new_id(&mut self, id_str: &str) -> Id {
        match self.str_map.get(id_str) {
            Some(id) => *id,
            None => {
                let id = Id(self.map.len() as u32);
                self.map.insert(id, id_str.to_string());
                self.str_map.insert(id_str.to_string(), id);
                id
            }
        }
    }

    pub fn get(&self, id_str: &str) -> Option<Id> {
        self.str_map.get(id_str)
            .map(|id| *id)
    }

    pub fn name(&self, id: &Id) -> &str {
        &self.map[id]
    }
}
