use std::collections::LinkedList;
use std::hash::Hash;

use rustc_hash::FxHashMap;

use crate::span::Span;

#[derive(Debug)]
pub struct HashMapWithScope<K: Hash + Eq, V> {
    pub(crate) maps: LinkedList<FxHashMap<K, V>>,
}

impl<K: Hash + Eq, V> HashMapWithScope<K, V> {
    pub fn new() -> Self {
        Self {
            maps: LinkedList::new(),
        }
    }

    pub fn push_scope(&mut self) {
        self.maps.push_front(FxHashMap::default());
    }

    pub fn pop_scope(&mut self) {
        self.maps.pop_front().unwrap();
    }

    pub fn find(&self, key: &K) -> Option<&V> {
        for map in self.maps.iter() {
            if let Some(value) = map.get(key) {
                return Some(value);
            }
        }

        None
    }

    pub fn insert(&mut self, key: K, value: V) {
        let front_map = self.maps.front_mut().unwrap();
        front_map.insert(key, value);
    }

    pub fn contains_key(&self, key: &K) -> bool {
        for map in self.maps.iter() {
            if map.contains_key(key) {
                return true;
            }
        }

        false
    }
}

pub fn escape_string(raw: &str) -> String {
    let mut s = String::new();
    for ch in raw.chars() {
        let n = ch as u32;
        match ch {
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            '\\' => s.push_str("\\\\"),
            '\0' => s.push_str("\\0"),
            '"' => s.push_str("\\\""),
            _ if n <= 0x1f || n == 0x7f => s.push_str(&format!("\\x{:02x}", n)),
            ch => s.push(ch),
        }
    }

    s
}

pub fn span_to_string(span: &Span) -> String {
    format!("\x1b[33m{}:{}-{}:{}\x1b[0m", span.start_line, span.start_col, span.end_line, span.end_col)
}

pub fn align(x: usize, n: usize) -> usize {
    (x + (n - 1)) & !(n - 1)
}
