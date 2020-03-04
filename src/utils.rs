use std::collections::{hash_map, LinkedList};
use std::fmt::Display;
use std::hash::Hash;
use std::iter::Iterator;

use rustc_hash::FxHashMap;

use crate::span::Span;

macro_rules! write_iter {
    ($f:expr, $iter:expr) => {{
        use crate::utils;

        let s = utils::format_iter($iter);
        write!($f, "{}", s)
    }};
}

pub fn format_iter<I: Iterator>(mut iter: I) -> String
where
    <I as Iterator>::Item: Display,
{
    let mut s = String::new();

    if let Some(first) = iter.next() {
        s += &format!("{}", first);
        for value in iter {
            s += &format!(", {}", value);
        }
    }

    s
}

pub fn format_bool(b: bool, s: &str) -> &str {
    if b {
        s
    } else {
        ""
    }
}

macro_rules! fn_next {
    () => {
        fn next(&mut self) -> Option<Self::Item> {
            loop {
                if self.curr.is_none() {
                    let new_iter = self.iter.pop()?;
                    self.curr = Some(new_iter);
                    self.level -= 1;
                }

                match self.curr.as_mut().unwrap().next() {
                    Some((key, value)) => return Some((self.level, key, value)),
                    None => {
                        self.curr = None;
                        continue;
                    }
                }
            }
        }
    };
}

pub struct HashMapWithScopeIter<'a, K, V> {
    iter: Vec<hash_map::Iter<'a, K, V>>,
    curr: Option<hash_map::Iter<'a, K, V>>,
    level: usize,
}

impl<'a, K, V> HashMapWithScopeIter<'a, K, V> {
    fn new(maps: &'a LinkedList<FxHashMap<K, V>>) -> Self {
        let mut iters = Vec::with_capacity(maps.len());
        for map in maps.iter().rev() {
            iters.push(map.iter());
        }

        Self {
            iter: iters,
            curr: None,
            level: maps.len() + 1,
        }
    }
}

impl<'a, K, V> Iterator for HashMapWithScopeIter<'a, K, V> {
    type Item = (usize, &'a K, &'a V);
    fn_next!();
}

pub struct HashMapWithScopeIterMut<'a, K, V> {
    iter: Vec<hash_map::IterMut<'a, K, V>>,
    curr: Option<hash_map::IterMut<'a, K, V>>,
    level: usize,
}

impl<'a, K, V> HashMapWithScopeIterMut<'a, K, V> {
    fn new(maps: &'a mut LinkedList<FxHashMap<K, V>>) -> Self {
        let level = maps.len() + 1;
        let mut iters = Vec::with_capacity(maps.len());
        for map in maps.iter_mut().rev() {
            iters.push(map.iter_mut());
        }

        Self {
            iter: iters,
            curr: None,
            level,
        }
    }
}

impl<'a, K, V> Iterator for HashMapWithScopeIterMut<'a, K, V> {
    type Item = (usize, &'a K, &'a mut V);
    fn_next!();
}

pub struct HashMapWithScopeIntoIter<K, V> {
    iter: Vec<hash_map::IntoIter<K, V>>,
    curr: Option<hash_map::IntoIter<K, V>>,
    level: usize,
}

impl<K, V> HashMapWithScopeIntoIter<K, V> {
    fn new(maps: LinkedList<FxHashMap<K, V>>) -> Self {
        let level = maps.len() + 1;
        let mut iters = Vec::with_capacity(maps.len());
        for map in maps.into_iter().rev() {
            iters.push(map.into_iter());
        }

        Self {
            iter: iters,
            curr: None,
            level,
        }
    }
}

impl<K, V> Iterator for HashMapWithScopeIntoIter<K, V> {
    type Item = (usize, K, V);
    fn_next!();
}

#[derive(Debug, Clone)]
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

    pub fn get(&self, key: &K) -> Option<&V> {
        for map in self.maps.iter() {
            if let Some(value) = map.get(key) {
                return Some(value);
            }
        }

        None
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        for map in self.maps.iter_mut() {
            if let Some(value) = map.get_mut(key) {
                return Some(value);
            }
        }

        None
    }

    #[allow(dead_code)]
    pub fn get_with_level(&self, key: &K) -> Option<(&V, usize)> {
        let mut level = self.level();
        for map in self.maps.iter() {
            if let Some(value) = map.get(key) {
                return Some((value, level));
            }

            level -= 1;
        }

        None
    }

    pub fn get_mut_with_level(&mut self, key: &K) -> Option<(&mut V, usize)> {
        let mut level = self.level();
        for map in self.maps.iter_mut() {
            if let Some(value) = map.get_mut(key) {
                return Some((value, level));
            }

            level -= 1;
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

    #[allow(dead_code)]
    pub fn last_scope(&self) -> Option<&FxHashMap<K, V>> {
        self.maps.front()
    }

    pub fn level(&self) -> usize {
        self.maps.len()
    }

    pub fn iter<'a>(&'a self) -> HashMapWithScopeIter<'a, K, V> {
        HashMapWithScopeIter::new(&self.maps)
    }

    pub fn iter_mut<'a>(&'a mut self) -> HashMapWithScopeIterMut<'a, K, V> {
        HashMapWithScopeIterMut::new(&mut self.maps)
    }

    pub fn into_iter(self) -> HashMapWithScopeIntoIter<K, V> {
        HashMapWithScopeIntoIter::new(self.maps)
    }
}

impl<K: Hash + Eq, V> IntoIterator for HashMapWithScope<K, V> {
    type Item = (usize, K, V);
    type IntoIter = HashMapWithScopeIntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_iter()
    }
}

impl<'a, K: Hash + Eq, V> IntoIterator for &'a HashMapWithScope<K, V> {
    type Item = (usize, &'a K, &'a V);
    type IntoIter = HashMapWithScopeIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Hash + Eq, V> IntoIterator for &'a mut HashMapWithScope<K, V> {
    type Item = (usize, &'a K, &'a mut V);
    type IntoIter = HashMapWithScopeIterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
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
    format!(
        "\x1b[33m{}:{}-{}:{}\x1b[0m",
        span.start_line, span.start_col, span.end_line, span.end_col
    )
}

pub fn align(x: usize, n: usize) -> usize {
    (x + (n - 1)) & !(n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iter() {
        let mut hm = HashMapWithScope::<i32, i32>::new();
        hm.push_scope();

        let vec: Vec<(usize, &i32, &i32)> = hm.iter().collect();
        assert!(vec.is_empty());

        hm.insert(100, 10);

        let vec: Vec<(usize, &i32, &i32)> = hm.iter().collect();
        assert_eq!(vec, vec![(1, &100, &10)]);

        hm.push_scope();

        hm.insert(101, 20);
        hm.insert(102, 23);
        hm.insert(103, 28);

        let vec: Vec<(usize, &i32, &i32)> = hm.iter().collect();
        assert_eq!(
            vec,
            vec![
                (2, &101, &20),
                (2, &102, &23),
                (2, &103, &28),
                (1, &100, &10),
            ]
        );

        hm.pop_scope();

        let vec: Vec<(usize, &i32, &i32)> = hm.iter().collect();
        assert_eq!(vec, vec![(1, &100, &10)]);
    }

    #[test]
    fn into_iter() {
        let mut hm = HashMapWithScope::<i32, i32>::new();
        hm.push_scope();

        let vec: Vec<(usize, i32, i32)> = hm.clone().into_iter().collect();
        assert!(vec.is_empty());

        hm.insert(100, 10);

        let vec: Vec<(usize, i32, i32)> = hm.clone().into_iter().collect();
        assert_eq!(vec, vec![(1, 100, 10)]);

        hm.push_scope();

        hm.insert(101, 20);
        hm.insert(102, 23);
        hm.insert(103, 28);

        let vec: Vec<(usize, i32, i32)> = hm.clone().into_iter().collect();
        assert_eq!(
            vec,
            vec![(2, 101, 20), (2, 102, 23), (2, 103, 28), (1, 100, 10),]
        );

        hm.pop_scope();

        let vec: Vec<(usize, i32, i32)> = hm.into_iter().collect();
        assert_eq!(vec, vec![(1, 100, 10)]);
    }
}
