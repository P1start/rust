// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rope types for efficient mutable data structures.

use core::prelude::*;

use core::{cmp, slice, mem};
use core::fmt::{Show, Formatter, FormatError};
use core::default::Default;
use core::iter::AdditiveIterator;
use core::iter::FlatMap;
use alloc::owned::Box;
use super::Mutable;
use super::hash::{Hash, Writer};
use super::vec::{mod, Vec};
use super::string::String;
use super::str::StrVector;
use super::MutableSeq;

struct ZipLength<T, U> {
    a: T,
    b: U,
}

impl<A, B, T: Iterator<A>, U: Iterator<B>> Iterator<Result<(A, B), bool>> for ZipLength<T, U> {
    #[inline]
    fn next(&mut self) -> Option<Result<(A, B), bool>> {
        match self.a.next() {
            None => match self.b.next() {
                None => None,
                Some(_) => Some(Err(true)),
            },
            Some(x) => match self.b.next() {
                None => Some(Err(false)),
                Some(y) => Some(Ok((x, y)))
            }
        }
    }
}

#[inline]
fn zip_length<A, B, T: Iterator<A>, U: Iterator<B>>(slf: T, other: U) -> ZipLength<T, U> {
    ZipLength { a: slf, b: other }
}

/// An owned rope type.
///
/// This is a type for a [rope](http://en.wikipedia.org/wiki/Rope_(data_structure)), a type of
/// tree-based data structure composed of smaller arrays that can be used for efficiently storing
/// and manipulating a very long vector.
///
/// # Example
///
/// ```rust
/// let mut rope: Rope<u8> = Rope::new();
/// rope.push(1);
/// rope.push(4);
/// rope.push(5);
///
/// let mut insert_rope = Rope::new();
/// rope.push(2);
/// rope.push(3);
///
/// rope.insert(1, insert_rope);
/// for i in rope.iter() {
///     println!("{}", i);
/// }
/// // => 1 2 3 4 5
/// ```
#[deriving(Clone)]
pub struct Rope<T> {
    tree: RopeTree<T>,
}

#[deriving(Clone)]
enum RopeTree<T> {
    Branch(uint, uint, Box<RopeTree<T>>, Box<RopeTree<T>>),
    Leaf(Vec<T>),
}

/// By-ref iterator over rope nodes (not items).
struct RopeNodeItems<'a, T: 'a> {
    stack: Vec<&'a RopeTree<T>>,
}

impl<'a, T> RopeNodeItems<'a, T> {
    fn from_ropetree(rt: &'a RopeTree<T>) -> RopeNodeItems<'a, T> {
        let mut v = Vec::with_capacity(rt.height() + 1);
        v.push(rt);
        RopeNodeItems {
            stack: v,
        }
    }
}

impl<'a, T> Iterator<&'a [T]> for RopeNodeItems<'a, T> {
    fn next(&mut self) -> Option<&'a [T]> {
        if self.stack.is_empty() { return None }
        loop {
            let curr = *self.stack.last().unwrap();
            match *curr {
                Branch(_, _, ref left, ref right) => {
                    self.stack.pop();
                    self.stack.push(&**right);
                    self.stack.push(&**left);
                }
                Leaf(ref vec) => {
                    self.stack.pop();
                    return Some(vec[])
                }
            }
        }
    }
}

/// A by-reference iterator over the elements in a `Rope`.
pub struct Items<'a, T: 'a>(FlatMap<'a, &'a [T],
                                RopeNodeItems<'a, T>,
                                slice::Items<'a, T>>);

impl<'a, T> Iterator<&'a T> for Items<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        self.0.next()
    }
}

/// A by-mutable-reference iterator over the elements in a `Rope`.
pub struct MutItems<'a, T: 'a>(Items<'a, T>);

impl<'a, T> Iterator<&'a mut T> for MutItems<'a, T> {
    fn next(&mut self) -> Option<&'a mut T> {
        unsafe { mem::transmute(self.0.next()) }
    }
}

/// A by-val iterator over rope nodes (not items).
struct RopeNodeMoveItems<T> {
    stack: Vec<RopeTree<T>>,
}

impl<T> RopeNodeMoveItems<T> {
    fn from_ropetree(rt: RopeTree<T>) -> RopeNodeMoveItems<T> {
        let mut v = Vec::with_capacity(rt.height() + 1);
        v.push(rt);
        RopeNodeMoveItems {
            stack: v,
        }
    }
}

impl<T> Iterator<Vec<T>> for RopeNodeMoveItems<T> {
    fn next(&mut self) -> Option<Vec<T>> {
        if self.stack.is_empty() { return None }
        loop {
            let curr = self.stack.pop().unwrap();
            match curr {
                Branch(_, _, box left, box right) => {
                    self.stack.push(right);
                    self.stack.push(left);
                }
                Leaf(vec) => {
                    return Some(vec)
                }
            }
        }
    }
}

/// A by-value iterator over the elements in a `Rope`.
pub struct MoveItems<T>(FlatMap<'static, Vec<T>,
                                    RopeNodeMoveItems<T>,
                                    vec::MoveItems<T>>);

impl<T> Iterator<T> for MoveItems<T> {
    fn next(&mut self) -> Option<T> {
        self.0.next()
    }
}

static MAX_VEC_LEN: uint = 100;

impl<T> RopeTree<T> {
    #[inline]
    fn height(&self) -> uint {
        match *self {
            Leaf(_) => 0,
            Branch(_, height, _, _) => height,
        }
    }

    #[inline]
    fn len(&self) -> uint {
        match *self {
            Branch(weight, _, _, ref right) => weight + right.len(),
            Leaf(ref v) => v.len(),
        }
    }

    fn index(&self, i: uint) -> Option<&T> {
        match *self {
            Branch(weight, _, _, ref right) if weight < i => right.index(i - weight),
            Branch(_, _, ref left, _) => left.index(i),
            Leaf(ref vec) => vec[].get(i),
        }
    }

    fn index_mut(&mut self, i: uint) -> Option<&mut T> {
        match *self {
            Branch(weight, _, _, ref mut right) if weight < i => right.index_mut(i - weight),
            Branch(_, _, ref mut left, _) => left.index_mut(i),
            Leaf(ref mut vec) => vec[mut].get_mut(i),
        }
    }

    fn split_vec(self, i: uint, mut v: Vec<RopeTree<T>>) -> (RopeTree<T>, Vec<RopeTree<T>>) {
        match self {
            Branch(weight, height, left, right) => {
                if weight < i {
                    let (rt, vrt) = right.split_vec(i - weight, v);
                    (Branch(weight, height, left, box rt), vrt)
                } else {
                    v.push(*right);
                    left.split_vec(i, v)
                }
            }
            Leaf(vec) => {
                let mut start = vec![];
                let mut end = vec![];
                if i > vec.len() {
                    fail!("index out of bounds: the len is {} but the index is {}", vec.len(), i);
                }
                for (index, v) in vec.into_iter().enumerate() {
                    if index < i {
                        start.push(v);
                    } else {
                        end.push(v)
                    }
                }
                v.push(Leaf(end));
                (Leaf(start), v)
            }
        }
    }

    fn split(self, i: uint) -> (RopeTree<T>, RopeTree<T>) {
        let (tree, vec) = self.split_vec(i, vec![]);
        let mut iter = vec.into_iter().rev();
        let first = match iter.next() {
            Some(i) => i,
            None => return (tree, Leaf(vec![])),
        };
        (tree, iter.fold(first, |x, y| x.append(y)))
    }

    #[inline]
    fn total_weight(&self) -> uint {
        match *self {
           Branch(len, _, _, ref r) => {
               len + r.total_weight()
           }
           Leaf(ref v) => {
               v.len()
           }
        }
    }

    fn append(self, other: RopeTree<T>) -> RopeTree<T> {
        match self {
            Branch(len, depth, l, r) => {
                match other {
                    Branch(o_len, o_depth, o_l, o_r) => {
                    // Branch + Branch
                        let new_depth = cmp::max(depth, o_depth);
                        Branch(len + r.total_weight(), new_depth + 1,
                               box Branch(len, depth, l, r),
                               box Branch(o_len, o_depth, o_l, o_r))
                    }
                    Leaf(o_v) => {
                        // Branch + Leaf
                        match r {
                            box r_br @ Branch(..) => {
                                Branch(len + r_br.total_weight(), depth + 1,
                                       box Branch(len, depth, l, box r_br),
                                       box Leaf(o_v))
                            }
                            box Leaf(mut self_r_v) => {
                                let result_len = o_v.len() + self_r_v.len();
                                if result_len <= MAX_VEC_LEN {
                                    self_r_v.extend(o_v.into_iter());
                                    Branch(len, depth, l,
                                           box Leaf(self_r_v))
                                } else {
                                    Branch(len + self_r_v.len(), depth + 1,
                                           box Branch(len, depth, l,
                                                      box Leaf(self_r_v)),
                                           box Leaf(o_v))
                                }
                            }
                        }
                    }
                }
            }
            Leaf(mut self_v) => {
                match other {
                    // Leaf + Branch
                    Branch(o_len, o_depth, o_l, o_r) => {
                        Branch(self_v.len(), o_depth + 1,
                               box Leaf(self_v),
                               box Branch(o_len, o_depth, o_l, o_r))
                    }
                    Leaf(o_v) => {
                        // Leaf + Leaf
                        let result_len = o_v.len() + self_v.len();
                        if result_len <= MAX_VEC_LEN {
                            self_v.extend(o_v.into_iter());
                            Leaf(self_v)
                        } else {
                            Branch(self_v.len(), 1,
                                   box Leaf(self_v),
                                   box Leaf(o_v))
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn iter(&self) -> Items<T> {
        Items(RopeNodeItems::from_ropetree(self).flat_map(|x| x.iter()))
    }

    #[inline]
    fn iter_mut(&mut self) -> MutItems<T> {
        MutItems(self.iter())
    }

    #[inline]
    fn into_iter(self) -> MoveItems<T> {
        MoveItems(RopeNodeMoveItems::from_ropetree(self).flat_map(|x| x.into_iter()))
    }
}

impl<T: PartialEq> RopeTree<T> {
    fn eq(&self, other: &RopeTree<T>) -> bool {
        for x in zip_length(self.iter(), other.iter()) {
            let (c, d) = match x {
                Ok(x) => x,
                Err(_) => return false,
            };
            if c != d {
                return false;
            }
        }
        true
    }
}

impl<T> Rope<T> {
    /// Constructs a new, empty `Rope`.
    #[inline]
    pub fn new() -> Rope<T> {
        Rope { tree: Leaf(vec![]) }
    }

    /// Constructs a `Rope<T>` from a `Vec<T>`.
    #[inline]
    pub fn from_vec(vec: Vec<T>) -> Rope<T> {
        Rope { tree: Leaf(vec) }
    }

    /// Returns an iterator that iterates over references to the elements in a `Rope`.
    #[inline]
    pub fn iter(&self) -> Items<T> {
        self.tree.iter()
    }

    /// Returns an iterator that iterates over mutable references to the elements in a `Rope`.
    #[inline]
    pub fn iter_mut(&mut self) -> MutItems<T> {
        self.tree.iter_mut()
    }

    /// Returns an iterator that iterates over the elements in a `Rope`, consuming the `Rope` in
    /// doing so.
    #[inline]
    pub fn into_iter(self) -> MoveItems<T> {
        self.tree.into_iter()
    }

    /// Returns a mutable reference to the value at index `index`.
    ///
    /// # Failure
    ///
    /// Fails if `index` is out of bounds.
    #[inline]
    pub fn index_mut(&mut self, &index: &uint) -> &mut T {
        let len = self.len();
        self.tree.index_mut(index)
            .unwrap_or(fail!("index out of bounds: the len is {} but the index is {}",
                             len, index))
    }

    /// Returns an immutable reference to the value at index `index`, or `None` if the index is out
    /// of bounds.
    #[inline]
    pub fn get(&self, index: uint) -> Option<&T> {
        self.tree.index(index)
    }

    /// Returns a mutable reference to the value at index `index`, or `None` if the index is out of
    /// bounds.
    #[inline]
    pub fn get_mut(&mut self, index: uint) -> Option<&mut T> {
        self.tree.index_mut(index)
    }

    /// Takes the root of a rope, replacing it with an empty rope.
    #[inline]
    fn take_root(&mut self) -> Rope<T> {
        Rope { tree: mem::replace(&mut self.tree, Leaf(vec![])) }
    }

    /// Sets the root of a rope to that of another.
    #[inline]
    fn set_root(&mut self, other: Rope<T>) {
        self.tree = other.tree;
    }

    /// Concatenates two ropes, consuming both and returning the joined rope.
    ///
    /// This function should compute in logarithmic time.
    #[inline]
    pub fn join(self, other: Rope<T>) -> Rope<T> {
        Rope { tree: self.tree.append(other.tree) }
    }

    /// Splits a rope at index `index`, consuming the original and returning the two resulting
    /// ropes. The first rope should have length `index`.
    ///
    /// This function should compute in logarithmic time.
    ///
    /// # Failure
    ///
    /// Fails if `index` is out of bounds.
    #[inline]
    pub fn split(self, i: uint) -> (Rope<T>, Rope<T>) {
        let (rt1, rt2) = self.tree.split(i);
        (Rope { tree: rt1 }, Rope { tree: rt2 })
    }

    /// Creates a rope from a single element.
    #[inline]
    pub fn from_elem(elem: T) -> Rope<T> {
        FromIterator::from_iter(Some(elem).into_iter())
    }

    /// Appends a rope to another rope in-place.
    ///
    /// This function should compute in logarithmic time.
    #[inline]
    pub fn append(&mut self, mut rope: Rope<T>) {
        let root = self.take_root();
        self.set_root(root.join(rope.take_root()));
    }

    /// Prepends a rope onto another rope.
    ///
    /// This function should compute in logarithmic time.
    #[inline]
    pub fn prepend(&mut self, mut rope: Rope<T>) {
        let root = self.take_root();
        self.set_root(rope.take_root().join(root));
    }

    /// Prepend a single element onto a rope.
    ///
    /// This function should compute in logarithmic time.
    #[inline]
    pub fn unshift(&mut self, elem: T) {
        self.prepend(Rope::from_elem(elem));
    }

    /// Inserts a rope into another rope in-place.
    ///
    /// This function should compute in logarithmic time.
    #[inline]
    pub fn insert(&mut self, index: uint, other: Rope<T>) {
        let root = self.take_root();
        let (r1, r2) = root.split(index);
        self.set_root(r1.join(other).join(r2));
    }

    /// Deletes a section of a rope in-place from index `j` (inclusive) to `u` (exclusive),
    /// returning the deleted section.
    ///
    /// This function should compute in logarithmic time.
    ///
    /// # Failure
    ///
    /// Fails if `i` or `u` is out of bounds.
    #[inline]
    pub fn delete(&mut self, i: uint, u: uint) -> Rope<T> {
        let root = self.take_root();
        let (r, r3) = root.split(u);
        let (r1, r2) = r.split(i);
        self.set_root(r1.join(r3));
        r2
    }
}

impl<T> FromIterator<T> for Rope<T> {
    #[inline]
    fn from_iter<I: Iterator<T>>(mut iterator: I) -> Rope<T> {
        let vec: Vec<_> = iterator.collect();
        Rope { tree: Leaf(vec) }
    }
}

impl<T> Extendable<T> for Rope<T> {
    #[inline]
    fn extend<I: Iterator<T>>(&mut self, iterator: I) {
        let mut v = Vec::new();
        v.extend(iterator);
        self.append(Rope::from_vec(v));
    }
}

impl<T> Collection for Rope<T> {
    #[inline]
    fn len(&self) -> uint {
        self.tree.len()
    }
}

impl<T> Index<uint, T> for Rope<T> {
    #[inline]
    fn index(&self, &index: &uint) -> &T {
        self.tree.index(index)
            .unwrap_or(fail!("index out of bounds: the len is {} but the index is {}",
                             self.len(), index))
    }
}

// FIXME(#12825) Indexing will always try IndexMut first and that causes issues.
/*impl<T> IndexMut<uint, T> for Rope<T> {
   #[inline]
    fn index_mut(&mut self, &index: &uint) -> &mut T {
        self.tree.index_mut(index)
            .expect(format!("index out of bounds: the len is {} but the index is {}",
                            self.len(), index).as_slice())
    }
}*/

impl<T: PartialEq> PartialEq for Rope<T> {
    #[inline]
    fn eq(&self, other: &Rope<T>) -> bool {
        self.tree.eq(&other.tree)
    }
}

impl<T: PartialEq> Eq for Rope<T> {}

impl<T: Show> Show for Rope<T> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FormatError> {
        try!(write!(f, "["));
        for (i, v) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")) }
            try!(v.fmt(f));
        }
        write!(f, "]")
    }
}

impl<T: PartialOrd> PartialOrd for Rope<T> {
    fn partial_cmp(&self, other: &Rope<T>) -> Option<Ordering> {
        for x in zip_length(self.iter(), other.iter()) {
            let (c, d) = match x {
                Ok(x) => x,
                // Err(false) means self terminated first
                // Err(true)  means other terminated first
                Err(false) => return Some(Greater),
                Err(true) => return Some(Less),
            };
            if c < d {
                return Some(Less);
            } else if c > d {
                return Some(Greater);
            }
        }
        Some(Equal)
    }
}

impl<T: Ord> Ord for Rope<T> {
    fn cmp(&self, other: &Rope<T>) -> Ordering {
        for x in zip_length(self.iter(), other.iter()) {
            let (c, d) = match x {
                Ok(x) => x,
                // Err(false) means self terminated first
                // Err(true)  means other terminated first
                Err(false) => return Greater,
                Err(true)  => return Less,
            };
            if c < d {
                return Less;
            } else if c > d {
                return Greater;
            }
        }
        Equal
    }
}

impl<T> Mutable for Rope<T> {
    #[inline]
    fn clear(&mut self) {
        self.tree = Leaf(vec![]);
    }
}

impl<T> MutableSeq<T> for Rope<T> {
    /// Pushes a single element onto a rope.
    ///
    /// This function should compute in logarithmic time.
    #[inline]
    fn push(&mut self, elem: T) {
        self.append(Rope::from_elem(elem));
    }

    /// Removes a single element from the end of the rope and returns it, or `None` if the rope was
    /// empty.
    #[inline]
    fn pop(&mut self) -> Option<T> {
        let len = self.len();
        if len == 0 { return None }
        Some(self.delete(len - 1, len).into_iter().next().unwrap())
    }
}

impl<T> Default for Rope<T> {
    #[inline]
    fn default() -> Rope<T> {
        Rope { tree: Leaf(vec![]) }
    }
}

impl<S: Writer, T: Hash<S>> Hash<S> for Rope<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.len().hash(state);
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}

impl<S: Str> StrVector for Rope<S> {
    fn concat(&self) -> String {
        if self.is_empty() {
            return String::new();
        }

        let len = self.iter().map(|s| s.as_slice().len()).sum();
        let mut result = String::with_capacity(len);

        for s in self.iter() {
            result.push_str(s.as_slice());
        }
        result
    }

    fn connect(&self, sep: &str) -> String {
        if self.is_empty() {
            return String::new();
        }

        if sep.is_empty() {
            return self.concat();
        }

        let len = sep.len() * (self.len() - 1) + self.iter().map(|s| s.as_slice().len()).sum();
        let mut result = String::with_capacity(len);
        let mut first = true;

        for s in self.iter() {
            if first {
                first = false;
            } else {
                result.push_str(sep);
            }
            result.push_str(s.as_slice());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::Rope;
    use super::super::hash::hash;
    use std::prelude::*;
    use std::rand::{mod, Rng};
    use super::super::{Mutable, MutableSeq};
    use super::super::str::StrVector;
    use test::Bencher;

    /// Constructs a new `Rope` from a list of elements.
    macro_rules! rope {
        ($($e:expr),*) => {
            Rope::from_vec(vec![$($e),*])
        }
    }

    macro_rules! rope_str {
        ($e:expr) => {
            Rope::from_vec($e.bytes().collect::<super::super::vec::Vec<u8>>())
        }
    }

    #[test]
    fn test_len() {
        assert_eq!(rope_str!("hello").len(), 5);
        assert_eq!(rope_str!("").len(), 0);
        assert_eq!(rope_str!("úÂḿ").len(), 7);

        assert_eq!(rope_str!("hel").join(rope_str!("lo")).len(), 5);
    }

    #[test]
    fn test_index() {
        let rope = rope_str!("helló").join(rope_str!(" wor")).join(rope_str!("ld"));
        assert_eq!(rope[3], b'l');
        assert_eq!(rope[7], b'w');
        assert_eq!(rope[11], b'd');
        assert_eq!(rope.get(3), Some(&b'l'));
        assert_eq!(rope.get(12), None);

        let mut rope = rope;
        *rope.index_mut(&1) = b'u';
        assert_eq!(rope, rope_str!("hulló world"));
    }

    #[test]
    fn test_join() {
        let rope = rope_str!("");
        let rope = rope.join(rope_str!("ab "));
        let rope = rope.join(rope_str!("cd"));
        println!("{}", rope);
        assert_eq!(rope, rope_str!("ab cd"));

        assert_eq!(rope_str!("hello ").join(rope_str!("world")), rope_str!("hello world"));
    }

    #[test]
    fn test_append() {
        let mut rope = rope_str!("hello");
        rope.append(rope_str!(" world"));
        assert_eq!(rope, rope_str!("hello world"));
    }

    #[test]
    fn test_push() {
        let mut rope = rope_str!("hello world");
        rope.push(b'!');
        assert_eq!(rope, rope_str!("hello world!"));
    }

    #[test]
    fn test_pop() {
        let mut rope = rope_str!("hello world");
        assert_eq!(rope.pop(), Some(b'd'));
        assert_eq!(rope.pop(), Some(b'l'));
        assert_eq!(rope.pop(), Some(b'r'));
        assert_eq!(rope, rope_str!("hello wo"));
    }

    #[test]
    fn test_prepend() {
        let mut rope = rope_str!("world");
        rope.prepend(rope_str!("hello "));
        assert_eq!(rope, rope_str!("hello world"));
    }

    #[test]
    fn test_unshift() {
        let mut rope = rope_str!("dlrow olleh");
        rope.unshift(b'!');
        assert_eq!(rope, rope_str!("!dlrow olleh"));
    }

    #[test]
    fn test_iter() {
        let mut rope = rope_str!("heó");
        rope.insert(2, rope_str!("ll"));
        rope.insert(6, rope_str!("!"));

        for (i, &c) in rope.iter().enumerate() {
            assert_eq!(c, match i {
                0 => b'h',
                1 => b'e',
                2 => b'l',
                3 => b'l',
                4 => 195,
                5 => 179,
                6 => b'!',
                _ => unreachable!(),
            });
        }
    }

    #[test]
    fn test_iter_mut() {
        let mut rope = rope_str!("heó");
        rope.insert(2, rope_str!("ll"));
        rope.insert(6, rope_str!("!"));

        for (i, c) in rope.iter_mut().enumerate() {
            assert_eq!(*c, match i {
                0 => b'h',
                1 => b'e',
                2 => b'l',
                3 => b'l',
                4 => 195,
                5 => 179,
                6 => b'!',
                _ => unreachable!(),
            });
            if i == 1 { *c = b'u' }
        }

        assert_eq!(rope, rope_str!("hulló!"));
    }

    #[test]
    fn test_into_iter() {
        let mut rope = rope_str!("heó");
        rope.insert(2, rope_str!("ll"));
        rope.insert(6, rope_str!("!"));

        for (i, c) in rope.into_iter().enumerate() {
            assert_eq!(c, match i {
                0 => b'h',
                1 => b'e',
                2 => b'l',
                3 => b'l',
                4 => 195,
                5 => 179,
                6 => b'!',
                _ => unreachable!(),
            });
        }
    }

    // Some more rigorous .iter() / .into_iter() testing
    #[test]
    fn test_iter_vec_comparison() {
        let mut rope: Rope<uint> = Rope::new();
        let mut vec: Vec<uint> = Vec::new();
        let mut len = 0u;
        let mut rng = rand::task_rng();
        for _ in range(0u, 100) {
            let v = rand::sample(&mut rng, range(0u, 100), 100);
            let i = rng.gen_range(0u, len + 1);
            len += v.len();
            rope.insert(i, Rope::from_vec(unsafe { ::std::mem::transmute(v.clone()) }));
            for (i2, a) in v.into_iter().enumerate() {
                vec.insert(i + i2, a);
            }
        }
        assert_eq!(rope, Rope::from_vec(unsafe { ::std::mem::transmute(vec) }));
    }

    #[test]
    fn test_into_iter_vec_comparison() {
        let mut rope: Rope<uint> = Rope::new();
        let mut vec: Vec<uint> = Vec::new();
        let mut len = 0u;
        let mut rng = rand::task_rng();
        for _ in range(0u, 100) {
            let v = rand::sample(&mut rng, range(0u, 100), 100);
            let i = rng.gen_range(0u, len + 1);
            len += v.len();
            rope.insert(i, Rope::from_vec(unsafe { ::std::mem::transmute(v.clone()) }));
            for (i2, a) in v.into_iter().enumerate() {
                vec.insert(i + i2, a);
            }
        }
        assert_eq!(rope.into_iter().collect::<Vec<uint>>(), vec);
    }

    #[test]
    fn test_eq() {
        let rope1 = rope_str!("hello");
        let rope2 = rope_str!("hello");
        let rope3 = rope_str!("hellol");
        let rope4 = rope_str!("hell");
        assert_eq!(rope1, rope2);
        assert!(rope1 != rope3);
        assert!(rope1 != rope4);
    }

    #[test]
    fn test_partialord() {
        let rope1 = rope_str!("hello");
        let rope2 = rope_str!("hello");
        let rope3 = rope_str!("hellol");
        let rope4 = rope_str!("hell");
        let rope5 = rope_str!("abc");
        let rope6 = rope_str!("zyx");
        assert!(rope1 < rope3);
        assert!(rope3 > rope1);
        assert!(rope4 < rope2);
        assert!(!(rope1 > rope2));
        assert!(!(rope1 < rope2));
        assert!(!(rope2 > rope1));
        assert!(!(rope2 < rope1));
        assert!(rope6 > rope1);
        assert!(rope5 < rope1);
    }

    #[test]
    fn test_ord() {
        let rope1 = rope_str!("hello");
        let rope2 = rope_str!("hello");
        let rope3 = rope_str!("hellol");
        let rope4 = rope_str!("hell");
        let rope5 = rope_str!("abc");
        let rope6 = rope_str!("zyx");
        assert_eq!(rope1.cmp(&rope3), Less);
        assert_eq!(rope3.cmp(&rope1), Greater);
        assert_eq!(rope4.cmp(&rope2), Less);
        assert_eq!(rope1.cmp(&rope2), Equal);
        assert_eq!(rope6.cmp(&rope1), Greater);
        assert_eq!(rope5.cmp(&rope1), Less);
    }

    #[test]
    fn test_split() {
        let rope1 = rope_str!("hel").join(rope_str!("lo ")).join(rope_str!("wor")
                    .join(rope_str!("ld")));
        let (start, end) = rope1.split(5);
        let sstr = start.iter().map(|&x| x).collect::<Vec<u8>>();
        let estr = end.iter().map(|&x| x).collect::<Vec<u8>>();
        assert_eq!(sstr, b"hello".to_vec());
        assert_eq!(estr, b" world".to_vec());
    }

    #[test]
    fn test_insert() {
        let mut rope1 = rope_str!("hel").join(rope_str!("ló ")).join(rope_str!("wòr")
                        .join(rope_str!("ld")));
        let rope2 = rope_str!("beau").join(rope_str!("tifü").join(rope_str!("l ")));
        rope1.insert(7, rope2);
        assert_eq!(rope1, rope_str!("helló beautifül wòrld"));
    }

    #[test]
    fn test_delete() {
        let mut rope = rope_str!("hel").join(rope_str!("ló ").join(rope_str!("beau")
                       .join(rope_str!("tifü").join(rope_str!("l ")))))
                       .join(rope_str!("wòr").join(rope_str!("ld")));
        let res = rope.delete(7, 18);
        assert_eq!(rope, rope_str!("helló wòrld"));
        assert_eq!(res, rope_str!("beautifül "));
    }

    #[test]
    fn test_from_iter() {
        let mut iter = vec![1u8, 2, 3, 4, 1, 5].into_iter();
        assert_eq!(iter.collect::<Rope<u8>>(), rope![1u8, 2, 3, 4, 1, 5]);
    }

    #[test]
    fn test_clear() {
        let mut rope = rope_str!("hello world");
        rope.clear();
        assert_eq!(rope, rope_str!(""));
    }

    #[test]
    fn test_hash() {
        let rope1 = rope_str!("hello world");
        let rope2 = rope_str!("hello ").join(rope_str!("world"));
        let rope3 = rope_str!("hello ").join(rope_str!("worl"));
        assert_eq!(hash(&rope1), hash(&rope2));
        assert!(hash(&rope1) != hash(&rope3));
    }

    #[test]
    fn test_strvector() {
        let rope = rope!("héllo", "beautifúl", "wörld");
        assert_eq!(rope.connect(" ").as_slice(), "héllo beautifúl wörld");
        assert_eq!(rope.concat().as_slice(), "héllobeautifúlwörld");
    }

    #[test]
    fn test_show() {
        let rope = rope![1i, 45, 2, 5, 6];
        assert_eq!(format!("{}", rope).as_slice(), "[1, 45, 2, 5, 6]");
    }


    #[test]
    #[should_fail]
    fn test_fail_index_out_of_bounds() {
        let rope = rope_str!("hel").join(rope_str!("ló ").join(rope_str!("beau")
                   .join(rope_str!("tifü").join(rope_str!("l ")))))
                   .join(rope_str!("wòr").join(rope_str!("ld")));
        println!("{}", rope[24]);
    }

    #[test]
    #[should_fail]
    fn test_fail_split_out_of_bounds() {
        let rope = rope_str!("hello world");
        println!("{}", rope.split(12));
    }


    #[bench]
    fn bench_random_inserts(b: &mut Bencher) {
        b.iter(|| {
            let mut rope: Rope<uint> = Rope::new();
            let mut len = 0u;
            let mut rng = rand::task_rng();
            for _ in range(0u, 30) {
                let v = rand::sample(&mut rng, range(0u, 300), 300);
                let i = rng.gen_range(0u, len + 1);
                len += v.len();
                rope.insert(i, Rope::from_vec(unsafe { ::std::mem::transmute(v) }));
            }
            rope
        })
    }

    #[bench]
    fn bench_random_extends(b: &mut Bencher) {
        b.iter(|| {
            let mut rope: Rope<uint> = Rope::new();
            let mut len = 0u;
            let mut rng = rand::task_rng();
            for _ in range(0u, 30) {
                let v = rand::sample(&mut rng, range(0u, 300), 300);
                len += v.len();
                rope.extend(v.into_iter());
            }
            rope
        })
    }

    #[bench]
    fn bench_random_pushes(b: &mut Bencher) {
        b.iter(|| {
            let mut rope: Rope<uint> = Rope::new();
            let mut len = 0u;
            let mut rng = rand::task_rng();
            for _ in range(0u, 50) {
                let v = rand::random();
                len += 1;
                rope.push(v);
            }
            rope
        })
    }
}
