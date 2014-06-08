//! Rope types for efficient mutable data structures.
//!
//! This provides a trait and type for a [rope](http://en.wikipedia.org/wiki/Rope_(data_structure)),
//! a type of data structure composed of smaller vectors that can be used for efficiently storing
//! and manipulating a very long vector.
//!
//! Example:
//!
//! ```rust
//! #![feature(phase)]
//!
//! #[phase(syntax, link)]
//! extern crate rope;
//!
//! use rope::{Rope, VecRope};
//!
//! fn main() {
//!     let rope: VecRope<u8> = rope![1, 4, 5];
//!     rope.insert(1, rope![2, 3]);
//!     assert_eq!(rope, rope![1, 2, 3, 4, 5]);
//! }
//! ```

use core::slice::Items;
use core::fmt::{Show, Formatter, FormatError};
use core::default::Default;
use core::iter::{Iterator, AdditiveIterator, FromIterator, DoubleEndedIterator, FlatMap};
use core::result::{Result, Ok, Err};
use core::option::{Option, Some, None};
use core::cmp::{Eq, PartialEq, Ord, PartialOrd, Ordering, Greater, Less, Equal};
use core::str::{Str, StrSlice};
use core::slice::{Vector, ImmutableVector};
use core::mem;
use core::cmp;
use alloc::owned::Box;
use super::{Collection, Mutable};
use super::hash::{Hash, Writer};
use super::vec::Vec;
use super::string::String;
use super::str::StrVector;

/*************
 * UTILITIES *
 *************/

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

/*********
 * TRAIT *
 *********/

/// Any rope-like type.
///
/// A rope must implement a few basic rope operations—`append_move`, `split_move`, and
/// `slice`—among some other methods. All other methods are based on those methods.
pub trait Rope<T>: FromIterator<T> {
    /// Takes the root of a rope, replacing it with an empty rope.
    fn take_root(&mut self) -> Self;

    /// Sets the root of a rope to that of another.
    fn set_root(&mut self, other: Self);

    /// Concatenates two ropes, consuming both and returning the joined rope.
    ///
    /// This function should compute in logarithmic time.
    fn append_move(self, other: Self) -> Self;
    
    /// Splits a rope at index `index`, consuming the original and returning the two resulting
    /// ropes. The first rope should have length `index`.
    ///
    /// This function should compute in logarithmic time.
    ///
    /// # Failure
    ///
    /// Fails if `index` is out of bounds.
    fn split(self, index: uint) -> (Self, Self);

    /// Creates a rope from a single element.
    #[inline]
    fn from_elem(elem: T) -> Self {
        FromIterator::from_iter(Some(elem).move_iter())
    }

    /// Appends a rope to another rope in-place.
    ///
    /// This function should compute in logarithmic time.
    #[inline]
    fn append(&mut self, mut rope: Self) {
        let root = self.take_root();
        self.set_root(root.append_move(rope.take_root()));
    }

    /// Pushes a single element onto a rope.
    ///
    /// This function should compute in logarithmic time.
    #[inline]
    fn push(&mut self, elem: T) {
        self.append(Rope::from_elem(elem));
    }

    /// Prepends a rope onto another rope.
    ///
    /// This function should compute in logarithmic time.
    #[inline]
    fn prepend(&mut self, mut rope: Self) {
        let root = self.take_root();
        self.set_root(rope.take_root().append_move(root));
    }

    /// Prepend a single element onto a rope.
    ///
    /// This function should compute in logarithmic time.
    #[inline]
    fn unshift(&mut self, elem: T) {
        self.prepend(Rope::from_elem(elem));
    }

    /// Inserts a rope into another rope in-place.
    ///
    /// This function should compute in logarithmic time.
    #[inline]
    fn insert(&mut self, index: uint, other: Self) {
        let root = self.take_root();
        let (r1, r2) = root.split(index);
        self.set_root(r1.append_move(other).append_move(r2));
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
    fn delete(&mut self, i: uint, u: uint) -> Self {
        let root = self.take_root();
        let (r, r3) = root.split(u);
        let (r1, r2) = r.split(i);
        self.set_root(r1.append_move(r3));
        r2
    }
}

/******************
 * IMPLEMENTATION *
 ******************/

/// A rope with a vector (type `Vec<T>`) at every leaf.
pub struct VecRope<T> {
    tree: VecRopeTree<T>,
}

enum VecRopeTree<T> {
    Branch(uint, uint, Box<VecRopeTree<T>>, Box<VecRopeTree<T>>),
    Leaf(uint, Vec<T>),
}

struct VecRopeNodeItems<'a, T> {
    stack: Vec<&'a VecRopeTree<T>>,
    pos: int,
}

impl<'a, T> VecRopeNodeItems<'a, T> {
    fn from_ropetree(rt: &'a VecRopeTree<T>) -> VecRopeNodeItems<'a, T> {
        VecRopeNodeItems {
            stack: Vec::from_elem(rt.height() + 1, rt),
            pos: 0,
        }
    }
}

impl<'a, T> Iterator<&'a [T]> for VecRopeNodeItems<'a, T> {
    fn next(&mut self) -> Option<&'a [T]> {
        if self.pos < 0 { return None; }
        loop {
            let curr = *self.stack.get(self.pos as uint);
            self.pos -= 1;
            match *curr {
                Branch(_, _, ref left, ref right) => {
                    self.pos += 1;
                    *self.stack.get_mut(self.pos as uint) = &**right;
                    self.pos += 1;
                    *self.stack.get_mut(self.pos as uint) = &**left;
                }
                Leaf(_, ref vec) => return Some(vec.as_slice()),
            }
        }
    }
}

/// A by-reference iterator over the elements in a `VecRope`.
pub type VecRopeItems<'a, T> = FlatMap<'a, &'a [T], VecRopeNodeItems<'a, T>,
                                                    Items<'a, T>>;

static MAX_VEC_LEN: uint = 100;

impl<T> VecRopeTree<T> {
    fn height(&self) -> uint {
        match *self {
            Leaf(_, _) => 0,
            Branch(_, height, _, _) => height,
        }
    }

    fn len(&self) -> uint {
        match *self {
            Branch(weight, _, _, ref right) => weight + right.len(),
            Leaf(weight, _) => weight,
        }
    }

    fn index<'a>(&'a self, i: uint) -> &'a T {
        match *self {
            Branch(weight, _, _, ref right) if weight < i => right.index(i - weight),
            Branch(_, _, ref left, _) => left.index(i),
            Leaf(_, ref vec) => vec.get(i),
        }
    }

    fn index_mut<'a>(&'a mut self, i: uint) -> &'a mut T {
        match *self {
            Branch(weight, _, _, ref mut right) if weight < i => right.index_mut(i - weight),
            Branch(_, _, ref mut left, _) => left.index_mut(i),
            Leaf(_, ref mut vec) => vec.get_mut(i),
        }
    }

    fn split_vec(self, i: uint, mut v: Vec<VecRopeTree<T>>)
                 -> (VecRopeTree<T>, Vec<VecRopeTree<T>>) {
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
            Leaf(len, vec) => {
                let mut start = vec![];
                let mut end = vec![];
                if i > len {
                    fail!("index out of bounds: the len is {} but the index is {}", len, i);
                }
                for (index, v) in vec.move_iter().enumerate() {
                    if index < i {
                        start.push(v);
                    } else {
                        end.push(v)
                    }
                }
                v.push(Leaf(len - i, end));
                (Leaf(i, start), v)
            }
        }
    }

    fn split(self, i: uint) -> (VecRopeTree<T>, VecRopeTree<T>) {
        let (tree, vec) = self.split_vec(i, vec![]);
        let mut iter = vec.move_iter().rev();
        let first = match iter.next() {
            Some(i) => i,
            None => return (tree, Leaf(0, vec![])),
        };
        (tree, iter.fold(first, |x, y| x.append(y)))
    }

    fn append(self, other: VecRopeTree<T>) -> VecRopeTree<T> {
        match self {
            Branch(len, height, l, r) => {
                match other {
                    Branch(o_len, o_height, o_l, o_r) => {
                        // Branch + Branch
                        let new_height = std::cmp::max(height, o_height);
                        Branch(len, new_height + 1,
                               box Branch(len, height, l, r),
                               box Branch(o_len, o_height, o_l, o_r))
                    }
                    Leaf(o_len, o_v) => {
                        // Branch + Leaf
                        match r {
                            box r_br @ Branch(..) => {
                                Branch(len, height + 1,
                                       box Branch(len, height, l, box r_br),
                                       box Leaf(o_len, o_v))
                            }
                            box Leaf(self_r_len, mut self_r_v) => {
                                let result_len = o_len + self_r_len;
                                if result_len <= MAX_VEC_LEN {
                                    self_r_v.extend(o_v.move_iter());
                                    Branch(result_len, height, l,
                                           box Leaf(result_len,  self_r_v))
                                } else {
                                    Branch(len, height + 1,
                                           box Branch(len, height, l, box Leaf(self_r_len, self_r_v)),
                                           box Leaf(o_len, o_v))
                                }
                            }
                        }
                    }
                }
            },
            Leaf(len, mut self_v) => {
                match other {
                    Branch(o_len, o_height, o_l, o_r) => {
                        // Leaf + Branch
                        Branch(len, o_height + 1,
                               box Leaf(len, self_v),
                               box Branch(o_len, o_height, o_l, o_r))
                    }
                    Leaf(o_len, o_v) => {
                        // Leaf + Leaf
                        let result_len = o_len + len;
                        if result_len <= MAX_VEC_LEN {
                            self_v.extend(o_v.move_iter());
                            Leaf(result_len, self_v)
                        } else {
                            Branch(len, 1,
                                   box Leaf(len, self_v),
                                   box Leaf(o_len, o_v))
                        }
                    }
                }
            }
        }
    }

    fn node_iter<'a>(&'a self) -> VecRopeNodeItems<'a, T> {
        VecRopeNodeItems::from_ropetree(self)
    }

    fn iter<'a>(&'a self) -> VecRopeItems<'a, T> {
        self.node_iter().flat_map(|x| x.iter())
    }
}

impl<T: PartialEq> VecRopeTree<T> {
    fn eq(&self, other: &VecRopeTree<T>) -> bool {
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


impl<T> VecRope<T> {
    /// Constructs a new, empty `VecRope`.
    pub fn new() -> VecRope<T> {
        VecRope { tree: Leaf(0, vec![]) }
    }

    /// Constructs a `VecRope<T>` from a `Vec<T>`.
    pub fn from_vec(vec: Vec<T>) -> VecRope<T> {
        VecRope { tree: Leaf(vec.as_slice().len(), vec) }
    }

    /// Creates an iterator that iterates over the elements in a `VecRope`.
    pub fn iter<'a>(&'a self) -> VecRopeItems<'a, T> {
        self.tree.iter()
    }

    /// Returns a reference to the value at index `index`.
    ///
    /// # Failure
    ///
    /// Fails if `index` is out of bounds.
    pub fn get<'a>(&'a self, index: uint) -> &'a T {
        self.tree.index(index)
    }

    /// Returns a mutable reference to the value at index `index`.
    ///
    /// # Failure
    ///
    /// Fails if `index` is out of bounds.
    pub fn get_mut<'a>(&'a mut self, index: uint) -> &'a mut T {
        self.tree.index_mut(index)
    }
}

impl<T> FromIterator<T> for VecRope<T> {
    fn from_iter<I: Iterator<T>>(mut iterator: I) -> VecRope<T> {
        let vec: Vec<_> = iterator.collect();
        VecRope { tree: Leaf(vec.len(), vec) }
    }
}

impl<T> Rope<T> for VecRope<T> {
    fn take_root(&mut self) -> VecRope<T> {
        VecRope { tree: mem::replace(&mut self.tree, Leaf(0, vec![])) }
    }

    fn set_root(&mut self, other: VecRope<T>) {
        self.tree = other.tree;
    }

    fn append_move(self, other: VecRope<T>) -> VecRope<T> {
        VecRope { tree: self.tree.append(other.tree) }
    }

    fn split(self, i: uint) -> (VecRope<T>, VecRope<T>) {
        let (rt1, rt2) = self.tree.split(i);
        (VecRope { tree: rt1 }, VecRope { tree: rt2 })
    }
}

impl<T> Collection for VecRope<T> {
    fn len(&self) -> uint {
        self.tree.len()
    }
}

impl<T: PartialEq> PartialEq for VecRope<T> {
    fn eq(&self, other: &VecRope<T>) -> bool {
        self.tree.eq(&other.tree)
    }
}

impl<T: PartialEq> Eq for VecRope<T> {}

impl<T: Show> Show for VecRope<T> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FormatError> {
        write!(f, "{}", self.iter().collect::<Vec<&T>>())
    }
}

impl<T: PartialOrd> PartialOrd for VecRope<T> {
    fn lt(&self, other: &VecRope<T>) -> bool {
        for x in zip_length(self.iter(), other.iter()) {
            let (c, d) = match x {
                Ok(x) => x,
                // Err(false) means self terminated first
                // Err(true)  means other terminated first
                Err(b) => return b,
            };
            if c < d {
                return true;
            } else if c > d {
                return false;
            }
        }
        false
    }
}

impl<T: Ord> Ord for VecRope<T> {
    fn cmp(&self, other: &VecRope<T>) -> Ordering {
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

impl<T> Mutable for VecRope<T> {
    #[inline]
    fn clear(&mut self) {
        self.tree = Leaf(0, vec![]);
    }
}

impl<T> Default for VecRope<T> {
    #[inline]
    fn default() -> VecRope<T> {
        VecRope { tree: Leaf(0, vec![]) }
    }
}

impl<S: Writer, T: Hash<S>> Hash<S> for VecRope<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.len().hash(state);
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}

impl<S: Str> StrVector for VecRope<S> {
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

/// Constructs a new `VecRope` from a list of elements.
#[macro_export]
macro_rules! rope {
    ($($e:expr),*) => {
        VecRope::from_vec(vec![$($e),*])
    }
}


/*********
 * TESTS *
 *********/

#[cfg(test)]
mod tests {
    use super::VecRope;
    use super::Rope;
    use super::super::hash::hash;
    use std::prelude::*;
    use super::super::Mutable;
    use super::super::str::StrVector;

    macro_rules! rope_str {
        ($e:expr) => {
            VecRope::from_vec($e.bytes().collect::<super::super::vec::Vec<u8>>())
        }
    }

    #[test]
    fn rope_len() {
        assert_eq!(rope_str!("hello").len(), 5);
        assert_eq!(rope_str!("").len(), 0);
        assert_eq!(rope_str!("úÂḿ").len(), 7);

        assert_eq!(rope_str!("hel").append_move(rope_str!("lo")).len(), 5);
    }

    #[test]
    fn rope_index() {
        let rope = rope_str!("helló").append_move(rope_str!(" wor")).append_move(rope_str!("ld"));
        assert_eq!(*rope.get(3), 'l' as u8);
        assert_eq!(*rope.get(7), 'w' as u8);
        assert_eq!(*rope.get(11), 'd' as u8);

        let mut rope = rope;
        *rope.get_mut(1) = 'u' as u8;
        assert_eq!(rope, rope_str!("hulló world"));
    }

    #[test]
    fn rope_append_move() {
        let rope = rope_str!("");
        let rope = rope.append_move(rope_str!("ab "));
        let rope = rope.append_move(rope_str!("cd"));
        println!("{}", rope);
        assert_eq!(rope, rope_str!("ab cd"));

        assert_eq!(rope_str!("hello ").append_move(rope_str!("world")), rope_str!("hello world"));
    }

    #[test]
    fn rope_append() {
        let mut rope = rope_str!("hello");
        rope.append(rope_str!(" world"));
        assert_eq!(rope, rope_str!("hello world"));
    }

    #[test]
    fn rope_push() {
        let mut rope = rope_str!("hello world");
        rope.push('!' as u8);
        assert_eq!(rope, rope_str!("hello world!"));
    }

    #[test]
    fn rope_prepend() {
        let mut rope = rope_str!("world");
        rope.prepend(rope_str!("hello "));
        assert_eq!(rope, rope_str!("hello world"));
    }

    #[test]
    fn rope_unshift() {
        let mut rope = rope_str!("dlrow olleh");
        rope.unshift('!' as u8);
        assert_eq!(rope, rope_str!("!dlrow olleh"));
    }

    #[test]
    fn rope_iter() {
        let rope = rope_str!("helló!");

        for (i, &c) in rope.iter().enumerate() {
            assert_eq!(c, match i {
                0 => 'h' as u8,
                1 => 'e' as u8,
                2 => 'l' as u8,
                3 => 'l' as u8,
                4 => 195,
                5 => 179,
                6 => '!' as u8,
                _ => unreachable!(),
            });
        }
    }

    #[test]
    fn rope_eq() {
        let rope1 = rope_str!("hello");
        let rope2 = rope_str!("hello");
        let rope3 = rope_str!("hellol");
        let rope4 = rope_str!("hell");
        assert_eq!(rope1, rope2);
        assert!(rope1 != rope3);
        assert!(rope1 != rope4);
    }

    #[test]
    fn rope_partialord() {
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
    fn rope_ord() {
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
    fn rope_split() {
        let rope1 = rope_str!("hel").append_move(rope_str!("lo ")).append_move(rope_str!("wor")
                    .append_move(rope_str!("ld")));
        let (start, end) = rope1.split(5);
        let sstr = start.iter().map(|&x| x).collect::<Vec<u8>>();
        let estr = end.iter().map(|&x| x).collect::<Vec<u8>>();
        assert_eq!(sstr, "hello".bytes().collect());
        assert_eq!(estr, " world".bytes().collect());
    }

    #[test]
    fn rope_insert() {
        let mut rope1 = rope_str!("hel").append_move(rope_str!("ló ")).append_move(rope_str!("wòr")
                        .append_move(rope_str!("ld")));
        let rope2 = rope_str!("beau").append_move(rope_str!("tifü").append_move(rope_str!("l ")));
        rope1.insert(7, rope2);
        assert_eq!(rope1, rope_str!("helló beautifül wòrld"));
    }

    #[test]
    fn rope_delete() {
        let mut rope = rope_str!("hel").append_move(rope_str!("ló ").append_move(rope_str!("beau")
                       .append_move(rope_str!("tifü").append_move(rope_str!("l ")))))
                       .append_move(rope_str!("wòr").append_move(rope_str!("ld")));
        let res = rope.delete(7, 18);
        assert_eq!(rope, rope_str!("helló wòrld"));
        assert_eq!(res, rope_str!("beautifül "));
    }

    #[test]
    fn rope_from_iter() {
        let mut iter = vec![1u8, 2, 3, 4, 1, 5].move_iter();
        assert_eq!(iter.collect::<VecRope<u8>>(), rope![1u8, 2, 3, 4, 1, 5]);
    }

    #[test]
    fn rope_clear() {
        let mut rope = rope_str!("hello world");
        rope.clear();
        assert_eq!(rope, rope_str!(""));
    }

    #[test]
    fn rope_hash() {
        let rope1 = rope_str!("hello world");
        let rope2 = rope_str!("hello ").append_move(rope_str!("world"));
        let rope3 = rope_str!("hello ").append_move(rope_str!("worl"));
        assert_eq!(hash(&rope1), hash(&rope2));
        assert!(hash(&rope1) != hash(&rope3));
    }

    #[test]
    fn rope_strvector() {
        let rope = rope!("héllo", "beautifúl", "wörld");
        assert_eq!(rope.connect(" ").to_str(), "héllo beautifúl wörld".to_string());
        assert_eq!(rope.concat().to_str(), "héllobeautifúlwörld".to_string());
    }

    // #[should_fail] tests //

    #[test]
    #[should_fail]
    fn rope_fail_index_out_of_bounds() {
        let rope = rope_str!("hel").append_move(rope_str!("ló ").append_move(rope_str!("beau")
                   .append_move(rope_str!("tifü").append_move(rope_str!("l ")))))
                   .append_move(rope_str!("wòr").append_move(rope_str!("ld")));
        println!("{}", rope.get(24));
    }

    #[test]
    #[should_fail]
    fn rope_fail_split_out_of_bounds() {
        let rope = rope_str!("hello world");
        println!("{}", rope.split(12));
    }
}
