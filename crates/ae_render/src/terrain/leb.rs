//! CPU reference of the Concurrent Binary Tree (CBT) + Longest Edge Bisection (LEB)
//! subdivision used on the GPU (see `shaders/cbt.glsl`, which mirrors this file).
//!
//! Based on J. Dupuy, "Concurrent Binary Trees (with application to longest edge
//! bisection)", HPG 2020, and the reference libleb/libcbt.
//!
//! * Every node of the LEB subdivision is a heap index: root = 1, children 2n / 2n+1.
//!   The domain is a square made of two root triangles (heap ids 2 and 3).
//! * The CBT stores one bit per node at the maximum depth `D` (the bitfield), plus a
//!   sum-reduction tree over that bitfield. A subdivision leaf at depth d < D is encoded
//!   by the bit of its left-most descendant at depth D. Splitting a node sets the bit
//!   of its right child's left-most descendant, merging clears it.
//! * To keep memory small, the sum tree is stored explicitly only down to depth
//!   `T = D - 5`; a node at depth T owns exactly one 32-bit bitfield word and the
//!   five levels below it are evaluated with popcounts.
//!
//! Triangle vertices live on an integer lattice (`ROOT_SIZE` units per square side), so
//! every bisection midpoint is exact and two neighbouring leaves always compute
//! bit-identical shared vertices -> no cracks.

#![allow(dead_code)] // CPU reference: most of it is exercised by the unit tests only

use glam::IVec2;

/// Lattice units per side of the root square. Supports up to `LATTICE_BITS * 2` levels
/// of bisection below the root triangles with exact midpoints.
pub const LATTICE_BITS: u32 = 20;
pub const ROOT_SIZE: i32 = 1 << LATTICE_BITS;

/// Levels of bisection in the per-leaf sub-mesh drawn by the vertex shader.
/// MUST be even: with an even number of uniform LEB levels every edge of a leaf (legs and
/// hypotenuse alike) is cut into the same number of segments, so a leaf and its
/// one-level-different neighbour still share identical vertices along their common edge.
pub const SUBMESH_LEVELS: u32 = 4;
const _: () = assert!(SUBMESH_LEVELS.is_multiple_of(2));
/// Weight denominator of sub-mesh vertices (barycentric weights sum to this).
pub const SUBMESH_DENOM: i32 = 1 << (SUBMESH_LEVELS / 2);

#[inline]
pub fn depth_of(id: u32) -> u32 {
    31 - id.leading_zeros()
}

#[inline]
fn bit(id: u32, b: i32) -> u32 {
    if b < 0 { 0 } else { (id >> b) & 1 }
}

/// Triangle (v0, v1, v2) of a heap node, v1 being the right-angle apex and v0-v2 the
/// hypotenuse. Requires depth >= 1.
pub fn decode_triangle(id: u32) -> [IVec2; 3] {
    let depth = depth_of(id) as i32;
    let mut v = [IVec2::new(0, ROOT_SIZE), IVec2::new(0, 0), IVec2::new(ROOT_SIZE, 0)];
    if bit(id, depth - 1) == 1 {
        v = [v[2], v[0] + v[2] - v[1], v[0]];
    }
    for b in (0..depth - 1).rev() {
        let m = (v[0] + v[2]) / 2;
        v = if bit(id, b) == 0 {
            [v[0], m, v[1]]
        } else {
            [v[1], m, v[2]]
        };
    }
    v
}

/// Same-depth neighbour ids (0 = none): [left (across v1-v2), right (across v0-v1),
/// edge (across hypotenuse v0-v2), node itself].
pub fn neighbors(id: u32) -> [u32; 4] {
    let depth = depth_of(id) as i32;
    let b = bit(id, (depth - 1).max(0));
    let mut n = [0u32, 0, 3 - b, 2 + b];
    for bb in (0..depth - 1).rev() {
        let [n1, n2, n3, n4] = n;
        let b2 = (n2 != 0) as u32;
        let b3 = (n3 != 0) as u32;
        n = if bit(id, bb) == 0 {
            [n4 << 1 | 1, n3 << 1 | b3, n2 << 1 | b2, n4 << 1]
        } else {
            [n3 << 1, n4 << 1, n1 << 1, n4 << 1 | 1]
        };
    }
    n
}

#[inline]
pub fn edge_neighbor(id: u32) -> u32 {
    neighbors(id)[2]
}

#[derive(Clone)]
pub struct Cbt {
    pub max_depth: u32,
    /// Sum tree, indexed by heap id, nodes of depth 0..=T. Index 0 unused.
    pub tree: Vec<u32>,
    /// One bit per node at max depth.
    pub bits: Vec<u32>,
}

impl Cbt {
    pub fn tree_depth(max_depth: u32) -> u32 {
        max_depth - 5
    }

    /// Two root triangles (heap ids 2 and 3).
    pub fn new(max_depth: u32) -> Self {
        assert!((6..=30).contains(&max_depth));
        let t = Self::tree_depth(max_depth);
        let mut c = Cbt {
            max_depth,
            tree: vec![0; 1 << (t + 1)],
            bits: vec![0; 1 << (max_depth - 5)],
        };
        c.set_bit(c.bit_index(2, 1), true);
        c.set_bit(c.bit_index(3, 1), true);
        c.reduce();
        c
    }

    fn t(&self) -> u32 {
        Self::tree_depth(self.max_depth)
    }

    /// Bitfield index of the left-most max-depth descendant of a node.
    #[inline]
    pub fn bit_index(&self, id: u32, depth: u32) -> u32 {
        (id << (self.max_depth - depth)) - (1 << self.max_depth)
    }

    fn set_bit(&mut self, idx: u32, v: bool) {
        let w = &mut self.bits[(idx >> 5) as usize];
        if v {
            *w |= 1 << (idx & 31)
        } else {
            *w &= !(1 << (idx & 31))
        }
    }

    pub fn reduce(&mut self) {
        let t = self.t();
        for j in 0..(1u32 << t) {
            self.tree[((1 << t) + j) as usize] = self.bits[j as usize].count_ones();
        }
        for d in (0..t).rev() {
            for j in 0..(1u32 << d) {
                let id = ((1 << d) + j) as usize;
                self.tree[id] = self.tree[2 * id] + self.tree[2 * id + 1];
            }
        }
    }

    pub fn leaf_count(&self) -> u32 {
        self.tree[1]
    }

    /// Number of set bits under a node.
    pub fn node_sum(&self, id: u32) -> u32 {
        let depth = depth_of(id);
        let t = self.t();
        if depth <= t {
            return self.tree[id as usize];
        }
        let s = depth - t;
        let word = self.bits[((id >> s) - (1 << t)) as usize];
        let width = 32 >> s;
        let offset = (id & ((1 << s) - 1)) * width;
        ((word >> offset) & ((1u32 << width) - 1)).count_ones()
    }

    pub fn is_leaf(&self, id: u32) -> bool {
        self.node_sum(id) == 1
    }

    /// Heap id of the `handle`-th leaf.
    pub fn decode(&self, mut handle: u32) -> u32 {
        let mut id = 1u32;
        while self.node_sum(id) > 1 {
            let c = self.node_sum(2 * id);
            if handle < c {
                id *= 2;
            } else {
                handle -= c;
                id = 2 * id + 1;
            }
        }
        id
    }

    pub fn split_node(&mut self, id: u32) {
        let d = depth_of(id);
        if d < self.max_depth {
            let idx = self.bit_index(2 * id + 1, d + 1);
            self.set_bit(idx, true);
        }
    }

    pub fn merge_node(&mut self, id: u32) {
        let idx = self.bit_index(id | 1, depth_of(id));
        self.set_bit(idx, false);
    }

    /// Splits a leaf and as many of its hypotenuse neighbours as needed to stay conforming.
    pub fn split_conforming(&mut self, id: u32) {
        if depth_of(id) >= self.max_depth {
            return;
        }
        self.split_node(id);
        let mut n = edge_neighbor(id);
        while n >= 2 {
            self.split_node(n);
            n >>= 1;
            if n >= 2 {
                self.split_node(n);
                n = edge_neighbor(n);
            }
        }
    }
}

/// The diamond a leaf belongs to: its parent and the parent's hypotenuse neighbour
/// (the parent itself when on the domain boundary).
pub fn diamond(id: u32) -> (u32, u32) {
    let base = id >> 1;
    let top = edge_neighbor(base);
    (base, if top == 0 { base } else { top })
}

/// Whether the leaf's diamond can be merged, judged on a snapshot of the tree.
pub fn can_merge(snapshot: &Cbt, id: u32) -> bool {
    if depth_of(id) <= 1 {
        return false;
    }
    let (_, top) = diamond(id);
    snapshot.is_leaf(id ^ 1) && snapshot.is_leaf(2 * top) && snapshot.is_leaf(2 * top + 1)
}

/// Applies a merge (to be called only when `can_merge` holds and the diamond agrees).
pub fn merge_conforming(cbt: &mut Cbt, id: u32) {
    let (_, top) = diamond(id);
    cbt.merge_node(id);
    cbt.merge_node(2 * top + 1);
}

/// Sub-mesh: `SUBMESH_LEVELS` uniform bisections of the reference triangle, as integer
/// barycentric weights (w0, w1, w2) over (v0, v1, v2) summing to `SUBMESH_DENOM`.
/// Returns (unique vertices, triangle indices). Triangles keep LEB order (v0, apex, v2).
pub fn build_submesh() -> (Vec<[i32; 3]>, Vec<u16>) {
    let n = SUBMESH_DENOM;
    // Work in a 2D lattice where the reference triangle is (0,n),(0,0),(n,0) scaled x2
    // so all midpoints stay integral, then convert back to weights.
    let s = 2 * n;
    let mut tris: Vec<[IVec2; 3]> = vec![[IVec2::new(0, s), IVec2::new(0, 0), IVec2::new(s, 0)]];
    for _ in 0..SUBMESH_LEVELS {
        let mut next = Vec::with_capacity(tris.len() * 2);
        for v in tris {
            let m = (v[0] + v[2]) / 2;
            next.push([v[0], m, v[1]]);
            next.push([v[1], m, v[2]]);
        }
        tris = next;
    }
    let mut verts: Vec<IVec2> = Vec::new();
    let mut idx = Vec::new();
    for t in &tris {
        for p in t {
            let i = verts.iter().position(|q| q == p).unwrap_or_else(|| {
                verts.push(*p);
                verts.len() - 1
            });
            idx.push(i as u16);
        }
    }
    // p = w0*(0,s) + w1*(0,0) + w2*(s,0)  (in units of s/n)
    let weights = verts
        .iter()
        .map(|p| {
            assert!(p.x % 2 == 0 && p.y % 2 == 0);
            let w2 = p.x / 2;
            let w0 = p.y / 2;
            [w0, n - w0 - w2, w2]
        })
        .collect();
    (weights, idx)
}

/// Checks that the leaves tile the square and every leaf edge is either on the boundary
/// or shared, with identical endpoints, by exactly one other leaf (no T-junctions, hence
/// no cracks). Returns the leaf count.
pub fn check_conforming(c: &Cbt) -> Result<u32, String> {
    use std::collections::HashMap;
    let mut edges: HashMap<(IVec2, IVec2), u32> = HashMap::new();
    let mut area2: i128 = 0;
    for h in 0..c.leaf_count() {
        let t = decode_triangle(c.decode(h));
        let a = (t[1] - t[0]).as_i64vec2();
        let b = (t[2] - t[0]).as_i64vec2();
        area2 += (a.x * b.y - a.y * b.x).unsigned_abs() as i128;
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let (p, q) = (t[i], t[j]);
            let key = if (p.x, p.y) < (q.x, q.y) { (p, q) } else { (q, p) };
            *edges.entry(key).or_default() += 1;
        }
    }
    if area2 != 2 * (ROOT_SIZE as i128).pow(2) {
        return Err(format!("leaves do not tile the domain (2*area = {area2})"));
    }
    for ((p, q), n) in edges {
        let boundary = (p.x == q.x && (p.x == 0 || p.x == ROOT_SIZE)) || (p.y == q.y && (p.y == 0 || p.y == ROOT_SIZE));
        if n != if boundary { 1 } else { 2 } {
            return Err(format!("T-junction at edge {p:?}-{q:?} (shared by {n})"));
        }
    }
    Ok(c.leaf_count())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shares_edge(a: [IVec2; 3], b: [IVec2; 3], ea: (usize, usize)) -> bool {
        let (p, q) = (a[ea.0], a[ea.1]);
        b.contains(&p) && b.contains(&q)
    }

    #[test]
    fn neighbours_share_edges() {
        for id in 2u32..(1 << 12) {
            let t = decode_triangle(id);
            let [l, r, e, me] = neighbors(id);
            assert_eq!(me, id);
            for (n, edge) in [(l, (1, 2)), (r, (0, 1)), (e, (0, 2))] {
                if n != 0 {
                    assert_eq!(depth_of(n), depth_of(id));
                    assert!(shares_edge(t, decode_triangle(n), edge), "id {id} nb {n}");
                } else {
                    // must be on the domain boundary
                    let (p, q) = (t[edge.0], t[edge.1]);
                    let on = |c: fn(IVec2) -> i32, v| c(p) == v && c(q) == v;
                    assert!(
                        on(|v| v.x, 0) || on(|v| v.y, 0) || on(|v| v.x, ROOT_SIZE) || on(|v| v.y, ROOT_SIZE),
                        "id {id} edge {edge:?} has no neighbour but is interior"
                    );
                }
            }
        }
    }

    fn hash(mut x: u64) -> u64 {
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        x ^ (x >> 33)
    }

    #[test]
    fn random_split_merge_stays_conforming() {
        let mut c = Cbt::new(14);
        for iter in 0..80u64 {
            let snap = c.clone();
            let n = snap.leaf_count();
            if iter % 2 == 0 {
                // split pass: concentrate refinement around a moving focus point
                let f = IVec2::new(
                    (hash(iter / 8) % ROOT_SIZE as u64) as i32,
                    (hash(iter / 8 + 99) % ROOT_SIZE as u64) as i32,
                );
                for h in 0..n {
                    let id = snap.decode(h);
                    let t = decode_triangle(id);
                    let ctr = (t[0] + t[1] + t[2]) / 3;
                    let d = (ctr - f).as_vec2().length() / ROOT_SIZE as f32;
                    let size = (depth_of(id) as f32 * -0.5).exp2();
                    if size > d * 0.5 || hash(id as u64 ^ iter << 40).is_multiple_of(7) {
                        c.split_conforming(id);
                    }
                }
            } else {
                for h in 0..n {
                    let id = snap.decode(h);
                    if !can_merge(&snap, id) {
                        continue;
                    }
                    let (b, t) = diamond(id);
                    let key = b.min(t) as u64 | (b.max(t) as u64) << 32;
                    if !hash(key ^ iter).is_multiple_of(3) {
                        merge_conforming(&mut c, id);
                    }
                }
            }
            c.reduce();
            check_conforming(&c).unwrap();
        }
        assert!(c.leaf_count() > 50);
    }

    #[test]
    fn submesh_is_uniform() {
        let (w, idx) = build_submesh();
        assert_eq!(idx.len() as u32, 3 << SUBMESH_LEVELS);
        assert!(
            w.iter()
                .all(|w| w.iter().sum::<i32>() == SUBMESH_DENOM && w.iter().all(|&x| x >= 0))
        );
        // each edge of the reference triangle carries SUBMESH_DENOM + 1 vertices
        let on01 = w.iter().filter(|w| w[2] == 0).count();
        let on12 = w.iter().filter(|w| w[0] == 0).count();
        let on02 = w.iter().filter(|w| w[1] == 0).count();
        let k = SUBMESH_DENOM as usize + 1;
        assert_eq!((on01, on12, on02), (k, k, k));
    }
}
