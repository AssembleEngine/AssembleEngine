//! Spatial queries over entity positions, in batches.
//!
//! Each tick the host builds a [`Grid`] from position columns (an O(n)
//! counting sort into uniform cells), then answers whole batches of queries
//! at once: [`Grid::radius`], [`Grid::boxes`], [`Grid::nearest`] and
//! [`Grid::sweep`]. A batch's answer is a [`Hits`] stream: one contiguous
//! run of hits per query, in query order (CSR layout), ready to hand to a
//! WASM system as columns.
//!
//! Determinism: every query's hits come out in a fixed order (item index;
//! distance then index for nearest-k; entry time then index for sweeps),
//! computed with plain f32 `+ - * /` and `sqrt` in a fixed order. Batches
//! are split into fixed-size chunks that run in parallel on the current
//! rayon pool and are concatenated in chunk order, so the output is the same
//! for any thread count. Items or queries outside the grid bounds are
//! clamped into the edge cells for lookup but always tested exactly.

use rayon::prelude::*;

/// Queries per parallel chunk. Fixed, so chunking never depends on threads.
pub const CHUNK: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

/// A circle query: items whose centre is within `r` of (x, y), boundary
/// included.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Circle {
    pub x: f32,
    pub y: f32,
    pub r: f32,
}

/// A box query: items whose centre lies in the box, boundary included.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

/// A swept circle from (x0, y0) to (x1, y1) with radius `pad` (0 for a ray):
/// hits every item whose disc (item radius) it touches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub pad: f32,
}

/// Nearest-k query: up to `k` items nearest (x, y) within `max_r`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nearest {
    pub x: f32,
    pub y: f32,
    pub max_r: f32,
    pub k: u32,
}

/// Answers to a batch: query `q` hit `items[offsets[q]..offsets[q + 1]]`.
/// `values` runs parallel to `items` for queries that have a per-hit value
/// (squared distance for nearest-k, entry time 0..=1 for sweeps).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hits {
    pub offsets: Vec<u32>,
    pub items: Vec<u32>,
    pub values: Vec<f32>,
}

impl Hits {
    pub fn query_count(&self) -> usize { self.offsets.len().saturating_sub(1) }
    pub fn of(&self, q: usize) -> &[u32] { &self.items[self.offsets[q] as usize..self.offsets[q + 1] as usize] }
    pub fn values_of(&self, q: usize) -> &[f32] { &self.values[self.offsets[q] as usize..self.offsets[q + 1] as usize] }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }
    fn clear(&mut self) {
        self.offsets.clear();
        self.items.clear();
        self.values.clear();
    }
}

/// Uniform grid over item centres; rebuilt each tick.
#[derive(Clone, Debug)]
pub struct Grid {
    bounds: Bounds,
    cell: f32,
    inv_cell: f32,
    cols: i32,
    rows: i32,
    /// Items of cell `c` are `order[start[c]..start[c + 1]]`, ascending.
    start: Vec<u32>,
    order: Vec<u32>,
    x: Vec<f32>,
    y: Vec<f32>,
    radius: Vec<f32>,
    max_radius: f32,
}

impl Grid {
    /// An empty grid covering `bounds` with square cells of `cell` units.
    pub fn new(bounds: Bounds, cell: f32) -> Self {
        assert!(cell > 0.0 && bounds.max_x > bounds.min_x && bounds.max_y > bounds.min_y, "bad grid shape");
        let cols = ((bounds.max_x - bounds.min_x) / cell).ceil().max(1.0) as i32;
        let rows = ((bounds.max_y - bounds.min_y) / cell).ceil().max(1.0) as i32;
        Self {
            bounds, cell, inv_cell: 1.0 / cell, cols, rows,
            start: vec![0; (cols * rows) as usize + 1], order: Vec::new(),
            x: Vec::new(), y: Vec::new(), radius: Vec::new(), max_radius: 0.0,
        }
    }

    /// Replace the items: centres `x`, `y` and radii `radius` (empty slice for
    /// all zero). Item `i` is row `i` of the columns. Reuses allocations.
    pub fn rebuild(&mut self, x: &[f32], y: &[f32], radius: &[f32]) {
        assert_eq!(x.len(), y.len());
        assert!(radius.is_empty() || radius.len() == x.len());
        let n = x.len();
        self.x.clear();
        self.x.extend_from_slice(x);
        self.y.clear();
        self.y.extend_from_slice(y);
        self.radius.clear();
        if radius.is_empty() { self.radius.resize(n, 0.0) } else { self.radius.extend_from_slice(radius) }
        self.max_radius = self.radius.iter().fold(0.0f32, |m, &r| m.max(r));
        // Counting sort by cell; items stay in index order within a cell.
        let cells = (self.cols * self.rows) as usize;
        self.start.clear();
        self.start.resize(cells + 1, 0);
        let mut cell_of = Vec::with_capacity(n);
        for i in 0..n {
            let c = self.cell_index(x[i], y[i]);
            cell_of.push(c as u32);
            self.start[c + 1] += 1;
        }
        for c in 0..cells {
            self.start[c + 1] += self.start[c];
        }
        let mut next: Vec<u32> = self.start[..cells].to_vec();
        self.order.clear();
        self.order.resize(n, 0);
        for (i, &c) in cell_of.iter().enumerate() {
            self.order[next[c as usize] as usize] = i as u32;
            next[c as usize] += 1;
        }
    }

    pub fn len(&self) -> usize { self.x.len() }
    pub fn is_empty(&self) -> bool { self.x.is_empty() }
    pub fn cell_size(&self) -> f32 { self.cell }

    #[inline]
    fn col(&self, x: f32) -> i32 { (((x - self.bounds.min_x) * self.inv_cell).floor() as i32).clamp(0, self.cols - 1) }
    #[inline]
    fn row(&self, y: f32) -> i32 { (((y - self.bounds.min_y) * self.inv_cell).floor() as i32).clamp(0, self.rows - 1) }
    #[inline]
    fn cell_index(&self, x: f32, y: f32) -> usize { (self.row(y) * self.cols + self.col(x)) as usize }

    #[inline]
    fn cell_items(&self, col: i32, row: i32) -> &[u32] {
        let c = (row * self.cols + col) as usize;
        &self.order[self.start[c] as usize..self.start[c + 1] as usize]
    }

    /// Visit every item in the cells overlapping the box (each item once).
    #[inline]
    fn for_cells(&self, min_x: f32, min_y: f32, max_x: f32, max_y: f32, mut f: impl FnMut(u32)) {
        let (c0, c1, r0, r1) = (self.col(min_x), self.col(max_x), self.row(min_y), self.row(max_y));
        for row in r0..=r1 {
            for col in c0..=c1 {
                for &i in self.cell_items(col, row) {
                    f(i);
                }
            }
        }
    }

    /// Items with centre within each circle; hits ascending by item index.
    pub fn radius(&self, queries: &[Circle], out: &mut Hits) {
        self.batch(queries.len(), out, false, |q, items, _| {
            let Circle { x, y, r } = queries[q];
            let r2 = r * r;
            self.for_cells(x - r, y - r, x + r, y + r, |i| {
                let (dx, dy) = (self.x[i as usize] - x, self.y[i as usize] - y);
                if dx * dx + dy * dy <= r2 { items.push(i); }
            });
            items.sort_unstable();
        });
    }

    /// Items with centre inside each box; hits ascending by item index.
    pub fn boxes(&self, queries: &[Aabb], out: &mut Hits) {
        self.batch(queries.len(), out, false, |q, items, _| {
            let b = queries[q];
            self.for_cells(b.min_x, b.min_y, b.max_x, b.max_y, |i| {
                let (x, y) = (self.x[i as usize], self.y[i as usize]);
                if x >= b.min_x && x <= b.max_x && y >= b.min_y && y <= b.max_y { items.push(i); }
            });
            items.sort_unstable();
        });
    }

    /// Up to `k` items nearest each point within `max_r`, ascending by
    /// (squared distance, item index); `values` holds the squared distances.
    pub fn nearest(&self, queries: &[Nearest], out: &mut Hits) {
        self.batch(queries.len(), out, true, |q, items, values| {
            let Nearest { x, y, max_r, k } = queries[q];
            if k == 0 { return; }
            // Grow the searched square ring by ring until the k-th best is
            // closer than any unsearched cell, or `max_r` is covered.
            let mut best: Vec<(f32, u32)> = Vec::new();
            let (cx, cy) = (self.col(x), self.row(y));
            let max_r2 = max_r * max_r;
            // Rings needed to cover `max_r` (any ring past the grid's size
            // adds nothing, which also bounds an infinite radius).
            let reach = (max_r * self.inv_cell).ceil().min(self.cols.max(self.rows) as f32) as i32 + 1;
            for ring in 0..=reach {
                for row in (cy - ring)..=(cy + ring) {
                    if row < 0 || row >= self.rows { continue; }
                    for col in (cx - ring)..=(cx + ring) {
                        if col < 0 || col >= self.cols { continue; }
                        if (row - cy).abs() != ring && (col - cx).abs() != ring { continue; }
                        for &i in self.cell_items(col, row) {
                            let (dx, dy) = (self.x[i as usize] - x, self.y[i as usize] - y);
                            let d2 = dx * dx + dy * dy;
                            if d2 <= max_r2 { best.push((d2, i)); }
                        }
                    }
                }
                if best.len() >= k as usize {
                    best.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    best.truncate(k as usize);
                    // Everything outside the searched rings is at least
                    // `ring` cells (minus the query's offset in its cell) away.
                    let safe = ring as f32 * self.cell;
                    if best[k as usize - 1].0 <= safe * safe { break; }
                }
            }
            best.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            best.truncate(k as usize);
            for (d2, i) in best {
                items.push(i);
                values.push(d2);
            }
        });
    }

    /// Items whose disc each swept circle touches, ascending by (entry time
    /// along the segment, 0..=1, item index); `values` holds entry times.
    pub fn sweep(&self, queries: &[Segment], out: &mut Hits) {
        self.batch(queries.len(), out, true, |q, items, values| {
            let s = queries[q];
            let reach = s.pad + self.max_radius;
            let (dx, dy) = (s.x1 - s.x0, s.y1 - s.y0);
            let len2 = dx * dx + dy * dy;
            let mut hits: Vec<(f32, u32)> = Vec::new();
            // Rows of cells the capsule crosses, and per row the x extent of
            // the segment within that row's slab (plus reach).
            let (y_lo, y_hi) = (s.y0.min(s.y1) - reach, s.y0.max(s.y1) + reach);
            for row in self.row(y_lo)..=self.row(y_hi) {
                // Edge rows also hold items beyond the bounds: open slabs.
                let base = self.bounds.min_y + row as f32 * self.cell;
                let slab_lo = if row == 0 { f32::NEG_INFINITY } else { base - reach };
                let slab_hi = if row == self.rows - 1 { f32::INFINITY } else { base + self.cell + reach };
                let (mut x_lo, mut x_hi) = (s.x0.min(s.x1), s.x0.max(s.x1));
                if dy != 0.0 {
                    let ta = ((slab_lo - s.y0) / dy).clamp(0.0, 1.0);
                    let tb = ((slab_hi - s.y0) / dy).clamp(0.0, 1.0);
                    let (xa, xb) = (s.x0 + dx * ta, s.x0 + dx * tb);
                    x_lo = xa.min(xb);
                    x_hi = xa.max(xb);
                }
                for col in self.col(x_lo - reach)..=self.col(x_hi + reach) {
                    for &i in self.cell_items(col, row) {
                        if let Some(t) = entry_time(&s, dx, dy, len2, self.x[i as usize], self.y[i as usize], s.pad + self.radius[i as usize]) {
                            hits.push((t, i));
                        }
                    }
                }
            }
            hits.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for (t, i) in hits {
                items.push(i);
                values.push(t);
            }
        });
    }

    /// Run `n` queries in fixed chunks on the current rayon pool. Each query
    /// gets empty scratch buffers for its hits (and values), which are then
    /// appended to the chunk's.
    fn batch(&self, n: usize, out: &mut Hits, with_values: bool, run: impl Fn(usize, &mut Vec<u32>, &mut Vec<f32>) + Sync) {
        out.clear();
        let parts: Vec<(Vec<u32>, Vec<u32>, Vec<f32>)> = (0..n.div_ceil(CHUNK))
            .into_par_iter()
            .map(|chunk| {
                let (lo, hi) = (chunk * CHUNK, ((chunk + 1) * CHUNK).min(n));
                let (mut counts, mut items, mut values) = (Vec::with_capacity(hi - lo), Vec::new(), Vec::new());
                let (mut q_items, mut q_values) = (Vec::new(), Vec::new());
                for q in lo..hi {
                    q_items.clear();
                    q_values.clear();
                    run(q, &mut q_items, &mut q_values);
                    counts.push(q_items.len() as u32);
                    items.extend_from_slice(&q_items);
                    values.extend_from_slice(&q_values);
                }
                (counts, items, values)
            })
            .collect();
        out.offsets.reserve(n + 1);
        out.offsets.push(0);
        let mut total = 0u32;
        for (counts, items, values) in parts {
            for c in counts {
                total += c;
                out.offsets.push(total);
            }
            out.items.extend_from_slice(&items);
            if with_values { out.values.extend_from_slice(&values); }
        }
    }
}

/// First time t in [0, 1] at which the point (x0, y0) + t (dx, dy) is
/// within `r` of (px, py), if any.
#[inline]
fn entry_time(s: &Segment, dx: f32, dy: f32, len2: f32, px: f32, py: f32, r: f32) -> Option<f32> {
    let (fx, fy) = (s.x0 - px, s.y0 - py);
    let c = fx * fx + fy * fy - r * r;
    if c <= 0.0 { return Some(0.0); }
    if len2 == 0.0 { return None; }
    let b = fx * dx + fy * dy;
    if b >= 0.0 { return None; }
    let disc = b * b - len2 * c;
    if disc < 0.0 { return None; }
    let t = (-b - disc.sqrt()) / len2;
    (t <= 1.0).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 40) as f32 / (1u64 << 24) as f32
    }

    #[test]
    fn counting_sort_keeps_index_order_per_cell() {
        let mut g = Grid::new(Bounds { min_x: 0.0, min_y: 0.0, max_x: 10.0, max_y: 10.0 }, 5.0);
        g.rebuild(&[1.0, 6.0, 2.0, -50.0, 7.0], &[1.0, 1.0, 2.0, 3.0, 9.0], &[]);
        assert_eq!(g.cell_items(0, 0), &[0, 2, 3]);
        assert_eq!(g.cell_items(1, 0), &[1]);
        assert_eq!(g.cell_items(1, 1), &[4]);
    }

    #[test]
    fn entry_time_of_a_ray_through_a_disc() {
        let s = Segment { x0: 0.0, y0: 0.0, x1: 10.0, y1: 0.0, pad: 0.0 };
        assert_eq!(entry_time(&s, 10.0, 0.0, 100.0, 5.0, 0.0, 1.0), Some(0.4));
        assert_eq!(entry_time(&s, 10.0, 0.0, 100.0, 5.0, 2.0, 1.0), None);
        assert_eq!(entry_time(&s, 10.0, 0.0, 100.0, 0.5, 0.0, 1.0), Some(0.0), "starts inside");
        assert_eq!(entry_time(&s, 10.0, 0.0, 100.0, 12.0, 0.0, 1.0), None, "beyond the end");
        let mut seed = 9u64;
        for _ in 0..1000 {
            let (px, py, r) = (rng(&mut seed) * 12.0 - 1.0, rng(&mut seed) * 4.0 - 2.0, rng(&mut seed) * 2.0);
            let hit = entry_time(&s, 10.0, 0.0, 100.0, px, py, r);
            let d2 = (px.clamp(0.0, 10.0) - px).powi(2) + py * py;
            if (d2 - r * r).abs() < 1e-3 { continue; } // tangent: either answer is fine
            assert_eq!(hit.is_some(), d2 < r * r, "px {px} py {py} r {r}");
        }
    }
}
