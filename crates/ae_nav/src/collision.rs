//! Basic circle collision response with a deterministic sorted-grid broadphase.
//! This is deliberately separate from navigation: it resolves local overlaps
//! and bounce, but makes no claim to solve congestion or deadlock globally.
use crate::math::WorldPos;
use crate::par::Par;
use crate::terrain::{Medium, Terrain};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub pos: WorldPos,
    pub vel: WorldPos,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ContactView {
    pub a: u32,
    pub b: u32,
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CollisionStats {
    pub broadphase_candidates: u32,
    pub contacts: u32,
    pub solver_iterations: u32,
    pub substeps: u32,
    /// Wall time of the whole collision step (integration, wall push-outs,
    /// broadphase and pair solve, all substeps).
    pub solver_ms: f64,
    pub obstacle_contacts: u32,
    /// Parts of `solver_ms`: integration and wall push-outs, the pair
    /// search, and the pair solve.
    #[serde(default)]
    pub integrate_ms: f64,
    #[serde(default)]
    pub broadphase_ms: f64,
    #[serde(default)]
    pub pair_solve_ms: f64,
    /// Time spent in velocity-obstacle avoidance this tick (steering).
    #[serde(default)]
    pub avoidance_ms: f64,
}

pub struct CollisionConfig {
    pub solver_iterations: u32,
    pub restitution: f32,
    pub correction: f32,
}

impl Default for CollisionConfig {
    fn default() -> Self {
        Self {
            solver_iterations: 3,
            restitution: 0.08,
            correction: 0.82,
        }
    }
}

/// Sorted-grid broadphase. Bodies are sorted by (bucket row, bucket
/// column, index) with buckets as wide as the largest pair distance, and
/// copied in that order, so the search reads memory sequentially and a
/// bucket row is a contiguous range. Each body then scans only forward:
/// the rest of its own and the next bucket in its row, and the three
/// buckets below it, so every pair is visited once. Memory scales with the
/// number of bodies, not the size of the map. Buffers persist between
/// calls.
#[derive(Default)]
struct Broadphase {
    /// (row, column, body index), sorted.
    keys: Vec<(u32, u32, u32)>,
    /// Bodies in `keys` order: x, y, radius.
    pts: Vec<[f32; 3]>,
    /// Start of each bucket row in `keys`, from row `min_row`.
    row_start: Vec<u32>,
    /// Start of each body's pairs, for the counting sort of pairs.
    pair_start: Vec<u32>,
    /// The result. Reused: a fresh buffer of this size is mapped anew by
    /// the allocator each time, and its page faults cost more than the
    /// ordering itself.
    pairs: Vec<(u32, u32)>,
}

impl Broadphase {
    /// Overlapping pairs `(a, b)` with `ids[a] < ids[b]`, sorted by IDs, and
    /// the candidate count. The pair set does not depend on bucket order or
    /// on how the search is split over threads.
    fn pairs(&mut self, ids: &[u32], bodies: &[Body], world_w: f32, world_h: f32, par: &Par) -> (&[(u32, u32)], u32) {
        if bodies.is_empty() {
            return (&[], 0);
        }
        let cell = bodies.iter().map(|b| b.radius * 2.0 + 0.001).fold(0.5f32, f32::max);
        let nx = (world_w / cell).ceil().max(1.0);
        let ny = (world_h / cell).ceil().max(1.0);
        let bucket = |p: WorldPos| {
            (
                (p.y / cell).floor().clamp(0.0, ny - 1.0) as u32,
                (p.x / cell).floor().clamp(0.0, nx - 1.0) as u32,
            )
        };
        // The previous order is nearly sorted still: bodies move less than
        // a bucket per substep.
        if self.keys.len() != bodies.len() {
            self.keys.clear();
            self.keys.extend((0..bodies.len() as u32).map(|i| (0, 0, i)));
        }
        par.chunks_mut(&mut self.keys, |chunk| {
            for k in chunk {
                (k.0, k.1) = bucket(bodies[k.2 as usize].pos);
            }
        });
        par.sort_unstable(&mut self.keys);
        self.pts.clear();
        self.pts.extend(self.keys.iter().map(|k| {
            let b = &bodies[k.2 as usize];
            [b.pos.x, b.pos.y, b.radius]
        }));
        let min_row = self.keys[0].0;
        let rows = (self.keys[self.keys.len() - 1].0 - min_row + 2) as usize;
        self.row_start.clear();
        self.row_start.resize(rows + 1, 0);
        for k in &self.keys {
            self.row_start[(k.0 - min_row + 1) as usize] += 1;
        }
        for r in 1..self.row_start.len() {
            self.row_start[r] += self.row_start[r - 1];
        }
        let (keys, pts, row_start) = (&self.keys, &self.pts, &self.row_start);
        let search = |lo: usize, hi: usize| -> (Vec<(u32, u32)>, u32) {
            let mut pairs = Vec::new();
            let mut candidates = 0u32;
            let mut test = |k: usize, m: usize, pairs: &mut Vec<(u32, u32)>| {
                candidates += 1;
                let (a, b) = (pts[k], pts[m]);
                let rr = a[2] + b[2];
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                if dx * dx + dy * dy <= rr * rr {
                    let (i, j) = (keys[k].2, keys[m].2);
                    // Sort by stable IDs, not storage order.
                    let (i, j) = if ids[i as usize] < ids[j as usize] {
                        (i, j)
                    } else {
                        (j, i)
                    };
                    pairs.push((i, j));
                }
            };
            for k in lo..hi {
                let (row, col, _) = keys[k];
                // Own row: the rest of this bucket and the next one.
                let row_end = row_start[(row - min_row + 1) as usize] as usize;
                let mut m = k + 1;
                while m < row_end && keys[m].1 <= col + 1 {
                    test(k, m, &mut pairs);
                    m += 1;
                }
                // Next row: columns col - 1 ..= col + 1.
                let r = (row - min_row + 1) as usize;
                if r + 1 < row_start.len() {
                    let below = &keys[row_start[r] as usize..row_start[r + 1] as usize];
                    let first = below.partition_point(|q| q.1 + 1 < col);
                    let base = row_start[r] as usize;
                    for (off, q) in below[first..].iter().enumerate() {
                        if q.1 > col + 1 {
                            break;
                        }
                        test(k, base + first + off, &mut pairs);
                    }
                }
            }
            (pairs, candidates)
        };
        let parts = par.ranges(keys.len(), search);
        let candidates = parts.iter().map(|p| p.1).sum();
        let found = || parts.iter().flat_map(|p| &p.0);
        self.pairs.clear();
        if ids.windows(2).all(|w| w[0] < w[1]) {
            // IDs ascend with storage index (always so in the simulation):
            // a counting sort by first body, then by second within each.
            self.pairs.resize(found().count(), (0, 0));
            let start = &mut self.pair_start;
            start.clear();
            start.resize(bodies.len(), 0);
            for p in found() {
                start[p.0 as usize] += 1;
            }
            let mut sum = 0;
            for c in start.iter_mut() {
                let n = *c;
                *c = sum;
                sum += n;
            }
            for &(i, j) in found() {
                self.pairs[start[i as usize] as usize] = (i, j);
                start[i as usize] += 1;
            }
            // Each `start[i]` is now the end of body i's pairs.
            let mut from = 0;
            for &end in start.iter() {
                if end - from > 1 {
                    self.pairs[from as usize..end as usize].sort_unstable_by_key(|p| p.1);
                }
                from = end;
            }
        } else {
            let mut keyed: Vec<(u32, u32, u32, u32)> = found()
                .map(|&(i, j)| (ids[i as usize], ids[j as usize], i, j))
                .collect();
            par.sort_unstable(&mut keyed);
            self.pairs.extend(keyed.into_iter().map(|k| (k.2, k.3)));
        }
        (&self.pairs, candidates)
    }
}

/// Per-body movement media: which cells are walls for which body. Body `i`
/// collides with every cell not open to `media[class_of[i]]`.
#[derive(Clone, Copy)]
pub struct BodyMedia<'a> {
    pub media: &'a [Medium],
    pub class_of: &'a [u8],
}

impl BodyMedia<'_> {
    fn of(&self, i: usize) -> &Medium {
        &self.media[self.class_of[i] as usize]
    }
}

fn resolve_obstacles(b: &mut Body, terrain: &Terrain, medium: &Medium, restitution: f32) -> u32 {
    let mut contacts = 0;
    let g = &terrain.geom;
    let cs = g.cell_size;
    {
        // World bounds are solid.
        for _ in 0..2 {
            if b.pos.x < b.radius {
                b.pos.x = b.radius;
                if b.vel.x < 0.0 {
                    b.vel.x = -b.vel.x * restitution;
                }
            }
            if b.pos.y < b.radius {
                b.pos.y = b.radius;
                if b.vel.y < 0.0 {
                    b.vel.y = -b.vel.y * restitution;
                }
            }
            if b.pos.x > g.world_width() - b.radius {
                b.pos.x = g.world_width() - b.radius;
                if b.vel.x > 0.0 {
                    b.vel.x = -b.vel.x * restitution;
                }
            }
            if b.pos.y > g.world_height() - b.radius {
                b.pos.y = g.world_height() - b.radius;
                if b.vel.y > 0.0 {
                    b.vel.y = -b.vel.y * restitution;
                }
            }
            let min_x = ((b.pos.x - b.radius) / cs).floor() as i32;
            let max_x = ((b.pos.x + b.radius) / cs).floor() as i32;
            let min_y = ((b.pos.y - b.radius) / cs).floor() as i32;
            let max_y = ((b.pos.y + b.radius) / cs).floor() as i32;
            for cy in min_y..=max_y {
                for cx in min_x..=max_x {
                    if !g.in_bounds(crate::math::CellPos::new(cx, cy)) {
                        continue;
                    }
                    if terrain.open_for((cy * g.cells_x + cx) as usize, medium) {
                        continue;
                    }
                    let rect_min_x = cx as f32 * cs;
                    let rect_min_y = cy as f32 * cs;
                    let rect_max_x = rect_min_x + cs;
                    let rect_max_y = rect_min_y + cs;
                    let nearest_x = b.pos.x.clamp(rect_min_x, rect_max_x);
                    let nearest_y = b.pos.y.clamp(rect_min_y, rect_max_y);
                    let mut dx = b.pos.x - nearest_x;
                    let mut dy = b.pos.y - nearest_y;
                    let mut d2 = dx * dx + dy * dy;
                    if d2 >= b.radius * b.radius {
                        continue;
                    }
                    contacts += 1;
                    if d2 <= 1e-12 {
                        // Center is inside the cell. Push out along the shortest
                        // axis deterministically.
                        let left = b.pos.x - rect_min_x;
                        let right = rect_max_x - b.pos.x;
                        let top = b.pos.y - rect_min_y;
                        let bottom = rect_max_y - b.pos.y;
                        let m = left.min(right).min(top).min(bottom);
                        if m == left {
                            dx = -1.0;
                            dy = 0.0;
                        } else if m == right {
                            dx = 1.0;
                            dy = 0.0;
                        } else if m == top {
                            dx = 0.0;
                            dy = -1.0;
                        } else {
                            dx = 0.0;
                            dy = 1.0;
                        }
                        d2 = 1.0;
                    } else {
                        let d = d2.sqrt();
                        dx /= d;
                        dy /= d;
                    }
                    let d = d2.sqrt();
                    let push = b.radius - d + 0.01;
                    b.pos.x += dx * push;
                    b.pos.y += dy * push;
                    let vn = b.vel.x * dx + b.vel.y * dy;
                    if vn < 0.0 {
                        b.vel.x -= (1.0 + restitution) * vn * dx;
                        b.vel.y -= (1.0 + restitution) * vn * dy;
                    }
                }
            }
        }
    }
    contacts
}

/// Collision response for one fixed tick, split into `substeps`. Keeps its
/// broadphase buffers between calls. Per substep:
/// 1. integrate positions;
/// 2. push bodies out of obstacles;
/// 3. find overlapping pairs and solve them in stable ID order;
/// 4. push bodies out of obstacles again, so separation never leaves a body
///    inside a wall.
///
/// Steps 1, 2 and 4 touch one body at a time and the pair search only
/// reads, so they run on the thread pool; the pair solve is sequential.
/// The result is the same for any thread count.
#[derive(Default)]
pub struct Collider {
    broadphase: Broadphase,
}

impl Collider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every body collides with obstacle cells only (pathsim's behaviour).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        ids: &[u32],
        bodies: &mut [Body],
        terrain: &Terrain,
        cfg: &CollisionConfig,
        dt: f32,
        substeps: u32,
        par: &Par,
        contacts_out: &mut Vec<ContactView>,
        contact_limit: usize,
    ) -> CollisionStats {
        self.step_in_media(
            ids,
            bodies,
            terrain,
            None,
            cfg,
            dt,
            substeps,
            par,
            contacts_out,
            contact_limit,
        )
    }

    /// As [`Self::step`]; with `media`, each body also collides with the
    /// cells its medium may not enter (a ship with the shore, a tank with
    /// deep water).
    #[allow(clippy::too_many_arguments)]
    pub fn step_in_media(
        &mut self,
        ids: &[u32],
        bodies: &mut [Body],
        terrain: &Terrain,
        media: Option<BodyMedia<'_>>,
        cfg: &CollisionConfig,
        dt: f32,
        substeps: u32,
        par: &Par,
        contacts_out: &mut Vec<ContactView>,
        contact_limit: usize,
    ) -> CollisionStats {
        if let Some(m) = &media {
            assert_eq!(m.class_of.len(), bodies.len());
        }
        let medium = |i: usize| media.as_ref().map_or(&Medium::ANY, |m| m.of(i));
        let start = std::time::Instant::now();
        let mut stats = CollisionStats {
            substeps,
            ..CollisionStats::default()
        };
        let r = cfg.restitution;
        for s in 0..substeps {
            // The previous substep's closing obstacle pass, this substep's
            // integration and opening obstacle pass, in one sweep per body.
            let closing = s > 0;
            let t = std::time::Instant::now();
            let obstacle: u32 = par
                .chunks_mut_at(bodies, |start, chunk| {
                    let mut n = 0;
                    for (k, b) in chunk.iter_mut().enumerate() {
                        let m = medium(start + k);
                        if closing {
                            n += resolve_obstacles(b, terrain, m, r);
                        }
                        b.pos = b.pos.add(b.vel.scaled(dt));
                        n += resolve_obstacles(b, terrain, m, r);
                    }
                    n
                })
                .into_iter()
                .sum();
            stats.obstacle_contacts += obstacle;
            let t = lap(&mut stats.integrate_ms, t);
            let (pairs, candidates) = self.broadphase.pairs(
                ids,
                bodies,
                terrain.geom.world_width(),
                terrain.geom.world_height(),
                par,
            );
            stats.broadphase_candidates += candidates;
            let t = lap(&mut stats.broadphase_ms, t);
            for _iter in 0..cfg.solver_iterations.max(1) {
                stats.solver_iterations += 1;
                stats.contacts += solve_pairs(ids, bodies, pairs, cfg, contacts_out, contact_limit);
            }
            lap(&mut stats.pair_solve_ms, t);
        }
        if substeps > 0 {
            let t = std::time::Instant::now();
            stats.obstacle_contacts += par
                .chunks_mut_at(bodies, |start, chunk| {
                    chunk
                        .iter_mut()
                        .enumerate()
                        .map(|(k, b)| resolve_obstacles(b, terrain, medium(start + k), r))
                        .sum::<u32>()
                })
                .into_iter()
                .sum::<u32>();
            lap(&mut stats.integrate_ms, t);
        }
        stats.solver_ms = start.elapsed().as_secs_f64() * 1000.0;
        stats
    }
}

/// Adds the time since `since` to `acc` (ms) and returns now.
fn lap(acc: &mut f64, since: std::time::Instant) -> std::time::Instant {
    let now = std::time::Instant::now();
    *acc += (now - since).as_secs_f64() * 1000.0;
    now
}

/// Overlap, as a fraction of the radii sum, that counts as touching. Large
/// enough that the correction of any overlap above it (a fraction of the
/// overlap) is several float steps even 20,000 world units from the origin;
/// below that, rounding would leave pairs creeping by one step a tick.
const OVERLAP_SLOP: f32 = 0.01;

/// One sequential pass over the pairs (Gauss-Seidel, in stable ID order).
/// Returns the number of overlapping pairs corrected.
fn solve_pairs(
    ids: &[u32],
    bodies: &mut [Body],
    pairs: &[(u32, u32)],
    cfg: &CollisionConfig,
    contacts_out: &mut Vec<ContactView>,
    contact_limit: usize,
) -> u32 {
    let mut contacts = 0;
    for &(ia, ib) in pairs {
        let (ia, ib) = (ia as usize, ib as usize);
        let (a, b) = {
            if ia < ib {
                let (left, right) = bodies.split_at_mut(ib);
                (&mut left[ia], &mut right[0])
            } else {
                let (left, right) = bodies.split_at_mut(ia);
                (&mut left[ib], &mut right[0])
            }
        };
        let mut dx = b.pos.x - a.pos.x;
        let mut dy = b.pos.y - a.pos.y;
        let rsum = a.radius + b.radius;
        let d2 = dx * dx + dy * dy;
        if d2 >= rsum * rsum {
            continue;
        }
        let mut d = d2.sqrt();
        if d <= 1e-6 {
            // Deterministic fallback normal based on stable IDs.
            let sign = if ids[ia] < ids[ib] { 1.0 } else { -1.0 };
            dx = sign;
            dy = 0.0;
            d = 1.0;
        } else {
            dx /= d;
            dy /= d;
        }
        // Each pass removes only part of an overlap, so without a floor a
        // pair at rest would shrink its overlap forever, nudging positions
        // by float rounding every tick.
        let overlap = rsum - d;
        if overlap <= rsum * OVERLAP_SLOP {
            continue;
        }
        contacts += 1;
        let correction = overlap * cfg.correction * 0.5;
        a.pos.x -= dx * correction;
        a.pos.y -= dy * correction;
        b.pos.x += dx * correction;
        b.pos.y += dy * correction;
        let rvx = b.vel.x - a.vel.x;
        let rvy = b.vel.y - a.vel.y;
        let vn = rvx * dx + rvy * dy;
        if vn < 0.0 {
            let impulse = -(1.0 + cfg.restitution) * vn * 0.5;
            a.vel.x -= impulse * dx;
            a.vel.y -= impulse * dy;
            b.vel.x += impulse * dx;
            b.vel.y += impulse * dy;
        }
        if contacts_out.len() < contact_limit {
            let px = (a.pos.x + b.pos.x) * 0.5;
            let py = (a.pos.y + b.pos.y) * 0.5;
            contacts_out.push(ContactView {
                a: ids[ia],
                b: ids[ib],
                x: px,
                y: py,
            });
        }
    }
    contacts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// The sorted-grid search finds exactly the overlapping pairs, in ID
    /// order, for ascending and shuffled IDs and any thread count, across
    /// repeated calls (which reuse the previous bucket order).
    #[test]
    fn broadphase_matches_brute_force() {
        let mut rng = Rng::new(7);
        let mut bodies: Vec<Body> = (0..3000)
            .map(|_| Body {
                pos: WorldPos::new(rng.range(0.0, 400.0), rng.range(0.0, 300.0)),
                vel: WorldPos::ZERO,
                radius: rng.range(1.5, 4.5),
            })
            .collect();
        let ascending: Vec<u32> = (1..=bodies.len() as u32).collect();
        let mut shuffled = ascending.clone();
        for i in (1..shuffled.len()).rev() {
            shuffled.swap(i, rng.next_u32() as usize % (i + 1));
        }
        for ids in [&ascending, &shuffled] {
            for threads in [1, 4] {
                let par = Par::new(threads);
                let mut bp = Broadphase::default();
                for round in 0..3 {
                    if round > 0 {
                        for b in &mut bodies {
                            b.pos = WorldPos::new(
                                (b.pos.x + rng.range(-3.0, 3.0)).clamp(0.0, 400.0),
                                (b.pos.y + rng.range(-3.0, 3.0)).clamp(0.0, 300.0),
                            );
                        }
                    }
                    let mut expected = Vec::new();
                    for a in 0..bodies.len() {
                        for b in a + 1..bodies.len() {
                            let rr = bodies[a].radius + bodies[b].radius;
                            let (dx, dy) = (bodies[a].pos.x - bodies[b].pos.x, bodies[a].pos.y - bodies[b].pos.y);
                            if dx * dx + dy * dy <= rr * rr {
                                expected.push(if ids[a] < ids[b] {
                                    (ids[a], ids[b])
                                } else {
                                    (ids[b], ids[a])
                                });
                            }
                        }
                    }
                    expected.sort_unstable();
                    let (pairs, _) = bp.pairs(ids, &bodies, 400.0, 300.0, &par);
                    let got: Vec<(u32, u32)> = pairs.iter().map(|&(a, b)| (ids[a as usize], ids[b as usize])).collect();
                    assert!(!expected.is_empty());
                    assert_eq!(got, expected, "threads {threads}, round {round}");
                }
            }
        }
    }
}
