//! Standing-crowd density for crowd-aware routing.
//!
//! Units standing still (idle at rest, unreachable, arrived a few seconds
//! ago, or queued: on their way but without progress for 1.5 s) are summed into a grid of `BLOCK_CELLS`² blocks as the
//! fraction of the block their bodies cover, and quantised to levels
//! 0..=`LEVELS` with hysteresis. From the levels, every edge of the portal
//! graph gets an extra cost: up to `PENALTY_FACTOR` times its own cost when
//! its straight line runs through blocks at the top level. A context built
//! for a large command (`ContextKey::crowd`) adds those extras, so its route
//! goes around a dense standing crowd when the detour is less than about
//! `1 + PENALTY_FACTOR` times the crowded stretch, and through it otherwise
//! (a crowd filling a chokepoint, a goal inside the crowd).
//!
//! Traffic (`Traffic`) adds how slowly units on their way advance through
//! each block: a block where they move at a tenth of their speed costs up
//! to 10× to cross, about the time it takes, in route edges and in fields,
//! so the tail of a queue at a narrow gap leaves for another gap when that
//! is quicker. Density alone missed queues that creep: units advancing a
//! little every second never counted as standing.
//!
//! The map only changes, and its `version` only bumps, when some block's
//! level changes; it is rebuilt every `UPDATE_TICKS` from simulation state,
//! so it is deterministic.
use crate::math::{CellPos, WorldGeometry, WorldPos};
use crate::nav::hierarchy::NavHierarchy;

pub const BLOCK_CELLS: i32 = 8;
pub const LEVELS: u8 = 3;
/// Covered fraction of a block at which levels 1, 2 and 3 start (measured,
/// mixed profiles): a crowd packed body to body (1.15 cells apart) covers
/// about 0.3 of its area, an army standing on its arrival slots about
/// 0.085 (0.2 in its core), a crowd 2.5 cells apart about 0.06. A large
/// command goes around all of them, most readily around packed ones.
const THRESHOLDS: [f32; LEVELS as usize] = [0.04, 0.07, 0.11];
/// A block's level changes only once the covered fraction is this far past
/// the threshold.
const HYSTERESIS: f32 = 0.01;
/// A new map is published only when at least this many blocks differ from
/// the published one (a quarter of a 64-cell sector): movers deforming a
/// crowd's edge, or a crowd shuffling in place, never re-route anyone,
/// while a crowd forming, leaving or drifting far does.
const MIN_CHANGED_BLOCKS: usize = 16;
/// Route costs and look-ahead see each block at the highest level within
/// this many blocks (16 cells). Sector-boundary crossing points otherwise
/// sat one cell outside a crowd's edge, and a stream tens of cells wide
/// following them walked half inside it.
const MARGIN_BLOCKS: i32 = 2;
/// Extra edge cost at the top level, as a multiple of the edge's cost.
pub const PENALTY_FACTOR: u64 = 2;
pub const UPDATE_TICKS: u64 = 30;
/// Commands with at least this many units route around standing crowds;
/// smaller ones go straight and slip through.
pub const CROWD_AWARE_MIN_UNITS: u32 = 64;
/// Traffic: a block counts once moving units cover this fraction of it.
const TRAFFIC_MIN_COVER: f32 = 0.04;
/// Weight of each new speed sample in a block's smoothed speed (about
/// 3 s of memory at one sample a second).
const TRAFFIC_SMOOTHING: f32 = 0.35;
/// Smoothed speed (fraction of top speed) below which traffic levels 1..=4
/// start, and the extra cost of crossing a block at each level as a
/// multiple of its length: about the extra travel time at that speed.
const SLOW_THRESHOLDS: [f32; 4] = [0.5, 0.33, 0.2, 0.1];
const SLOW_HYSTERESIS: f32 = 0.03;
const SLOW_EXTRA: [u64; 5] = [0, 1, 2, 4, 9];
/// A traffic change alone publishes a new map once this many blocks
/// changed level: a queue at a narrow gap is only a few blocks.
const MIN_CHANGED_SLOW_BLOCKS: usize = 4;

#[derive(Clone, Debug, Default)]
pub struct CrowdMap {
    /// Identity of this map's levels: 0 while no block has ever been dense;
    /// every new map gets a fresh number from the simulation (unique across
    /// the global and per-command maps), so contexts keyed by it never mix
    /// two maps.
    pub version: u32,
    pub blocks_x: i32,
    pub blocks_y: i32,
    pub levels: Vec<u8>,
    /// `levels` grown by `MARGIN_BLOCKS`: what routes and look-ahead see.
    pub route_levels: Vec<u8>,
    /// Traffic level of each block (`Traffic::slow`): how slowly the
    /// units moving through it advance.
    pub slow: Vec<u8>,
    /// Navigation version the edge extras belong to.
    pub nav_version: u32,
    /// Extra cost of each incoming edge, per profile and node, in
    /// `ProfileHierarchy::in_edges` order. Empty when no block is dense.
    extra: Vec<Vec<Vec<u32>>>,
}

impl CrowdMap {
    pub fn empty(geom: &WorldGeometry) -> Self {
        let (bx, by) = (
            (geom.cells_x + BLOCK_CELLS - 1) / BLOCK_CELLS,
            (geom.cells_y + BLOCK_CELLS - 1) / BLOCK_CELLS,
        );
        let n = (bx * by) as usize;
        Self {
            version: 0,
            blocks_x: bx,
            blocks_y: by,
            levels: vec![0; n],
            route_levels: vec![0; n],
            slow: vec![0; n],
            nav_version: 0,
            extra: Vec::new(),
        }
    }

    /// Level of the block at `c` as routes see it (with the margin).
    pub fn level_at(&self, c: CellPos) -> u8 {
        self.block(&self.route_levels, c)
    }

    /// Level of the block at `c` without the margin. Look-ahead lines use
    /// it: a unit at a crowd's edge stands in its margin, and judged by
    /// that level any line through the crowd was acceptable.
    pub fn raw_level_at(&self, c: CellPos) -> u8 {
        self.block(&self.levels, c)
    }

    /// Extra cost (`block_extra`) of the blocks of sector `s` (row-major)
    /// from raw levels (no margin) and traffic, or empty when all are 0: a
    /// field's crowd costs (`FieldKey::crowd`). With the margin, the 16
    /// cells around a crowd cost as much as the crowd, so a unit at its
    /// edge had no cheaper way out than through it. With traffic, units
    /// behind a queue in the same sector as its gap leave by another exit
    /// when that is quicker, while the queue's front goes through.
    pub fn sector_levels(&self, geom: &WorldGeometry, s: crate::math::SectorPos) -> Vec<u8> {
        let (x0, y0, x1, y1) = geom.sector_bounds(s);
        let mut out = Vec::new();
        for by in (y0..y1).step_by(BLOCK_CELLS as usize) {
            for bx in (x0..x1).step_by(BLOCK_CELLS as usize) {
                let c = CellPos::new(bx, by);
                out.push(block_extra(self.raw_level_at(c), self.block(&self.slow, c)) as u8);
            }
        }
        if out.iter().all(|&l| l == 0) {
            out.clear();
        }
        out
    }

    fn block(&self, levels: &[u8], c: CellPos) -> u8 {
        let (x, y) = (c.x / BLOCK_CELLS, c.y / BLOCK_CELLS);
        if x < 0 || y < 0 || x >= self.blocks_x || y >= self.blocks_y {
            return 0;
        }
        levels[(y * self.blocks_x + x) as usize]
    }

    /// Extra cost of `ph.in_edges[node][k]` for profile `pid` (0 when the
    /// extras belong to another navigation version).
    pub fn edge_extra(&self, nav_version: u32, pid: u8, node: u32, k: usize) -> u32 {
        if nav_version != self.nav_version {
            return 0;
        }
        self.extra
            .get(pid as usize)
            .and_then(|p| p.get(node as usize))
            .and_then(|e| e.get(k))
            .copied()
            .unwrap_or(0)
    }

    /// The map for standing bodies `(position, radius)` and traffic levels
    /// `slow` as `version`, or `None` when fewer than `MIN_CHANGED_BLOCKS`
    /// blocks changed level and fewer than `MIN_CHANGED_SLOW_BLOCKS`
    /// changed traffic level.
    pub fn update(
        &self,
        h: &NavHierarchy,
        bodies: impl Iterator<Item = (WorldPos, f32)>,
        slow: &[u8],
        version: u32,
    ) -> Option<CrowdMap> {
        let geom = &h.terrain.geom;
        let mut area = vec![0.0f32; self.levels.len()];
        for (p, r) in bodies {
            let c = geom.world_to_cell(p);
            let (x, y) = (c.x / BLOCK_CELLS, c.y / BLOCK_CELLS);
            if x < 0 || y < 0 || x >= self.blocks_x || y >= self.blocks_y {
                continue;
            }
            area[(y * self.blocks_x + x) as usize] += std::f32::consts::PI * r * r;
        }
        let block_area = (BLOCK_CELLS as f32 * geom.cell_size).powi(2);
        let mut changed = 0;
        let levels: Vec<u8> = area
            .iter()
            .zip(&self.levels)
            .map(|(&a, &old)| {
                let new = level_with_hysteresis(a / block_area, old);
                changed += (new != old) as usize;
                new
            })
            .collect();
        let changed_slow = slow.iter().zip(&self.slow).filter(|(a, b)| a != b).count();
        if changed < MIN_CHANGED_BLOCKS && changed_slow < MIN_CHANGED_SLOW_BLOCKS {
            return None;
        }
        let mut next = CrowdMap {
            version,
            levels,
            slow: slow.to_vec(),
            ..self.clone()
        };
        next.route_levels = grow(&next.levels, next.blocks_x, next.blocks_y, MARGIN_BLOCKS);
        next.rebuild_extras(h);
        Some(next)
    }

    /// The same levels and version with edge extras for hierarchy `h` (after
    /// a terrain repair).
    pub fn for_hierarchy(&self, h: &NavHierarchy) -> CrowdMap {
        let mut next = self.clone();
        next.rebuild_extras(h);
        next
    }

    fn rebuild_extras(&mut self, h: &NavHierarchy) {
        self.nav_version = h.version;
        self.extra.clear();
        if self.levels.iter().all(|&l| l == 0) && self.slow.iter().all(|&l| l == 0) {
            return;
        }
        let geom = &h.terrain.geom;
        // Sectors with any dense block: only edges touching them can cost
        // extra.
        let mut dense = vec![false; (geom.sectors_x() * geom.sectors_y()) as usize];
        for (i, (&l, &sl)) in self.route_levels.iter().zip(&self.slow).enumerate() {
            if l == 0 && sl == 0 {
                continue;
            }
            let c = CellPos::new(
                (i as i32 % self.blocks_x) * BLOCK_CELLS,
                (i as i32 / self.blocks_x) * BLOCK_CELLS,
            );
            dense[geom.sector_index(geom.sector_of(c))] = true;
        }
        for ph in &h.hierarchies {
            let per_node: Vec<Vec<u32>> = ph
                .in_edges
                .iter()
                .enumerate()
                .map(|(node, edges)| {
                    let to = &ph.nodes[node];
                    edges
                        .iter()
                        .map(|&(from, c)| {
                            let from = &ph.nodes[from as usize];
                            if !dense[from.sector as usize] && !dense[to.sector as usize] {
                                return 0;
                            }
                            self.line_extra(from.cell, to.cell, c)
                        })
                        .collect()
                })
                .collect();
            self.extra.push(per_node);
        }
    }

    /// `cost` times the mean extra along the straight line a → b (sampled
    /// every 4 cells): at each sample the larger of the crowd's
    /// (`PENALTY_FACTOR` × level / `LEVELS`) and the traffic's
    /// (`SLOW_EXTRA`).
    fn line_extra(&self, a: CellPos, b: CellPos, cost: u32) -> u32 {
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let m = (dx.abs().max(dy.abs()) / 4).max(1);
        let sum: u64 = (0..=m)
            .map(|k| {
                let c = CellPos::new(a.x + dx * k / m, a.y + dy * k / m);
                block_extra(self.level_at(c), self.block(&self.slow, c))
            })
            .sum();
        (cost as u64 * sum / (LEVELS as u64 * (m as u64 + 1))) as u32
    }
}

/// How fast moving units advance through each block: their speed toward
/// their steering waypoints as a fraction of top speed, smoothed over a
/// few samples, and quantised to traffic levels 0..=4 with hysteresis.
/// Blocks with too few movers count as free.
#[derive(Clone, Debug, Default)]
pub struct Traffic {
    speed: Vec<f32>,
    pub slow: Vec<u8>,
}

impl Traffic {
    pub fn new(map: &CrowdMap) -> Self {
        let n = map.levels.len();
        Self {
            speed: vec![1.0; n],
            slow: vec![0; n],
        }
    }

    /// One sample of movers `(position, radius, speed fraction)`.
    pub fn update(&mut self, map: &CrowdMap, geom: &WorldGeometry, movers: impl Iterator<Item = (WorldPos, f32, f32)>) {
        let n = self.speed.len();
        let (mut area, mut sum, mut count) = (vec![0.0f32; n], vec![0.0f32; n], vec![0u32; n]);
        for (p, r, f) in movers {
            let c = geom.world_to_cell(p);
            let (x, y) = (c.x / BLOCK_CELLS, c.y / BLOCK_CELLS);
            if x < 0 || y < 0 || x >= map.blocks_x || y >= map.blocks_y {
                continue;
            }
            let i = (y * map.blocks_x + x) as usize;
            area[i] += std::f32::consts::PI * r * r;
            sum[i] += f;
            count[i] += 1;
        }
        let block_area = (BLOCK_CELLS as f32 * geom.cell_size).powi(2);
        for i in 0..n {
            let sample = if area[i] / block_area >= TRAFFIC_MIN_COVER {
                sum[i] / count[i] as f32
            } else {
                1.0
            };
            self.speed[i] += TRAFFIC_SMOOTHING * (sample - self.speed[i]);
            self.slow[i] = slow_with_hysteresis(self.speed[i], self.slow[i]);
        }
    }
}

/// Extra cost of crossing a block, in `1 / LEVELS` of its own cost: the
/// larger of the crowd's (`PENALTY_FACTOR` × level) and the traffic's
/// (`SLOW_EXTRA`).
pub fn block_extra(level: u8, slow: u8) -> u64 {
    (PENALTY_FACTOR * level as u64).max(LEVELS as u64 * SLOW_EXTRA[slow as usize])
}

fn slow_with_hysteresis(speed: f32, old: u8) -> u8 {
    let mut level = old;
    while (level as usize) < SLOW_THRESHOLDS.len() && speed < SLOW_THRESHOLDS[level as usize] - SLOW_HYSTERESIS {
        level += 1;
    }
    while level > 0 && speed >= SLOW_THRESHOLDS[level as usize - 1] + SLOW_HYSTERESIS {
        level -= 1;
    }
    level
}

/// Each block's level becomes the highest within `r` blocks (separable
/// max: rows, then columns).
fn grow(levels: &[u8], w: i32, h: i32, r: i32) -> Vec<u8> {
    let at = |v: &[u8], x: i32, y: i32| v[(y * w + x) as usize];
    let rows: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| {
            ((x - r).max(0)..=(x + r).min(w - 1))
                .map(|xx| at(levels, xx, y))
                .max()
                .unwrap_or(0)
        })
        .collect();
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| {
            ((y - r).max(0)..=(y + r).min(h - 1))
                .map(|yy| at(&rows, x, yy))
                .max()
                .unwrap_or(0)
        })
        .collect()
}

fn level_with_hysteresis(covered: f32, old: u8) -> u8 {
    let mut level = old;
    while (level as usize) < THRESHOLDS.len() && covered >= THRESHOLDS[level as usize] + HYSTERESIS {
        level += 1;
    }
    while level > 0 && covered < THRESHOLDS[level as usize - 1] - HYSTERESIS {
        level -= 1;
    }
    level
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_grows_by_two_blocks() {
        let mut v = vec![0u8; 7 * 7];
        v[3 * 7 + 3] = 2;
        let g = grow(&v, 7, 7, 2);
        for y in 0..7 {
            for x in 0..7 {
                let inside = (x as i32 - 3).abs() <= 2 && (y as i32 - 3).abs() <= 2;
                assert_eq!(g[y * 7 + x], if inside { 2 } else { 0 }, "({x}, {y})");
            }
        }
    }

    #[test]
    fn levels_hold_near_thresholds() {
        assert_eq!(level_with_hysteresis(0.0, 0), 0);
        assert_eq!(level_with_hysteresis(0.045, 0), 0);
        assert_eq!(level_with_hysteresis(0.055, 0), 1);
        assert_eq!(level_with_hysteresis(0.085, 0), 2);
        assert_eq!(level_with_hysteresis(0.6, 0), 3);
        assert_eq!(level_with_hysteresis(0.105, 3), 3);
        assert_eq!(level_with_hysteresis(0.095, 3), 2);
        assert_eq!(level_with_hysteresis(0.0, 3), 0);
    }

    #[test]
    fn traffic_levels_hold_near_thresholds() {
        assert_eq!(slow_with_hysteresis(1.0, 0), 0);
        assert_eq!(slow_with_hysteresis(0.48, 0), 0);
        assert_eq!(slow_with_hysteresis(0.45, 0), 1);
        assert_eq!(slow_with_hysteresis(0.05, 0), 4);
        assert_eq!(slow_with_hysteresis(0.52, 1), 1);
        assert_eq!(slow_with_hysteresis(0.55, 1), 0);
        assert_eq!(slow_with_hysteresis(1.0, 4), 0);
    }

    #[test]
    fn a_slow_queue_raises_edge_costs_more_than_a_crowd() {
        assert_eq!(block_extra(0, 0), 0);
        assert_eq!(block_extra(3, 0), PENALTY_FACTOR * 3);
        assert_eq!(block_extra(3, 4), LEVELS as u64 * 9);
        assert_eq!(block_extra(3, 1), PENALTY_FACTOR * 3);
    }
}
