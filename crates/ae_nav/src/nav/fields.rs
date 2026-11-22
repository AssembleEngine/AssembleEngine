//! Demand-generated, shared flow-field tiles.
//!
//! A field solves a single directed grid problem inside one sector. It is
//! built by reverse Dijkstra from its seeds: destination fields are seeded at
//! the goal cell with 0, transit fields at every exit cell with
//! `crossing_cost + downstream cost-to-go` of the abstract node on the far
//! side. Incoming edges are traversed with their original forward costs, so
//! `D(u) = min_v c(u,v) + D(v)` holds for direction-dependent costs. Units move
//! by minimising `edge_cost(u,v) + D(v)`, never `D(v)` alone.
//!
//! All arithmetic is integer: costs are `u64` and additions saturate at
//! `u64::MAX`, which doubles as "unreachable". Seeds are stored relative to
//! their minimum (`u32`, saturating) so fields that differ only by a constant
//! offset share one cache entry.
use crate::math::{CellPos, SectorPos, NEIGHBORS4, NEIGHBORS8};
use crate::nav::base::{diagonal_corner_ok, traversal_cost_units, ProfileGrid};
use crate::nav::crowd::{BLOCK_CELLS, LEVELS};
use crate::terrain::Terrain;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::Arc;

/// Marks a seed or cell without an explicit exit move.
pub const NO_EXIT: u8 = u8::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GateSeed {
    pub cell: CellPos,
    /// Index into `NEIGHBORS4` for the boundary crossing this seed stands for,
    /// or `NO_EXIT` for a destination seed.
    pub exit_dir: u8,
    /// Cost-to-go from `cell` (crossing included), relative to the field's
    /// minimum seed.
    pub downstream_cost: u32,
}

/// Cache identity of a field: profile, the sector's own terrain version,
/// the complete exit boundary condition and the standing-crowd levels it
/// was costed with.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FieldKey {
    pub profile_id: u8,
    pub sector: SectorPos,
    pub sector_version: u32,
    /// Sorted by cell, one seed per cell, minimum value 0.
    pub seeds: Vec<GateSeed>,
    /// Extra cost of the sector's `crowd::BLOCK_CELLS` blocks in
    /// `1 / crowd::LEVELS` of a step (`crowd::block_extra`), row-major;
    /// empty for no crowd costs (`nav::crowd`).
    #[serde(default)]
    pub crowd: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct FlowField {
    pub key: FieldKey,
    pub sector: SectorPos,
    pub x0: i32,
    pub y0: i32,
    pub w: i32,
    pub h: i32,
    pub cost_to_go: Vec<u64>,
    /// Per cell: `NEIGHBORS4` index when the optimal action is to cross the
    /// sector boundary from here, else `NO_EXIT`.
    pub exit_dir: Vec<u8>,
    pub build_ms: f64,
    pub expansions: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Open {
    cost: u64,
    cell: u32,
}
impl PartialOrd for Open {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Open {
    // Min-heap on (cost, local cell index): a total, deterministic order.
    fn cmp(&self, o: &Self) -> Ordering {
        o.cost.cmp(&self.cost).then_with(|| o.cell.cmp(&self.cell))
    }
}

/// Dijkstra restricted to one sector's passable cells. `reverse = true`
/// computes cost-to-go to the sources (walking incoming edges with their
/// forward cost); `reverse = false` computes cost-from the sources.
/// Returns (distances over the sector rect, expansions).
pub fn sector_dijkstra(
    terrain: &Terrain,
    grid: &ProfileGrid,
    sector: SectorPos,
    sources: &[(CellPos, u64)],
    reverse: bool,
) -> (Vec<u64>, u32) {
    sector_dijkstra_crowd(terrain, grid, sector, sources, reverse, &[])
}

/// `sector_dijkstra` where leaving a cell also costs its block's standing
/// crowd and traffic: `crowd` holds the sector's block extras
/// (`FieldKey::crowd`), and a step costs `1 + extra / LEVELS` times its
/// terrain cost, as portal-graph edges do (`nav::crowd`).
pub fn sector_dijkstra_crowd(
    terrain: &Terrain,
    grid: &ProfileGrid,
    sector: SectorPos,
    sources: &[(CellPos, u64)],
    reverse: bool,
    crowd: &[u8],
) -> (Vec<u64>, u32) {
    let g = &terrain.geom;
    let (x0, y0, x1, y1) = g.sector_bounds(sector);
    let w = x1 - x0;
    let n = (w * (y1 - y0)).max(0) as usize;
    let passable = |x: i32, y: i32| -> bool {
        x >= x0 && y >= y0 && x < x1 && y < y1 && grid.passable[(y * g.cells_x + x) as usize] != 0
    };
    let off = |x: i32, y: i32| -> usize { ((y - y0) * w + (x - x0)) as usize };
    let bw = (w + BLOCK_CELLS - 1) / BLOCK_CELLS;
    let crowd_cost = |x: i32, y: i32, step: u64| -> u64 {
        if crowd.is_empty() {
            return step;
        }
        let extra = crowd[(((y - y0) / BLOCK_CELLS) * bw + (x - x0) / BLOCK_CELLS) as usize] as u64;
        step + step * extra / LEVELS as u64
    };
    let mut dist = vec![u64::MAX; n];
    let mut open = BinaryHeap::new();
    for &(c, v) in sources {
        if !passable(c.x, c.y) {
            continue;
        }
        let i = off(c.x, c.y);
        if v < dist[i] {
            dist[i] = v;
            open.push(Open {
                cost: v,
                cell: i as u32,
            });
        }
    }
    let mut expansions = 0u32;
    while let Some(node) = open.pop() {
        let ci = node.cell as usize;
        if node.cost != dist[ci] {
            continue;
        }
        expansions += 1;
        let cx = x0 + ci as i32 % w;
        let cy = y0 + ci as i32 / w;
        for &(dx, dy) in &NEIGHBORS8 {
            let ox = cx + dx;
            let oy = cy + dy;
            if !passable(ox, oy) {
                continue;
            }
            let step = if reverse {
                // Edge other -> current, costed in its forward direction.
                if !diagonal_corner_ok(passable, ox, oy, -dx, -dy) {
                    continue;
                }
                crowd_cost(ox, oy, traversal_cost_units(terrain, CellPos::new(ox, oy), -dx, -dy))
            } else {
                if !diagonal_corner_ok(passable, cx, cy, dx, dy) {
                    continue;
                }
                crowd_cost(cx, cy, traversal_cost_units(terrain, CellPos::new(cx, cy), dx, dy))
            };
            let cand = node.cost.saturating_add(step);
            let oi = off(ox, oy);
            if cand < dist[oi] {
                dist[oi] = cand;
                open.push(Open {
                    cost: cand,
                    cell: oi as u32,
                });
            }
        }
    }
    (dist, expansions)
}

/// Build a flow field for the key's sector. Returns `None` if no seed lies on
/// a clearance-passable cell of the sector (the boundary condition is invalid
/// at this terrain version).
pub fn build_flow_field(terrain: &Terrain, grid: &ProfileGrid, key: FieldKey) -> Option<FlowField> {
    let start = std::time::Instant::now();
    let g = &terrain.geom;
    let sector = key.sector;
    let (x0, y0, x1, y1) = g.sector_bounds(sector);
    let (w, h) = (x1 - x0, y1 - y0);
    if w <= 0 || h <= 0 {
        return None;
    }
    let sources: Vec<(CellPos, u64)> = key.seeds.iter().map(|s| (s.cell, s.downstream_cost as u64)).collect();
    let (dist, expansions) = sector_dijkstra_crowd(terrain, grid, sector, &sources, true, &key.crowd);
    let mut exit_dir = vec![NO_EXIT; dist.len()];
    let mut any = false;
    for s in &key.seeds {
        if s.cell.x < x0 || s.cell.y < y0 || s.cell.x >= x1 || s.cell.y >= y1 {
            continue;
        }
        let i = ((s.cell.y - y0) * w + (s.cell.x - x0)) as usize;
        if dist[i] == u64::MAX {
            continue;
        }
        any = true;
        // The seed's own value survived: crossing here is optimal.
        if s.exit_dir != NO_EXIT && dist[i] == s.downstream_cost as u64 {
            exit_dir[i] = s.exit_dir;
        }
    }
    if !any {
        return None;
    }
    Some(FlowField {
        key,
        sector,
        x0,
        y0,
        w,
        h,
        cost_to_go: dist,
        exit_dir,
        build_ms: start.elapsed().as_secs_f64() * 1000.0,
        expansions,
    })
}

impl FlowField {
    #[inline]
    pub fn idx(&self, x: i32, y: i32) -> Option<usize> {
        if x < self.x0 || y < self.y0 || x >= self.x0 + self.w || y >= self.y0 + self.h {
            None
        } else {
            Some(((y - self.y0) * self.w + (x - self.x0)) as usize)
        }
    }
    pub fn cost_at(&self, c: CellPos) -> u64 {
        self.idx(c.x, c.y).map(|i| self.cost_to_go[i]).unwrap_or(u64::MAX)
    }
    pub fn is_reachable(&self, c: CellPos) -> bool {
        self.cost_at(c) != u64::MAX
    }

    /// Boundary crossing to take from `at`, if crossing is optimal there.
    pub fn exit_from(&self, at: CellPos) -> Option<CellPos> {
        let d = self.exit_dir[self.idx(at.x, at.y)?];
        if d == NO_EXIT {
            return None;
        }
        let (dx, dy) = NEIGHBORS4[d as usize];
        Some(CellPos::new(at.x + dx, at.y + dy))
    }

    /// Strictly best step by `edge cost + downstream cost`; ties broken by
    /// (y, x) of the target cell.
    pub fn best_step(&self, terrain: &Terrain, grid: &ProfileGrid, at: CellPos) -> Option<(CellPos, u64, u64)> {
        self.best_step_spread(terrain, grid, at, 0)
    }

    /// Lane-spread variant used by moving units. Candidates must strictly
    /// decrease cost-to-go (guaranteed progress, no oscillation); among those
    /// within one cheap cardinal step of the best `edge + D(v)` total, the
    /// pick is a deterministic hash of (seed, cell). Spreading equal-cost
    /// followers across lanes keeps a large group off a single cell line.
    /// Returns (next cell, edge cost, downstream cost). An optimal boundary
    /// crossing is returned as a cell outside the sector.
    pub fn best_step_spread(
        &self,
        terrain: &Terrain,
        grid: &ProfileGrid,
        at: CellPos,
        seed: u64,
    ) -> Option<(CellPos, u64, u64)> {
        if !self.is_reachable(at) {
            return None;
        }
        if let Some(out) = self.exit_from(at) {
            return Some((out, 0, self.cost_at(at)));
        }
        let current_down = self.cost_at(at);
        let passable = |x: i32, y: i32| -> bool {
            x >= self.x0
                && y >= self.y0
                && x < self.x0 + self.w
                && y < self.y0 + self.h
                && grid.passable[(y * terrain.geom.cells_x + x) as usize] != 0
        };
        let mut candidates: [(CellPos, u64, u64, u64); 8] = [(CellPos::default(), 0, 0, 0); 8];
        let mut n = 0usize;
        let mut best_total = u64::MAX;
        for &(dx, dy) in &NEIGHBORS8 {
            let nc = CellPos::new(at.x + dx, at.y + dy);
            if !passable(nc.x, nc.y) || !diagonal_corner_ok(passable, at.x, at.y, dx, dy) {
                continue;
            }
            let down = self.cost_at(nc);
            if down >= current_down {
                continue;
            }
            let edge = traversal_cost_units(terrain, at, dx, dy);
            let total = down.saturating_add(edge);
            best_total = best_total.min(total);
            candidates[n] = (nc, edge, down, total);
            n += 1;
        }
        if n == 0 {
            return None;
        }
        let band = best_total.saturating_add(16);
        let near = &mut candidates[..n];
        near.sort_unstable_by_key(|c| (c.3, c.0.y, c.0.x));
        let m = near.iter().take_while(|c| c.3 <= band).count();
        let pick = if seed == 0 {
            0
        } else {
            (crate::rng::hash2(seed, at.x, at.y) as usize) % m
        };
        let c = near[pick];
        Some((c.0, c.1, c.2))
    }

    /// Step used to walk a unit's look-ahead chain. Candidates must strictly
    /// decrease cost-to-go; among those within `band` of the best
    /// `edge + D(v)` total, the one closest to the straight line
    /// `from -> toward` wins (then the one making more progress along it,
    /// then (y, x)). Equal-cost octile paths then interleave diagonal and
    /// straight steps along the direct line, Bresenham-style, instead of
    /// forming an L.
    pub fn best_step_toward(
        &self,
        terrain: &Terrain,
        grid: &ProfileGrid,
        at: CellPos,
        from: CellPos,
        toward: CellPos,
        band: u64,
    ) -> Option<CellPos> {
        if !self.is_reachable(at) {
            return None;
        }
        if let Some(out) = self.exit_from(at) {
            return Some(out);
        }
        let current_down = self.cost_at(at);
        let passable = |x: i32, y: i32| -> bool {
            x >= self.x0
                && y >= self.y0
                && x < self.x0 + self.w
                && y < self.y0 + self.h
                && grid.passable[(y * terrain.geom.cells_x + x) as usize] != 0
        };
        let (lx, ly) = ((toward.x - from.x) as i64, (toward.y - from.y) as i64);
        // (cell, total, off-line distance (scaled), negative progress)
        let mut candidates: [(CellPos, u64, i64, i64); 8] = [(CellPos::default(), 0, 0, 0); 8];
        let mut n = 0usize;
        let mut best_total = u64::MAX;
        for &(dx, dy) in &NEIGHBORS8 {
            let nc = CellPos::new(at.x + dx, at.y + dy);
            if !passable(nc.x, nc.y) || !diagonal_corner_ok(passable, at.x, at.y, dx, dy) {
                continue;
            }
            let down = self.cost_at(nc);
            if down >= current_down {
                continue;
            }
            let total = down.saturating_add(traversal_cost_units(terrain, at, dx, dy));
            best_total = best_total.min(total);
            let (rx, ry) = ((nc.x - from.x) as i64, (nc.y - from.y) as i64);
            candidates[n] = (nc, total, (rx * ly - ry * lx).abs(), -(rx * lx + ry * ly));
            n += 1;
        }
        let band = best_total.saturating_add(band);
        candidates[..n]
            .iter()
            .filter(|c| c.1 <= band)
            .min_by_key(|c| (c.2, c.3, c.0.y, c.0.x))
            .map(|c| c.0)
    }

    pub fn reachable_count(&self) -> usize {
        self.cost_to_go.iter().filter(|v| **v != u64::MAX).count()
    }
    pub fn bytes(&self) -> usize {
        self.cost_to_go.len() * (std::mem::size_of::<u64>() + 1)
            + self.key.seeds.len() * std::mem::size_of::<GateSeed>()
    }
}

#[derive(Clone, Debug, Default)]
pub struct FieldCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub builds: u64,
    pub evictions: u64,
    pub build_ms_total: f64,
    pub last_build_ms: f64,
}

/// Bounded LRU cache of immutable fields. Only the simulation thread touches
/// it, and only at tick boundaries, so hits/evictions are deterministic.
pub struct FieldCache {
    fields: BTreeMap<FieldKey, (Arc<FlowField>, u64)>,
    pub max_fields: usize,
    pub stats: FieldCacheStats,
}

impl FieldCache {
    pub fn new(max_fields: usize) -> Self {
        Self {
            fields: BTreeMap::new(),
            max_fields: max_fields.max(1),
            stats: Default::default(),
        }
    }
    pub fn get(&mut self, key: &FieldKey, tick: u64) -> Option<Arc<FlowField>> {
        if let Some((f, used)) = self.fields.get_mut(key) {
            *used = tick;
            self.stats.hits += 1;
            Some(Arc::clone(f))
        } else {
            self.stats.misses += 1;
            None
        }
    }
    pub fn contains(&self, key: &FieldKey) -> bool {
        self.fields.contains_key(key)
    }
    pub fn insert(&mut self, field: FlowField, tick: u64) -> Arc<FlowField> {
        self.stats.builds += 1;
        self.stats.last_build_ms = field.build_ms;
        self.stats.build_ms_total += field.build_ms;
        let f = Arc::new(field);
        self.fields.insert(f.key.clone(), (Arc::clone(&f), tick));
        self.evict_if_needed();
        f
    }
    /// Evict least-recently-used entries, ordered by (last use tick, key):
    /// never by pointer or hash order. Holders keep evicted fields alive
    /// through their `Arc`, so eviction never invalidates a live route.
    fn evict_if_needed(&mut self) {
        while self.fields.len() > self.max_fields {
            let victim = self
                .fields
                .iter()
                .min_by_key(|(k, (_, used))| (*used, *k))
                .map(|(k, _)| k.clone());
            match victim {
                Some(k) => {
                    self.fields.remove(&k);
                    self.stats.evictions += 1;
                }
                None => break,
            }
        }
    }
    /// Drop fields of a sector whose terrain version moved on.
    pub fn purge_sector_versions(&mut self, current: impl Fn(u8, SectorPos) -> u32) {
        let stale: Vec<FieldKey> = self
            .fields
            .keys()
            .filter(|k| k.sector_version != current(k.profile_id, k.sector))
            .cloned()
            .collect();
        for k in stale {
            self.fields.remove(&k);
            self.stats.evictions += 1;
        }
    }
    pub fn len(&self) -> usize {
        self.fields.len()
    }
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
    pub fn bytes(&self) -> usize {
        self.fields.values().map(|(f, _)| f.bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::WorldGeometry;
    use crate::nav::ProfileGrid;
    use crate::terrain::TestTerrain;

    fn setup() -> (Terrain, ProfileGrid) {
        let mut tt = TestTerrain::new(32, 32);
        tt.wall(12, 8, 12, 14);
        let t = tt.into_terrain(1);
        let p = crate::nav::MovementProfile {
            id: 0,
            name: "t".into(),
            clearance_cells: 0,
            max_slope: 1000,
            unit_radius: 0.4,
            base_speed: 10.0,
            medium: crate::terrain::Medium::ANY,
        };
        let grid = ProfileGrid::build(&t, &p);
        (t, grid)
    }

    fn key(seeds: Vec<GateSeed>) -> FieldKey {
        FieldKey {
            profile_id: 0,
            sector: SectorPos::new(1, 1),
            sector_version: 1,
            seeds,
            crowd: Vec::new(),
        }
    }

    #[test]
    fn crowd_levels_route_the_field_around_a_dense_block() {
        // One 64-cell sector; a level-3 crowd 16 cells wide and 32 tall in
        // its middle (blocks x 3..4, y 2..5); exit seed on the east edge.
        let mut tt = TestTerrain::new(64, 64);
        tt.geom = WorldGeometry::new(64, 64, 1.0, 64);
        let t = tt.into_terrain(1);
        let p = crate::nav::MovementProfile {
            id: 0,
            name: "t".into(),
            clearance_cells: 0,
            max_slope: 1000,
            unit_radius: 0.4,
            base_speed: 10.0,
            medium: crate::terrain::Medium::ANY,
        };
        let grid = ProfileGrid::build(&t, &p);
        let mut crowd = vec![0u8; 64];
        for by in 2..6 {
            for bx in 3..5 {
                crowd[by * 8 + bx] = crate::nav::crowd::block_extra(3, 0) as u8;
            }
        }
        let seeds = vec![GateSeed {
            cell: CellPos::new(63, 32),
            exit_dir: 0,
            downstream_cost: 0,
        }];
        let walk = |crowd: Vec<u8>| -> usize {
            let f = build_flow_field(
                &t,
                &grid,
                FieldKey {
                    profile_id: 0,
                    sector: SectorPos::new(0, 0),
                    sector_version: 1,
                    seeds: seeds.clone(),
                    crowd,
                },
            )
            .unwrap();
            let (mut at, mut inside) = (CellPos::new(4, 32), 0);
            while let Some((next, _, _)) = f.best_step(&t, &grid, at) {
                if (24..40).contains(&next.x) && (16..48).contains(&next.y) {
                    inside += 1;
                }
                at = next;
                if at.x >= 64 {
                    break;
                }
            }
            assert_eq!(at, CellPos::new(64, 32), "leaves through the exit");
            inside
        };
        assert!(
            walk(Vec::new()) >= 16,
            "without crowd costs the field goes straight through"
        );
        assert_eq!(walk(crowd), 0, "with them it goes around");
    }

    #[test]
    fn field_routes_around_wall_and_respects_seed_identity() {
        let (t, grid) = setup();
        let k = key(vec![GateSeed {
            cell: CellPos::new(14, 14),
            exit_dir: NO_EXIT,
            downstream_cost: 123,
        }]);
        let f = build_flow_field(&t, &grid, k.clone()).unwrap();
        assert!(f.is_reachable(CellPos::new(9, 9)));
        assert!(f.cost_at(CellPos::new(9, 9)) >= 123);
        // The wall at x=12, y=8..14 forces a detour below it.
        let step = f.best_step(&t, &grid, CellPos::new(11, 10)).unwrap();
        assert!(step.0.y >= 10);
        let k2 = FieldKey {
            sector_version: 2,
            ..k.clone()
        };
        assert_ne!(k, k2);
    }

    #[test]
    fn exit_seeds_use_downstream_values_not_zero() {
        // Two exits on the east edge of sector (1,1) (cells x=15). The
        // northern one has a much larger downstream cost (a cardinal step
        // costs 100 here), so a unit standing right next to it must still
        // walk 7 cells to the southern exit.
        let (t, grid) = setup();
        let east = 0u8; // NEIGHBORS4[0] == (1, 0)
        let seeds = vec![
            GateSeed {
                cell: CellPos::new(15, 8),
                exit_dir: east,
                downstream_cost: 2000,
            },
            GateSeed {
                cell: CellPos::new(15, 15),
                exit_dir: east,
                downstream_cost: 0,
            },
        ];
        let f = build_flow_field(&t, &grid, key(seeds.clone())).unwrap();
        assert_eq!(f.exit_from(CellPos::new(15, 15)), Some(CellPos::new(16, 15)));
        assert_eq!(
            f.exit_from(CellPos::new(15, 8)),
            None,
            "expensive exit must not be taken"
        );
        let step = f.best_step(&t, &grid, CellPos::new(15, 8)).unwrap();
        assert!(step.0.y > 8, "unit next to the expensive exit heads south");
        // Same cells with all-zero seeds is a different boundary condition:
        // different key, different behaviour.
        let zero: Vec<GateSeed> = seeds
            .iter()
            .map(|s| GateSeed {
                downstream_cost: 0,
                ..*s
            })
            .collect();
        let fz = build_flow_field(&t, &grid, key(zero.clone())).unwrap();
        assert_ne!(key(zero), key(seeds));
        assert_eq!(fz.exit_from(CellPos::new(15, 8)), Some(CellPos::new(16, 8)));
    }

    #[test]
    fn cache_evicts_deterministically() {
        let (t, grid) = setup();
        let run = || {
            let mut cache = FieldCache::new(2);
            for (n, x, y) in [(0u64, 9, 9), (1, 14, 14), (2, 10, 14)] {
                let k = key(vec![GateSeed {
                    cell: CellPos::new(x, y),
                    exit_dir: NO_EXIT,
                    downstream_cost: 0,
                }]);
                cache.insert(build_flow_field(&t, &grid, k).unwrap(), n);
            }
            (
                cache.len(),
                cache.stats.evictions,
                cache.fields.keys().cloned().collect::<Vec<_>>(),
            )
        };
        let a = run();
        assert_eq!(a.0, 2);
        assert_eq!(a, run());
    }

    #[test]
    fn world_geometry_sectors() {
        let g = WorldGeometry::new(100, 50, 2.0, 16);
        assert_eq!(g.sectors_x(), 7);
        assert_eq!(g.sector_of(CellPos::new(99, 49)), SectorPos::new(6, 3));
    }
}
