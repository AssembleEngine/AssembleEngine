//! Movement profiles, clearance-aware traversal, edge costs, and the
//! deterministic single-level A*/Dijkstra reference solver.
use crate::math::{CellPos, NEIGHBORS8};
use crate::terrain::{Medium, Terrain};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub const INF_COST: u64 = u64::MAX;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MovementProfile {
    pub id: u8,
    pub name: String,
    /// Conservative clearance radius in navigation cells. A path is valid only
    /// where this entire square is empty and slope-passable.
    pub clearance_cells: i32,
    pub max_slope: i16,
    pub unit_radius: f32,
    pub base_speed: f32,
    /// Water depths the unit may enter (default: anything not an obstacle).
    #[serde(default)]
    pub medium: Medium,
}

/// Largest random radius factor applied to a profile's nominal radius at
/// spawn (radii vary by ±8%).
pub const RADIUS_JITTER_MAX: f32 = 1.08;

/// Smallest clearance (cells) that keeps a circle of `radius` world units,
/// centred anywhere in its cell's centre region, clear of blocked cells.
pub fn min_clearance_cells(radius: f32, cell_size: f32) -> i32 {
    ((radius / cell_size) - 0.5).ceil().max(0.0) as i32
}

pub fn default_profiles(cell_size: f32) -> Vec<MovementProfile> {
    profiles_for(cell_size, 1.0)
}

/// Movement profiles for a cell size and radius scale. Clearance is the
/// larger of the profile's design value (heavies need wider gaps) and the
/// physical minimum for its largest body, so a cell path is always wide
/// enough for the unit that follows it.
pub fn profiles_for(cell_size: f32, radius_scale: f32) -> Vec<MovementProfile> {
    let light_radius = (cell_size * 0.34).max(1.6) * radius_scale;
    let heavy_radius = (cell_size * 0.52).max(2.6) * radius_scale;
    let clearance = |design: i32, r: f32| design.max(min_clearance_cells(r * RADIUS_JITTER_MAX, cell_size));
    vec![
        MovementProfile {
            id: 0,
            name: "Scout".into(),
            clearance_cells: clearance(1, light_radius),
            max_slope: 135,
            unit_radius: light_radius,
            base_speed: cell_size * 14.0,
            medium: Medium::ANY,
        },
        MovementProfile {
            id: 1,
            name: "Heavy".into(),
            clearance_cells: clearance(2, heavy_radius),
            max_slope: 95,
            unit_radius: heavy_radius,
            base_speed: cell_size * 10.0,
            medium: Medium::ANY,
        },
    ]
}

/// Deepest water (decimetres) the land profiles of [`water_profiles`] ford.
pub const FORD_DEPTH: i16 = 20;
/// Shallowest water (decimetres) the ship profile of [`water_profiles`] sails.
pub const SHIP_DRAFT: i16 = 30;

/// pathsim's two land profiles (now fording up to `FORD_DEPTH`) plus a ship,
/// a hovercraft and an amphibious profile, ids 0..=4 in that order. For maps
/// with water (`WaterMode::Surface`).
pub fn water_profiles(cell_size: f32) -> Vec<MovementProfile> {
    let mut out = profiles_for(cell_size, 1.0);
    for p in &mut out {
        p.medium = Medium::land(FORD_DEPTH);
    }
    let clearance = |design: i32, r: f32| design.max(min_clearance_cells(r * RADIUS_JITTER_MAX, cell_size));
    let ship_radius = (cell_size * 0.62).max(3.2);
    let hover_radius = (cell_size * 0.40).max(2.0);
    let amphibious_radius = (cell_size * 0.48).max(2.4);
    out.push(MovementProfile {
        id: 2,
        name: "Ship".into(),
        clearance_cells: clearance(2, ship_radius),
        max_slope: 60,
        unit_radius: ship_radius,
        base_speed: cell_size * 9.0,
        medium: Medium::ship(SHIP_DRAFT),
    });
    out.push(MovementProfile {
        id: 3,
        name: "Hover".into(),
        clearance_cells: clearance(1, hover_radius),
        max_slope: 110,
        unit_radius: hover_radius,
        base_speed: cell_size * 13.0,
        medium: Medium::HOVER,
    });
    out.push(MovementProfile {
        id: 4,
        name: "Amphibious".into(),
        clearance_cells: clearance(1, amphibious_radius),
        max_slope: 110,
        unit_radius: amphibious_radius,
        base_speed: cell_size * 9.0,
        medium: Medium::AMPHIBIOUS,
    });
    out
}

/// Clearance/slope-filtered passability for one movement profile.
///
/// Connectivity is not stored per cell here: components are derived from the
/// sector-region graph in `hierarchy`, which can be repaired locally.
#[derive(Clone, Debug)]
pub struct ProfileGrid {
    pub profile: MovementProfile,
    pub passable: Vec<u8>,
    /// Number of passable cells, kept up to date by `recompute_rect`.
    pub passable_count: usize,
    /// Lower bound of `move_cost` over open cells, used by the heuristic. A
    /// local repair only ever lowers it, so it stays admissible.
    pub min_cost: u8,
}

impl ProfileGrid {
    pub fn build(terrain: &Terrain, profile: &MovementProfile) -> Self {
        let g = &terrain.geom;
        let mut grid = Self {
            profile: profile.clone(),
            passable: vec![0u8; terrain.len()],
            passable_count: 0,
            min_cost: u8::MAX,
        };
        grid.recompute_rect(terrain, 0, 0, g.cells_x - 1, g.cells_y - 1);
        if grid.min_cost == u8::MAX {
            grid.min_cost = 1;
        }
        grid
    }

    /// Cells whose passability can change when terrain cells in the inclusive
    /// rect change: the slope test reads 8-neighbours, then clearance widens
    /// the influence by `clearance_cells`.
    pub fn influence_margin(&self) -> i32 {
        self.profile.clearance_cells.max(0) + 1
    }

    /// Recompute passability for every cell in the inclusive rect. A cell is
    /// passable only when the whole (2r+1)^2 square around it is open,
    /// slope-passable and inside the map: deliberately conservative for
    /// circular bodies.
    pub fn recompute_rect(&mut self, terrain: &Terrain, x0: i32, y0: i32, x1: i32, y1: i32) {
        let g = &terrain.geom;
        let r = self.profile.clearance_cells.max(0);
        let x0 = x0.max(0);
        let y0 = y0.max(0);
        let x1 = x1.min(g.cells_x - 1);
        let y1 = y1.min(g.cells_y - 1);
        if x0 > x1 || y0 > y1 {
            return;
        }
        // base_ok over the rect expanded by r (clamped), then a summed-area
        // table over that window.
        let wx0 = (x0 - r).max(0);
        let wy0 = (y0 - r).max(0);
        let wx1 = (x1 + r).min(g.cells_x - 1);
        let wy1 = (y1 + r).min(g.cells_y - 1);
        let ww = (wx1 - wx0 + 1) as usize;
        let wh = (wy1 - wy0 + 1) as usize;
        let mut integral = vec![0u32; (ww + 1) * (wh + 1)];
        for yy in 0..wh {
            let y = wy0 + yy as i32;
            let mut row = 0u32;
            for xx in 0..ww {
                let x = wx0 + xx as i32;
                let ok = self.base_ok(terrain, x, y);
                if ok {
                    self.min_cost = self
                        .min_cost
                        .min(terrain.move_cost[(y * g.cells_x + x) as usize].max(1));
                }
                row += (!ok) as u32;
                integral[(yy + 1) * (ww + 1) + xx + 1] = integral[yy * (ww + 1) + xx + 1] + row;
            }
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = (y * g.cells_x + x) as usize;
                let inside = x - r >= 0 && y - r >= 0 && x + r < g.cells_x && y + r < g.cells_y;
                let before = self.passable[i] as usize;
                if !inside {
                    self.passable[i] = 0;
                    self.passable_count -= before;
                    continue;
                }
                let ax = (x - r - wx0) as usize;
                let ay = (y - r - wy0) as usize;
                let bx = (x + r - wx0) as usize + 1;
                let by = (y + r - wy0) as usize + 1;
                let bad = integral[by * (ww + 1) + bx] + integral[ay * (ww + 1) + ax]
                    - integral[ay * (ww + 1) + bx]
                    - integral[by * (ww + 1) + ax];
                self.passable[i] = (bad == 0) as u8;
                self.passable_count = self.passable_count - before + self.passable[i] as usize;
            }
        }
    }

    /// Terrain open to the profile's medium whose height step to every such
    /// open 8-neighbour is within the profile's slope limit. Surface movers
    /// measure slope on the water surface; ground movers on the ground.
    fn base_ok(&self, terrain: &Terrain, x: i32, y: i32) -> bool {
        let g = &terrain.geom;
        let i = (y * g.cells_x + x) as usize;
        if !terrain.has_water() {
            // Dry map: pathsim's test, without the per-neighbour depth work.
            if terrain.blocked[i] != 0 {
                return false;
            }
            let h = terrain.height[i];
            for &(dx, dy) in &NEIGHBORS8 {
                if let Some(j) = g.cell_index_xy(x + dx, y + dy) {
                    if terrain.blocked[j] != 0 {
                        continue;
                    }
                    if (terrain.height[j] - h).abs() > self.profile.max_slope {
                        return false;
                    }
                }
            }
            return true;
        }
        let m = &self.profile.medium;
        if !terrain.open_for(i, m) {
            return false;
        }
        let h = terrain.standing_height(i, m);
        for &(dx, dy) in &NEIGHBORS8 {
            if let Some(j) = g.cell_index_xy(x + dx, y + dy) {
                if !terrain.open_for(j, m) {
                    continue;
                }
                if (terrain.standing_height(j, m) - h).abs() > self.profile.max_slope {
                    return false;
                }
            }
        }
        true
    }

    #[inline]
    pub fn is_passable(&self, terrain: &Terrain, c: CellPos) -> bool {
        if !terrain.geom.in_bounds(c) {
            return false;
        }
        self.passable[terrain.geom.cell_index(c)] != 0
    }
}

#[inline]
pub fn step_base(dx: i32, dy: i32) -> u64 {
    if dx != 0 && dy != 0 {
        14
    } else {
        10
    }
}

#[inline]
pub fn traversal_cost_units(terrain: &Terrain, from: CellPos, dx: i32, dy: i32) -> u64 {
    let to = CellPos::new(from.x + dx, from.y + dy);
    let fi = terrain.geom.cell_index(from);
    let ti = terrain.geom.cell_index(to);
    let base = ((terrain.move_cost[fi] as u64 + terrain.move_cost[ti] as u64) / 2).max(1);
    let mut cost = base * step_base(dx, dy);
    // `height` is the water surface over water, so climbs there are
    // measured on the surface (the same for every profile, so fields stay
    // shareable); on dry ground this is pathsim's climb, unchanged.
    let dh = terrain.height[ti] as i64 - terrain.height[fi] as i64;
    if dh > 0 {
        // Uphill has a deterministic surcharge. Downhill is not made cheaper so
        // that the octile heuristic remains admissible for all directed edges.
        cost += dh as u64 * 3;
    }
    cost
}

pub fn diagonal_corner_ok(passable: impl Fn(i32, i32) -> bool, x: i32, y: i32, dx: i32, dy: i32) -> bool {
    if dx == 0 || dy == 0 {
        return true;
    }
    passable(x + dx, y) && passable(x, y + dy)
}

pub fn octile_heuristic(dx: i32, dy: i32, min_cost: u8) -> u64 {
    let dx = dx.unsigned_abs();
    let dy = dy.unsigned_abs();
    let (lo, hi) = if dx < dy { (dx, dy) } else { (dy, dx) };
    (hi as u64 * 10 + lo as u64 * 4) * min_cost.max(1) as u64
}

#[derive(Clone, Copy, Debug)]
struct OpenNode {
    f: u64,
    h: u64,
    g: u64,
    cell: u32,
}
impl PartialEq for OpenNode {
    fn eq(&self, o: &Self) -> bool {
        self.f == o.f && self.h == o.h && self.g == o.g && self.cell == o.cell
    }
}
impl Eq for OpenNode {}
impl PartialOrd for OpenNode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OpenNode {
    fn cmp(&self, o: &Self) -> Ordering {
        // BinaryHeap is a max heap: reverse all comparisons.
        o.f.cmp(&self.f)
            .then_with(|| o.h.cmp(&self.h))
            .then_with(|| o.g.cmp(&self.g))
            .then_with(|| o.cell.cmp(&self.cell))
    }
}

#[derive(Clone, Debug)]
pub struct CellPath {
    pub cells: Vec<CellPos>,
    pub cost: u64,
    pub expansions: u32,
}

/// Deterministic reference A*. `zero_heuristic` turns it into Dijkstra.
/// Directed edge costs, diagonal corner rules and profile clearance are all
/// shared with the scalable navigator.
pub fn astar(
    terrain: &Terrain,
    grid: &ProfileGrid,
    start: CellPos,
    goal: CellPos,
    zero_heuristic: bool,
) -> Option<CellPath> {
    astar_impl(terrain, grid, start, goal, zero_heuristic, u32::MAX)
}

fn astar_impl(
    terrain: &Terrain,
    grid: &ProfileGrid,
    start: CellPos,
    goal: CellPos,
    zero_heuristic: bool,
    max_expansions: u32,
) -> Option<CellPath> {
    let g = &terrain.geom;
    if !g.in_bounds(start) || !g.in_bounds(goal) {
        return None;
    }
    if !grid.is_passable(terrain, start) || !grid.is_passable(terrain, goal) {
        return None;
    }
    let n = terrain.len();
    let mut dist = vec![u64::MAX; n];
    let mut parent = vec![u32::MAX; n];
    let si = g.cell_index(start);
    let gi = g.cell_index(goal);
    dist[si] = 0;
    let h0 = if zero_heuristic {
        0
    } else {
        octile_heuristic(start.x - goal.x, start.y - goal.y, grid.min_cost)
    };
    let mut open = BinaryHeap::new();
    open.push(OpenNode {
        f: h0,
        h: h0,
        g: 0,
        cell: si as u32,
    });
    let mut expansions = 0u32;
    let passable = |x: i32, y: i32| -> bool {
        if !g.in_bounds(CellPos::new(x, y)) {
            return false;
        }
        grid.passable[(y * g.cells_x + x) as usize] != 0
    };
    while let Some(node) = open.pop() {
        let ci = node.cell as usize;
        if node.g != dist[ci] {
            continue;
        }
        if expansions >= max_expansions {
            return None;
        }
        expansions += 1;
        if ci == gi {
            let mut cells = Vec::new();
            let mut c = gi as u32;
            while c != u32::MAX {
                let cc = c as usize;
                cells.push(CellPos::new((cc as i32) % g.cells_x, (cc as i32) / g.cells_x));
                if c == si as u32 {
                    break;
                }
                c = parent[cc];
            }
            cells.reverse();
            return Some(CellPath {
                cells,
                cost: node.g,
                expansions,
            });
        }
        let cx = (ci as i32) % g.cells_x;
        let cy = (ci as i32) / g.cells_x;
        let from = CellPos::new(cx, cy);
        for &(dx, dy) in &NEIGHBORS8 {
            let nx = cx + dx;
            let ny = cy + dy;
            if !passable(nx, ny) {
                continue;
            }
            if !diagonal_corner_ok(passable, cx, cy, dx, dy) {
                continue;
            }
            let ni = (ny * g.cells_x + nx) as usize;
            let step = traversal_cost_units(terrain, from, dx, dy);
            let ng = node.g.saturating_add(step);
            if ng < dist[ni] {
                dist[ni] = ng;
                parent[ni] = ci as u32;
                let h = if zero_heuristic {
                    0
                } else {
                    octile_heuristic(nx - goal.x, ny - goal.y, grid.min_cost)
                };
                open.push(OpenNode {
                    f: ng.saturating_add(h),
                    h,
                    g: ng,
                    cell: ni as u32,
                });
            }
        }
    }
    None
}

/// Bounded A*: gives up (returns `None`) once `max_expansions` nodes have
/// been expanded. Used for individual fallback searches with a deterministic
/// work quota.
pub fn astar_bounded(
    terrain: &Terrain,
    grid: &ProfileGrid,
    start: CellPos,
    goal: CellPos,
    max_expansions: u32,
) -> Option<CellPath> {
    astar_impl(terrain, grid, start, goal, false, max_expansions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TestTerrain;

    fn profile() -> MovementProfile {
        MovementProfile {
            id: 0,
            name: "t".into(),
            clearance_cells: 0,
            max_slope: 1000,
            unit_radius: 0.4,
            base_speed: 10.0,
            medium: crate::terrain::Medium::ANY,
        }
    }

    #[test]
    fn diagonal_requires_both_ordinals() {
        let mut tt = TestTerrain::new(8, 8);
        tt.wall(3, 2, 3, 2);
        let t = tt.into_terrain(1);
        let grid = ProfileGrid::build(&t, &profile());
        let p = astar(&t, &grid, CellPos::new(2, 2), CellPos::new(4, 4), false).unwrap();
        // The solver must go around the blocked cell, never cut the corner.
        for w in p.cells.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if a.x != b.x && a.y != b.y {
                assert!(grid.passable[t.geom.cell_index(CellPos::new(b.x, a.y))] != 0);
                assert!(grid.passable[t.geom.cell_index(CellPos::new(a.x, b.y))] != 0);
            }
        }
    }

    #[test]
    fn heuristic_matches_dijkstra_on_small_random_costs() {
        use crate::rng::Rng;
        let mut tt = TestTerrain::new(24, 24);
        let mut rng = Rng::new(7);
        for y in 0..24 {
            for x in 0..24 {
                let i = (y * 24 + x) as usize;
                tt.cost[i] = rng.range_i32(6, 25) as u8;
                tt.height[i] = (rng.range_i32(0, 250)) as i16;
            }
        }
        let t = tt.into_terrain(3);
        let grid = ProfileGrid::build(&t, &profile());
        let a = astar(&t, &grid, CellPos::new(1, 1), CellPos::new(22, 21), false).unwrap();
        let d = astar(&t, &grid, CellPos::new(1, 1), CellPos::new(22, 21), true).unwrap();
        assert_eq!(a.cost, d.cost);
        assert!(a.expansions <= d.expansions);
    }

    #[test]
    fn uphill_is_more_expensive_than_downhill() {
        let mut tt = TestTerrain::new(10, 3);
        for y in 0..3 {
            for x in 0..10 {
                tt.height[x as usize + y * 10] = (x * 10) as i16;
            }
        }
        let t = tt.into_terrain(1);
        let grid = ProfileGrid::build(&t, &profile());
        let up = astar(&t, &grid, CellPos::new(1, 1), CellPos::new(8, 1), false).unwrap();
        let down = astar(&t, &grid, CellPos::new(8, 1), CellPos::new(1, 1), false).unwrap();
        assert!(up.cost > down.cost, "{} vs {}", up.cost, down.cost);
        assert_eq!(up.cells.len(), down.cells.len());
    }
}
