//! Shared destination contexts and field recipes.
//!
//! A `DestContext` is the abstract half of navigation towards one destination
//! for one movement profile: a backward Dijkstra over the directed portal
//! graph that gives every abstract node its cost-to-go. It is computed once on
//! a worker and shared (via `Arc`) by every cohort, command and stage whose
//! destination quantises to the same key. From it, the field for *any*
//! (sector, region) a unit happens to be in is fully determined: its exits are
//! seeded with `crossing cost + label(far node)`. Fields are built on demand,
//! ahead of moving units, and cached by boundary condition.
use crate::math::{CellPos, SectorPos, WorldPos, NEIGHBORS4};
use crate::nav::base::traversal_cost_units;
use crate::nav::crowd::CrowdMap;
use crate::nav::fields::{sector_dijkstra, FieldKey, GateSeed, NO_EXIT};
use crate::nav::hierarchy::{perimeter_index, perimeter_len, perimeter_offset, NavHierarchy, ProfileHierarchy};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

/// Destinations are quantised to blocks of this many cells (within the same
/// sector region) for sharing the abstract search. Exact arrival is handled
/// separately by the destination field and per-unit slots.
pub const CONTEXT_QUANTUM_CELLS: i32 = 8;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContextKey {
    pub profile_id: u8,
    pub nav_version: u32,
    pub goal_sector: u32,
    pub goal_region: u16,
    /// Quantised anchor cell the abstract labels are measured to.
    pub anchor: CellPos,
    /// Crowd map version whose standing-crowd costs the labels include
    /// (`nav::crowd`); 0 for none.
    pub crowd: u32,
}

#[derive(Clone, Debug)]
pub struct DestContext {
    pub key: ContextKey,
    /// Cost-to-go per abstract node (`u64::MAX` = cannot reach the goal).
    pub labels: Vec<u64>,
    /// Successor toward the goal per node (`u32::MAX` = in the goal region).
    pub next_hop: Vec<u32>,
    /// In-sector cost from each boundary cell of the goal sector to the
    /// anchor (by `perimeter_index`; `u64::MAX` = none).
    pub goal_boundary: Vec<u64>,
    pub expansions: u32,
    pub build_ms: f64,
    /// The crowd map whose costs the labels include (`key.crowd`); fields
    /// for this context cost the same crowds inside their sectors.
    pub crowd: Option<Arc<CrowdMap>>,
}

/// Context key for a target cell, or `None` if the cell is not passable for
/// the profile.
pub fn context_key(h: &NavHierarchy, profile_id: u8, target: CellPos) -> Option<ContextKey> {
    let geom = &h.terrain.geom;
    let ph = h.hierarchy(profile_id);
    let (sector, region) = ph.region_at(geom, target)?;
    let (x0, y0, x1, y1) = geom.sector_bounds(geom.sector_of(target));
    let q = CONTEXT_QUANTUM_CELLS;
    let ax = ((target.x / q) * q + q / 2).clamp(x0, x1 - 1);
    let ay = ((target.y / q) * q + q / 2).clamp(y0, y1 - 1);
    let anchor = CellPos::new(ax, ay);
    let anchor = if ph.region_at(geom, anchor) == Some((sector, region)) { anchor } else { target };
    Some(ContextKey { profile_id, nav_version: h.version, goal_sector: sector, goal_region: region, anchor, crowd: 0 })
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Open { cost: u64, node: u32 }
impl PartialOrd for Open { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for Open { fn cmp(&self, o: &Self) -> Ordering { o.cost.cmp(&self.cost).then_with(|| o.node.cmp(&self.node)) } }

/// Backward Dijkstra over the portal graph from the goal region. Initial
/// labels of the goal region's nodes are exact in-sector costs to the anchor.
/// The search covers the goal's whole component: the graph is small (a few
/// thousand nodes per profile) so this costs milliseconds, and it means a unit
/// pushed off its expected corridor never lacks a valid route.
///
/// With `key.crowd` set, edges also cost their standing-crowd extra from
/// `crowd` (which must be that version). There is no exemption near the
/// goal: when the goal is inside a crowd every route enters it, and the
/// cheapest entry is the most direct one. (Exempting edges within one
/// sector of the goal's let routes to a point just outside a crowd cut
/// straight through its nearest part.)
pub fn build_context(h: &NavHierarchy, key: &ContextKey, crowd: Option<&Arc<CrowdMap>>) -> Option<DestContext> {
    let start = std::time::Instant::now();
    if key.nav_version != h.version { return None; }
    let geom = &h.terrain.geom;
    let ph = h.hierarchy(key.profile_id);
    let grid = h.grid(key.profile_id);
    let gsec = SectorPos::new(key.goal_sector as i32 % geom.sectors_x(), key.goal_sector as i32 / geom.sectors_x());
    if ph.region_at(geom, key.anchor) != Some((key.goal_sector, key.goal_region)) { return None; }
    let (dist, mut expansions) = sector_dijkstra(&h.terrain, grid, gsec, &[(key.anchor, 0)], true);
    let gbounds = geom.sector_bounds(gsec);
    let (x0, y0, x1, _) = gbounds;
    let w = x1 - x0;
    let mut goal_boundary = vec![u64::MAX; perimeter_len(gbounds)];
    for (k, slot) in goal_boundary.iter_mut().enumerate() { *slot = dist[perimeter_offset(gbounds, k)]; }
    let n = ph.nodes.len();
    let mut labels = vec![u64::MAX; n];
    let mut next_hop = vec![u32::MAX; n];
    let mut open = BinaryHeap::new();
    for &id in &ph.sector_nodes[key.goal_sector as usize] {
        let node = &ph.nodes[id as usize];
        if node.region != key.goal_region { continue; }
        let d = dist[((node.cell.y - y0) * w + (node.cell.x - x0)) as usize];
        if d < labels[id as usize] { labels[id as usize] = d; open.push(Open { cost: d, node: id }); }
    }
    let crowd = crowd.filter(|m| key.crowd != 0 && m.version == key.crowd);
    while let Some(Open { cost, node }) = open.pop() {
        if cost != labels[node as usize] { continue; }
        expansions += 1;
        for (k, &(from, c)) in ph.in_edges[node as usize].iter().enumerate() {
            let extra = crowd.map_or(0, |m| m.edge_extra(h.version, key.profile_id, node, k));
            let cand = cost.saturating_add(c as u64 + extra as u64);
            if cand < labels[from as usize] {
                labels[from as usize] = cand;
                next_hop[from as usize] = node;
                open.push(Open { cost: cand, node: from });
            }
        }
    }
    Some(DestContext { key: key.clone(), labels, next_hop, goal_boundary, expansions, build_ms: start.elapsed().as_secs_f64() * 1000.0,
        crowd: crowd.cloned() })
}

/// Cost to go from `cell`, a boundary cell on abstract node `node`'s side
/// of its portal: the cheapest in-sector path to one of the region's nodes
/// plus that node's label, or the in-sector cost to the anchor in the goal
/// region. Every candidate is the cost of a real path, so this is never
/// below the true cost. It leans toward run middles (labels are measured
/// from there): on a uniform ramp a route 1 cell off the middle row costs
/// 1.7% more than optimal. Measuring to the whole run instead removed that
/// bias but was worse on average (long routes 3.7% vs 1.8% over optimal on
/// the default map), so it was not kept.
fn cost_to_go_from_boundary(h: &NavHierarchy, ctx: &DestContext, node: u32, cell: CellPos) -> Option<u64> {
    let geom = &h.terrain.geom;
    let ph = h.hierarchy(ctx.key.profile_id);
    let n = &ph.nodes[node as usize];
    let bounds = geom.sector_bounds(geom.sector_of(cell));
    let slot = perimeter_index(bounds, cell)?;
    if n.sector == ctx.key.goal_sector && n.region == ctx.key.goal_region {
        return Some(ctx.goal_boundary[slot]).filter(|v| *v != u64::MAX);
    }
    let sg = &ph.sectors[n.sector as usize];
    let rim = perimeter_len(bounds);
    let mut best: Option<u64> = None;
    for (k, &id) in ph.sector_nodes[n.sector as usize].iter().enumerate() {
        if ph.nodes[id as usize].region != n.region { continue; }
        let label = ctx.labels[id as usize];
        let b = sg.boundary_cost[k * rim + slot];
        if label == u64::MAX || b == u32::MAX { continue; }
        let v = label.saturating_add(b as u64);
        if best.map(|x| v < x).unwrap_or(true) { best = Some(v); }
    }
    best
}

/// Field recipe for a unit standing in (sector, region): the cache key, the
/// constant removed from the seeds (add it back for absolute cost-to-go) and
/// the best exit's far-side node (for prefetching and route display).
#[derive(Clone, Debug)]
pub struct FieldRecipe {
    pub key: FieldKey,
    pub offset: u64,
    pub best_exit_far_node: Option<u32>,
}

/// `target` is the exact stage target (used inside the goal region; the
/// context anchor is only an abstraction for sharing).
///
/// Exit seeds are per cell: the cell's directed crossing cost plus the cost
/// to go from the cell it crosses into (`cost_to_go_from_boundary`).
/// Seeding a whole portal run with the far node's label instead (measured
/// from the run's middle cell) sent units to the nearest cell of the run and
/// made long routes on the default map about 20% dearer than optimal.
pub fn field_recipe(h: &NavHierarchy, ctx: &DestContext, sector: u32, region: u16, target: CellPos) -> Option<FieldRecipe> {
    let geom = &h.terrain.geom;
    let ph = h.hierarchy(ctx.key.profile_id);
    let spos = SectorPos::new(sector as i32 % geom.sectors_x(), sector as i32 / geom.sectors_x());
    let sector_version = ph.sectors[sector as usize].version;
    let crowd = ctx.crowd.as_ref().map(|m| m.sector_levels(geom, spos)).unwrap_or_default();
    if sector == ctx.key.goal_sector && region == ctx.key.goal_region {
        return Some(FieldRecipe {
            key: FieldKey { profile_id: ctx.key.profile_id, sector: spos, sector_version,
                seeds: vec![GateSeed { cell: target, exit_dir: NO_EXIT, downstream_cost: 0 }], crowd },
            offset: 0, best_exit_far_node: None,
        });
    }
    let mut raw: Vec<(CellPos, u8, u64)> = Vec::new();
    let mut best: Option<(u64, u32)> = None;
    for &id in &ph.sector_nodes[sector as usize] {
        let node = &ph.nodes[id as usize];
        if node.region != region { continue; }
        let p = &ph.portals[node.portal as usize];
        let far = if node.a_side { p.node_b } else { p.node_a };
        let label = ctx.labels[far as usize];
        // Downhill-only exits: seed a crossing only if it strictly lowers the
        // abstract cost-to-go. Every finite-label region keeps at least one
        // such exit (Dijkstra optimality), and no unit can be sent back across
        // a portal it just used, because that would need both
        // label(a) < label(b) and label(b) < label(a).
        if label == u64::MAX || label >= ctx.labels[id as usize] { continue; }
        let (dx, dy) = if node.a_side { p.a_to_b() } else { let (x, y) = p.a_to_b(); (-x, -y) };
        let dir = NEIGHBORS4.iter().position(|&d| d == (dx, dy)).unwrap() as u8;
        for along in p.lo..=p.hi {
            let (near, far_cell) = if node.a_side { (p.a_cell(along), p.b_cell(along)) } else { (p.b_cell(along), p.a_cell(along)) };
            let down = cost_to_go_from_boundary(h, ctx, far, far_cell).unwrap_or(label);
            let v = traversal_cost_units(&h.terrain, near, dx, dy).saturating_add(down);
            raw.push((near, dir, v));
            if best.map(|(b, bn)| (v, far) < (b, bn)).unwrap_or(true) { best = Some((v, far)); }
        }
    }
    if raw.is_empty() { return None; }
    // One seed per cell (a corner cell can exit two ways): keep the cheapest.
    raw.sort_unstable_by_key(|&(c, d, v)| (c, v, d));
    raw.dedup_by_key(|s| s.0);
    let offset = raw.iter().map(|s| s.2).min().unwrap();
    let seeds = raw.into_iter().map(|(cell, exit_dir, v)| GateSeed {
        cell, exit_dir, downstream_cost: (v - offset).min(u32::MAX as u64) as u32,
    }).collect();
    Some(FieldRecipe {
        key: FieldKey { profile_id: ctx.key.profile_id, sector: spos, sector_version, seeds, crowd },
        offset, best_exit_far_node: best.map(|b| b.1),
    })
}

/// Follow next-hop pointers from a node, returning the (sector, region)
/// sequence entered, up to `max_sectors` entries. Used to prefetch fields
/// ahead of moving units.
pub fn regions_ahead(ph: &ProfileHierarchy, ctx: &DestContext, from_node: u32, max_sectors: usize) -> Vec<(u32, u16)> {
    let mut out = Vec::new();
    let mut n = from_node;
    let mut guard = 0;
    while out.len() < max_sectors && guard < 4096 {
        guard += 1;
        let node = &ph.nodes[n as usize];
        let key = (node.sector, node.region);
        if out.last() != Some(&key) { out.push(key); }
        let nx = ctx.next_hop[n as usize];
        if nx == u32::MAX { break; }
        n = nx;
    }
    out
}

/// Abstract route from a (sector, region) to the goal, as world points
/// through portal centres. Debug display only; never used by movement.
pub fn route_polyline(h: &NavHierarchy, ctx: &DestContext, from: CellPos, target: CellPos) -> Vec<WorldPos> {
    let geom = &h.terrain.geom;
    let ph = h.hierarchy(ctx.key.profile_id);
    let mut pts = vec![geom.cell_center(from)];
    if let Some((s, r)) = ph.region_at(geom, from) {
        if let Some(rec) = field_recipe(h, ctx, s, r, target) {
            if let Some(mut n) = rec.best_exit_far_node {
                let mut guard = 0;
                loop {
                    guard += 1;
                    if guard > 4096 { break; }
                    pts.push(ph.portals[ph.nodes[n as usize].portal as usize].center_world(geom));
                    let nx = ctx.next_hop[n as usize];
                    if nx == u32::MAX { break; }
                    // Skip the intra-sector hop; record each crossing once.
                    if ph.nodes[nx as usize].portal == ph.nodes[n as usize].portal { n = nx; continue; }
                    n = nx;
                }
            }
        }
    }
    pts.push(geom.cell_center(target));
    pts.dedup_by(|a, b| a.dist(*b) < 1e-3);
    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nav::{astar, build_flow_field, default_profiles, MovementProfile};
    use crate::rng::Rng;
    use crate::terrain::TestTerrain;
    use std::sync::Arc;

    /// Follow fields cell by cell from start to goal, exactly as units do,
    /// returning the summed directed edge cost. Panics on an invalid step.
    fn follow(h: &NavHierarchy, profile: u8, start: CellPos, goal: CellPos) -> Option<(u64, usize)> {
        let key = context_key(h, profile, goal)?;
        let ctx = build_context(h, &key, None)?;
        let grid = h.grid(profile);
        let t = &h.terrain;
        let mut at = start;
        let mut cost = 0u64;
        for steps in 0..100_000usize {
            if at == goal { return Some((cost, steps)); }
            assert!(grid.is_passable(t, at), "walked onto impassable {at:?}");
            let (s, r) = h.region_at(profile, at).unwrap();
            let rec = field_recipe(h, &ctx, s, r, goal)?;
            let f = build_flow_field(t, grid, rec.key)?;
            let (next, _, _) = f.best_step(t, grid, at)?;
            let (dx, dy) = (next.x - at.x, next.y - at.y);
            assert!(dx.abs() <= 1 && dy.abs() <= 1 && (dx, dy) != (0, 0));
            if dx != 0 && dy != 0 {
                assert!(grid.is_passable(t, CellPos::new(at.x + dx, at.y)) && grid.is_passable(t, CellPos::new(at.x, at.y + dy)), "corner cut");
            }
            cost += traversal_cost_units(t, at, dx, dy);
            at = next;
        }
        panic!("did not converge");
    }

    fn random_map(seed: u64, w: i32, h: i32, walls: usize) -> crate::terrain::Terrain {
        let mut tt = TestTerrain::new(w, h);
        let mut rng = Rng::new(seed);
        for i in 0..(w * h) as usize {
            tt.cost[i] = rng.range_i32(8, 20) as u8;
            tt.height[i] = rng.range_i32(0, 60) as i16;
        }
        for _ in 0..walls {
            let x = rng.range_i32(0, w); let y = rng.range_i32(0, h);
            if rng.chance(0.5) { tt.wall(x, y, (x + rng.range_i32(2, 14)).min(w - 1), y); }
            else { tt.wall(x, y, x, (y + rng.range_i32(2, 14)).min(h - 1)); }
        }
        tt.into_terrain(seed)
    }

    #[test]
    fn hierarchical_fields_match_reference_on_random_maps() {
        let profile = vec![MovementProfile { id: 0, name: "p".into(), clearance_cells: 0, max_slope: 1000, unit_radius: 0.4, base_speed: 1.0 }];
        let mut compared = 0;
        let mut worst_ratio = 1.0f64;
        for seed in 1..=12u64 {
            let t = Arc::new(random_map(seed, 48, 40, 40));
            let h = NavHierarchy::build(t.clone(), &profile);
            let grid = h.grid(0);
            let mut rng = Rng::new(seed * 77);
            for _ in 0..12 {
                let a = CellPos::new(rng.range_i32(0, 48), rng.range_i32(0, 40));
                let b = CellPos::new(rng.range_i32(0, 48), rng.range_i32(0, 40));
                if !grid.is_passable(&t, a) || !grid.is_passable(&t, b) { continue; }
                let reference = astar(&t, grid, a, b, false);
                let same_comp = h.component_at(0, a) == h.component_at(0, b);
                assert_eq!(reference.is_some(), same_comp, "component disagrees with reference for {a:?}->{b:?}");
                let Some(reference) = reference else { continue; };
                let (cost, _) = follow(&h, 0, a, b).expect("hierarchical route exists when reference does");
                assert!(cost >= reference.cost, "cheaper than optimal: {cost} < {}", reference.cost);
                let ratio = cost as f64 / reference.cost.max(1) as f64;
                worst_ratio = worst_ratio.max(ratio);
                assert!(ratio <= 1.6, "route {a:?}->{b:?} cost {cost} vs optimal {} (ratio {ratio:.2})", reference.cost);
                compared += 1;
            }
        }
        assert!(compared > 60, "only {compared} pairs compared");
        eprintln!("hierarchical vs A*: {compared} pairs, worst cost ratio {worst_ratio:.3}");
    }

    #[test]
    fn asymmetric_costs_are_respected_in_both_directions() {
        // A ramp rising to the east; uphill is surcharged, downhill is not.
        let mut tt = TestTerrain::new(40, 8);
        for y in 0..8 { for x in 0..40 { tt.height[(y * 40 + x) as usize] = (x * 12) as i16; } }
        let t = Arc::new(tt.into_terrain(1));
        let profile = vec![MovementProfile { id: 0, name: "p".into(), clearance_cells: 0, max_slope: 1000, unit_radius: 0.4, base_speed: 1.0 }];
        let h = NavHierarchy::build(t.clone(), &profile);
        let (a, b) = (CellPos::new(2, 4), CellPos::new(37, 4));
        let (up, _) = follow(&h, 0, a, b).unwrap();
        let (down, _) = follow(&h, 0, b, a).unwrap();
        let ref_up = astar(&t, h.grid(0), a, b, false).unwrap().cost;
        let ref_down = astar(&t, h.grid(0), b, a, false).unwrap().cost;
        assert!(up > down);
        // Exit seeds lean toward run middles (see `cost_to_go_from_boundary`):
        // here the route steps diagonally to the middle row and back, +80.
        assert!(up >= ref_up && up * 100 <= ref_up * 103, "uphill {up} vs optimal {ref_up}");
        assert!(down >= ref_down && down * 100 <= ref_down * 103, "downhill {down} vs optimal {ref_down}");
    }

    #[test]
    fn narrow_gap_passes_scout_but_not_heavy() {
        // 1-unit cells: scouts (radius <= 1.73) get clearance 2 and need a
        // 5-cell gap; heavies (radius <= 2.81) get clearance 3 and need 7.
        let profiles = default_profiles(1.0);
        assert_eq!((profiles[0].clearance_cells, profiles[1].clearance_cells), (2, 3));
        let mut tt = TestTerrain::new(48, 48);
        tt.wall(24, 0, 24, 20);
        tt.wall(24, 26, 24, 47);
        let t = Arc::new(tt.into_terrain(1));
        let h = NavHierarchy::build(t.clone(), &profiles);
        let (a, b) = (CellPos::new(8, 23), CellPos::new(40, 23));
        assert_eq!(h.component_at(0, a), h.component_at(0, b), "scout fits a 5-cell gap");
        assert!(follow(&h, 0, a, b).is_some());
        assert_ne!(h.component_at(1, a), h.component_at(1, b), "heavy must not fit a 5-cell gap");
        assert!(astar(&t, h.grid(1), a, b, false).is_none());
    }

    #[test]
    fn context_is_shared_for_nearby_destinations() {
        let t = Arc::new(TestTerrain::new(64, 64).into_terrain(1));
        let h = NavHierarchy::build(t, &default_profiles(1.0));
        let k1 = context_key(&h, 0, CellPos::new(41, 41)).unwrap();
        let k2 = context_key(&h, 0, CellPos::new(43, 42)).unwrap();
        let k3 = context_key(&h, 0, CellPos::new(52, 41)).unwrap();
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }
}
