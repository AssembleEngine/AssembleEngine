//! Sector / region / portal hierarchy (one per movement profile).
//!
//! * Each sector's clearance-passable cells are split into sector-local
//!   **regions** (8-connected inside the sector, same corner rule as A*).
//! * A **portal** is a maximal run of boundary cells where both sides are
//!   passable and the region on each side does not change. Every portal
//!   contributes two abstract **nodes**, one per side, at the run's middle.
//! * Directed **edges**: a crossing edge per direction (forward traversal cost
//!   of the representative cell), plus intra-sector edges between all nodes of
//!   the same region, costed by a forward Dijkstra inside the sector. Costs are
//!   therefore direction dependent end to end.
//! * Connectivity **components** come from a union-find over regions joined
//!   by portals. No per-cell flood fill of the whole map is needed, so a local
//!   terrain edit only recomputes the sectors it touches.
//!
//! Abstract costs are an approximation of grid costs (one representative cell
//! per portal); movement itself always follows exact per-sector fields.
use crate::math::{CellPos, SectorPos, WorldGeometry, WorldPos};
use crate::nav::base::{diagonal_corner_ok, traversal_cost_units, MovementProfile, ProfileGrid};
use crate::nav::fields::sector_dijkstra;
use crate::terrain::Terrain;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const NO_REGION: u16 = u16::MAX;

/// Ordered as scanned: vertical boundaries first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PortalAxis { Vertical, Horizontal }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Portal {
    pub axis: PortalAxis,
    /// Sector on the low-coordinate side (left or top) and its region.
    pub sector_a: u32,
    pub sector_b: u32,
    pub region_a: u16,
    pub region_b: u16,
    /// Vertical: x of B's first column. Horizontal: y of B's first row.
    pub boundary: i32,
    /// Inclusive run along y (vertical) or x (horizontal).
    pub lo: i32,
    pub hi: i32,
    pub node_a: u32,
    pub node_b: u32,
}

impl Portal {
    pub fn a_cell(&self, along: i32) -> CellPos {
        match self.axis {
            PortalAxis::Vertical => CellPos::new(self.boundary - 1, along),
            PortalAxis::Horizontal => CellPos::new(along, self.boundary - 1),
        }
    }
    pub fn b_cell(&self, along: i32) -> CellPos {
        match self.axis {
            PortalAxis::Vertical => CellPos::new(self.boundary, along),
            PortalAxis::Horizontal => CellPos::new(along, self.boundary),
        }
    }
    pub fn mid(&self) -> i32 { (self.lo + self.hi) / 2 }
    /// Cell on the given side where the straight line `from -> to` crosses
    /// this boundary line, clamped to `lo..=hi` along it (the run, or a
    /// stretch of touching runs). Integer arithmetic.
    pub fn crossing_cell(&self, a_side: bool, from: CellPos, to: CellPos, lo: i32, hi: i32) -> CellPos {
        let (f_n, f_a, t_n, t_a) = match self.axis {
            PortalAxis::Vertical => (from.x, from.y, to.x, to.y),
            PortalAxis::Horizontal => (from.y, from.x, to.y, to.x),
        };
        let normal = if a_side { self.boundary - 1 } else { self.boundary };
        let along = if t_n == f_n { t_a } else {
            f_a + ((normal - f_n) as i64 * (t_a - f_a) as i64 / (t_n - f_n) as i64) as i32
        };
        let along = along.clamp(lo, hi);
        if a_side { self.a_cell(along) } else { self.b_cell(along) }
    }
    /// Cardinal step from the A side to the B side.
    pub fn a_to_b(&self) -> (i32, i32) {
        match self.axis { PortalAxis::Vertical => (1, 0), PortalAxis::Horizontal => (0, 1) }
    }
    pub fn center_world(&self, geom: &WorldGeometry) -> WorldPos {
        let cs = geom.cell_size;
        let mid = (self.lo + self.hi + 1) as f32 * 0.5 * cs;
        match self.axis {
            PortalAxis::Vertical => WorldPos::new(self.boundary as f32 * cs, mid),
            PortalAxis::Horizontal => WorldPos::new(mid, self.boundary as f32 * cs),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AbsNode {
    pub sector: u32,
    pub region: u16,
    /// Representative cell (inside `sector`).
    pub cell: CellPos,
    pub portal: u32,
    pub a_side: bool,
}

/// Per-sector cached data. `version` changes whenever the sector's own cells
/// (or anything within the clearance influence margin) change; it is part of
/// every field cache key in this sector.
#[derive(Clone, Debug, Default)]
pub struct SectorGraph {
    pub version: u32,
    pub region_count: u16,
    /// (representative cell, region) per local node, used to reuse edges
    /// and boundary costs.
    pub signature: Vec<(CellPos, u16)>,
    /// Directed intra-sector edges (local from, local to, cost).
    pub edges: Vec<(u16, u16, u32)>,
    /// In-sector cost from every boundary cell to each local node's cell:
    /// `boundary_cost[node * perimeter + perimeter_index(cell)]`
    /// (`u32::MAX` = no in-sector path). Gives fields per-cell exit costs.
    /// Shared, so the per-repair copy of every sector graph stays cheap.
    pub boundary_cost: Arc<[u32]>,
}

/// Number of boundary slots of a sector (top row, bottom row, left column,
/// right column; corner cells appear twice).
pub fn perimeter_len(bounds: (i32, i32, i32, i32)) -> usize {
    let (x0, y0, x1, y1) = bounds;
    (2 * (x1 - x0) + 2 * (y1 - y0)) as usize
}

/// Boundary slot of a cell on the sector's edge (`None` for interior cells).
pub fn perimeter_index(bounds: (i32, i32, i32, i32), c: CellPos) -> Option<usize> {
    let (x0, y0, x1, y1) = bounds;
    let (w, h) = (x1 - x0, y1 - y0);
    if c.x < x0 || c.y < y0 || c.x >= x1 || c.y >= y1 { return None; }
    let i = if c.y == y0 { c.x - x0 }
        else if c.y == y1 - 1 { w + c.x - x0 }
        else if c.x == x0 { 2 * w + c.y - y0 }
        else if c.x == x1 - 1 { 2 * w + h + c.y - y0 }
        else { return None };
    Some(i as usize)
}

/// Offset into a sector-rect array (row-major) of boundary slot `k`.
pub fn perimeter_offset(bounds: (i32, i32, i32, i32), k: usize) -> usize {
    let (x0, y0, x1, y1) = bounds;
    let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let (x, y) = if k < w { (k, 0) } else if k < 2 * w { (k - w, h - 1) }
        else if k < 2 * w + h { (0, k - 2 * w) } else { (w - 1, k - 2 * w - h) };
    y * w + x
}

/// Cells of every boundary slot, in slot order.
fn perimeter_cells(bounds: (i32, i32, i32, i32)) -> Vec<CellPos> {
    let (x0, y0, x1, y1) = bounds;
    let mut out = Vec::with_capacity(perimeter_len(bounds));
    for x in x0..x1 { out.push(CellPos::new(x, y0)); }
    for x in x0..x1 { out.push(CellPos::new(x, y1 - 1)); }
    for y in y0..y1 { out.push(CellPos::new(x0, y)); }
    for y in y0..y1 { out.push(CellPos::new(x1 - 1, y)); }
    out
}

#[derive(Clone, Debug)]
pub struct ProfileHierarchy {
    pub profile_id: u8,
    /// Sector-local region per cell (`NO_REGION` if not passable).
    /// Sector-local region per cell (`NO_REGION` if not passable), one
    /// row-major array per sector. Shared between hierarchy versions: a local
    /// repair replaces only the dirty sectors' arrays.
    pub region_of: Vec<Arc<[u16]>>,
    /// Shared like `region_of`; replaced only when recomputed.
    pub sectors: Vec<Arc<SectorGraph>>,
    pub portals: Vec<Portal>,
    pub nodes: Vec<AbsNode>,
    pub sector_nodes: Vec<Vec<u32>>,
    pub out_edges: Vec<Vec<(u32, u32)>>,
    pub in_edges: Vec<Vec<(u32, u32)>>,
    pub region_base: Vec<u32>,
    pub region_comp: Vec<u32>,
    pub component_count: u32,
}

#[derive(Clone, Debug)]
pub struct NavHierarchy {
    pub terrain: Arc<Terrain>,
    /// Equals the terrain version this hierarchy was built from.
    pub version: u32,
    pub grids: Vec<ProfileGrid>,
    pub hierarchies: Vec<ProfileHierarchy>,
    pub build_ms: f64,
    /// Sectors recomputed by the most recent (re)build, for debug overlays.
    pub dirty_sectors: Vec<u32>,
    /// Sectors whose intra-sector edges were recomputed (vs reused).
    pub sectors_recomputed: usize,
    /// Computed once per (re)build; reading it is free.
    pub stats: NavStats,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NavStats {
    pub version: u32,
    pub portals: usize,
    pub nodes: usize,
    pub edges: usize,
    pub regions: usize,
    pub sectors: usize,
    pub build_ms: f64,
    pub pass_cells: usize,
    pub bytes: usize,
    pub sectors_recomputed: usize,
}

fn sector_pos(geom: &WorldGeometry, s: u32) -> SectorPos {
    SectorPos::new(s as i32 % geom.sectors_x(), s as i32 / geom.sectors_x())
}

/// Label sector-local regions: a row-major array over the sector (as
/// stored in `ProfileHierarchy::region_of`) and the region count.
fn label_sector_regions(terrain: &Terrain, grid: &ProfileGrid, sector: SectorPos) -> (Arc<[u16]>, u16) {
    let g = &terrain.geom;
    let (x0, y0, x1, y1) = g.sector_bounds(sector);
    let w = x1 - x0;
    let pass = |x: i32, y: i32| x >= x0 && y >= y0 && x < x1 && y < y1 && grid.passable[(y * g.cells_x + x) as usize] != 0;
    let local = |x: i32, y: i32| ((y - y0) * w + (x - x0)) as usize;
    let mut region_of = vec![NO_REGION; (w * (y1 - y0)).max(0) as usize];
    let mut count: u16 = 0;
    let mut queue: Vec<(i32, i32)> = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let i = local(x, y);
            if !pass(x, y) || region_of[i] != NO_REGION { continue; }
            let label = count;
            count += 1;
            region_of[i] = label;
            queue.clear();
            queue.push((x, y));
            let mut head = 0;
            while head < queue.len() {
                let (cx, cy) = queue[head]; head += 1;
                for &(dx, dy) in &crate::math::NEIGHBORS8 {
                    let (nx, ny) = (cx + dx, cy + dy);
                    if !pass(nx, ny) || !diagonal_corner_ok(pass, cx, cy, dx, dy) { continue; }
                    let ni = local(nx, ny);
                    if region_of[ni] == NO_REGION { region_of[ni] = label; queue.push((nx, ny)); }
                }
            }
        }
    }
    (region_of.into(), count)
}

/// Region of an in-bounds cell, from per-sector arrays.
#[inline]
fn region_in(region_of: &[Arc<[u16]>], g: &WorldGeometry, c: CellPos) -> u16 {
    let sp = g.sector_of(c);
    let (x0, y0, x1, _) = g.sector_bounds(sp);
    region_of[g.sector_index(sp)][((c.y - y0) * (x1 - x0) + (c.x - x0)) as usize]
}

/// Portals in (axis, boundary, run start) order: runs that touch along one
/// boundary line are adjacent in the result. A run never crosses from one
/// pair of sectors to the next, so each boundary line splits into segments
/// of one sector pair. Segments between two clean sectors are copied from
/// `previous` (their cells and regions did not change); the rest are
/// scanned. The result equals a full scan.
fn scan_portals(terrain: &Terrain, grid: &ProfileGrid, region_of: &[Arc<[u16]>], previous: Option<&[Portal]>, dirty: &[bool]) -> Vec<Portal> {
    let g = &terrain.geom;
    let sc = g.sector_cells;
    let mut out = Vec::new();
    let pass = |c: CellPos| grid.passable[g.cell_index(c)] != 0;
    let reg = |c: CellPos| region_in(region_of, g, c);
    let mut prev = previous.unwrap_or(&[]).iter().peekable();
    for axis in [PortalAxis::Vertical, PortalAxis::Horizontal] {
        let (count, len) = match axis {
            PortalAxis::Vertical => (g.sectors_x(), g.cells_y),
            PortalAxis::Horizontal => (g.sectors_y(), g.cells_x),
        };
        for k in 1..count {
            let boundary = k * sc;
            let cells = |along: i32| match axis {
                PortalAxis::Vertical => (CellPos::new(boundary - 1, along), CellPos::new(boundary, along)),
                PortalAxis::Horizontal => (CellPos::new(along, boundary - 1), CellPos::new(along, boundary)),
            };
            let mut seg_lo = 0;
            while seg_lo < len {
                let seg_hi = (seg_lo + sc).min(len);
                let (a, b) = cells(seg_lo);
                let (sa, sb) = (g.sector_index(g.sector_of(a)), g.sector_index(g.sector_of(b)));
                let same_segment = |p: &Portal| p.axis == axis && p.boundary == boundary && p.lo >= seg_lo && p.lo < seg_hi;
                // Previous portals are in the same order: skip any before
                // this segment, then take or drop the segment's own.
                while prev.peek().is_some_and(|p| !same_segment(p) && (p.axis, p.boundary, p.lo) < (axis, boundary, seg_lo)) { prev.next(); }
                if previous.is_some() && !dirty[sa] && !dirty[sb] {
                    while let Some(p) = prev.next_if(|p| same_segment(p)) { out.push(p.clone()); }
                    seg_lo = seg_hi;
                    continue;
                }
                while prev.next_if(|p| same_segment(p)).is_some() {}
                let mut along = seg_lo;
                while along < seg_hi {
                    let (a, b) = cells(along);
                    if !pass(a) || !pass(b) { along += 1; continue; }
                    let (ra, rb) = (reg(a), reg(b));
                    let lo = along;
                    let mut hi = along;
                    while hi + 1 < seg_hi {
                        let (a2, b2) = cells(hi + 1);
                        if !pass(a2) || !pass(b2) || reg(a2) != ra || reg(b2) != rb { break; }
                        hi += 1;
                    }
                    out.push(Portal {
                        axis, sector_a: sa as u32, sector_b: sb as u32,
                        region_a: ra, region_b: rb, boundary, lo, hi, node_a: 0, node_b: 0,
                    });
                    along = hi + 1;
                }
                seg_lo = seg_hi;
            }
        }
    }
    out
}

/// For local node `j` of a sector: the boundary costs to it (one value per
/// perimeter slot) and the directed edges into it from the other nodes of its
/// region, from one reverse Dijkstra inside the sector.
fn node_costs(terrain: &Terrain, grid: &ProfileGrid, sector: SectorPos, sig: &[(CellPos, u16)], j: usize) -> (Vec<u32>, Vec<(u16, u16, u32)>) {
    let g = &terrain.geom;
    let bounds = g.sector_bounds(sector);
    let (x0, y0, x1, _) = bounds;
    let w = x1 - x0;
    let (cell, region) = sig[j];
    let (dist, _) = sector_dijkstra(terrain, grid, sector, &[(cell, 0)], true);
    let at = |c: CellPos| dist[((c.y - y0) * w + (c.x - x0)) as usize];
    let boundary = perimeter_cells(bounds).iter().map(|&c| at(c).min(u32::MAX as u64) as u32).collect();
    let mut edges = Vec::new();
    for (i, &(other, oregion)) in sig.iter().enumerate() {
        if i == j || oregion != region { continue; }
        let d = at(other);
        if d != u64::MAX { edges.push((i as u16, j as u16, d.min(u32::MAX as u64 - 1) as u32)); }
    }
    (boundary, edges)
}

/// Run `f` over `items` on up to `threads` scoped threads. Results are
/// returned in item order, so the output is independent of thread count.
fn par_map<T: Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = threads.max(1).min(items.len().max(1));
    if threads <= 1 { return items.iter().map(&f).collect(); }
    let chunk = items.len().div_ceil(threads);
    crate::thread_load::idle(|| std::thread::scope(|scope| {
        let handles: Vec<_> = items.chunks(chunk).enumerate().map(|(i, c)| {
            let f = &f;
            scope.spawn(move || {
                crate::thread_load::register(&format!("hierarchy-helper-{i}"));
                let _b = crate::thread_load::busy();
                c.iter().map(f).collect::<Vec<R>>()
            })
        }).collect();
        handles.into_iter().flat_map(|h| h.join().expect("hierarchy worker panicked")).collect()
    }))
}

impl ProfileHierarchy {
    /// Build or locally repair. With `previous` and `dirty` (sector flags),
    /// regions are relabelled only in dirty sectors and intra-sector edges are
    /// reused for every sector that is clean and whose node signature did not
    /// change. Portal scanning and union-find are global but linear in the
    /// number of boundary cells / regions.
    fn build(terrain: &Terrain, grid: &ProfileGrid, previous: Option<&ProfileHierarchy>, dirty: &[bool], threads: usize) -> (Self, usize) {
        let g = &terrain.geom;
        let sector_count = (g.sectors_x() * g.sectors_y()) as usize;
        let mut region_of: Vec<Arc<[u16]>> = match previous { Some(p) => p.region_of.clone(), None => vec![Arc::from(Vec::new()); sector_count] };
        let mut sectors: Vec<Arc<SectorGraph>> = match previous { Some(p) => p.sectors.clone(), None => vec![Arc::new(SectorGraph::default()); sector_count] };
        let relabel: Vec<u32> = (0..sector_count as u32).filter(|s| previous.is_none() || dirty[*s as usize]).collect();
        let labels = par_map(&relabel, threads, |&s| label_sector_regions(terrain, grid, sector_pos(g, s)));
        for (&s, (regions, count)) in relabel.iter().zip(labels) {
            region_of[s as usize] = regions;
            let sg = Arc::make_mut(&mut sectors[s as usize]);
            sg.region_count = count;
            sg.version = sg.version.wrapping_add(1);
        }
        let mut portals = scan_portals(terrain, grid, &region_of, previous.map(|p| p.portals.as_slice()), dirty);
        let mut nodes = Vec::with_capacity(portals.len() * 2);
        let mut sector_nodes: Vec<Vec<u32>> = vec![Vec::new(); sector_count];
        for (pi, p) in portals.iter_mut().enumerate() {
            let mid = p.mid();
            p.node_a = nodes.len() as u32;
            nodes.push(AbsNode { sector: p.sector_a, region: p.region_a, cell: p.a_cell(mid), portal: pi as u32, a_side: true });
            p.node_b = nodes.len() as u32;
            nodes.push(AbsNode { sector: p.sector_b, region: p.region_b, cell: p.b_cell(mid), portal: pi as u32, a_side: false });
            sector_nodes[p.sector_a as usize].push(p.node_a);
            sector_nodes[p.sector_b as usize].push(p.node_b);
        }
        // Intra-sector edges: reuse where possible, recompute the rest in
        // parallel (each sector is independent).
        let signatures: Vec<Vec<(CellPos, u16)>> = sector_nodes.iter()
            .map(|ids| ids.iter().map(|&n| (nodes[n as usize].cell, nodes[n as usize].region)).collect()).collect();
        let todo: Vec<u32> = (0..sector_count as u32).filter(|&s| {
            let su = s as usize;
            previous.is_none() || dirty[su] || sectors[su].signature != signatures[su]
        }).collect();
        // One task per node, so a repair touching a few sectors still uses
        // every thread.
        let tasks: Vec<(u32, usize)> = todo.iter().flat_map(|&s| (0..signatures[s as usize].len()).map(move |j| (s, j))).collect();
        let costs = par_map(&tasks, threads, |&(s, j)| node_costs(terrain, grid, sector_pos(g, s), &signatures[s as usize], j));
        let mut costs = costs.into_iter();
        for &s in &todo {
            let sig = &signatures[s as usize];
            let mut boundary = Vec::new();
            let mut edges = Vec::new();
            for (b, e) in costs.by_ref().take(sig.len()) { boundary.extend(b); edges.extend(e); }
            edges.sort_unstable_by_key(|&(i, j, _)| (i, j));
            let sg = Arc::make_mut(&mut sectors[s as usize]);
            sg.edges = edges;
            sg.boundary_cost = boundary.into();
            sg.signature = sig.clone();
        }
        // Adjacency, one task per sector: each node's portal edge first, then
        // its sector's edges in (i, j) order.
        let sector_ids: Vec<u32> = (0..sector_count as u32).collect();
        let lists = par_map(&sector_ids, threads, |&s| {
            let ids = &sector_nodes[s as usize];
            let mut lists: Vec<(Vec<(u32, u32)>, Vec<(u32, u32)>)> = ids.iter().map(|&n| {
                let node = &nodes[n as usize];
                let p = &portals[node.portal as usize];
                let (dx, dy) = p.a_to_b();
                let ab = traversal_cost_units(terrain, nodes[p.node_a as usize].cell, dx, dy).min(u32::MAX as u64) as u32;
                let ba = traversal_cost_units(terrain, nodes[p.node_b as usize].cell, -dx, -dy).min(u32::MAX as u64) as u32;
                let (other, out_cost, in_cost) = if node.a_side { (p.node_b, ab, ba) } else { (p.node_a, ba, ab) };
                (vec![(other, out_cost)], vec![(other, in_cost)])
            }).collect();
            for &(i, j, c) in &sectors[s as usize].edges {
                lists[i as usize].0.push((ids[j as usize], c));
                lists[j as usize].1.push((ids[i as usize], c));
            }
            lists
        });
        let mut out_edges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nodes.len()];
        let mut in_edges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nodes.len()];
        for (s, lists) in lists.into_iter().enumerate() {
            for (k, (o, i)) in lists.into_iter().enumerate() {
                let n = sector_nodes[s][k] as usize;
                out_edges[n] = o;
                in_edges[n] = i;
            }
        }
        // Components: union-find over (sector, region) joined by portals.
        let mut region_base = Vec::with_capacity(sector_count + 1);
        let mut acc = 0u32;
        for sg in &sectors { region_base.push(acc); acc += sg.region_count as u32; }
        region_base.push(acc);
        let mut parent: Vec<u32> = (0..acc).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x { p[x as usize] = p[p[x as usize] as usize]; x = p[x as usize]; }
            x
        }
        for p in &portals {
            let a = find(&mut parent, region_base[p.sector_a as usize] + p.region_a as u32);
            let b = find(&mut parent, region_base[p.sector_b as usize] + p.region_b as u32);
            if a != b { let (lo, hi) = if a < b { (a, b) } else { (b, a) }; parent[hi as usize] = lo; }
        }
        let mut region_comp = vec![u32::MAX; acc as usize];
        let mut root_label = vec![u32::MAX; acc as usize];
        let mut component_count = 0u32;
        for r in 0..acc {
            let root = find(&mut parent, r);
            if root_label[root as usize] == u32::MAX { root_label[root as usize] = component_count; component_count += 1; }
            region_comp[r as usize] = root_label[root as usize];
        }
        let h = Self {
            profile_id: grid.profile.id, region_of, sectors, portals, nodes, sector_nodes,
            out_edges, in_edges, region_base, region_comp, component_count,
        };
        (h, todo.len())
    }

    #[inline]
    pub fn region_at(&self, geom: &WorldGeometry, c: CellPos) -> Option<(u32, u16)> {
        if !geom.in_bounds(c) { return None; }
        let r = region_in(&self.region_of, geom, c);
        if r == NO_REGION { None } else { Some((geom.sector_index(geom.sector_of(c)) as u32, r)) }
    }

    pub fn component_of(&self, sector: u32, region: u16) -> u32 {
        self.region_comp[(self.region_base[sector as usize] + region as u32) as usize]
    }

    pub fn bytes(&self) -> usize {
        self.region_of.iter().map(|r| r.len() * 2).sum::<usize>()
            + self.nodes.len() * std::mem::size_of::<AbsNode>()
            + self.portals.len() * std::mem::size_of::<Portal>()
            + self.out_edges.iter().chain(self.in_edges.iter()).map(|e| e.len() * 8 + 24).sum::<usize>()
            + self.sectors.iter().map(|s| s.edges.len() * 8 + s.signature.len() * 12 + s.boundary_cost.len() * 4 + 64).sum::<usize>()
            + self.region_comp.len() * 4 + self.region_base.len() * 4
    }
}

impl NavHierarchy {
    pub fn build(terrain: Arc<Terrain>, profiles: &[MovementProfile]) -> Self {
        Self::build_parallel(terrain, profiles, 1)
    }

    /// Full build. `threads` only affects speed: the output is identical.
    pub fn build_parallel(terrain: Arc<Terrain>, profiles: &[MovementProfile], threads: usize) -> Self {
        let start = std::time::Instant::now();
        let mut grids = Vec::with_capacity(profiles.len());
        let mut hierarchies = Vec::with_capacity(profiles.len());
        let sectors = (terrain.geom.sectors_x() * terrain.geom.sectors_y()) as usize;
        let mut recomputed = 0;
        for p in profiles {
            let grid = ProfileGrid::build(&terrain, p);
            let (h, n) = ProfileHierarchy::build(&terrain, &grid, None, &[], threads);
            recomputed += n;
            grids.push(grid);
            hierarchies.push(h);
        }
        let mut h = Self {
            version: terrain.version, terrain, grids, hierarchies,
            build_ms: start.elapsed().as_secs_f64() * 1000.0,
            dirty_sectors: (0..sectors as u32).collect(), sectors_recomputed: recomputed,
            stats: NavStats::default(),
        };
        h.stats = h.compute_stats();
        h
    }

    /// Local repair after a terrain edit whose changed cells lie inside the
    /// inclusive rect. Passability is recomputed only within the clearance
    /// influence margin; regions and intra-sector edges only for sectors that
    /// margin touches (plus sectors whose boundary nodes changed).
    pub fn rebuild_local(&self, terrain: Arc<Terrain>, rect: (i32, i32, i32, i32), threads: usize) -> Self {
        let start = std::time::Instant::now();
        let g = terrain.geom.clone();
        let (x0, y0, x1, y1) = rect;
        let sector_count = (g.sectors_x() * g.sectors_y()) as usize;
        // Profiles are independent: repair them at the same time.
        let profiles: Vec<usize> = (0..self.grids.len()).collect();
        let repaired = par_map(&profiles, profiles.len(), |&k| {
            let (grid, prev) = (&self.grids[k], &self.hierarchies[k]);
            let mut grid = grid.clone();
            let m = grid.influence_margin();
            let (ax0, ay0, ax1, ay1) = ((x0 - m).max(0), (y0 - m).max(0), (x1 + m).min(g.cells_x - 1), (y1 + m).min(g.cells_y - 1));
            grid.recompute_rect(&terrain, ax0, ay0, ax1, ay1);
            let mut dirty = vec![false; sector_count];
            // Field costs read the cell across a portal too, so widen by one.
            let (dx0, dy0, dx1, dy1) = ((ax0 - 1).max(0), (ay0 - 1).max(0), (ax1 + 1).min(g.cells_x - 1), (ay1 + 1).min(g.cells_y - 1));
            for sy in dy0 / g.sector_cells..=dy1 / g.sector_cells {
                for sx in dx0 / g.sector_cells..=dx1 / g.sector_cells { dirty[g.sector_index(SectorPos::new(sx, sy))] = true; }
            }
            let (h, n) = ProfileHierarchy::build(&terrain, &grid, Some(prev), &dirty, threads);
            (grid, h, n, dirty)
        });
        let mut grids = Vec::with_capacity(self.grids.len());
        let mut hierarchies = Vec::with_capacity(self.grids.len());
        let mut all_dirty = std::collections::BTreeSet::new();
        let mut recomputed = 0;
        for (grid, h, n, dirty) in repaired {
            all_dirty.extend((0..sector_count as u32).filter(|&s| dirty[s as usize]));
            recomputed += n;
            grids.push(grid);
            hierarchies.push(h);
        }
        let mut h = Self {
            version: terrain.version, terrain, grids, hierarchies,
            build_ms: start.elapsed().as_secs_f64() * 1000.0,
            dirty_sectors: all_dirty.into_iter().collect(), sectors_recomputed: recomputed,
            stats: NavStats::default(),
        };
        h.stats = h.compute_stats();
        h
    }

    fn profile_index(&self, profile_id: u8) -> usize {
        self.grids.iter().position(|g| g.profile.id == profile_id).unwrap_or(0)
    }
    pub fn grid(&self, profile_id: u8) -> &ProfileGrid { &self.grids[self.profile_index(profile_id)] }
    pub fn hierarchy(&self, profile_id: u8) -> &ProfileHierarchy { &self.hierarchies[self.profile_index(profile_id)] }

    /// Connected component of a clearance-passable cell for a profile.
    pub fn component_at(&self, profile_id: u8, c: CellPos) -> Option<u32> {
        let h = self.hierarchy(profile_id);
        let (s, r) = h.region_at(&self.terrain.geom, c)?;
        Some(h.component_of(s, r))
    }
    pub fn region_at(&self, profile_id: u8, c: CellPos) -> Option<(u32, u16)> {
        self.hierarchy(profile_id).region_at(&self.terrain.geom, c)
    }
    pub fn sector_version(&self, profile_id: u8, sector: SectorPos) -> u32 {
        self.hierarchy(profile_id).sectors[self.terrain.geom.sector_index(sector)].version
    }

    pub fn stats(&self) -> NavStats { self.stats.clone() }
    pub fn bytes(&self) -> usize { self.stats.bytes }

    fn compute_stats(&self) -> NavStats {
        NavStats {
            version: self.version,
            portals: self.hierarchies.iter().map(|h| h.portals.len()).sum(),
            nodes: self.hierarchies.iter().map(|h| h.nodes.len()).sum(),
            edges: self.hierarchies.iter().map(|h| h.out_edges.iter().map(|e| e.len()).sum::<usize>()).sum(),
            regions: self.hierarchies.iter().map(|h| h.region_comp.len()).sum(),
            sectors: (self.terrain.geom.sectors_x() * self.terrain.geom.sectors_y()) as usize,
            build_ms: self.build_ms,
            pass_cells: self.grids.iter().map(|g| g.passable_count).sum(),
            bytes: self.compute_bytes(),
            sectors_recomputed: self.sectors_recomputed,
        }
    }

    fn compute_bytes(&self) -> usize {
        self.grids.iter().map(|g| g.passable.len()).sum::<usize>() + self.hierarchies.iter().map(|h| h.bytes()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nav::default_profiles;
    use crate::terrain::TestTerrain;

    fn zero_clearance() -> Vec<MovementProfile> {
        vec![MovementProfile { id: 0, name: "pt".into(), clearance_cells: 0, max_slope: 1000, unit_radius: 0.4, base_speed: 10.0, medium: crate::terrain::Medium::ANY }]
    }

    #[test]
    fn sector_split_into_two_regions_gets_separate_nodes() {
        // A full-height wall inside sector column 1 (x=8..16) splits it into
        // two regions that are only connected through other sectors... and
        // here not at all, so they are different components too.
        let mut tt = TestTerrain::new(32, 16);
        tt.wall(12, 0, 12, 15);
        let t = Arc::new(tt.into_terrain(1));
        let h = NavHierarchy::build(t.clone(), &zero_clearance());
        let ph = h.hierarchy(0);
        let s = t.geom.sector_index(SectorPos::new(1, 0)) as u32;
        assert_eq!(ph.sectors[s as usize].region_count, 2);
        let left = h.region_at(0, CellPos::new(10, 4)).unwrap();
        let right = h.region_at(0, CellPos::new(14, 4)).unwrap();
        assert_eq!(left.0, right.0);
        assert_ne!(left.1, right.1);
        assert_ne!(h.component_at(0, CellPos::new(2, 2)), h.component_at(0, CellPos::new(30, 2)));
        // No intra edge may join nodes of different regions.
        for &(i, j, _) in &ph.sectors[s as usize].edges {
            let ni = ph.sector_nodes[s as usize][i as usize];
            let nj = ph.sector_nodes[s as usize][j as usize];
            assert_eq!(ph.nodes[ni as usize].region, ph.nodes[nj as usize].region);
        }
    }

    #[test]
    fn local_rebuild_matches_full_build() {
        let mut tt = TestTerrain::new(64, 64);
        tt.wall(20, 0, 22, 40);
        let base = Arc::new(tt.into_terrain(1));
        let profiles = default_profiles(1.0);
        let h0 = NavHierarchy::build(base.clone(), &profiles);
        let mut patch = crate::terrain::TerrainPatch::new();
        for y in 30..34 { for x in 40..50 { patch.push(crate::terrain::CellEdit { x, y, blocked: Some(true), height: None, cost: None, material: None }); } }
        patch.finish();
        let (t2, applied) = base.with_patch(patch);
        let t2 = Arc::new(t2);
        let rect = applied.dirty_cells().unwrap();
        let local = h0.rebuild_local(t2.clone(), rect, 2);
        let full = NavHierarchy::build(t2.clone(), &profiles);
        assert!(local.sectors_recomputed < full.sectors_recomputed);
        for (a, b) in local.hierarchies.iter().zip(&full.hierarchies) {
            assert_eq!(a.region_of, b.region_of);
            assert_eq!(a.portals.len(), b.portals.len());
            assert_eq!(a.out_edges, b.out_edges);
            assert_eq!(a.region_comp, b.region_comp);
            for (sa, sb) in a.sectors.iter().zip(&b.sectors) { assert_eq!(sa.boundary_cost, sb.boundary_cost); }
        }
        for (a, b) in local.grids.iter().zip(&full.grids) { assert_eq!(a.passable, b.passable); }
    }

    /// Everything a full build produces, for exact comparison.
    fn assert_same(local: &NavHierarchy, full: &NavHierarchy) {
        for (a, b) in local.hierarchies.iter().zip(&full.hierarchies) {
            assert_eq!(a.region_of, b.region_of);
            assert_eq!(format!("{:?}", a.portals), format!("{:?}", b.portals));
            assert_eq!(format!("{:?}", a.nodes), format!("{:?}", b.nodes));
            assert_eq!(a.out_edges, b.out_edges);
            assert_eq!(a.in_edges, b.in_edges);
            assert_eq!(a.region_comp, b.region_comp);
            for (sa, sb) in a.sectors.iter().zip(&b.sectors) {
                assert_eq!((sa.region_count, &sa.edges, &sa.boundary_cost), (sb.region_count, &sb.edges, &sb.boundary_cost));
            }
        }
        for (a, b) in local.grids.iter().zip(&full.grids) {
            assert_eq!(a.passable, b.passable);
            assert_eq!(a.passable_count, b.passable_count);
        }
    }

    #[test]
    fn chained_local_repairs_match_full_builds() {
        // Repairs build on previous repairs (reusing their portals and
        // sector data), so compare after each of a series of random edits:
        // walls, openings, costs and heights, some across sector boundaries.
        // 99 × 83 with 8-cell sectors: the last column and row of sectors
        // are partial.
        let mut tt = TestTerrain::new(99, 83);
        tt.wall(30, 0, 31, 50);
        tt.wall(0, 40, 60, 40);
        let profiles = default_profiles(1.0);
        let mut t = Arc::new(tt.into_terrain(1));
        let mut h = NavHierarchy::build(t.clone(), &profiles);
        let mut rng = crate::rng::Rng::new(7);
        let mut repairs = 0;
        for _ in 0..24 {
            let (x, y) = (rng.range_i32(0, 99), rng.range_i32(0, 83));
            let (w, hh) = (rng.range_i32(1, 12), rng.range_i32(1, 12));
            let kind = rng.range_i32(0, 4);
            let mut patch = crate::terrain::TerrainPatch::new();
            for yy in y..(y + hh).min(83) { for xx in x..(x + w).min(99) {
                let (blocked, cost, height) = match kind {
                    0 => (Some(true), None, None),
                    1 => (Some(false), Some(10), None),
                    2 => (None, Some(rng.range_i32(8, 20) as u8), None),
                    _ => (None, None, Some(rng.range_i32(0, 400) as i16)),
                };
                patch.push(crate::terrain::CellEdit { x: xx, y: yy, blocked, height, cost, material: None });
            }}
            patch.finish();
            let (t2, applied) = t.with_patch(patch);
            t = Arc::new(t2);
            let Some(rect) = applied.dirty_cells() else { continue };
            h = h.rebuild_local(t.clone(), rect, 3);
            assert_same(&h, &NavHierarchy::build(t.clone(), &profiles));
            repairs += 1;
        }
        assert!(repairs >= 20, "only {repairs} edits changed the terrain");
    }

    #[test]
    fn parallel_build_is_identical() {
        let mut tt = TestTerrain::new(80, 60);
        tt.wall(39, 0, 41, 20);
        let t = Arc::new(tt.into_terrain(1));
        let a = NavHierarchy::build_parallel(t.clone(), &default_profiles(1.0), 1);
        let b = NavHierarchy::build_parallel(t, &default_profiles(1.0), 7);
        for (x, y) in a.hierarchies.iter().zip(&b.hierarchies) {
            assert_eq!(x.out_edges, y.out_edges);
            for (sa, sb) in x.sectors.iter().zip(&y.sectors) { assert_eq!(sa.boundary_cost, sb.boundary_cost); }
        }
    }
}
