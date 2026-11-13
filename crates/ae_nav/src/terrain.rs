//! Terrain generation and versioned editing.
//!
//! `Terrain` is immutable once wrapped in an `Arc`. An edit creates a new
//! version; navigation workers never observe a partially mutated terrain.
use crate::math::{CellPos, Rect, WorldGeometry, WorldPos};
use crate::rng::{fbm, hash2, Rng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const TERRAIN_VERSION_INITIAL: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TerrainConfig {
    pub seed: u64,
    pub geom: WorldGeometry,
}

impl TerrainConfig {
    pub fn new(seed: u64, cells: i32, cell_size: f32, sector_cells: i32) -> Self {
        Self { seed, geom: WorldGeometry::new(cells, cells, cell_size, sector_cells) }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Terrain {
    pub geom: WorldGeometry,
    pub seed: u64,
    pub version: u32,
    /// 0 = open, 1 = solid rock/water obstacle.
    pub blocked: Vec<u8>,
    /// Decimetres above an arbitrary datum, 0..=2400.
    pub height: Vec<i16>,
    /// Integer traversal cost in tenths (10 = ordinary ground).
    pub move_cost: Vec<u8>,
    /// Palette / material id for display only.
    pub material: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Material {
    Grass = 0,
    Forest = 1,
    Sand = 2,
    Rock = 3,
    Water = 4,
    Road = 5,
    Mud = 6,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellEdit {
    pub x: i32,
    pub y: i32,
    pub blocked: Option<bool>,
    pub height: Option<i16>,
    pub cost: Option<u8>,
    pub material: Option<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TerrainPatch {
    pub version: u32,
    pub cells: Vec<CellEdit>,
}

impl TerrainPatch {
    pub fn new() -> Self { Self { version: 0, cells: Vec::new() } }
    pub fn push(&mut self, e: CellEdit) { self.cells.push(e); }
    /// Sort and collapse duplicate cells. Later values win, deterministically.
    pub fn finish(&mut self) {
        self.cells.sort_by_key(|e| (e.y, e.x));
        let mut out: Vec<CellEdit> = Vec::with_capacity(self.cells.len());
        for e in self.cells.drain(..) {
            if let Some(prev) = out.last_mut() {
                if prev.x == e.x && prev.y == e.y {
                    if e.blocked.is_some() { prev.blocked = e.blocked; }
                    if e.height.is_some() { prev.height = e.height; }
                    if e.cost.is_some() { prev.cost = e.cost; }
                    if e.material.is_some() { prev.material = e.material; }
                    continue;
                }
            }
            out.push(e);
        }
        self.cells = out;
    }
    pub fn dirty_cells(&self) -> Option<(i32, i32, i32, i32)> {
        if self.cells.is_empty() { return None; }
        let mut x0 = i32::MAX; let mut y0 = i32::MAX;
        let mut x1 = i32::MIN; let mut y1 = i32::MIN;
        for c in &self.cells {
            x0 = x0.min(c.x); y0 = y0.min(c.y);
            x1 = x1.max(c.x); y1 = y1.max(c.y);
        }
        Some((x0, y0, x1, y1))
    }
    pub fn dirty_world_rect(&self, geom: &WorldGeometry) -> Option<Rect> {
        self.dirty_cells().map(|(x0, y0, x1, y1)| Rect {
            min_x: x0 as f32 * geom.cell_size,
            min_y: y0 as f32 * geom.cell_size,
            max_x: (x1 + 1) as f32 * geom.cell_size,
            max_y: (y1 + 1) as f32 * geom.cell_size,
        })
    }
}

impl Terrain {
    pub fn generate(cfg: &TerrainConfig) -> Self {
        let g = &cfg.geom;
        let n = (g.cells_x * g.cells_y) as usize;
        let mut t = Self {
            geom: g.clone(), seed: cfg.seed, version: TERRAIN_VERSION_INITIAL,
            blocked: vec![0; n], height: vec![0; n], move_cost: vec![10; n],
            material: vec![Material::Grass as u8; n],
        };
        let mut rng = Rng::new(cfg.seed);
        let water_seed = rng.next_u64();
        let forest_seed = rng.next_u64();
        let rock_seed = rng.next_u64();
        for y in 0..g.cells_y {
            for x in 0..g.cells_x {
                let i = (y * g.cells_x + x) as usize;
                let wx = x as f32; let wy = y as f32;
                let macro_n = fbm(cfg.seed, wx / 720.0, wy / 720.0, 3, 2.0, 0.52);
                let ridge = 1.0 - (fbm(rock_seed, wx / 260.0, wy / 260.0, 3, 2.1, 0.5) * 2.0 - 1.0).abs();
                let detail = fbm(forest_seed, wx / 58.0, wy / 58.0, 2, 2.0, 0.5);
                let h = 120.0 + macro_n * 920.0 + ridge * 520.0 + detail * 170.0;
                let h = h.round().clamp(0.0, 2400.0) as i16;
                t.height[i] = h;
                let moist = fbm(water_seed, wx / 410.0 + 31.0, wy / 410.0 - 17.0, 3, 2.0, 0.55);
                if h < 168 {
                    t.blocked[i] = 1;
                    t.material[i] = Material::Water as u8;
                    t.move_cost[i] = 255;
                } else if h < 198 {
                    t.material[i] = Material::Sand as u8;
                    t.move_cost[i] = 13;
                } else if moist > 0.60 {
                    t.material[i] = Material::Forest as u8;
                    t.move_cost[i] = 15;
                } else if moist < 0.36 {
                    t.material[i] = Material::Mud as u8;
                    t.move_cost[i] = 17;
                }
            }
        }
        // Natural (noise) heights. Roads and plazas restore these instead of
        // forcing constant heights, which used to leave slope cliffs along
        // their edges and cut the spawn area off from the rest of the map.
        let natural_height = t.height.clone();
        // Hand-placed features are authored for a 2560 x 2560 map and scaled
        // to the configured size; widths that matter for clearance (walls,
        // passages, roads) stay in absolute cells.
        let (cx, cy) = (g.cells_x, g.cells_y);
        let sx = |v: i32| v * cx / 2560;
        let sy = |v: i32| v * cy / 2560;
        let sr = |v: i32| (v * cx.min(cy) / 2560).max(2);
        let spans = |v: &[(i32, i32)], f: &dyn Fn(i32) -> i32| v.iter().map(|&(a, b)| (f(a), f(b).max(f(a) + 3))).collect::<Vec<_>>();
        // A few irregular lake basins, independent of the global noise field.
        stamp_ellipse(&mut t, sx(1330), sy(1620), sr(92), sr(56), sr(27), water_seed);
        stamp_ellipse(&mut t, sx(520), sy(470), sr(54), sr(34), sr(22), rock_seed);
        stamp_ellipse(&mut t, sx(2200), sy(1760), sr(78), sr(45), sr(25), water_seed ^ 9);
        stamp_ellipse(&mut t, sx(1550), sy(430), sr(55), sr(38), sr(18), rock_seed ^ 4);
        stamp_ellipse(&mut t, sx(1930), sy(650), sr(40), sr(26), sr(14), rock_seed ^ 7);
        stamp_ellipse(&mut t, sx(340), sy(1950), sr(32), sr(22), sr(18), rock_seed ^ 3);
        // Broken fortification walls make deliberate chokepoints and force
        // routes to make meaningful decisions. Gaps are generated at fixed
        // positions so every scenario is reproducible.
        stamp_vwall(&mut t, sx(820), sy(160), sy(2360), 7, &spans(&[(590, 638), (1420, 1450), (1986, 2014)], &sy));
        stamp_hwall(&mut t, sy(1280), sx(1010), sx(2240), 6, &spans(&[(1180, 1220), (1690, 1722)], &sx));
        stamp_vwall(&mut t, sx(1770), sy(720), sy(2100), 5, &spans(&[(1200, 1260)], &sy));
        stamp_hwall(&mut t, sy(2190), sx(300), sx(900), 5, &spans(&[(620, 650)], &sx));
        // Fully enclosed rock chambers: their floors are disconnected regions.
        stamp_ring(&mut t, sx(2050), sy(1190), sr(58), sr(47), 6);
        stamp_ring(&mut t, sx(1850), sy(2130), sr(36), sr(30), 5);
        // Rock outcrops split by a narrow 3-cell passage: scouts fit, heavy
        // units (5-cell clearance square) do not.
        for &(px, py, r) in &[(1040, 920, 44), (1130, 2220, 48), (690, 1280, 34), (2300, 430, 38)] {
            let (px, py, r) = (sx(px), sy(py), sr(r));
            stamp_ellipse(&mut t, px, py, r, r * 2 / 3, sr(13), rock_seed ^ (px as u64));
            for dy in -(r * 2 / 3) - sr(13)..=(r * 2 / 3) + sr(13) {
                for dx in 0..=2 {
                    let Some(i) = idx(g, px + 2 + dx, py + dy) else { continue };
                    t.blocked[i] = 0;
                    t.material[i] = Material::Sand as u8;
                    t.move_cost[i] = 12;
                    t.height[i] = natural_height[i];
                }
            }
        }
        // Clear reproducible staging and destination plazas so benchmarks do
        // not depend on a rare noisy starting cell.
        clear_plaza(&mut t, &natural_height, sx(250), sy(2300), sr(64));
        clear_plaza(&mut t, &natural_height, sx(2290), sy(300), sr(64));
        // Carve a broad road: cheaper terrain but never special-cased in
        // navigation beyond the ordinary weighted grid.
        let road: Vec<(i32, i32)> = [(250, 2300), (700, 2100), (1020, 1900), (1400, 1560), (1740, 1100), (2050, 650), (2290, 300)]
            .iter().map(|&(x, y)| (sx(x), sy(y))).collect();
        stamp_road(&mut t, &natural_height, &road, 5);
        t
    }

    pub fn len(&self) -> usize { self.blocked.len() }
    pub fn is_blocked(&self, c: CellPos) -> bool {
        self.geom.in_bounds(c) && self.blocked[self.geom.cell_index(c)] != 0
    }
    pub fn in_bounds_blocked_or_out(&self, c: CellPos) -> bool { !self.geom.in_bounds(c) || self.is_blocked(c) }
    pub fn height_at(&self, c: CellPos) -> i16 { self.height[self.geom.cell_index(c)] }
    pub fn cost_at(&self, c: CellPos) -> u8 { self.move_cost[self.geom.cell_index(c)] }
    pub fn material_at(&self, c: CellPos) -> u8 { self.material[self.geom.cell_index(c)] }
    pub fn world_of_cell(&self, c: CellPos) -> WorldPos { self.geom.cell_center(c) }
    pub fn cell_of_world(&self, p: WorldPos) -> CellPos { self.geom.world_to_cell(p) }

    /// Apply an edit against this immutable base, producing the next version.
    pub fn with_patch(&self, patch: TerrainPatch) -> (Self, TerrainPatch) {
        let mut next = self.clone();
        let patch = next.apply_patch(patch);
        (next, patch)
    }

    /// Apply `patch` in place as the next version; returns it stamped with
    /// that version.
    pub fn apply_patch(&mut self, mut patch: TerrainPatch) -> TerrainPatch {
        self.version += 1;
        patch.version = self.version;
        for e in &patch.cells {
            if e.x < 0 || e.y < 0 || e.x >= self.geom.cells_x || e.y >= self.geom.cells_y { continue; }
            let i = (e.y * self.geom.cells_x + e.x) as usize;
            if let Some(b) = e.blocked { self.blocked[i] = b as u8; }
            if let Some(h) = e.height { self.height[i] = h.clamp(0, 2400); }
            if let Some(c) = e.cost { self.move_cost[i] = c; }
            if let Some(m) = e.material { self.material[i] = m; }
            if e.blocked == Some(true) {
                self.material[i] = Material::Rock as u8;
                self.move_cost[i] = 255;
            }
            if e.blocked == Some(false) && self.move_cost[i] == 255 {
                self.move_cost[i] = 10;
                if self.material[i] == Material::Rock as u8 { self.material[i] = Material::Grass as u8; }
            }
        }
        patch
    }

    /// Deterministic digest of authoritative terrain, for replay checks:
    /// the version and the digest of every band of `DIGEST_BAND_ROWS` rows.
    /// Callers cache band digests so an edit rehashes only its own bands.
    pub fn digest(&self) -> u64 {
        let bands: Vec<u64> = (0..self.digest_bands()).map(|b| self.band_digest(b)).collect();
        Self::digest_of_bands(self.version, &bands)
    }

    pub fn digest_bands(&self) -> usize { (self.geom.cells_y.max(0) as usize).div_ceil(DIGEST_BAND_ROWS) }

    /// The band holding row `y`.
    pub fn digest_band_of(&self, y: i32) -> usize { (y.clamp(0, self.geom.cells_y - 1) as usize) / DIGEST_BAND_ROWS }

    pub fn band_digest(&self, band: usize) -> u64 {
        let w = self.geom.cells_x.max(0) as usize;
        let lo = band * DIGEST_BAND_ROWS * w;
        let hi = (lo + DIGEST_BAND_ROWS * w).min(self.blocked.len());
        let mut h = crate::rng::StateHasher::new();
        for i in lo..hi {
            h.u64(((self.blocked[i] as u64) << 40) | ((self.move_cost[i] as u64) << 16) | (self.height[i] as u16 as u64));
        }
        h.finish()
    }

    pub fn digest_of_bands(version: u32, bands: &[u64]) -> u64 {
        let mut h = crate::rng::StateHasher::new();
        h.u32(version);
        for &b in bands { h.u64(b); }
        h.finish()
    }

    pub fn nearest_open_cell(&self, c: CellPos, max_radius: i32, accept: impl Fn(CellPos) -> bool) -> Option<CellPos> {
        let c = self.geom.clamp_cell(c);
        if accept(c) { return Some(c); }
        for r in 1..=max_radius {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs() != r && dy.abs() != r { continue; }
                    let p = CellPos::new(c.x + dx, c.y + dy);
                    if self.geom.in_bounds(p) && accept(p) { return Some(p); }
                }
            }
        }
        None
    }
}

fn idx(g: &WorldGeometry, x: i32, y: i32) -> Option<usize> { g.cell_index_xy(x, y) }

/// Rows per terrain digest band.
pub const DIGEST_BAND_ROWS: usize = 16;

fn paint_block(t: &mut Terrain, x: i32, y: i32, blocked: bool) {
    if let Some(i) = idx(&t.geom, x, y) {
        t.blocked[i] = blocked as u8;
        if blocked {
            t.material[i] = Material::Rock as u8;
            t.move_cost[i] = 255;
            t.height[i] = t.height[i].max(620);
        } else {
            if t.material[i] == Material::Rock as u8 || t.material[i] == Material::Water as u8 {
                t.material[i] = Material::Sand as u8;
                t.move_cost[i] = 11;
            }
        }
    }
}

fn stamp_ellipse(t: &mut Terrain, cx: i32, cy: i32, rx: i32, ry: i32, rough: i32, seed: u64) {
    for y in cy - ry - rough..=cy + ry + rough {
        for x in cx - rx - rough..=cx + rx + rough {
            if idx(&t.geom, x, y).is_none() { continue; }
            let nx = (x - cx) as f32 / rx as f32;
            let ny = (y - cy) as f32 / ry as f32;
            let d = nx * nx + ny * ny;
            let jitter = (hash2(seed, x, y) as f32 / u32::MAX as f32 - 0.5) * 0.16;
            if d < 1.0 + jitter {
                paint_block(t, x, y, true);
                // Lake material for very low basins.
                let i = t.geom.cell_index_xy(x, y).unwrap();
                if t.height[i] < 380 {
                    t.material[i] = Material::Water as u8;
                }
            }
        }
    }
}

fn stamp_ring(t: &mut Terrain, cx: i32, cy: i32, rx: i32, ry: i32, thickness: i32) {
    for y in cy - ry - 2..=cy + ry + 2 {
        for x in cx - rx - 2..=cx + rx + 2 {
            let nx = (x - cx) as f32 / rx as f32;
            let ny = (y - cy) as f32 / ry as f32;
            let d = (nx * nx + ny * ny).sqrt();
            let ring_width = thickness as f32 / rx.min(ry) as f32;
            if (d - 1.0).abs() < ring_width {
                paint_block(t, x, y, true);
            }
        }
    }
    // The interior stays open but is sealed by the ring.
}

fn stamp_vwall(t: &mut Terrain, x0: i32, y0: i32, y1: i32, thickness: i32, gaps: &[(i32, i32)]) {
    for y in y0..=y1 {
        if gaps.iter().any(|&(a, b)| y >= a && y <= b) { continue; }
        let wobble = ((y / 43) % 3) - 1;
        for x in x0 + wobble - thickness / 2..=x0 + wobble + thickness / 2 {
            paint_block(t, x, y, true);
        }
    }
}

fn stamp_hwall(t: &mut Terrain, y0: i32, x0: i32, x1: i32, thickness: i32, gaps: &[(i32, i32)]) {
    for x in x0..=x1 {
        if gaps.iter().any(|&(a, b)| x >= a && x <= b) { continue; }
        let wobble = ((x / 37) % 3) - 1;
        for y in y0 + wobble - thickness / 2..=y0 + wobble + thickness / 2 {
            paint_block(t, x, y, true);
        }
    }
}

fn clear_plaza(t: &mut Terrain, natural_height: &[i16], cx: i32, cy: i32, r: i32) {
    for y in cy - r..=cy + r {
        for x in cx - r..=cx + r {
            if (x - cx).pow(2) + (y - cy).pow(2) <= r * r {
                if let Some(i) = idx(&t.geom, x, y) {
                    t.blocked[i] = 0;
                    t.move_cost[i] = 10;
                    t.material[i] = Material::Grass as u8;
                    t.height[i] = natural_height[i];
                }
            }
        }
    }
}

fn stamp_road(t: &mut Terrain, natural_height: &[i16], points: &[(i32, i32)], half_width: i32) {
    // The road is also the generator's guaranteed-safe trunk corridor. It
    // clears walls/water rather than merely discounting them, so the staging
    // and destination plazas are always connected for the default scenario.
    // Heights follow the natural terrain so the road joins its surroundings
    // without slope cliffs.
    for w in points.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        let steps = ((x1 - x0).abs()).max((y1 - y0).abs()).max(1);
        for s in 0..=steps {
            let x = x0 + (x1 - x0) * s / steps;
            let y = y0 + (y1 - y0) * s / steps;
            for dy in -half_width..=half_width {
                for dx in -half_width..=half_width {
                    if let Some(i) = idx(&t.geom, x + dx, y + dy) {
                        if dx * dx + dy * dy > (half_width + 1) * (half_width + 1) { continue; }
                        t.blocked[i] = 0;
                        t.move_cost[i] = 8;
                        t.material[i] = Material::Road as u8;
                        t.height[i] = natural_height[i];
                    }
                }
            }
        }
    }
}

/// Small mutable map builder for unit tests.
pub struct TestTerrain {
    pub geom: WorldGeometry,
    pub blocked: Vec<u8>,
    pub height: Vec<i16>,
    pub cost: Vec<u8>,
}

impl TestTerrain {
    pub fn new(w: i32, h: i32) -> Self {
        Self {
            geom: WorldGeometry::new(w, h, 1.0, 8),
            blocked: vec![0; (w * h) as usize],
            height: vec![0; (w * h) as usize],
            cost: vec![10; (w * h) as usize],
        }
    }
    pub fn wall(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) {
        for y in y0..=y1 {
            for x in x0..=x1 {
                if let Some(i) = self.geom.cell_index_xy(x, y) { self.blocked[i] = 1; }
            }
        }
    }
    pub fn into_terrain(self, seed: u64) -> Terrain {
        let material = self.cost.iter().map(|_| Material::Grass as u8).collect();
        Terrain {
            geom: self.geom, seed, version: 1, blocked: self.blocked,
            height: self.height, move_cost: self.cost, material,
        }
    }
}

pub type SharedTerrain = Arc<Terrain>;

/// Map a cell given in the 2560 x 2560 reference layout (where scenario
/// features are authored) onto a map of another size.
pub fn reference_cell(geom: &WorldGeometry, x: i32, y: i32) -> CellPos {
    geom.clamp_cell(CellPos::new(x * geom.cells_x / 2560, y * geom.cells_y / 2560))
}
