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
    /// How the generator treats low ground and lake basins.
    #[serde(default)]
    pub water: WaterMode,
}

impl TerrainConfig {
    pub fn new(seed: u64, cells: i32, cell_size: f32, sector_cells: i32) -> Self {
        Self {
            seed,
            geom: WorldGeometry::new(cells, cells, cell_size, sector_cells),
            water: WaterMode::Obstacle,
        }
    }
}

/// Water in generated terrain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaterMode {
    /// pathsim's behaviour: low ground and lakes are solid obstacles; no
    /// water level. Every unit is a land unit.
    #[default]
    Obstacle,
    /// Low ground and the two lake basins lie under a water surface at
    /// `SURFACE_WATER_LEVEL`; they are open cells, and movement profiles
    /// decide by depth who may enter them (ships, hover, amphibious).
    Surface,
}

/// Water surface height (decimetres) of `WaterMode::Surface` maps: the
/// height below which pathsim's generator made cells water.
pub const SURFACE_WATER_LEVEL: i16 = 168;

/// `Terrain::water_level` of a map without water.
pub const NO_WATER: i16 = i16::MIN;

/// Which water depths a unit may stand in, and whether it moves on the
/// surface (ships, hover) or on the ground (land, amphibious).
///
/// Depth is decimetres of water above the ground; dry cells have depth 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Medium {
    pub min_depth: i16,
    pub max_depth: i16,
    /// Floats over water of any depth (and still crosses dry land).
    pub hover: bool,
}

impl Medium {
    /// Anything that is not an obstacle: pathsim's units.
    pub const ANY: Medium = Medium {
        min_depth: 0,
        max_depth: i16::MAX,
        hover: false,
    };
    /// Ground units that ford water up to `max_depth`.
    pub const fn land(max_depth: i16) -> Medium {
        Medium {
            min_depth: 0,
            max_depth,
            hover: false,
        }
    }
    /// Ships: water at least `min_depth` deep.
    pub const fn ship(min_depth: i16) -> Medium {
        Medium {
            min_depth,
            max_depth: i16::MAX,
            hover: false,
        }
    }
    /// Hovercraft: land and water of any depth.
    pub const HOVER: Medium = Medium {
        min_depth: 0,
        max_depth: i16::MAX,
        hover: true,
    };
    /// Amphibious: drives along the bottom at any depth.
    pub const AMPHIBIOUS: Medium = Medium {
        min_depth: 0,
        max_depth: i16::MAX,
        hover: false,
    };

    #[inline]
    pub fn allows(&self, depth: i16) -> bool {
        self.hover || (depth >= self.min_depth && depth <= self.max_depth)
    }
    /// Moves on the water surface, so slopes and climbs are measured there.
    #[inline]
    pub fn on_surface(&self) -> bool {
        self.hover || self.min_depth > 0
    }
}

impl Default for Medium {
    fn default() -> Self {
        Medium::ANY
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Terrain {
    pub geom: WorldGeometry,
    pub seed: u64,
    pub version: u32,
    /// 0 = open, 1 = solid rock/water obstacle.
    pub blocked: Vec<u8>,
    /// What a unit stands on, in decimetres above an arbitrary datum: the
    /// ground (0..=2400), or the water surface where the ground is under
    /// water. Ground height is `height - depth`.
    pub height: Vec<i16>,
    /// Integer traversal cost in tenths (10 = ordinary ground).
    pub move_cost: Vec<u8>,
    /// Palette / material id for display only.
    pub material: Vec<u8>,
    /// Water surface height in decimetres; `NO_WATER` for a dry map. Cells
    /// whose ground is below it are under water.
    #[serde(default = "no_water")]
    pub water_level: i16,
    /// Water depth per cell (decimetres); empty on a dry map. Kept apart
    /// from `height` so dry maps run pathsim's code with no water work.
    #[serde(default)]
    pub water_depth: Vec<i16>,
    /// Cost overlays: extra traversal cost per cell (tenths) on top of
    /// `base_cost`; `move_cost` is their sum (`effective_cost`). Both are
    /// empty until the first overlay edit, so maps without overlays run
    /// with no overlay work and keep their digests.
    #[serde(default)]
    pub overlay: Vec<u8>,
    #[serde(default)]
    pub base_cost: Vec<u8>,
}

/// `move_cost` of a cell under overlays: blocked stays 255, open cells cost
/// the base plus the overlay, at most 254.
pub fn effective_cost(base: u8, overlay: u8) -> u8 {
    if base == 255 {
        255
    } else {
        (base as u16 + overlay as u16).min(254) as u8
    }
}

fn no_water() -> i16 {
    NO_WATER
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
    /// `(x, y, overlay)`: set the cost overlay of these cells (applied
    /// after `cells`).
    #[serde(default)]
    pub overlay: Vec<(i32, i32, u8)>,
}

impl TerrainPatch {
    pub fn new() -> Self {
        Self {
            version: 0,
            cells: Vec::new(),
            overlay: Vec::new(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty() && self.overlay.is_empty()
    }
    /// Whether the patch changes the ground (blocked, height or material),
    /// not only traversal costs.
    pub fn changes_ground(&self) -> bool {
        self.cells
            .iter()
            .any(|e| e.blocked.is_some() || e.height.is_some() || e.material.is_some())
    }
    pub fn push(&mut self, e: CellEdit) {
        self.cells.push(e);
    }
    /// Sort and collapse duplicate cells. Later values win, deterministically.
    pub fn finish(&mut self) {
        self.cells.sort_by_key(|e| (e.y, e.x));
        let mut out: Vec<CellEdit> = Vec::with_capacity(self.cells.len());
        for e in self.cells.drain(..) {
            if let Some(prev) = out.last_mut() {
                if prev.x == e.x && prev.y == e.y {
                    if e.blocked.is_some() {
                        prev.blocked = e.blocked;
                    }
                    if e.height.is_some() {
                        prev.height = e.height;
                    }
                    if e.cost.is_some() {
                        prev.cost = e.cost;
                    }
                    if e.material.is_some() {
                        prev.material = e.material;
                    }
                    continue;
                }
            }
            out.push(e);
        }
        self.cells = out;
    }
    pub fn dirty_cells(&self) -> Option<(i32, i32, i32, i32)> {
        if self.is_empty() {
            return None;
        }
        let mut x0 = i32::MAX;
        let mut y0 = i32::MAX;
        let mut x1 = i32::MIN;
        let mut y1 = i32::MIN;
        for (x, y) in self
            .cells
            .iter()
            .map(|c| (c.x, c.y))
            .chain(self.overlay.iter().map(|o| (o.0, o.1)))
        {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
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
            geom: g.clone(),
            seed: cfg.seed,
            version: TERRAIN_VERSION_INITIAL,
            blocked: vec![0; n],
            height: vec![0; n],
            move_cost: vec![10; n],
            material: vec![Material::Grass as u8; n],
            water_level: NO_WATER,
            water_depth: Vec::new(),
            overlay: Vec::new(),
            base_cost: Vec::new(),
        };
        let surface = cfg.water == WaterMode::Surface;
        if surface {
            t.water_level = SURFACE_WATER_LEVEL;
        }
        let mut rng = Rng::new(cfg.seed);
        let water_seed = rng.next_u64();
        let forest_seed = rng.next_u64();
        let rock_seed = rng.next_u64();
        for y in 0..g.cells_y {
            for x in 0..g.cells_x {
                let i = (y * g.cells_x + x) as usize;
                let wx = x as f32;
                let wy = y as f32;
                let macro_n = fbm(cfg.seed, wx / 720.0, wy / 720.0, 3, 2.0, 0.52);
                let ridge = 1.0 - (fbm(rock_seed, wx / 260.0, wy / 260.0, 3, 2.1, 0.5) * 2.0 - 1.0).abs();
                let detail = fbm(forest_seed, wx / 58.0, wy / 58.0, 2, 2.0, 0.5);
                let h = 120.0 + macro_n * 920.0 + ridge * 520.0 + detail * 170.0;
                let h = h.round().clamp(0.0, 2400.0) as i16;
                t.height[i] = h;
                let moist = fbm(water_seed, wx / 410.0 + 31.0, wy / 410.0 - 17.0, 3, 2.0, 0.55);
                if h < 168 && surface {
                    t.material[i] = Material::Water as u8;
                } else if h < 168 {
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
        let spans = |v: &[(i32, i32)], f: &dyn Fn(i32) -> i32| {
            v.iter().map(|&(a, b)| (f(a), f(b).max(f(a) + 3))).collect::<Vec<_>>()
        };
        // A few irregular lake basins, independent of the global noise field.
        let lake = if surface { stamp_lake } else { stamp_ellipse };
        lake(&mut t, sx(1330), sy(1620), sr(92), sr(56), sr(27), water_seed);
        stamp_ellipse(&mut t, sx(520), sy(470), sr(54), sr(34), sr(22), rock_seed);
        lake(&mut t, sx(2200), sy(1760), sr(78), sr(45), sr(25), water_seed ^ 9);
        if surface {
            // A river from the north edge to the south edge separating the
            // spawn plaza (west) from the destinations (east), with two
            // fords; the road crosses it on a causeway (`stamp_road`).
            stamp_river(
                &mut t,
                (sx(600), 0),
                (sx(1700), cy - 1),
                sr(22),
                sr(60),
                &[sy(700), sy(2000)],
                sr(7),
                water_seed ^ 0x51,
            );
        }
        stamp_ellipse(&mut t, sx(1550), sy(430), sr(55), sr(38), sr(18), rock_seed ^ 4);
        stamp_ellipse(&mut t, sx(1930), sy(650), sr(40), sr(26), sr(14), rock_seed ^ 7);
        stamp_ellipse(&mut t, sx(340), sy(1950), sr(32), sr(22), sr(18), rock_seed ^ 3);
        // Broken fortification walls make deliberate chokepoints and force
        // routes to make meaningful decisions. Gaps are generated at fixed
        // positions so every scenario is reproducible.
        stamp_vwall(
            &mut t,
            sx(820),
            sy(160),
            sy(2360),
            7,
            &spans(&[(590, 638), (1420, 1450), (1986, 2014)], &sy),
        );
        stamp_hwall(
            &mut t,
            sy(1280),
            sx(1010),
            sx(2240),
            6,
            &spans(&[(1180, 1220), (1690, 1722)], &sx),
        );
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
                    let Some(i) = idx(g, px + 2 + dx, py + dy) else {
                        continue;
                    };
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
        let road: Vec<(i32, i32)> = [
            (250, 2300),
            (700, 2100),
            (1020, 1900),
            (1400, 1560),
            (1740, 1100),
            (2050, 650),
            (2290, 300),
        ]
        .iter()
        .map(|&(x, y)| (sx(x), sy(y)))
        .collect();
        stamp_road(&mut t, &natural_height, &road, 5);
        // Generation worked on ground heights; now put water on top.
        if surface {
            t.flood(SURFACE_WATER_LEVEL);
        }
        t
    }

    /// Put water up to `level` over a map whose `height` is still bare
    /// ground: cells below it get a depth, and `height` becomes the water
    /// surface there. Once per map.
    pub fn flood(&mut self, level: i16) {
        assert!(self.water_depth.is_empty(), "map already has water");
        self.water_level = level;
        self.water_depth = self
            .height
            .iter()
            .map(|&g| (level as i32 - g as i32).max(0) as i16)
            .collect();
        for h in &mut self.height {
            *h = (*h).max(level);
        }
    }

    pub fn len(&self) -> usize {
        self.blocked.len()
    }
    pub fn is_blocked(&self, c: CellPos) -> bool {
        self.geom.in_bounds(c) && self.blocked[self.geom.cell_index(c)] != 0
    }
    pub fn in_bounds_blocked_or_out(&self, c: CellPos) -> bool {
        !self.geom.in_bounds(c) || self.is_blocked(c)
    }
    pub fn height_at(&self, c: CellPos) -> i16 {
        self.height[self.geom.cell_index(c)]
    }
    pub fn cost_at(&self, c: CellPos) -> u8 {
        self.move_cost[self.geom.cell_index(c)]
    }
    pub fn material_at(&self, c: CellPos) -> u8 {
        self.material[self.geom.cell_index(c)]
    }

    #[inline]
    pub fn has_water(&self) -> bool {
        !self.water_depth.is_empty()
    }

    /// Water depth (decimetres) over cell index `i`; 0 when dry.
    #[inline]
    pub fn depth(&self, i: usize) -> i16 {
        if self.water_depth.is_empty() {
            0
        } else {
            self.water_depth[i]
        }
    }

    /// Ground height at cell index `i`, under any water.
    #[inline]
    pub fn ground_height(&self, i: usize) -> i16 {
        self.height[i] - self.depth(i)
    }

    /// Height of what a unit in `medium` stands on at cell index `i`: the
    /// water surface for surface movers, the ground (river bed) otherwise.
    #[inline]
    pub fn standing_height(&self, i: usize, medium: &Medium) -> i16 {
        if medium.on_surface() {
            self.height[i]
        } else {
            self.ground_height(i)
        }
    }

    /// Whether a unit in `medium` may occupy cell index `i` (ignoring slope
    /// and clearance): not an obstacle and a depth the medium allows.
    #[inline]
    pub fn open_for(&self, i: usize, medium: &Medium) -> bool {
        self.blocked[i] == 0 && medium.allows(self.depth(i))
    }
    pub fn world_of_cell(&self, c: CellPos) -> WorldPos {
        self.geom.cell_center(c)
    }
    pub fn cell_of_world(&self, p: WorldPos) -> CellPos {
        self.geom.world_to_cell(p)
    }

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
            if e.x < 0 || e.y < 0 || e.x >= self.geom.cells_x || e.y >= self.geom.cells_y {
                continue;
            }
            let i = (e.y * self.geom.cells_x + e.x) as usize;
            if let Some(b) = e.blocked {
                self.blocked[i] = b as u8;
            }
            if let Some(h) = e.height {
                // An edit sets the ground; the water surface stays where it is.
                let ground = h.clamp(0, 2400);
                if self.water_depth.is_empty() {
                    self.height[i] = ground;
                } else {
                    self.water_depth[i] = (self.water_level as i32 - ground as i32).max(0) as i16;
                    self.height[i] = ground.max(self.water_level);
                }
            }
            // Cost edits set the base cost under any overlays.
            let layered = !self.overlay.is_empty();
            let cost = if layered {
                &mut self.base_cost[i]
            } else {
                &mut self.move_cost[i]
            };
            if let Some(c) = e.cost {
                *cost = c;
            }
            if let Some(m) = e.material {
                self.material[i] = m;
            }
            if e.blocked == Some(true) {
                self.material[i] = Material::Rock as u8;
                *cost = 255;
            }
            if e.blocked == Some(false) && *cost == 255 {
                *cost = 10;
                if self.material[i] == Material::Rock as u8 {
                    self.material[i] = Material::Grass as u8;
                }
            }
            if layered {
                self.move_cost[i] = effective_cost(self.base_cost[i], self.overlay[i]);
            }
        }
        if !patch.overlay.is_empty() && self.overlay.is_empty() {
            self.base_cost = self.move_cost.clone();
            self.overlay = vec![0; self.move_cost.len()];
        }
        for &(x, y, o) in &patch.overlay {
            if x < 0 || y < 0 || x >= self.geom.cells_x || y >= self.geom.cells_y {
                continue;
            }
            let i = (y * self.geom.cells_x + x) as usize;
            self.overlay[i] = o;
            self.move_cost[i] = effective_cost(self.base_cost[i], o);
        }
        patch
    }

    /// Deterministic digest of authoritative terrain, for replay checks:
    /// the version and the digest of every band of `DIGEST_BAND_ROWS` rows.
    /// Callers cache band digests so an edit rehashes only its own bands.
    pub fn digest(&self) -> u64 {
        let bands: Vec<u64> = (0..self.digest_bands()).map(|b| self.band_digest(b)).collect();
        Self::digest_of_bands_with_water(self.version, self.water_level, &bands)
    }

    pub fn digest_bands(&self) -> usize {
        (self.geom.cells_y.max(0) as usize).div_ceil(DIGEST_BAND_ROWS)
    }

    /// The band holding row `y`.
    pub fn digest_band_of(&self, y: i32) -> usize {
        (y.clamp(0, self.geom.cells_y - 1) as usize) / DIGEST_BAND_ROWS
    }

    pub fn band_digest(&self, band: usize) -> u64 {
        let w = self.geom.cells_x.max(0) as usize;
        let lo = band * DIGEST_BAND_ROWS * w;
        let hi = (lo + DIGEST_BAND_ROWS * w).min(self.blocked.len());
        let mut h = crate::rng::StateHasher::new();
        let wet = !self.water_depth.is_empty();
        let layered = !self.overlay.is_empty();
        for i in lo..hi {
            h.u64(
                ((self.blocked[i] as u64) << 40) | ((self.move_cost[i] as u64) << 16) | (self.height[i] as u16 as u64),
            );
            if wet {
                h.u32(self.water_depth[i] as u16 as u32);
            }
            if layered {
                h.u32((self.overlay[i] as u32) << 8 | self.base_cost[i] as u32);
            }
        }
        h.finish()
    }

    pub fn digest_of_bands(version: u32, bands: &[u64]) -> u64 {
        Self::digest_of_bands_with_water(version, NO_WATER, bands)
    }

    /// As `digest_of_bands`; the water level is hashed only on a map with
    /// water, so dry maps keep pathsim's digests.
    pub fn digest_of_bands_with_water(version: u32, water_level: i16, bands: &[u64]) -> u64 {
        let mut h = crate::rng::StateHasher::new();
        h.u32(version);
        if water_level != NO_WATER {
            h.u32(0x57A7_E400 | water_level as u16 as u32);
        }
        for &b in bands {
            h.u64(b);
        }
        h.finish()
    }

    pub fn nearest_open_cell(&self, c: CellPos, max_radius: i32, accept: impl Fn(CellPos) -> bool) -> Option<CellPos> {
        let c = self.geom.clamp_cell(c);
        if accept(c) {
            return Some(c);
        }
        for r in 1..=max_radius {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs() != r && dy.abs() != r {
                        continue;
                    }
                    let p = CellPos::new(c.x + dx, c.y + dy);
                    if self.geom.in_bounds(p) && accept(p) {
                        return Some(p);
                    }
                }
            }
        }
        None
    }
}

fn idx(g: &WorldGeometry, x: i32, y: i32) -> Option<usize> {
    g.cell_index_xy(x, y)
}

/// Rows per terrain digest band.
pub const DIGEST_BAND_ROWS: usize = 16;

fn paint_block(t: &mut Terrain, x: i32, y: i32, blocked: bool) {
    if let Some(i) = idx(&t.geom, x, y) {
        // Walls and rocks stop at open water: they never dam a river.
        if blocked && t.water_level != NO_WATER && t.blocked[i] == 0 && t.material[i] == Material::Water as u8 {
            return;
        }
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
            if idx(&t.geom, x, y).is_none() {
                continue;
            }
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

/// Height gain (decimetres per cell) of lake and river banks: gentler than
/// any profile's slope limit.
const BANK_RAMP: f32 = 50.0;

/// Widest bank needed to climb from the water to the highest ground.
fn bank_cells(t: &Terrain) -> i32 {
    ((2400.0 - t.water_level as f32) / BANK_RAMP).ceil() as i32 + 1
}

/// A lake basin under the water surface: the ellipse of `stamp_ellipse`,
/// carved down to between 4 and 16 m deep (deepest in the middle) instead of
/// raised into an obstacle, inside banks that ramp up from the water at
/// `BANK_RAMP` until they meet the natural ground, so the shore is no cliff.
fn stamp_lake(t: &mut Terrain, cx: i32, cy: i32, rx: i32, ry: i32, rough: i32, seed: u64) {
    let level = t.water_level as f32;
    let bank = bank_cells(t);
    let r_min = rx.min(ry).max(1) as f32;
    for y in cy - ry - rough - bank..=cy + ry + rough + bank {
        for x in cx - rx - rough - bank..=cx + rx + rough + bank {
            let Some(i) = idx(&t.geom, x, y) else { continue };
            let nx = (x - cx) as f32 / rx as f32;
            let ny = (y - cy) as f32 / ry as f32;
            let d = nx * nx + ny * ny;
            let jitter = (hash2(seed, x, y) as f32 / u32::MAX as f32 - 0.5) * 0.16;
            let edge = 1.0 + jitter;
            if d < edge {
                let depth = 40.0 + 120.0 * (1.0 - d.min(1.0));
                let floor = (level - depth).round() as i16;
                t.height[i] = t.height[i].min(floor).max(0);
                t.blocked[i] = 0;
                t.material[i] = Material::Water as u8;
                t.move_cost[i] = 10;
            } else {
                // Distance beyond the shore, in cells along the short axis.
                let beyond = (d.sqrt() - edge.sqrt()) * r_min;
                if beyond > bank as f32 {
                    continue;
                }
                let bank_height = (level + 6.0 + beyond * BANK_RAMP).round() as i16;
                if bank_height < t.height[i] {
                    t.height[i] = bank_height;
                    if t.blocked[i] == 0 {
                        t.material[i] = Material::Sand as u8;
                        t.move_cost[i] = 12;
                    }
                }
            }
        }
    }
}

/// A river along the line `from`..`to` (cells), wobbling sideways by up to
/// `wobble` cells: `half_width` cells of water each side of its centre (1 m
/// deep at the edge, 10 m in the middle), then banks ramping up at
/// `BANK_RAMP` until they meet the natural ground. Rows within `ford_half` of
/// each `fords` row are only 1.2 m deep (land units of `water_profiles` wade
/// across).
#[allow(clippy::too_many_arguments)]
fn stamp_river(
    t: &mut Terrain,
    from: (i32, i32),
    to: (i32, i32),
    half_width: i32,
    wobble: i32,
    fords: &[i32],
    ford_half: i32,
    seed: u64,
) {
    let level = t.water_level as f32;
    let bank = bank_cells(t);
    let (y0, y1) = (from.1.min(to.1), from.1.max(to.1));
    let phase = (seed % 628) as f32 / 100.0;
    for y in y0..=y1 {
        let f = (y - from.1) as f32 / (to.1 - from.1).max(1) as f32;
        let wob = (f * 9.0 + phase).sin() * 0.7 + (f * 23.0 + phase * 2.0).sin() * 0.3;
        let cx = from.0 as f32 + (to.0 - from.0) as f32 * f + wob * wobble as f32;
        let ford = fords.iter().any(|&fy| (y - fy).abs() <= ford_half);
        let reach = half_width + bank;
        for x in (cx as i32 - reach)..=(cx as i32 + reach) {
            let Some(i) = idx(&t.geom, x, y) else { continue };
            let dist = (x as f32 + 0.5 - cx).abs();
            if dist <= half_width as f32 {
                let k = dist / half_width.max(1) as f32;
                let depth = if ford { 12.0 } else { 10.0 + 90.0 * (1.0 - k * k) };
                t.height[i] = t.height[i].min((level - depth).round() as i16).max(0);
                t.blocked[i] = 0;
                t.material[i] = Material::Water as u8;
                t.move_cost[i] = 10;
            } else {
                let bank_height = (level + 6.0 + (dist - half_width as f32) * BANK_RAMP).round() as i16;
                if bank_height < t.height[i] {
                    t.height[i] = bank_height;
                    if t.blocked[i] == 0 {
                        t.material[i] = Material::Sand as u8;
                        t.move_cost[i] = 12;
                    }
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
        if gaps.iter().any(|&(a, b)| y >= a && y <= b) {
            continue;
        }
        let wobble = ((y / 43) % 3) - 1;
        for x in x0 + wobble - thickness / 2..=x0 + wobble + thickness / 2 {
            paint_block(t, x, y, true);
        }
    }
}

fn stamp_hwall(t: &mut Terrain, y0: i32, x0: i32, x1: i32, thickness: i32, gaps: &[(i32, i32)]) {
    for x in x0..=x1 {
        if gaps.iter().any(|&(a, b)| x >= a && x <= b) {
            continue;
        }
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
                        if dx * dx + dy * dy > (half_width + 1) * (half_width + 1) {
                            continue;
                        }
                        t.blocked[i] = 0;
                        t.move_cost[i] = 8;
                        t.material[i] = Material::Road as u8;
                        // On a map with water the road crosses it on a causeway.
                        t.height[i] = natural_height[i].max(t.water_level);
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
                if let Some(i) = self.geom.cell_index_xy(x, y) {
                    self.blocked[i] = 1;
                }
            }
        }
    }
    pub fn into_terrain(self, seed: u64) -> Terrain {
        let material = self.cost.iter().map(|_| Material::Grass as u8).collect();
        Terrain {
            geom: self.geom,
            seed,
            version: 1,
            blocked: self.blocked,
            height: self.height,
            move_cost: self.cost,
            material,
            water_level: NO_WATER,
            water_depth: Vec::new(),
            overlay: Vec::new(),
            base_cost: Vec::new(),
        }
    }
}

pub type SharedTerrain = Arc<Terrain>;

/// Map a cell given in the 2560 x 2560 reference layout (where scenario
/// features are authored) onto a map of another size.
pub fn reference_cell(geom: &WorldGeometry, x: i32, y: i32) -> CellPos {
    geom.clamp_cell(CellPos::new(x * geom.cells_x / 2560, y * geom.cells_y / 2560))
}
