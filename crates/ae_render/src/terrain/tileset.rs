//! Material tilesets: reusable terrain surface layers (MrBob's proposal,
//! `docs/tilesets.md`). A tileset is a few tileable layers, each an albedo
//! with a height channel and a matching normal map, plus a table from the
//! map's gameplay materials to layers. The terrain shader projects the
//! layers by surface orientation and blends them in two levels (broad
//! masks from the geometry and the material cells, broken up per pixel by
//! the layers' heights and normals), so a map supplies only its heights
//! and material ids, and the same tileset works on any map.
//!
//! Until a texture source is chosen the layers are procedural ([`temperate`],
//! [`desert`]): periodic gradient and cellular noise, so they tile exactly.
//!
//! GPU layout (see [`Tileset::ubo`] and `shaders/terrain.frag`):
//! - albedo: `R8G8B8A8_SRGB` 2D array, one layer per tileset layer, full
//!   mip chain (made on the GPU from mip 0, `upload::image_mipped`); RGB =
//!   albedo, A = height (linear in an sRGB format);
//! - normals: `R8G8_UNORM` 2D array, same shape; tangent-space x, y (z is
//!   reconstructed), y along the texture's v axis.

pub const MAX_LAYERS: usize = 8;
/// Gameplay material ids the table covers (`ae_nav::terrain::Material`
/// uses 0..=6).
pub const MATERIAL_SLOTS: usize = 8;

/// How a layer is projected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    /// Planar (from above) when it covers ground, duoplanar (from the two
    /// horizontal axes) when it covers a cliff.
    ByOrientation = 0,
    /// Triplanar wherever it is used (rough materials on mixed slopes).
    Triplanar = 1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerSettings {
    pub name: String,
    /// World units one repeat of the texture covers.
    pub tile_size: f32,
    /// Scale of the detail normal (0: flat, 1: as generated).
    pub normal_strength: f32,
    /// GGX roughness of the layer's speculars (0 mirror, 1 fully rough).
    pub roughness: f32,
    pub projection: Projection,
    /// The texture has a direction anti-tiling must keep (strata, ripples):
    /// its copies are shifted, not rotated.
    pub directional: bool,
}

/// One layer's textures: mip 0, `size` square, tightly packed (the GPU
/// makes the mips).
#[derive(Clone, Debug)]
pub struct Layer {
    pub settings: LayerSettings,
    /// `R8G8B8A8_SRGB` texels (A = height).
    pub albedo: Vec<u8>,
    /// `R8G8_UNORM` texels (tangent-space normal x, y).
    pub normal: Vec<u8>,
}

impl Layer {
    /// The mean albedo (linear light) and height over mip 0.
    pub fn mean(&self) -> [f32; 4] {
        let texels = self.albedo.len() / 4;
        if texels == 0 {
            return [0.0; 4];
        }
        let mut sum = [0.0f64; 4];
        for t in self.albedo.as_chunks::<4>().0 {
            for c in 0..3 {
                sum[c] += srgb_decode(t[c]) as f64;
            }
            sum[3] += t[3] as f64 / 255.0;
        }
        sum.map(|c| (c / texels as f64) as f32)
    }
}

/// The layers a gameplay material uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialLayers {
    /// On flat ground.
    pub ground: u8,
    /// On slopes past the cliff threshold.
    pub cliff: u8,
}

#[derive(Clone, Debug)]
pub struct Tileset {
    pub name: String,
    /// Side of mip 0 in texels (a power of two).
    pub size: u32,
    pub layers: Vec<Layer>,
    pub materials: [MaterialLayers; MATERIAL_SLOTS],
    /// Anti-tiling degree, 0..=1 (`docs/antitiling.md`): 0 samples each
    /// layer once, 1 uses the most versions of its texture. The shader
    /// always carries the path, so the degree alone sets the look and the
    /// cost.
    pub anti_tiling: f32,
    /// Geometric slope (1 − normal.y) where cliffs start and where they are
    /// complete.
    pub cliff_slope: (f32, f32),
    /// Height-blend transition depth: smaller gives sharper, more
    /// irregular boundaries between layers (in layer height units, 0..1).
    pub blend_depth: f32,
    /// How much the ground layer's detail normal moves the cliff mask
    /// (the second level of blending on slopes).
    pub detail_tilt: f32,
    /// Material boundaries: the cell lookup is displaced by up to `.0`
    /// cells by a smooth noise with features of `.1` world units, so
    /// boundaries meander instead of following the cell grid.
    pub warp: (f32, f32),
    /// Anti-tiling: albedo brightness varies by up to ± this fraction with
    /// a smooth noise of features `.1` world units, so a layer's repeats do
    /// not line up into a visible grid at a distance.
    pub macro_variation: (f32, f32),
}

/// The tileset's uniform block (`TilesetUBO` in `terrain.frag`),
/// std140 with only 16-byte members.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TilesetUbo {
    /// Per material id: x = ground layer, y = cliff layer.
    pub materials: [[u32; 4]; MATERIAL_SLOTS],
    /// Per layer: x = repeats per world unit, y = normal strength,
    /// z = projection (bit 0: 0 by orientation, 1 triplanar) and bit 1
    /// directional (anti-tiling shifts instead of rotating), w unused.
    pub layers: [[f32; 4]; MAX_LAYERS],
    /// Per layer: the mean albedo (linear) and height, which the
    /// anti-tiling blend keeps its contrast around.
    pub means: [[f32; 4]; MAX_LAYERS],
    /// Per layer: x = GGX roughness.
    pub roughness: [[f32; 4]; MAX_LAYERS],
    /// x, y: cliff slope start and end; z: blend depth; w: detail tilt.
    pub params: [f32; 4],
    /// x: boundary warp (cells); y: 1 / its feature size (world units);
    /// z: macro variation; w: 1 / its feature size.
    pub warp: [f32; 4],
    /// x: anti-tiling degree (0..1).
    pub anti_tiling: [f32; 4],
}

pub const TILESET_UBO_BYTES: usize = 560;
const _: () = assert!(size_of::<TilesetUbo>() == TILESET_UBO_BYTES);

/// Finite and above zero (NaN is not).
fn positive(x: f32) -> bool {
    x.is_finite() && x > 0.0
}

impl Tileset {
    /// Whether the textures and table are consistent.
    pub fn validate(&self) -> Result<(), String> {
        if !self.size.is_power_of_two() || self.size < 4 {
            return Err(format!("tileset `{}`: size {} is not a power of two of at least 4", self.name, self.size));
        }
        if self.layers.is_empty() || self.layers.len() > MAX_LAYERS {
            return Err(format!("tileset `{}`: {} layers (1..={MAX_LAYERS})", self.name, self.layers.len()));
        }
        let side = self.size as usize;
        for l in &self.layers {
            let s = &l.settings;
            if l.albedo.len() != side * side * 4 || l.normal.len() != side * side * 2 {
                return Err(format!("layer `{}`: {} albedo and {} normal bytes, not {side}² texels", s.name, l.albedo.len(), l.normal.len()));
            }
            if !positive(s.tile_size) || !(0.0..=4.0).contains(&s.normal_strength) || !(0.0..=1.0).contains(&s.roughness) {
                return Err(format!("layer `{}`: tile size {}, normal strength {} or roughness {} out of range", s.name, s.tile_size, s.normal_strength, s.roughness));
            }
        }
        for (id, m) in self.materials.iter().enumerate() {
            if m.ground as usize >= self.layers.len() || m.cliff as usize >= self.layers.len() {
                return Err(format!("tileset `{}`: material {id} uses layer {} / {} of {}", self.name, m.ground, m.cliff, self.layers.len()));
            }
        }
        let (a, b) = self.cliff_slope;
        if !(0.0..=4.0).contains(&self.warp.0) || !positive(self.warp.1) || !(0.0..=0.9).contains(&self.macro_variation.0) || !positive(self.macro_variation.1) {
            return Err(format!("tileset `{}`: warp {:?} out of range", self.name, self.warp));
        }
        if !(0.0..=1.0).contains(&self.anti_tiling) {
            return Err(format!("tileset `{}`: anti-tiling degree {} out of range", self.name, self.anti_tiling));
        }
        if !(0.0 <= a && a < b && b <= 1.0) || !positive(self.blend_depth) {
            return Err(format!("tileset `{}`: cliff slope {a}..{b} or blend depth {} out of range", self.name, self.blend_depth));
        }
        Ok(())
    }

    pub fn ubo(&self) -> TilesetUbo {
        let mut u = TilesetUbo::default();
        for (slot, m) in u.materials.iter_mut().zip(&self.materials) {
            *slot = [m.ground as u32, m.cliff as u32, 0, 0];
        }
        for (slot, l) in u.layers.iter_mut().zip(&self.layers) {
            let s = &l.settings;
            let flags = s.projection as u32 | u32::from(s.directional) << 1;
            *slot = [1.0 / s.tile_size, s.normal_strength, flags as f32, 0.0];
        }
        for (slot, l) in u.means.iter_mut().zip(&self.layers) {
            *slot = l.mean();
        }
        for (slot, l) in u.roughness.iter_mut().zip(&self.layers) {
            *slot = [l.settings.roughness, 0.0, 0.0, 0.0];
        }
        u.params = [self.cliff_slope.0, self.cliff_slope.1, self.blend_depth, self.detail_tilt];
        u.warp = [self.warp.0, 1.0 / self.warp.1, self.macro_variation.0, 1.0 / self.macro_variation.1];
        u.anti_tiling = [self.anti_tiling, 0.0, 0.0, 0.0];
        u
    }

    /// Texture bytes on the GPU (both arrays, every mip: a third more than
    /// mip 0).
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(|l| (l.albedo.len() + l.normal.len()) * 4 / 3).sum()
    }
}

/// Side of the anti-tiling index noise ([`anti_tiling_noise`]).
pub const NOISE_SIZE: u32 = 256;

/// The anti-tiling index: a smooth noise that tiles, `R8_UNORM`, mip 0
/// ([`NOISE_SIZE`] square; the GPU makes the mips), 16 features across. Its values are
/// equalised (ranked onto 0..1), so each of a layer's versions covers about
/// the same share of the ground. The octaves of the gradient noise are
/// offset from one another by fractions of a cell: gradient noise is 0 at
/// its lattice points, and with aligned octaves the sum would be too, a
/// regular grid of one version.
pub fn anti_tiling_noise() -> Vec<u8> {
    let n = NOISE_SIZE as usize;
    let mut v: Vec<(f32, usize)> = (0..n * n)
        .map(|i| {
            let (x, y) = ((i % n) as f32 / n as f32, (i / n) as f32 / n as f32);
            let (mut sum, mut amp, mut p) = (0.0, 1.0, 16);
            for o in 0..3u32 {
                let (ox, oy) = (0.37 + 0.29 * o as f32, 0.61 + 0.17 * o as f32);
                sum += amp * gradient(x * p as f32 + ox, y * p as f32 + oy, p, 0xa71 + o * 977);
                amp *= 0.5;
                p *= 2;
            }
            (sum, i)
        })
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut mip = vec![0u8; n * n];
    for (rank, &(_, i)) in v.iter().enumerate() {
        mip[i] = ((rank as f32 + 0.5) / (n * n) as f32 * 255.0).round() as u8;
    }
    mip
}

/// A procedural tileset by name (`temperate`, `desert`) with `size`-texel
/// layers.
pub fn by_name(name: &str, size: u32) -> Option<Tileset> {
    match name {
        "temperate" => Some(temperate(size)),
        "desert" => Some(desert(size)),
        _ => None,
    }
}

pub const NAMES: [&str; 2] = ["temperate", "desert"];

// ---------------------------------------------------------------------------
// Periodic noise (everything wraps at the tile edge)
// ---------------------------------------------------------------------------

fn hash(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^ (h >> 15)
}

fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Gradient noise on a lattice of `period` cells that wraps (`x`, `y` in
/// lattice units); roughly −1..1.
fn gradient(x: f32, y: f32, period: i32, seed: u32) -> f32 {
    let (fx, fy) = (x.floor(), y.floor());
    let (tx, ty) = (x - fx, y - fy);
    let (ix, iy) = (fx as i32, fy as i32);
    let corner = |dx: i32, dy: i32| {
        let h = hash((ix + dx).rem_euclid(period), (iy + dy).rem_euclid(period), seed);
        let a = unit(h) * std::f32::consts::TAU;
        a.cos() * (tx - dx as f32) + a.sin() * (ty - dy as f32)
    };
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v) = (fade(tx), fade(ty));
    let a = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * u;
    let b = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * u;
    (a + (b - a) * v) * 1.4
}

/// Fractal sum over `octaves`, `period` lattice cells per tile at the base
/// (`u`, `v` in tile units 0..1).
fn fbm(u: f32, v: f32, period: i32, octaves: u32, gain: f32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut norm, mut p) = (0.0, 1.0, 0.0, period);
    for o in 0..octaves {
        sum += amp * gradient(u * p as f32, v * p as f32, p, seed.wrapping_add(o * 977));
        norm += amp;
        amp *= gain;
        p *= 2;
    }
    sum / norm
}

/// Cellular noise on `period` cells per tile: distances to the nearest and
/// second-nearest feature point (in cell units) and the nearest point's
/// hash.
fn cellular(u: f32, v: f32, period: i32, seed: u32) -> (f32, f32, u32) {
    let (x, y) = (u * period as f32, v * period as f32);
    let (cx, cy) = (x.floor() as i32, y.floor() as i32);
    let (mut d1, mut d2, mut id) = (f32::MAX, f32::MAX, 0);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (gx, gy) = (cx + dx, cy + dy);
            let h = hash(gx.rem_euclid(period), gy.rem_euclid(period), seed);
            let px = gx as f32 + unit(h);
            let py = gy as f32 + unit(hash(h as i32, 7, seed));
            let d = ((px - x).powi(2) + (py - y).powi(2)).sqrt();
            if d < d1 {
                (d2, d1, id) = (d1, d, h);
            } else if d < d2 {
                d2 = d;
            }
        }
    }
    (d1, d2, id)
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    a.map(|c| c * s)
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

/// What a layer looks like: height (0..1) and linear albedo at tile
/// coordinates `(u, v)`.
#[derive(Clone, Copy, Debug)]
enum Recipe {
    Grass { base: [f32; 3], tip: [f32; 3] },
    Forest,
    Sand { light: [f32; 3], dark: [f32; 3], ripples: f32 },
    Rock { light: [f32; 3], dark: [f32; 3], strata: f32 },
    Pebbles { stone: [f32; 3], fill: [f32; 3], cells: i32 },
    Mud { wet: [f32; 3], dry: [f32; 3] },
    Cracked { clay: [f32; 3], crack: [f32; 3] },
}

fn sample(recipe: Recipe, u: f32, v: f32, seed: u32) -> (f32, [f32; 3]) {
    match recipe {
        Recipe::Grass { base, tip } => {
            let clumps = fbm(u, v, 6, 4, 0.5, seed) * 0.5 + 0.5;
            let blades = fbm(u, v, 48, 2, 0.5, seed + 1) * 0.5 + 0.5;
            let h = clumps * 0.6 + blades * 0.4;
            let tint = fbm(u, v, 3, 3, 0.5, seed + 2) * 0.5 + 0.5;
            (h, scale3(mix3(base, tip, tint * 0.7 + blades * 0.3), 0.75 + 0.5 * h))
        }
        Recipe::Forest => {
            let (d1, _, id) = cellular(u, v, 14, seed);
            let leaf = 1.0 - smooth(0.25, 0.55, d1);
            let ground = fbm(u, v, 8, 4, 0.55, seed + 3) * 0.5 + 0.5;
            let h = (ground * 0.5 + leaf * 0.5 * (0.6 + 0.4 * unit(id))).clamp(0.0, 1.0);
            let leaf_col = mix3([0.20, 0.12, 0.05], [0.10, 0.17, 0.05], unit(id));
            (h, mix3(scale3([0.09, 0.11, 0.05], 0.7 + 0.6 * ground), leaf_col, leaf))
        }
        Recipe::Sand { light, dark, ripples } => {
            let warp = fbm(u, v, 3, 3, 0.5, seed) * 0.35;
            let r = (std::f32::consts::TAU * (v * ripples + warp * ripples * 0.15)).sin() * 0.5 + 0.5;
            let grain = fbm(u, v, 64, 2, 0.5, seed + 1) * 0.5 + 0.5;
            let broad = fbm(u, v, 4, 4, 0.5, seed + 2) * 0.5 + 0.5;
            let h = r * 0.45 + broad * 0.4 + grain * 0.15;
            (h, scale3(mix3(dark, light, broad * 0.6 + r * 0.25 + grain * 0.15), 1.0))
        }
        Recipe::Rock { light, dark, strata } => {
            let ridged = 1.0 - fbm(u, v, 4, 5, 0.55, seed).abs();
            let (d1, d2, id) = cellular(u, v, 6, seed + 1);
            let crack = smooth(0.0, 0.08, d2 - d1);
            let bands = (std::f32::consts::TAU * (v * strata + fbm(u, v, 2, 2, 0.5, seed + 2) * 0.6)).sin() * 0.5 + 0.5;
            let h = (ridged * 0.6 + bands * 0.15 + 0.25 * unit(id)) * (0.35 + 0.65 * crack);
            let col = mix3(dark, light, (ridged * 0.5 + bands * 0.3 + unit(id) * 0.2).clamp(0.0, 1.0));
            (h, scale3(col, 0.55 + 0.45 * crack))
        }
        Recipe::Pebbles { stone, fill, cells } => {
            let (d1, _, id) = cellular(u, v, cells, seed);
            let dome = (1.0 - d1 / 0.6).max(0.0);
            let stone_h = dome.sqrt();
            let is_stone = smooth(0.05, 0.2, stone_h);
            let fine = fbm(u, v, 32, 2, 0.5, seed + 1) * 0.5 + 0.5;
            let h = stone_h * (0.7 + 0.3 * unit(id)) * 0.85 + fine * 0.15;
            let col = mix3(scale3(fill, 0.8 + 0.4 * fine), scale3(stone, 0.75 + 0.5 * unit(id)), is_stone);
            (h, col)
        }
        Recipe::Mud { wet, dry } => {
            let puddles = fbm(u, v, 4, 4, 0.5, seed) * 0.5 + 0.5;
            let fine = fbm(u, v, 24, 3, 0.5, seed + 1) * 0.5 + 0.5;
            let h = puddles * 0.7 + fine * 0.3;
            (h, mix3(wet, dry, smooth(0.35, 0.65, h)))
        }
        Recipe::Cracked { clay, crack } => {
            let (d1, d2, id) = cellular(u, v, 10, seed);
            let edge = smooth(0.0, 0.06, d2 - d1);
            let plate = fbm(u, v, 16, 2, 0.5, seed + 1) * 0.5 + 0.5;
            let h = edge * (0.7 + 0.2 * unit(id) + 0.1 * plate);
            (h, mix3(crack, scale3(clay, 0.85 + 0.3 * plate), edge))
        }
    }
}

fn srgb_encode(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0).round() as u8
}

fn srgb_decode(b: u8) -> f32 {
    let s = b as f32 / 255.0;
    if s <= 0.040_45 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
}

fn unorm(x: f32) -> u8 {
    ((x * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
fn snorm(b: u8) -> f32 {
    b as f32 / 255.0 * 2.0 - 1.0
}

/// Mip 0 of a layer: albedo + height (RGBA8 sRGB) and normals (RG8) from
/// the height's wrapped central differences; `bump` scales the slopes.
fn generate(recipe: Recipe, size: u32, seed: u32, bump: f32) -> (Vec<u8>, Vec<u8>) {
    let n = size as usize;
    let mut heights = vec![0.0f32; n * n];
    let mut albedo = vec![0u8; n * n * 4];
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(n);
    let rows = n.div_ceil(threads);
    std::thread::scope(|s| {
        for (chunk, (hs, al)) in heights.chunks_mut(rows * n).zip(albedo.chunks_mut(rows * n * 4)).enumerate() {
            s.spawn(move || {
                for (i, h) in hs.iter_mut().enumerate() {
                    let (x, y) = (i % n, chunk * rows + i / n);
                    let (height, col) = sample(recipe, (x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32, seed);
                    *h = height.clamp(0.0, 1.0);
                    al[i * 4..i * 4 + 4].copy_from_slice(&[srgb_encode(col[0]), srgb_encode(col[1]), srgb_encode(col[2]), (*h * 255.0).round() as u8]);
                }
            });
        }
    });
    let at = |x: usize, y: usize| heights[(y % n) * n + (x % n)];
    let mut normal = vec![0u8; n * n * 2];
    // Slopes in height units per texel, as if a tile were `n` texels of
    // height range `bump * n / 64` (so the look does not depend on size).
    let k = bump * n as f32 / 64.0;
    for y in 0..n {
        for x in 0..n {
            let du = (at(x + 1, y) - at(x + n - 1, y)) * 0.5 * k;
            let dv = (at(x, y + 1) - at(x, y + n - 1)) * 0.5 * k;
            let len = (du * du + dv * dv + 1.0).sqrt();
            normal[(y * n + x) * 2] = unorm(-du / len);
            normal[(y * n + x) * 2 + 1] = unorm(-dv / len);
        }
    }
    (albedo, normal)
}

#[allow(clippy::too_many_arguments)]
fn layer(name: &str, recipe: Recipe, size: u32, seed: u32, tile_size: f32, bump: f32, projection: Projection, directional: bool) -> Layer {
    let (albedo, normal) = generate(recipe, size, seed, bump);
    Layer { settings: LayerSettings { name: name.into(), tile_size, normal_strength: 1.0, roughness: 0.85, projection, directional }, albedo, normal }
}

/// The per-layer roughness of a tileset, in its layer order (the sets below
/// give each surface a plausible specular response: grass and scrub matte,
/// sand a little sheen, rock smoother, wet beds and mud glossy).
fn roughness(mut t: Tileset, values: &[f32]) -> Tileset {
    assert_eq!(values.len(), t.layers.len(), "one roughness per layer");
    for (l, &r) in t.layers.iter_mut().zip(values) {
        l.settings.roughness = r;
    }
    debug_assert_eq!(t.validate(), Ok(()));
    t
}

fn build(name: &str, size: u32, specs: Vec<(&str, Recipe, f32, f32, Projection, bool)>, materials: [MaterialLayers; MATERIAL_SLOTS]) -> Tileset {
    assert!(size.is_power_of_two() && size >= 4, "tileset size {size}");
    let layers = std::thread::scope(|s| {
        let handles: Vec<_> = specs.iter().enumerate().map(|(i, &(n, r, tile, bump, p, a))| s.spawn(move || layer(n, r, size, 0x5eed + i as u32 * 131, tile, bump, p, a))).collect();
        handles.into_iter().map(|h| h.join().expect("layer generation")).collect()
    });
    let t = Tileset { name: name.into(), size, layers, materials, anti_tiling: 1.0, cliff_slope: (0.18, 0.42), blend_depth: 0.22, detail_tilt: 0.35, warp: (2.0, 40.0), macro_variation: (0.22, 150.0) };
    debug_assert_eq!(t.validate(), Ok(()));
    t
}

const fn m(ground: u8, cliff: u8) -> MaterialLayers {
    MaterialLayers { ground, cliff }
}

/// Grass, forest floor, sand, grey rock, pebbled river bed, gravel road,
/// mud. Materials (`ae_nav::terrain::Material`): grass, forest, sand, rock,
/// water (the bed), road, mud; every cliff is rock.
pub fn temperate(size: u32) -> Tileset {
    use Projection::*;
    let specs = vec![
        ("grass", Recipe::Grass { base: [0.08, 0.14, 0.03], tip: [0.20, 0.26, 0.06] }, 24.0, 1.0, ByOrientation, false),
        ("forest floor", Recipe::Forest, 20.0, 1.2, ByOrientation, false),
        ("sand", Recipe::Sand { light: [0.62, 0.52, 0.33], dark: [0.42, 0.34, 0.20], ripples: 9.0 }, 32.0, 0.8, ByOrientation, true),
        ("rock", Recipe::Rock { light: [0.33, 0.31, 0.29], dark: [0.12, 0.11, 0.10], strata: 5.0 }, 48.0, 2.0, ByOrientation, true),
        ("river bed", Recipe::Pebbles { stone: [0.30, 0.28, 0.24], fill: [0.16, 0.14, 0.10], cells: 12 }, 16.0, 1.5, ByOrientation, false),
        ("gravel", Recipe::Pebbles { stone: [0.38, 0.35, 0.31], fill: [0.25, 0.22, 0.18], cells: 22 }, 12.0, 1.0, ByOrientation, false),
        ("mud", Recipe::Mud { wet: [0.07, 0.05, 0.03], dry: [0.20, 0.14, 0.08] }, 20.0, 2.5, ByOrientation, false),
    ];
    // grass, forest floor, sand, rock, river bed, gravel, mud
    roughness(build("temperate", size, specs, [m(0, 3), m(1, 3), m(2, 3), m(3, 3), m(4, 3), m(5, 3), m(6, 3), m(0, 3)]), &[0.92, 0.95, 0.75, 0.52, 0.45, 0.82, 0.48])
}

/// Dry scrub, dark scrub, dune sand, red rock (triplanar), cracked clay
/// (river beds), gravel, salt pan; cliffs are red rock.
pub fn desert(size: u32) -> Tileset {
    use Projection::*;
    let specs = vec![
        ("dry scrub", Recipe::Grass { base: [0.30, 0.24, 0.12], tip: [0.42, 0.36, 0.18] }, 24.0, 0.9, ByOrientation, false),
        ("dark scrub", Recipe::Grass { base: [0.14, 0.13, 0.06], tip: [0.28, 0.24, 0.10] }, 20.0, 1.1, ByOrientation, false),
        ("dune sand", Recipe::Sand { light: [0.78, 0.55, 0.30], dark: [0.55, 0.36, 0.18], ripples: 14.0 }, 40.0, 0.9, ByOrientation, true),
        ("red rock", Recipe::Rock { light: [0.55, 0.28, 0.15], dark: [0.22, 0.10, 0.06], strata: 9.0 }, 56.0, 2.2, Triplanar, true),
        ("cracked clay", Recipe::Cracked { clay: [0.50, 0.38, 0.26], crack: [0.18, 0.12, 0.08] }, 18.0, 1.4, ByOrientation, false),
        ("gravel", Recipe::Pebbles { stone: [0.45, 0.33, 0.24], fill: [0.36, 0.26, 0.17], cells: 22 }, 12.0, 1.0, ByOrientation, false),
        ("salt pan", Recipe::Cracked { clay: [0.40, 0.36, 0.30], crack: [0.24, 0.20, 0.15] }, 26.0, 0.6, ByOrientation, false),
    ];
    // dry scrub, dark scrub, dune sand, red rock, cracked clay, gravel, salt
    roughness(build("desert", size, specs, [m(0, 3), m(1, 3), m(2, 3), m(3, 3), m(4, 3), m(5, 3), m(6, 3), m(0, 3)]), &[0.92, 0.95, 0.78, 0.50, 0.68, 0.82, 0.62])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean absolute difference of horizontally (or vertically) adjacent
    /// texels of channel `c`, inside the image and across the wrap edge.
    fn neighbour_diffs(img: &[u8], side: usize, comps: usize, c: usize, vertical: bool) -> (f64, f64) {
        let at = |x: usize, y: usize| img[(y * side + x) * comps + c] as f64;
        let (mut inner, mut edge) = (0.0, 0.0);
        for i in 0..side {
            for j in 0..side - 1 {
                inner += if vertical { (at(i, j) - at(i, j + 1)).abs() } else { (at(j, i) - at(j + 1, i)).abs() };
            }
            edge += if vertical { (at(i, side - 1) - at(i, 0)).abs() } else { (at(side - 1, i) - at(0, i)).abs() };
        }
        (inner / (side * (side - 1)) as f64, edge / side as f64)
    }

    /// Every layer of both sets tiles: the step across the wrap edge is
    /// like a step inside, for albedo, height and normals (the GPU's 2x2
    /// box mips of a power-of-two tile never cross its edge, so they tile
    /// as mip 0 does); normals are unit vectors; generation is
    /// deterministic; the tables are valid.
    #[test]
    fn procedural_layers_tile_and_validate() {
        for name in NAMES {
            let t = by_name(name, 128).unwrap();
            t.validate().unwrap();
            assert_eq!(t.layers.len(), 7);
            let again = by_name(name, 128).unwrap();
            for (a, b) in t.layers.iter().zip(&again.layers) {
                assert!(a.albedo == b.albedo && a.normal == b.normal, "{name}: deterministic");
            }
            for l in &t.layers {
                for (img, comps, channels) in [(&l.albedo, 4, 0..4), (&l.normal, 2, 0..2)] {
                    for c in channels {
                        for vertical in [false, true] {
                            let (inner, edge) = neighbour_diffs(img, 128, comps, c, vertical);
                            assert!(edge <= inner * 2.5 + 1.0, "{name} `{}` channel {c} ({}): edge step {edge:.2} vs inside {inner:.2}", l.settings.name, if vertical { "v" } else { "u" });
                        }
                    }
                }
                // Heights use the range; normals are tilted somewhere.
                let h: Vec<u8> = l.albedo.chunks(4).map(|p| p[3]).collect();
                let (lo, hi) = (*h.iter().min().unwrap(), *h.iter().max().unwrap());
                assert!(hi - lo > 80, "{name} `{}`: heights {lo}..{hi}", l.settings.name);
                let tilt = l.normal.chunks(2).map(|p| snorm(p[0]).abs() + snorm(p[1]).abs()).fold(0.0f32, f32::max);
                assert!(tilt > 0.2, "{name} `{}`: normals tilt", l.settings.name);
                for p in l.normal.chunks(2) {
                    let (x, y) = (snorm(p[0]), snorm(p[1]));
                    assert!(x * x + y * y <= 1.0 + 0.02, "a unit normal's x, y");
                }
            }
        }
    }

    /// A broken edge (as an unwrapped generator would make) fails the
    /// tiling measure: the test above can fail.
    #[test]
    fn the_tiling_measure_sees_a_seam() {
        let t = temperate(64);
        let mut img = t.layers[3].albedo.clone();
        for y in 0..64 {
            let i = (y * 64 + 63) * 4;
            img[i] = img[i].wrapping_add(90);
        }
        let (inner, edge) = neighbour_diffs(&img, 64, 4, 0, false);
        assert!(edge > inner * 2.5 + 1.0, "{edge} vs {inner}");
    }

    #[test]
    fn broken_tilesets_are_rejected() {
        let mut t = temperate(16);
        t.materials[2].cliff = 9;
        assert!(t.validate().unwrap_err().contains("material 2"));
        let mut t = temperate(16);
        t.layers[0].normal.pop();
        assert!(t.validate().unwrap_err().contains("normal bytes"));
        let mut t = temperate(16);
        t.cliff_slope = (0.5, 0.2);
        assert!(t.validate().is_err());
        let mut t = temperate(16);
        t.layers[0].settings.roughness = 1.5;
        assert!(t.validate().unwrap_err().contains("roughness"));
        let u = temperate(16).ubo();
        assert_eq!(u.materials[4], [4, 3, 0, 0]);
        assert_eq!(u.layers[0][0], 1.0 / 24.0);
        assert_eq!(u.roughness[0][0], 0.92);
        assert_eq!(desert(16).ubo().layers[3][2], 3.0, "red rock is triplanar and directional");
        assert_eq!(desert(16).ubo().roughness[3][0], 0.50);
    }

    /// The table's anti-tiling: the tileset's degree, each layer's
    /// directional flag, and means over mip 0; out-of-range degrees
    /// rejected.
    #[test]
    fn anti_tiling_in_the_table() {
        let mut t = temperate(16);
        t.anti_tiling = 0.5;
        t.layers[0].settings.projection = Projection::Triplanar;
        t.layers[0].settings.directional = false;
        t.layers[1].settings.directional = true;
        let u = t.ubo();
        assert_eq!(u.anti_tiling[0], 0.5, "the tileset's degree");
        assert_eq!((u.layers[0][2], u.layers[1][2]), (1.0, 2.0), "triplanar, directional");
        // The mean over mip 0, colour in linear light.
        let texels = t.layers[4].albedo.chunks(4);
        let n = texels.len() as f32;
        let red = texels.map(|p| srgb_decode(p[0])).sum::<f32>() / n;
        assert!((u.means[4][0] - red).abs() < 1e-4, "{} vs {red}", u.means[4][0]);
        assert!(u.means[4][..3].iter().all(|&c| c > 0.0 && c < 1.0));
        for bad in [-0.1, 1.1, f32::NAN] {
            let mut t = temperate(16);
            t.anti_tiling = bad;
            assert!(t.validate().unwrap_err().contains("anti-tiling"), "{bad:?}");
        }
    }
}
