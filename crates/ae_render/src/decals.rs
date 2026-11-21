//! Decals: marks painted on the terrain (scorch marks, craters), applied by
//! the terrain's own fragment shader (`terrain_decals.glsl`), so nothing
//! else (units, water) is ever painted and there is no projection volume to
//! size.
//!
//! * A decal is a footprint on the heightfield: a centre (x, z; its height
//!   is the ground's there, read on the GPU, so it follows terrain edits),
//!   half-sizes along its own axes, a rotation, a layer of the
//!   [`DecalAtlas`], an opacity and a fade-in.
//! * Its layer also carries a material ([`LayerMaterial`]: GGX roughness and
//!   metallic), blended into the ground's by the decal's coverage, so soot
//!   takes the gloss off whatever it lands on.
//! * Channels: the albedo texture is RGB colour with coverage in alpha; the
//!   surface texture is normal x/y, roughness (times the layer's factor, as
//!   glTF's factor × texture) and the glow mask in alpha. Normal z is rebuilt
//!   in the shader; metallic stays a per-layer factor.
//! * It is unfolded over the ground from its centre: the plan ray to each
//!   point, read every heightmap cell, flattened segment by segment along
//!   the ground's slope. So a decal across a cliff's edge or at its foot
//!   wraps onto the face with its texels unstretched, and the image is
//!   continuous over any ground. Its extent is the square in unfolded
//!   metres.
//! * Binning: the world's cells of [`GRID_CELL`] metres, hashed into
//!   [`buckets`] (about four per decal, whatever the map's size), list the
//!   decals touching them; the shader loops over its bucket's list, oldest
//!   first, so newer decals cover older ones (another cell's decals in the
//!   same bucket fail its footprint test). A decal's bounding radius is clamped to
//!   [`MAX_RADIUS`], so it lists in at most 3x3 cells and the buffers have a
//!   fixed size.
//! * A ring of [`DecalsSetup::capacity`] decals: the oldest goes first.
//!   The terrain pipeline with decals is a specialization (constant 1) used
//!   only while there are any; with none the frame is the one without
//!   decals.
//!
//! Decals come from a [`DecalFeed`] (the client fills it from what the
//! observer sees, as the particles' bursts). A decal lives
//! [`Decal::life`] seconds (0: until evicted), fading out over its last
//! [`Decal::fade_out`]; it ends under the fog too, so a mark seen once may
//! be gone when the fog lifts.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Side of a binning cell (metres).
pub const GRID_CELL: f32 = 32.0;
/// Largest bounding radius of a decal (metres); bigger ones are scaled down.
pub const MAX_RADIUS: f32 = GRID_CELL;
/// Seconds a scorch mark lives (Recoil's scars: 0.5 to 60 s by the
/// explosion's damage, times the decal level setting).
pub const SCORCH_LIFE: f32 = 60.0;
/// Seconds a scar glows (Recoil's default glow TTL, 60 frames).
pub const SCAR_GLOW_LIFE: f32 = 2.0;
/// Most decals one binning cell lists (the newest are kept).
pub const MAX_PER_CELL: usize = 255;
/// Header words before the decals.
const HEADER_WORDS: usize = 4;
/// Words per decal.
const DECAL_WORDS: usize = 16;
/// Cells a decal lists in at most.
const CELLS_PER_DECAL: usize = 9;

/// The default atlas's layers.
pub mod layer {
    pub const SCORCH: u32 = 0;
    pub const CRATER: u32 = 1;
}

/// One decal to place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decal {
    /// World x, z of the centre.
    pub pos: [f32; 2],
    /// Half-sizes along the decal's own axes (metres).
    pub half: [f32; 2],
    /// Rotation about the vertical (radians).
    pub angle: f32,
    /// Atlas layer (out of range: drawn as nothing).
    pub layer: u32,
    pub opacity: f32,
    /// Seconds to fade in from placement (0: at once).
    pub fade_in: f32,
    /// Seconds it lives from placement (0: until evicted). It is removed
    /// when its life ends, under the fog too.
    pub life: f32,
    /// Seconds it fades out over at the end of its life (0: it vanishes;
    /// longer than `life`: it starts partly faded).
    pub fade_out: f32,
    /// Glow at placement, 0..1: the atlas's glow mask lit as hot metal.
    /// It falls by 1 every `glow_life` seconds (Recoil's decay: a glow of
    /// 0.5 ends halfway through), and the fade-in does not dim it.
    pub glow: f32,
    pub glow_life: f32,
    pub blend: Blend,
}

/// How a decal's colour meets the ground's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Blend {
    /// Its colour over the ground's, by its alpha.
    #[default]
    Replace,
    /// The ground's colour times the decal's (as Recoil's explosion scars,
    /// `2 * decal * ground` in gamma space): mid grey (sRGB 0.5) leaves
    /// the ground as it is, darker chars it, so a scar darkens whatever it
    /// lands on instead of painting one colour.
    Modulate,
}

impl Decal {
    /// A scorch mark of `radius` metres at `pos`, rotated by `seed`: it
    /// lives [`SCORCH_LIFE`] seconds, fading out over the last third.
    pub fn scorch(pos: [f32; 2], radius: f32, seed: u32) -> Self {
        let angle = (seed.wrapping_mul(0x9e37_79b9) >> 8) as f32 / (1u32 << 24) as f32 * std::f32::consts::TAU;
        Self { pos, half: [radius, radius], angle, layer: layer::SCORCH, opacity: 0.9, fade_in: 0.4, life: SCORCH_LIFE, fade_out: SCORCH_LIFE / 3.0, glow: 0.0, glow_life: 0.0, blend: Blend::Replace }
    }

    /// An explosion's scar on atlas layer `layer`, as Recoil sets one from
    /// the explosion's `damage` (`GroundDecalHandler`): opacity
    /// `clamp(2 damage / 255, 0.8, 1)`, glow `clamp(2 damage / 255, 0, 1)`
    /// for [`SCAR_GLOW_LIFE`] seconds. It lives [`SCORCH_LIFE`] (Recoil:
    /// `clamp(3 damage, 15, 1800)` frames times the decal level setting).
    pub fn scar(pos: [f32; 2], radius: f32, seed: u32, damage: f32, layer: u32, blend: Blend) -> Self {
        let k = 2.0 * damage / 255.0;
        Self { layer, blend, opacity: k.clamp(0.8, 1.0), glow: k.clamp(0.0, 1.0), glow_life: SCAR_GLOW_LIFE, ..Self::scorch(pos, radius, seed) }
    }

    /// A decal that stays until evicted, at full opacity at once.
    pub fn permanent(pos: [f32; 2], half: [f32; 2], angle: f32, layer: u32) -> Self {
        Self { pos, half, angle, layer, opacity: 1.0, fade_in: 0.0, life: 0.0, fade_out: 0.0, glow: 0.0, glow_life: 0.0, blend: Blend::Replace }
    }

    /// Bounding radius in plan.
    fn radius(&self) -> f32 {
        self.half[0].hypot(self.half[1])
    }
}

/// Where decals are queued for the render thread (taken every frame).
#[derive(Debug, Default)]
pub struct DecalFeed {
    queued: Mutex<Vec<Decal>>,
}

impl DecalFeed {
    pub fn place(&self, decal: Decal) {
        self.queued.lock().unwrap_or_else(|p| p.into_inner()).push(decal);
    }

    pub fn place_all(&self, decals: &[Decal]) {
        self.queued.lock().unwrap_or_else(|p| p.into_inner()).extend_from_slice(decals);
    }

    /// Every queued decal, oldest first (the render thread takes them each
    /// frame).
    pub fn take(&self) -> Vec<Decal> {
        std::mem::take(&mut *self.queued.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

/// How a decal layer's surface responds to light: the GGX roughness and
/// metallic of the material it paints (`docs/lighting.md`, "Speculars").
/// A decal blends them into the ground's by its coverage, so soot (a rough
/// dielectric) takes the gloss off whatever it lands on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerMaterial {
    /// 0 mirror, 1 matte.
    pub roughness: f32,
    /// 0 dielectric, 1 metal (the layer's colour is then its F0).
    pub metallic: f32,
}

impl Default for LayerMaterial {
    /// A matte dielectric: the safe default for soot and disturbed earth.
    fn default() -> Self {
        Self { roughness: 0.9, metallic: 0.0 }
    }
}

/// The decals' textures: square layers of `size`² texels, row-major, the
/// first row at the decal's -z edge; mips are made from them.
#[derive(Clone, Debug)]
pub struct DecalAtlas {
    pub size: u32,
    /// Straight (not premultiplied) linear RGBA.
    pub layers: Vec<Vec<[f32; 4]>>,
    /// Layer for layer, the tangent-space normal (x toward the decal's +x,
    /// y toward its +z, the first rows' side being -z; z out of the ground;
    /// each -1..1) and the glow mask (0..1). Empty: flat, no glow.
    pub normals: Vec<Vec<[f32; 4]>>,
    /// Layer for layer, an optional roughness texture (0..1, one value per
    /// texel): the shader multiplies it by the layer's
    /// [`LayerMaterial::roughness`]. Empty: the texture value is 1, so the
    /// factor alone sets the roughness.
    pub roughness: Vec<Vec<f32>>,
    /// Layer for layer, its surface material. Empty: every layer is
    /// [`LayerMaterial::default`].
    pub materials: Vec<LayerMaterial>,
}

impl DecalAtlas {
    /// Each layer's mips as sRGB RGBA8 bytes (alpha linear). Colour is
    /// averaged weighted by alpha, so transparent texels do not darken the
    /// edges of smaller mips.
    pub(crate) fn mips(&self) -> Vec<Vec<Vec<u8>>> {
        let encode = |c: f32| {
            let c = c.clamp(0.0, 1.0);
            let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
            (s * 255.0).round() as u8
        };
        self.layers
            .iter()
            .map(|base| {
                let mut level = base.clone();
                let mut side = self.size as usize;
                let mut mips = Vec::new();
                loop {
                    mips.push(level.iter().flat_map(|t| [encode(t[0]), encode(t[1]), encode(t[2]), (t[3].clamp(0.0, 1.0) * 255.0).round() as u8]).collect());
                    if side == 1 {
                        break;
                    }
                    let half = side / 2;
                    level = (0..half * half)
                        .map(|i| {
                            let (x, y) = (i % half * 2, i / half * 2);
                            let texels = [level[y * side + x], level[y * side + x + 1], level[(y + 1) * side + x], level[(y + 1) * side + x + 1]];
                            let a: f32 = texels.iter().map(|t| t[3]).sum();
                            let colour = |k: usize| if a > 0.0 { texels.iter().map(|t| t[k] * t[3]).sum::<f32>() / a } else { texels.iter().map(|t| t[k]).sum::<f32>() * 0.25 };
                            [colour(0), colour(1), colour(2), a * 0.25]
                        })
                        .collect();
                    side = half;
                }
                mips
            })
            .collect()
    }

    /// Each layer's surface as RGBA8 UNORM mips: R, G the tangent normal
    /// (`n * 0.5 + 0.5`; its z is reconstructed in the shader), B the
    /// roughness texture (multiplied by the layer's factor), A the glow
    /// mask. Normals are averaged in xyz and renormalised; roughness and
    /// the mask are averaged. Flat, roughness 1, no glow, when [`Self::normals`]
    /// is empty; roughness 1 where [`Self::roughness`] has no map.
    pub(crate) fn surface_mips(&self) -> Vec<Vec<Vec<u8>>> {
        let texels = (self.size * self.size) as usize;
        let flat = vec![[0.0, 0.0, 1.0, 0.0]; texels];
        let unorm = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
        let normal = |c: f32| unorm(c * 0.5 + 0.5);
        (0..self.layers.len())
            .map(|i| {
                let mut level = self.normals.get(i).unwrap_or(&flat).clone();
                let mut rough = self.roughness.get(i).cloned().unwrap_or_else(|| vec![1.0; texels]);
                let mut side = self.size as usize;
                let mut mips = Vec::new();
                loop {
                    mips.push(level.iter().zip(&rough).flat_map(|(t, r)| [normal(t[0]), normal(t[1]), unorm(*r), unorm(t[3])]).collect());
                    if side == 1 {
                        break;
                    }
                    let half = side / 2;
                    level = (0..half * half)
                        .map(|i| {
                            let (x, y) = (i % half * 2, i / half * 2);
                            let texels = [level[y * side + x], level[y * side + x + 1], level[(y + 1) * side + x], level[(y + 1) * side + x + 1]];
                            let sum = |k: usize| texels.iter().map(|t| t[k]).sum::<f32>();
                            let n = [sum(0), sum(1), sum(2)];
                            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                            let n = if len > 1e-6 { n.map(|c| c / len) } else { [0.0, 0.0, 1.0] };
                            [n[0], n[1], n[2], sum(3) * 0.25]
                        })
                        .collect();
                    rough = (0..half * half)
                        .map(|i| {
                            let (x, y) = (i % half * 2, i / half * 2);
                            (rough[y * side + x] + rough[y * side + x + 1] + rough[(y + 1) * side + x] + rough[(y + 1) * side + x + 1]) * 0.25
                        })
                        .collect();
                    side = half;
                }
                mips
            })
            .collect()
    }

    /// The atlas at `size`² texels per layer ([`resample`]).
    pub fn resized(&self, size: u32) -> Self {
        let from = self.size;
        let colour = |l: &Vec<[f32; 4]>| resample(l, from, size, false);
        Self {
            size,
            layers: self.layers.iter().map(colour).collect(),
            normals: self.normals.iter().map(|l| resample(l, from, size, true)).collect(),
            roughness: self.roughness.iter().map(|m| resample_scalar(m, from, size)).collect(),
            materials: self.materials.clone(),
        }
    }

    /// Set layer `layer`'s surface material (roughness and metallic). Every
    /// other layer keeps its own (or the default).
    pub fn set_layer_material(&mut self, layer: u32, material: LayerMaterial) -> Result<(), String> {
        let n = self.layers.len();
        if layer as usize >= n {
            return Err(format!("material for decal layer {layer}, but the atlas has {n}"));
        }
        if self.materials.is_empty() {
            self.materials = vec![LayerMaterial::default(); n];
        }
        self.materials.resize(n, LayerMaterial::default());
        self.materials[layer as usize] = material;
        Ok(())
    }

    /// The per-layer materials, packed for the shader: x roughness factor,
    /// y metallic factor, z specular scale (reserved, 1), w reserved. A
    /// missing entry is [`LayerMaterial::default`].
    pub(crate) fn gpu_materials(&self) -> Vec<[f32; 4]> {
        (0..self.layers.len())
            .map(|i| {
                let m = self.materials.get(i).copied().unwrap_or_default();
                [m.roughness.clamp(0.0, 1.0), m.metallic.clamp(0.0, 1.0), 1.0, 0.0]
            })
            .collect()
    }

    /// Add a layer from 8-bit images, rows top first (as [`read_tga`]
    /// returns them; the top row is the decal's -z edge), resampled to the
    /// atlas's size: `colour` sRGB with straight alpha, `normals` a tangent
    /// normal map (rgb, x right) with the glow mask in alpha and whether
    /// its green points up the image (OpenGL style; flipped to the atlas's
    /// convention, green down the rows). Returns the layer.
    pub fn add_layer(&mut self, colour: &Image, normals: Option<(&Image, bool)>) -> Result<u32, String> {
        let linear = |c: u8| {
            let c = c as f32 / 255.0;
            if c <= 0.040_45 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
        };
        let square = |img: &Image| -> Result<(), String> {
            if img.width != img.height || !img.width.is_power_of_two() {
                return Err(format!("decal image {}x{} is not a power-of-two square", img.width, img.height));
            }
            Ok(())
        };
        square(colour)?;
        let texels: Vec<[f32; 4]> = colour.texels.iter().map(|t| [linear(t[0]), linear(t[1]), linear(t[2]), t[3] as f32 / 255.0]).collect();
        let layer = self.layers.len() as u32;
        let normal_layer = match normals {
            Some((img, up)) => {
                square(img)?;
                let unorm = |c: u8| c as f32 / 255.0 * 2.0 - 1.0;
                let ys = if up { -1.0 } else { 1.0 };
                Some(resample(&img.texels.iter().map(|t| [unorm(t[0]), ys * unorm(t[1]), unorm(t[2]), t[3] as f32 / 255.0]).collect::<Vec<_>>(), img.width, self.size, true))
            }
            None => None,
        };
        // Normals for every layer once any layer has them.
        let flat = || vec![[0.0, 0.0, 1.0, 0.0]; (self.size * self.size) as usize];
        if normal_layer.is_some() && self.normals.is_empty() {
            self.normals = (0..self.layers.len()).map(|_| flat()).collect();
        }
        if !self.normals.is_empty() {
            self.normals.push(normal_layer.unwrap_or_else(flat));
        }
        if !self.materials.is_empty() {
            self.materials.push(LayerMaterial::default());
        }
        if !self.roughness.is_empty() {
            self.roughness.push(vec![1.0; (self.size * self.size) as usize]);
        }
        self.layers.push(resample(&texels, colour.width, self.size, false));
        Ok(layer)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.size.is_power_of_two() {
            return Err(format!("decal atlas size {} is not a power of two", self.size));
        }
        if self.layers.is_empty() {
            return Err("decal atlas has no layers".into());
        }
        if !self.normals.is_empty() && self.normals.len() != self.layers.len() {
            return Err(format!("decal atlas has {} normal layers for {} colour layers", self.normals.len(), self.layers.len()));
        }
        if !self.materials.is_empty() && self.materials.len() != self.layers.len() {
            return Err(format!("decal atlas has {} material layers for {} colour layers", self.materials.len(), self.layers.len()));
        }
        if !self.roughness.is_empty() && self.roughness.len() != self.layers.len() {
            return Err(format!("decal atlas has {} roughness layers for {} colour layers", self.roughness.len(), self.layers.len()));
        }
        if let Some((i, m)) = self.materials.iter().enumerate().find(|(_, m)| !(0.0..=1.0).contains(&m.roughness) || !(0.0..=1.0).contains(&m.metallic)) {
            return Err(format!("decal atlas layer {i} material roughness {} or metallic {} out of range", m.roughness, m.metallic));
        }
        let texels = (self.size * self.size) as usize;
        if let Some(i) = self.layers.iter().position(|l| l.len() != texels) {
            return Err(format!("decal atlas layer {i} has {} texels, not {texels}", self.layers[i].len()));
        }
        if let Some(i) = self.normals.iter().position(|l| l.len() != texels) {
            return Err(format!("decal atlas normal layer {i} has {} texels, not {texels}", self.normals[i].len()));
        }
        if let Some((i, m)) = self.roughness.iter().enumerate().find(|(_, m)| m.len() != texels || m.iter().any(|v| !(0.0..=1.0).contains(v))) {
            return Err(format!("decal atlas roughness layer {i} has {} texels (not {texels}) or a value out of range", m.len()));
        }
        Ok(())
    }
}

/// An 8-bit RGBA image, rows top first.
#[derive(Clone, Debug)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<[u8; 4]>,
}

/// Decode a 32-bit TGA (uncompressed or run-length encoded, either row
/// order), as Recoil's scars are stored.
pub fn read_tga(bytes: &[u8]) -> Result<Image, String> {
    let header = bytes.get(..18).ok_or("TGA: short header")?;
    let (id_len, kind, bpp, descriptor) = (header[0] as usize, header[2], header[16], header[17]);
    let (width, height) = (u16::from_le_bytes([header[12], header[13]]) as u32, u16::from_le_bytes([header[14], header[15]]) as u32);
    if header[1] != 0 || !(kind == 2 || kind == 10) || bpp != 32 {
        return Err(format!("TGA: only 32-bit true colour is read (colour map {}, type {kind}, {bpp} bits)", header[1]));
    }
    let n = (width * height) as usize;
    let mut data = bytes.get(18 + id_len..).ok_or("TGA: short data")?;
    let mut bgra: Vec<[u8; 4]> = Vec::with_capacity(n);
    let texel = |d: &[u8]| -> Result<[u8; 4], String> { d.get(..4).map(|t| [t[2], t[1], t[0], t[3]]).ok_or_else(|| "TGA: data ends early".to_string()) };
    while bgra.len() < n {
        if kind == 2 {
            bgra.push(texel(data)?);
            data = &data[4..];
            continue;
        }
        let (&c, rest) = data.split_first().ok_or("TGA: data ends early")?;
        let count = (c & 127) as usize + 1;
        if c & 128 != 0 {
            let t = texel(rest)?;
            bgra.extend(std::iter::repeat_n(t, count));
            data = &rest[4..];
        } else {
            for k in 0..count {
                bgra.push(texel(&rest[4 * k..])?);
            }
            data = &rest[4 * count..];
        }
    }
    bgra.truncate(n);
    // Bit 5: rows top first; otherwise bottom first.
    if descriptor & 32 == 0 {
        bgra = bgra.chunks(width as usize).rev().flatten().copied().collect();
    }
    Ok(Image { width, height, texels: bgra })
}

/// Whether a tangent normal map's green points up the image (OpenGL/glTF
/// style), from the curl of its red and green channels over the texels of
/// `colour` whose alpha is at least 0.3. A positive curl means green down
/// the rows (DirectX style), a negative one green up. A map with too little
/// signal (flat, or two images of different sizes) is taken as green up,
/// glTF's convention.
pub fn normal_green_up(colour: &Image, normals: &Image) -> bool {
    let (w, h) = (normals.width as usize, normals.height as usize);
    if colour.width as usize != w || colour.height as usize != h || w < 3 || h < 3 {
        return true;
    }
    let normal = |x: usize, y: usize| {
        let t = normals.texels[y * w + x];
        (t[0] as f64 / 127.5 - 1.0, t[1] as f64 / 127.5 - 1.0)
    };
    let (mut sum, mut nx2, mut ny2) = (0.0f64, 0.0f64, 0.0f64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if colour.texels[y * w + x][3] < 76 {
                continue;
            }
            let (nx0, _) = normal(x, y - 1);
            let (nx1, _) = normal(x, y + 1);
            let (_, ny0) = normal(x - 1, y);
            let (_, ny1) = normal(x + 1, y);
            let dx = nx1 - nx0;
            let dy = ny1 - ny0;
            sum += dx * dy;
            nx2 += dx * dx;
            ny2 += dy * dy;
        }
    }
    if nx2 * ny2 <= 1e-12 {
        return true;
    }
    sum / (nx2 * ny2).sqrt() < 0.0
}

/// A square layer of `from`² texels at `to`²: box-filtered when shrinking
/// by a whole factor, bilinear otherwise; `normals` renormalizes xyz.
pub fn resample(texels: &[[f32; 4]], from: u32, to: u32, normals: bool) -> Vec<[f32; 4]> {
    let (f, t) = (from as usize, to as usize);
    let at = |x: usize, y: usize| texels[y.min(f - 1) * f + x.min(f - 1)];
    let mut out: Vec<[f32; 4]> = (0..t * t)
        .map(|i| {
            let (x, y) = (i % t, i / t);
            if f >= t && f % t == 0 {
                let k = f / t;
                let mut sum = [0.0f32; 4];
                for dy in 0..k {
                    for dx in 0..k {
                        let s = at(x * k + dx, y * k + dy);
                        (0..4).for_each(|c| sum[c] += s[c]);
                    }
                }
                sum.map(|c| c / (k * k) as f32)
            } else {
                let (u, v) = (((x as f32 + 0.5) * f as f32 / t as f32 - 0.5).max(0.0), ((y as f32 + 0.5) * f as f32 / t as f32 - 0.5).max(0.0));
                let (x0, y0, fx, fy) = (u as usize, v as usize, u.fract(), v.fract());
                let (a, b, c, d) = (at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1));
                std::array::from_fn(|k| (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy)
            }
        })
        .collect();
    if normals {
        for n in &mut out {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-6 {
                (0..3).for_each(|c| n[c] /= len);
            } else {
                *n = [0.0, 0.0, 1.0, n[3]];
            }
        }
    }
    out
}

/// [`resample`] for a one-channel map (roughness).
fn resample_scalar(values: &[f32], from: u32, to: u32) -> Vec<f32> {
    let packed: Vec<[f32; 4]> = values.iter().map(|&v| [v, 0.0, 0.0, 0.0]).collect();
    resample(&packed, from, to, false).iter().map(|t| t[0]).collect()
}

/// The engine's decals ([`layer`]), 128² each: a scorch mark (charred
/// centre, a burnt brown ring, a ragged edge; a shallow rough hollow whose
/// centre can glow) and a crater (a dark pit, a pale rim of thrown-up
/// earth, in relief).
pub fn default_atlas() -> DecalAtlas {
    const SIZE: usize = 128;
    fn hash(x: i32, y: i32, octave: i32) -> f32 {
        let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ (octave as u32).wrapping_mul(0xcb1a_b31f);
        h ^= h >> 13;
        h = h.wrapping_mul(0x5bd1_e995);
        h ^= h >> 15;
        (h & 0xffff) as f32 / 65535.0
    }
    fn value_noise(x: f32, y: f32, octave: i32) -> f32 {
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let (fx, fy) = (x - x.floor(), y - y.floor());
        let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
        let top = hash(ix, iy, octave) + (hash(ix + 1, iy, octave) - hash(ix, iy, octave)) * sx;
        let bottom = hash(ix, iy + 1, octave) + (hash(ix + 1, iy + 1, octave) - hash(ix, iy + 1, octave)) * sx;
        top + (bottom - top) * sy
    }
    let fbm = |u: f32, v: f32, seed: i32| {
        let (mut n, mut amplitude) = (0.0, 0.5);
        for octave in 0..4 {
            let f = 6.0 * (1 << octave) as f32;
            n += amplitude * value_noise(u * f, v * f, octave + seed);
            amplitude *= 0.5;
        }
        n / 0.9375
    };
    let smooth = |e0: f32, e1: f32, x: f32| {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let mix = |a: [f32; 3], b: [f32; 3], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    let layer = |f: &dyn Fn(f32, f32, f32, f32) -> [f32; 4]| -> Vec<[f32; 4]> {
        (0..SIZE * SIZE)
            .map(|i| {
                let (u, v) = (((i % SIZE) as f32 + 0.5) / SIZE as f32, ((i / SIZE) as f32 + 0.5) / SIZE as f32);
                let (x, y) = (u * 2.0 - 1.0, v * 2.0 - 1.0);
                f(u, v, x.hypot(y), y.atan2(x))
            })
            .collect()
    };
    let scalar = |f: &dyn Fn(f32, f32, f32, f32) -> f32| -> Vec<f32> {
        (0..SIZE * SIZE)
            .map(|i| {
                let (u, v) = (((i % SIZE) as f32 + 0.5) / SIZE as f32, ((i / SIZE) as f32 + 0.5) / SIZE as f32);
                let (x, y) = (u * 2.0 - 1.0, v * 2.0 - 1.0);
                f(u, v, x.hypot(y), y.atan2(x))
            })
            .collect()
    };
    let char_ = [0.025, 0.022, 0.02];
    let burnt = [0.12, 0.075, 0.04];
    let scorch = layer(&|u, v, r, a| {
        let n = fbm(u, v, 0);
        // A ragged edge: the radius wobbles with the angle and the noise.
        let edge = 0.78 + 0.03 * (a * 5.0).sin() + (n - 0.5) * 0.35;
        let alpha = 1.0 - smooth(edge - 0.25, edge, r);
        let colour = mix(char_, burnt, smooth(0.15, edge * 0.9, r + (n - 0.5) * 0.3));
        [colour[0], colour[1], colour[2], alpha * (0.75 + 0.25 * n)]
    });
    let crater = layer(&|u, v, r, _| {
        let n = fbm(u, v, 7);
        let pit = [0.05, 0.04, 0.035];
        let rim = [0.36, 0.30, 0.22];
        let rr = r + (n - 0.5) * 0.18;
        let colour = mix(pit, rim, smooth(0.35, 0.6, rr));
        let alpha = 1.0 - smooth(0.7, 0.95, rr);
        [colour[0], colour[1], colour[2], alpha]
    });
    // Relief: the scorch a shallow blast hollow, rough with cinders; the
    // crater a pit inside a raised rim. Normals from the heights (in
    // texels of depth per texel), and the scorch's glow mask: its centre,
    // cracked by the noise.
    let scorch_height = |u: f32, v: f32, r: f32| -2.5 * (1.0 - smooth(0.0, 0.7, r)) + 2.0 * fbm(u * 2.0, v * 2.0, 3);
    let crater_height = |u: f32, v: f32, r: f32| {
        let rr = r + (fbm(u, v, 7) - 0.5) * 0.18;
        -5.0 * (1.0 - smooth(0.2, 0.55, rr)) + 3.0 * smooth(0.45, 0.6, rr) * (1.0 - smooth(0.6, 0.9, rr))
    };
    let normals = |height: &dyn Fn(f32, f32, f32) -> f32, glow: &dyn Fn(f32, f32, f32) -> f32| -> Vec<[f32; 4]> {
        let h = |x: f32, y: f32| {
            let (u, v) = (x / SIZE as f32, y / SIZE as f32);
            height(u, v, (u * 2.0 - 1.0).hypot(v * 2.0 - 1.0))
        };
        (0..SIZE * SIZE)
            .map(|i| {
                let (x, y) = ((i % SIZE) as f32 + 0.5, (i / SIZE) as f32 + 0.5);
                let (dx, dy) = ((h(x + 1.0, y) - h(x - 1.0, y)) * 0.5, (h(x, y + 1.0) - h(x, y - 1.0)) * 0.5);
                let len = (dx * dx + dy * dy + 1.0).sqrt();
                let (u, v) = (x / SIZE as f32, y / SIZE as f32);
                [-dx / len, -dy / len, 1.0 / len, glow(u, v, (u * 2.0 - 1.0).hypot(v * 2.0 - 1.0))]
            })
            .collect()
    };
    let scorch_glow = |u: f32, v: f32, r: f32| ((1.0 - smooth(0.05, 0.6, r)) * (0.35 + 0.8 * fbm(u * 1.5, v * 1.5, 11))).clamp(0.0, 0.85);
    let normals = vec![normals(&scorch_height, &scorch_glow), normals(&crater_height, &|_, _, _| 0.0)];
    // Roughness textures (multipliers on the layer's factor): char and
    // cinders are rougher than the burnt ring, the crater's pit than its
    // rim. 8 bits is plenty for both.
    let scorch_rough = scalar(&|u, v, r, a| {
        let n = fbm(u, v, 0);
        let edge = 0.78 + 0.03 * (a * 5.0).sin() + (n - 0.5) * 0.35;
        1.0 - 0.1 * smooth(0.15, edge * 0.9, r + (n - 0.5) * 0.3)
    });
    let crater_rough = scalar(&|u, v, r, _| {
        let n = fbm(u, v, 7);
        1.0 - 0.1 * smooth(0.35, 0.6, r + (n - 0.5) * 0.18)
    });
    // Char and ash are matte dielectrics; the crater's loose earth less so.
    let materials = vec![LayerMaterial { roughness: 0.95, metallic: 0.0 }, LayerMaterial { roughness: 0.9, metallic: 0.0 }];
    DecalAtlas { size: SIZE as u32, layers: vec![scorch, crater], normals, roughness: vec![scorch_rough, crater_rough], materials }
}

pub struct DecalsSetup {
    pub feed: Arc<DecalFeed>,
    /// Most decals at once (the oldest is evicted first).
    pub capacity: u32,
    /// The textures; [`default_atlas`] when `None`.
    pub atlas: Option<Arc<DecalAtlas>>,
}

impl DecalsSetup {
    pub fn new(feed: Arc<DecalFeed>, capacity: u32) -> Self {
        Self { feed, capacity, atlas: None }
    }
}

/// Buckets of the spatial hash for `capacity` decals: a power of two,
/// about four per decal (a decal lists in up to nine cells).
pub fn buckets(capacity: u32) -> u32 {
    (capacity * 4).next_power_of_two().max(64)
}

/// Words of the GPU buffer for `capacity` decals: header, decals, buckets
/// (first index << 8 | count), indices.
pub fn buffer_words(capacity: u32) -> usize {
    HEADER_WORDS + capacity as usize * (DECAL_WORDS + CELLS_PER_DECAL) + buckets(capacity) as usize
}

/// The bucket of world cell `(x, z)` ([`GRID_CELL`] metres; the shader
/// hashes the same way).
pub fn bucket(x: i32, z: i32, mask: u32) -> usize {
    ((x as u32).wrapping_mul(73_856_093) ^ (z as u32).wrapping_mul(19_349_663)) as usize & mask as usize
}

/// A placed decal: its words in the buffer and the buckets it lists in
/// (for the set's table size), computed once.
#[derive(Clone, Copy, Debug)]
struct Placed {
    words: [u32; DECAL_WORDS],
    buckets: [u32; CELLS_PER_DECAL],
    n_buckets: u8,
    /// Centre x, z and bounding radius in plan.
    bounds: [f32; 3],
    /// When its life ends (0: never).
    end: f32,
}

/// The live decals, oldest first, and a version bumped by every change.
#[derive(Debug)]
pub(crate) struct DecalSet {
    capacity: usize,
    live: VecDeque<Placed>,
    pub version: u64,
    /// Kept between [`Self::encode`] calls.
    words: Vec<u32>,
    count: Vec<u32>,
    cursor: Vec<u32>,
}

impl DecalSet {
    pub fn new(capacity: u32) -> Self {
        Self { capacity: capacity as usize, live: VecDeque::new(), version: 0, words: Vec::new(), count: Vec::new(), cursor: Vec::new() }
    }

    /// Place `decal` at `time` (seconds, the frame's animation time).
    pub fn insert(&mut self, decal: Decal, time: f32) {
        if self.capacity == 0 {
            return;
        }
        let mut decal = decal;
        let r = decal.radius();
        if r > MAX_RADIUS {
            let k = MAX_RADIUS / r;
            decal.half = decal.half.map(|h| h * k);
        }
        if self.live.len() == self.capacity {
            self.live.pop_front();
        }
        let d = &decal;
        let (s, c) = d.angle.sin_cos();
        let fade = if d.fade_in > 0.0 { 1.0 / d.fade_in } else { 0.0 };
        // The end of its life, and 1 / its fade-out (0: lives until evicted;
        // no fade-out: a step a microsecond wide).
        let end = if d.life > 0.0 { time + d.life } else { 0.0 };
        let fade_out = if d.life > 0.0 { 1.0 / d.fade_out.max(1e-6) } else { 0.0 };
        // 1 / the glow's life (0: no glow).
        let glow_inv = if d.glow > 0.0 && d.glow_life > 0.0 { 1.0 / d.glow_life } else { 0.0 };
        let blend = match d.blend {
            Blend::Replace => 0.0,
            Blend::Modulate => 1.0,
        };
        let words = [
            d.pos[0],
            d.pos[1],
            d.half[0].max(1e-3),
            d.half[1].max(1e-3),
            c,
            s,
            d.layer as f32,
            d.opacity,
            time,
            fade,
            end,
            fade_out,
            d.glow.clamp(0.0, 1.0),
            0.0,
            glow_inv,
            blend,
        ]
        .map(f32::to_bits);
        // The buckets of the distinct cells it touches (two of its cells
        // may share a bucket: listed once).
        let mask = buckets(self.capacity as u32) - 1;
        let r = d.radius();
        let lo = |v: f32| ((v - r) / GRID_CELL).floor() as i32;
        let hi = |v: f32| ((v + r) / GRID_CELL).floor() as i32;
        let mut list = [0u32; CELLS_PER_DECAL];
        let mut n = 0;
        for z in lo(d.pos[1])..=hi(d.pos[1]) {
            for x in lo(d.pos[0])..=hi(d.pos[0]) {
                let b = bucket(x, z, mask) as u32;
                if !list[..n].contains(&b) {
                    list[n] = b;
                    n += 1;
                }
            }
        }
        self.live.push_back(Placed { words, buckets: list, n_buckets: n as u8, bounds: [d.pos[0], d.pos[1], r], end });
        self.version += 1;
    }

    /// Remove the decals whose life has ended by `time`.
    pub fn expire(&mut self, time: f32) {
        let before = self.live.len();
        self.live.retain(|p| p.end <= 0.0 || p.end > time);
        if self.live.len() != before {
            self.version += 1;
        }
    }

    /// Whether any decal's box (its square in plan, `heights` in the
    /// vertical) is inside every plane (`dot(n, p) + d >= 0`) of a view.
    pub fn any_in_view(&self, planes: &[[f32; 4]; 6], heights: (f32, f32)) -> bool {
        self.live.iter().any(|p| {
            let [x, z, r] = p.bounds;
            planes.iter().all(|pl| {
                // The box's corner farthest along the plane's normal.
                let px = x + r * pl[0].signum();
                let py = if pl[1] >= 0.0 { heights.1 } else { heights.0 };
                let pz = z + r * pl[2].signum();
                pl[0] * px + pl[1] * py + pl[2] * pz + pl[3] >= 0.0
            })
        })
    }

    /// The GPU buffer's words ([`buffer_words`]; the unused tail of the
    /// index list left out). A counting sort into buffers kept between
    /// calls.
    pub fn encode(&mut self) -> &[u32] {
        let cap = self.capacity;
        let mask = buckets(cap as u32) - 1;
        let n_buckets = mask as usize + 1;
        let buckets_at = HEADER_WORDS + cap * DECAL_WORDS;
        let index_at = buckets_at + n_buckets;
        self.words.resize(buffer_words(cap as u32), 0);
        self.count.clear();
        self.count.resize(n_buckets, 0);
        let words = &mut self.words;
        words[..HEADER_WORDS].copy_from_slice(&[mask, 0, GRID_CELL.to_bits(), cap as u32]);
        for (i, p) in self.live.iter().enumerate() {
            let at = HEADER_WORDS + i * DECAL_WORDS;
            words[at..at + DECAL_WORDS].copy_from_slice(&p.words);
            for &b in &p.buckets[..p.n_buckets as usize] {
                self.count[b as usize] += 1;
            }
        }
        // Each bucket's first index; `count` becomes the listings to skip
        // (a full bucket keeps its newest decals, still oldest first).
        self.cursor.clear();
        let mut next = 0u32;
        for (b, n) in self.count.iter_mut().enumerate() {
            let kept = (*n).min(MAX_PER_CELL as u32);
            words[buckets_at + b] = next << 8 | kept;
            self.cursor.push(next);
            next += kept;
            *n -= kept;
        }
        for (i, p) in self.live.iter().enumerate() {
            for &b in &p.buckets[..p.n_buckets as usize] {
                let b = b as usize;
                if self.count[b] > 0 {
                    self.count[b] -= 1;
                } else {
                    words[index_at + self.cursor[b] as usize] = i as u32;
                    self.cursor[b] += 1;
                }
            }
        }
        &self.words[..index_at + next as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decal(x: f32, z: f32, r: f32) -> Decal {
        Decal::permanent([x, z], [r, r], 0.0, 0)
    }

    /// The decals listed in the bucket of world cell `(x, z)`.
    fn listed(words: &[u32], cap: u32, x: i32, z: i32) -> Vec<u32> {
        let mask = buckets(cap) - 1;
        let buckets_at = HEADER_WORDS + cap as usize * DECAL_WORDS;
        let index_at = buckets_at + mask as usize + 1;
        let e = words[buckets_at + bucket(x, z, mask)];
        let (first, n) = ((e >> 8) as usize, (e & 255) as usize);
        words[index_at + first..index_at + first + n].to_vec()
    }

    #[test]
    fn decals_list_in_the_cells_they_touch_oldest_first_and_the_oldest_is_evicted() {
        let mut set = DecalSet::new(3);
        set.insert(decal(10.0, 10.0, 4.0), 0.0); // cell (0, 0)
        set.insert(decal(32.0, 32.0, 4.0), 0.0); // the corner of four cells
        set.insert(decal(40.0, 10.0, 100.0), 0.0); // clamped to MAX_RADIUS
        let words = set.encode().to_vec();
        assert!(words.len() <= buffer_words(3));
        // Distinct buckets for these cells at this table size.
        let cells = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (-1, 0), (0, -1), (3, 0), (0, 2), (5, 5)];
        let mask = buckets(3) - 1;
        let mut seen: Vec<usize> = cells.iter().map(|&(x, z)| bucket(x, z, mask)).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), cells.len(), "test cells collide");
        assert_eq!(listed(&words, 3, 0, 0), vec![0, 1, 2]);
        for (x, z) in [(1, 0), (0, 1), (1, 1)] {
            assert_eq!(listed(&words, 3, x, z), vec![1, 2], "cell {x},{z}");
        }
        // Radius 32 about (40, 10): x 8..72 (cells 0..=2), z -22..42 (-1..=1).
        for (x, z) in [(2, 0), (2, 1), (0, -1)] {
            assert_eq!(listed(&words, 3, x, z), vec![2], "cell {x},{z}");
        }
        assert_eq!(listed(&words, 3, -1, 0), Vec::<u32>::new());
        assert_eq!(listed(&words, 3, 3, 0), Vec::<u32>::new());
        assert_eq!(listed(&words, 3, 0, 2), Vec::<u32>::new());
        let half = f32::from_bits(words[HEADER_WORDS + 2 * DECAL_WORDS + 2]);
        assert!((half * std::f32::consts::SQRT_2 - MAX_RADIUS).abs() < 1e-3, "{half}");

        set.insert(decal(170.0, 170.0, 4.0), 1.0);
        let words = set.encode().to_vec();
        // The first decal went; the others moved down one.
        assert_eq!(listed(&words, 3, 0, 0), vec![0, 1]);
        assert_eq!(listed(&words, 3, 5, 5), vec![2]);
        assert_eq!(set.version, 4);
    }

    #[test]
    fn decals_expire_at_the_end_of_their_life() {
        let mut set = DecalSet::new(4);
        set.insert(decal(0.0, 0.0, 4.0), 0.0);
        set.insert(Decal { life: 2.0, fade_out: 0.5, ..decal(1.0, 0.0, 4.0) }, 1.0);
        set.insert(Decal { life: 1.0, ..decal(2.0, 0.0, 4.0) }, 1.0);
        let words = set.encode().to_vec();
        let at = |i: usize, k: usize| f32::from_bits(words[HEADER_WORDS + i * DECAL_WORDS + k]);
        assert_eq!((at(0, 10), at(0, 11)), (0.0, 0.0), "permanent");
        assert_eq!((at(1, 10), at(1, 11)), (3.0, 2.0), "ends at 3, fades over 0.5 s");
        assert_eq!(at(2, 10), 2.0);
        assert!(at(2, 11) >= 1e5, "no fade-out: a step");
        let v = set.version;
        set.expire(1.9);
        assert_eq!((set.live.len(), set.version), (3, v), "nothing ended yet");
        set.expire(2.0);
        assert_eq!((set.live.len(), set.version), (2, v + 1));
        set.expire(3.5);
        assert_eq!(set.live.len(), 1);
        assert_eq!(set.live[0].bounds[0], 0.0, "the permanent one stays");
    }

    #[test]
    fn decals_out_of_view_are_not_in_view() {
        let mut set = DecalSet::new(4);
        // A view of x in [0, 100] (two planes; the rest always pass).
        let mut planes = [[0.0, 0.0, 0.0, 1.0]; 6];
        planes[0] = [1.0, 0.0, 0.0, 0.0];
        planes[1] = [-1.0, 0.0, 0.0, 100.0];
        assert!(!set.any_in_view(&planes, (0.0, 10.0)));
        set.insert(decal(-10.0, 0.0, 4.0), 0.0);
        assert!(!set.any_in_view(&planes, (0.0, 10.0)), "5.7 m short of the view");
        set.insert(decal(-4.0, 0.0, 4.0), 0.0);
        assert!(set.any_in_view(&planes, (0.0, 10.0)), "its square reaches x = 0");
    }

    #[test]
    fn a_full_bucket_keeps_its_newest_decals() {
        let n = MAX_PER_CELL + 5;
        let mut set = DecalSet::new(n as u32);
        for _ in 0..n {
            set.insert(decal(16.0, 16.0, 2.0), 0.0);
        }
        let words = set.encode().to_vec();
        assert_eq!(listed(&words, n as u32, 0, 0), (5..n as u32).collect::<Vec<_>>());
    }

    #[test]
    fn atlas_mips_reach_one_texel_and_ignore_transparent_colour() {
        let mut layer = vec![[1.0, 0.0, 0.0, 1.0]; 4];
        layer[3] = [0.0, 0.0, 1.0, 0.0];
        let atlas = DecalAtlas { size: 2, layers: vec![layer], normals: Vec::new(), roughness: Vec::new(), materials: Vec::new() };
        let mips = atlas.mips();
        assert_eq!(mips[0].len(), 2);
        assert_eq!(mips[0][1], vec![255, 0, 0, 191]);
        assert!(default_atlas().validate().is_ok());
    }

    #[test]
    fn layers_carry_their_material_and_default_to_matte() {
        let plain = DecalAtlas { size: 2, layers: vec![vec![[0.0; 4]; 4]; 2], normals: Vec::new(), roughness: Vec::new(), materials: Vec::new() };
        assert_eq!(plain.gpu_materials(), vec![[0.9, 0.0, 1.0, 0.0]; 2], "without materials: matte dielectric");
        let mut atlas = default_atlas();
        assert_eq!(atlas.materials.len(), atlas.layers.len());
        assert_eq!(atlas.roughness.len(), atlas.layers.len(), "the default atlas has texture roughness");
        let scorch = atlas.materials[layer::SCORCH as usize];
        assert!((scorch.roughness - 0.95).abs() < 1e-6 && scorch.metallic == 0.0);
        assert_eq!(atlas.materials[layer::CRATER as usize].metallic, 0.0);
        // A new layer takes the default until it is told otherwise, and a
        // 1.0 roughness texture while the atlas has roughness maps.
        let grey = Image { width: 4, height: 4, texels: vec![[128; 4]; 16] };
        let added = atlas.add_layer(&grey, None).unwrap();
        assert_eq!(atlas.materials[added as usize], LayerMaterial::default());
        assert_eq!(atlas.roughness[added as usize].len(), 128 * 128);
        assert!(atlas.roughness[added as usize].iter().all(|&v| v == 1.0));
        atlas.set_layer_material(added, LayerMaterial { roughness: 0.2, metallic: 1.0 }).unwrap();
        assert_eq!(atlas.gpu_materials()[added as usize], [0.2, 1.0, 1.0, 0.0]);
        assert!(atlas.set_layer_material(99, LayerMaterial::default()).is_err());
        assert_eq!(atlas.resized(8).materials, atlas.materials);
        atlas.materials[0].metallic = 1.5;
        assert!(atlas.validate().unwrap_err().contains("out of range"));
    }

    #[test]
    fn the_surface_packs_normal_xy_roughness_and_glow() {
        // A flat normal, roughness 0.25 and glow 0.5: R G 128, B 64, A 128.
        let normal = vec![[0.0, 0.0, 1.0, 0.5]; 4];
        let atlas = DecalAtlas { size: 2, layers: vec![vec![[0.5; 4]; 4]], normals: vec![normal], roughness: vec![vec![0.25; 4]], materials: vec![LayerMaterial::default()] };
        let mips = atlas.surface_mips();
        assert_eq!(mips[0][0], vec![128, 128, 64, 128, 128, 128, 64, 128, 128, 128, 64, 128, 128, 128, 64, 128]);
        // The mip chain reaches one texel (4 bytes).
        assert_eq!(mips[0].len(), 2);
        assert_eq!(mips[0][1], vec![128, 128, 64, 128]);
        // A layer with no map writes texture roughness 1 (the factor then
        // carries it).
        let flat = DecalAtlas { size: 2, layers: vec![vec![[0.5; 4]; 4]], normals: Vec::new(), roughness: Vec::new(), materials: Vec::new() };
        let flat = flat.surface_mips();
        assert_eq!(&flat[0][0][..4], &[128, 128, 255, 0]);
        assert_eq!(flat[0][0].len() / 4, 4);
        let mut bad = atlas;
        bad.roughness[0][0] = 1.5;
        assert!(bad.validate().unwrap_err().contains("roughness"));
    }

    /// A 32-bit TGA of `texels` (rows top first), run-length encoded or
    /// not, stored top or bottom row first.
    fn tga(width: u16, texels: &[[u8; 4]], rle: bool, top_first: bool) -> Vec<u8> {
        let height = (texels.len() / width as usize) as u16;
        let mut out = vec![0, 0, if rle { 10 } else { 2 }, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        out.extend(width.to_le_bytes());
        out.extend(height.to_le_bytes());
        out.extend([32, if top_first { 8 | 32 } else { 8 }]);
        let rows: Vec<&[[u8; 4]]> = if top_first { texels.chunks(width as usize).collect() } else { texels.chunks(width as usize).rev().collect() };
        let bgra = |t: &[u8; 4]| [t[2], t[1], t[0], t[3]];
        for row in rows {
            if rle {
                // A run of the first texel, then the rest raw.
                out.push(128);
                out.extend(bgra(&row[0]));
                out.push(row.len() as u8 - 2);
                row[1..].iter().for_each(|t| out.extend(bgra(t)));
            } else {
                row.iter().for_each(|t| out.extend(bgra(t)));
            }
        }
        out
    }

    #[test]
    fn tga_reads_raw_and_run_length_either_row_order() {
        let texels: Vec<[u8; 4]> = (0..16u8).map(|i| [i, 2 * i, 3 * i, 255 - i]).collect();
        for rle in [false, true] {
            for top_first in [false, true] {
                let img = read_tga(&tga(4, &texels, rle, top_first)).unwrap();
                assert_eq!((img.width, img.height), (4, 4));
                assert_eq!(img.texels, texels, "rle {rle}, top first {top_first}");
            }
        }
        assert!(read_tga(&tga(4, &texels, false, true)[..40]).is_err());
    }

    #[test]
    fn layers_are_resampled_and_normals_take_the_atlas_convention() {
        let mut atlas = DecalAtlas { size: 2, layers: vec![vec![[0.0; 4]; 4]], normals: Vec::new(), roughness: Vec::new(), materials: Vec::new() };
        // 4x4 colour, mid grey: box-filtered to 2x2, linear.
        let grey = Image { width: 4, height: 4, texels: vec![[188, 188, 188, 255]; 16] };
        // A normal map leaning up the image (green 1, OpenGL style), glow 0.5.
        let up = Image { width: 4, height: 4, texels: vec![[128, 255, 128, 128]; 16] };
        let layer = atlas.add_layer(&grey, Some((&up, true))).unwrap();
        assert_eq!(layer, 1);
        assert!((atlas.layers[1][0][0] - 0.5).abs() < 0.01 && atlas.layers[1][3][3] == 1.0);
        // Earlier layers get flat normals; the new one leans down the rows
        // (the atlas's +v) once flipped, renormalized.
        assert_eq!(atlas.normals[0], vec![[0.0, 0.0, 1.0, 0.0]; 4]);
        let n = atlas.normals[1][0];
        assert!(n[1] < -0.7 && (n[0] * n[0] + n[1] * n[1] + n[2] * n[2] - 1.0).abs() < 1e-4 && (n[3] - 0.5).abs() < 0.01, "{n:?}");
        let down = atlas.add_layer(&grey, Some((&up, false))).unwrap();
        assert!(atlas.normals[down as usize][0][1] > 0.7);
        assert!(atlas.validate().is_ok());
        // Bilinear when growing: a ramp stays monotonic.
        let ramp: Vec<[f32; 4]> = (0..4).map(|i| [i as f32, 0.0, 0.0, 1.0]).collect();
        let big = resample(&ramp, 2, 4, false);
        assert!(big[0][0] <= big[1][0] && big[1][0] <= big[2][0] && big[2][0] <= big[3][0] && big[3][0] > big[0][0]);
        assert!(DecalAtlas { size: 2, layers: vec![vec![[0.0; 4]; 4]], normals: vec![], roughness: Vec::new(), materials: Vec::new() }.add_layer(&Image { width: 4, height: 2, texels: vec![[0; 4]; 8] }, None).is_err());
    }

    /// A normal map's handedness from its own curl: with y down the rows,
    /// d(nx)/dy and d(ny)/dx agree when green points down the rows.
    fn curl_sign(n: &[[f32; 4]], size: usize, alpha: &[[f32; 4]]) -> f32 {
        let at = |x: usize, y: usize| n[y * size + x];
        let (mut sum, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for y in 1..size - 1 {
            for x in 1..size - 1 {
                if alpha[y * size + x][3] < 0.3 {
                    continue;
                }
                let a = (at(x, y + 1)[0] - at(x, y - 1)[0]) as f64;
                let b = (at(x + 1, y)[1] - at(x - 1, y)[1]) as f64;
                sum += a * b;
                na += a * a;
                nb += b * b;
            }
        }
        (sum / (na * nb).sqrt().max(1e-12)) as f32
    }

    #[test]
    fn default_atlas_normals_follow_the_atlas_convention() {
        let atlas = default_atlas();
        let opaque = vec![[0.0, 0.0, 0.0, 1.0]; (atlas.size * atlas.size) as usize];
        for k in 0..2 {
            let c = curl_sign(&atlas.normals[k], atlas.size as usize, &opaque);
            assert!(c > 0.5, "layer {k}: {c}");
        }
    }

    /// The handedness detector: a DirectX-style (green down) normal map and
    /// an OpenGL-style (green up) one are told apart; a flat map defaults
    /// to green up.
    #[test]
    fn normal_maps_report_their_green_handedness() {
        let size = 8u32;
        let colour = Image { width: size, height: size, texels: vec![[0, 0, 0, 255]; (size * size) as usize] };
        let map = |up: bool| -> Image {
            let texels = (0..size * size)
                .map(|i| {
                    let (x, y) = ((i % size) as f32 - 3.5, (i / size) as f32 - 3.5);
                    let nx = y * 0.05;
                    let ny = x * 0.05 * if up { -1.0 } else { 1.0 };
                    let enc = |v: f32| ((v.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8;
                    [enc(nx), enc(ny), 128, 255]
                })
                .collect();
            Image { width: size, height: size, texels }
        };
        assert!(!normal_green_up(&colour, &map(false)), "green down is DirectX style");
        assert!(normal_green_up(&colour, &map(true)), "green up is OpenGL style");
        let flat = Image { width: size, height: size, texels: vec![[128, 128, 255, 255]; (size * size) as usize] };
        assert!(normal_green_up(&colour, &flat), "a flat map defaults to green up");
    }

    #[test]
    fn recoil_scars_load_into_one_convention() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/decals/scars");
        let read = |name: &str| read_tga(&std::fs::read(dir.join(name)).unwrap()).unwrap();
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tga") && !n.ends_with("_normal.tga"))
            .collect();
        names.sort();
        assert!(!names.is_empty(), "no bundled scars");
        let mut atlas = default_atlas().resized(256);
        for name in names {
            let stem = name.strip_suffix(".tga").unwrap();
            let (colour, normal) = (read(&name), read(&format!("{stem}_normal.tga")));
            let up = normal_green_up(&colour, &normal);
            let layer = atlas.add_layer(&colour, Some((&normal, up))).unwrap() as usize;
            let size = atlas.size as usize;
            let sign = curl_sign(&atlas.normals[layer], size, &atlas.layers[layer]);
            println!("{name}: green up {up}, curl {sign:.2}");
            assert!(sign > 0.3, "{name}: {sign}");
        }
        atlas.validate().unwrap();
    }

    /// `cargo test --release -p ae_render --lib decal_encode_time -- --ignored --nocapture`
    #[test]
    #[ignore = "timing"]
    fn decal_encode_time() {
        let mut set = DecalSet::new(1024);
        for i in 0..1024 {
            set.insert(Decal::scorch([5000.0 + (i % 37) as f32 * 16.0, 5000.0 + (i / 37) as f32 * 16.0], 5.0, i), 0.0);
        }
        let t = std::time::Instant::now();
        for _ in 0..100 {
            std::hint::black_box(set.encode());
        }
        println!("{:.1} µs per encode of 1024 decals", t.elapsed().as_secs_f64() * 1e4);
    }
}
