//! Decals: marks painted on the terrain (scorch marks, craters), applied by
//! the terrain's own fragment shader (`terrain_decals.glsl`), so nothing
//! else (units, water) is ever painted and there is no projection volume to
//! size.
//!
//! * A decal is a footprint on the heightfield: a centre (x, z; its height
//!   is the ground's there, read on the GPU, so it follows terrain edits),
//!   half-sizes along its own axes, a rotation, a layer of the
//!   [`DecalAtlas`], an opacity and a fade-in.
//! * Its layer also carries a material ([`LayerMaterial`]: a metallic
//!   factor), blended into the ground's by the decal's coverage; its
//!   roughness comes from the layer's surface texture.
//! * Channels: the albedo texture is RGB colour with coverage in alpha; the
//!   material texture is linear height, metalness, roughness and glow.
//!   Signed UV slopes are precomputed and mipmapped independently. Legacy
//!   normal layers are converted to equivalent slopes at upload.
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
/// Seconds a scorch mark fades in over: about the explosion's fire's
/// longest life (0.9 s in `assets/effects.json`), so the mark forms as the
/// fire burns out instead of lying whole under it.
pub const SCORCH_FADE_IN: f32 = 1.0;
/// Seconds a scar glows (Recoil's default glow TTL, 60 frames).
pub const SCAR_GLOW_LIFE: f32 = 2.0;
/// Most decals one binning cell lists (the newest are kept).
pub const MAX_PER_CELL: usize = 255;
/// Default full height range as a fraction of the decal's width.
pub const DEFAULT_HEIGHT_RATIO: f32 = 24.0 / 512.0;
/// Additional height multiplier for RCSM displacement and normal strength.
pub const RCSM_HEIGHT_SCALE: f32 = 2.5;
/// Full height range for explosion scars, as a fraction of their width.
pub const SCAR_HEIGHT_RATIO: f32 = 4.0 * DEFAULT_HEIGHT_RATIO;
/// Header words before the decals.
const HEADER_WORDS: usize = 4;
/// Words per decal.
const DECAL_WORDS: usize = 17;
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
    /// Full world-space height range for this instance's relative relief.
    pub relief: Relief,
}

/// How a decal's authored micro-height scales when its footprint changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Relief {
    /// Height range as a fraction of the decal's final full width.
    Proportional(f32),
    /// Height range in world metres.
    World(f32),
}

impl Default for Relief {
    fn default() -> Self {
        Self::Proportional(DEFAULT_HEIGHT_RATIO)
    }
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
    /// fades in over [`SCORCH_FADE_IN`] and lives [`SCORCH_LIFE`] seconds,
    /// fading out over the last third.
    pub fn scorch(pos: [f32; 2], radius: f32, seed: u32) -> Self {
        let angle = (seed.wrapping_mul(0x9e37_79b9) >> 8) as f32 / (1u32 << 24) as f32 * std::f32::consts::TAU;
        Self {
            pos,
            half: [radius, radius],
            angle,
            layer: layer::SCORCH,
            opacity: 0.9,
            fade_in: SCORCH_FADE_IN,
            life: SCORCH_LIFE,
            fade_out: SCORCH_LIFE / 3.0,
            glow: 0.0,
            glow_life: 0.0,
            blend: Blend::Replace,
            relief: Relief::default(),
        }
    }

    /// An explosion's scar on atlas layer `layer`, as Recoil sets one from
    /// the explosion's `damage` (`GroundDecalHandler`): opacity
    /// `clamp(2 damage / 255, 0.8, 1)`, glow `clamp(2 damage / 255, 0, 1)`
    /// for [`SCAR_GLOW_LIFE`] seconds. It lives [`SCORCH_LIFE`] (Recoil:
    /// `clamp(3 damage, 15, 1800)` frames times the decal level setting).
    pub fn scar(pos: [f32; 2], radius: f32, seed: u32, damage: f32, layer: u32, blend: Blend) -> Self {
        let k = 2.0 * damage / 255.0;
        Self {
            layer,
            blend,
            opacity: k.clamp(0.8, 1.0),
            glow: k.clamp(0.0, 1.0),
            glow_life: SCAR_GLOW_LIFE,
            relief: Relief::Proportional(SCAR_HEIGHT_RATIO),
            ..Self::scorch(pos, radius, seed)
        }
    }

    /// A decal that stays until evicted, at full opacity at once.
    pub fn permanent(pos: [f32; 2], half: [f32; 2], angle: f32, layer: u32) -> Self {
        Self {
            pos,
            half,
            angle,
            layer,
            opacity: 1.0,
            fade_in: 0.0,
            life: 0.0,
            fade_out: 0.0,
            glow: 0.0,
            glow_life: 0.0,
            blend: Blend::Replace,
            relief: Relief::default(),
        }
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
        self.queued
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .extend_from_slice(decals);
    }

    /// Every queued decal, oldest first (the render thread takes them each
    /// frame).
    pub fn take(&self) -> Vec<Decal> {
        std::mem::take(&mut *self.queued.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

/// How a decal layer's surface responds to light: its metallic factor
/// (roughness is the layer's own texture, per texel). A layer blends its
/// material into the ground's by its coverage, so soot (a rough dielectric)
/// takes the gloss off whatever it lands on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerMaterial {
    /// 0 dielectric, 1 metal (the layer's colour is then its F0).
    pub metallic: f32,
}

impl Default for LayerMaterial {
    /// A dielectric: metallic stays 0, and the roughness comes wholly from
    /// the layer's texture (fully rough where it has none).
    fn default() -> Self {
        Self { metallic: 0.0 }
    }
}

/// The decals' textures, mip 0 of each layer as the GPU takes it: square
/// layers of `size`² texels, row-major, the first row at the decal's -z
/// edge. Their mips are made on the GPU ([`crate::upload::image_mipped`]).
#[derive(Clone, Debug)]
pub struct DecalAtlas {
    pub size: u32,
    /// Layer for layer, the colour (`R8G8B8A8_SRGB`): sRGB RGB, straight
    /// (not premultiplied) alpha, the coverage, linear.
    pub colour: Vec<Vec<[u8; 4]>>,
    /// Layer for layer, linear surface texels. Height layers store height
    /// (128 is neutral), metalness, roughness and glow. Legacy layers
    /// store tangent normal XY, roughness and glow. Empty uses flat legacy
    /// surfaces ([`FLAT_SURFACE`]).
    pub surface: Vec<Vec<[u8; 4]>>,
    /// Indices of surfaces with height and metalness in RG.
    pub height_layers: Vec<u32>,
    /// Layer for layer, its surface material. Empty: every layer is
    /// [`LayerMaterial::default`].
    pub materials: Vec<LayerMaterial>,
}

/// A surface texel with a flat normal, roughness 1 and no glow.
pub const FLAT_SURFACE: [u8; 4] = [128, 128, 255, 0];

fn unorm8(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn srgb_encode(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    unorm8(if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    })
}

fn srgb_decode(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// A colour texel from linear RGB and alpha.
pub fn colour_texel(linear: [f32; 4]) -> [u8; 4] {
    [
        srgb_encode(linear[0]),
        srgb_encode(linear[1]),
        srgb_encode(linear[2]),
        unorm8(linear[3]),
    ]
}

/// A surface texel from a tangent normal (any length; z out of the
/// ground), the roughness and the glow mask.
pub fn surface_texel(normal: [f32; 3], roughness: f32, glow: f32) -> [u8; 4] {
    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    let (x, y) = if len > 1e-6 {
        (normal[0] / len, normal[1] / len)
    } else {
        (0.0, 0.0)
    };
    [
        unorm8(x * 0.5 + 0.5),
        unorm8(y * 0.5 + 0.5),
        unorm8(roughness),
        unorm8(glow),
    ]
}

impl DecalAtlas {
    /// Each layer's surface, [`FLAT_SURFACE`] throughout when the atlas
    /// has none.
    pub(crate) fn surface_layers(&self) -> Vec<Vec<[u8; 4]>> {
        if self.surface.is_empty() {
            vec![vec![FLAT_SURFACE; (self.size * self.size) as usize]; self.colour.len()]
        } else {
            self.surface.clone()
        }
    }

    /// Set a legacy layer's uniform metallic factor.
    pub fn set_layer_material(&mut self, layer: u32, material: LayerMaterial) -> Result<(), String> {
        let n = self.colour.len();
        if layer as usize >= n {
            return Err(format!("material for decal layer {layer}, but the atlas has {n}"));
        }
        if self.height_layers.contains(&layer) {
            return Err(format!(
                "decal layer {layer} takes its metalness from the material texture"
            ));
        }
        if self.materials.is_empty() {
            self.materials = vec![LayerMaterial::default(); n];
        }
        self.materials.resize(n, LayerMaterial::default());
        self.materials[layer as usize] = material;
        Ok(())
    }

    /// The per-layer materials, packed for the shader: x legacy metallic,
    /// y the weight of the metalness texture. Missing legacy entries are dielectric.
    pub(crate) fn gpu_materials(&self) -> Vec<[f32; 4]> {
        (0..self.colour.len())
            .map(|i| {
                let m = self.materials.get(i).copied().unwrap_or_default();
                if self.height_layers.contains(&(i as u32)) {
                    [0.0, 1.0, 0.0, 0.0]
                } else {
                    [m.metallic.clamp(0.0, 1.0), 0.0, 0.0, 0.0]
                }
            })
            .collect()
    }

    /// Add a layer from 8-bit images, rows top first (as [`read_tga`]
    /// returns them; the top row is the decal's -z edge): `colour` sRGB
    /// with straight alpha; `normals` the surface as the atlas stores it
    /// (tangent normal x/y in red/green, roughness in blue, the glow mask
    /// in alpha), but with green up the image when `up` (OpenGL style;
    /// flipped to the atlas's green-down convention, `255 - g`). A source
    /// the atlas's size is stored as it is; any other square is resampled
    /// ([`resample`]: colour in linear light, the surface's channels as
    /// they are). Without `normals` the layer is [`FLAT_SURFACE`]; its
    /// material starts at metallic 0. Returns the layer.
    pub fn add_layer(&mut self, colour: &Image, normals: Option<(&Image, bool)>) -> Result<u32, String> {
        self.add_surface_layer(colour, normals)
    }

    /// Add a material layer with linear height, metalness, roughness and glow.
    /// Increasing V follows the top-first image rows. Coverage comes from colour.
    pub fn add_material_layer(&mut self, colour: &Image, material: &Image) -> Result<u32, String> {
        let layer = self.add_surface_layer(colour, Some((material, false)))?;
        self.height_layers.push(layer);
        Ok(layer)
    }

    fn add_surface_layer(&mut self, colour: &Image, surface: Option<(&Image, bool)>) -> Result<u32, String> {
        let square = |img: &Image| -> Result<(), String> {
            if img.width == 0 || img.width != img.height || img.texels.len() != img.width as usize * img.height as usize
            {
                return Err(format!(
                    "invalid square decal image {}x{} with {} texels",
                    img.width,
                    img.height,
                    img.texels.len()
                ));
            }
            Ok(())
        };
        square(colour)?;
        let size = self.size;
        let colour_layer = if colour.width == size {
            colour.texels.clone()
        } else {
            let linear: Vec<[f32; 4]> = colour
                .texels
                .iter()
                .map(|t| {
                    [
                        srgb_decode(t[0]),
                        srgb_decode(t[1]),
                        srgb_decode(t[2]),
                        t[3] as f32 / 255.0,
                    ]
                })
                .collect();
            resample(&linear, colour.width, size)
                .into_iter()
                .map(colour_texel)
                .collect()
        };
        let surface_layer = match surface {
            Some((img, up)) => {
                square(img)?;
                let flipped: Vec<[u8; 4]> = img
                    .texels
                    .iter()
                    .map(|t| [t[0], if up { 255 - t[1] } else { t[1] }, t[2], t[3]])
                    .collect();
                Some(if img.width == size {
                    flipped
                } else {
                    let unorm: Vec<[f32; 4]> = flipped.iter().map(|t| t.map(|c| c as f32 / 255.0)).collect();
                    resample(&unorm, img.width, size)
                        .into_iter()
                        .map(|t| t.map(unorm8))
                        .collect()
                })
            }
            None => None,
        };
        let layer = self.colour.len() as u32;
        let texels = (size * size) as usize;
        // A surface for every layer once any layer has one.
        if surface_layer.is_some() || !self.surface.is_empty() {
            self.surface.resize(self.colour.len(), vec![FLAT_SURFACE; texels]);
            self.surface
                .push(surface_layer.unwrap_or_else(|| vec![FLAT_SURFACE; texels]));
        }
        if !self.materials.is_empty() {
            self.materials.push(LayerMaterial::default());
        }
        self.colour.push(colour_layer);
        Ok(layer)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.size.is_power_of_two() {
            return Err(format!("decal atlas size {} is not a power of two", self.size));
        }
        if self.colour.is_empty() {
            return Err("decal atlas has no layers".into());
        }
        if !self.surface.is_empty() && self.surface.len() != self.colour.len() {
            return Err(format!(
                "decal atlas has {} surface layers for {} colour layers",
                self.surface.len(),
                self.colour.len()
            ));
        }
        if let Some(i) = self.height_layers.iter().find(|&&i| i as usize >= self.surface.len()) {
            return Err(format!("decal material layer {i} has no surface"));
        }
        if !self.materials.is_empty() && self.materials.len() != self.colour.len() {
            return Err(format!(
                "decal atlas has {} material layers for {} colour layers",
                self.materials.len(),
                self.colour.len()
            ));
        }
        if let Some((i, m)) = self
            .materials
            .iter()
            .enumerate()
            .find(|(_, m)| !(0.0..=1.0).contains(&m.metallic))
        {
            return Err(format!(
                "decal atlas layer {i} material metallic {} out of range",
                m.metallic
            ));
        }
        let texels = (self.size * self.size) as usize;
        if let Some(i) = self.colour.iter().position(|l| l.len() != texels) {
            return Err(format!(
                "decal atlas layer {i} has {} texels, not {texels}",
                self.colour[i].len()
            ));
        }
        if let Some(i) = self.surface.iter().position(|l| l.len() != texels) {
            return Err(format!(
                "decal atlas surface layer {i} has {} texels, not {texels}",
                self.surface[i].len()
            ));
        }
        Ok(())
    }
}

/// Signed height slopes in normalized UV units. Top-first rows increase V.
/// The neutral-height bias cancels when subtracting samples.
fn height_derivatives(surface: &[[u8; 4]], size: usize) -> impl Iterator<Item = [f32; 2]> + '_ {
    let height = move |x: usize, y: usize| f32::from(surface[y * size + x][0]);
    let difference = move |lo, hi, span: usize| (hi - lo) * (size as f32 / 255.0) / span.max(1) as f32;
    (0..surface.len()).map(move |i| {
        let (x, y) = (i % size, i / size);
        let (l, r) = (x.saturating_sub(1), (x + 1).min(size - 1));
        let (u, d) = (y.saturating_sub(1), (y + 1).min(size - 1));
        [
            difference(height(l, y), height(r, y), r - l),
            difference(height(x, u), height(x, d), d - u),
        ]
    })
}

pub(crate) struct GpuSurfaceLayer {
    pub material: Vec<[u8; 4]>,
    pub normals: Vec<[u8; 4]>,
}

impl DecalAtlas {
    /// Convert legacy normals and heightfield slopes to signed tangent normals.
    pub(crate) fn gpu_surfaces(&self) -> impl Iterator<Item = GpuSurfaceLayer> + '_ {
        self.surface_layers().into_iter().enumerate().map(|(i, mut layer)| {
            let pack = |slope: [f32; 2]| {
                let n = glam::Vec3::new(-slope[0] * DEFAULT_HEIGHT_RATIO, -slope[1] * DEFAULT_HEIGHT_RATIO, 1.0)
                    .normalize();
                let [x, y] = [n.x, n.y].map(|c| (c * 32767.0).round() as i16);
                let [a, b] = x.to_le_bytes();
                let [c, d] = y.to_le_bytes();
                [a, b, c, d]
            };
            let normals = if self.height_layers.contains(&(i as u32)) {
                height_derivatives(&layer, self.size as usize).map(pack).collect()
            } else {
                layer
                    .iter_mut()
                    .map(|t| {
                        let xy = [t[0], t[1]].map(|c| match c {
                            127 | 128 => 0.0,
                            _ => f32::from(c) / 127.5 - 1.0,
                        });
                        let z = (1.0 - xy[0] * xy[0] - xy[1] * xy[1]).max(0.0025).sqrt();
                        t[0] = 128;
                        t[1] = 0;
                        pack(xy.map(|n| -n / (DEFAULT_HEIGHT_RATIO * z)))
                    })
                    .collect()
            };
            GpuSurfaceLayer {
                material: layer,
                normals,
            }
        })
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
    let (width, height) = (
        u16::from_le_bytes([header[12], header[13]]) as u32,
        u16::from_le_bytes([header[14], header[15]]) as u32,
    );
    if width == 0 || height == 0 {
        return Err("TGA: empty image".into());
    }
    if header[1] != 0 || !(kind == 2 || kind == 10) || bpp != 32 {
        return Err(format!(
            "TGA: only 32-bit true colour is read (colour map {}, type {kind}, {bpp} bits)",
            header[1]
        ));
    }
    let n = (width * height) as usize;
    let mut data = bytes.get(18 + id_len..).ok_or("TGA: short data")?;
    let mut bgra: Vec<[u8; 4]> = Vec::with_capacity(n);
    let texel = |d: &[u8]| -> Result<[u8; 4], String> {
        d.get(..4)
            .map(|t| [t[2], t[1], t[0], t[3]])
            .ok_or_else(|| "TGA: data ends early".to_string())
    };
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
    Ok(Image {
        width,
        height,
        texels: bgra,
    })
}

/// Whether a tangent normal map's green points up the image (OpenGL/glTF
/// style), from the curl of its red and green channels over the texels of
/// `colour` whose alpha is at least 0.3. A positive curl means green down
/// the rows (DirectX style), a negative one green up. A map with too little
/// signal (flat, or two images of different sizes) is taken as green up,
/// glTF's convention. The blue channel is the layer's roughness (see
/// [`DecalAtlas::add_layer`]) and is not read here.
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
/// by a whole factor, bilinear otherwise.
pub fn resample(texels: &[[f32; 4]], from: u32, to: u32) -> Vec<[f32; 4]> {
    let (f, t) = (from as usize, to as usize);
    let at = |x: usize, y: usize| texels[y.min(f - 1) * f + x.min(f - 1)];
    (0..t * t)
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
                let (u, v) = (
                    ((x as f32 + 0.5) * f as f32 / t as f32 - 0.5).max(0.0),
                    ((y as f32 + 0.5) * f as f32 / t as f32 - 0.5).max(0.0),
                );
                let (x0, y0, fx, fy) = (u as usize, v as usize, u.fract(), v.fract());
                let (a, b, c, d) = (at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1));
                std::array::from_fn(|k| {
                    (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy
                })
            }
        })
        .collect()
}

/// The engine's decals ([`layer`]), `size`² each (a power of two; 128 is
/// plenty, the relief keeps its depth at any size): a scorch mark (charred
/// centre, a burnt brown ring, a ragged edge; a shallow rough hollow whose
/// centre can glow) and a crater (a dark pit, a pale rim of thrown-up
/// earth, in relief).
pub fn default_atlas(size: u32) -> DecalAtlas {
    let n = size as usize;
    // Heights are in texels of depth per texel at 128².
    let relief = size as f32 / 128.0;
    fn hash(x: i32, y: i32, octave: i32) -> f32 {
        let mut h = (x as u32).wrapping_mul(0x8da6_b343)
            ^ (y as u32).wrapping_mul(0xd816_3841)
            ^ (octave as u32).wrapping_mul(0xcb1a_b31f);
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
    let mix = |a: [f32; 3], b: [f32; 3], t: f32| {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    };
    let layer = |f: &dyn Fn(f32, f32, f32, f32) -> [f32; 4]| -> Vec<[f32; 4]> {
        (0..n * n)
            .map(|i| {
                let (u, v) = (((i % n) as f32 + 0.5) / n as f32, ((i / n) as f32 + 0.5) / n as f32);
                let (x, y) = (u * 2.0 - 1.0, v * 2.0 - 1.0);
                f(u, v, x.hypot(y), y.atan2(x))
            })
            .collect()
    };
    let scalar = |f: &dyn Fn(f32, f32, f32, f32) -> f32| -> Vec<f32> {
        (0..n * n)
            .map(|i| {
                let (u, v) = (((i % n) as f32 + 0.5) / n as f32, ((i / n) as f32 + 0.5) / n as f32);
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
    let scorch_height =
        |u: f32, v: f32, r: f32| relief * (-2.5 * (1.0 - smooth(0.0, 0.7, r)) + 2.0 * fbm(u * 2.0, v * 2.0, 3));
    let crater_height = |u: f32, v: f32, r: f32| {
        let rr = r + (fbm(u, v, 7) - 0.5) * 0.18;
        relief * (-5.0 * (1.0 - smooth(0.2, 0.55, rr)) + 3.0 * smooth(0.45, 0.6, rr) * (1.0 - smooth(0.6, 0.9, rr)))
    };
    let normals = |height: &dyn Fn(f32, f32, f32) -> f32, glow: &dyn Fn(f32, f32, f32) -> f32| -> Vec<[f32; 4]> {
        let h = |x: f32, y: f32| {
            let (u, v) = (x / n as f32, y / n as f32);
            height(u, v, (u * 2.0 - 1.0).hypot(v * 2.0 - 1.0))
        };
        (0..n * n)
            .map(|i| {
                let (x, y) = ((i % n) as f32 + 0.5, (i / n) as f32 + 0.5);
                let (dx, dy) = (
                    (h(x + 1.0, y) - h(x - 1.0, y)) * 0.5,
                    (h(x, y + 1.0) - h(x, y - 1.0)) * 0.5,
                );
                let len = (dx * dx + dy * dy + 1.0).sqrt();
                let (u, v) = (x / n as f32, y / n as f32);
                [
                    -dx / len,
                    -dy / len,
                    1.0 / len,
                    glow(u, v, (u * 2.0 - 1.0).hypot(v * 2.0 - 1.0)),
                ]
            })
            .collect()
    };
    let scorch_glow = |u: f32, v: f32, r: f32| {
        ((1.0 - smooth(0.05, 0.6, r)) * (0.35 + 0.8 * fbm(u * 1.5, v * 1.5, 11))).clamp(0.0, 0.85)
    };
    let normals = [
        normals(&scorch_height, &scorch_glow),
        normals(&crater_height, &|_, _, _| 0.0),
    ];
    // Roughness textures: the final roughness, since layers have no roughness
    // factor. Char and cinders are rougher than the burnt ring, the crater's
    // pit than its rim. 8 bits is plenty for both.
    let scorch_rough = scalar(&|u, v, r, a| {
        let n = fbm(u, v, 0);
        let edge = 0.78 + 0.03 * (a * 5.0).sin() + (n - 0.5) * 0.35;
        0.95 * (1.0 - 0.1 * smooth(0.15, edge * 0.9, r + (n - 0.5) * 0.3))
    });
    let crater_rough = scalar(&|u, v, r, _| {
        let n = fbm(u, v, 7);
        0.9 * (1.0 - 0.1 * smooth(0.35, 0.6, r + (n - 0.5) * 0.18))
    });
    // Char and ash are matte dielectrics; the crater's loose earth less so.
    let materials = vec![LayerMaterial::default(); 2];
    let surface = |normals: &[[f32; 4]], rough: &[f32]| -> Vec<[u8; 4]> {
        normals
            .iter()
            .zip(rough)
            .map(|(t, &r)| surface_texel([t[0], t[1], t[2]], r, t[3]))
            .collect()
    };
    let surface = vec![surface(&normals[0], &scorch_rough), surface(&normals[1], &crater_rough)];
    let colour = vec![
        scorch.into_iter().map(colour_texel).collect(),
        crater.into_iter().map(colour_texel).collect(),
    ];
    DecalAtlas {
        size,
        colour,
        surface,
        height_layers: Vec::new(),
        materials,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DecalReliefMode {
    Disabled = 0,
    #[default]
    NormalMapping = 1,
    RobustConeStepMapping = 2,
}

impl DecalReliefMode {
    pub(crate) const ALL: [Self; 3] = [Self::Disabled, Self::NormalMapping, Self::RobustConeStepMapping];

    pub fn next(self) -> Self {
        Self::ALL[(self as usize + 1) % Self::ALL.len()]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DecalParallaxOverlapMode {
    #[default]
    NormalMappingFallback = 0,
    DominantDecal = 1,
    IndependentExperimental = 2,
}

#[derive(Clone, Copy, Debug)]
pub struct DecalReliefSettings {
    pub mode: DecalReliefMode,
    pub overlap_mode: DecalParallaxOverlapMode,
    pub strength: f32,
    pub max_cone_steps: u32,
    pub max_refine_steps: u32,
    pub min_parallax_pixels: f32,
    pub full_parallax_pixels: f32,
    pub min_view_normal_dot: f32,
    pub adaptive_quality: bool,
}

impl Default for DecalReliefSettings {
    fn default() -> Self {
        Self {
            mode: DecalReliefMode::default(),
            overlap_mode: DecalParallaxOverlapMode::default(),
            strength: 1.0,
            max_cone_steps: 48,
            max_refine_steps: 6,
            min_parallax_pixels: 0.75,
            full_parallax_pixels: 2.0,
            min_view_normal_dot: 0.18,
            adaptive_quality: true,
        }
    }
}

/// Unlit terrain decal views. Individual views show the newest decal with
/// nonzero coverage at the fragment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DecalDebug {
    #[default]
    Off = 0,
    Height = 1,
    Coverage = 2,
    Normal = 3,
    TangentNormal = 4,
    Cone = 5,
    CellBounds = 6,
    Iterations = 7,
    IntersectionDepth = 8,
    DisplacedUv = 9,
    IntersectionFailure = 10,
    OverlapFallback = 11,
    ReliefLod = 12,
    FrameError = 13,
    UnfoldGrid = 14,
}

impl DecalDebug {
    pub const VIEWS: [(&'static str, Self); 15] = [
        ("off", Self::Off),
        ("height", Self::Height),
        ("coverage", Self::Coverage),
        ("normal", Self::Normal),
        ("tangent-normal", Self::TangentNormal),
        ("cone", Self::Cone),
        ("cell-bounds", Self::CellBounds),
        ("iterations", Self::Iterations),
        ("depth", Self::IntersectionDepth),
        ("uv", Self::DisplacedUv),
        ("failed", Self::IntersectionFailure),
        ("overlap", Self::OverlapFallback),
        ("lod", Self::ReliefLod),
        ("frame", Self::FrameError),
        ("grid", Self::UnfoldGrid),
    ];

    pub fn next(self) -> Self {
        Self::VIEWS[(self as usize + 1) % Self::VIEWS.len()].1
    }
}

pub struct DecalsSetup {
    pub feed: Arc<DecalFeed>,
    /// Most decals at once (the oldest is evicted first).
    pub capacity: u32,
    /// The textures; [`default_atlas`] when `None`.
    pub atlas: Option<Arc<DecalAtlas>>,
    pub debug: DecalDebug,
    pub relief: DecalReliefSettings,
}

impl DecalsSetup {
    pub fn new(feed: Arc<DecalFeed>, capacity: u32) -> Self {
        Self {
            feed,
            capacity,
            atlas: None,
            debug: DecalDebug::Off,
            relief: DecalReliefSettings::default(),
        }
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
    clearance: Vec<f32>,
    clearance_version: Option<u64>,
}

impl DecalSet {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity: capacity as usize,
            live: VecDeque::new(),
            version: 0,
            words: Vec::new(),
            count: Vec::new(),
            cursor: Vec::new(),
            clearance: Vec::new(),
            clearance_version: None,
        }
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
        let glow_inv = if d.glow > 0.0 && d.glow_life > 0.0 {
            1.0 / d.glow_life
        } else {
            0.0
        };
        let blend = match d.blend {
            Blend::Replace => 0.0,
            Blend::Modulate => 1.0,
        };
        let width = 2.0 * d.half[0].max(1e-3);
        let height_range = match d.relief {
            Relief::Proportional(ratio) => width * ratio,
            Relief::World(metres) => metres,
        };
        let height_range = if height_range.is_finite() {
            height_range.max(0.0)
        } else {
            0.0
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
            height_range,
            glow_inv,
            blend,
            0.0, // Filled by encode when the safe overlap policy needs it.
        ]
        .map(f32::to_bits);
        // The buckets of the distinct cells it touches (two of its cells
        // may share a bucket: listed once).
        let mask = buckets(self.capacity as u32) - 1;
        let r = f32::from_bits(words[2]).hypot(f32::from_bits(words[3]));
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
        self.live.push_back(Placed {
            words,
            buckets: list,
            n_buckets: n as u8,
            bounds: [d.pos[0], d.pos[1], r],
            end,
        });
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

    /// A lower bound on distance from each footprint disc to any other.
    /// The shader compares it to the entire ray's planar displacement.
    /// Looking past GRID_CELL can only make this bound less pessimistic.
    fn update_clearance(&mut self) {
        let mut by_x: Vec<_> = self
            .live
            .iter()
            .enumerate()
            // Only decals with a relief height occlude a cone trace; word 13
            // is the height range (word 7 is opacity, which every decal has).
            .filter(|(_, p)| f32::from_bits(p.words[13]) > 0.0)
            .map(|(i, p)| (i, p.bounds))
            .collect();
        by_x.sort_unstable_by(|a, b| a.1[0].total_cmp(&b.1[0]));
        self.clearance.clear();
        self.clearance.resize(self.live.len(), GRID_CELL);
        for (at, &(i, [x, z, r])) in by_x.iter().enumerate() {
            for &(j, [other_x, other_z, other_r]) in &by_x[at + 1..] {
                let dx = other_x - x;
                if dx >= r + MAX_RADIUS + GRID_CELL {
                    break;
                }
                let limit = r + other_r + self.clearance[i].max(self.clearance[j]);
                let dz = other_z - z;
                if dx.abs() >= limit || dz.abs() >= limit {
                    continue;
                }
                let distance = (dx.hypot(dz) - r - other_r - 1e-3).max(0.0);
                self.clearance[i] = self.clearance[i].min(distance);
                self.clearance[j] = self.clearance[j].min(distance);
            }
        }
        self.clearance_version = Some(self.version);
    }

    /// The GPU buffer's words ([`buffer_words`]; the unused tail of the
    /// index list left out). A counting sort into buffers kept between
    /// calls.
    pub fn encode(&mut self, debug: DecalDebug, check_overlap: bool) -> &[u32] {
        if check_overlap && self.clearance_version != Some(self.version) {
            self.update_clearance();
        }
        let cap = self.capacity;
        let mask = buckets(cap as u32) - 1;
        let n_buckets = mask as usize + 1;
        let buckets_at = HEADER_WORDS + cap * DECAL_WORDS;
        let index_at = buckets_at + n_buckets;
        self.words.resize(buffer_words(cap as u32), 0);
        self.count.clear();
        self.count.resize(n_buckets, 0);
        let words = &mut self.words;
        words[..HEADER_WORDS].copy_from_slice(&[mask, debug as u32, GRID_CELL.to_bits(), cap as u32]);
        for (i, p) in self.live.iter().enumerate() {
            let at = HEADER_WORDS + i * DECAL_WORDS;
            words[at..at + DECAL_WORDS].copy_from_slice(&p.words);
            if check_overlap {
                words[at + 16] = self.clearance[i].to_bits();
            }
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
        let words = set.encode(DecalDebug::Off, false).to_vec();
        assert!(words.len() <= buffer_words(3));
        // Distinct buckets for these cells at this table size.
        let cells = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (2, 1),
            (-1, 0),
            (0, -1),
            (3, 0),
            (0, 2),
            (5, 5),
        ];
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
        let words = set.encode(DecalDebug::Off, false).to_vec();
        // The first decal went; the others moved down one.
        assert_eq!(listed(&words, 3, 0, 0), vec![0, 1]);
        assert_eq!(listed(&words, 3, 5, 5), vec![2]);
        assert_eq!(set.version, 4);
    }

    #[test]
    fn decals_expire_at_the_end_of_their_life() {
        let mut set = DecalSet::new(4);
        set.insert(decal(0.0, 0.0, 4.0), 0.0);
        set.insert(
            Decal {
                life: 2.0,
                fade_out: 0.5,
                ..decal(1.0, 0.0, 4.0)
            },
            1.0,
        );
        set.insert(
            Decal {
                life: 1.0,
                ..decal(2.0, 0.0, 4.0)
            },
            1.0,
        );
        let words = set.encode(DecalDebug::Off, false).to_vec();
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
        let words = set.encode(DecalDebug::Off, false).to_vec();
        assert_eq!(listed(&words, n as u32, 0, 0), (5..n as u32).collect::<Vec<_>>());
    }

    #[test]
    fn layers_carry_their_material_and_default_to_matte() {
        let plain = DecalAtlas {
            size: 2,
            colour: vec![vec![[0; 4]; 4]; 2],
            surface: Vec::new(),
            height_layers: Vec::new(),
            materials: Vec::new(),
        };
        assert_eq!(
            plain.gpu_materials(),
            vec![[0.0; 4]; 2],
            "without materials: dielectric, roughness from the texture"
        );
        assert_eq!(
            plain.surface_layers(),
            vec![vec![FLAT_SURFACE; 4]; 2],
            "without a surface: flat, fully rough"
        );
        let mut atlas = default_atlas(32);
        assert!(atlas.validate().is_ok());
        assert_eq!(atlas.materials.len(), atlas.colour.len());
        assert_eq!(
            atlas.surface.len(),
            atlas.colour.len(),
            "the default atlas has a surface"
        );
        assert_eq!(atlas.materials[layer::SCORCH as usize].metallic, 0.0);
        assert_eq!(atlas.materials[layer::CRATER as usize].metallic, 0.0);
        // A new layer without a normal map takes the default material until
        // it is told otherwise, and a flat, fully rough surface.
        let grey = Image {
            width: 4,
            height: 4,
            texels: vec![[128; 4]; 16],
        };
        let added = atlas.add_layer(&grey, None).unwrap();
        assert_eq!(atlas.materials[added as usize], LayerMaterial::default());
        assert_eq!(atlas.surface[added as usize], vec![FLAT_SURFACE; 32 * 32]);
        atlas
            .set_layer_material(added, LayerMaterial { metallic: 1.0 })
            .unwrap();
        assert_eq!(atlas.gpu_materials()[added as usize], [1.0, 0.0, 0.0, 0.0]);
        assert!(atlas.set_layer_material(99, LayerMaterial::default()).is_err());
        atlas.materials[0].metallic = 1.5;
        assert!(atlas.validate().unwrap_err().contains("out of range"));
    }

    /// The default atlas keeps its look at any size: the relief's slopes
    /// (per texel) scale with it, so the normals match at 128 and 512.
    #[test]
    fn the_default_atlas_keeps_its_relief_at_any_size() {
        let (small, big) = (default_atlas(128), default_atlas(512));
        let tilt = |a: &DecalAtlas, l: usize| -> f32 {
            let s = &a.surface[l];
            s.iter()
                .map(|t| ((t[0] as f32 - 127.5).powi(2) + (t[1] as f32 - 127.5).powi(2)).sqrt())
                .sum::<f32>()
                / s.len() as f32
        };
        for l in 0..2 {
            let (t0, t1) = (tilt(&small, l), tilt(&big, l));
            assert!(
                t0 > 5.0 && (t0 - t1).abs() < t0 * 0.15,
                "layer {l}: mean tilt {t0} at 128, {t1} at 512"
            );
        }
    }

    #[test]
    fn texels_pack_colour_and_surface() {
        // A flat normal, roughness 0.25 and glow 0.5: R G 128, B 64, A 128.
        assert_eq!(surface_texel([0.0, 0.0, 1.0], 0.25, 0.5), [128, 128, 64, 128]);
        // Any length: normalised first.
        assert_eq!(surface_texel([3.0, 0.0, 4.0], 1.0, 0.0), [204, 128, 255, 0]);
        assert_eq!(colour_texel([0.2159, 1.0, 0.0, 0.5]), [128, 255, 0, 128]);
    }

    /// A 32-bit TGA of `texels` (rows top first), run-length encoded or
    /// not, stored top or bottom row first.
    fn tga(width: u16, texels: &[[u8; 4]], rle: bool, top_first: bool) -> Vec<u8> {
        let height = (texels.len() / width as usize) as u16;
        let mut out = vec![0, 0, if rle { 10 } else { 2 }, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        out.extend(width.to_le_bytes());
        out.extend(height.to_le_bytes());
        out.extend([32, if top_first { 8 | 32 } else { 8 }]);
        let rows: Vec<&[[u8; 4]]> = if top_first {
            texels.chunks(width as usize).collect()
        } else {
            texels.chunks(width as usize).rev().collect()
        };
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
        let mut atlas = DecalAtlas {
            size: 2,
            colour: vec![vec![[0; 4]; 4]],
            surface: Vec::new(),
            height_layers: Vec::new(),
            materials: Vec::new(),
        };
        // 4x4 colour, mid grey: box-filtered to 2x2 in linear light.
        let grey = Image {
            width: 4,
            height: 4,
            texels: vec![[188, 188, 188, 255]; 16],
        };
        // A normal map leaning up the image (green 1, OpenGL style),
        // roughness 0.5, glow 0.5.
        let up = Image {
            width: 4,
            height: 4,
            texels: vec![[128, 255, 128, 128]; 16],
        };
        let layer = atlas.add_layer(&grey, Some((&up, true))).unwrap();
        assert_eq!(layer, 1);
        assert_eq!(atlas.colour[1][0], [188, 188, 188, 255]);
        // Earlier layers get a flat surface; the new one leans down the rows
        // (the atlas's +v) once green is flipped; blue (roughness) and alpha
        // (glow) pass through.
        assert_eq!(atlas.surface[0], vec![FLAT_SURFACE; 4]);
        assert_eq!(atlas.surface[1][0], [128, 0, 128, 128]);
        assert_eq!(
            atlas.gpu_materials()[1],
            [0.0; 4],
            "roughness is the texture; the material stays a dielectric"
        );
        let down = atlas.add_layer(&grey, Some((&up, false))).unwrap();
        assert_eq!(atlas.surface[down as usize][0], [128, 255, 128, 128]);
        assert!(atlas.validate().is_ok());
        // A source the atlas's size is stored byte for byte.
        let mut same = DecalAtlas {
            size: 4,
            colour: Vec::new(),
            surface: Vec::new(),
            height_layers: Vec::new(),
            materials: Vec::new(),
        };
        let texels: Vec<[u8; 4]> = (0..16u8).map(|i| [i * 9, 255 - i, i, 200]).collect();
        let img = Image {
            width: 4,
            height: 4,
            texels: texels.clone(),
        };
        same.add_layer(&img, Some((&img, false))).unwrap();
        assert_eq!((&same.colour[0], &same.surface[0]), (&texels, &texels));
        // Bilinear when growing: a ramp stays monotonic.
        let ramp: Vec<[f32; 4]> = (0..4).map(|i| [i as f32, 0.0, 0.0, 1.0]).collect();
        let big = resample(&ramp, 2, 4);
        assert!(big[0][0] <= big[1][0] && big[1][0] <= big[2][0] && big[2][0] <= big[3][0] && big[3][0] > big[0][0]);
        assert!(
            DecalAtlas {
                size: 2,
                colour: vec![vec![[0; 4]; 4]],
                surface: Vec::new(),
                height_layers: Vec::new(),
                materials: Vec::new()
            }
            .add_layer(
                &Image {
                    width: 4,
                    height: 2,
                    texels: vec![[0; 4]; 8]
                },
                None
            )
            .is_err()
        );
        // Any square resamples; only the atlas itself needs power-of-two
        // mips. (A 6x6 source shrinks into a 4x4 atlas here.)
        let mut odd = DecalAtlas {
            size: 4,
            colour: vec![vec![[0; 4]; 16]],
            surface: Vec::new(),
            height_layers: Vec::new(),
            materials: Vec::new(),
        };
        assert!(
            odd.add_layer(
                &Image {
                    width: 6,
                    height: 6,
                    texels: vec![[128, 128, 128, 255]; 36]
                },
                None
            )
            .is_ok()
        );
        assert!(odd.validate().is_ok());
        let mut bad = odd;
        bad.surface = vec![vec![FLAT_SURFACE; 3]];
        assert!(bad.validate().is_err());
    }

    /// A normal map's handedness from its own curl: with y down the rows,
    /// d(nx)/dy and d(ny)/dx agree when green points down the rows.
    fn curl_sign(n: &[[u8; 4]], size: usize, alpha: &[[u8; 4]]) -> f32 {
        let at = |x: usize, y: usize| n[y * size + x].map(|c| c as f32 / 127.5 - 1.0);
        let (mut sum, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for y in 1..size - 1 {
            for x in 1..size - 1 {
                if alpha[y * size + x][3] < 76 {
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
        let atlas = default_atlas(128);
        let opaque = vec![[0, 0, 0, 255]; (atlas.size * atlas.size) as usize];
        for k in 0..2 {
            let c = curl_sign(&atlas.surface[k], atlas.size as usize, &opaque);
            assert!(c > 0.5, "layer {k}: {c}");
        }
    }

    /// The handedness detector: a DirectX-style (green down) normal map and
    /// an OpenGL-style (green up) one are told apart; a flat map defaults
    /// to green up.
    #[test]
    fn normal_maps_report_their_green_handedness() {
        let size = 8u32;
        let colour = Image {
            width: size,
            height: size,
            texels: vec![[0, 0, 0, 255]; (size * size) as usize],
        };
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
            Image {
                width: size,
                height: size,
                texels,
            }
        };
        assert!(!normal_green_up(&colour, &map(false)), "green down is DirectX style");
        assert!(normal_green_up(&colour, &map(true)), "green up is OpenGL style");
        let flat = Image {
            width: size,
            height: size,
            texels: vec![[128, 128, 255, 255]; (size * size) as usize],
        };
        assert!(normal_green_up(&colour, &flat), "a flat map defaults to green up");
    }

    #[test]
    fn bundled_scar_materials_preserve_the_source_channels() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/decals/scars");
        let read = |name: &str| read_tga(&std::fs::read(dir.join(name)).unwrap()).unwrap();
        let mut atlas = default_atlas(512);
        for i in 1..=10 {
            let colour = read(&format!("scar{i}.tga"));
            let material = read(&format!("scar{i}_mat.tga"));
            assert_eq!((colour.width, colour.height), (512, 512));
            assert_eq!((material.width, material.height), (512, 512));
            let layer = atlas.add_material_layer(&colour, &material).unwrap() as usize;
            assert_eq!(atlas.surface[layer], material.texels);
            assert_eq!(atlas.gpu_materials()[layer], [0.0, 1.0, 0.0, 0.0]);
        }
        atlas.validate().unwrap();
    }

    #[test]
    fn height_derivatives_use_normalized_uv_and_one_sided_borders() {
        assert_eq!(
            height_derivatives(&[[128, 0, 0, 0]; 16], 4).collect::<Vec<_>>(),
            [[0.0; 2]; 16]
        );
        assert_eq!(height_derivatives(&[[42; 4]], 1).next(), Some([0.0; 2]));
        for size in [4, 8, 16] {
            let pixels: Vec<[u8; 4]> = (0..size * size)
                .map(|i| {
                    [
                        128 + (64 / size * (i % size)) as u8 - (32 / size * (i / size)) as u8,
                        42,
                        70,
                        0,
                    ]
                })
                .collect();
            // Decode both TGA row orders before differentiating.
            for top_first in [false, true] {
                let img = read_tga(&tga(size as u16, &pixels, true, top_first)).unwrap();
                for [u, v] in height_derivatives(&img.texels, size) {
                    assert!((u - 64.0 / 255.0).abs() < 1e-6, "U: {u}");
                    assert!((v + 32.0 / 255.0).abs() < 1e-6, "V: {v}");
                }
            }
        }
    }

    #[test]
    fn relief_range_is_encoded_after_the_footprint_is_clamped() {
        let mut set = DecalSet::new(4);
        let decal = Decal::permanent([0.0, 0.0], [64.0, 48.0], 0.0, 0);
        set.insert(decal, 0.0);
        set.insert(
            Decal {
                relief: Relief::World(0.75),
                ..decal
            },
            0.0,
        );
        set.insert(
            Decal {
                relief: Relief::World(0.0),
                ..decal
            },
            0.0,
        );
        let words = set.encode(DecalDebug::Normal, false);
        assert_eq!(words[1], DecalDebug::Normal as u32);
        let range = |i| f32::from_bits(words[HEADER_WORDS + DECAL_WORDS * i + 13]);
        assert!((range(0) - DEFAULT_HEIGHT_RATIO * (128.0 * MAX_RADIUS / 80.0)).abs() < 1e-6);
        assert_eq!(range(1), 0.75);
        assert_eq!(range(2), 0.0);
    }

    /// `cargo test --release -p ae_render --lib decal_encode_time -- --ignored --nocapture`
    #[test]
    #[ignore = "timing"]
    fn decal_encode_time() {
        let mut set = DecalSet::new(1024);
        for i in 0..1024 {
            set.insert(
                Decal::scorch(
                    [5000.0 + (i % 37) as f32 * 16.0, 5000.0 + (i / 37) as f32 * 16.0],
                    5.0,
                    i,
                ),
                0.0,
            );
        }
        let t = std::time::Instant::now();
        for _ in 0..100 {
            std::hint::black_box(set.encode(DecalDebug::Off, false));
        }
        println!("{:.1} µs per encode of 1024 decals", t.elapsed().as_secs_f64() * 1e4);
    }
}
