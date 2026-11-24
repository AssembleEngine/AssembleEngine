//! GPU-driven terrain (lifted from `~/map-rendering/src/renderer.rs`): the
//! CBT/LEB subdivision is updated by compute shaders and drawn with one
//! indexed-indirect instanced draw (one instance = one CBT leaf = a
//! 16-triangle sub-mesh).
//!
//! In the frame graph it is two passes:
//! * `record_lod` (a compute pass) splits and merges leaves against the
//!   camera, sum-reduces the tree and writes the culled draw list
//!   ([`TerrainRenderer::leaves`]) and the indirect arguments
//!   ([`TerrainRenderer::indirect`]); the pass declares both as storage
//!   writes. Its internal steps (snapshot copy, dispatches) are separated by
//!   full barriers it records itself, as in the prototype.
//! * `record_draw` (inside a graphics pass with a colour and a reverse-Z
//!   depth attachment) reads the draw list as a vertex-stage storage buffer
//!   and the arguments as an indirect buffer.
//!
//! * `record_water` (inside a graphics pass that loads the colour target
//!   and reads the depth buffer read-only, after everything opaque) draws
//!   the water surface over the map, when the map has water.
//!
//! The tree state lives in buffers the renderer owns, shared by every frame
//! slot (the subdivision evolves frame to frame); the per-frame uniforms are
//! one buffer per slot. Static data (sub-mesh, height texture, error
//! pyramid) is uploaded once.

use super::heightmap::{Heightmap, PYRAMID_BASE_LEVEL, Pyramid};
use super::leb::{self, Cbt};
use super::tileset::{TILESET_UBO_BYTES, Tileset, TilesetUbo};
use crate::decals::{DecalAtlas, DecalReliefMode, RCSM_HEIGHT_SCALE};
use crate::device::GpuDevice;
use crate::executor::{BufferBinding, PassContext};
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::pipeline::{self, spirv};
use crate::shadow::{Cascades, ShadowMap};
use crate::upload::{self, MipFilter};
use crate::water::{
    WaterParams,
    reflect::{self, MirrorFrame},
    underwater, volume,
};
use ae_framegraph::BufferDesc;
use anyhow::{Context as _, Result};
use ash::vk;
use glam::{Mat4, Vec3, Vec4};
use gpu_allocator::MemoryLocation;

const MAX_DRAW_LEAVES: u32 = 1 << 21;
const LEAF_BYTES: u64 = 32;
const IND_WORDS: u64 = 16;

const MODE_SPLIT: u32 = 0;
const MODE_MERGE: u32 = 1;
const MODE_NONE: u32 = 2;

/// Binding identities of the renderer's graph-visible buffers.
const IDENTITY_LEAVES: u64 = (0xAE7 << 40) | 1;
const IDENTITY_INDIRECT: u64 = (0xAE7 << 40) | 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct FrameUbo {
    view_proj: [f32; 16],
    cam_pos: [f32; 4],
    lod_cam_pos: [f32; 4],
    lod_frustum: [[f32; 4]; 6],
    draw_frustum: [[f32; 4]; 6],
    lod: [f32; 4],
    map: [f32; 4],
    height: [f32; 4],
    sun: [f32; 4],
    cfg: [u32; 4],
    fog: [f32; 4],
    water: [f32; 4],
    /// Vision grid: width, height (cells), cell size, u32 words per row.
    vision: [f32; 4],
    /// The units' shadow cascades ([`crate::shadow`]).
    unit_shadow: crate::shadow::CascadeUniforms,
    /// x: effect time (decals).
    effects: [f32; 4],
    /// x: exposure, y: unused, z: the sun's illuminance (lux), w: the
    /// sky's (lux).
    photometry: [f32; 4],
    decal_relief: [u32; 4],
}

/// The cell layout of the vision (fog of war) planes a renderer can show:
/// one bit per cell, rows of `stride` 64-bit words.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionGrid {
    pub width: u32,
    pub height: u32,
    pub cell_size: f32,
}

impl VisionGrid {
    /// 64-bit words per row.
    pub fn stride(&self) -> usize {
        (self.width as usize).div_ceil(64)
    }

    /// 64-bit words of one plane.
    pub fn words(&self) -> usize {
        self.stride() * self.height as usize
    }
}

/// What the LOD is computed for (kept apart from the render camera so it
/// can be frozen).
#[derive(Clone, Copy, Debug)]
pub struct LodView {
    pub pos: Vec3,
    pub view_proj: Mat4,
    /// Pixels per world unit at distance 1 along the view axis.
    pub proj_scale: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameParams {
    pub view_proj: Mat4,
    pub cam_pos: Vec3,
    pub lod_view: LodView,
    /// Target screen-space error in pixels.
    pub tau: f32,
    pub update_lod: bool,
    pub wireframe: Wireframe,
    pub lod_colors: bool,
    pub fog: bool,
    /// Shade with the material tileset (when the renderer has one) instead
    /// of the palette.
    pub tileset: bool,
    /// With the tileset: show each pixel's layer blend weights as flat
    /// colours, unlit (a debug view).
    pub weights: bool,
    /// Shade by the vision planes last written (`write_vision`): full in
    /// line of sight, dimmed where explored, dark elsewhere.
    pub vision: bool,
    /// The water surface is drawn this frame (`record_water`): the terrain
    /// adds caustics under it.
    pub water: bool,
    /// Animation time in seconds (waves, foam, caustics).
    pub time: f32,
    /// Effect time in seconds (decals' fades and glow, particles): the sim
    /// time drawn when a unit stream is live, so effects follow the sim's
    /// speed; otherwise `time`.
    pub effect_time: f32,
    pub decal_relief: crate::decals::DecalReliefSettings,
    /// Shade by the hills' shadows (when the renderer has them).
    pub shadows: bool,
    /// Shade by the units' shadow map with these cascades (the graph
    /// draws and samples it this frame).
    pub unit_shadows: Option<Cascades>,
    /// Photometric lighting: exposure and the sun's and sky's illuminances
    /// in lux (y: 1 when the working space is BT.2020).
    pub photometry: [f32; 4],
    /// The scene's working space (the sky's radiance is its fog and clear
    /// colour).
    pub working: crate::colorspace::WorkingSpace,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wireframe {
    Off,
    Triangles,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct FrameStats {
    /// CBT leaves (the whole subdivision).
    pub leaves: u32,
    /// Leaves in the draw list (inside the view frustum).
    pub drawn: u32,
}

/// The map as the terrain shaders see it.
#[derive(Clone, Copy, Debug)]
pub struct TerrainInfo {
    pub cells: (u32, u32),
    pub root_cells: u32,
    pub cell_size: f32,
    /// World height = normalised sample × scale + offset.
    pub height_scale: f32,
    pub height_offset: f32,
    /// World height of the water surface, if the map has water.
    pub water_level: Option<f32>,
    /// A material id per cell (`cells.0 × cells.1`, row-major) colours the
    /// ground; without it the colour comes from height and slope.
    pub has_materials: bool,
}

impl TerrainInfo {
    pub fn new(
        heightmap: &Heightmap,
        pyramid: &Pyramid,
        cell_size: f32,
        height_scale: f32,
        height_offset: f32,
        water_level: Option<f32>,
    ) -> Self {
        let (cx, cy) = heightmap.cells();
        Self {
            cells: (cx as u32, cy as u32),
            root_cells: pyramid.root_cells as u32,
            cell_size,
            height_scale,
            height_offset,
            water_level,
            has_materials: false,
        }
    }

    pub fn extent(&self) -> (f32, f32) {
        (
            self.cells.0 as f32 * self.cell_size,
            self.cells.1 as f32 * self.cell_size,
        )
    }
}

struct TerrainReflection {
    prepare: vk::Pipeline,
    cull: vk::Pipeline,
    draw: vk::Pipeline,
    surface: vk::Pipeline,
}

struct TerrainRefraction {
    terrain: vk::Pipeline,
    decals: Option<[vk::Pipeline; DecalReliefMode::ALL.len()]>,
    underwater: vk::Pipeline,
    refract: vk::Pipeline,
    full: vk::Pipeline,
    seabed: OwnedImage,
    seabed_index: u32,
    seabed_group: crate::bindless::GroupId,
}

pub struct TerrainRenderer {
    pub max_depth: u32,
    info: TerrainInfo,
    color_format: vk::Format,
    depth_format: vk::Format,

    pipeline_layout: vk::PipelineLayout,
    /// The buffers' handles (terrain_common.glsl): the frame block, the
    /// tree, bits, snapshot, leaves, indirect, the tileset block, the
    /// vision, the decal list, the decal materials, then the lights.
    table: crate::bindless::HandleTable,
    /// The vision planes' slotted index (exposure reads them too).
    vision_handle: u32,
    prepare: vk::Pipeline,
    update: vk::Pipeline,
    reduce: vk::Pipeline,
    terrain: vk::Pipeline,
    /// `None` without `fillModeNonSolid`.
    wire: Option<vk::Pipeline>,
    /// The water surface; `None` on a map without water.
    water: Option<vk::Pipeline>,
    reflection: Option<TerrainReflection>,
    refraction: Option<TerrainRefraction>,
    wave_surfaces: Option<[vk::Pipeline; 4]>,
    wave_underwater: Option<vk::Pipeline>,
    volume: [Option<(vk::Pipeline, vk::Pipeline)>; 4],
    frame_index: u32,

    tree: OwnedBuffer,
    bits: OwnedBuffer,
    snap: OwnedBuffer,
    leaves: OwnedBuffer,
    indirect: OwnedBuffer,
    submesh_vb: OwnedBuffer,
    submesh_ib: OwnedBuffer,
    ubos: Vec<OwnedBuffer>,
    stats: Vec<OwnedBuffer>,
    height_img: OwnedImage,
    pyramid_img: OwnedImage,
    error_img: OwnedImage,
    material_img: OwnedImage,
    /// Tileset albedo + height and normal arrays (1x1 placeholders without
    /// a tileset), and its table.
    tile_albedo: OwnedImage,
    tile_normal: OwnedImage,
    tile_ubo: OwnedBuffer,
    /// The anti-tiling index noise (`tileset::anti_tiling_noise`; 1x1
    /// without a tileset that uses it).
    tile_noise: OwnedImage,
    has_tileset: bool,
    /// World shadow height per corner ([`super::sun`]); 1x1 without shadows.
    shadow_img: OwnedImage,
    has_shadows: bool,
    /// The units' shadow map (a 1x1 stand-in without it).
    unit_shadow_map: ShadowMap,
    /// The terrain pipeline with decals (specialization constant 1), when
    /// made with them; used for a slot while it has any.
    terrain_decals: Option<[vk::Pipeline; DecalReliefMode::ALL.len()]>,
    /// Per slot: the decals ([`crate::decals`]; host-visible, 16 bytes
    /// without), and whether it has any.
    decal_bufs: Vec<OwnedBuffer>,
    decal_live: Vec<Option<DecalReliefMode>>,
    /// The decal atlas (1x1 without decals).
    decal_atlas: OwnedImage,
    /// Its linear height, metalness, roughness and glow, layer for layer.
    decal_surface: OwnedImage,
    /// Its signed tangent normals, layer for layer.
    decal_normals: OwnedImage,
    decal_cones: OwnedImage,
    /// Legacy metallic factors and the weight of material texture metalness.
    decal_materials: OwnedBuffer,
    /// Per slot: line-of-sight and explored bit planes (host-visible;
    /// 16 bytes without a vision grid).
    vision_bufs: Vec<OwnedBuffer>,
    vision: Option<VisionGrid>,
    /// The global sampled indices of the items drawn above, and the pass's
    /// 128-byte push block (terrain.frag/water.frag macros name them).
    sampled: [u32; 12],
    push: crate::bindless::Push,
    /// The global-set handles, given back by [`Self::retire`].
    group: Option<crate::bindless::GroupId>,
}

/// Texels to rewrite after the terrain changed (see
/// [`super::edit::TerrainEditor`]): regions of the height texture, the
/// pyramid and error mips, and the material texture. Applied all at once,
/// so a frame never sees new heights with old pyramid bounds.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainUpdate {
    /// Terrain version this brings the textures to (for reports).
    pub version: u32,
    pub heights: Vec<upload::Region>,
    pub pyramid: Vec<upload::Region>,
    pub errors: Vec<upload::Region>,
    pub materials: Vec<upload::Region>,
    /// Shadow heights ([`super::sun`]).
    pub shadows: Vec<upload::Region>,
}

impl TerrainUpdate {
    pub fn bytes(&self) -> usize {
        [
            &self.heights,
            &self.pyramid,
            &self.errors,
            &self.materials,
            &self.shadows,
        ]
        .iter()
        .flat_map(|r| r.iter())
        .map(|r| r.bytes.len())
        .sum()
    }
}

fn bytes_of<T: Copy>(value: &T) -> &[u8] {
    pipeline::push_bytes(value)
}

fn slice_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // Safety: plain-old-data (integers and floats) only.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

impl TerrainRenderer {
    /// The renderer and the shadow map it owns, as one group of global-set
    /// handles: [`Self::retire`] gives them back.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &GpuDevice,
        bindless: &mut crate::bindless::Bindless,
        heightmap: &Heightmap,
        pyramid: &Pyramid,
        info: TerrainInfo,
        materials: Option<&[u8]>,
        tileset: Option<&Tileset>,
        vision: Option<VisionGrid>,
        shadows: Option<&[f32]>,
        unit_shadows: bool,
        decals: Option<(&DecalAtlas, u32)>,
        lights: &crate::lights::LightRenderer,
        color_format: vk::Format,
        depth_format: vk::Format,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let (mut this, group) = bindless.group(|bindless| {
            Self::build(
                gpu,
                bindless,
                heightmap,
                pyramid,
                info,
                materials,
                tileset,
                vision,
                shadows,
                unit_shadows,
                decals,
                lights,
                color_format,
                depth_format,
                frames_in_flight,
            )
        })?;
        this.group = Some(group);
        Ok(this)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        gpu: &GpuDevice,
        bindless: &mut crate::bindless::Bindless,
        heightmap: &Heightmap,
        pyramid: &Pyramid,
        info: TerrainInfo,
        materials: Option<&[u8]>,
        tileset: Option<&Tileset>,
        vision: Option<VisionGrid>,
        shadows: Option<&[f32]>,
        unit_shadows: bool,
        decals: Option<(&DecalAtlas, u32)>,
        lights: &crate::lights::LightRenderer,
        color_format: vk::Format,
        depth_format: vk::Format,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let device = &gpu.device;
        let mut info = info;
        info.has_materials = materials.is_some();
        if let Some(materials) = materials {
            anyhow::ensure!(
                materials.len() == (info.cells.0 * info.cells.1) as usize,
                "one material per cell"
            );
        }
        let k = pyramid.root_cells.trailing_zeros();
        // Splitting stops by itself once a leaf's sub-mesh has one vertex
        // per sample (around depth 2k-2). One spare level, and T = D-5 >= 8
        // for the shared-memory reduce.
        let max_depth = (2 * k - 1).clamp(13, 30);
        let cbt = Cbt::new(max_depth);

        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let tree = upload::buffer(gpu, slice_bytes(&cbt.tree), storage, "terrain tree")?;
        let bits = upload::buffer(
            gpu,
            slice_bytes(&cbt.bits),
            storage | vk::BufferUsageFlags::TRANSFER_SRC,
            "terrain bits",
        )?;
        let snap = OwnedBuffer::new(
            gpu,
            BufferDesc {
                size: bits.size,
                usage: storage | vk::BufferUsageFlags::TRANSFER_DST,
            },
            MemoryLocation::GpuOnly,
            "terrain snapshot",
        )?;
        let capacity = MAX_DRAW_LEAVES.min(1 << max_depth);
        let leaves = OwnedBuffer::new(
            gpu,
            BufferDesc {
                size: capacity as u64 * LEAF_BYTES,
                usage: storage,
            },
            MemoryLocation::GpuOnly,
            "terrain leaves",
        )?;

        let (weights, indices) = leb::build_submesh();
        let index_count = indices.len() as u32;
        let mut words = [0u32; IND_WORDS as usize];
        words[3] = index_count;
        let indirect = upload::buffer(
            gpu,
            slice_bytes(&words),
            storage | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "terrain indirect",
        )?;
        let vertices: Vec<[i32; 4]> = weights.iter().map(|w| [w[0], w[1], w[2], 0]).collect();
        let submesh_vb = upload::buffer(
            gpu,
            slice_bytes(&vertices),
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "terrain sub-mesh vertices",
        )?;
        let submesh_ib = upload::buffer(
            gpu,
            slice_bytes(&indices),
            vk::BufferUsageFlags::INDEX_BUFFER,
            "terrain sub-mesh indices",
        )?;

        let ubos = (0..frames_in_flight)
            .map(|_| {
                OwnedBuffer::new(
                    gpu,
                    BufferDesc {
                        size: (size_of::<FrameUbo>()
                            + usize::from(info.water_level.is_some())
                                * (size_of::<MirrorFrame>() + underwater::FRAME_BYTES))
                            as u64,
                        usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                    },
                    MemoryLocation::CpuToGpu,
                    "terrain frame",
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let vision_bytes = vision.map_or(16, |v| 2 * v.words() as u64 * 8);
        let vision_bufs = (0..frames_in_flight)
            .map(|_| {
                OwnedBuffer::new(
                    gpu,
                    BufferDesc {
                        size: vision_bytes,
                        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                    },
                    MemoryLocation::CpuToGpu,
                    "terrain vision",
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let decal_bytes = decals.map_or(16, |(_, capacity)| crate::decals::buffer_words(capacity) as u64 * 4);
        let decal_bufs = (0..frames_in_flight)
            .map(|_| {
                OwnedBuffer::new(
                    gpu,
                    BufferDesc {
                        size: decal_bytes,
                        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                    },
                    MemoryLocation::CpuToGpu,
                    "terrain decals",
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let (decal_atlas, decal_surface, decal_normals, decal_cones) = match decals {
            Some((atlas, _)) => {
                atlas.validate().map_err(anyhow::Error::msg)?;
                // Mip 0 of each layer; the mips are made on the GPU.
                let colour: Vec<&[u8]> = atlas.colour.iter().map(|l| l.as_flattened()).collect();
                let layers: Vec<_> = atlas.gpu_surfaces().collect();
                let surface: Vec<&[u8]> = layers.iter().map(|l| l.material.as_flattened()).collect();
                let normals: Vec<&[u8]> = layers.iter().map(|l| l.normals.as_flattened()).collect();
                let array = vk::ImageViewType::TYPE_2D_ARRAY;
                (
                    upload::image_mipped(
                        gpu,
                        vk::Format::R8G8B8A8_SRGB,
                        atlas.size,
                        atlas.size,
                        &colour,
                        MipFilter::Coverage,
                        array,
                        "decal atlas",
                    )?,
                    upload::image_mipped(
                        gpu,
                        vk::Format::R8G8B8A8_UNORM,
                        atlas.size,
                        atlas.size,
                        &surface,
                        MipFilter::Average,
                        array,
                        "decal surface",
                    )?,
                    upload::image_mipped(
                        gpu,
                        vk::Format::R16G16_SNORM,
                        atlas.size,
                        atlas.size,
                        &normals,
                        MipFilter::NormalSigned,
                        array,
                        "decal tangent normals",
                    )?,
                    crate::decals_relief::upload_cones(gpu, atlas, &layers)?,
                )
            }
            None => {
                let one = |format, texel: [u8; 4], name| {
                    upload::image_layers(
                        gpu,
                        format,
                        1,
                        1,
                        &[vec![&texel[..]]],
                        vk::ImageViewType::TYPE_2D_ARRAY,
                        name,
                    )
                };
                (
                    one(vk::Format::R8G8B8A8_SRGB, [0; 4], "decal atlas")?,
                    one(vk::Format::R8G8B8A8_UNORM, [128, 0, 255, 0], "decal surface")?,
                    one(vk::Format::R16G16_SNORM, [0; 4], "decal tangent normals")?,
                    upload::image_layers(
                        gpu,
                        vk::Format::R32G32B32A32_SFLOAT,
                        1,
                        1,
                        &[vec![&[0; 16]]],
                        vk::ImageViewType::TYPE_2D_ARRAY,
                        "decal cones",
                    )?,
                )
            }
        };
        let mut material_texels = decals.map_or_else(|| vec![[0.0; 4]], |(atlas, _)| atlas.gpu_materials());
        if let Some((atlas, _)) = decals {
            for &layer in &atlas.height_layers {
                let (lo, hi) = atlas.surface[layer as usize]
                    .iter()
                    .fold((128u8, 128u8), |(lo, hi), t| (lo.min(t[0]), hi.max(t[0])));
                material_texels[layer as usize][2] = f32::from(lo) / 255.0;
                material_texels[layer as usize][3] = f32::from(hi) / 255.0;
            }
        }
        let decal_materials = upload::buffer(gpu, slice_bytes(&material_texels), storage, "decal materials")?;
        let stats = (0..frames_in_flight)
            .map(|_| {
                OwnedBuffer::new(
                    gpu,
                    BufferDesc {
                        size: IND_WORDS * 4,
                        usage: vk::BufferUsageFlags::TRANSFER_DST,
                    },
                    MemoryLocation::GpuToCpu,
                    "terrain stats",
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let height_img = upload::image(
            gpu,
            vk::Format::R32_SFLOAT,
            heightmap.width as u32,
            heightmap.height as u32,
            &[slice_bytes(&heightmap.data)],
            "terrain heights",
        )?;
        let mips: Vec<&[u8]> = pyramid.levels.iter().map(|l| slice_bytes(l.as_slice())).collect();
        let base = (pyramid.root_cells >> PYRAMID_BASE_LEVEL) as u32;
        let pyramid_img = upload::image(
            gpu,
            vk::Format::R32G32B32A32_SFLOAT,
            base,
            base,
            &mips,
            "terrain pyramid",
        )?;
        let error_mips: Vec<&[u8]> = pyramid.errors.iter().map(|l| slice_bytes(l.as_slice())).collect();
        let half = (pyramid.root_cells >> 1) as u32;
        let error_img = upload::image(gpu, vk::Format::R32_SFLOAT, half, half, &error_mips, "terrain errors")?;

        let material_img = match materials {
            Some(m) => upload::image(
                gpu,
                vk::Format::R8_UINT,
                info.cells.0,
                info.cells.1,
                &[m],
                "terrain materials",
            )?,
            None => upload::image(gpu, vk::Format::R8_UINT, 1, 1, &[&[0]], "terrain materials")?,
        };
        let shadow_img = match shadows {
            Some(s) => {
                anyhow::ensure!(s.len() == heightmap.data.len(), "one shadow height per corner");
                upload::image(
                    gpu,
                    vk::Format::R32_SFLOAT,
                    heightmap.width as u32,
                    heightmap.height as u32,
                    &[slice_bytes(s)],
                    "terrain shadow heights",
                )?
            }
            None => upload::image(gpu, vk::Format::R32_SFLOAT, 1, 1, &[&[0; 4]], "terrain shadow heights")?,
        };
        let unit_shadow_map = ShadowMap::new(gpu, bindless, unit_shadows)?;
        let array = vk::ImageViewType::TYPE_2D_ARRAY;
        let (tile_albedo, tile_normal, tile_table) = match tileset {
            Some(t) => {
                t.validate().map_err(anyhow::Error::msg)?;
                // Mip 0 of each layer; the mips are made on the GPU (a box on
                // power-of-two sides never crosses the tile edge, so they tile).
                let albedo: Vec<&[u8]> = t.layers.iter().map(|l| l.albedo.as_slice()).collect();
                let normal: Vec<&[u8]> = t.layers.iter().map(|l| l.normal.as_slice()).collect();
                (
                    upload::image_mipped(
                        gpu,
                        vk::Format::R8G8B8A8_SRGB,
                        t.size,
                        t.size,
                        &albedo,
                        MipFilter::Average,
                        array,
                        "tileset albedo",
                    )?,
                    upload::image_mipped(
                        gpu,
                        vk::Format::R8G8_UNORM,
                        t.size,
                        t.size,
                        &normal,
                        MipFilter::Normal,
                        array,
                        "tileset normals",
                    )?,
                    t.ubo(),
                )
            }
            None => (
                upload::image_layers(
                    gpu,
                    vk::Format::R8G8B8A8_SRGB,
                    1,
                    1,
                    &[vec![&[0u8; 4][..]]],
                    array,
                    "tileset albedo",
                )?,
                upload::image_layers(
                    gpu,
                    vk::Format::R8G8_UNORM,
                    1,
                    1,
                    &[vec![&[128u8, 128][..]]],
                    array,
                    "tileset normals",
                )?,
                TilesetUbo::default(),
            ),
        };
        let tile_noise = match tileset {
            Some(t) if t.anti_tiling > 0.0 => {
                let noise = super::tileset::anti_tiling_noise();
                let size = super::tileset::NOISE_SIZE;
                upload::image_mipped(
                    gpu,
                    vk::Format::R8_UNORM,
                    size,
                    size,
                    &[&noise],
                    MipFilter::Average,
                    vk::ImageViewType::TYPE_2D,
                    "anti-tiling noise",
                )?
            }
            _ => upload::image(gpu, vk::Format::R8_UNORM, 1, 1, &[&[128u8][..]], "anti-tiling noise")?,
        };
        let tile_ubo = upload::buffer(
            gpu,
            &bytes_of(&tile_table)[..TILESET_UBO_BYTES],
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "tileset table",
        )?;

        let make_sampler = |filter: vk::Filter| {
            vk::SamplerCreateInfo::default()
                .mag_filter(filter)
                .min_filter(filter)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(vk::LOD_CLAMP_NONE)
        };
        let samplers = [
            make_sampler(vk::Filter::NEAREST),
            make_sampler(vk::Filter::LINEAR),
            vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(vk::LOD_CLAMP_NONE),
        ];
        let sampled = [
            bindless
                .sampled_2d(pyramid_img.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
            bindless
                .sampled_2d(height_img.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
            bindless
                .sampled_2d(error_img.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
            bindless
                .sampled_u2d(material_img.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
            bindless
                .sampled_2d_array(tile_albedo.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
            bindless
                .sampled_2d_array(tile_normal.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
            bindless
                .sampled_2d(shadow_img.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
            unit_shadow_map.sampled_index(),
            unit_shadow_map.color_sampled_index(),
            bindless
                .sampled_2d_array(decal_atlas.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
            bindless
                .sampled_2d_array(decal_surface.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
            bindless
                .sampled_2d(tile_noise.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
        ];
        let mut push = crate::bindless::Push::default();
        for (at, &index) in sampled.iter().enumerate() {
            push.image(at, index);
        }
        push.sampler(0, bindless.sampler(&samplers[0])?.index);
        push.sampler(1, bindless.sampler(&samplers[1])?.index);
        push.sampler(2, bindless.sampler(&samplers[2])?.index);
        push.sampler(3, unit_shadow_map.color_sampler_index());
        push.sampler(4, unit_shadow_map.sampler_index());
        let decal_sampler = make_sampler(vk::Filter::LINEAR)
            .anisotropy_enable(decals.is_some() && gpu.max_sampler_anisotropy.is_some())
            .max_anisotropy(gpu.max_sampler_anisotropy.unwrap_or(1.0).min(8.0));
        push.sampler(5, bindless.sampler(&decal_sampler)?.index);
        push.image(
            14,
            bindless
                .sampled_2d_array(decal_normals.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
        );
        push.image(
            15,
            bindless
                .sampled_2d_array(decal_cones.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?
                .0
                .index,
        );

        // Every buffer is in the global set (M15 step D); the per-slot ones
        // are slotted, so the table is the same every frame.
        let vision_handle = bindless.per_slot_buffer(&vision_bufs)?.index;
        let mut handles = vec![
            bindless.per_slot_uniform_buffer(&ubos)?.index,
            bindless.whole_buffer(&tree)?.index,
            bindless.whole_buffer(&bits)?.index,
            bindless.whole_buffer(&snap)?.index,
            bindless.whole_buffer(&leaves)?.index,
            bindless.whole_buffer(&indirect)?.index,
            bindless.whole_uniform_buffer(&tile_ubo)?.index,
            vision_handle,
            bindless.per_slot_buffer(&decal_bufs)?.index,
            bindless.whole_buffer(&decal_materials)?.index,
        ];
        handles.extend(lights.handles());
        let table = crate::bindless::HandleTable::new(gpu, bindless, &handles, "terrain handles")?;
        push.table(table.index);
        let pipeline_layout = bindless.pipeline_layout();

        let prepare = pipeline::compute_pipeline(device, spirv!("cbt_prepare.comp"), pipeline_layout)?;
        let update = pipeline::compute_pipeline(device, spirv!("cbt_update.comp"), pipeline_layout)?;
        let reduce = pipeline::compute_pipeline(device, spirv!("cbt_reduce.comp"), pipeline_layout)?;
        let terrain = graphics_pipeline(
            device,
            pipeline_layout,
            color_format,
            depth_format,
            spirv!("terrain.frag"),
            false,
            unit_shadows,
            None,
            false,
        )?;
        let terrain_decals = decals
            .map(|_| {
                reflect::pipelines(device, |i| {
                    graphics_pipeline(
                        device,
                        pipeline_layout,
                        color_format,
                        depth_format,
                        spirv!("terrain.frag"),
                        false,
                        unit_shadows,
                        Some(DecalReliefMode::ALL[i]),
                        false,
                    )
                })
            })
            .transpose()?;
        let wire = if gpu.fill_mode_non_solid {
            Some(graphics_pipeline(
                device,
                pipeline_layout,
                color_format,
                depth_format,
                spirv!("terrain_wire.frag"),
                true,
                false,
                None,
                false,
            )?)
        } else {
            None
        };
        let water = info
            .water_level
            .map(|_| {
                water_pipeline(
                    device,
                    pipeline_layout,
                    color_format,
                    depth_format,
                    spirv!("water.frag"),
                )
            })
            .transpose()?;

        Ok(Self {
            max_depth,
            info,
            color_format,
            depth_format,
            pipeline_layout,
            table,
            vision_handle,
            prepare,
            update,
            reduce,
            terrain,
            wire,
            water,
            reflection: None,
            refraction: None,
            wave_surfaces: None,
            wave_underwater: None,
            volume: [None; 4],
            frame_index: handles[0],
            tree,
            bits,
            snap,
            leaves,
            indirect,
            submesh_vb,
            submesh_ib,
            ubos,
            stats,
            height_img,
            pyramid_img,
            error_img,
            material_img,
            tile_albedo,
            tile_normal,
            tile_ubo,
            tile_noise,
            has_tileset: tileset.is_some(),
            shadow_img,
            has_shadows: shadows.is_some(),
            unit_shadow_map,
            terrain_decals,
            decal_live: vec![None; frames_in_flight],
            decal_bufs,
            decal_atlas,
            decal_surface,
            decal_normals,
            decal_cones,
            decal_materials,
            vision_bufs,
            vision,
            sampled,
            push,
            group: None,
        })
    }

    /// Give the global-set handles back (the shadow map's too); the device
    /// objects stay until [`Self::destroy`].
    pub fn retire(&mut self, bindless: &mut crate::bindless::Bindless) {
        if let Some(refraction) = &self.refraction {
            bindless.retire_group(refraction.seabed_group);
        }
        if let Some(group) = self.group.take() {
            bindless.retire_group(group);
        }
    }

    /// Rewrite the textures `update` covers. The device must be idle (no
    /// frame in flight samples them); waits for the copy.
    pub fn apply(&mut self, gpu: &GpuDevice, update: &TerrainUpdate) -> Result<()> {
        let materials: &[upload::Region] = if self.info.has_materials {
            &update.materials
        } else {
            &[]
        };
        let shadows: &[upload::Region] = if self.has_shadows { &update.shadows } else { &[] };
        upload::update_images(
            gpu,
            &[
                (&self.height_img, &update.heights),
                (&self.pyramid_img, &update.pyramid),
                (&self.error_img, &update.errors),
                (&self.material_img, materials),
                (&self.shadow_img, shadows),
            ],
        )
    }

    /// The units' shadow map (the 1x1 stand-in unless made with
    /// `unit_shadows`). Lives as long as the renderer.
    pub fn unit_shadow_map(&self) -> &ShadowMap {
        &self.unit_shadow_map
    }

    /// Whether it was made with the hills' shadows.
    pub fn has_shadows(&self) -> bool {
        self.has_shadows
    }

    pub fn has_tileset(&self) -> bool {
        self.has_tileset
    }

    pub fn info(&self) -> &TerrainInfo {
        &self.info
    }

    pub fn formats(&self) -> (vk::Format, vk::Format) {
        (self.color_format, self.depth_format)
    }

    /// The draw list, for the graph (written by the LOD pass, read by the
    /// draw pass).
    pub fn leaves(&self) -> BufferBinding {
        BufferBinding {
            buffer: self.leaves.buffer,
            offset: 0,
            size: self.leaves.size,
            identity: IDENTITY_LEAVES,
            storage: None,
        }
    }

    /// The indirect arguments, for the graph.
    pub fn indirect(&self) -> BufferBinding {
        BufferBinding {
            buffer: self.indirect.buffer,
            offset: 0,
            size: self.indirect.size,
            identity: IDENTITY_INDIRECT,
            storage: None,
        }
    }

    pub fn leaves_desc(&self) -> BufferDesc {
        BufferDesc {
            size: self.leaves.size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        }
    }

    pub fn indirect_desc(&self) -> BufferDesc {
        BufferDesc {
            size: self.indirect.size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC,
        }
    }

    /// Counts written by the frame that last used `slot` (call after its
    /// fence, i.e. in `prepare`).
    pub fn read_stats(&mut self, slot: usize) -> FrameStats {
        let Some(bytes) = self.stats[slot].mapped() else {
            return FrameStats::default();
        };
        let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        FrameStats {
            leaves: word(8),
            drawn: word(4),
        }
    }

    /// Per-frame uniforms for `slot` (its previous submission has completed).
    pub fn write_frame(&mut self, slot: usize, p: &FrameParams) {
        let planes = frustum_planes;
        let i = &self.info;
        let unit_shadow = Cascades::uniforms(p.unit_shadows.as_ref().filter(|_| self.unit_shadow_map.is_real()));
        let relief_scale = if p.decal_relief.mode == DecalReliefMode::RobustConeStepMapping {
            RCSM_HEIGHT_SCALE
        } else {
            1.0
        };
        let ubo = FrameUbo {
            view_proj: p.view_proj.to_cols_array(),
            cam_pos: p.cam_pos.extend(1.0).to_array(),
            lod_cam_pos: p.lod_view.pos.extend(1.0).to_array(),
            lod_frustum: planes(p.lod_view.view_proj),
            draw_frustum: planes(p.view_proj),
            lod: [p.tau, p.lod_view.proj_scale, 4.0, 0.7],
            map: [i.cells.0 as f32, i.cells.1 as f32, i.root_cells as f32, i.cell_size],
            height: [
                i.height_scale,
                i.height_offset,
                1.0 / (i.cells.0 + 1) as f32,
                1.0 / (i.cells.1 + 1) as f32,
            ],
            sun: super::sun::direction().extend(0.0).to_array(),
            cfg: [
                self.max_depth,
                self.max_depth - 5,
                p.lod_colors as u32
                    | if p.wireframe == Wireframe::Triangles { 2 } else { 0 }
                    | if i.has_materials { 8 } else { 0 }
                    | if self.has_tileset && p.tileset { 16 } else { 0 }
                    | if self.has_tileset && p.tileset && p.weights {
                        32
                    } else {
                        0
                    }
                    | if self.vision.is_some() && p.vision { 64 } else { 0 }
                    | if self.water.is_some() && p.water { 128 } else { 0 }
                    | if self.has_shadows && p.shadows { 256 } else { 0 },
                MAX_DRAW_LEAVES.min(1 << self.max_depth),
            ],
            // Distance fog: the sky's radiance in the working space.
            fog: {
                let c = crate::lighting::sky_radiance(p.working);
                [
                    c[0],
                    c[1],
                    c[2],
                    if p.fog {
                        1.0 / (i.cells.0.max(i.cells.1) as f32 * i.cell_size * 1.2)
                    } else {
                        0.0
                    },
                ]
            },
            water: match i.water_level {
                Some(level) => [level, 1.0, 0.08, p.time],
                None => [0.0, 0.0, 0.0, p.time],
            },
            vision: self.vision_uniform().unwrap_or([0.0; 4]),
            unit_shadow,
            effects: [
                p.effect_time,
                if p.decal_relief.strength.is_finite() {
                    p.decal_relief.strength.max(0.0) * relief_scale
                } else {
                    0.0
                },
                p.decal_relief.min_parallax_pixels.max(0.0),
                p.decal_relief
                    .full_parallax_pixels
                    .max(p.decal_relief.min_parallax_pixels.max(0.0) + 1e-3),
            ],
            decal_relief: [
                p.decal_relief.overlap_mode as u32 | ((p.decal_relief.adaptive_quality as u32) << 2),
                p.decal_relief.min_view_normal_dot.clamp(0.03, 0.97).to_bits(),
                p.decal_relief.max_cone_steps.min(256),
                p.decal_relief.max_refine_steps.min(24),
            ],
            photometry: p.photometry,
        };
        if let Some(bytes) = self.ubos[slot].mapped() {
            bytes[..size_of::<FrameUbo>()].copy_from_slice(bytes_of(&ubo));
            if self.reflection.is_some()
                && let Some(level) = i.water_level
            {
                MirrorFrame::new(p.view_proj, p.cam_pos, level).write(bytes, size_of::<FrameUbo>());
            }
        }
    }

    pub(crate) fn write_water(&mut self, slot: usize, view_proj: Mat4, params: WaterParams) {
        if self.refraction.is_some()
            && let Some(bytes) = self.ubos[slot].mapped()
        {
            underwater::write(
                &mut bytes[size_of::<FrameUbo>() + size_of::<MirrorFrame>()..],
                view_proj,
                params,
            );
        }
    }

    /// The vision grid this renderer was made for.
    pub fn vision_grid(&self) -> Option<VisionGrid> {
        self.vision
    }

    /// The vision planes' layout as the shaders read it (`u.vision`: cells
    /// wide, high, the cell size and the u32 words per row), or `None`
    /// without a vision grid.
    pub fn vision_uniform(&self) -> Option<[f32; 4]> {
        self.vision
            .map(|v| [v.width as f32, v.height as f32, v.cell_size, (v.stride() * 2) as f32])
    }

    /// Slot `slot`'s vision planes (line of sight, then explored), as
    /// [`Self::write_vision`] last wrote them; 16 bytes without a grid.
    /// The vision planes' global (slotted) index.
    pub fn vision_handle(&self) -> u32 {
        self.vision_handle
    }

    pub fn vision_buffer(&self, slot: usize) -> vk::Buffer {
        self.vision_bufs[slot].buffer
    }

    /// Write `slot`'s vision planes (its previous submission has completed):
    /// line of sight and explored, [`VisionGrid::words`] each.
    pub fn write_vision(&mut self, slot: usize, los: &[u64], explored: &[u64]) -> Result<()> {
        let v = self.vision.context("this terrain renderer has no vision grid")?;
        anyhow::ensure!(
            los.len() == v.words() && explored.len() == v.words(),
            "vision planes of {} and {} words for a {}x{} grid",
            los.len(),
            explored.len(),
            v.width,
            v.height
        );
        let bytes = self.vision_bufs[slot].mapped().context("vision buffer is not mapped")?;
        let half = v.words() * 8;
        bytes[..half].copy_from_slice(slice_bytes(los));
        bytes[half..2 * half].copy_from_slice(slice_bytes(explored));
        Ok(())
    }

    /// Whether it was made with decals.
    pub fn has_decals(&self) -> bool {
        self.terrain_decals.is_some()
    }

    /// Write `slot`'s decals (its previous submission has completed): the
    /// words of [`crate::decals`]' buffer. Whether the draw uses the
    /// pipeline with decals is [`Self::set_decals_live`].
    pub fn write_decals(&mut self, slot: usize, words: &[u32]) -> Result<()> {
        anyhow::ensure!(self.terrain_decals.is_some(), "this terrain renderer has no decals");
        let bytes = self.decal_bufs[slot].mapped().context("decal buffer is not mapped")?;
        anyhow::ensure!(
            bytes.len() >= words.len() * 4,
            "{} decal words for a buffer of {} bytes",
            words.len(),
            bytes.len()
        );
        bytes[..words.len() * 4].copy_from_slice(slice_bytes(words));
        Ok(())
    }

    /// Whether `slot` draws with decals (made with them, written, some in
    /// view); set every frame.
    pub fn set_decals_live(&mut self, slot: usize, live: bool, mode: DecalReliefMode) {
        self.decal_live[slot] = (live && self.terrain_decals.is_some()).then_some(mode);
    }

    /// The LOD update: split and merge passes, reduction, draw list. Records
    /// inside a compute pass that declares `leaves` and `indirect` as
    /// storage writes.
    pub fn record_lod(&self, ctx: &PassContext<'_>, update_lod: bool) {
        let (device, cmd) = (ctx.device, ctx.cmd);
        let full = || unsafe {
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&barrier));
        };
        let all = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let push = |mode: u32, write_draw: u32, level: u32| unsafe {
            let mut block = self.push;
            block.words = [mode, write_draw, level, 0];
            device.cmd_push_constants(cmd, self.pipeline_layout, all, 0, bytes_of(&block));
        };
        let t = self.max_depth - 5;
        unsafe {
            // The previous frame's draw (and the snapshot copy) read the
            // buffers about to be rewritten.
            full();
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[ctx.global_set],
                &[],
            );
            let pass = |mode: u32, write_draw: bool| {
                // Decode against a snapshot so in-flight splits and merges
                // cannot disturb it.
                device.cmd_copy_buffer(
                    cmd,
                    self.bits.buffer,
                    self.snap.buffer,
                    &[vk::BufferCopy::default().size(self.bits.size)],
                );
                full();
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.prepare);
                push(0, write_draw as u32, 0);
                device.cmd_dispatch(cmd, 1, 1, 1);
                full();
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.update);
                push(mode, write_draw as u32, 0);
                device.cmd_dispatch_indirect(cmd, self.indirect.buffer, 0);
                full();
                if mode != MODE_NONE {
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.reduce);
                    push(0, 0, 0);
                    device.cmd_dispatch(cmd, ((1u32 << t) / 256).max(1), 1, 1);
                    full();
                    for level in (8..t).rev() {
                        push(1, 0, level);
                        device.cmd_dispatch(cmd, ((1u32 << level) / 256).max(1), 1, 1);
                        full();
                    }
                    push(2, 0, 0);
                    device.cmd_dispatch(cmd, 1, 1, 1);
                    full();
                }
            };
            if update_lod {
                pass(MODE_SPLIT, false);
                pass(MODE_MERGE, true);
            } else {
                pass(MODE_NONE, true);
            }
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.prepare);
            push(1, 0, 0);
            device.cmd_dispatch(cmd, 1, 1, 1);
            full();
            device.cmd_copy_buffer(
                cmd,
                self.indirect.buffer,
                self.stats[ctx.slot].buffer,
                &[vk::BufferCopy::default().size(IND_WORDS * 4)],
            );
            // The graph sees the last write as a compute write: make the
            // copy's read finish before the draw pass too.
            full();
        }
    }

    /// Draw the leaves. Records inside a graphics pass with a colour
    /// attachment of `color_format` and a depth attachment of
    /// `depth_format` (cleared to 0: reverse-Z).
    pub fn record_draw(&self, ctx: &PassContext<'_>, wireframe: Wireframe) {
        self.draw_terrain(ctx, wireframe, self.terrain, self.terrain_decals, &self.push);
    }

    pub(crate) fn record_draw_refract(&self, ctx: &PassContext<'_>, wireframe: Wireframe) -> Result<()> {
        let refract = self.refraction.as_ref().context("terrain refraction pipelines")?;
        let mut push = self.push;
        push.image(12, refract.seabed_index);
        self.draw_terrain(ctx, wireframe, refract.terrain, refract.decals, &push);
        Ok(())
    }

    fn draw_terrain(
        &self,
        ctx: &PassContext<'_>,
        wireframe: Wireframe,
        terrain: vk::Pipeline,
        decals: Option<[vk::Pipeline; DecalReliefMode::ALL.len()]>,
        push: &crate::bindless::Push,
    ) {
        let (device, cmd) = (ctx.device, ctx.cmd);
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[ctx.global_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(push),
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.submesh_vb.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.submesh_ib.buffer, 0, vk::IndexType::UINT16);
            let terrain = decals
                .zip(self.decal_live[ctx.slot])
                .map_or(terrain, |(p, mode)| p[mode as usize]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, terrain);
            device.cmd_draw_indexed_indirect(cmd, self.indirect.buffer, 3 * 4, 1, 0);
            if let (Wireframe::Triangles, Some(wire)) = (wireframe, self.wire) {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, wire);
                device.cmd_draw_indexed_indirect(cmd, self.indirect.buffer, 3 * 4, 1, 0);
            }
        }
    }

    pub(crate) fn prepare_reflection(&mut self, device: &ash::Device) -> Result<()> {
        if self.reflection.is_some() {
            return Ok(());
        }
        let [prepare, cull, draw, surface] = reflect::pipelines(device, |i| match i {
            0 => pipeline::compute_pipeline(device, spirv!("cbt_prepare_mirror.comp"), self.pipeline_layout),
            1 => pipeline::compute_pipeline(device, spirv!("cbt_update_mirror.comp"), self.pipeline_layout),
            2 => graphics_pipeline(
                device,
                self.pipeline_layout,
                self.color_format,
                self.depth_format,
                spirv!("terrain_mirror.frag"),
                false,
                self.unit_shadow_map.is_real(),
                None,
                true,
            ),
            _ => water_pipeline(
                device,
                self.pipeline_layout,
                self.color_format,
                self.depth_format,
                spirv!("water_reflect.frag"),
            ),
        })?;
        self.reflection = Some(TerrainReflection {
            prepare,
            cull,
            draw,
            surface,
        });
        Ok(())
    }

    pub(crate) fn prepare_refraction(
        &mut self,
        gpu: &GpuDevice,
        bindless: &mut crate::bindless::Bindless,
    ) -> Result<()> {
        let device = &gpu.device;
        if self.refraction.is_some() {
            return Ok(());
        }
        let terrain = |decals| {
            graphics_pipeline(
                device,
                self.pipeline_layout,
                self.color_format,
                self.depth_format,
                spirv!("terrain_refract_owner.frag"),
                false,
                self.unit_shadow_map.is_real(),
                decals,
                false,
            )
        };
        let surface = |fragment| {
            water_pipeline(
                device,
                self.pipeline_layout,
                self.color_format,
                self.depth_format,
                fragment,
            )
        };
        let [draw, underwater, refract, full] = reflect::pipelines(device, |i| match i {
            0 => terrain(None),
            1 => underwater::pipeline(device, self.pipeline_layout, self.color_format),
            2 => surface(spirv!("water_refract.frag")),
            _ => surface(spirv!("water_full.frag")),
        })?;
        let pending = (|| -> Result<_> {
            let decals = self
                .terrain_decals
                .map(|_| reflect::pipelines(device, |i| terrain(Some(DecalReliefMode::ALL[i]))))
                .transpose()?;
            let layers = super::tileset::seabed_layers();
            let size = super::tileset::SEABED_SIZE;
            let image = upload::image_mipped(
                gpu,
                vk::Format::R8G8B8A8_SRGB,
                size,
                size,
                &layers.each_ref().map(Vec::as_slice),
                MipFilter::Average,
                vk::ImageViewType::TYPE_2D_ARRAY,
                "seabed layers",
            );
            let result = image.and_then(|mut image| {
                match bindless.group(|b| b.sampled_2d_array(image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)) {
                    Ok((index, group)) => Ok((image, index.0.index, group)),
                    Err(err) => {
                        image.destroy(gpu);
                        Err(err)
                    }
                }
            });
            match result {
                Ok((seabed, seabed_index, seabed_group)) => Ok(TerrainRefraction {
                    terrain: draw,
                    decals,
                    underwater,
                    refract,
                    full,
                    seabed,
                    seabed_index,
                    seabed_group,
                }),
                Err(err) => {
                    for p in decals.into_iter().flatten() {
                        unsafe { device.destroy_pipeline(p, None) };
                    }
                    Err(err)
                }
            }
        })();
        match pending {
            Ok(variants) => self.refraction = Some(variants),
            Err(err) => {
                for pipeline in [draw, underwater, refract, full] {
                    unsafe { device.destroy_pipeline(pipeline, None) };
                }
                return Err(err);
            }
        }
        Ok(())
    }

    pub(crate) fn frame_index(&self) -> u32 {
        self.frame_index
    }

    pub(crate) fn prepare_wave_surfaces(&mut self, device: &ash::Device) -> Result<()> {
        if self.wave_surfaces.is_none() {
            let fragments = [
                spirv!("water_mesh.frag").as_slice(),
                spirv!("water_mesh_reflect.frag").as_slice(),
                spirv!("water_mesh_refract.frag").as_slice(),
                spirv!("water_mesh_full.frag").as_slice(),
            ];
            self.wave_surfaces = Some(reflect::pipelines(device, |i| {
                water_pipeline_with_vertex(
                    device,
                    self.pipeline_layout,
                    self.color_format,
                    self.depth_format,
                    spirv!("water_mesh.vert"),
                    fragments[i],
                )
            })?);
        }
        Ok(())
    }

    pub(crate) fn prepare_wave_underwater(&mut self, device: &ash::Device) -> Result<()> {
        if self.wave_underwater.is_none() {
            self.wave_underwater = Some(underwater::pipeline_with_shader(
                device,
                self.pipeline_layout,
                self.color_format,
                spirv!("water_underwater_waves.frag"),
            )?);
        }
        Ok(())
    }

    fn volume_index(waves: bool, reflect: bool) -> usize {
        2 * usize::from(waves) + usize::from(reflect)
    }

    pub(crate) fn prepare_volume(&mut self, device: &ash::Device, waves: bool, reflect: bool) -> Result<()> {
        let index = Self::volume_index(waves, reflect);
        if self.volume[index].is_none() {
            let fragment = [
                spirv!("water_rich_refract.frag").as_slice(),
                spirv!("water_rich_full.frag").as_slice(),
                spirv!("water_mesh_rich_refract.frag").as_slice(),
                spirv!("water_mesh_rich_full.frag").as_slice(),
            ][index];
            let vertex = if waves {
                spirv!("water_mesh.vert").as_slice()
            } else {
                spirv!("water.vert").as_slice()
            };
            let [volume, surface] = reflect::pipelines(device, |i| {
                if i == 0 {
                    volume::pipeline(device, self.pipeline_layout, self.color_format, waves)
                } else {
                    water_pipeline_with_vertex(
                        device,
                        self.pipeline_layout,
                        self.color_format,
                        self.depth_format,
                        vertex,
                        fragment,
                    )
                }
            })?;
            self.volume[index] = Some((volume, surface));
        }
        Ok(())
    }

    pub(crate) fn record_volume(
        &self,
        ctx: &PassContext<'_>,
        inputs: [u32; 3],
        waves: Option<(crate::water::waves::WavePatch, u32)>,
        reflect: bool,
        enabled: bool,
    ) -> Result<()> {
        let mut push = self.push;
        push.words[0] = u32::from(enabled);
        for (slot, image) in inputs.into_iter().enumerate() {
            push.image(12 + slot, image);
        }
        if let Some((patch, image)) = waves {
            patch.push(&mut push);
            push.image(15, image);
        }
        let (volume, _) = self.volume[Self::volume_index(waves.is_some(), reflect)].context("water volume pipeline")?;
        self.draw_water_vertices(ctx, volume, &push, 3);
        Ok(())
    }

    pub(crate) fn water_push(&self) -> crate::bindless::Push {
        self.push
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_wave_surface(
        &self,
        ctx: &PassContext<'_>,
        reflection: Option<u32>,
        refraction: Option<u32>,
        surface: crate::water::waves::WaveSurface,
        leaves: BufferBinding,
        indirect: BufferBinding,
        volume: bool,
    ) -> Result<()> {
        let surfaces = self.wave_surfaces.context("water mesh pipelines")?;
        let mut push = self.push;
        surface.patch.push(&mut push);
        push.image(14, surface.waves);
        // The surface samples none of the terrain's error, material,
        // tileset or decal images: their slots carry the water's own data
        // (`water.frag` WATER_MESH), and the push words the swell.
        let [static_map, ripples, foam] = surface.detail;
        push.image(2, surface.shore);
        push.image(3, surface.churn);
        push.image(4, ripples);
        push.image(5, static_map);
        push.image(9, foam);
        push.words = surface.swell;
        push.image(15, leaves.storage.context("water mesh leaf storage")?);
        let index = usize::from(reflection.is_some()) + 2 * usize::from(refraction.is_some());
        if let Some(image) = reflection {
            push.image(12, image);
        }
        if let Some(image) = refraction {
            push.image(13, image);
        }
        let pipeline = if volume {
            self.volume[Self::volume_index(true, reflection.is_some())]
                .context("displaced underside pipeline")?
                .1
        } else {
            surfaces[index]
        };
        self.bind_water(ctx, pipeline, &push);
        unsafe {
            ctx.device
                .cmd_draw_indirect(ctx.cmd, indirect.buffer, indirect.offset + 3 * 4, 1, 0)
        };
        Ok(())
    }

    pub(crate) fn record_underwater(
        &self,
        ctx: &PassContext<'_>,
        scene: u32,
        depth: u32,
        caustics: u32,
        waves: Option<(crate::water::waves::WavePatch, u32)>,
    ) -> Result<()> {
        let refract = self.refraction.as_ref().context("terrain refraction pipelines")?;
        let mut push = self.push;
        push.image(12, scene);
        push.image(13, depth);
        push.image(14, caustics);
        let pipeline = if let Some((patch, image)) = waves {
            patch.push(&mut push);
            push.image(15, image);
            self.wave_underwater.context("displaced refraction pipeline")?
        } else {
            refract.underwater
        };
        self.draw_water_vertices(ctx, pipeline, &push, 3);
        Ok(())
    }

    pub(crate) fn record_water_refract(
        &self,
        ctx: &PassContext<'_>,
        reflection: Option<u32>,
        refraction: u32,
        volume: bool,
    ) -> Result<()> {
        let refract = self.refraction.as_ref().context("terrain refraction pipelines")?;
        let mut push = self.push;
        let pipeline = if let Some(reflection) = reflection {
            push.image(12, reflection);
            refract.full
        } else {
            refract.refract
        };
        let pipeline = if volume {
            self.volume[Self::volume_index(false, reflection.is_some())]
                .context("water underside pipeline")?
                .1
        } else {
            pipeline
        };
        push.image(13, refraction);
        self.draw_water(ctx, pipeline, &push);
        Ok(())
    }

    pub(crate) fn record_cull_reflect(
        &self,
        ctx: &PassContext<'_>,
        leaves: BufferBinding,
        indirect: BufferBinding,
    ) -> Result<()> {
        let mirror = self.reflection.as_ref().context("terrain reflection pipelines")?;
        let (device, cmd) = (ctx.device, ctx.cmd);
        let mut push = self.push;
        push.words[1] = (self.submesh_ib.size / 2) as u32;
        reflect::push_buffers(&mut push, leaves, indirect)?;
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[ctx.global_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                crate::bindless::PUSH_STAGES,
                0,
                bytes_of(&push),
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, mirror.prepare);
            device.cmd_dispatch(cmd, 1, 1, 1);
            reflect::cull_barrier(ctx);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, mirror.cull);
            device.cmd_dispatch_indirect(cmd, indirect.buffer, indirect.offset);
            reflect::cull_barrier(ctx);
            push.words[0] = 1;
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                crate::bindless::PUSH_STAGES,
                0,
                bytes_of(&push),
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, mirror.prepare);
            device.cmd_dispatch(cmd, 1, 1, 1);
        }
        Ok(())
    }

    pub(crate) fn record_draw_reflect(
        &self,
        ctx: &PassContext<'_>,
        leaves: BufferBinding,
        indirect: BufferBinding,
    ) -> Result<()> {
        let mirror = self.reflection.as_ref().context("terrain reflection pipelines")?;
        let (device, cmd) = (ctx.device, ctx.cmd);
        let mut push = self.push;
        reflect::push_buffers(&mut push, leaves, indirect)?;
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[ctx.global_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                crate::bindless::PUSH_STAGES,
                0,
                bytes_of(&push),
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.submesh_vb.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.submesh_ib.buffer, 0, vk::IndexType::UINT16);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, mirror.draw);
            device.cmd_set_front_face(cmd, vk::FrontFace::CLOCKWISE);
            device.cmd_draw_indexed_indirect(cmd, indirect.buffer, indirect.offset + 3 * 4, 1, 0);
        }
        Ok(())
    }

    pub(crate) fn record_water_reflect(&self, ctx: &PassContext<'_>, reflection: u32) -> Result<()> {
        let mirror = self.reflection.as_ref().context("terrain reflection pipelines")?;
        let mut push = self.push;
        push.image(12, reflection);
        self.draw_water(ctx, mirror.surface, &push);
        Ok(())
    }

    /// Whether this renderer can draw water (the map has a water level).
    pub fn has_water(&self) -> bool {
        self.water.is_some()
    }

    /// Draw the water surface. Records inside a graphics pass with a colour
    /// attachment of `color_format` (loaded: it blends over the scene) and
    /// the depth buffer read-only. Nothing on a map without water.
    pub fn record_water(&self, ctx: &PassContext<'_>) {
        if let Some(water) = self.water {
            self.draw_water(ctx, water, &self.push);
        }
    }

    fn draw_water(&self, ctx: &PassContext<'_>, water: vk::Pipeline, push: &crate::bindless::Push) {
        self.draw_water_vertices(ctx, water, push, 6);
    }

    fn draw_water_vertices(
        &self,
        ctx: &PassContext<'_>,
        water: vk::Pipeline,
        push: &crate::bindless::Push,
        vertices: u32,
    ) {
        self.bind_water(ctx, water, push);
        unsafe { ctx.device.cmd_draw(ctx.cmd, vertices, 1, 0, 0) };
    }

    fn bind_water(&self, ctx: &PassContext<'_>, water: vk::Pipeline, push: &crate::bindless::Push) {
        let (device, cmd) = (ctx.device, ctx.cmd);
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[ctx.global_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(push),
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, water);
        }
    }

    /// Global sampled indices the units and particles reuse: the height
    /// texture, the hills' shadow heights, the units' shadow map and the
    /// particles' coloured shadows.
    pub fn height_index(&self) -> u32 {
        self.sampled[1]
    }

    pub fn sun_shadow_index(&self) -> u32 {
        self.sampled[6]
    }

    pub fn unit_shadow_index(&self) -> u32 {
        self.sampled[7]
    }

    pub fn particle_shadow_index(&self) -> u32 {
        self.sampled[8]
    }

    /// The terrain's linear sampler and the shadows' colour sampler.
    pub fn linear_sampler_index(&self) -> u32 {
        self.push.sampler_at(1)
    }

    pub fn shadow_color_sampler_index(&self) -> u32 {
        self.push.sampler_at(3)
    }

    /// The units' shadow map's comparison sampler.
    pub fn shadow_compare_sampler_index(&self) -> u32 {
        self.push.sampler_at(4)
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        let device = &gpu.device;
        unsafe {
            for p in [self.prepare, self.update, self.reduce, self.terrain]
                .into_iter()
                .chain(self.terrain_decals.into_iter().flatten())
                .chain(self.wire)
                .chain(self.water)
                .chain(
                    self.reflection
                        .take()
                        .into_iter()
                        .flat_map(|m| [m.prepare, m.cull, m.draw, m.surface]),
                )
            {
                device.destroy_pipeline(p, None);
            }
        }
        if let Some(mut r) = self.refraction.take() {
            for p in [r.terrain, r.underwater, r.refract, r.full]
                .into_iter()
                .chain(r.decals.into_iter().flatten())
            {
                unsafe { device.destroy_pipeline(p, None) };
            }
            r.seabed.destroy(gpu);
        }
        if let Some(pipeline) = self.wave_underwater.take() {
            unsafe { device.destroy_pipeline(pipeline, None) };
        }
        if let Some(surfaces) = self.wave_surfaces.take() {
            for pipeline in surfaces {
                unsafe { device.destroy_pipeline(pipeline, None) };
            }
        }
        for (volume, surface) in self.volume.iter_mut().filter_map(Option::take) {
            unsafe {
                device.destroy_pipeline(volume, None);
                device.destroy_pipeline(surface, None);
            }
        }
        self.table.destroy(gpu);
        for b in [
            &mut self.tree,
            &mut self.bits,
            &mut self.snap,
            &mut self.leaves,
            &mut self.indirect,
            &mut self.submesh_vb,
            &mut self.submesh_ib,
        ] {
            b.destroy(gpu);
        }
        for b in self
            .ubos
            .iter_mut()
            .chain(self.stats.iter_mut())
            .chain(self.vision_bufs.iter_mut())
            .chain(self.decal_bufs.iter_mut())
        {
            b.destroy(gpu);
        }
        self.height_img.destroy(gpu);
        self.pyramid_img.destroy(gpu);
        self.error_img.destroy(gpu);
        self.material_img.destroy(gpu);
        self.tile_albedo.destroy(gpu);
        self.tile_normal.destroy(gpu);
        self.tile_ubo.destroy(gpu);
        self.tile_noise.destroy(gpu);
        self.shadow_img.destroy(gpu);
        self.unit_shadow_map.destroy(gpu);
        self.decal_atlas.destroy(gpu);
        self.decal_surface.destroy(gpu);
        self.decal_normals.destroy(gpu);
        self.decal_cones.destroy(gpu);
        self.decal_materials.destroy(gpu);
    }
}

/// The six planes of `view_proj` (Vulkan clip space, reverse-Z with an
/// infinite far plane), normalised, inside where `dot(n, p) + d >= 0`.
pub fn frustum_planes(view_proj: Mat4) -> [[f32; 4]; 6] {
    let r = |i: usize| view_proj.row(i);
    let ps: [Vec4; 6] = [
        r(3) + r(0),
        r(3) - r(0),
        r(3) + r(1),
        r(3) - r(1),
        r(3) - r(2),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    ];
    ps.map(|p| (p / p.truncate().length().max(1e-20)).to_array())
}

#[allow(clippy::too_many_arguments)]
fn graphics_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
    depth_format: vk::Format,
    fragment: &[u8],
    wire: bool,
    unit_shadows: bool,
    decals: Option<DecalReliefMode>,
    mirrored: bool,
) -> Result<vk::Pipeline> {
    let vertex: &[u8] = if mirrored {
        spirv!("terrain_mirror.vert")
    } else {
        spirv!("terrain.vert")
    };
    let vert = pipeline::shader_module(device, vertex)?;
    let frag = pipeline::shader_module(device, fragment)?;
    // UNIT_SHADOWS (unit_shadows.glsl): without a shadow map the lookup is
    // compiled out, not just skipped; DECALS (terrain_decals.glsl) likewise.
    // Anti-tiling is a tileset degree in the table, not a compiled variant.
    let spec_words = [
        u32::from(unit_shadows),
        u32::from(decals.is_some()),
        decals.unwrap_or_default() as u32,
    ];
    let spec_value: Vec<u8> = spec_words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let spec_entries: [vk::SpecializationMapEntry; 3] = std::array::from_fn(|i| vk::SpecializationMapEntry {
        constant_id: i as u32,
        offset: 4 * i as u32,
        size: 4,
    });
    let spec = vk::SpecializationInfo::default()
        .map_entries(&spec_entries)
        .data(&spec_value);
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(c"main")
            .specialization_info(&spec),
    ];
    let vb = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: 16,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let va = [vk::VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: vk::Format::R32G32B32A32_SINT,
        offset: 0,
    }];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vb)
        .vertex_attribute_descriptions(&va);
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vp = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    // LEB triangles alternate winding; the heightfield is seen from above.
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(if wire {
            vk::PolygonMode::LINE
        } else {
            vk::PolygonMode::FILL
        })
        .cull_mode(vk::CullModeFlags::NONE)
        .line_width(1.0)
        .depth_bias_enable(wire)
        .depth_bias_constant_factor(if wire { 4.0 } else { 0.0 })
        .depth_bias_slope_factor(if wire { 1.0 } else { 0.0 });
    let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // Reverse-Z.
    let ds = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(!wire)
        .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);
    // The glow target (the bloom's input) is the second attachment; the
    // scene's pipelines are built for it so a graph may declare it or not
    // (the fragment shader always writes location 1).
    let formats = [
        color_format,
        vk::Format::from_raw(ae_render_abi::module::GLOW_FORMAT as i32),
    ];
    let blend = [
        vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA),
        vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA),
    ];
    let attachments = 2 - usize::from(mirrored);
    let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend[..attachments]);
    let dyn_states = [
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::FRONT_FACE,
    ];
    let dynamic =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states[..2 + usize::from(mirrored)]);
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats[..attachments])
        .depth_attachment_format(depth_format);
    if ae_framegraph::has_stencil(depth_format) {
        rendering = rendering.stencil_attachment_format(depth_format);
    }
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&ia)
        .viewport_state(&vp)
        .rasterization_state(&raster)
        .multisample_state(&ms)
        .depth_stencil_state(&ds)
        .color_blend_state(&cb)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    let result = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    Ok(result.map_err(|(_, err)| err).context("terrain pipeline")?[0])
}

fn water_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
    depth_format: vk::Format,
    fragment: &[u8],
) -> Result<vk::Pipeline> {
    water_pipeline_with_vertex(
        device,
        layout,
        color_format,
        depth_format,
        spirv!("water.vert"),
        fragment,
    )
}

fn water_pipeline_with_vertex(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
    depth_format: vk::Format,
    vertex: &[u8],
    fragment: &[u8],
) -> Result<vk::Pipeline> {
    pipeline::graphics_pipeline(
        device,
        &pipeline::GraphicsDesc {
            vertex,
            fragment,
            layout,
            color_formats: &[
                color_format,
                vk::Format::from_raw(ae_render_abi::module::GLOW_FORMAT as i32),
            ],
            depth: Some(pipeline::DepthState {
                format: depth_format,
                write: false,
                compare: vk::CompareOp::GREATER_OR_EQUAL,
            }),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            blend: pipeline::Blend::Premultiplied,
            glow_blend: pipeline::Blend::Off,
        },
    )
    .context("water pipeline")
}
