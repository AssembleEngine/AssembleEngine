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
use crate::decals::DecalAtlas;
use crate::device::GpuDevice;
use crate::executor::{BufferBinding, PassContext};
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::pipeline::{self, spirv};
use crate::shadow::{Cascades, ShadowMap};
use crate::upload;
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

#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    mode: u32,
    write_draw: u32,
    level: u32,
    pad: u32,
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
    /// Shade by the hills' shadows (when the renderer has them).
    pub shadows: bool,
    /// Shade by the units' shadow map with these cascades (the graph
    /// draws and samples it this frame).
    pub unit_shadows: Option<Cascades>,
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
    pub fn new(heightmap: &Heightmap, pyramid: &Pyramid, cell_size: f32, height_scale: f32, height_offset: f32, water_level: Option<f32>) -> Self {
        let (cx, cy) = heightmap.cells();
        Self { cells: (cx as u32, cy as u32), root_cells: pyramid.root_cells as u32, cell_size, height_scale, height_offset, water_level, has_materials: false }
    }

    pub fn extent(&self) -> (f32, f32) {
        (self.cells.0 as f32 * self.cell_size, self.cells.1 as f32 * self.cell_size)
    }
}

pub struct TerrainRenderer {
    pub max_depth: u32,
    info: TerrainInfo,
    color_format: vk::Format,
    depth_format: vk::Format,

    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    desc_pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    /// The clustered lights' sets (set 1 of the draws; crate::lights).
    light_sets: Vec<vk::DescriptorSet>,
    prepare: vk::Pipeline,
    update: vk::Pipeline,
    reduce: vk::Pipeline,
    terrain: vk::Pipeline,
    /// `None` without `fillModeNonSolid`.
    wire: Option<vk::Pipeline>,
    /// The water surface; `None` on a map without water.
    water: Option<vk::Pipeline>,

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
    terrain_decals: Option<vk::Pipeline>,
    /// Per slot: the decals ([`crate::decals`]; host-visible, 16 bytes
    /// without), and whether it has any.
    decal_bufs: Vec<OwnedBuffer>,
    decal_live: Vec<bool>,
    /// The decal atlas (1x1 without decals).
    decal_atlas: OwnedImage,
    /// Its normals and glow mask, layer for layer.
    decal_normals: OwnedImage,
    /// Per slot: line-of-sight and explored bit planes (host-visible;
    /// 16 bytes without a vision grid).
    vision_bufs: Vec<OwnedBuffer>,
    vision: Option<VisionGrid>,
    /// Nearest clamp, linear clamp, trilinear repeat (tilesets).
    samplers: [vk::Sampler; 3],
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
        [&self.heights, &self.pyramid, &self.errors, &self.materials, &self.shadows].iter().flat_map(|r| r.iter()).map(|r| r.bytes.len()).sum()
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &GpuDevice,
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
            anyhow::ensure!(materials.len() == (info.cells.0 * info.cells.1) as usize, "one material per cell");
        }
        let k = pyramid.root_cells.trailing_zeros();
        // Splitting stops by itself once a leaf's sub-mesh has one vertex
        // per sample (around depth 2k-2). One spare level, and T = D-5 >= 8
        // for the shared-memory reduce.
        let max_depth = (2 * k - 1).clamp(13, 30);
        let cbt = Cbt::new(max_depth);

        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let tree = upload::buffer(gpu, slice_bytes(&cbt.tree), storage, "terrain tree")?;
        let bits = upload::buffer(gpu, slice_bytes(&cbt.bits), storage | vk::BufferUsageFlags::TRANSFER_SRC, "terrain bits")?;
        let snap = OwnedBuffer::new(gpu, BufferDesc { size: bits.size, usage: storage | vk::BufferUsageFlags::TRANSFER_DST }, MemoryLocation::GpuOnly, "terrain snapshot")?;
        let capacity = MAX_DRAW_LEAVES.min(1 << max_depth);
        let leaves = OwnedBuffer::new(gpu, BufferDesc { size: capacity as u64 * LEAF_BYTES, usage: storage }, MemoryLocation::GpuOnly, "terrain leaves")?;

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
        let submesh_vb = upload::buffer(gpu, slice_bytes(&vertices), vk::BufferUsageFlags::VERTEX_BUFFER, "terrain sub-mesh vertices")?;
        let submesh_ib = upload::buffer(gpu, slice_bytes(&indices), vk::BufferUsageFlags::INDEX_BUFFER, "terrain sub-mesh indices")?;

        let ubos = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: size_of::<FrameUbo>() as u64, usage: vk::BufferUsageFlags::UNIFORM_BUFFER }, MemoryLocation::CpuToGpu, "terrain frame"))
            .collect::<Result<Vec<_>>>()?;
        let vision_bytes = vision.map_or(16, |v| 2 * v.words() as u64 * 8);
        let vision_bufs = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: vision_bytes, usage: vk::BufferUsageFlags::STORAGE_BUFFER }, MemoryLocation::CpuToGpu, "terrain vision"))
            .collect::<Result<Vec<_>>>()?;
        let decal_bytes = decals.map_or(16, |(_, capacity)| crate::decals::buffer_words(capacity) as u64 * 4);
        let decal_bufs = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: decal_bytes, usage: vk::BufferUsageFlags::STORAGE_BUFFER }, MemoryLocation::CpuToGpu, "terrain decals"))
            .collect::<Result<Vec<_>>>()?;
        // The decal atlas: colour (sRGB) and, layer for layer, the
        // tangent-space normal with the glow mask (UNORM).
        let (decal_atlas, decal_normals) = match decals {
            Some((atlas, _)) => {
                atlas.validate().map_err(anyhow::Error::msg)?;
                let upload = |mips: Vec<Vec<Vec<u8>>>, format, name| {
                    let layers: Vec<Vec<&[u8]>> = mips.iter().map(|l| l.iter().map(Vec::as_slice).collect()).collect();
                    upload::image_layers(gpu, format, atlas.size, atlas.size, &layers, vk::ImageViewType::TYPE_2D_ARRAY, name)
                };
                (upload(atlas.mips(), vk::Format::R8G8B8A8_SRGB, "decal atlas")?, upload(atlas.normal_mips(), vk::Format::R8G8B8A8_UNORM, "decal normals")?)
            }
            None => {
                let one = |format, texel: [u8; 4], name| upload::image_layers(gpu, format, 1, 1, &[vec![&texel[..]]], vk::ImageViewType::TYPE_2D_ARRAY, name);
                (one(vk::Format::R8G8B8A8_SRGB, [0; 4], "decal atlas")?, one(vk::Format::R8G8B8A8_UNORM, [128, 128, 255, 0], "decal normals")?)
            }
        };
        let stats = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: IND_WORDS * 4, usage: vk::BufferUsageFlags::TRANSFER_DST }, MemoryLocation::GpuToCpu, "terrain stats"))
            .collect::<Result<Vec<_>>>()?;

        let height_img = upload::image(gpu, vk::Format::R32_SFLOAT, heightmap.width as u32, heightmap.height as u32, &[slice_bytes(&heightmap.data)], "terrain heights")?;
        let mips: Vec<&[u8]> = pyramid.levels.iter().map(|l| slice_bytes(l.as_slice())).collect();
        let base = (pyramid.root_cells >> PYRAMID_BASE_LEVEL) as u32;
        let pyramid_img = upload::image(gpu, vk::Format::R32G32B32A32_SFLOAT, base, base, &mips, "terrain pyramid")?;
        let error_mips: Vec<&[u8]> = pyramid.errors.iter().map(|l| slice_bytes(l.as_slice())).collect();
        let half = (pyramid.root_cells >> 1) as u32;
        let error_img = upload::image(gpu, vk::Format::R32_SFLOAT, half, half, &error_mips, "terrain errors")?;

        let material_img = match materials {
            Some(m) => upload::image(gpu, vk::Format::R8_UINT, info.cells.0, info.cells.1, &[m], "terrain materials")?,
            None => upload::image(gpu, vk::Format::R8_UINT, 1, 1, &[&[0]], "terrain materials")?,
        };
        let shadow_img = match shadows {
            Some(s) => {
                anyhow::ensure!(s.len() == heightmap.data.len(), "one shadow height per corner");
                upload::image(gpu, vk::Format::R32_SFLOAT, heightmap.width as u32, heightmap.height as u32, &[slice_bytes(s)], "terrain shadow heights")?
            }
            None => upload::image(gpu, vk::Format::R32_SFLOAT, 1, 1, &[&[0; 4]], "terrain shadow heights")?,
        };
        let unit_shadow_map = ShadowMap::new(gpu, unit_shadows)?;
        let array = vk::ImageViewType::TYPE_2D_ARRAY;
        let (tile_albedo, tile_normal, tile_table) = match tileset {
            Some(t) => {
                t.validate().map_err(anyhow::Error::msg)?;
                let albedo: Vec<Vec<&[u8]>> = t.layers.iter().map(|l| l.albedo.iter().map(Vec::as_slice).collect()).collect();
                let normal: Vec<Vec<&[u8]>> = t.layers.iter().map(|l| l.normal.iter().map(Vec::as_slice).collect()).collect();
                (
                    upload::image_layers(gpu, vk::Format::R8G8B8A8_SRGB, t.size, t.size, &albedo, array, "tileset albedo")?,
                    upload::image_layers(gpu, vk::Format::R8G8_UNORM, t.size, t.size, &normal, array, "tileset normals")?,
                    t.ubo(),
                )
            }
            None => (
                upload::image_layers(gpu, vk::Format::R8G8B8A8_SRGB, 1, 1, &[vec![&[0u8; 4][..]]], array, "tileset albedo")?,
                upload::image_layers(gpu, vk::Format::R8G8_UNORM, 1, 1, &[vec![&[128u8, 128][..]]], array, "tileset normals")?,
                TilesetUbo::default(),
            ),
        };
        let tile_noise = match tileset {
            Some(t) if t.anti_tiled() => {
                let mips = super::tileset::anti_tiling_noise();
                let mips: Vec<&[u8]> = mips.iter().map(Vec::as_slice).collect();
                upload::image(gpu, vk::Format::R8_UNORM, super::tileset::NOISE_SIZE, super::tileset::NOISE_SIZE, &mips, "anti-tiling noise")?
            }
            _ => upload::image(gpu, vk::Format::R8_UNORM, 1, 1, &[&[128u8][..]], "anti-tiling noise")?,
        };
        let tile_ubo = upload::buffer(gpu, &bytes_of(&tile_table)[..TILESET_UBO_BYTES], vk::BufferUsageFlags::UNIFORM_BUFFER, "tileset table")?;

        let make_sampler = |filter: vk::Filter| unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(filter)
                    .min_filter(filter)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        };
        let repeat = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        }?;
        let samplers = [make_sampler(vk::Filter::NEAREST)?, make_sampler(vk::Filter::LINEAR)?, repeat];

        let all = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let (set_layout, pipeline_layout, desc_pool, sets) = unsafe {
            let b = |i: u32, ty: vk::DescriptorType| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(1).stage_flags(all);
            let bindings = [
                b(0, vk::DescriptorType::UNIFORM_BUFFER),
                b(1, vk::DescriptorType::STORAGE_BUFFER),
                b(2, vk::DescriptorType::STORAGE_BUFFER),
                b(3, vk::DescriptorType::STORAGE_BUFFER),
                b(4, vk::DescriptorType::STORAGE_BUFFER),
                b(5, vk::DescriptorType::STORAGE_BUFFER),
                b(6, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                b(7, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                b(8, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                b(9, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                b(10, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                b(11, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                b(12, vk::DescriptorType::UNIFORM_BUFFER),
                b(13, vk::DescriptorType::STORAGE_BUFFER),
                b(14, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                b(15, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                b(16, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                b(17, vk::DescriptorType::STORAGE_BUFFER),
                b(18, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                b(19, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                b(20, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            ];
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            // Set 1: the clustered lights (the terrain and the water read them).
            let pipeline_layout = pipeline::pipeline_layout(device, &[set_layout, lights.set_layout()], size_of::<PushConstants>() as u32, all)?;
            let n = frames_in_flight as u32;
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 2 * n },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 7 * n },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 12 * n },
            ];
            let desc_pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes), None)?;
            let layouts = vec![set_layout; frames_in_flight];
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&layouts))?;
            for (slot, &set) in sets.iter().enumerate() {
                let info = |buffer: &OwnedBuffer| [vk::DescriptorBufferInfo { buffer: buffer.buffer, offset: 0, range: vk::WHOLE_SIZE }];
                let infos = [info(&ubos[slot]), info(&tree), info(&bits), info(&snap), info(&leaves), info(&indirect)];
                let image = |sampler, view| [vk::DescriptorImageInfo { sampler, image_view: view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
                let images = [
                    image(samplers[0], pyramid_img.view),
                    image(samplers[1], height_img.view),
                    image(samplers[0], error_img.view),
                    image(samplers[0], material_img.view),
                    image(samplers[2], tile_albedo.view),
                    image(samplers[2], tile_normal.view),
                ];
                let shadow = image(samplers[1], shadow_img.view);
                let mut writes: Vec<vk::WriteDescriptorSet> = infos
                    .iter()
                    .enumerate()
                    .map(|(i, info)| {
                        let ty = if i == 0 { vk::DescriptorType::UNIFORM_BUFFER } else { vk::DescriptorType::STORAGE_BUFFER };
                        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(ty).buffer_info(info)
                    })
                    .collect();
                for (i, info) in images.iter().enumerate() {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(6 + i as u32)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(info),
                    );
                }
                let table = [vk::DescriptorBufferInfo { buffer: tile_ubo.buffer, offset: 0, range: vk::WHOLE_SIZE }];
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(12).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&table));
                let planes = [vk::DescriptorBufferInfo { buffer: vision_bufs[slot].buffer, offset: 0, range: vk::WHOLE_SIZE }];
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(13).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&planes));
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(14).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&shadow));
                let (view, sampler) = unit_shadow_map.texture();
                let units = image(sampler, view);
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(15).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&units));
                let (view, sampler) = unit_shadow_map.color_texture();
                let tint = image(sampler, view);
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(16).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&tint));
                let decal_list = [vk::DescriptorBufferInfo { buffer: decal_bufs[slot].buffer, offset: 0, range: vk::WHOLE_SIZE }];
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(17).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&decal_list));
                let atlas = image(samplers[2], decal_atlas.view);
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(18).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&atlas));
                let normals = image(samplers[2], decal_normals.view);
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(19).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&normals));
                let noise = image(samplers[2], tile_noise.view);
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(20).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&noise));
                device.update_descriptor_sets(&writes, &[]);
            }
            (set_layout, pipeline_layout, desc_pool, sets)
        };

        let prepare = pipeline::compute_pipeline(device, spirv!("cbt_prepare.comp"), pipeline_layout)?;
        let update = pipeline::compute_pipeline(device, spirv!("cbt_update.comp"), pipeline_layout)?;
        let reduce = pipeline::compute_pipeline(device, spirv!("cbt_reduce.comp"), pipeline_layout)?;
        let anti_tiling = tileset.is_some_and(Tileset::anti_tiled);
        let terrain = graphics_pipeline(device, pipeline_layout, color_format, depth_format, spirv!("terrain.frag"), false, unit_shadows, false, anti_tiling)?;
        let terrain_decals = match decals {
            Some(_) => Some(graphics_pipeline(device, pipeline_layout, color_format, depth_format, spirv!("terrain.frag"), false, unit_shadows, true, anti_tiling)?),
            None => None,
        };
        let wire = if gpu.fill_mode_non_solid {
            Some(graphics_pipeline(device, pipeline_layout, color_format, depth_format, spirv!("terrain_wire.frag"), true, false, false, false)?)
        } else {
            None
        };
        let water = match info.water_level {
            Some(_) => Some(
                pipeline::graphics_pipeline(
                    device,
                    &pipeline::GraphicsDesc {
                        vertex: spirv!("water.vert"),
                        fragment: spirv!("water.frag"),
                        layout: pipeline_layout,
                        color_formats: &[color_format],
                        // Reverse-Z; tested against the terrain and units, never written.
                        depth: Some(pipeline::DepthState { format: depth_format, write: false, compare: vk::CompareOp::GREATER_OR_EQUAL }),
                        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                        blend: pipeline::Blend::Premultiplied,
                    },
                )
                .context("water pipeline")?,
            ),
            None => None,
        };

        Ok(Self {
            max_depth,
            info,
            color_format,
            depth_format,
            set_layout,
            pipeline_layout,
            desc_pool,
            sets,
            light_sets: lights.sets(),
            prepare,
            update,
            reduce,
            terrain,
            wire,
            water,
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
            decal_live: vec![false; frames_in_flight],
            decal_bufs,
            decal_atlas,
            decal_normals,
            vision_bufs,
            vision,
            samplers,
        })
    }

    /// Rewrite the textures `update` covers. The device must be idle (no
    /// frame in flight samples them); waits for the copy.
    pub fn apply(&mut self, gpu: &GpuDevice, update: &TerrainUpdate) -> Result<()> {
        let materials: &[upload::Region] = if self.info.has_materials { &update.materials } else { &[] };
        let shadows: &[upload::Region] = if self.has_shadows { &update.shadows } else { &[] };
        upload::update_images(
            gpu,
            &[(&self.height_img, &update.heights), (&self.pyramid_img, &update.pyramid), (&self.error_img, &update.errors), (&self.material_img, materials), (&self.shadow_img, shadows)],
        )
    }

    /// The height texture (one corner per texel, normalised: world height =
    /// sample × `height_scale` + `height_offset`) with its bilinear sampler,
    /// for passes that place things on the ground. Lives as long as the
    /// renderer.
    pub fn height_texture(&self) -> (vk::ImageView, vk::Sampler) {
        (self.height_img.view, self.samplers[1])
    }

    /// The shadow height texture ([`super::sun`]; world heights, one corner
    /// per texel like [`Self::height_texture`]) with its bilinear sampler;
    /// 1x1 when the renderer has no shadows. Lives as long as the renderer.
    pub fn shadow_texture(&self) -> (vk::ImageView, vk::Sampler) {
        (self.shadow_img.view, self.samplers[1])
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
        BufferBinding { buffer: self.leaves.buffer, offset: 0, size: self.leaves.size, identity: IDENTITY_LEAVES }
    }

    /// The indirect arguments, for the graph.
    pub fn indirect(&self) -> BufferBinding {
        BufferBinding { buffer: self.indirect.buffer, offset: 0, size: self.indirect.size, identity: IDENTITY_INDIRECT }
    }

    pub fn leaves_desc(&self) -> BufferDesc {
        BufferDesc { size: self.leaves.size, usage: vk::BufferUsageFlags::STORAGE_BUFFER }
    }

    pub fn indirect_desc(&self) -> BufferDesc {
        BufferDesc { size: self.indirect.size, usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC }
    }

    /// Counts written by the frame that last used `slot` (call after its
    /// fence, i.e. in `prepare`).
    pub fn read_stats(&mut self, slot: usize) -> FrameStats {
        let Some(bytes) = self.stats[slot].mapped() else { return FrameStats::default() };
        let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        FrameStats { leaves: word(8), drawn: word(4) }
    }

    /// Per-frame uniforms for `slot` (its previous submission has completed).
    pub fn write_frame(&mut self, slot: usize, p: &FrameParams) {
        let planes = frustum_planes;
        let i = &self.info;
        let unit_shadow = Cascades::uniforms(p.unit_shadows.as_ref().filter(|_| self.unit_shadow_map.is_real()));
        let ubo = FrameUbo {
            view_proj: p.view_proj.to_cols_array(),
            cam_pos: p.cam_pos.extend(1.0).to_array(),
            lod_cam_pos: p.lod_view.pos.extend(1.0).to_array(),
            lod_frustum: planes(p.lod_view.view_proj),
            draw_frustum: planes(p.view_proj),
            lod: [p.tau, p.lod_view.proj_scale, 4.0, 0.7],
            map: [i.cells.0 as f32, i.cells.1 as f32, i.root_cells as f32, i.cell_size],
            height: [i.height_scale, i.height_offset, 1.0 / (i.cells.0 + 1) as f32, 1.0 / (i.cells.1 + 1) as f32],
            sun: super::sun::direction().extend(0.0).to_array(),
            cfg: [
                self.max_depth,
                self.max_depth - 5,
                p.lod_colors as u32
                    | if p.wireframe == Wireframe::Triangles { 2 } else { 0 }
                    | if i.has_materials { 8 } else { 0 }
                    | if self.has_tileset && p.tileset { 16 } else { 0 }
                    | if self.has_tileset && p.tileset && p.weights { 32 } else { 0 }
                    | if self.vision.is_some() && p.vision { 64 } else { 0 }
                    | if self.water.is_some() && p.water { 128 } else { 0 }
                    | if self.has_shadows && p.shadows { 256 } else { 0 },
                MAX_DRAW_LEAVES.min(1 << self.max_depth),
            ],
            fog: [0.62, 0.70, 0.80, if p.fog { 1.0 / (i.cells.0.max(i.cells.1) as f32 * i.cell_size * 1.2) } else { 0.0 }],
            water: match i.water_level {
                Some(level) => [level, 1.0, 0.08, p.time],
                None => [0.0, 0.0, 0.0, p.time],
            },
            vision: self.vision.map_or([0.0; 4], |v| [v.width as f32, v.height as f32, v.cell_size, (v.stride() * 2) as f32]),
            unit_shadow,
            effects: [p.effect_time, 0.0, 0.0, 0.0],
        };
        if let Some(bytes) = self.ubos[slot].mapped() {
            bytes[..size_of::<FrameUbo>()].copy_from_slice(bytes_of(&ubo));
        }
    }

    /// The vision grid this renderer was made for.
    pub fn vision_grid(&self) -> Option<VisionGrid> {
        self.vision
    }

    /// Write `slot`'s vision planes (its previous submission has completed):
    /// line of sight and explored, [`VisionGrid::words`] each.
    pub fn write_vision(&mut self, slot: usize, los: &[u64], explored: &[u64]) -> Result<()> {
        let v = self.vision.context("this terrain renderer has no vision grid")?;
        anyhow::ensure!(los.len() == v.words() && explored.len() == v.words(), "vision planes of {} and {} words for a {}x{} grid", los.len(), explored.len(), v.width, v.height);
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
        anyhow::ensure!(bytes.len() >= words.len() * 4, "{} decal words for a buffer of {} bytes", words.len(), bytes.len());
        bytes[..words.len() * 4].copy_from_slice(slice_bytes(words));
        Ok(())
    }

    /// Whether `slot` draws with decals (made with them, written, some in
    /// view); set every frame.
    pub fn set_decals_live(&mut self, slot: usize, live: bool) {
        self.decal_live[slot] = live && self.terrain_decals.is_some();
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
            device.cmd_push_constants(cmd, self.pipeline_layout, all, 0, bytes_of(&PushConstants { mode, write_draw, level, pad: 0 }));
        };
        let t = self.max_depth - 5;
        unsafe {
            // The previous frame's draw (and the snapshot copy) read the
            // buffers about to be rewritten.
            full();
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.sets[ctx.slot]], &[]);
            let pass = |mode: u32, write_draw: bool| {
                // Decode against a snapshot so in-flight splits and merges
                // cannot disturb it.
                device.cmd_copy_buffer(cmd, self.bits.buffer, self.snap.buffer, &[vk::BufferCopy::default().size(self.bits.size)]);
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
            device.cmd_copy_buffer(cmd, self.indirect.buffer, self.stats[ctx.slot].buffer, &[vk::BufferCopy::default().size(IND_WORDS * 4)]);
            // The graph sees the last write as a compute write: make the
            // copy's read finish before the draw pass too.
            full();
        }
    }

    /// Draw the leaves. Records inside a graphics pass with a colour
    /// attachment of `color_format` and a depth attachment of
    /// `depth_format` (cleared to 0: reverse-Z).
    pub fn record_draw(&self, ctx: &PassContext<'_>, wireframe: Wireframe) {
        let (device, cmd) = (ctx.device, ctx.cmd);
        unsafe {
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.sets[ctx.slot], self.light_sets[ctx.slot]], &[]);
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.submesh_vb.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.submesh_ib.buffer, 0, vk::IndexType::UINT16);
            let terrain = match self.terrain_decals {
                Some(decals) if self.decal_live[ctx.slot] => decals,
                _ => self.terrain,
            };
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, terrain);
            device.cmd_draw_indexed_indirect(cmd, self.indirect.buffer, 3 * 4, 1, 0);
            if let (Wireframe::Triangles, Some(wire)) = (wireframe, self.wire) {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, wire);
                device.cmd_draw_indexed_indirect(cmd, self.indirect.buffer, 3 * 4, 1, 0);
            }
        }
    }

    /// Whether this renderer can draw water (the map has a water level).
    pub fn has_water(&self) -> bool {
        self.water.is_some()
    }

    /// Draw the water surface. Records inside a graphics pass with a colour
    /// attachment of `color_format` (loaded: it blends over the scene) and
    /// the depth buffer read-only. Nothing on a map without water.
    pub fn record_water(&self, ctx: &PassContext<'_>) {
        let Some(water) = self.water else { return };
        let (device, cmd) = (ctx.device, ctx.cmd);
        unsafe {
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.sets[ctx.slot], self.light_sets[ctx.slot]], &[]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, water);
            device.cmd_draw(cmd, 6, 1, 0, 0);
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        let device = &gpu.device;
        unsafe {
            for p in [self.prepare, self.update, self.reduce, self.terrain].into_iter().chain(self.terrain_decals).chain(self.wire).chain(self.water) {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.desc_pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            for s in self.samplers {
                device.destroy_sampler(s, None);
            }
        }
        for b in [&mut self.tree, &mut self.bits, &mut self.snap, &mut self.leaves, &mut self.indirect, &mut self.submesh_vb, &mut self.submesh_ib] {
            b.destroy(gpu);
        }
        for b in self.ubos.iter_mut().chain(self.stats.iter_mut()).chain(self.vision_bufs.iter_mut()).chain(self.decal_bufs.iter_mut()) {
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
        self.decal_normals.destroy(gpu);
    }
}

/// The six planes of `view_proj` (Vulkan clip space, reverse-Z with an
/// infinite far plane), normalised, inside where `dot(n, p) + d >= 0`.
pub fn frustum_planes(view_proj: Mat4) -> [[f32; 4]; 6] {
    let r = |i: usize| view_proj.row(i);
    let ps: [Vec4; 6] = [r(3) + r(0), r(3) - r(0), r(3) + r(1), r(3) - r(1), r(3) - r(2), Vec4::new(0.0, 0.0, 0.0, 1.0)];
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
    decals: bool,
    anti_tiling: bool,
) -> Result<vk::Pipeline> {
    let vert = pipeline::shader_module(device, spirv!("terrain.vert"))?;
    let frag = pipeline::shader_module(device, fragment)?;
    // UNIT_SHADOWS (unit_shadows.glsl): without a shadow map the lookup is
    // compiled out, not just skipped; DECALS (terrain_decals.glsl) and
    // ANTI_TILING (terrain.frag, when a tileset layer uses it) likewise.
    let spec_words = [u32::from(unit_shadows), u32::from(decals), u32::from(anti_tiling)];
    let spec_value: Vec<u8> = spec_words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let spec_entries: [vk::SpecializationMapEntry; 3] = std::array::from_fn(|i| vk::SpecializationMapEntry { constant_id: i as u32, offset: 4 * i as u32, size: 4 });
    let spec = vk::SpecializationInfo::default().map_entries(&spec_entries).data(&spec_value);
    let stages = [
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert).name(c"main"),
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(c"main").specialization_info(&spec),
    ];
    let vb = [vk::VertexInputBindingDescription { binding: 0, stride: 16, input_rate: vk::VertexInputRate::VERTEX }];
    let va = [vk::VertexInputAttributeDescription { location: 0, binding: 0, format: vk::Format::R32G32B32A32_SINT, offset: 0 }];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&vb).vertex_attribute_descriptions(&va);
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    // LEB triangles alternate winding; the heightfield is seen from above.
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(if wire { vk::PolygonMode::LINE } else { vk::PolygonMode::FILL })
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
    let blend = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
    let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
    let formats = [color_format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats).depth_attachment_format(depth_format);
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
