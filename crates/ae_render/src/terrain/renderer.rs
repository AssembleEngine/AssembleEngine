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
//! The tree state lives in buffers the renderer owns, shared by every frame
//! slot (the subdivision evolves frame to frame); the per-frame uniforms are
//! one buffer per slot. Static data (sub-mesh, height texture, error
//! pyramid) is uploaded once.

use super::heightmap::{Heightmap, PYRAMID_BASE_LEVEL, Pyramid};
use super::leb::{self, Cbt};
use crate::device::GpuDevice;
use crate::executor::{BufferBinding, PassContext};
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::pipeline::{self, spirv};
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
    prepare: vk::Pipeline,
    update: vk::Pipeline,
    reduce: vk::Pipeline,
    terrain: vk::Pipeline,
    /// `None` without `fillModeNonSolid`.
    wire: Option<vk::Pipeline>,

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
    samplers: [vk::Sampler; 2],
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
        let samplers = [make_sampler(vk::Filter::NEAREST)?, make_sampler(vk::Filter::LINEAR)?];

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
            ];
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let pipeline_layout = pipeline::pipeline_layout(device, &[set_layout], size_of::<PushConstants>() as u32, all)?;
            let n = frames_in_flight as u32;
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: n },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 5 * n },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 4 * n },
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
                ];
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
                device.update_descriptor_sets(&writes, &[]);
            }
            (set_layout, pipeline_layout, desc_pool, sets)
        };

        let prepare = pipeline::compute_pipeline(device, spirv!("cbt_prepare.comp"), pipeline_layout)?;
        let update = pipeline::compute_pipeline(device, spirv!("cbt_update.comp"), pipeline_layout)?;
        let reduce = pipeline::compute_pipeline(device, spirv!("cbt_reduce.comp"), pipeline_layout)?;
        let terrain = graphics_pipeline(device, pipeline_layout, color_format, depth_format, spirv!("terrain.frag"), false)?;
        let wire = if gpu.fill_mode_non_solid {
            Some(graphics_pipeline(device, pipeline_layout, color_format, depth_format, spirv!("terrain_wire.frag"), true)?)
        } else {
            None
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
            prepare,
            update,
            reduce,
            terrain,
            wire,
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
            samplers,
        })
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
        let planes = |m: Mat4| -> [[f32; 4]; 6] {
            let r = |i: usize| m.row(i);
            // Vulkan clip space, reverse-Z with an infinite far plane.
            let ps: [Vec4; 6] = [r(3) + r(0), r(3) - r(0), r(3) + r(1), r(3) - r(1), r(3) - r(2), Vec4::new(0.0, 0.0, 0.0, 1.0)];
            ps.map(|p| (p / p.truncate().length().max(1e-20)).to_array())
        };
        let i = &self.info;
        let ubo = FrameUbo {
            view_proj: p.view_proj.to_cols_array(),
            cam_pos: p.cam_pos.extend(1.0).to_array(),
            lod_cam_pos: p.lod_view.pos.extend(1.0).to_array(),
            lod_frustum: planes(p.lod_view.view_proj),
            draw_frustum: planes(p.view_proj),
            lod: [p.tau, p.lod_view.proj_scale, 4.0, 0.7],
            map: [i.cells.0 as f32, i.cells.1 as f32, i.root_cells as f32, i.cell_size],
            height: [i.height_scale, i.height_offset, 1.0 / (i.cells.0 + 1) as f32, 1.0 / (i.cells.1 + 1) as f32],
            sun: Vec3::new(-0.5, 0.55, 0.35).normalize().extend(0.0).to_array(),
            cfg: [
                self.max_depth,
                self.max_depth - 5,
                p.lod_colors as u32 | if p.wireframe == Wireframe::Triangles { 2 } else { 0 } | if i.has_materials { 8 } else { 0 },
                MAX_DRAW_LEAVES.min(1 << self.max_depth),
            ],
            fog: [0.62, 0.70, 0.80, if p.fog { 1.0 / (i.cells.0.max(i.cells.1) as f32 * i.cell_size * 1.2) } else { 0.0 }],
            water: match i.water_level {
                Some(level) => [level, 1.0, 0.08, 0.0],
                None => [0.0, 0.0, 0.0, 0.0],
            },
        };
        if let Some(bytes) = self.ubos[slot].mapped() {
            bytes[..size_of::<FrameUbo>()].copy_from_slice(bytes_of(&ubo));
        }
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
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.sets[ctx.slot]], &[]);
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.submesh_vb.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.submesh_ib.buffer, 0, vk::IndexType::UINT16);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.terrain);
            device.cmd_draw_indexed_indirect(cmd, self.indirect.buffer, 3 * 4, 1, 0);
            if let (Wireframe::Triangles, Some(wire)) = (wireframe, self.wire) {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, wire);
                device.cmd_draw_indexed_indirect(cmd, self.indirect.buffer, 3 * 4, 1, 0);
            }
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        let device = &gpu.device;
        unsafe {
            for p in [self.prepare, self.update, self.reduce, self.terrain].into_iter().chain(self.wire) {
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
        for b in self.ubos.iter_mut().chain(self.stats.iter_mut()) {
            b.destroy(gpu);
        }
        self.height_img.destroy(gpu);
        self.pyramid_img.destroy(gpu);
        self.error_img.destroy(gpu);
        self.material_img.destroy(gpu);
    }
}

fn graphics_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
    depth_format: vk::Format,
    fragment: &[u8],
    wire: bool,
) -> Result<vk::Pipeline> {
    let vert = pipeline::shader_module(device, spirv!("terrain.vert"))?;
    let frag = pipeline::shader_module(device, fragment)?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert).name(c"main"),
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(c"main"),
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
