//! Units as GPU instances: an instance stream (one row per unit, a fixed set
//! of 4-byte attributes) culled on the GPU and drawn with one indirect
//! instanced draw on top of the terrain.
//!
//! * [`InstanceStream`] is the render side's whole view of the units: the
//!   attributes in [`ATTRIBUTES`], column by column. Whoever feeds it maps
//!   its own fields onto them (today the client, from an end-of-tick
//!   capture of the sim's `Body` and `NavAgent` columns; from M4 the copy
//!   layer's view tables), so the renderer never sees the sim.
//! * [`UnitFeed`] holds the latest stream. The producer swaps in a new one
//!   after each tick; the render thread takes whatever is latest at the
//!   start of a frame, so neither waits for the other.
//! * Positions are interpolated between the previous and the latest tick
//!   (`prev_x/prev_y` → `x/y`) by the time since the stream was published.
//!
//! In the frame graph, five passes: `units reset` (transfer: the indirect
//! instance count to 0), `units cull` (compute: frustum test per unit,
//! appends visible indices and counts them), `units` (graphics: draws
//! the visible list over the terrain, loading its colour target and depth
//! buffer, reverse-Z), and `units count` + `units count read` (copy the
//! indirect arguments to a host buffer per slot, read back in `prepare`
//! as the visible count). The instance data is one host-visible buffer per
//! frame slot, written in `prepare` only when a newer tick arrived.

use crate::device::GpuDevice;
use crate::executor::{BufferBinding, PassContext};
use crate::memory::OwnedBuffer;
use crate::pipeline::{self, Blend, DepthState, GraphicsDesc, spirv};
use crate::terrain::renderer::{TerrainInfo, TerrainRenderer, frustum_planes};
use crate::upload;
use ae_framegraph::BufferDesc;
use anyhow::{Context as _, Result, ensure};
use ash::vk;
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The attributes of an instance stream, in column order. Positions are on
/// the ground plane in world units (the nav world's x and y; render world
/// x and z); `heading` need not be normalised (zero: facing +x); `class`
/// is a `u32` indexing [`UnitsSetup::classes`], every other attribute an
/// `f32`.
pub const ATTRIBUTES: [&str; 8] = ["x", "y", "prev_x", "prev_y", "heading_x", "heading_y", "radius", "class"];
pub const ATTRIBUTE_COUNT: usize = ATTRIBUTES.len();
/// Classes with their own colour; higher class ids use the last one.
pub const MAX_CLASSES: usize = 16;
/// Vertices of one unit (a four-faced dart).
const UNIT_VERTICES: u32 = 12;

const IDENTITY_INSTANCES: u64 = (0xAE7 << 40) | (2 << 32);
const IDENTITY_VISIBLE: u64 = (0xAE7 << 40) | (2 << 32) | 0x100;
const IDENTITY_INDIRECT: u64 = (0xAE7 << 40) | (2 << 32) | 0x101;
const IDENTITY_COUNTS: u64 = (0xAE7 << 40) | (2 << 32) | 0x200;

/// One snapshot of the units.
#[derive(Clone, Debug)]
pub struct InstanceStream {
    /// The sim tick it shows.
    pub tick: u64,
    /// When it was published: interpolation runs from `prev` to the latest
    /// position over the `tick_seconds` after this.
    pub published: Instant,
    /// Length of a tick; 0 shows the latest positions only.
    pub tick_seconds: f32,
    /// `columns[attribute][row]`, raw 4-byte words, every column the same
    /// length.
    pub columns: [Vec<u32>; ATTRIBUTE_COUNT],
}

impl InstanceStream {
    pub fn len(&self) -> usize {
        self.columns[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns[0].is_empty()
    }

    /// Interpolation weight of the latest position at `now`.
    pub fn alpha(&self, now: Instant) -> f32 {
        if self.tick_seconds <= 0.0 {
            return 1.0;
        }
        (now.saturating_duration_since(self.published).as_secs_f32() / self.tick_seconds).clamp(0.0, 1.0)
    }
}

/// The latest stream, shared by its producer and the render thread.
#[derive(Default)]
pub struct UnitFeed {
    latest: Mutex<Option<Arc<InstanceStream>>>,
}

impl UnitFeed {
    pub fn publish(&self, stream: Arc<InstanceStream>) {
        *self.latest.lock().unwrap_or_else(|p| p.into_inner()) = Some(stream);
    }

    pub fn latest(&self) -> Option<Arc<InstanceStream>> {
        self.latest.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
}

/// How one class of units looks.
#[derive(Clone, Copy, Debug)]
pub struct UnitClass {
    /// Linear RGB.
    pub color: [f32; 3],
    /// Sits on the water surface where the ground is below it (ships,
    /// hovercraft); otherwise on the ground, under water or not.
    pub floats: bool,
}

/// Units for a [`crate::terrain::app::TerrainApp`].
#[derive(Clone)]
pub struct UnitsSetup {
    pub feed: Arc<UnitFeed>,
    /// Instances per frame; rows beyond it are not drawn.
    pub capacity: u32,
    pub classes: Vec<UnitClass>,
    /// Units smaller than this on screen are drawn this large (pixels,
    /// across), so they stay visible from far away.
    pub min_pixels: f32,
}

/// Per-frame inputs besides the stream.
#[derive(Clone, Copy, Debug)]
pub struct UnitFrame {
    pub view_proj: Mat4,
    pub cam_pos: Vec3,
    /// Pixels per world unit at distance 1.
    pub proj_scale: f32,
    /// Fog colour (linear) and density; density 0 turns fog off.
    pub fog: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FrameUbo {
    view_proj: [f32; 16],
    cam_pos: [f32; 4],
    frustum: [[f32; 4]; 6],
    map: [f32; 4],
    height: [f32; 4],
    sun: [f32; 4],
    fog: [f32; 4],
    counts: [u32; 4],
    classes: [[f32; 4]; MAX_CLASSES],
}

pub struct UnitRenderer {
    capacity: u32,
    info: TerrainInfo,
    classes: [[f32; 4]; MAX_CLASSES],
    min_pixels: f32,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    desc_pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    cull: vk::Pipeline,
    draw: vk::Pipeline,
    /// Per slot: the stream's columns, `capacity` words each.
    instances: Vec<OwnedBuffer>,
    /// Per slot: `(tick, rows)` of the stream in `instances`.
    uploaded: Vec<Option<(u64, u32)>>,
    ubos: Vec<OwnedBuffer>,
    visible: OwnedBuffer,
    /// `VkDrawIndirectCommand`: vertex count fixed, instance count from the cull.
    indirect: OwnedBuffer,
    /// Per slot: a host copy of `indirect` after the draw.
    counts: Vec<OwnedBuffer>,
}

impl UnitRenderer {
    /// Units over `terrain` (its height texture places them), drawn into a
    /// pass with a `color_format` target and a `depth_format` reverse-Z
    /// depth buffer.
    pub fn new(gpu: &GpuDevice, terrain: &TerrainRenderer, setup: &UnitsSetup, color_format: vk::Format, depth_format: vk::Format, frames_in_flight: usize) -> Result<Self> {
        ensure!(setup.capacity > 0, "unit capacity 0");
        let device = &gpu.device;
        let capacity = setup.capacity;
        let mut classes = [[0.8, 0.8, 0.8, 0.0]; MAX_CLASSES];
        for (slot, class) in classes.iter_mut().zip(&setup.classes) {
            *slot = [class.color[0], class.color[1], class.color[2], class.floats as u32 as f32];
        }
        let stream_bytes = capacity as u64 * ATTRIBUTE_COUNT as u64 * 4;
        let instances = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: stream_bytes, usage: vk::BufferUsageFlags::STORAGE_BUFFER }, MemoryLocation::CpuToGpu, "unit instances"))
            .collect::<Result<Vec<_>>>()?;
        let ubos = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: size_of::<FrameUbo>() as u64, usage: vk::BufferUsageFlags::UNIFORM_BUFFER }, MemoryLocation::CpuToGpu, "unit frame"))
            .collect::<Result<Vec<_>>>()?;
        let visible = OwnedBuffer::new(gpu, BufferDesc { size: capacity as u64 * 4, usage: vk::BufferUsageFlags::STORAGE_BUFFER }, MemoryLocation::GpuOnly, "unit visible")?;
        let words = [UNIT_VERTICES, 0, 0, 0];
        let indirect = upload::buffer(gpu, pipeline::push_bytes(&words), Self::indirect_usage(), "unit indirect")?;
        let counts = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: 16, usage: vk::BufferUsageFlags::TRANSFER_DST }, MemoryLocation::GpuToCpu, "unit counts"))
            .collect::<Result<Vec<_>>>()?;

        let all = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let (height_view, height_sampler) = terrain.height_texture();
        let (set_layout, pipeline_layout, desc_pool, sets) = unsafe {
            let set_layout = pipeline::set_layout(
                device,
                &[
                    (vk::DescriptorType::UNIFORM_BUFFER, all),
                    (vk::DescriptorType::STORAGE_BUFFER, all),
                    (vk::DescriptorType::STORAGE_BUFFER, all),
                    (vk::DescriptorType::STORAGE_BUFFER, all),
                    (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, all),
                ],
            )?;
            let pipeline_layout = pipeline::pipeline_layout(device, &[set_layout], 0, vk::ShaderStageFlags::empty())?;
            let n = frames_in_flight as u32;
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: n },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 * n },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: n },
            ];
            let desc_pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes), None)?;
            let layouts = vec![set_layout; frames_in_flight];
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&layouts))?;
            for (slot, &set) in sets.iter().enumerate() {
                let info = |buffer: &OwnedBuffer| [vk::DescriptorBufferInfo { buffer: buffer.buffer, offset: 0, range: vk::WHOLE_SIZE }];
                let buffers = [info(&ubos[slot]), info(&instances[slot]), info(&visible), info(&indirect)];
                let image = [vk::DescriptorImageInfo { sampler: height_sampler, image_view: height_view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
                let mut writes: Vec<vk::WriteDescriptorSet> = buffers
                    .iter()
                    .enumerate()
                    .map(|(i, info)| {
                        let ty = if i == 0 { vk::DescriptorType::UNIFORM_BUFFER } else { vk::DescriptorType::STORAGE_BUFFER };
                        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(ty).buffer_info(info)
                    })
                    .collect();
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(4).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&image));
                device.update_descriptor_sets(&writes, &[]);
            }
            (set_layout, pipeline_layout, desc_pool, sets)
        };
        let cull = pipeline::compute_pipeline(device, spirv!("units_cull.comp"), pipeline_layout).context("unit cull pipeline")?;
        let draw = pipeline::graphics_pipeline(
            device,
            &GraphicsDesc {
                vertex: spirv!("units.vert"),
                fragment: spirv!("units.frag"),
                layout: pipeline_layout,
                color_formats: &[color_format],
                depth: Some(DepthState { format: depth_format, write: true, compare: vk::CompareOp::GREATER_OR_EQUAL }),
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                blend: Blend::Off,
            },
        )
        .context("unit pipeline")?;
        Ok(Self {
            capacity,
            info: *terrain.info(),
            classes,
            min_pixels: setup.min_pixels,
            set_layout,
            pipeline_layout,
            desc_pool,
            sets,
            cull,
            draw,
            instances,
            uploaded: vec![None; frames_in_flight],
            ubos,
            visible,
            indirect,
            counts,
        })
    }

    fn indirect_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC
    }

    /// The host copy of the indirect arguments for `slot` (the copy pass
    /// writes it, a host-read pass makes it visible).
    pub fn counts(&self, slot: usize) -> BufferBinding {
        let b = &self.counts[slot];
        BufferBinding { buffer: b.buffer, offset: 0, size: b.size, identity: IDENTITY_COUNTS | slot as u64 }
    }

    pub fn counts_desc(&self) -> BufferDesc {
        BufferDesc { size: self.counts[0].size, usage: vk::BufferUsageFlags::TRANSFER_DST }
    }

    /// Units the cull passed in the frame that last used `slot` (call once
    /// that frame has completed: in `prepare`, or after the device idles).
    pub fn read_visible(&mut self, slot: usize) -> u32 {
        let Some(bytes) = self.counts[slot].mapped() else { return 0 };
        u32::from_le_bytes(bytes[4..8].try_into().unwrap())
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The instance buffer of `slot` (host-written in [`Self::write_frame`]).
    pub fn instances(&self, slot: usize) -> BufferBinding {
        let b = &self.instances[slot];
        BufferBinding { buffer: b.buffer, offset: 0, size: b.size, identity: IDENTITY_INSTANCES | slot as u64 }
    }

    pub fn instances_desc(&self) -> BufferDesc {
        BufferDesc { size: self.instances[0].size, usage: vk::BufferUsageFlags::STORAGE_BUFFER }
    }

    /// Visible instance indices (cull writes, draw reads).
    pub fn visible(&self) -> BufferBinding {
        BufferBinding { buffer: self.visible.buffer, offset: 0, size: self.visible.size, identity: IDENTITY_VISIBLE }
    }

    pub fn visible_desc(&self) -> BufferDesc {
        BufferDesc { size: self.visible.size, usage: vk::BufferUsageFlags::STORAGE_BUFFER }
    }

    /// The indirect draw (reset, then counted by the cull, read by the draw).
    pub fn indirect(&self) -> BufferBinding {
        BufferBinding { buffer: self.indirect.buffer, offset: 0, size: self.indirect.size, identity: IDENTITY_INDIRECT }
    }

    pub fn indirect_desc(&self) -> BufferDesc {
        BufferDesc { size: self.indirect.size, usage: Self::indirect_usage() }
    }

    /// Rows of `slot`'s stream (after [`Self::write_frame`]).
    pub fn rows(&self, slot: usize) -> u32 {
        self.uploaded[slot].map_or(0, |(_, rows)| rows)
    }

    /// Uniforms for `slot`, and the stream's columns when it is newer than
    /// the slot's copy (the slot's previous submission has completed).
    pub fn write_frame(&mut self, slot: usize, frame: &UnitFrame, stream: Option<&InstanceStream>, now: Instant) {
        let mut alpha = 1.0;
        match stream {
            Some(stream) => {
                alpha = stream.alpha(now);
                let rows = stream.len().min(self.capacity as usize) as u32;
                if self.uploaded[slot] != Some((stream.tick, rows)) {
                    let capacity = self.capacity as usize;
                    if let Some(bytes) = self.instances[slot].mapped() {
                        for (a, column) in stream.columns.iter().enumerate() {
                            let words = &column[..rows as usize];
                            bytes[a * capacity * 4..a * capacity * 4 + words.len() * 4].copy_from_slice(pipeline::slice_bytes(words));
                        }
                    }
                    self.uploaded[slot] = Some((stream.tick, rows));
                }
            }
            None => self.uploaded[slot] = None,
        }
        let i = &self.info;
        let ubo = FrameUbo {
            view_proj: frame.view_proj.to_cols_array(),
            cam_pos: frame.cam_pos.extend(frame.proj_scale).to_array(),
            frustum: frustum_planes(frame.view_proj),
            map: [i.cell_size, 1.0 / (i.cells.0 + 1) as f32, 1.0 / (i.cells.1 + 1) as f32, alpha],
            height: [i.height_scale, i.height_offset, i.water_level.unwrap_or(0.0), i.water_level.is_some() as u32 as f32],
            sun: Vec3::new(-0.5, 0.55, 0.35).normalize().extend(self.min_pixels).to_array(),
            fog: frame.fog,
            counts: [self.rows(slot), self.capacity, 0, 0],
            classes: self.classes,
        };
        if let Some(bytes) = self.ubos[slot].mapped() {
            bytes[..size_of::<FrameUbo>()].copy_from_slice(pipeline::push_bytes(&ubo));
        }
    }

    /// Zero the instance count. Records inside a transfer pass that
    /// declares [`Self::indirect`] as a transfer destination.
    pub fn record_reset(&self, ctx: &PassContext<'_>) {
        unsafe { ctx.device.cmd_fill_buffer(ctx.cmd, self.indirect.buffer, 4, 4, 0) };
    }

    /// Copy the indirect arguments to `slot`'s host buffer. Records inside a
    /// transfer pass that reads [`Self::indirect`] and writes [`Self::counts`].
    pub fn record_count(&self, ctx: &PassContext<'_>) {
        unsafe { ctx.device.cmd_copy_buffer(ctx.cmd, self.indirect.buffer, self.counts[ctx.slot].buffer, &[vk::BufferCopy::default().size(16)]) };
    }

    /// Frustum culling. Records inside a compute pass that reads
    /// [`Self::instances`] and writes [`Self::visible`] and
    /// [`Self::indirect`].
    pub fn record_cull(&self, ctx: &PassContext<'_>) {
        let rows = self.rows(ctx.slot);
        if rows == 0 {
            return;
        }
        let (device, cmd) = (ctx.device, ctx.cmd);
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cull);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.sets[ctx.slot]], &[]);
            device.cmd_dispatch(cmd, rows.div_ceil(64), 1, 1);
        }
    }

    /// The visible units. Records inside a graphics pass that reads
    /// [`Self::instances`] and [`Self::visible`] in the vertex stage and
    /// [`Self::indirect`] as an indirect buffer, with a colour attachment
    /// and a reverse-Z depth attachment.
    pub fn record_draw(&self, ctx: &PassContext<'_>) {
        let (device, cmd) = (ctx.device, ctx.cmd);
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.draw);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.sets[ctx.slot]], &[]);
            device.cmd_draw_indirect(cmd, self.indirect.buffer, 0, 1, 0);
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        let device = &gpu.device;
        unsafe {
            device.destroy_pipeline(self.cull, None);
            device.destroy_pipeline(self.draw, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.desc_pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
        for b in self.instances.iter_mut().chain(self.ubos.iter_mut()).chain(self.counts.iter_mut()) {
            b.destroy(gpu);
        }
        self.visible.destroy(gpu);
        self.indirect.destroy(gpu);
    }
}

/// The cull of `units_cull.comp` on the CPU, for tests: per row of `stream`
/// (all rows; the GPU sees at most the capacity), the smallest signed
/// distance of the unit's bounding sphere to the outside of a frustum plane
/// (`>= 0`: kept), the plane it is (`0..6`, as [`frustum_planes`]) and the
/// sphere's radius (a kept unit with a margin below it has its centre
/// outside the view).
/// `ground(x, y, class)` is the height the unit stands at; positions are
/// interpolated with `alpha` as the shader does.
pub fn cull_margins(stream: &InstanceStream, frame: &UnitFrame, min_pixels: f32, alpha: f32, ground: impl Fn(f32, f32, u32) -> f32) -> Vec<(f32, usize, f32)> {
    let planes = frustum_planes(frame.view_proj);
    let f = |a: usize, r: usize| f32::from_bits(stream.columns[a][r]);
    (0..stream.len())
        .map(|r| {
            let (x, y) = (f(2, r) + (f(0, r) - f(2, r)) * alpha, f(3, r) + (f(1, r) - f(3, r)) * alpha);
            let class = stream.columns[7][r].min(MAX_CLASSES as u32 - 1);
            let pos = Vec3::new(x, ground(x, y, class), y);
            let radius = f(6, r);
            let pixels = 2.0 * radius * frame.proj_scale / pos.distance(frame.cam_pos).max(1e-3);
            let size = radius * (min_pixels / pixels.max(1e-6)).max(1.0);
            let centre = pos + Vec3::new(0.0, 0.4 * size, 0.0);
            let reach = 1.6 * size;
            let (margin, plane) = planes
                .iter()
                .enumerate()
                .map(|(k, p)| (Vec3::new(p[0], p[1], p[2]).dot(centre) + p[3] + reach, k))
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .unwrap();
            (margin, plane, reach)
        })
        .collect()
}
