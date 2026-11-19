//! Clustered lights (M12): the frame's light list and, per view cluster,
//! the lights that reach it.
//!
//! * A [`Light`] is a point, a cone or a beam (a capsule along a segment),
//!   with `~/particles`' falloff (`shaders/lights.glsl`): `(1 − d²/r²)² /
//!   (1 + d²/s²)`, `d` the distance, `r` the range and `s` the source size,
//!   a cone's cut by a smoothstep between two cosines.
//! * The view is cut into [`TILES_X`] × [`TILES_Y`] screen tiles (about
//!   32 pixels at 1280×720) and [`SLICES`] slices of view depth, spaced
//!   exponentially over the depths the map can occupy in view
//!   ([`depth_range`]; slice 0 from the near plane to them, the last
//!   reaching to the horizon); the grid is fixed, so a resize reallocates
//!   nothing.
//! * The `lights.cluster` library pass (compute) copies the host's list
//!   into device memory, bounds each light by the tiles and slices its
//!   bounding sphere reaches (`shaders/lights_bounds.comp`), lists each
//!   tile's lights (`lights_tiles.comp`), then each cluster's from its
//!   tile's by their slices (`lights_cluster.comp`), in light order, so per-pixel and per-particle sums are the same every
//!   run. A tile holds at most [`PER_TILE`], a cluster 256, the index list
//!   [`INDEX_CAPACITY`] entries; what is cut is counted ([`LightStats`]).
//! * The lit passes (the terrain, the units, the water and the particles'
//!   simulation) read the list and the clusters through this module's
//!   descriptor set ([`LightRenderer::set_layout`]), at the set number their
//!   pipeline layouts give it. In a graph without the lights the set is
//!   bound all the same and reads as off.
//!
//! Per frame slot: a host-written uniform block and light list, and the
//! clusters (device memory, written by the pass).

use crate::device::GpuDevice;
use crate::executor::{BufferBinding, PassContext};
use crate::memory::OwnedBuffer;
use crate::pipeline::{self, spirv};
use ae_framegraph::BufferDesc;
use anyhow::{Context as _, Result};
use ash::vk;
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;

pub const TILES_X: u32 = 40;
pub const TILES_Y: u32 = 24;
pub const TILES: u32 = TILES_X * TILES_Y;
pub const SLICES: u32 = 16;
pub const CLUSTERS: u32 = TILES * SLICES;
/// Most lights a tile lists (`LIGHT_PER_TILE` in `lights.glsl`).
pub const PER_TILE: u32 = 1024;
/// The index list's length (`LIGHT_INDEX_CAPACITY`).
pub const INDEX_CAPACITY: u32 = CLUSTERS * 32;
/// Most lights a frame lists (more are dropped and counted).
pub const MAX_LIGHTS: usize = 8192;
/// The slices' spacing never ends past this view depth (metres); the last
/// slice reaches past it.
pub const LAST_SLICE: f32 = 8000.0;

const IDENTITY_LIST: u64 = (0xAE7 << 40) | (6 << 32);
const IDENTITY_CLUSTERS: u64 = (0xAE7 << 40) | (6 << 32) | 0x100;
const HEADER_BYTES: u64 = 16;

pub const POINT: u32 = 1;
pub const CONE: u32 = 2;
pub const BEAM: u32 = 4;

/// One light, as `lights.glsl` reads it (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    pub position: [f32; 3],
    /// Nothing past it (metres).
    pub range: f32,
    /// Linear, may exceed 1.
    pub color: [f32; 3],
    pub kind: u32,
    /// A cone's axis, a beam's segment (unit).
    pub direction: [f32; 3],
    /// A beam's half length.
    pub half_length: f32,
    /// x, y: a cone's inner and outer cosines; z: the source size (> 0).
    pub profile: [f32; 4],
}

impl Light {
    /// A point light of `color` reaching `range`, from a source of `size`.
    pub fn point(position: Vec3, color: Vec3, range: f32, size: f32) -> Self {
        Self { position: position.to_array(), range, color: color.to_array(), kind: POINT, direction: [0.0, -1.0, 0.0], half_length: 0.0, profile: [0.0, 0.0, size.max(1e-3), 0.0] }
    }

    /// A cone along `direction`, full inside `inner` and dark outside
    /// `outer` (half-angles, radians).
    pub fn cone(position: Vec3, direction: Vec3, color: Vec3, range: f32, size: f32, inner: f32, outer: f32) -> Self {
        let outer = outer.max(inner + 1e-3);
        Self { position: position.to_array(), range, color: color.to_array(), kind: CONE, direction: direction.normalize_or(Vec3::NEG_Y).to_array(), half_length: 0.0, profile: [inner.cos(), outer.cos(), size.max(1e-3), 0.0] }
    }

    /// A beam from `a` to `b`, `range` around the segment.
    pub fn beam(a: Vec3, b: Vec3, color: Vec3, range: f32, size: f32) -> Self {
        let d = b - a;
        Self { position: ((a + b) * 0.5).to_array(), range, color: color.to_array(), kind: BEAM, direction: d.normalize_or(Vec3::X).to_array(), half_length: d.length() * 0.5, profile: [0.0, 0.0, size.max(1e-3), 0.0] }
    }

    /// Centre and radius of a sphere holding everything it lights.
    pub fn bounds(&self) -> (Vec3, f32) {
        (Vec3::from(self.position), self.range + if self.kind == BEAM { self.half_length } else { 0.0 })
    }
}

/// `AE_LIGHT_SLICES` (1 to [`SLICES`]), else [`SLICES`].
fn slices_from_env() -> u32 {
    std::env::var("AE_LIGHT_SLICES").ok().and_then(|v| v.parse::<u32>().ok()).map_or(SLICES, |n| n.clamp(1, SLICES))
}

/// `n` point lights scattered over a disc of `radius` around `centre`
/// (x, z), each `lift` metres over the ground (`ground` gives its height):
/// warm and cool colours, ranges of 15 to 45 m. Seeded: the same lights
/// for the same arguments (a test and benchmark of the clusters).
pub fn scatter(n: u32, centre: glam::Vec2, radius: f32, lift: f32, seed: u32, ground: &dyn Fn(glam::Vec2) -> f32) -> Vec<Light> {
    let mut state = seed.wrapping_mul(0x9e37_79b9) ^ 0x85eb_ca6b;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as f32 / u32::MAX as f32
    };
    (0..n)
        .map(|_| {
            let (a, r) = (next() * std::f32::consts::TAU, radius * next().sqrt());
            let p = centre + glam::Vec2::new(a.cos(), a.sin()) * r;
            let hue = next();
            let color = Vec3::new(1.0, 0.55, 0.2).lerp(Vec3::new(0.35, 0.6, 1.0), hue) * (1.0 + 2.0 * next());
            let range = 15.0 + 30.0 * next();
            Light::point(Vec3::new(p.x, ground(p) + lift, p.y), color, range, range * 0.25)
        })
        .collect()
}

/// Per-frame inputs.
#[derive(Clone, Copy, Debug)]
pub struct LightFrame {
    pub view_proj: Mat4,
    pub cam_pos: Vec3,
    /// The camera's right, up and forward axes.
    pub axes: [Vec3; 3],
    /// The projection's near plane (infinite reverse-Z: depth = near / z).
    pub near: f32,
    pub extent: vk::Extent2D,
    /// The graph clusters the lights this frame (a `lights.cluster`
    /// pass); else the lit passes read them as off.
    pub on: bool,
    /// The view depths the scene can occupy ([`depth_range`]): slice 0
    /// reaches to the first, the others are spaced exponentially to the
    /// second, the last reaching past it.
    pub depths: (f32, f32),
}

/// The view depths of the part of the view between heights `y_lo` and
/// `y_hi` (the map's slab, with room above for what stands on it): the
/// least and the most over where the frustum's corner rays meet the two
/// planes (the vertices of the frustum cut by the slab). From inside the
/// slab the first is the near plane; with a corner ray over the horizon
/// the second is [`LAST_SLICE`].
pub fn depth_range(view_proj: Mat4, cam_pos: Vec3, forward: Vec3, near: f32, y_lo: f32, y_hi: f32) -> (f32, f32) {
    let inv = view_proj.inverse();
    let (mut lo, mut hi) = (f32::INFINITY, 0.0f32);
    let inside = cam_pos.y >= y_lo && cam_pos.y <= y_hi;
    for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        // A point on the corner's ray (reverse-Z: depth near / z), z = 100.
        let p = inv * glam::Vec4::new(x, y, near / 100.0, 1.0);
        let d = (p.truncate() / p.w - cam_pos).normalize_or_zero();
        let depth_at = |plane: f32| {
            let t = (plane - cam_pos.y) / d.y;
            (d.y.abs() > 1e-6 && t > 0.0).then(|| t * d.dot(forward))
        };
        // The far plane the ray can reach: the lower one from above, the
        // upper one from below.
        let far = if cam_pos.y > y_lo { depth_at(y_lo) } else { depth_at(y_hi) };
        hi = hi.max(far.unwrap_or(LAST_SLICE));
        let close = if cam_pos.y > y_hi { depth_at(y_hi) } else if cam_pos.y < y_lo { depth_at(y_lo) } else { None };
        if let Some(z) = close {
            lo = lo.min(z);
        }
    }
    let lo = if inside || !lo.is_finite() { near } else { lo };
    let lo = lo.max(near * 2.0).min(LAST_SLICE * 0.5);
    (lo, hi.clamp(lo * 1.5, LAST_SLICE))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FrameUbo {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    cam_pos: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    forward: [f32; 4],
    grid: [u32; 4],
    slicing: [f32; 4],
}

/// What the last completed frame of a slot cut.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LightStats {
    /// Lights listed.
    pub lights: u32,
    /// Index list entries used.
    pub indices: u32,
    /// Clusters cut short (more than 256 lights, or the list full).
    pub overflow: u32,
    /// Lights past [`MAX_LIGHTS`], not listed.
    pub dropped: u32,
}

pub struct LightRenderer {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    bounds: vk::Pipeline,
    tiles: vk::Pipeline,
    cull: vk::Pipeline,
    ubos: Vec<OwnedBuffer>,
    /// Host-written: the header and the lights, copied into `lists` when
    /// the pass records (the passes read device memory).
    staging: Vec<OwnedBuffer>,
    lists: Vec<OwnedBuffer>,
    clusters: Vec<OwnedBuffer>,
    on: Vec<bool>,
    /// Per slot: the bytes `staging` holds this frame.
    staged: Vec<u64>,
    /// Per slot: the list's header after the cull (its counts), read when
    /// the slot is next written.
    counts: Vec<OwnedBuffer>,
    stats: LightStats,
    /// Slices of view depth in use (at most [`SLICES`]; 1: screen tiles
    /// only). `AE_LIGHT_SLICES` sets it, to measure.
    slices: u32,
    /// Every light at every pixel, no culls ([`Self::set_unculled`]).
    unculled: bool,
}

impl LightRenderer {
    pub fn new(gpu: &GpuDevice, frames_in_flight: usize) -> Result<Self> {
        let device = &gpu.device;
        let mut ubos = Vec::new();
        let mut staging = Vec::new();
        let mut counts = Vec::new();
        let mut lists = Vec::new();
        let mut clusters = Vec::new();
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        for slot in 0..frames_in_flight {
            ubos.push(OwnedBuffer::new(gpu, BufferDesc { size: size_of::<FrameUbo>() as u64, usage: vk::BufferUsageFlags::UNIFORM_BUFFER }, MemoryLocation::CpuToGpu, &format!("light frame {slot}"))?);
            let size = HEADER_BYTES + (MAX_LIGHTS * size_of::<Light>()) as u64;
            let mut stage = OwnedBuffer::new(gpu, BufferDesc { size, usage: vk::BufferUsageFlags::TRANSFER_SRC }, MemoryLocation::CpuToGpu, &format!("lights staging {slot}"))?;
            if let Some(bytes) = stage.mapped() {
                bytes[..HEADER_BYTES as usize].fill(0);
            }
            staging.push(stage);
            let mut count = OwnedBuffer::new(gpu, BufferDesc { size: HEADER_BYTES, usage: vk::BufferUsageFlags::TRANSFER_DST }, MemoryLocation::GpuToCpu, &format!("light counts {slot}"))?;
            if let Some(bytes) = count.mapped() {
                bytes.fill(0);
            }
            counts.push(count);
            lists.push(OwnedBuffer::new(gpu, BufferDesc { size, usage: storage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC }, MemoryLocation::GpuOnly, &format!("lights {slot}"))?);
            let words = 2 * MAX_LIGHTS as u64 + TILES as u64 * (1 + PER_TILE as u64) + INDEX_CAPACITY as u64;
            clusters.push(OwnedBuffer::new(gpu, BufferDesc { size: (CLUSTERS as u64) * 8 + words * 4, usage: storage | vk::BufferUsageFlags::TRANSFER_DST }, MemoryLocation::GpuOnly, &format!("light clusters {slot}"))?);
        }
        let all = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let ssbo = vk::DescriptorType::STORAGE_BUFFER;
        let (set_layout, pipeline_layout, pool, sets) = unsafe {
            let set_layout = pipeline::set_layout(device, &[(vk::DescriptorType::UNIFORM_BUFFER, all), (ssbo, all), (ssbo, all)])?;
            let pipeline_layout = pipeline::pipeline_layout(device, &[set_layout], 0, vk::ShaderStageFlags::empty())?;
            let n = frames_in_flight as u32;
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: n }, vk::DescriptorPoolSize { ty: ssbo, descriptor_count: 2 * n }];
            let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes), None)?;
            let layouts = vec![set_layout; frames_in_flight];
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))?;
            for (slot, &set) in sets.iter().enumerate() {
                let info = |b: &OwnedBuffer| [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: vk::WHOLE_SIZE }];
                let (u, l, c) = (info(&ubos[slot]), info(&lists[slot]), info(&clusters[slot]));
                let writes = [
                    vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&u),
                    vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(ssbo).buffer_info(&l),
                    vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(ssbo).buffer_info(&c),
                ];
                device.update_descriptor_sets(&writes, &[]);
            }
            (set_layout, pipeline_layout, pool, sets)
        };
        let bounds = pipeline::compute_pipeline(device, spirv!("lights_bounds.comp"), pipeline_layout).context("light bounds pipeline")?;
        let tiles = pipeline::compute_pipeline(device, spirv!("lights_tiles.comp"), pipeline_layout).context("light tile pipeline")?;
        let cull = pipeline::compute_pipeline(device, spirv!("lights_cluster.comp"), pipeline_layout).context("light cluster pipeline")?;
        Ok(Self { set_layout, pipeline_layout, pool, sets, bounds, tiles, cull, ubos, staging, lists, clusters, on: vec![false; frames_in_flight], staged: vec![0; frames_in_flight], counts, stats: LightStats::default(), slices: slices_from_env(), unculled: false })
    }

    /// Light every pixel and particle by every light, skipping the culls:
    /// the reference the clusters must match exactly (a light left out of a
    /// cluster adds exactly 0 where it reaches no further than its range).
    pub fn set_unculled(&mut self, unculled: bool) {
        self.unculled = unculled;
    }

    /// The lit passes' set (bind [`Self::set`] at the number their layouts
    /// give it).
    pub fn set_layout(&self) -> vk::DescriptorSetLayout {
        self.set_layout
    }

    pub fn set(&self, slot: usize) -> vk::DescriptorSet {
        self.sets[slot]
    }

    /// Every slot's set, for renderers that keep them.
    pub fn sets(&self) -> Vec<vk::DescriptorSet> {
        self.sets.clone()
    }

    /// Write `slot`'s uniforms and light list (its previous submission has
    /// completed). Lights past [`MAX_LIGHTS`] are dropped.
    pub fn write_frame(&mut self, slot: usize, frame: &LightFrame, lights: &[Light]) {
        let n = lights.len().min(MAX_LIGHTS);
        let count = if frame.on { n } else { 0 };
        if let Some(bytes) = self.staging[slot].mapped() {
            let header = [count as u32, 0, 0, (lights.len() - n) as u32];
            bytes[..16].copy_from_slice(pipeline::slice_bytes(&header));
            let raw = pipeline::slice_bytes(&lights[..count]);
            bytes[16..16 + raw.len()].copy_from_slice(raw);
        }
        self.staged[slot] = HEADER_BYTES + (count * size_of::<Light>()) as u64;
        if self.on[slot]
            && let Some(bytes) = self.counts[slot].mapped()
        {
            let word = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            self.stats = LightStats { lights: word(0), indices: word(1), overflow: word(2), dropped: word(3) };
        }
        self.on[slot] = frame.on;
        let [right, up, forward] = frame.axes;
        let ubo = FrameUbo {
            view_proj: frame.view_proj.to_cols_array(),
            inv_view_proj: frame.view_proj.inverse().to_cols_array(),
            cam_pos: frame.cam_pos.extend(frame.near).to_array(),
            right: right.extend(0.0).to_array(),
            up: up.extend(0.0).to_array(),
            forward: forward.extend(0.0).to_array(),
            grid: [TILES_X, TILES_Y, self.slices, if frame.on { 1 + self.unculled as u32 } else { 0 }],
            slicing: [frame.depths.0, (self.slices.max(2) - 1) as f32 / (frame.depths.1 / frame.depths.0).ln(), frame.extent.width.max(1) as f32, frame.extent.height.max(1) as f32],
        };
        if let Some(bytes) = self.ubos[slot].mapped() {
            bytes[..size_of::<FrameUbo>()].copy_from_slice(pipeline::push_bytes(&ubo));
        }
    }

    /// The counts of the slot's last frame with lights (read back when the
    /// slot is written again).
    pub fn stats(&self) -> LightStats {
        self.stats
    }

    /// The light list (the host writes it; `lights.cluster` counts into its
    /// header).
    pub fn list(&self, slot: usize) -> BufferBinding {
        let b = &self.lists[slot];
        BufferBinding { buffer: b.buffer, offset: 0, size: b.size, identity: IDENTITY_LIST | slot as u64 }
    }

    pub fn list_desc(&self) -> BufferDesc {
        BufferDesc { size: self.lists[0].size, usage: vk::BufferUsageFlags::STORAGE_BUFFER }
    }

    pub fn clusters(&self, slot: usize) -> BufferBinding {
        let b = &self.clusters[slot];
        BufferBinding { buffer: b.buffer, offset: 0, size: b.size, identity: IDENTITY_CLUSTERS | slot as u64 }
    }

    pub fn clusters_desc(&self) -> BufferDesc {
        BufferDesc { size: self.clusters[0].size, usage: vk::BufferUsageFlags::STORAGE_BUFFER }
    }

    /// The clusters' lists. Records inside a compute pass that writes
    /// [`Self::list`] and [`Self::clusters`]: copies the host's list in
    /// (with its own barrier to the cull), then the cull.
    pub fn record_cluster(&self, ctx: &PassContext<'_>) {
        if !self.on[ctx.slot] || self.unculled {
            return;
        }
        unsafe {
            let copy = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: self.staged[ctx.slot] };
            ctx.device.cmd_copy_buffer(ctx.cmd, self.staging[ctx.slot].buffer, self.lists[ctx.slot].buffer, &[copy]);
            let memory = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)];
            ctx.device.cmd_pipeline_barrier2(ctx.cmd, &vk::DependencyInfo::default().memory_barriers(&memory));
            ctx.device.cmd_bind_descriptor_sets(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.sets[ctx.slot]], &[]);
            let step = || {
                let memory = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)];
                ctx.device.cmd_pipeline_barrier2(ctx.cmd, &vk::DependencyInfo::default().memory_barriers(&memory));
            };
            let listed = ((self.staged[ctx.slot] - HEADER_BYTES) / size_of::<Light>() as u64) as u32;
            ctx.device.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.bounds);
            ctx.device.cmd_dispatch(ctx.cmd, listed.div_ceil(64), 1, 1);
            step();
            ctx.device.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.tiles);
            ctx.device.cmd_dispatch(ctx.cmd, TILES, 1, 1);
            step();
            ctx.device.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.cull);
            ctx.device.cmd_dispatch(ctx.cmd, TILES * self.slices, 1, 1);
            // The counts back to the host (read when the slot comes round).
            let memory = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
            ctx.device.cmd_pipeline_barrier2(ctx.cmd, &vk::DependencyInfo::default().memory_barriers(&memory));
            let copy = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: HEADER_BYTES };
            ctx.device.cmd_copy_buffer(ctx.cmd, self.lists[ctx.slot].buffer, self.counts[ctx.slot].buffer, &[copy]);
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        unsafe {
            gpu.device.destroy_pipeline(self.bounds, None);
            gpu.device.destroy_pipeline(self.tiles, None);
            gpu.device.destroy_pipeline(self.cull, None);
            gpu.device.destroy_pipeline_layout(self.pipeline_layout, None);
            gpu.device.destroy_descriptor_pool(self.pool, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
        for b in self.ubos.iter_mut().chain(&mut self.counts).chain(&mut self.staging).chain(&mut self.lists).chain(&mut self.clusters) {
            b.destroy(gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_range_spans_the_slab_in_view() {
        // From 500 m up, looking 45° down along +z: the slab 0..100 m.
        let eye = Vec3::new(0.0, 500.0, 0.0);
        let forward = Vec3::new(0.0, -1.0, 1.0).normalize();
        let view = Mat4::look_to_rh(eye, forward, Vec3::Y);
        let proj = Mat4::perspective_infinite_reverse_rh(0.8, 16.0 / 9.0, 1.0);
        let (lo, hi) = depth_range(proj * view, eye, forward, 1.0, 0.0, 100.0);
        // Straight ahead the ground is 707 m away and the slab's top 566 m.
        assert!(lo > 300.0 && lo < 566.0, "{lo}");
        assert!(hi > 707.0 && hi < 2000.0, "{hi}");
        // Looking at the horizon, the far end is open.
        let level = Vec3::new(0.0, -0.05, 1.0).normalize();
        let view = Mat4::look_to_rh(eye, level, Vec3::Y);
        assert_eq!(depth_range(proj * view, eye, level, 1.0, 0.0, 100.0).1, LAST_SLICE);
        // From inside the slab, from the near plane.
        let low = Vec3::new(0.0, 50.0, 0.0);
        let view = Mat4::look_to_rh(low, forward, Vec3::Y);
        assert_eq!(depth_range(proj * view, low, forward, 1.0, 0.0, 100.0).0, 2.0);
    }

    #[test]
    fn light_is_64_bytes() {
        assert_eq!(size_of::<Light>(), 64);
        assert_eq!(size_of::<FrameUbo>(), 128 + 16 * 6);
    }
}
