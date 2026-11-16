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
//! The passes: `units pose` (`units.pose`, compute: the skinning palettes
//! of the units with joint keys into `units.palette`, which the draws
//! read),
//! `units reset` (transfer: the indirect instance counts to 0), `units
//! cull` (compute: frustum test per unit, appends visible indices and
//! counts them; with models and shadows, also each model unit per shadow
//! cascade), with shadows `shadows` (`shadows.draw`: depth only, every
//! dart and the model units the cull found per cascade, into each cascade
//! of the units' shadow map, [`crate::shadow`]), `units` (graphics: draws
//! the visible list over the terrain, loading its colour target and depth
//! buffer, reverse-Z), and `units count` + `units count read` (copy the
//! indirect arguments to a host buffer per slot, read back in `prepare`
//! as the visible count). The instance data is one host-visible buffer per
//! frame slot, written in `prepare` only when a newer tick arrived.

use crate::device::GpuDevice;
use crate::executor::{BufferBinding, PassContext};
use crate::memory::OwnedBuffer;
use crate::models::{self, ModelSet};
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
/// is a `u32` indexing [`UnitsSetup::classes`]. `pose` (a `u32`) is where
/// a unit with a model finds its joint keys in [`InstanceStream::keys`]:
/// its model's bones' keys from there, two per bone (older, newer), or
/// [`NO_POSE`] for the rest pose. Every other attribute is an `f32`.
pub const ATTRIBUTES: [&str; 9] = ["x", "y", "prev_x", "prev_y", "heading_x", "heading_y", "radius", "class", "pose"];
pub const ATTRIBUTE_COUNT: usize = ATTRIBUTES.len();
const ATTR_CLASS: usize = 7;
const ATTR_POSE: usize = 8;
/// The `pose` of a unit drawn at its model's rest pose.
pub const NO_POSE: u32 = u32::MAX;

pub use ae_model::anim::Key;
/// Classes with their own colour; higher class ids use the last one.
pub const MAX_CLASSES: usize = 16;
/// Vertices of one unit (a four-faced dart).
pub(crate) const UNIT_VERTICES: u32 = 12;

const IDENTITY_INSTANCES: u64 = (0xAE7 << 40) | (2 << 32);
const IDENTITY_VISIBLE: u64 = (0xAE7 << 40) | (2 << 32) | 0x100;
const IDENTITY_INDIRECT: u64 = (0xAE7 << 40) | (2 << 32) | 0x101;
const IDENTITY_COUNTS: u64 = (0xAE7 << 40) | (2 << 32) | 0x200;
const IDENTITY_PALETTE: u64 = (0xAE7 << 40) | (2 << 32) | 0x102;

/// One snapshot of the units.
#[derive(Clone, Debug)]
pub struct InstanceStream {
    /// The sim tick it shows.
    pub tick: u64,
    /// When it was published: interpolation runs from `prev` to the latest
    /// position over the `tick_seconds` after this.
    pub published: Instant,
    /// Length of a tick in wall time (the sim's tick over its speed); 0
    /// shows the latest positions only.
    pub tick_seconds: f32,
    /// Sim time of a tick, whatever the sim's speed (effects follow it).
    pub sim_seconds: f32,
    /// `columns[attribute][row]`, raw 4-byte words, every column the same
    /// length.
    pub columns: [Vec<u32>; ATTRIBUTE_COUNT],
    /// Joint keys (`ae_model::anim::Key`): per posed unit, from its `pose`,
    /// two per bone of its model. A bone is drawn on the line through its
    /// two keys at the drawn tick (`tick - 1 + alpha`, past the newer key
    /// too).
    pub keys: Vec<Key>,
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

    /// The sim time drawn at `alpha` (seconds since tick 0; units are
    /// drawn a tick behind, between the previous tick and this one). It
    /// speeds up, slows down and stops with the sim, and never steps back
    /// when the next stream replaces this one.
    pub fn sim_time(&self, alpha: f32) -> f64 {
        (self.tick as f64 - 1.0 + alpha as f64) * self.sim_seconds.max(0.0) as f64
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
    /// A model per class (`models[class]`, [`crate::models`]); a class
    /// without one (or beyond the list) is drawn as a dart. A model's
    /// class colour is not used. Empty: darts only, and the renderer is
    /// exactly the darts' one.
    pub models: Vec<Option<Arc<ae_model::Model>>>,
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
    /// Shade by the hills' shadows (when the terrain has them).
    pub shadows: bool,
    /// Draw and shade by the units' shadow map with these cascades (the
    /// graph has it this frame).
    pub unit_shadows: Option<crate::shadow::Cascades>,
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
    unit_shadow: crate::shadow::CascadeUniforms,
    /// x: the tick drawn from (the stream's tick - 1), y: the fraction
    /// past it (`f32` bits; joint keys are evaluated there).
    anim: [u32; 4],
}

/// Makes a graphics pipeline from a vertex and a fragment shader, with the
/// colour target (`true`) or for the shadow map.
type PipelineMaker<'a> = dyn Fn(&[u8], &[u8], bool) -> Result<vk::Pipeline> + 'a;

/// The model side of a [`UnitRenderer`] whose classes have models.
struct Models {
    set: ModelSet,
    set_layout: vk::DescriptorSetLayout,
    /// Per slot: set 1 (`shaders/units_models.glsl`).
    sets: Vec<vk::DescriptorSet>,
    /// Per slot: pose offsets, the pose jobs (rows with joint keys, their
    /// scratch offsets).
    rows: Vec<OwnedBuffer>,
    /// Per slot: the stream's joint keys (as many as fit).
    keys: Vec<OwnedBuffer>,
    /// Per slot: pose jobs in `rows`.
    jobs: Vec<u32>,
    pose: vk::Pipeline,
    /// The indirect buffer's initial contents, copied in by the reset.
    template: OwnedBuffer,
    opaque: vk::Pipeline,
    masked: vk::Pipeline,
    /// Opaque and masked casters.
    caster: Option<(vk::Pipeline, vk::Pipeline)>,
}

pub struct UnitRenderer {
    capacity: u32,
    /// With models: their GPU side (the darts' pipelines then use the
    /// models' pipeline layout, and the cull is `units_models_cull.comp`).
    models: Option<Models>,
    info: TerrainInfo,
    classes: [[f32; 4]; MAX_CLASSES],
    min_pixels: f32,
    /// The terrain has shadow heights to shade the units by.
    shadows: bool,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    desc_pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    cull: vk::Pipeline,
    draw: vk::Pipeline,
    /// Into the units' shadow map (`None` when the terrain renderer has
    /// only its stand-in).
    caster: Option<vk::Pipeline>,
    /// Per slot: the frame draws the units' shadows.
    casting: Vec<bool>,
    /// Per slot: the stream's columns, `capacity` words each.
    instances: Vec<OwnedBuffer>,
    /// Per slot: `(tick, rows)` of the stream in `instances`.
    uploaded: Vec<Option<(u64, u32)>>,
    /// The skinning palettes (`units.palette`): with models, each class's
    /// rest palette at its head; `units.pose` writes the rest.
    palette: OwnedBuffer,
    ubos: Vec<OwnedBuffer>,
    visible: OwnedBuffer,
    /// `VkDrawIndirectCommand`: vertex count fixed, instance count from the cull.
    indirect: OwnedBuffer,
    /// Per slot: a host copy of `indirect` after the draw.
    counts: Vec<OwnedBuffer>,
}

impl Models {
    /// `slot`'s rows and keys: each row's palette offset (its model's rest
    /// palette, or a range of its own when it has joint keys: a pose job),
    /// the pose jobs (rows with keys, their scratch offsets), and the keys.
    fn write_rows(&mut self, slot: usize, stream: &InstanceStream, rows: usize, capacity: usize) {
        let class_of = |r: usize| (stream.columns[ATTR_CLASS][r] as usize).min(MAX_CLASSES - 1);
        let mut words = vec![0u32; 3 * capacity];
        let (ranges, scratch, _) = self.set.palette_layout(capacity as u32);
        let (mut range, mut scratch, mut jobs) = (ranges as u32, scratch as u32, 0);
        let fit = stream.keys.len().min((self.keys[slot].size / size_of::<Key>() as u64) as usize);
        if let Some(bytes) = self.keys[slot].mapped() {
            bytes[..fit * size_of::<Key>()].copy_from_slice(pipeline::slice_bytes(&stream.keys[..fit]));
        }
        for r in 0..rows {
            let Some(mc) = self.set.class_model[class_of(r)] else { continue };
            let mc = &self.set.classes[mc];
            let pose = stream.columns[ATTR_POSE][r];
            words[r] = match mc.bones > 0 && pose != NO_POSE && pose as usize + 2 * mc.bones as usize <= fit {
                true => {
                    words[capacity + jobs] = r as u32;
                    words[2 * capacity + jobs] = scratch;
                    jobs += 1;
                    scratch += mc.bones;
                    range += mc.joints;
                    range - mc.joints
                }
                false => mc.palette_offset,
            };
        }
        self.jobs[slot] = jobs as u32;
        if let Some(bytes) = self.rows[slot].mapped() {
            bytes[..words.len() * 4].copy_from_slice(pipeline::slice_bytes(&words));
        }
    }
}

impl UnitRenderer {
    /// Units over `terrain` (its height texture places them), drawn into a
    /// pass with a `color_format` target and a `depth_format` reverse-Z
    /// depth buffer.
    pub fn new(gpu: &GpuDevice, terrain: &TerrainRenderer, setup: &UnitsSetup, color_format: vk::Format, depth_format: vk::Format, frames_in_flight: usize) -> Result<Self> {
        ensure!(setup.capacity > 0, "unit capacity 0");
        let device = &gpu.device;
        let capacity = setup.capacity;
        let model_set = match setup.models.iter().any(Option::is_some) {
            true => Some(ModelSet::new(gpu, &setup.models, capacity, MAX_CLASSES).context("unit models")?),
            false => None,
        };
        let mut classes = [[0.8, 0.8, 0.8, 0.0]; MAX_CLASSES];
        for (slot, class) in classes.iter_mut().zip(&setup.classes) {
            *slot = [class.color[0], class.color[1], class.color[2], class.floats as u32 as f32];
        }
        if let Some(set) = &model_set {
            for c in &set.classes {
                classes[c.class as usize][3] += 2.0;
            }
        }
        let stream_bytes = capacity as u64 * ATTRIBUTE_COUNT as u64 * 4;
        let instances = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: stream_bytes, usage: vk::BufferUsageFlags::STORAGE_BUFFER }, MemoryLocation::CpuToGpu, "unit instances"))
            .collect::<Result<Vec<_>>>()?;
        let ubos = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: size_of::<FrameUbo>() as u64, usage: vk::BufferUsageFlags::UNIFORM_BUFFER }, MemoryLocation::CpuToGpu, "unit frame"))
            .collect::<Result<Vec<_>>>()?;
        // With models: regions of the visible list per model class after
        // the darts' (the camera's and each shadow cascade's), and their
        // commands after the darts'.
        let visible_len = model_set.as_ref().map_or(capacity as u64, |s| s.visible_len(capacity));
        let visible = OwnedBuffer::new(gpu, BufferDesc { size: visible_len * 4, usage: vk::BufferUsageFlags::STORAGE_BUFFER }, MemoryLocation::GpuOnly, "unit visible")?;
        let words = [UNIT_VERTICES, 0, 0, 0];
        let initial: &[u8] = match &model_set {
            Some(set) => &set.template,
            None => pipeline::push_bytes(&words),
        };
        let indirect = upload::buffer(gpu, initial, Self::indirect_usage(), "unit indirect")?;
        let counts = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: initial.len() as u64, usage: vk::BufferUsageFlags::TRANSFER_DST }, MemoryLocation::GpuToCpu, "unit counts"))
            .collect::<Result<Vec<_>>>()?;
        let palette_usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC;
        let palette = match &model_set {
            Some(set) => {
                let entries = set.palette_layout(capacity).2;
                upload::buffer_sized(gpu, entries * models::PALETTE_ENTRY_BYTES, set.rest_palettes(), palette_usage, "unit palette")?
            }
            None => OwnedBuffer::new(gpu, BufferDesc { size: 64, usage: palette_usage }, MemoryLocation::GpuOnly, "unit palette")?,
        };

        let all = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let (height_view, height_sampler) = terrain.height_texture();
        let (shadow_view, shadow_sampler) = terrain.shadow_texture();
        let (unit_shadow_view, unit_shadow_sampler) = terrain.unit_shadow_map().texture();
        let (tint_view, tint_sampler) = terrain.unit_shadow_map().color_texture();
        // Set 1 of the models: seven storage buffers, the texture array
        // (partially bound), the animation table, the joint keys.
        let model_layout = match &model_set {
            Some(_) => Some(unsafe {
                let storage = |b: u32| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(all);
                let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..7).map(storage).collect();
                bindings.push(vk::DescriptorSetLayoutBinding::default().binding(7).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(models::MAX_TEXTURES).stage_flags(vk::ShaderStageFlags::FRAGMENT));
                bindings.push(storage(8).stage_flags(vk::ShaderStageFlags::COMPUTE));
                bindings.push(storage(9).stage_flags(vk::ShaderStageFlags::COMPUTE));
                let none = vk::DescriptorBindingFlags::empty();
                let flags = [none; 7].into_iter().chain([vk::DescriptorBindingFlags::PARTIALLY_BOUND, none, none]).collect::<Vec<_>>();
                let mut flag_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
                device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings).push_next(&mut flag_info), None)?
            }),
            None => None,
        };
        let (set_layout, pipeline_layout, desc_pool, sets) = unsafe {
            let set_layout = pipeline::set_layout(
                device,
                &[
                    (vk::DescriptorType::UNIFORM_BUFFER, all),
                    (vk::DescriptorType::STORAGE_BUFFER, all),
                    (vk::DescriptorType::STORAGE_BUFFER, all),
                    (vk::DescriptorType::STORAGE_BUFFER, all),
                    (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, all),
                    (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, all),
                    (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, all),
                    (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, all),
                ],
            )?;
            // The caster's push constant: the cascade it draws; with models
            // also the multi-draw's first command (every stage).
            let pipeline_layout = match model_layout {
                Some(models) => pipeline::pipeline_layout(device, &[set_layout, models], 8, all)?,
                None => pipeline::pipeline_layout(device, &[set_layout], 4, vk::ShaderStageFlags::VERTEX)?,
            };
            let n = frames_in_flight as u32;
            let m = model_layout.is_some() as u32;
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: n },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: (3 + 9 * m) * n },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: (4 + models::MAX_TEXTURES * m) * n },
            ];
            let desc_pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n * (1 + m)).pool_sizes(&sizes), None)?;
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
                let shadow = [vk::DescriptorImageInfo { sampler: shadow_sampler, image_view: shadow_view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(5).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&shadow));
                let units = [vk::DescriptorImageInfo { sampler: unit_shadow_sampler, image_view: unit_shadow_view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(6).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&units));
                let tint = [vk::DescriptorImageInfo { sampler: tint_sampler, image_view: tint_view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(7).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&tint));
                device.update_descriptor_sets(&writes, &[]);
            }
            (set_layout, pipeline_layout, desc_pool, sets)
        };
        let cull_code: &[u8] = if model_set.is_some() { spirv!("units_models_cull.comp") } else { spirv!("units_cull.comp") };
        let cull = pipeline::compute_pipeline(device, cull_code, pipeline_layout).context("unit cull pipeline")?;
        let color_formats = [color_format];
        let graphics = |vertex: &[u8], fragment: &[u8], color: bool| {
            pipeline::graphics_pipeline(
                device,
                &GraphicsDesc {
                    vertex,
                    fragment,
                    layout: pipeline_layout,
                    color_formats: if color { &color_formats } else { &[] },
                    depth: Some(match color {
                        true => DepthState { format: depth_format, write: true, compare: vk::CompareOp::GREATER_OR_EQUAL },
                        false => DepthState { format: crate::shadow::format(), write: true, compare: vk::CompareOp::GREATER_OR_EQUAL },
                    }),
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    blend: Blend::Off,
                },
            )
        };
        let draw = graphics(spirv!("units.vert"), spirv!("units.frag"), true).context("unit pipeline")?;
        let casts = terrain.unit_shadow_map().is_real();
        let caster = match casts {
            true => Some(graphics(spirv!("units_shadow.vert"), spirv!("units_shadow.frag"), false).context("unit shadow pipeline")?),
            false => None,
        };
        let models = match (model_set, model_layout) {
            (Some(set), Some(layout)) => Some(Self::models(gpu, set, layout, desc_pool, frames_in_flight, capacity, &palette, pipeline_layout, &graphics, casts)?),
            _ => None,
        };
        Ok(Self {
            capacity,
            models,
            info: *terrain.info(),
            classes,
            min_pixels: setup.min_pixels,
            shadows: terrain.has_shadows(),
            set_layout,
            pipeline_layout,
            desc_pool,
            sets,
            cull,
            draw,
            caster,
            casting: vec![false; frames_in_flight],
            instances,
            uploaded: vec![None; frames_in_flight],
            palette,
            ubos,
            visible,
            indirect,
            counts,
        })
    }

    /// The model side: per-slot buffers, set 1 and the pipelines.
    #[allow(clippy::too_many_arguments)]
    fn models(
        gpu: &GpuDevice,
        set: ModelSet,
        set_layout: vk::DescriptorSetLayout,
        pool: vk::DescriptorPool,
        frames_in_flight: usize,
        capacity: u32,
        palette: &OwnedBuffer,
        pipeline_layout: vk::PipelineLayout,
        graphics: &PipelineMaker<'_>,
        casts: bool,
    ) -> Result<Models> {
        let device = &gpu.device;
        let host = |size: u64, usage: vk::BufferUsageFlags, name: &str| OwnedBuffer::new(gpu, BufferDesc { size, usage }, MemoryLocation::CpuToGpu, name);
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let rows_bytes = 3 * capacity as u64 * 4;
        let rows = (0..frames_in_flight).map(|_| host(rows_bytes, storage, "unit model rows")).collect::<Result<Vec<_>>>()?;
        // Two keys per bone of every unit.
        let keys_bytes = (capacity as u64 * set.max_bones as u64 * 2 * size_of::<Key>() as u64).max(size_of::<Key>() as u64);
        let keys = (0..frames_in_flight).map(|_| host(keys_bytes, storage, "unit joint keys")).collect::<Result<Vec<_>>>()?;
        let template = upload::buffer(gpu, &set.template, vk::BufferUsageFlags::TRANSFER_SRC, "unit indirect template")?;
        let layouts = vec![set_layout; frames_in_flight];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }?;
        for (slot, &ds) in sets.iter().enumerate() {
            let info = |buffer: &OwnedBuffer| [vk::DescriptorBufferInfo { buffer: buffer.buffer, offset: 0, range: vk::WHOLE_SIZE }];
            let geometry = [vk::DescriptorBufferInfo { buffer: set.geometry.buffer, offset: 0, range: set.index_offset.max(4) }];
            let buffers = [geometry, info(&set.materials), info(palette), info(&set.class_table), info(&set.draw_table), info(&set.class_commands), info(&rows[slot])];
            let mut writes: Vec<vk::WriteDescriptorSet> =
                buffers.iter().enumerate().map(|(i, b)| vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(b)).collect();
            let images: Vec<vk::DescriptorImageInfo> =
                set.slots.iter().map(|&(view, sampler)| vk::DescriptorImageInfo { sampler, image_view: view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }).collect();
            writes.push(vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(7).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&images));
            let anim = info(&set.anim);
            writes.push(vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(8).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&anim));
            let keys = info(&keys[slot]);
            writes.push(vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(9).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&keys));
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        let pose = pipeline::compute_pipeline(device, spirv!("units_pose.comp"), pipeline_layout).context("unit pose pipeline")?;
        let opaque = graphics(spirv!("units_models.vert"), spirv!("units_models_opaque.frag"), true).context("unit model pipeline")?;
        let masked = graphics(spirv!("units_models.vert"), spirv!("units_models_masked.frag"), true).context("unit masked model pipeline")?;
        let caster = match casts {
            true => Some((
                graphics(spirv!("units_models_shadow.vert"), spirv!("units_shadow.frag"), false).context("unit model shadow pipeline")?,
                graphics(spirv!("units_models_shadow_masked.vert"), spirv!("units_models_shadow_masked.frag"), false).context("unit masked model shadow pipeline")?,
            )),
            false => None,
        };
        Ok(Models { set, set_layout, sets, rows, keys, jobs: vec![0; frames_in_flight], pose, template, opaque, masked, caster })
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
        let first: Vec<u32> = self.models.as_ref().map_or(Vec::new(), |m| m.set.classes.iter().map(|c| c.commands[0]).collect());
        let Some(bytes) = self.counts[slot].mapped() else { return 0 };
        let word = |w: usize| u32::from_le_bytes(bytes[w * 4..w * 4 + 4].try_into().unwrap());
        // The darts', then each model class's (its first command's).
        word(1) + first.iter().map(|&k| word(4 + 5 * k as usize + 1)).sum::<u32>()
    }

    /// Per model class (in class order), the units the cull found in each
    /// shadow cascade of `slot`'s frame (its first command's instance
    /// count in that cascade's copy; read once the slot's frame finished).
    pub fn read_cascade_counts(&mut self, slot: usize) -> Vec<[u32; crate::shadow::CASCADES]> {
        let Some(m) = &self.models else { return Vec::new() };
        let Some(bytes) = self.counts[slot].mapped() else { return Vec::new() };
        let word = |w: usize| u32::from_le_bytes(bytes[w * 4..w * 4 + 4].try_into().unwrap());
        let total = m.set.commands() as usize;
        m.set.classes.iter().map(|c| std::array::from_fn(|k| word(4 + 5 * ((k + 1) * total + c.commands[0] as usize) + 1))).collect()
    }

    /// Each row's skinning palette in `slot`'s stream as the pose pass left
    /// it (its model's joints, 3x4 rows; empty for a dart). For tests and
    /// tools: it reads the device back (call once the frame completed).
    pub fn read_poses(&mut self, gpu: &GpuDevice, slot: usize) -> Result<Vec<Vec<[f32; 12]>>> {
        let rows = self.rows(slot) as usize;
        let capacity = self.capacity as usize;
        let Some(m) = &mut self.models else { return Ok(vec![Vec::new(); rows]) };
        let palette = upload::read_buffer(gpu, self.palette.buffer, 0, self.palette.size)?;
        let entry = |e: usize| -> [f32; 12] { std::array::from_fn(|k| f32::from_le_bytes(palette[(e * 12 + k) * 4..(e * 12 + k) * 4 + 4].try_into().unwrap())) };
        let word = |bytes: &[u8], w: usize| u32::from_le_bytes(bytes[w * 4..w * 4 + 4].try_into().unwrap());
        let classes: Vec<u32> = {
            let inst = self.instances[slot].mapped().context("instances not host-visible")?;
            (0..rows).map(|r| word(inst, ATTR_CLASS * capacity + r)).collect()
        };
        let offsets = m.rows[slot].mapped().context("rows not host-visible")?;
        let mut out = Vec::with_capacity(rows);
        for (r, &class) in classes.iter().enumerate() {
            let joints = m.set.class_model[(class as usize).min(MAX_CLASSES - 1)].map_or(0, |mc| m.set.classes[mc].joints as usize);
            let at = word(offsets, r) as usize;
            out.push((0..joints).map(|j| entry(at + j)).collect());
        }
        Ok(out)
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

    /// The skinning palettes (`units.pose` writes, the draws read).
    pub fn palette(&self) -> BufferBinding {
        BufferBinding { buffer: self.palette.buffer, offset: 0, size: self.palette.size, identity: IDENTITY_PALETTE }
    }

    pub fn palette_desc(&self) -> BufferDesc {
        BufferDesc { size: self.palette.size, usage: vk::BufferUsageFlags::STORAGE_BUFFER }
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
                    if let Some(m) = &mut self.models {
                        m.write_rows(slot, stream, rows as usize, self.capacity as usize);
                    }
                    self.uploaded[slot] = Some((stream.tick, rows));
                }
            }
            None => self.uploaded[slot] = None,
        }
        let i = &self.info;
        let cascades = frame.unit_shadows.filter(|_| self.caster.is_some());
        self.casting[slot] = cascades.is_some();
        let unit_shadow = crate::shadow::Cascades::uniforms(cascades.as_ref());
        let ubo = FrameUbo {
            view_proj: frame.view_proj.to_cols_array(),
            cam_pos: frame.cam_pos.extend(frame.proj_scale).to_array(),
            frustum: frustum_planes(frame.view_proj),
            map: [i.cell_size, 1.0 / (i.cells.0 + 1) as f32, 1.0 / (i.cells.1 + 1) as f32, alpha],
            height: [i.height_scale, i.height_offset, i.water_level.unwrap_or(0.0), i.water_level.is_some() as u32 as f32],
            sun: crate::terrain::sun::direction().extend(self.min_pixels).to_array(),
            fog: frame.fog,
            counts: [self.rows(slot), self.capacity, (frame.shadows && self.shadows) as u32, self.models.as_ref().map_or(0, |m| m.set.commands())],
            classes: self.classes,
            unit_shadow,
            anim: [stream.map_or(0, |s| s.tick.wrapping_sub(1) as u32), alpha.to_bits(), 0, 0],
        };
        if let Some(bytes) = self.ubos[slot].mapped() {
            bytes[..size_of::<FrameUbo>()].copy_from_slice(pipeline::push_bytes(&ubo));
        }
    }

    /// Zero the instance count. Records inside a transfer pass that
    /// declares [`Self::indirect`] as a transfer destination.
    pub fn record_reset(&self, ctx: &PassContext<'_>) {
        match &self.models {
            // Every command's instance count (and the darts'), from the
            // template.
            Some(m) => unsafe { ctx.device.cmd_copy_buffer(ctx.cmd, m.template.buffer, self.indirect.buffer, &[vk::BufferCopy::default().size(self.indirect.size)]) },
            None => unsafe { ctx.device.cmd_fill_buffer(ctx.cmd, self.indirect.buffer, 4, 4, 0) },
        }
    }

    /// Copy the indirect arguments to `slot`'s host buffer. Records inside a
    /// transfer pass that reads [`Self::indirect`] and writes [`Self::counts`].
    pub fn record_count(&self, ctx: &PassContext<'_>) {
        let size = if self.models.is_some() { self.indirect.size } else { 16 };
        unsafe { ctx.device.cmd_copy_buffer(ctx.cmd, self.indirect.buffer, self.counts[ctx.slot].buffer, &[vk::BufferCopy::default().size(size)]) };
    }

    /// The units' poses. Records inside a compute pass that reads
    /// [`Self::instances`] and writes [`Self::palette`].
    pub fn record_pose(&self, ctx: &PassContext<'_>) {
        let Some(m) = &self.models else { return };
        let jobs = m.jobs[ctx.slot];
        if self.rows(ctx.slot) == 0 || jobs == 0 {
            return;
        }
        let (device, cmd) = (ctx.device, ctx.cmd);
        let all = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, m.pose);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &self.bound_sets(ctx.slot), &[]);
            device.cmd_push_constants(cmd, self.pipeline_layout, all, 0, pipeline::slice_bytes(&[jobs, 0u32]));
            device.cmd_dispatch(cmd, jobs.div_ceil(64), 1, 1);
        }
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
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &self.bound_sets(ctx.slot), &[]);
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
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &self.bound_sets(ctx.slot), &[]);
            device.cmd_draw_indirect(cmd, self.indirect.buffer, 0, 1, 0);
            if let Some(m) = &self.models {
                // Every model command in two multi-draws: opaque, masked.
                device.cmd_bind_index_buffer(cmd, m.set.geometry.buffer, m.set.index_offset, vk::IndexType::UINT32);
                let (opaque, total) = (m.set.opaque, m.set.commands());
                for (pipeline, first, count) in [(m.opaque, 0, opaque), (m.masked, opaque, total - opaque)] {
                    if count == 0 {
                        continue;
                    }
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pipeline::slice_bytes(&[0u32, first]));
                    let offset = models::DART_COMMAND_BYTES + first as u64 * models::COMMAND_BYTES;
                    device.cmd_draw_indexed_indirect(cmd, self.indirect.buffer, offset, count, models::COMMAND_BYTES as u32);
                }
            }
        }
    }

    /// The descriptor sets the pipelines bind: set 0, and with models set 1.
    fn bound_sets(&self, slot: usize) -> Vec<vk::DescriptorSet> {
        let mut sets = vec![self.sets[slot]];
        if let Some(m) = &self.models {
            sets.push(m.sets[slot]);
        }
        sets
    }

    /// The units into each cascade of the units' shadow map: every dart,
    /// and the model units the cull found in the cascade (nothing when the
    /// frame casts no shadows: the map stays cleared). Records inside a
    /// depth-only graphics pass that reads [`Self::instances`],
    /// [`Self::visible`] and [`Self::palette`] in the vertex stage and
    /// [`Self::indirect`] as an indirect buffer, and writes the map.
    pub fn record_shadows(&self, ctx: &PassContext<'_>) {
        let rows = self.rows(ctx.slot);
        let Some(caster) = self.caster.filter(|_| self.casting[ctx.slot] && rows > 0) else { return };
        let (device, cmd) = (ctx.device, ctx.cmd);
        let side = crate::shadow::CASCADE_TEXELS;
        unsafe {
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &self.bound_sets(ctx.slot), &[]);
            let push_stages = match self.models {
                Some(_) => vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                None => vk::ShaderStageFlags::VERTEX,
            };
            let model_caster = self.models.as_ref().and_then(|m| m.caster.map(|c| (m, c))).filter(|(m, _)| m.set.commands() > 0);
            if let Some((m, _)) = model_caster {
                device.cmd_bind_index_buffer(cmd, m.set.geometry.buffer, m.set.index_offset, vk::IndexType::UINT32);
            }
            for k in 0..crate::shadow::CASCADES as u32 {
                let viewport = vk::Viewport { x: (k * side) as f32, y: 0.0, width: side as f32, height: side as f32, min_depth: 0.0, max_depth: 1.0 };
                device.cmd_set_viewport(cmd, 0, &[viewport]);
                device.cmd_set_scissor(cmd, 0, &[vk::Rect2D { offset: vk::Offset2D { x: (k * side) as i32, y: 0 }, extent: vk::Extent2D { width: side, height: side } }]);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, caster);
                if self.models.is_some() {
                    device.cmd_push_constants(cmd, self.pipeline_layout, push_stages, 0, pipeline::slice_bytes(&[k, 0u32]));
                } else {
                    device.cmd_push_constants(cmd, self.pipeline_layout, push_stages, 0, &k.to_le_bytes());
                }
                device.cmd_draw(cmd, UNIT_VERTICES, rows, 0, 0);
                // The model units the cull found in this cascade (the darts'
                // caster skipped model classes), by this cascade's copy of
                // the commands: the opaque ones in one multi-draw, the
                // masked ones (alpha-tested) in another.
                if let Some((m, (opaque_caster, masked_caster))) = model_caster {
                    let (opaque, total) = (m.set.opaque, m.set.commands());
                    let cascade_commands = models::DART_COMMAND_BYTES + (k + 1) as u64 * total as u64 * models::COMMAND_BYTES;
                    for (pipeline, first, count) in [(opaque_caster, 0, opaque), (masked_caster, opaque, total - opaque)] {
                        if count == 0 {
                            continue;
                        }
                        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                        device.cmd_push_constants(cmd, self.pipeline_layout, push_stages, 0, pipeline::slice_bytes(&[k, first]));
                        device.cmd_draw_indexed_indirect(cmd, self.indirect.buffer, cascade_commands + first as u64 * models::COMMAND_BYTES, count, models::COMMAND_BYTES as u32);
                    }
                }
            }
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        let device = &gpu.device;
        if let Some(mut m) = self.models.take() {
            unsafe {
                for p in [m.pose, m.opaque, m.masked].into_iter().chain(m.caster.into_iter().flat_map(|(a, b)| [a, b])) {
                    device.destroy_pipeline(p, None);
                }
                device.destroy_descriptor_set_layout(m.set_layout, None);
            }
            for b in m.rows.iter_mut().chain(m.keys.iter_mut()) {
                b.destroy(gpu);
            }
            m.template.destroy(gpu);
            m.set.destroy(gpu);
        }
        unsafe {
            if let Some(caster) = self.caster.take() {
                device.destroy_pipeline(caster, None);
            }
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
        self.palette.destroy(gpu);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_time_follows_the_sim() {
        let stream = |tick: u64, tick_seconds: f32| InstanceStream { tick, published: Instant::now(), tick_seconds, sim_seconds: 0.1, columns: Default::default(), keys: Vec::new() };
        let (a, b) = (stream(10, 0.05), stream(11, 0.05));
        assert!((a.sim_time(0.0) - 0.9).abs() < 1e-6 && (a.sim_time(1.0) - 1.0).abs() < 1e-6, "a tick behind, by sim seconds");
        assert!(b.sim_time(0.0) >= a.sim_time(1.0), "the next stream never steps back");
        // At 2x (half the wall time per tick) a tick still gains 0.1 s.
        assert_eq!(stream(11, 0.025).sim_time(0.5), stream(11, 0.2).sim_time(0.5));
    }
}
