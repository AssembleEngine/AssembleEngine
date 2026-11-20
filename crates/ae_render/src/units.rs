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
//!   start of a frame, so neither waits for the other. It also keeps the
//!   animation table writes the streams carry ([`AnimWrites`]) until the
//!   renderer takes them, each cell once (its newest value) however many
//!   streams passed; a renderer that missed writes asks the producer for
//!   its whole table.
//! * Positions are interpolated between the previous and the latest tick
//!   (`prev_x/prev_y` → `x/y`) by the time since the stream was published.
//!
//! The passes: `units pose` (`units.pose`, compute: the skinning palettes
//! of the units with animation records into `units.palette`, which the
//! draws read; their models' clips sampled on the GPU),
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The attributes of an instance stream, in column order. Positions are on
/// the ground plane in world units (the nav world's x and y; render world
/// x and z); `heading` need not be normalised (zero: facing +x); `class`
/// is a `u32` indexing [`UnitsSetup::classes`]. `pose` (a `u32`) is where
/// a unit with a model finds its two animation records in the animation
/// table (what the streams' [`InstanceStream::anim`] wrote; see
/// [`RECORD_CELLS`]), with [`POSE_MANUAL`] set when either record shows
/// a manual joint, or [`NO_POSE`] for the rest pose. Every other
/// attribute is an `f32`.
pub const ATTRIBUTES: [&str; 9] = ["x", "y", "prev_x", "prev_y", "heading_x", "heading_y", "radius", "class", "pose"];
pub const ATTRIBUTE_COUNT: usize = ATTRIBUTES.len();
const ATTR_CLASS: usize = 7;
/// The `pose` attribute's column.
pub const ATTR_POSE: usize = 8;
/// The `pose` of a unit drawn at its model's rest pose.
pub const NO_POSE: u32 = u32::MAX;
/// Set in a `pose` other than [`NO_POSE`] when either of the unit's
/// records shows a manual joint: the cull takes the sphere covering any
/// turn (`models::turned_sphere`) from it, not from the table. The rest
/// is the first record's cell ([`pose_cell`]); a table never reaches
/// `1 << 31` cells.
pub const POSE_MANUAL: u32 = 1 << 31;

/// The first record cell of `pose` (not [`NO_POSE`]), without
/// [`POSE_MANUAL`].
pub const fn pose_cell(pose: u32) -> u32 {
    pose & !POSE_MANUAL
}

/// A cell of the animation table: four 32-bit words (a vec4 on the GPU).
pub type Cell = [u32; 4];

/// Cells of one animation record: what a unit's model does at a tick, as
/// the pose needs it (its `ae_model::anim::Channels` less what only moves
/// them on). A posed unit has two from its `pose`, each stamped with its
/// tick; it is drawn between their poses at the drawn tick (`tick - 1 +
/// alpha`; before the older or past the newer, at that one). The cells
/// (`shaders/units_pose.comp` reads them):
/// * 0: (tick, manual entries shown `n`, 0, 0);
/// * 1..=4: per layer (clip `| mask << 16`, mode `| kind << 8`, weight,
///   phase), floats as their bits; clip 0 is an empty layer;
/// * 5, 6: per manual entry `k < n` word `k`: bone `| kind << 16`;
/// * 7..=14: per manual entry `k < n` its value (a quaternion, or xyz).
///
/// The entries are the channels' that show (not empty, not waiting for
/// their start: `Manual::shows`), in order.
pub const RECORD_CELLS: usize = 15;
/// Cells of a posed unit's two records.
pub const POSE_CELLS: usize = 2 * RECORD_CELLS;
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
    /// Writes into the animation table: a posed unit's two records from
    /// its `pose` ([`RECORD_CELLS`]). Every stream is to be published:
    /// each one writes only what changed.
    pub anim: AnimWrites,
}

/// Writes into the animation table: cell `at[i]` becomes `cells[i]`, in
/// order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnimWrites {
    /// A whole table: every cell a reader needs (what came before is
    /// written over; a table it is applied to is emptied first).
    pub reset: bool,
    pub at: Vec<u32>,
    pub cells: Vec<Cell>,
}

impl AnimWrites {
    pub fn push(&mut self, at: u32, cell: Cell) {
        self.at.push(at);
        self.cells.push(cell);
    }

    pub fn len(&self) -> usize {
        self.at.len()
    }

    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }

    /// Onto `table` (grown with zeros as needed).
    pub fn apply(&self, table: &mut Vec<Cell>) {
        if self.reset {
            table.clear();
        }
        for (&at, cell) in self.at.iter().zip(&self.cells) {
            let at = at as usize;
            if at >= table.len() {
                table.resize(at + 1, [0; 4]);
            }
            table[at] = *cell;
        }
    }

    /// The whole of `table`, from empty.
    pub fn whole(table: &[Cell]) -> Self {
        Self { reset: true, at: (0..table.len() as u32).collect(), cells: table.to_vec() }
    }
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

/// The latest stream, shared by its producer and the render thread, and
/// the animation table writes its reader (the renderer) has not taken yet.
#[derive(Default)]
pub struct UnitFeed {
    state: Mutex<FeedState>,
}

/// Write batches kept for a reader that does not take them: past this,
/// they are dropped and the producer asked for the whole table.
const MAX_PENDING: usize = 64;

#[derive(Default)]
struct FeedState {
    latest: Option<Arc<InstanceStream>>,
    /// Writes since the last take, in order.
    pending: Vec<AnimWrites>,
    /// Batches taken, for their memory.
    pool: Vec<AnimWrites>,
    /// A bit per cell staged in a take (scratch: the newest write wins).
    marked: Vec<u64>,
    /// Takes so far: a reader whose last take is not the latest one has
    /// missed writes.
    takes: u64,
    /// A reader missed writes and no whole table is pending.
    want_whole: bool,
}

impl FeedState {
    fn push(&mut self, writes: AnimWrites) {
        if writes.reset {
            // Older batches are written over.
            self.pool.append(&mut self.pending);
        } else if writes.is_empty() {
            self.pool.push(writes);
            return;
        }
        self.pending.push(writes);
        if self.pending.len() > MAX_PENDING {
            self.pool.append(&mut self.pending);
            self.want_whole = true;
        }
    }
}

impl UnitFeed {
    /// Publish `stream`; its table writes are kept for the reader, and
    /// memory from batches already taken is handed back for the
    /// producer's next stream (its contents stale; the published stream
    /// has no writes).
    pub fn publish_reusing(&self, mut stream: InstanceStream) -> AnimWrites {
        let writes = std::mem::take(&mut stream.anim);
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        s.push(writes);
        s.latest = Some(Arc::new(stream));
        s.pool.pop().unwrap_or_default()
    }

    /// Publish `stream` (its table writes copied for the reader).
    pub fn publish(&self, stream: Arc<InstanceStream>) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        s.push(stream.anim.clone());
        s.latest = Some(stream);
    }

    pub fn latest(&self) -> Option<Arc<InstanceStream>> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).latest.clone()
    }

    /// A reader missed table writes since the last whole table: the
    /// producer is to write its whole table next (`AnimWrites::reset`).
    /// Cleared by asking.
    pub fn wants_whole(&self) -> bool {
        std::mem::take(&mut self.state.lock().unwrap_or_else(|p| p.into_inner()).want_whole)
    }

    /// The latest stream, and the table cells its reader lacks: `write(at,
    /// cell)` once for each cell written since the reader's last take
    /// (`seen`, 0 at first), its newest value, in no particular order.
    /// A reader that missed writes (its first take, a frame dropped,
    /// another reader's take between) gets the cells from the last whole
    /// table pending, or none and [`Self::wants_whole`] is set (its table
    /// is stale until the whole one comes). Under the feed's lock, so the
    /// cells are the stream's. One table reader per feed.
    pub fn take(&self, seen: &mut u64, mut write: impl FnMut(u32, &Cell)) -> Option<Arc<InstanceStream>> {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let s = &mut *s;
        let current = *seen != 0 && *seen == s.takes;
        let from = match current {
            true => Some(0),
            false => s.pending.iter().rposition(|w| w.reset),
        };
        if let Some(from) = from {
            for w in s.pending[from..].iter().rev() {
                for (&at, cell) in w.at.iter().zip(&w.cells).rev() {
                    let i = at as usize;
                    if i / 64 >= s.marked.len() {
                        s.marked.resize(i / 64 + 1, 0);
                    }
                    if s.marked[i / 64] & (1 << (i % 64)) == 0 {
                        s.marked[i / 64] |= 1 << (i % 64);
                        write(at, cell);
                    }
                }
            }
            for w in &s.pending[from..] {
                for &at in &w.at {
                    s.marked[at as usize / 64] = 0;
                }
            }
            s.takes += 1;
            *seen = s.takes;
        } else {
            s.want_whole = true;
            *seen = 0;
        }
        s.pool.append(&mut s.pending);
        s.latest.clone()
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
    /// The rig each model is posed by (`rigs[class]`: the sim's, with its
    /// masks); none or beyond the list: one made from the model.
    pub rigs: Vec<Option<Arc<ae_model::anim::Rig>>>,
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
    /// Photometric lighting: exposure and the sun's and sky's illuminances
    /// in lux (y unused).
    pub photometry: [f32; 4],
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
    /// past it (`f32` bits; poses are drawn there), z: animation table
    /// writes this frame.
    anim: [u32; 4],
    /// x: exposure, y: the legacy emission scale, z: sun lux, w: sky lux.
    photometry: [f32; 4],
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
    /// Per slot: pose offsets, the pose jobs (rows with animation records,
    /// their scratch offsets).
    rows: Vec<OwnedBuffer>,
    /// The animation table (`units_anim_writes.comp` writes it, the poses
    /// read it), `table_cells` cells: one for all slots, written in queue
    /// order.
    table: OwnedBuffer,
    table_cells: usize,
    /// Per slot: the frame's table writes (cells, their places),
    /// `writes` of them.
    staged: Vec<OwnedBuffer>,
    staged_at: Vec<OwnedBuffer>,
    writes: Vec<u32>,
    /// Per slot: writes staged and not yet recorded (a frame dropped
    /// after `write_frame`: the next take is of the whole table).
    unrecorded: Vec<AtomicBool>,
    /// The last take from the feed ([`UnitFeed::take`]), and its cells and
    /// milliseconds (staging them included).
    seen: u64,
    last_take: (u32, f64),
    /// Per slot: pose jobs in `rows`.
    jobs: Vec<u32>,
    pose: vk::Pipeline,
    scatter: vk::Pipeline,
    /// Per slot: the offset of each unit's model lights in the frame's
    /// light list (the host writes it with the rows;
    /// `units_lights.comp` reads it), and how many there are in all.
    light_offsets: Vec<OwnedBuffer>,
    light_totals: Vec<u32>,
    lights_append: vk::Pipeline,
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
    /// Set 1 of the layout without models.
    empty_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    desc_pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    /// The clustered lights' sets (set 2 of the draws; crate::lights).
    light_sets: Vec<vk::DescriptorSet>,
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
    /// The stream from `feed`, and the table writes `slot`'s frame makes
    /// (what the table lacks).
    fn take(&mut self, slot: usize, feed: &UnitFeed) -> Option<Arc<InstanceStream>> {
        if self.unrecorded.iter().any(|f| f.swap(false, Ordering::Relaxed)) {
            self.seen = 0;
        }
        let started = Instant::now();
        let cap = self.table_cells;
        let (mut n, mut cells, mut at) = (0usize, self.staged[slot].mapped(), self.staged_at[slot].mapped());
        let stream = feed.take(&mut self.seen, |i, cell| {
            if let (true, Some(cells), Some(at)) = ((i as usize) < cap, cells.as_deref_mut(), at.as_deref_mut()) {
                cells[n * size_of::<Cell>()..(n + 1) * size_of::<Cell>()].copy_from_slice(pipeline::slice_bytes(cell));
                at[n * 4..n * 4 + 4].copy_from_slice(&i.to_ne_bytes());
                n += 1;
            }
        });
        self.writes[slot] = n as u32;
        self.unrecorded[slot].store(n > 0, Ordering::Relaxed);
        self.last_take = (n as u32, started.elapsed().as_secs_f64() * 1e3);
        stream
    }

    /// `slot`'s rows: each row's palette offset (its model's rest palette,
    /// or a range of its own when it has records in the table: a pose
    /// job), the pose jobs (rows with records, their scratch offsets).
    fn write_rows(&mut self, slot: usize, stream: &InstanceStream, rows: usize, capacity: usize) {
        let class_of = |r: usize| (stream.columns[ATTR_CLASS][r] as usize).min(MAX_CLASSES - 1);
        let mut words = vec![0u32; 3 * capacity];
        // Model light offsets, in row order (the appended lights keep the
        // list's order deterministic; the host has the rows already).
        let mut offsets = vec![0u32; capacity];
        let mut lights = 0u32;
        let (ranges, scratch, _) = self.set.palette_layout(capacity as u32);
        let (mut range, mut scratch, mut jobs) = (ranges as u32, scratch as u32, 0);
        let fit = self.table_cells;
        for r in 0..rows {
            let n = self.set.class_lights[class_of(r)][1];
            if n > 0 {
                offsets[r] = lights;
                lights = lights.saturating_add(n);
            }
            let Some(mc) = self.set.class_model[class_of(r)] else { continue };
            let mc = &self.set.classes[mc];
            let pose = stream.columns[ATTR_POSE][r];
            words[r] = match mc.bones > 0 && pose != NO_POSE && pose_cell(pose) as usize + POSE_CELLS <= fit {
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
        self.light_totals[slot] = lights;
        if let Some(bytes) = self.rows[slot].mapped() {
            bytes[..words.len() * 4].copy_from_slice(pipeline::slice_bytes(&words));
        }
        if let Some(bytes) = self.light_offsets[slot].mapped() {
            bytes[..offsets.len() * 4].copy_from_slice(pipeline::slice_bytes(&offsets));
        }
    }
}

impl UnitRenderer {
    /// Units over `terrain` (its height texture places them), drawn into a
    /// pass with a `color_format` target and a `depth_format` reverse-Z
    /// depth buffer.
    pub fn new(gpu: &GpuDevice, terrain: &TerrainRenderer, lights: &crate::lights::LightRenderer, setup: &UnitsSetup, color_format: vk::Format, depth_format: vk::Format, frames_in_flight: usize) -> Result<Self> {
        ensure!(setup.capacity > 0, "unit capacity 0");
        let device = &gpu.device;
        let capacity = setup.capacity;
        let model_set = match setup.models.iter().any(Option::is_some) {
            true => Some(ModelSet::new(gpu, &setup.models, &setup.rigs, capacity, MAX_CLASSES).context("unit models")?),
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
        // (partially bound), the models' animation table, the units'
        // animation table, the frame's writes to it and their places, and
        // the models' lights (their tables and the frame's offsets).
        let model_layout = match &model_set {
            Some(_) => Some(unsafe {
                let storage = |b: u32| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(all);
                let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..7).map(storage).collect();
                bindings.push(vk::DescriptorSetLayoutBinding::default().binding(7).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(models::MAX_TEXTURES).stage_flags(vk::ShaderStageFlags::FRAGMENT));
                bindings.push(storage(8).stage_flags(vk::ShaderStageFlags::COMPUTE));
                for b in 9..15 {
                    bindings.push(storage(b).stage_flags(vk::ShaderStageFlags::COMPUTE));
                }
                let none = vk::DescriptorBindingFlags::empty();
                let flags = [none; 7].into_iter().chain([vk::DescriptorBindingFlags::PARTIALLY_BOUND, none, none, none, none, none, none, none]).collect::<Vec<_>>();
                let mut flag_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
                device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings).push_next(&mut flag_info), None)?
            }),
            None => None,
        };
        let (set_layout, empty_layout, pipeline_layout, desc_pool, sets) = unsafe {
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
            // Set 2: the clustered lights (the draws' fragments read them;
            // an empty set 1 without models).
            let empty_layout = pipeline::set_layout(device, &[])?;
            let pipeline_layout = match model_layout {
                Some(models) => pipeline::pipeline_layout(device, &[set_layout, models, lights.set_layout()], 8, all)?,
                None => pipeline::pipeline_layout(device, &[set_layout, empty_layout, lights.set_layout()], 4, vk::ShaderStageFlags::VERTEX)?,
            };
            let n = frames_in_flight as u32;
            let m = model_layout.is_some() as u32;
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: n },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: (3 + 14 * m) * n },
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
            (set_layout, empty_layout, pipeline_layout, desc_pool, sets)
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
            empty_layout,
            pipeline_layout,
            desc_pool,
            sets,
            light_sets: lights.sets(),
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
        // Two records per unit; a frame may write all of them.
        let table_cells = capacity as usize * POSE_CELLS;
        let table = OwnedBuffer::new(gpu, BufferDesc { size: (table_cells * size_of::<Cell>()) as u64, usage: storage }, MemoryLocation::GpuOnly, "unit animation table")?;
        let staged = (0..frames_in_flight).map(|_| host((table_cells * size_of::<Cell>()) as u64, storage, "unit animation writes")).collect::<Result<Vec<_>>>()?;
        let staged_at = (0..frames_in_flight).map(|_| host(table_cells as u64 * 4, storage, "unit animation write places")).collect::<Result<Vec<_>>>()?;
        let light_offsets = (0..frames_in_flight)
            .map(|_| OwnedBuffer::new(gpu, BufferDesc { size: capacity as u64 * 4, usage: storage }, MemoryLocation::CpuToGpu, "unit light offsets"))
            .collect::<Result<Vec<_>>>()?;
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
            let cells = [info(&table), info(&staged[slot]), info(&staged_at[slot])];
            for (b, k) in cells.iter().enumerate() {
                writes.push(vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(9 + b as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(k));
            }
            let lights = [info(&set.class_light_table), info(&set.model_light_table), info(&light_offsets[slot])];
            for (b, k) in lights.iter().enumerate() {
                writes.push(vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(12 + b as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(k));
            }
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        let pose = pipeline::compute_pipeline(device, spirv!("units_pose.comp"), pipeline_layout).context("unit pose pipeline")?;
        let scatter = pipeline::compute_pipeline(device, spirv!("units_anim_writes.comp"), pipeline_layout).context("unit animation write pipeline")?;
        let lights_append = pipeline::compute_pipeline(device, spirv!("units_lights.comp"), pipeline_layout).context("unit light append pipeline")?;
        let opaque = graphics(spirv!("units_models.vert"), spirv!("units_models_opaque.frag"), true).context("unit model pipeline")?;
        let masked = graphics(spirv!("units_models.vert"), spirv!("units_models_masked.frag"), true).context("unit masked model pipeline")?;
        let caster = match casts {
            true => Some((
                graphics(spirv!("units_models_shadow.vert"), spirv!("units_shadow.frag"), false).context("unit model shadow pipeline")?,
                graphics(spirv!("units_models_shadow_masked.vert"), spirv!("units_models_shadow_masked.frag"), false).context("unit masked model shadow pipeline")?,
            )),
            false => None,
        };
        Ok(Models {
            set,
            set_layout,
            sets,
            rows,
            table,
            table_cells,
            staged,
            staged_at,
            writes: vec![0; frames_in_flight],
            unrecorded: (0..frames_in_flight).map(|_| AtomicBool::new(false)).collect(),
            seen: 0,
            last_take: (0, 0.0),
            jobs: vec![0; frames_in_flight],
            pose,
            scatter,
            light_offsets,
            light_totals: vec![0; frames_in_flight],
            lights_append,
            template,
            opaque,
            masked,
            caster,
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

    /// Cells the last [`Self::write_frame`] staged for the animation
    /// table, and its milliseconds doing it (under the feed's lock).
    pub fn anim_upload(&self) -> (u32, f64) {
        self.models.as_ref().map_or((0, 0.0), |m| m.last_take)
    }

    /// Rows of `slot`'s stream (after [`Self::write_frame`]).
    pub fn rows(&self, slot: usize) -> u32 {
        self.uploaded[slot].map_or(0, |(_, rows)| rows)
    }

    /// Uniforms for `slot`, the latest stream's columns from `feed` when it
    /// is newer than the slot's copy, and the cells the animation table
    /// lacks (the slot's previous submission has completed).
    pub fn write_frame(&mut self, slot: usize, frame: &UnitFrame, feed: &UnitFeed, now: Instant) {
        let stream = match &mut self.models {
            Some(m) => m.take(slot, feed),
            None => feed.latest(),
        };
        let stream = stream.as_deref();
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
            anim: [stream.map_or(0, |s| s.tick.wrapping_sub(1) as u32), alpha.to_bits(), self.models.as_ref().map_or(0, |m| m.writes[slot]), 0],
            photometry: frame.photometry,
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

    /// The frame's writes into the animation table, then the units'
    /// poses. Records inside a compute pass that reads [`Self::instances`]
    /// and writes [`Self::palette`] (the animation table is the renderer's
    /// own: its barriers are here).
    pub fn record_pose(&self, ctx: &PassContext<'_>) {
        let Some(m) = &self.models else { return };
        let (device, cmd) = (ctx.device, ctx.cmd);
        let writes = m.writes[ctx.slot];
        if writes > 0 {
            let barrier = |src: vk::AccessFlags, dst: vk::AccessFlags| unsafe {
                let b = [vk::MemoryBarrier::default().src_access_mask(src).dst_access_mask(dst)];
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &b, &[], &[]);
            };
            // After the earlier frames' poses read the table (queue order).
            barrier(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_WRITE);
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, m.scatter);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &self.bound_sets(ctx.slot), &[]);
                device.cmd_dispatch(cmd, writes.div_ceil(64), 1, 1);
            }
            barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ);
            m.unrecorded[ctx.slot].store(false, Ordering::Relaxed);
        }
        let jobs = m.jobs[ctx.slot];
        if self.rows(ctx.slot) == 0 || jobs == 0 {
            return;
        }
        let all = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, m.pose);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &self.bound_sets(ctx.slot), &[]);
            device.cmd_push_constants(cmd, self.pipeline_layout, all, 0, pipeline::slice_bytes(&[jobs, 0u32]));
            device.cmd_dispatch(cmd, jobs.div_ceil(64), 1, 1);
        }
    }

    /// Whether any unit class's model has lights (the lights pass appends
    /// them, even with no host light).
    pub fn has_model_lights(&self) -> bool {
        self.models.as_ref().is_some_and(|m| m.set.light_count > 0)
    }

    /// `slot`'s model lights in all (one per unit of a class whose model
    /// has any), as its stream's rows were written.
    pub fn model_light_total(&self, slot: usize) -> u32 {
        self.models.as_ref().map_or(0, |m| m.light_totals[slot])
    }

    /// This frame's model lights, appended to the frame's light list: one
    /// per unit of a class whose model has any, at the pose the units are
    /// drawn at. The `lights.cluster` pass calls this between its copy and
    /// its bounds; it records one dispatch, reading [`Self::instances`] and
    /// [`Self::palette`] (its offsets by row are host-written with the
    /// rows) and writing the light list. Returns whether any were
    /// appended.
    pub fn record_lights(&self, ctx: &PassContext<'_>) -> bool {
        let Some(m) = &self.models else { return false };
        let rows = self.rows(ctx.slot);
        if rows == 0 || m.set.light_count == 0 || m.light_totals[ctx.slot] == 0 {
            return false;
        }
        let (device, cmd) = (ctx.device, ctx.cmd);
        let mut sets = self.bound_sets(ctx.slot);
        sets.push(self.light_sets[ctx.slot]);
        unsafe {
            // The pose pass's palette writes (an earlier pass) to the
            // append's reads.
            let memory = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)];
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&memory));
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &sets, &[]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, m.lights_append);
            device.cmd_dispatch(cmd, rows.div_ceil(64), 1, 1);
        }
        true
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
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 2, &[self.light_sets[ctx.slot]], &[]);
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
                for p in [m.pose, m.scatter, m.opaque, m.masked, m.lights_append].into_iter().chain(m.caster.into_iter().flat_map(|(a, b)| [a, b])) {
                    device.destroy_pipeline(p, None);
                }
                device.destroy_descriptor_set_layout(m.set_layout, None);
            }
            for b in m.rows.iter_mut().chain(&mut m.staged).chain(&mut m.staged_at).chain(&mut m.light_offsets) {
                b.destroy(gpu);
            }
            m.table.destroy(gpu);
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
            device.destroy_descriptor_set_layout(self.empty_layout, None);
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
        let stream = |tick: u64, tick_seconds: f32| InstanceStream { tick, published: Instant::now(), tick_seconds, sim_seconds: 0.1, columns: Default::default(), anim: AnimWrites::default() };
        let (a, b) = (stream(10, 0.05), stream(11, 0.05));
        assert!((a.sim_time(0.0) - 0.9).abs() < 1e-6 && (a.sim_time(1.0) - 1.0).abs() < 1e-6, "a tick behind, by sim seconds");
        assert!(b.sim_time(0.0) >= a.sim_time(1.0), "the next stream never steps back");
        // At 2x (half the wall time per tick) a tick still gains 0.1 s.
        assert_eq!(stream(11, 0.025).sim_time(0.5), stream(11, 0.2).sim_time(0.5));
    }

    /// The feed keeps the streams' table writes for its reader: each cell
    /// once, its newest value, from the last whole table on at first; a
    /// reader that missed writes gets none and the producer is asked for
    /// a whole table.
    #[test]
    fn feed_hands_out_table_writes() {
        let k = |v: u32| [v; 4];
        let stream = |anim: AnimWrites| InstanceStream { tick: 1, published: Instant::now(), tick_seconds: 0.0, sim_seconds: 0.1, columns: Default::default(), anim };
        let writes = |w: &[(u32, u32)]| {
            let mut keys = AnimWrites::default();
            for &(at, v) in w {
                keys.push(at, k(v));
            }
            keys
        };
        let feed = UnitFeed::default();
        let take = |seen: &mut u64| {
            let mut got = Vec::new();
            feed.take(seen, |at, key| got.push((at, key[0])));
            got.sort();
            got
        };
        let (mut a, mut b) = (0, 0);
        feed.publish_reusing(stream(writes(&[(0, 1)])));
        assert_eq!(take(&mut a), [], "no whole table yet");
        assert!(feed.wants_whole() && !feed.wants_whole(), "asked once");
        let mut whole = AnimWrites::whole(&[k(1), k(0), k(20)]);
        whole.reset = true;
        feed.publish_reusing(stream(whole));
        assert_eq!(take(&mut a), [(0, 1), (1, 0), (2, 20)], "from the whole table");
        assert_eq!(take(&mut a), [], "nothing new");
        let spare = feed.publish_reusing(stream(writes(&[(1, 10), (2, 21)])));
        assert!(spare.at.capacity() >= 3, "the taken batch's memory back");
        feed.publish_reusing(stream(writes(&[(1, 11), (4, 40), (1, 12)])));
        assert_eq!(take(&mut a), [(1, 12), (2, 21), (4, 40)], "coalesced, the newest write each");
        feed.publish_reusing(stream(writes(&[(0, 2)])));
        assert_eq!(take(&mut b), [], "another reader missed writes");
        assert!(feed.wants_whole());
        assert_eq!(take(&mut a), [], "a take came between");
        feed.publish_reusing(stream(AnimWrites::whole(&[k(7), k(8)])));
        feed.publish_reusing(stream(writes(&[(1, 9)])));
        assert_eq!(take(&mut a), [(0, 7), (1, 9)], "the whole table and what came after");
        // Too many batches for a reader that does not take: dropped.
        for v in 0..=MAX_PENDING as u32 {
            feed.publish_reusing(stream(writes(&[(0, v)])));
        }
        assert!(feed.wants_whole());
        let mut table = vec![k(9); 6];
        AnimWrites::whole(&[k(7), k(8)]).apply(&mut table);
        assert_eq!(table, [k(7), k(8)]);
    }
}
