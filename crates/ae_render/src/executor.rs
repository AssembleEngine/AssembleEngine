//! Frame-graph executor (§3.5): realizes the baked plan (one physical set per
//! frame slot), records every frame into one primary command buffer on the
//! single graphics queue (or one per pass, recorded on several threads, see
//! [`Executor::execute`]), and derives barriers from the access state machine
//! in `frame_graph::barriers`. The executor owns the baked plan and all
//! execution state; declarations hold none.
//!
//! Frame protocol (render thread):
//! `begin_frame` (waits the slot's fence) → caller acquires a swapchain
//! image with `image_available(slot)` → `execute` (records, submits with the
//! caller's waits/signals, fences the slot) → caller presents.

use crate::bindless::{Bindless, BufferHandle, ImageHandle, SampledKind, StorageImageHandle};
use crate::device::GpuDevice;
use crate::memory::{OwnedBuffer, OwnedImage};
use ae_framegraph::barriers::{AccessState, BarrierDesc, StateKey, StateStore, pass_barriers};
use ae_framegraph::{
    BakeOptions, BakedGraph, FrameGraph, LoadOp, PassId, PassKind, ResourceDesc, ResourceId, Role, bake,
};
use anyhow::{Context as _, Result, anyhow, bail};
use ash::vk;
use gpu_allocator::MemoryLocation;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct ImageBinding {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub range: vk::ImageSubresourceRange,
    /// Unique per bound object; keys the access state of external images
    /// (e.g. one state per swapchain image).
    pub identity: u64,
    /// The global sampled-image index, when the resource has one.
    pub sampled: Option<u32>,
    /// The global storage-image index, when the resource has one.
    pub storage: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct BufferBinding {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
    pub identity: u64,
    /// The global storage-buffer index, when the resource has one.
    pub storage: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub enum Binding {
    Image(ImageBinding),
    Buffer(BufferBinding),
}

impl Binding {
    fn identity(&self) -> u64 {
        match self {
            Binding::Image(image) => image.identity,
            Binding::Buffer(buffer) => buffer.identity,
        }
    }
}

/// The graph handles a resource keeps across bakes, keyed by name.
#[derive(Clone, Copy, Debug, Default)]
struct GraphHandles {
    sampled: Option<ImageHandle>,
    storage: Option<StorageImageHandle>,
    buffer: Option<BufferHandle>,
}

/// What a pass may see while recording: its command buffer, its render
/// area and the resources bound this frame. No renderer, no world.
pub struct PassContext<'a> {
    pub device: &'a ash::Device,
    pub cmd: vk::CommandBuffer,
    pub slot: usize,
    pub pass: PassId,
    /// Graphics passes: the attachment extent.
    pub render_area: vk::Rect2D,
    bindings: &'a [Option<Binding>],
}

impl PassContext<'_> {
    pub fn image(&self, resource: ResourceId) -> Result<ImageBinding> {
        match self.bindings.get(resource.index()).copied().flatten() {
            Some(Binding::Image(image)) => Ok(image),
            _ => Err(anyhow!("resource #{} is not a bound image", resource.0)),
        }
    }

    pub fn buffer(&self, resource: ResourceId) -> Result<BufferBinding> {
        match self.bindings.get(resource.index()).copied().flatten() {
            Some(Binding::Buffer(buffer)) => Ok(buffer),
            _ => Err(anyhow!("resource #{} is not a bound buffer", resource.0)),
        }
    }
}

/// Records pass payloads `R`. Implemented by each renderer (demo, view
/// renderer); the executor only drives barriers and rendering scopes.
pub trait Recorder<R> {
    /// Skip a conditional pass this frame.
    fn skip(&self, _pass: &R) -> bool {
        false
    }

    fn clear_color(&self, _pass: &R, _resource: ResourceId) -> [f32; 4] {
        [0.0, 0.0, 0.0, 1.0]
    }

    /// Depth and stencil a clearing depth attachment starts from (default
    /// 1.0, the far plane of a conventional depth range; reversed-Z
    /// recorders return 0.0).
    fn clear_depth(&self, _pass: &R, _resource: ResourceId) -> (f32, u32) {
        (1.0, 0)
    }

    fn record(&mut self, pass: &R, ctx: &PassContext<'_>) -> Result<()>;

    /// A view of this recorder that records from several threads at once,
    /// for one frame. Recorders that have one let the executor record passes
    /// in parallel (`ExecutorConfig::record_threads`); the view must record
    /// exactly what [`Recorder::record`] would.
    fn shared(&self) -> Option<Box<dyn SharedRecorder<R> + '_>> {
        None
    }
}

/// Thread-safe recording of pass payloads (see [`Recorder::shared`]).
pub trait SharedRecorder<R>: Sync {
    fn record(&self, pass: &R, ctx: &PassContext<'_>) -> Result<()>;
}

#[derive(Clone, Copy, Debug)]
pub struct ExecutorConfig {
    /// Frames in flight: data, not a constant baked into allocation sizes.
    pub frames_in_flight: usize,
    /// GPU timestamps per pass.
    pub profile: bool,
    pub bake: BakeOptions,
    /// Threads that record passes (one command buffer per pass) when the
    /// recorder supports it; 1 records every pass on the calling thread.
    pub record_threads: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
            record_threads: 1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExecutorStats {
    pub bakes: u64,
    pub frames: u64,
    pub barriers: u64,
    pub skipped_passes: u64,
    /// Frames whose passes were recorded on the recording threads.
    pub parallel_frames: u64,
    pub physical_resources: usize,
    pub estimated_bytes: u64,
    /// CPU phases of the last frame.
    pub wait_ms: f32,
    pub record_ms: f32,
    pub submit_ms: f32,
    /// Average GPU time per baked pass (profiling only), baked order.
    pub pass_gpu_ms: Vec<f32>,
    /// Names of the baked passes, baked order (as `pass_gpu_ms`).
    pub pass_names: Vec<String>,
    /// Descriptor writes the bindless registry has applied.
    pub bindless_writes: u64,
    /// Live bindless entries, binding order (sampled images, samplers,
    /// storage images, storage buffers).
    pub bindless_live: [u32; 4],
}

struct Physical {
    image: Option<OwnedImage>,
    buffer: Option<OwnedBuffer>,
}

struct Slot {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    image_available: vk::Semaphore,
    physical: Vec<Physical>,
    /// Access states of this slot's physical resources.
    states: BTreeMap<StateKey, AccessState>,
    queries: vk::QueryPool,
    query_capacity: u32,
    /// Timestamps written by the last submit of this slot.
    pending_queries: u32,
    /// With recording threads: a pool and a primary command buffer per
    /// baked pass (a pool is used by one thread at a time).
    pass_cmds: Vec<(vk::CommandPool, vk::CommandBuffer)>,
}

/// A slot whose fence has been waited; valid until `execute`.
#[derive(Clone, Copy, Debug)]
pub struct FrameToken {
    pub slot: usize,
}

struct Store<'a> {
    slot: &'a mut BTreeMap<StateKey, AccessState>,
    external: &'a mut BTreeMap<u64, AccessState>,
    bindings: &'a [Option<Binding>],
}

impl StateStore for Store<'_> {
    fn state(&mut self, key: StateKey) -> &mut AccessState {
        match key {
            StateKey::Physical(_) => self.slot.entry(key).or_default(),
            StateKey::External(resource) => {
                let identity = self.bindings[resource.index()]
                    .map(|b| b.identity())
                    .unwrap_or(u64::MAX);
                self.external.entry(identity).or_default()
            }
        }
    }
}

pub struct Executor<R> {
    gpu: Arc<GpuDevice>,
    config: ExecutorConfig,
    graph: FrameGraph<R>,
    baked: Option<BakedGraph>,
    slots: Vec<Slot>,
    bindless: Bindless,
    graph_handles: BTreeMap<String, GraphHandles>,
    external_states: BTreeMap<u64, AccessState>,
    frame: u64,
    stats: ExecutorStats,
    gpu_accum: Vec<(f64, u64)>,
    /// Per-frame GPU pass times (ms, baked order), oldest first, while
    /// profiling; capped at [`MAX_GPU_FRAMES`] and drained by
    /// [`Executor::take_gpu_frames`].
    gpu_frames: Vec<Vec<f32>>,
    /// The recording threads (`record_threads > 1`).
    pool: Option<rayon::ThreadPool>,
}

/// Frames of per-pass GPU times kept until taken (older ones are dropped).
pub const MAX_GPU_FRAMES: usize = 16_384;

impl<R> Executor<R> {
    pub fn new(gpu: Arc<GpuDevice>, config: ExecutorConfig) -> Result<Self> {
        if config.frames_in_flight == 0 {
            bail!("frames_in_flight must be positive");
        }
        let device = &gpu.device;
        let mut slots = Vec::with_capacity(config.frames_in_flight);
        for index in 0..config.frames_in_flight {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(gpu.queue_family)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let pool = unsafe { device.create_command_pool(&pool_info, None) }?;
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe { device.allocate_command_buffers(&alloc) }?[0];
            let fence = unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            }?;
            let image_available = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
            gpu.set_name(cmd, &format!("frame slot {index}"));
            slots.push(Slot {
                pool,
                cmd,
                fence,
                image_available,
                physical: Vec::new(),
                states: BTreeMap::new(),
                queries: vk::QueryPool::null(),
                query_capacity: 0,
                pending_queries: 0,
                pass_cmds: Vec::new(),
            });
        }
        let bindless = Bindless::new(&gpu, config.frames_in_flight)?;
        Ok(Self {
            gpu,
            config,
            graph: FrameGraph::new(),
            baked: None,
            slots,
            bindless,
            graph_handles: BTreeMap::new(),
            external_states: BTreeMap::new(),
            frame: 0,
            stats: ExecutorStats::default(),
            gpu_accum: Vec::new(),
            gpu_frames: Vec::new(),
            pool: match config.record_threads {
                0 | 1 => None,
                n => Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(n)
                        .thread_name(|i| format!("ae-record-{i}"))
                        .build()?,
                ),
            },
        })
    }

    pub fn frames_in_flight(&self) -> usize {
        self.config.frames_in_flight
    }

    pub fn graph(&self) -> &FrameGraph<R> {
        &self.graph
    }

    pub fn baked(&self) -> Option<&BakedGraph> {
        self.baked.as_ref()
    }

    pub fn stats(&self) -> ExecutorStats {
        let mut stats = self.stats.clone();
        stats.pass_gpu_ms = self
            .gpu_accum
            .iter()
            .map(|(sum, count)| if *count > 0 { (*sum / *count as f64) as f32 } else { 0.0 })
            .collect();
        stats.pass_names = self
            .baked
            .iter()
            .flat_map(|b| b.order())
            .map(|p| self.graph.passes[p.0 as usize].name.clone())
            .collect();
        stats.bindless_live = self.bindless.live();
        stats
    }

    /// The global descriptor set's registry.
    pub fn bindless(&self) -> &Bindless {
        &self.bindless
    }

    pub fn bindless_mut(&mut self) -> &mut Bindless {
        &mut self.bindless
    }

    /// Install the declarations for the next frames. Rebakes and
    /// re-realizes only when the topology signature changed; otherwise only
    /// the per-frame payloads are replaced. Returns whether it rebaked.
    /// Forget every external resource's access state (its last layout and
    /// accesses). Call it when the app has replaced the objects it binds
    /// (a rebuild after the device went idle): a new image under a reused
    /// identity starts undefined, instead of inheriting the old image's
    /// layout.
    pub fn forget_external_states(&mut self) {
        self.external_states.clear();
    }

    pub fn set_graph(&mut self, graph: FrameGraph<R>) -> Result<bool> {
        let signature = graph.signature();
        if let Some(baked) = self.baked.clone()
            && baked.signature == signature
        {
            // Names are not part of the signature, so a renamed resource
            // still needs its handles re-keyed even though nothing rebaked.
            // Direct descriptor writes need an idle device, which a rebake
            // waits for anyway.
            self.gpu.wait_idle();
            self.sync_graph_handles(&graph, &baked)?;
            self.graph = graph;
            return Ok(false);
        }
        let baked = bake(&graph, self.config.bake).map_err(|err| anyhow!("frame graph bake: {err}"))?;
        // Topology changes are rare (resize, output toggled): wait for the
        // GPU and rebuild the physical resources from scratch.
        self.gpu.wait_idle();
        self.release_physical();
        for (slot_index, slot) in self.slots.iter_mut().enumerate() {
            for (index, physical) in baked.physical.iter().enumerate() {
                let names: Vec<&str> = physical.members.iter().map(|m| graph.resource_name(*m)).collect();
                let name = format!("{} [slot {slot_index}]", names.join("+"));
                let entry = match physical.desc {
                    ResourceDesc::Image(desc) => Physical {
                        image: Some(OwnedImage::new(&self.gpu, desc, &name)?),
                        buffer: None,
                    },
                    ResourceDesc::Buffer(desc) => Physical {
                        image: None,
                        buffer: Some(OwnedBuffer::new(&self.gpu, desc, MemoryLocation::GpuOnly, &name)?),
                    },
                };
                let _ = index;
                slot.physical.push(entry);
            }
            slot.states.clear();
            if self.config.profile && self.gpu.timestamp_period > 0.0 {
                let needed = 2 * baked.passes.len() as u32;
                if slot.query_capacity < needed {
                    if slot.queries != vk::QueryPool::null() {
                        unsafe { self.gpu.device.destroy_query_pool(slot.queries, None) };
                    }
                    let info = vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(needed);
                    slot.queries = unsafe { self.gpu.device.create_query_pool(&info, None) }?;
                    slot.query_capacity = needed;
                }
            }
            slot.pending_queries = 0;
            if self.pool.is_some() {
                destroy_pass_cmds(&self.gpu.device, &mut slot.pass_cmds);
                for index in 0..baked.passes.len() {
                    let pool = unsafe {
                        self.gpu.device.create_command_pool(
                            &vk::CommandPoolCreateInfo::default()
                                .queue_family_index(self.gpu.queue_family)
                                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                            None,
                        )
                    }?;
                    let alloc = vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1);
                    let cmd = unsafe { self.gpu.device.allocate_command_buffers(&alloc) }?[0];
                    self.gpu.set_name(cmd, &format!("frame slot {slot_index} pass {index}"));
                    slot.pass_cmds.push((pool, cmd));
                }
            }
        }
        self.sync_graph_handles(&graph, &baked)?;
        self.stats.bakes += 1;
        self.stats.physical_resources = baked.physical.len();
        self.stats.estimated_bytes = baked.estimated_bytes();
        self.gpu_accum = vec![(0.0, 0); baked.passes.len()];
        self.gpu_frames.clear();
        self.baked = Some(baked);
        self.graph = graph;
        Ok(true)
    }

    /// Keep each graph resource's name and handles across bakes. The GPU is
    /// idle here, so the new physical views can be written into every slot
    /// directly.
    fn sync_graph_handles(&mut self, graph: &FrameGraph<R>, baked: &BakedGraph) -> Result<()> {
        let mut old = std::mem::take(&mut self.graph_handles);
        let mut new = BTreeMap::new();
        let mut seen = BTreeSet::new();
        for (index, resource) in baked.resources.iter().enumerate() {
            if !resource.reachable {
                continue;
            }
            let Some(physical) = resource.physical else {
                continue;
            };
            let id = ResourceId(index as u32);
            let name = graph.resource_name(id).to_string();
            if !seen.insert(name.clone()) {
                continue;
            }
            let mut handles = old.remove(&name).unwrap_or_default();
            match baked.physical[physical as usize].desc {
                ResourceDesc::Image(image) => {
                    if image.usage.contains(vk::ImageUsageFlags::SAMPLED) {
                        let views: Vec<vk::ImageView> = self
                            .slots
                            .iter()
                            .map(|slot| {
                                slot.physical[physical as usize]
                                    .image
                                    .as_ref()
                                    .expect("a physical image")
                                    .view
                            })
                            .collect();
                        let layout = sampled_layout(baked, id);
                        let kind = if image.layers > 1 {
                            SampledKind::Image2DArray
                        } else {
                            SampledKind::Image2D
                        };
                        match handles.sampled {
                            Some(handle) => {
                                self.bindless
                                    .update_graph_sampled(&self.gpu.device, handle, &views, layout);
                            }
                            None => {
                                handles.sampled = Some(self.bindless.register_graph_sampled(
                                    &self.gpu.device,
                                    kind,
                                    &views,
                                    layout,
                                )?);
                            }
                        }
                    } else if let Some(handle) = handles.sampled.take() {
                        self.bindless.retire_sampled(handle);
                    }
                    if image.usage.contains(vk::ImageUsageFlags::STORAGE) {
                        let views: Vec<vk::ImageView> = self
                            .slots
                            .iter()
                            .map(|slot| {
                                slot.physical[physical as usize]
                                    .image
                                    .as_ref()
                                    .expect("a physical image")
                                    .view
                            })
                            .collect();
                        match handles.storage {
                            Some(handle) => {
                                self.bindless.update_graph_storage_image(
                                    &self.gpu.device,
                                    handle,
                                    &views,
                                    vk::ImageLayout::GENERAL,
                                );
                            }
                            None => {
                                handles.storage = Some(self.bindless.register_graph_storage_image(
                                    &self.gpu.device,
                                    &views,
                                    vk::ImageLayout::GENERAL,
                                )?);
                            }
                        }
                    } else if let Some(handle) = handles.storage.take() {
                        self.bindless.retire_storage_image(handle);
                    }
                }
                ResourceDesc::Buffer(buffer) => {
                    if buffer.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
                        let buffers: Vec<(vk::Buffer, u64, u64)> = self
                            .slots
                            .iter()
                            .map(|slot| {
                                let buffer = slot.physical[physical as usize]
                                    .buffer
                                    .as_ref()
                                    .expect("a physical buffer");
                                (buffer.buffer, 0, buffer.size)
                            })
                            .collect();
                        match handles.buffer {
                            Some(handle) => {
                                self.bindless.update_graph_buffer(&self.gpu.device, handle, &buffers);
                            }
                            None => {
                                handles.buffer = Some(self.bindless.register_graph_buffer(&self.gpu.device, &buffers)?);
                            }
                        }
                    } else if let Some(handle) = handles.buffer.take() {
                        self.bindless.retire_buffer(handle);
                    }
                }
            }
            new.insert(name, handles);
        }
        for (_, handles) in old {
            if let Some(handle) = handles.sampled {
                self.bindless.retire_sampled(handle);
            }
            if let Some(handle) = handles.storage {
                self.bindless.retire_storage_image(handle);
            }
            if let Some(handle) = handles.buffer {
                self.bindless.retire_buffer(handle);
            }
        }
        self.graph_handles = new;
        Ok(())
    }

    /// A graph resource's bindless indices by name, for tests and tools:
    /// `[sampled, storage image, storage buffer]`.
    pub fn graph_index(&self, name: &str) -> Option<[Option<u32>; 3]> {
        self.graph_handles.get(name).map(|handles| {
            [
                handles.sampled.map(|handle| handle.index),
                handles.storage.map(|handle| handle.index),
                handles.buffer.map(|handle| handle.index),
            ]
        })
    }

    fn release_physical(&mut self) {
        for slot in &mut self.slots {
            for mut physical in slot.physical.drain(..) {
                if let Some(image) = physical.image.as_mut() {
                    image.destroy(&self.gpu);
                }
                if let Some(buffer) = physical.buffer.as_mut() {
                    buffer.destroy(&self.gpu);
                }
            }
        }
    }

    /// Wait for the next slot's previous submission (bounded by frames in
    /// flight) and collect its timestamps.
    pub fn begin_frame(&mut self) -> Result<FrameToken> {
        let slot_index = (self.frame % self.config.frames_in_flight as u64) as usize;
        let started = Instant::now();
        let slot = &mut self.slots[slot_index];
        wait_fence(&self.gpu, slot.fence).context("waiting for a frame fence")?;
        self.stats.wait_ms = started.elapsed().as_secs_f32() * 1000.0;
        // The slot's fence has signalled, so its set is idle: applying the
        // queued writes here needs no update-after-bind feature.
        self.stats.bindless_writes += self.bindless.begin_frame(&self.gpu.device, slot_index);
        if slot.pending_queries > 0 {
            let mut data = vec![0u64; slot.pending_queries as usize];
            let result = unsafe {
                self.gpu
                    .device
                    .get_query_pool_results(slot.queries, 0, &mut data, vk::QueryResultFlags::TYPE_64)
            };
            if result.is_ok() {
                let period = self.gpu.timestamp_period as f64;
                let mut frame = Vec::with_capacity(data.len() / 2);
                for (pass, pair) in data.as_chunks::<2>().0.iter().enumerate() {
                    let ms = pair[1].saturating_sub(pair[0]) as f64 * period / 1.0e6;
                    frame.push(ms as f32);
                    if let Some(accum) = self.gpu_accum.get_mut(pass) {
                        accum.0 += ms;
                        accum.1 += 1;
                    }
                }
                if self.gpu_frames.len() >= MAX_GPU_FRAMES {
                    self.gpu_frames.remove(0);
                }
                self.gpu_frames.push(frame);
            }
            slot.pending_queries = 0;
        }
        Ok(FrameToken { slot: slot_index })
    }

    /// The per-pass GPU times (ms, baked order as `stats().pass_names`) of
    /// the frames whose timestamps were collected since the last call (or
    /// the last rebake), oldest first. Empty unless profiling.
    pub fn take_gpu_frames(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.gpu_frames)
    }

    pub fn image_available(&self, token: FrameToken) -> vk::Semaphore {
        self.slots[token.slot].image_available
    }

    /// The slot's fence (signalled when its last submission completes).
    pub fn fence(&self, token: FrameToken) -> vk::Fence {
        self.slots[token.slot].fence
    }

    /// Wait for the given slot's submission to complete (headless readback).
    pub fn wait(&self, token: FrameToken) -> Result<()> {
        wait_fence(&self.gpu, self.slots[token.slot].fence)
    }

    /// Record and submit the baked plan. `externals` binds every reachable
    /// external/backbuffer resource for this frame.
    ///
    /// Barriers, skips and attachments are worked out serially in baked
    /// order first. With `record_threads > 1`, a recorder that offers
    /// [`Recorder::shared`] and at least two passes to record, each pass is
    /// then recorded into its own primary command buffer on the recording
    /// pool's threads; otherwise all passes go into one buffer on this
    /// thread. Either way one submit runs them in baked order, with the same
    /// commands per pass.
    pub fn execute(
        &mut self,
        token: FrameToken,
        recorder: &mut dyn Recorder<R>,
        externals: &dyn Fn(ResourceId) -> Option<Binding>,
        waits: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        signals: &[vk::Semaphore],
    ) -> Result<()>
    where
        R: Sync,
    {
        let record_started = Instant::now();
        let gpu = self.gpu.clone();
        let device = &gpu.device;
        let baked = self.baked.as_ref().ok_or_else(|| anyhow!("execute before set_graph"))?;
        let graph = &self.graph;
        let slot = &mut self.slots[token.slot];

        // Bind every reachable resource for this frame.
        let mut bindings: Vec<Option<Binding>> = vec![None; graph.resources.len()];
        for (index, resource) in baked.resources.iter().enumerate() {
            if !resource.reachable {
                continue;
            }
            bindings[index] = match resource.physical {
                Some(physical) => {
                    let entry = &slot.physical[physical as usize];
                    let identity = (u64::from(token.slot as u32) << 32) | physical as u64;
                    let handles = self
                        .graph_handles
                        .get(graph.resource_name(ResourceId(index as u32)))
                        .copied()
                        .unwrap_or_default();
                    if let Some(image) = &entry.image {
                        Some(Binding::Image(ImageBinding {
                            image: image.image,
                            view: image.view,
                            format: image.desc.format,
                            extent: image.desc.extent(),
                            range: image.range(),
                            identity,
                            sampled: handles.sampled.map(|handle| handle.index),
                            storage: handles.storage.map(|handle| handle.index),
                        }))
                    } else {
                        entry.buffer.as_ref().map(|buffer| {
                            Binding::Buffer(BufferBinding {
                                buffer: buffer.buffer,
                                offset: 0,
                                size: buffer.size,
                                identity,
                                storage: handles.buffer.map(|handle| handle.index),
                            })
                        })
                    }
                }
                None => Some(externals(ResourceId(index as u32)).ok_or_else(|| {
                    anyhow!(
                        "external resource {:?} is not bound",
                        graph.resource_name(ResourceId(index as u32))
                    )
                })?),
            };
        }

        // Serial: what each pass records around its payload.
        let mut store = Store {
            slot: &mut slot.states,
            external: &mut self.external_states,
            bindings: &bindings,
        };
        let mut barrier_count = 0u64;
        let mut plans = Vec::with_capacity(baked.passes.len());
        for (index, baked_pass) in baked.passes.iter().enumerate() {
            let pass = &graph.passes[baked_pass.pass.index()];
            let skipped = baked_pass.conditional && recorder.skip(&pass.record);
            if skipped {
                self.stats.skipped_passes += 1;
                plans.push(PassPlan {
                    index,
                    pass: baked_pass.pass,
                    skipped,
                    payload: false,
                    barriers: Vec::new(),
                    rendering: None,
                });
                continue;
            }
            let barriers = pass_barriers(baked, index, &mut store);
            barrier_count += barriers.len() as u64;
            let rendering = match baked_pass.kind {
                PassKind::Graphics => {
                    let mut rendering = Rendering {
                        colors: Vec::new(),
                        depth: None,
                        stencil: false,
                        area: vk::Extent2D::default(),
                    };
                    for access in &baked_pass.accesses {
                        let Some(load) = access.role.load_op() else { continue };
                        let Some(Binding::Image(image)) = bindings[access.resource.index()] else {
                            bail!("attachment of {:?} is not an image", pass.name);
                        };
                        rendering.area = image.extent;
                        // A load from never-written contents (first use of a
                        // persistent or external image) is a clear instead,
                        // so the first frame is deterministic.
                        let first_use = barriers.iter().any(|barrier| {
                            barrier.resource == access.resource && barrier.old_layout == vk::ImageLayout::UNDEFINED
                        });
                        let load = if load == LoadOp::Load && first_use {
                            LoadOp::Clear
                        } else {
                            load
                        };
                        let store = access.store.unwrap_or(vk::AttachmentStoreOp::STORE);
                        if access.role.is_depth() {
                            let (depth, stencil) = recorder.clear_depth(&pass.record, access.resource);
                            rendering.stencil = image.range.aspect_mask.contains(vk::ImageAspectFlags::STENCIL);
                            rendering.depth = Some(Attachment {
                                view: image.view,
                                layout: access.usage.layout,
                                load: load.vk(),
                                store,
                                clear: vk::ClearValue {
                                    depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
                                },
                            });
                            continue;
                        }
                        let clear = recorder.clear_color(&pass.record, access.resource);
                        rendering.colors.push(Attachment {
                            view: image.view,
                            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            load: load.vk(),
                            store,
                            clear: vk::ClearValue {
                                color: vk::ClearColorValue { float32: clear },
                            },
                        });
                    }
                    Some(rendering)
                }
                PassKind::Compute | PassKind::Transfer | PassKind::Present => None,
            };
            let payload = baked_pass.kind != PassKind::Present;
            plans.push(PassPlan {
                index,
                pass: baked_pass.pass,
                skipped,
                payload,
                barriers,
                rendering,
            });
        }

        let timestamps = slot.query_capacity >= 2 * baked.passes.len() as u32 && slot.queries != vk::QueryPool::null();
        let frame = FrameRecording {
            gpu: &gpu,
            slot: token.slot,
            bindings: &bindings,
            queries: if timestamps {
                slot.queries
            } else {
                vk::QueryPool::null()
            },
        };
        let recorded = plans.iter().filter(|p| !p.skipped).count();
        let mut cmds = Vec::with_capacity(plans.len() + 1);
        let parallel = match (&self.pool, recorded >= 2 && slot.pass_cmds.len() == plans.len()) {
            (Some(pool), true) => recorder.shared().map(|shared| {
                let results: Vec<Result<()>> = pool.install(|| {
                    use rayon::prelude::*;
                    plans
                        .par_iter()
                        .zip(&slot.pass_cmds)
                        .map(|(plan, &(pass_pool, cmd))| {
                            let pass = &graph.passes[plan.pass.index()];
                            unsafe {
                                device.reset_command_pool(pass_pool, vk::CommandPoolResetFlags::empty())?;
                                device.begin_command_buffer(
                                    cmd,
                                    &vk::CommandBufferBeginInfo::default()
                                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                                )?;
                            }
                            let result = frame
                                .record_pass(cmd, plan, &pass.name, &pass.record, &mut |r, ctx| shared.record(r, ctx));
                            unsafe { device.end_command_buffer(cmd) }?;
                            result
                        })
                        .collect()
                });
                results
            }),
            _ => None,
        };
        let head = slot.cmd;
        unsafe {
            device.reset_command_pool(slot.pool, vk::CommandPoolResetFlags::empty())?;
            device.begin_command_buffer(
                head,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            if timestamps {
                device.cmd_reset_query_pool(head, slot.queries, 0, 2 * baked.passes.len() as u32);
            }
        }
        cmds.push(head);
        match parallel {
            Some(results) => {
                unsafe { device.end_command_buffer(head) }?;
                // The earliest failing pass's error.
                results.into_iter().collect::<Result<Vec<()>>>()?;
                cmds.extend(slot.pass_cmds.iter().map(|&(_, cmd)| cmd));
                self.stats.parallel_frames += 1;
            }
            None => {
                for plan in &plans {
                    let pass = &graph.passes[plan.pass.index()];
                    frame.record_pass(head, plan, &pass.name, &pass.record, &mut |r, ctx| {
                        recorder.record(r, ctx)
                    })?;
                }
                unsafe { device.end_command_buffer(head) }?;
            }
        }
        self.stats.record_ms = record_started.elapsed().as_secs_f32() * 1000.0;

        let submit_started = Instant::now();
        let wait_infos: Vec<_> = waits
            .iter()
            .map(|(semaphore, stages)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*semaphore)
                    .stage_mask(*stages)
            })
            .collect();
        let signal_infos: Vec<_> = signals
            .iter()
            .map(|semaphore| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*semaphore)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();
        let cmd_infos: Vec<_> = cmds
            .iter()
            .map(|&cmd| vk::CommandBufferSubmitInfo::default().command_buffer(cmd))
            .collect();
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
            .command_buffer_infos(&cmd_infos)
            .signal_semaphore_infos(&signal_infos);
        unsafe {
            device.reset_fences(&[slot.fence])?;
            device
                .queue_submit2(gpu.queue, &[submit], slot.fence)
                .map_err(crate::device::fatal)
                .context("vkQueueSubmit2")?;
        }
        slot.pending_queries = if timestamps { 2 * baked.passes.len() as u32 } else { 0 };
        self.stats.submit_ms = submit_started.elapsed().as_secs_f32() * 1000.0;
        self.stats.barriers += barrier_count;
        self.stats.frames += 1;
        self.frame += 1;
        Ok(())
    }
}

/// One attachment of a graphics pass, as worked out before recording.
#[derive(Clone, Copy)]
struct Attachment {
    view: vk::ImageView,
    layout: vk::ImageLayout,
    load: vk::AttachmentLoadOp,
    store: vk::AttachmentStoreOp,
    clear: vk::ClearValue,
}

impl Attachment {
    fn info(&self) -> vk::RenderingAttachmentInfo<'static> {
        vk::RenderingAttachmentInfo::default()
            .image_view(self.view)
            .image_layout(self.layout)
            .load_op(self.load)
            .store_op(self.store)
            .clear_value(self.clear)
    }
}

struct Rendering {
    colors: Vec<Attachment>,
    depth: Option<Attachment>,
    /// The depth attachment has a stencil aspect (bound as both).
    stencil: bool,
    area: vk::Extent2D,
}

/// Everything recorded around one pass's payload, decided serially.
struct PassPlan {
    /// Baked order (timestamp pair index).
    index: usize,
    pass: PassId,
    skipped: bool,
    /// Calls the recorder (all but present passes).
    payload: bool,
    barriers: Vec<BarrierDesc>,
    /// Graphics passes: the dynamic rendering scope.
    rendering: Option<Rendering>,
}

/// What every pass of one frame records with; shared by the recording
/// threads.
struct FrameRecording<'a> {
    gpu: &'a GpuDevice,
    slot: usize,
    bindings: &'a [Option<Binding>],
    /// Null without timestamps.
    queries: vk::QueryPool,
}

impl FrameRecording<'_> {
    /// Timestamps, label, barriers, rendering scope and the payload of one
    /// pass, into `cmd`.
    fn record_pass<R>(
        &self,
        cmd: vk::CommandBuffer,
        plan: &PassPlan,
        name: &str,
        payload: &R,
        record: &mut dyn FnMut(&R, &PassContext<'_>) -> Result<()>,
    ) -> Result<()> {
        let device = &self.gpu.device;
        let timestamp = |at: u32| {
            if self.queries != vk::QueryPool::null() {
                unsafe { device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, self.queries, at) };
            }
        };
        timestamp(2 * plan.index as u32);
        if !plan.skipped {
            self.gpu.begin_label(cmd, name);
            emit_barriers(device, cmd, &plan.barriers, self.bindings)?;
            let mut ctx = PassContext {
                device,
                cmd,
                slot: self.slot,
                pass: plan.pass,
                render_area: vk::Rect2D::default(),
                bindings: self.bindings,
            };
            match &plan.rendering {
                Some(r) => {
                    let render_area = vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: r.area,
                    };
                    let colors: Vec<_> = r.colors.iter().map(Attachment::info).collect();
                    let depth = r.depth.as_ref().map(Attachment::info);
                    let mut info = vk::RenderingInfo::default()
                        .render_area(render_area)
                        .layer_count(1)
                        .color_attachments(&colors);
                    if let Some(depth) = depth.as_ref() {
                        info = info.depth_attachment(depth);
                        if r.stencil {
                            info = info.stencil_attachment(depth);
                        }
                    }
                    unsafe {
                        device.cmd_begin_rendering(cmd, &info);
                        let viewport = vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: r.area.width as f32,
                            height: r.area.height as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        };
                        device.cmd_set_viewport(cmd, 0, &[viewport]);
                        device.cmd_set_scissor(cmd, 0, &[render_area]);
                    }
                    ctx.render_area = render_area;
                    let result = record(payload, &ctx);
                    unsafe { device.cmd_end_rendering(cmd) };
                    result.with_context(|| format!("recording {name:?}"))?;
                }
                None if plan.payload => record(payload, &ctx).with_context(|| format!("recording {name:?}"))?,
                None => {}
            }
            self.gpu.end_label(cmd);
        }
        timestamp(2 * plan.index as u32 + 1);
        Ok(())
    }
}

/// The layout the graph's sampled access transitions an image to. A
/// sampled depth image is read in the depth read-only layout; everything
/// else is `SHADER_READ_ONLY_OPTIMAL`.
fn sampled_layout(baked: &BakedGraph, id: ResourceId) -> vk::ImageLayout {
    for pass in &baked.passes {
        for access in &pass.accesses {
            if access.resource == id && matches!(access.role, Role::Sampled(_) | Role::DepthRead) {
                return access.usage.layout;
            }
        }
    }
    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
}

fn emit_barriers(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    barriers: &[BarrierDesc],
    bindings: &[Option<Binding>],
) -> Result<()> {
    if barriers.is_empty() {
        return Ok(());
    }
    let mut images = Vec::new();
    let mut buffers = Vec::new();
    for barrier in barriers {
        match bindings[barrier.resource.index()] {
            Some(Binding::Image(image)) => images.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(barrier.src_stages)
                    .src_access_mask(barrier.src_access)
                    .dst_stage_mask(barrier.dst_stages)
                    .dst_access_mask(barrier.dst_access)
                    .old_layout(barrier.old_layout)
                    .new_layout(barrier.new_layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.image)
                    .subresource_range(image.range),
            ),
            Some(Binding::Buffer(buffer)) => buffers.push(
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(barrier.src_stages)
                    .src_access_mask(barrier.src_access)
                    .dst_stage_mask(barrier.dst_stages)
                    .dst_access_mask(barrier.dst_access)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffer.buffer)
                    .offset(buffer.offset)
                    .size(buffer.size),
            ),
            None => bail!("barrier on unbound resource #{}", barrier.resource.0),
        }
    }
    let dependency = vk::DependencyInfo::default()
        .image_memory_barriers(&images)
        .buffer_memory_barriers(&buffers);
    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency) };
    Ok(())
}

impl<R> Drop for Executor<R> {
    fn drop(&mut self) {
        self.gpu.wait_idle();
        self.release_physical();
        self.bindless.destroy(&self.gpu);
        let device = &self.gpu.device;
        for mut slot in self.slots.drain(..) {
            destroy_pass_cmds(device, &mut slot.pass_cmds);
            unsafe {
                device.destroy_command_pool(slot.pool, None);
                device.destroy_fence(slot.fence, None);
                device.destroy_semaphore(slot.image_available, None);
                if slot.queries != vk::QueryPool::null() {
                    device.destroy_query_pool(slot.queries, None);
                }
            }
        }
    }
}

fn destroy_pass_cmds(device: &ash::Device, cmds: &mut Vec<(vk::CommandPool, vk::CommandBuffer)>) {
    for (pool, _) in cmds.drain(..) {
        unsafe { device.destroy_command_pool(pool, None) };
    }
}

/// How long a frame fence may take before the GPU is considered hung.
const FENCE_TIMEOUT_NS: u64 = 5_000_000_000;

/// Wait for a frame fence, bounded: a hung GPU becomes an error (the render
/// thread exits) instead of an unbounded wait that would also block `stop`.
fn wait_fence(gpu: &GpuDevice, fence: vk::Fence) -> Result<()> {
    match unsafe { gpu.device.wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS) } {
        Ok(()) => Ok(()),
        Err(vk::Result::TIMEOUT) => {
            gpu.mark_hung();
            bail!("the GPU did not finish a frame within 5 s (hang)")
        }
        Err(err) => Err(crate::device::fatal(err)),
    }
}
