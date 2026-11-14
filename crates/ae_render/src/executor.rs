//! Frame-graph executor (§3.5): realizes the baked plan (one physical set per
//! frame slot), records every frame into one primary command buffer on the
//! single graphics queue, and derives barriers from the access state machine
//! in `frame_graph::barriers`. The executor owns the baked plan and all
//! execution state; declarations hold none.
//!
//! Frame protocol (render thread):
//! `begin_frame` (waits the slot's fence) → caller acquires a swapchain
//! image with `image_available(slot)` → `execute` (records, submits with the
//! caller's waits/signals, fences the slot) → caller presents.

use crate::device::GpuDevice;
use ae_framegraph::barriers::{AccessState, BarrierDesc, StateKey, StateStore, pass_barriers};
use ae_framegraph::{
    BakeOptions, BakedGraph, FrameGraph, LoadOp, PassId, PassKind, ResourceDesc, ResourceId, bake,
};
use crate::memory::{OwnedBuffer, OwnedImage};
use anyhow::{Context as _, Result, anyhow, bail};
use ash::vk;
use gpu_allocator::MemoryLocation;
use std::collections::BTreeMap;
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
}

#[derive(Clone, Copy, Debug)]
pub struct BufferBinding {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
    pub identity: u64,
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
}

#[derive(Clone, Copy, Debug)]
pub struct ExecutorConfig {
    /// Frames in flight: data, not a constant baked into allocation sizes.
    pub frames_in_flight: usize,
    /// GPU timestamps per pass.
    pub profile: bool,
    pub bake: BakeOptions,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            frames_in_flight: 2,
            profile: false,
            bake: BakeOptions::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExecutorStats {
    pub bakes: u64,
    pub frames: u64,
    pub barriers: u64,
    pub skipped_passes: u64,
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
                let identity = self.bindings[resource.index()].map(|b| b.identity()).unwrap_or(u64::MAX);
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
    external_states: BTreeMap<u64, AccessState>,
    frame: u64,
    stats: ExecutorStats,
    gpu_accum: Vec<(f64, u64)>,
}

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
                device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)
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
            });
        }
        Ok(Self {
            gpu,
            config,
            graph: FrameGraph::new(),
            baked: None,
            slots,
            external_states: BTreeMap::new(),
            frame: 0,
            stats: ExecutorStats::default(),
            gpu_accum: Vec::new(),
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
        stats.pass_names = self.baked.iter().flat_map(|b| b.order()).map(|p| self.graph.passes[p.0 as usize].name.clone()).collect();
        stats
    }

    /// Install the declarations for the next frames. Rebakes and
    /// re-realizes only when the topology signature changed; otherwise only
    /// the per-frame payloads are replaced. Returns whether it rebaked.
    pub fn set_graph(&mut self, graph: FrameGraph<R>) -> Result<bool> {
        let signature = graph.signature();
        if self.baked.as_ref().is_some_and(|baked| baked.signature == signature) {
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
        }
        self.stats.bakes += 1;
        self.stats.physical_resources = baked.physical.len();
        self.stats.estimated_bytes = baked.estimated_bytes();
        self.gpu_accum = vec![(0.0, 0); baked.passes.len()];
        self.baked = Some(baked);
        self.graph = graph;
        Ok(true)
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
        if slot.pending_queries > 0 {
            let mut data = vec![0u64; slot.pending_queries as usize];
            let result = unsafe {
                self.gpu
                    .device
                    .get_query_pool_results(slot.queries, 0, &mut data, vk::QueryResultFlags::TYPE_64)
            };
            if result.is_ok() {
                let period = self.gpu.timestamp_period as f64;
                for (pass, pair) in data.as_chunks::<2>().0.iter().enumerate() {
                    if let Some(accum) = self.gpu_accum.get_mut(pass) {
                        accum.0 += pair[1].saturating_sub(pair[0]) as f64 * period / 1.0e6;
                        accum.1 += 1;
                    }
                }
            }
            slot.pending_queries = 0;
        }
        Ok(FrameToken { slot: slot_index })
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
    pub fn execute(
        &mut self,
        token: FrameToken,
        recorder: &mut dyn Recorder<R>,
        externals: &dyn Fn(ResourceId) -> Option<Binding>,
        waits: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        signals: &[vk::Semaphore],
    ) -> Result<()> {
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
                    if let Some(image) = &entry.image {
                        Some(Binding::Image(ImageBinding {
                            image: image.image,
                            view: image.view,
                            format: image.desc.format,
                            extent: image.desc.extent(),
                            range: image.range(),
                            identity,
                        }))
                    } else {
                        entry.buffer.as_ref().map(|buffer| {
                            Binding::Buffer(BufferBinding {
                                buffer: buffer.buffer,
                                offset: 0,
                                size: buffer.size,
                                identity,
                            })
                        })
                    }
                }
                None => Some(externals(ResourceId(index as u32)).ok_or_else(|| {
                    anyhow!("external resource {:?} is not bound", graph.resource_name(ResourceId(index as u32)))
                })?),
            };
        }

        unsafe {
            device.reset_command_pool(slot.pool, vk::CommandPoolResetFlags::empty())?;
            device.begin_command_buffer(
                slot.cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let cmd = slot.cmd;
        let timestamps = slot.query_capacity >= 2 * baked.passes.len() as u32 && slot.queries != vk::QueryPool::null();
        if timestamps {
            unsafe { device.cmd_reset_query_pool(cmd, slot.queries, 0, 2 * baked.passes.len() as u32) };
        }

        let mut store = Store {
            slot: &mut slot.states,
            external: &mut self.external_states,
            bindings: &bindings,
        };
        let mut barrier_count = 0u64;
        for (index, baked_pass) in baked.passes.iter().enumerate() {
            let pass = &graph.passes[baked_pass.pass.index()];
            if timestamps {
                unsafe {
                    device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, slot.queries, 2 * index as u32)
                };
            }
            let skipped = baked_pass.conditional && recorder.skip(&pass.record);
            if skipped {
                self.stats.skipped_passes += 1;
            } else {
                gpu.begin_label(cmd, &pass.name);
                let barriers = pass_barriers(baked, index, &mut store);
                barrier_count += barriers.len() as u64;
                emit_barriers(device, cmd, &barriers, &bindings)?;
                match baked_pass.kind {
                    PassKind::Graphics => {
                        let mut colors = Vec::new();
                        let mut depth = None;
                        let mut stencil = false;
                        let mut area = vk::Extent2D::default();
                        for access in &baked_pass.accesses {
                            let Some(load) = access.role.load_op() else { continue };
                            let Some(Binding::Image(image)) = bindings[access.resource.index()] else {
                                bail!("attachment of {:?} is not an image", pass.name);
                            };
                            area = image.extent;
                            // A load from never-written contents (first use of a
                            // persistent or external image) is a clear instead,
                            // so the first frame is deterministic.
                            let first_use = barriers.iter().any(|barrier| {
                                barrier.resource == access.resource
                                    && barrier.old_layout == vk::ImageLayout::UNDEFINED
                            });
                            let load = if load == LoadOp::Load && first_use { LoadOp::Clear } else { load };
                            if access.role.is_depth() {
                                let (clear_depth, clear_stencil) = recorder.clear_depth(&pass.record, access.resource);
                                stencil = image.range.aspect_mask.contains(vk::ImageAspectFlags::STENCIL);
                                depth = Some(
                                    vk::RenderingAttachmentInfo::default()
                                        .image_view(image.view)
                                        .image_layout(access.usage.layout)
                                        .load_op(load.vk())
                                        .store_op(access.store.unwrap_or(vk::AttachmentStoreOp::STORE))
                                        .clear_value(vk::ClearValue {
                                            depth_stencil: vk::ClearDepthStencilValue { depth: clear_depth, stencil: clear_stencil },
                                        }),
                                );
                                continue;
                            }
                            let clear = recorder.clear_color(&pass.record, access.resource);
                            colors.push(
                                vk::RenderingAttachmentInfo::default()
                                    .image_view(image.view)
                                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                    .load_op(load.vk())
                                    .store_op(access.store.unwrap_or(vk::AttachmentStoreOp::STORE))
                                    .clear_value(vk::ClearValue {
                                        color: vk::ClearColorValue { float32: clear },
                                    }),
                            );
                        }
                        let render_area = vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent: area,
                        };
                        let mut info = vk::RenderingInfo::default()
                            .render_area(render_area)
                            .layer_count(1)
                            .color_attachments(&colors);
                        if let Some(depth) = depth.as_ref() {
                            info = info.depth_attachment(depth);
                            if stencil {
                                info = info.stencil_attachment(depth);
                            }
                        }
                        unsafe {
                            device.cmd_begin_rendering(cmd, &info);
                            device.cmd_set_viewport(
                                cmd,
                                0,
                                &[vk::Viewport {
                                    x: 0.0,
                                    y: 0.0,
                                    width: area.width as f32,
                                    height: area.height as f32,
                                    min_depth: 0.0,
                                    max_depth: 1.0,
                                }],
                            );
                            device.cmd_set_scissor(cmd, 0, &[render_area]);
                        }
                        let ctx = PassContext {
                            device,
                            cmd,
                            slot: token.slot,
                            pass: baked_pass.pass,
                            render_area,
                            bindings: &bindings,
                        };
                        let result = recorder.record(&pass.record, &ctx);
                        unsafe { device.cmd_end_rendering(cmd) };
                        result.with_context(|| format!("recording {:?}", pass.name))?;
                    }
                    PassKind::Compute | PassKind::Transfer => {
                        let ctx = PassContext {
                            device,
                            cmd,
                            slot: token.slot,
                            pass: baked_pass.pass,
                            render_area: vk::Rect2D::default(),
                            bindings: &bindings,
                        };
                        recorder
                            .record(&pass.record, &ctx)
                            .with_context(|| format!("recording {:?}", pass.name))?;
                    }
                    PassKind::Present => {}
                }
                gpu.end_label(cmd);
            }
            if timestamps {
                unsafe {
                    device.cmd_write_timestamp2(
                        cmd,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        slot.queries,
                        2 * index as u32 + 1,
                    )
                };
            }
        }
        unsafe { device.end_command_buffer(cmd) }?;
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
        let cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
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
        let device = &self.gpu.device;
        for slot in self.slots.drain(..) {
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
