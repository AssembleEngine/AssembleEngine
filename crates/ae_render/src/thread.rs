//! The GPU render thread (§3.6, §4.1): the only owner of the device,
//! swapchain, frame graph executor and queue. The main thread keeps SDL (the
//! window and the event pump), creates the instance and surface, and
//! forwards resize/minimize/close as messages; it never touches the
//! swapchain.
//!
//! A `RenderApp` supplies what to draw: per frame it declares the graph
//! (unchanged declarations do not rebake), fills its per-slot data after the
//! slot's fence was waited, binds its external resources and records its
//! passes. The thread never waits on a simulation or presentation lock.

use crate::device::{GpuDevice, GpuInstance};
use crate::executor::{Binding, Executor, ExecutorConfig, ExecutorStats, ImageBinding, Recorder};
use ae_framegraph::{FrameGraph, ResourceId};
use crate::swapchain::{Acquired, HdrRequest, PresentPolicy, Swapchain, SwapchainOptions, TargetFormat, color_range};
use crate::window::{SdlWindow, SurfaceSource, WindowEvents};
use anyhow::{Context as _, Result};
use ash::vk;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The render target of the current topology: its size, the backbuffer's
/// format and the colour space it stores (the pair drives the encoding and
/// the graph redeclare; M13 stage 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetInfo {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
}

impl TargetInfo {
    /// The pair a swapchain target stores.
    pub fn target(self) -> TargetFormat {
        TargetFormat { format: self.format, color_space: self.color_space }
    }
}

pub trait RenderApp: Send + 'static {
    type Pass: Send + Sync + 'static;

    /// (Re)create format-dependent GPU objects (pipelines). Called once and
    /// whenever the swapchain's (format, colour space) pair changes.
    fn create(&mut self, gpu: &Arc<GpuDevice>, target: TargetInfo, frames_in_flight: usize) -> Result<()>;

    /// The swapchain's (format, colour space) pair changed after the first
    /// `create` (device idle). Returns whether objects the app binds as
    /// externals were replaced: the executor then forgets their layouts.
    /// The default destroys and creates everything; an app that keeps its
    /// format-independent objects (state that builds up over frames)
    /// rebuilds only the rest and returns `false`.
    fn retarget(&mut self, gpu: &Arc<GpuDevice>, target: TargetInfo, frames_in_flight: usize) -> Result<bool> {
        self.destroy(gpu);
        self.create(gpu, target, frames_in_flight)?;
        Ok(true)
    }

    /// The HDR metadata the swapchain should carry after a recreate (the
    /// levels and the output gamut); `None` for an SDR output.
    fn hdr_metadata(&self) -> Option<(crate::encode::DisplayLevels, crate::colorspace::Gamut)> {
        None
    }

    /// Declarations for this frame plus the backbuffer resource (`None`
    /// for offscreen-only graphs). Equal declarations keep the baked plan.
    fn declare(&mut self, target: TargetInfo) -> Result<(FrameGraph<Self::Pass>, Option<ResourceId>)>;

    /// The plan was rebaked (physical resources replaced; device idle).
    fn on_rebake(&mut self) -> Result<()> {
        Ok(())
    }

    /// The previous frame's executor statistics (CPU phases), before
    /// `prepare`. Used for budget rules.
    fn frame_stats(&mut self, _stats: &ExecutorStats) {}

    /// Fill per-slot data (staging, instance data). `slot`'s previous
    /// submission has completed.
    fn prepare(&mut self, slot: usize) -> Result<()>;

    /// Bind an app-owned external resource for `slot`.
    fn external(&self, resource: ResourceId, slot: usize) -> Option<Binding>;

    fn recorder(&mut self) -> &mut dyn Recorder<Self::Pass>;

    /// One status line for the `[render]` report.
    fn status(&self) -> String {
        String::new()
    }

    /// Release GPU objects (device still alive, idle).
    fn destroy(&mut self, gpu: &Arc<GpuDevice>);
}

/// The window's HDR state as the main thread sees it (SDL owns it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrControl {
    pub request: HdrRequest,
    /// SDL reports the window's display HDR-capable.
    pub capable: bool,
    /// SDL's headroom above SDR white, for the app's level defaults.
    pub headroom: f32,
}

impl Default for HdrControl {
    fn default() -> Self {
        Self { request: HdrRequest::Off, capable: false, headroom: 1.0 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GpuRenderConfig {
    pub frames_in_flight: usize,
    pub present: PresentPolicy,
    /// Cap on frames per second (0 = present-paced only).
    pub max_fps: u32,
    pub profile: bool,
    pub verbose: bool,
    /// `ExecutorConfig::record_threads`.
    pub record_threads: usize,
    /// The swapchain's encoding request and the window's HDR state.
    pub hdr: HdrControl,
}

#[derive(Clone, Debug, Default)]
pub struct GpuRenderStats {
    pub frames: u64,
    pub skipped_minimized: u64,
    pub swapchain_recreates: u64,
    /// Frames skipped because no swapchain image arrived in time.
    pub acquire_timeouts: u64,
    pub executor: ExecutorStats,
    pub app_status: String,
    pub device: String,
    pub present_mode: String,
    /// The output the swapchain carries: the encoding, and in HDR the
    /// levels the frame is encoded at (signal nits), e.g.
    /// `HDR10 (PQ), white 203, peak 361 nits`.
    pub output: String,
    /// The validation layer is on (the counts below mean nothing otherwise).
    pub validation: bool,
    pub validation_errors: u64,
    pub validation_warnings: u64,
    pub error: Option<String>,
    pub acquire_ms: f32,
    pub present_ms: f32,
}

#[derive(Default)]
struct Control {
    stop: AtomicBool,
    resize: Mutex<Option<(u32, u32)>>,
    /// A new present policy (vsync on/off) to apply.
    present: Mutex<Option<PresentPolicy>>,
    minimized: AtomicBool,
    /// A new HDR state (toggle, SDL event, display change).
    hdr: Mutex<Option<HdrControl>>,
    /// The render thread leaked its GPU objects after a hang; the surface
    /// must not be destroyed under the leaked swapchain.
    leaked: AtomicBool,
}

pub struct GpuRenderThread {
    control: Arc<Control>,
    stats: Arc<Mutex<GpuRenderStats>>,
    join: Option<std::thread::JoinHandle<()>>,
    instance: Arc<GpuInstance>,
    surface: vk::SurfaceKHR,
}

impl GpuRenderThread {
    /// Create the instance and surface here (main thread), then start the
    /// render thread with `app`.
    pub fn start<A: RenderApp>(window: &SdlWindow, app: A, config: GpuRenderConfig) -> Result<Self> {
        let extensions = window.instance_extensions()?;
        let instance = Arc::new(GpuInstance::new(&extensions, None)?);
        let surface = window.create_surface(&instance.entry, &instance.instance)?;
        let control = Arc::new(Control::default());
        let stats = Arc::new(Mutex::new(GpuRenderStats::default()));
        let size = window.pixel_size();
        let thread_instance = instance.clone();
        let thread_control = control.clone();
        let thread_stats = stats.clone();
        let join = std::thread::Builder::new()
            .name("gpu-render".into())
            .spawn(move || {
                let result = render_main(thread_instance, surface, size, app, config, &thread_control, &thread_stats);
                if let Err(err) = result {
                    eprintln!("[render] GPU render thread failed: {err:#}");
                    if let Ok(mut stats) = thread_stats.lock() {
                        stats.error = Some(format!("{err:#}"));
                    }
                }
            })
            .context("spawning the render thread")?;
        Ok(Self {
            control,
            stats,
            join: Some(join),
            instance,
            surface,
        })
    }

    /// Forward window events (main thread).
    pub fn handle_events(&self, events: &WindowEvents) {
        if let Some(size) = events.resized
            && let Ok(mut pending) = self.control.resize.lock()
        {
            *pending = Some(size);
        }
        if events.restored {
            self.control.minimized.store(false, Ordering::Release);
        } else if events.minimized {
            self.control.minimized.store(true, Ordering::Release);
        }
    }

    /// Ask the render thread to recreate the swapchain for a new HDR state
    /// (a toggle key, `SDL_EVENT_WINDOW_HDR_STATE_CHANGED`, or a display
    /// change).
    pub fn set_hdr(&self, hdr: HdrControl) {
        if let Ok(mut pending) = self.control.hdr.lock() {
            *pending = Some(hdr);
        }
    }

    /// Ask the render thread for a present policy; the swapchain is
    /// recreated at the next frame. `Fifo` is vsync, `Mailbox` is uncapped.
    pub fn set_present(&self, policy: PresentPolicy) {
        if let Ok(mut pending) = self.control.present.lock() {
            *pending = Some(policy);
        }
    }

    pub fn running(&self) -> bool {
        self.join.as_ref().is_some_and(|join| !join.is_finished())
    }

    pub fn stats(&self) -> GpuRenderStats {
        let mut stats = self.stats.lock().map(|stats| stats.clone()).unwrap_or_default();
        stats.validation = self.instance.validation;
        stats.validation_errors = self.instance.counters.errors();
        stats.validation_warnings = self.instance.counters.warnings();
        stats
    }

    /// Stop and join the render thread, then destroy the surface. Idempotent.
    pub fn stop(&mut self) -> GpuRenderStats {
        self.control.stop.store(true, Ordering::Release);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        if self.surface != vk::SurfaceKHR::null() && !self.control.leaked.load(Ordering::Acquire) {
            self.instance.destroy_surface(self.surface);
            self.surface = vk::SurfaceKHR::null();
        }
        self.stats()
    }
}

impl Drop for GpuRenderThread {
    fn drop(&mut self) {
        self.stop();
    }
}

fn render_main<A: RenderApp>(
    instance: Arc<GpuInstance>,
    surface: vk::SurfaceKHR,
    initial_size: (u32, u32),
    mut app: A,
    config: GpuRenderConfig,
    control: &Control,
    shared: &Mutex<GpuRenderStats>,
) -> Result<()> {
    let gpu = Arc::new(GpuDevice::new(instance, Some(surface))?);
    let mut hdr = config.hdr;
    let mut swapchain = Swapchain::new(&gpu, surface, initial_size, SwapchainOptions { present: config.present, hdr: hdr.request})?;
    let mut executor: Executor<A::Pass> = Executor::new(
        gpu.clone(),
        ExecutorConfig {
            frames_in_flight: config.frames_in_flight,
            profile: config.profile,
            record_threads: config.record_threads,
            ..ExecutorConfig::default()
        },
    )?;
    let mut target = TargetInfo { extent: swapchain.extent, format: swapchain.format.format, color_space: swapchain.format.color_space };
    app.create(&gpu, target, config.frames_in_flight)?;
    if let Some((levels, gamut)) = app.hdr_metadata() {
        swapchain.set_hdr_metadata(&gpu, levels, gamut)?;
    }
    let mut stats = GpuRenderStats {
        device: gpu.device_name.clone(),
        present_mode: format!("{:?}", swapchain.present_mode),
        output: output_label(target, app.hdr_metadata()),
        ..GpuRenderStats::default()
    };
    if config.verbose {
        println!(
            "[render] GPU: {} | swapchain {} {}x{} {:?} | {} frames in flight | validation {}",
            gpu.device_name,
            swapchain.target().describe(),
            swapchain.extent.width,
            swapchain.extent.height,
            swapchain.present_mode,
            config.frames_in_flight,
            gpu.instance.validation,
        );
    }
    let mut dirty = false;
    let mut metadata_pending = false;
    let mut size = initial_size;
    let min_frame = (config.max_fps > 0).then(|| Duration::from_secs_f64(1.0 / config.max_fps as f64));
    let mut last_frame = Instant::now();
    let mut last_report = Instant::now();
    let trace = std::env::var("AE_RENDER_TRACE").is_ok_and(|raw| raw.trim() == "1");
    let result = (|| -> Result<()> {
        while !control.stop.load(Ordering::Acquire) {
            if let Some(new_size) = control.resize.lock().ok().and_then(|mut pending| pending.take()) {
                size = new_size;
                dirty = true;
            }
            if let Some(policy) = control.present.lock().ok().and_then(|mut pending| pending.take())
                && swapchain.policy() != policy
            {
                swapchain.set_policy(policy);
                dirty = true;
            }
            if let Some(next) = control.hdr.lock().ok().and_then(|mut pending| pending.take()) {
                // An HDR state change always recreates (the compositor's
                // description changed, even if SDL's numbers look the same).
                println!("[render] HDR state: --hdr {} | SDL HDR {} | headroom {:.3}x", next.request.label(), if next.capable { "capable" } else { "off" }, next.headroom);
                hdr = next;
                dirty = true;
            }
            if control.minimized.load(Ordering::Acquire) || size.0 == 0 || size.1 == 0 {
                stats.skipped_minimized += 1;
                std::thread::sleep(Duration::from_millis(16));
                continue;
            }
            if dirty || !swapchain.ready() {
                swapchain.set_options(SwapchainOptions { present: swapchain.policy(), hdr: hdr.request});
                if !swapchain.recreate(&gpu, size)? {
                    std::thread::sleep(Duration::from_millis(16));
                    continue;
                }
                stats.swapchain_recreates += 1;
                stats.present_mode = format!("{:?}", swapchain.present_mode);
                dirty = false;
                let next = TargetInfo { extent: swapchain.extent, format: swapchain.format.format, color_space: swapchain.format.color_space };
                let pair_changed = next.target() != target.target();
                target = next;
                if pair_changed {
                    // The pair (format, colour space) drives the encoding and
                    // the graph: rebuild the app's target-dependent objects.
                    // A resize only moves the extent (the graph redeclares).
                    // When the app replaced what it binds (the recreate
                    // waited for the device), no external image keeps the
                    // old one's layout under its reused identity.
                    if app.retarget(&gpu, target, config.frames_in_flight)? {
                        executor.forget_external_states();
                    }
                }
                metadata_pending = true;
            }
            if let Some(min_frame) = min_frame {
                let elapsed = last_frame.elapsed();
                if elapsed < min_frame {
                    std::thread::sleep(min_frame - elapsed);
                }
            }
            last_frame = Instant::now();

            let (graph, backbuffer) = app.declare(target)?;
            if executor.set_graph(graph)? {
                app.on_rebake()?;
                if trace {
                    dump_plan(&executor, &[], true);
                }
            }
            let token = executor.begin_frame()?;
            let acquire_started = Instant::now();
            let (image, suboptimal) = match swapchain.acquire(executor.image_available(token))? {
                Acquired::Image { index, suboptimal } => (index, suboptimal),
                Acquired::OutOfDate => {
                    dirty = true;
                    continue;
                }
                Acquired::TimedOut => {
                    stats.acquire_timeouts += 1;
                    continue;
                }
            };
            stats.acquire_ms = acquire_started.elapsed().as_secs_f32() * 1000.0;
            dirty |= suboptimal;
            app.frame_stats(&stats.executor);
            app.prepare(token.slot)?;
            if metadata_pending {
                if let Some((levels, gamut)) = app.hdr_metadata() {
                    swapchain.set_hdr_metadata(&gpu, levels, gamut)?;
                }
                stats.output = output_label(target, app.hdr_metadata());
                metadata_pending = false;
            }
            let resources = executor.graph().resources.len();
            let back = Binding::Image(ImageBinding {
                image: swapchain.images[image as usize],
                view: swapchain.views[image as usize],
                format: target.format,
                extent: swapchain.extent,
                range: color_range(),
                identity: (swapchain.generation << 32) | image as u64,
            });
            let externals: Vec<Option<Binding>> = (0..resources as u32)
                .map(|index| {
                    let resource = ResourceId(index);
                    if Some(resource) == backbuffer {
                        Some(back)
                    } else {
                        app.external(resource, token.slot)
                    }
                })
                .collect();
            executor.execute(
                token,
                app.recorder(),
                &|resource| externals.get(resource.index()).copied().flatten(),
                &[(executor_wait_semaphore(&executor, token), vk::PipelineStageFlags2::ALL_COMMANDS)],
                &[swapchain.render_finished[image as usize]],
            )?;
            let present_started = Instant::now();
            if !swapchain.present(gpu.queue, image)? {
                dirty = true;
            }
            stats.present_ms = present_started.elapsed().as_secs_f32() * 1000.0;
            stats.frames += 1;
            stats.executor = executor.stats();
            if last_report.elapsed() >= Duration::from_millis(250) {
                stats.app_status = app.status();
                stats.validation_errors = gpu.instance.counters.errors();
                stats.validation_warnings = gpu.instance.counters.warnings();
                if let Ok(mut shared) = shared.lock() {
                    *shared = stats.clone();
                }
                last_report = Instant::now();
            }
        }
        Ok(())
    })();
    gpu.wait_idle();
    stats.executor = executor.stats();
    if trace {
        dump_plan(&executor, &stats.executor.pass_gpu_ms, false);
    }
    stats.app_status = app.status();
    if let Ok(mut shared) = shared.lock() {
        let error = shared.error.take();
        *shared = stats;
        shared.error = error;
    }
    if gpu.is_hung() {
        // The GPU may still be using these objects and no wait is bounded:
        // leak them (and the surface, see `stop`) so the thread can exit.
        control.leaked.store(true, Ordering::Release);
        std::mem::forget(executor);
        std::mem::forget(app);
        std::mem::forget(swapchain);
        std::mem::forget(gpu);
        return result;
    }
    drop(executor);
    app.destroy(&gpu);
    swapchain.destroy(&gpu);
    drop(gpu);
    result
}

/// [`GpuRenderStats::output`] for `target` and the app's HDR metadata.
fn output_label(target: TargetInfo, metadata: Option<(crate::encode::DisplayLevels, crate::colorspace::Gamut)>) -> String {
    let encoding = target.target().encoding().label();
    match metadata {
        Some((levels, _)) => format!("{encoding}, white {:.0}, peak {:.0} nits", levels.paper_white, levels.peak),
        None => encoding.into(),
    }
}

/// Drives an app offscreen in steps, keeping one executor (so rebakes
/// across steps are counted and only happen when the topology changes).
pub struct Headless<P> {
    gpu: Arc<GpuDevice>,
    executor: Executor<P>,
    target: TargetInfo,
    last: Option<crate::executor::FrameToken>,
}

impl<P: Sync> Headless<P> {
    /// Create the executor and the app's GPU objects.
    pub fn new<A: RenderApp<Pass = P>>(gpu: &Arc<GpuDevice>, app: &mut A, target: TargetInfo, config: ExecutorConfig) -> Result<Self> {
        let executor = Executor::new(gpu.clone(), config)?;
        app.create(gpu, target, config.frames_in_flight)?;
        Ok(Self { gpu: gpu.clone(), executor, target, last: None })
    }

    /// Render `frames` frames.
    pub fn frames<A: RenderApp<Pass = P>>(&mut self, app: &mut A, frames: usize) -> Result<()> {
        for _ in 0..frames {
            let (graph, backbuffer) = app.declare(self.target)?;
            anyhow::ensure!(backbuffer.is_none(), "headless graphs have no backbuffer");
            if self.executor.set_graph(graph)? {
                app.on_rebake()?;
            }
            let token = self.executor.begin_frame()?;
            app.frame_stats(&self.executor.stats());
            app.prepare(token.slot)?;
            let externals: Vec<Option<Binding>> = (0..self.executor.graph().resources.len() as u32).map(|index| app.external(ResourceId(index), token.slot)).collect();
            self.executor.execute(token, app.recorder(), &|resource| externals.get(resource.index()).copied().flatten(), &[], &[])?;
            self.last = Some(token);
        }
        Ok(())
    }

    /// Wait until everything submitted has finished (readbacks are then
    /// valid).
    pub fn wait(&mut self) -> Result<()> {
        if let Some(token) = self.last.take() {
            self.executor.wait(token)?;
        }
        self.gpu.wait_idle();
        Ok(())
    }

    pub fn stats(&self) -> ExecutorStats {
        self.executor.stats()
    }

    /// See [`Executor::take_gpu_frames`].
    pub fn take_gpu_frames(&mut self) -> Vec<Vec<f32>> {
        self.executor.take_gpu_frames()
    }

    /// Wait, then drop the executor; its statistics.
    pub fn finish(mut self) -> Result<ExecutorStats> {
        self.wait()?;
        Ok(self.executor.stats())
    }
}

/// Render `frames` frames offscreen (see [`Headless`]) and wait for them.
pub fn run_headless<A: RenderApp>(gpu: &Arc<GpuDevice>, app: &mut A, target: TargetInfo, frames: usize, config: ExecutorConfig) -> Result<ExecutorStats> {
    let mut headless = Headless::new(gpu, app, target, config)?;
    headless.frames(app, frames)?;
    headless.finish()
}

fn executor_wait_semaphore<R>(executor: &Executor<R>, token: crate::executor::FrameToken) -> vk::Semaphore {
    executor.image_available(token)
}

/// AE_RENDER_TRACE=1: print the baked plan and write `frame_graph.dot`
/// (with average GPU pass times once they exist).
fn dump_plan<R>(executor: &Executor<R>, times_ms: &[f32], print: bool) {
    let Some(baked) = executor.baked() else { return };
    let graph = executor.graph();
    let dot = ae_framegraph::dot::to_dot_with_times(graph, baked, times_ms);
    if let Err(err) = std::fs::write("frame_graph.dot", dot) {
        eprintln!("[render] writing frame_graph.dot: {err}");
    }
    if !print {
        return;
    }
    println!(
        "[render] baked plan {:#018x}: {} passes, {} physical resources (~{} KiB/slot), {} alias handoffs",
        baked.signature,
        baked.passes.len(),
        baked.physical.len(),
        baked.estimated_bytes() / 1024,
        baked.handoffs.len()
    );
    for (index, pass) in baked.passes.iter().enumerate() {
        let accesses: Vec<String> = pass
            .accesses
            .iter()
            .map(|access| {
                format!(
                    "{}:{:?}{}",
                    graph.resource_name(access.resource),
                    access.role,
                    if access.discard { " (discard)" } else { "" }
                )
            })
            .collect();
        println!("[render]   {index}: {} [{:?}] {}", graph.pass_name(pass.pass), pass.kind, accesses.join(", "));
    }
}
