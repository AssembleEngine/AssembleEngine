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
use crate::swapchain::{Acquired, PresentPolicy, Swapchain, color_range};
use crate::window::{SdlWindow, SurfaceSource, WindowEvents};
use anyhow::{Context as _, Result};
use ash::vk;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The render target of the current topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetInfo {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
}

pub trait RenderApp: Send + 'static {
    type Pass: Send + 'static;

    /// (Re)create format-dependent GPU objects (pipelines). Called once and
    /// whenever the swapchain format changes.
    fn create(&mut self, gpu: &Arc<GpuDevice>, format: vk::Format, frames_in_flight: usize) -> Result<()>;

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

#[derive(Clone, Copy, Debug)]
pub struct GpuRenderConfig {
    pub frames_in_flight: usize,
    pub present: PresentPolicy,
    /// Cap on frames per second (0 = present-paced only).
    pub max_fps: u32,
    pub profile: bool,
    pub verbose: bool,
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
    minimized: AtomicBool,
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
    let mut swapchain = Swapchain::new(&gpu, surface, initial_size, config.present)?;
    let mut executor: Executor<A::Pass> = Executor::new(
        gpu.clone(),
        ExecutorConfig {
            frames_in_flight: config.frames_in_flight,
            profile: config.profile,
            ..ExecutorConfig::default()
        },
    )?;
    let mut format = swapchain.format.format;
    app.create(&gpu, format, config.frames_in_flight)?;
    let mut stats = GpuRenderStats {
        device: gpu.device_name.clone(),
        present_mode: format!("{:?}", swapchain.present_mode),
        ..GpuRenderStats::default()
    };
    if config.verbose {
        println!(
            "[render] GPU: {} | swapchain {:?} {}x{} {:?} | {} frames in flight | validation {}",
            gpu.device_name,
            swapchain.format.format,
            swapchain.extent.width,
            swapchain.extent.height,
            swapchain.present_mode,
            config.frames_in_flight,
            gpu.instance.validation,
        );
    }
    let mut dirty = false;
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
            if control.minimized.load(Ordering::Acquire) || size.0 == 0 || size.1 == 0 {
                stats.skipped_minimized += 1;
                std::thread::sleep(Duration::from_millis(16));
                continue;
            }
            if dirty || !swapchain.ready() {
                if !swapchain.recreate(&gpu, size)? {
                    std::thread::sleep(Duration::from_millis(16));
                    continue;
                }
                stats.swapchain_recreates += 1;
                dirty = false;
                if swapchain.format.format != format {
                    format = swapchain.format.format;
                    app.destroy(&gpu);
                    app.create(&gpu, format, config.frames_in_flight)?;
                }
            }
            if let Some(min_frame) = min_frame {
                let elapsed = last_frame.elapsed();
                if elapsed < min_frame {
                    std::thread::sleep(min_frame - elapsed);
                }
            }
            last_frame = Instant::now();

            let target = TargetInfo {
                extent: swapchain.extent,
                format,
            };
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
            let resources = executor.graph().resources.len();
            let back = Binding::Image(ImageBinding {
                image: swapchain.images[image as usize],
                view: swapchain.views[image as usize],
                format,
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

/// Drive `app` for `frames` frames without a window or swapchain (tests,
/// offscreen captures). The app's graph must not contain a backbuffer.
pub fn run_headless<A: RenderApp>(
    gpu: &Arc<GpuDevice>,
    app: &mut A,
    target: TargetInfo,
    frames: usize,
    config: ExecutorConfig,
) -> Result<ExecutorStats> {
    let mut executor: Executor<A::Pass> = Executor::new(gpu.clone(), config)?;
    app.create(gpu, target.format, config.frames_in_flight)?;
    let mut last = None;
    for _ in 0..frames {
        let (graph, backbuffer) = app.declare(target)?;
        anyhow::ensure!(backbuffer.is_none(), "headless graphs have no backbuffer");
        if executor.set_graph(graph)? {
            app.on_rebake()?;
        }
        let token = executor.begin_frame()?;
        app.frame_stats(&executor.stats());
        app.prepare(token.slot)?;
        let externals: Vec<Option<Binding>> = (0..executor.graph().resources.len() as u32)
            .map(|index| app.external(ResourceId(index), token.slot))
            .collect();
        executor.execute(
            token,
            app.recorder(),
            &|resource| externals.get(resource.index()).copied().flatten(),
            &[],
            &[],
        )?;
        last = Some(token);
    }
    if let Some(token) = last {
        executor.wait(token)?;
    }
    gpu.wait_idle();
    let stats = executor.stats();
    drop(executor);
    Ok(stats)
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
