//! The terrain as a [`RenderApp`]: a LOD compute pass, a draw pass into the
//! backbuffer (or an offscreen target read back to the host) with a
//! transient reverse-Z depth buffer, and present.
//!
//! The view (camera, error target, debug switches) is shared with the main
//! thread through [`TerrainApp::view`]; the render thread reads it once per
//! frame and never waits for anyone else.

use super::camera::Camera;
use super::heightmap::{Heightmap, Pyramid};
use super::renderer::{FrameParams, FrameStats, LodView, TerrainInfo, TerrainRenderer, Wireframe};
use crate::device::GpuDevice;
use crate::executor::{Binding, BufferBinding, ImageBinding, PassContext, Recorder};
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::thread::{RenderApp, TargetInfo};
use ae_framegraph::{BufferDesc, FrameGraph, ImageDesc, LoadOp, PassKind, ResourceClass, ResourceId, Role, ShaderStage};
use anyhow::{Context as _, Result};
use ash::vk;
use glam::Vec2;
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, Mutex};

pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
/// Sky / fog colour (linear).
pub const SKY: [f32; 4] = [0.62, 0.70, 0.80, 1.0];
/// sRGB like the swapchain, so a capture shows what the window shows.
const READBACK_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

/// The map: heights, error pyramid and how they map to the world. Built
/// once on the CPU, shared by the main thread (camera ground height) and the
/// render thread (uploads).
pub struct TerrainSource {
    pub heightmap: Heightmap,
    pub pyramid: Pyramid,
    pub info: TerrainInfo,
    /// Material id per cell (see [`TerrainInfo::has_materials`]).
    pub materials: Option<Vec<u8>>,
}

impl TerrainSource {
    /// World height of the terrain under world point (x, z).
    pub fn height_at(&self, world: Vec2) -> f32 {
        self.heightmap.sample(world / self.info.cell_size) * self.info.height_scale + self.info.height_offset
    }

    /// A camera looking at the middle of the map from the south.
    pub fn overview_camera(&self) -> Camera {
        let (w, h) = self.info.extent();
        let target = Vec2::new(w * 0.5, h * 0.5);
        Camera { target, ground: self.height_at(target), distance: w.max(h) * 0.6, yaw: 0.0, pitch: 0.75, fov_y: 0.8 }
    }
}

/// What the main thread controls.
#[derive(Clone, Copy, Debug)]
pub struct TerrainView {
    pub camera: Camera,
    pub tau: f32,
    pub wireframe: Wireframe,
    pub lod_colors: bool,
    pub fog: bool,
    /// Keep the subdivision of this view (fly around a frozen mesh).
    pub frozen: Option<LodView>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainPass {
    Lod,
    Draw,
    Copy,
    HostRead,
    Present,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainOutput {
    Present,
    /// Render into an app-owned image and copy it to a host buffer
    /// ([`TerrainApp::readback`]); for tests and captures.
    Readback,
}

struct Ids {
    leaves: ResourceId,
    indirect: ResourceId,
    target: Option<ResourceId>,
    readback: Option<ResourceId>,
}

struct Offscreen {
    image: OwnedImage,
    buffer: OwnedBuffer,
    extent: vk::Extent2D,
}

pub struct TerrainApp {
    source: Arc<TerrainSource>,
    view: Arc<Mutex<TerrainView>>,
    stats: Arc<Mutex<FrameStats>>,
    output: TerrainOutput,
    renderer: Option<TerrainRenderer>,
    offscreen: Option<Offscreen>,
    ids: Option<Ids>,
    extent: vk::Extent2D,
    update_lod: bool,
    wireframe: Wireframe,
}

impl TerrainApp {
    pub fn new(source: Arc<TerrainSource>, output: TerrainOutput) -> Self {
        let view = TerrainView { camera: source.overview_camera(), tau: 2.0, wireframe: Wireframe::Off, lod_colors: false, fog: true, frozen: None };
        Self {
            source,
            view: Arc::new(Mutex::new(view)),
            stats: Arc::default(),
            output,
            renderer: None,
            offscreen: None,
            ids: None,
            extent: vk::Extent2D { width: 1, height: 1 },
            update_lod: true,
            wireframe: Wireframe::Off,
        }
    }

    /// The shared view; the main thread writes it, the render thread reads
    /// it once per frame.
    pub fn view(&self) -> Arc<Mutex<TerrainView>> {
        self.view.clone()
    }

    /// Leaf counts of the most recently completed frame.
    pub fn stats(&self) -> Arc<Mutex<FrameStats>> {
        self.stats.clone()
    }

    /// The last offscreen frame (RGBA8, sRGB-encoded, row-major), after
    /// `run_headless`.
    pub fn readback(&mut self) -> Option<Vec<u8>> {
        let offscreen = self.offscreen.as_mut()?;
        let bytes = (offscreen.extent.width * offscreen.extent.height * 4) as usize;
        offscreen.buffer.mapped().map(|data| data[..bytes].to_vec())
    }

    /// The camera matrices of `view` for the current target.
    fn params(&self, view: &TerrainView) -> FrameParams {
        let aspect = self.extent.width as f32 / self.extent.height.max(1) as f32;
        let mut camera = view.camera;
        camera.ground = self.source.height_at(camera.target);
        let eye = camera.eye(|p| self.source.height_at(p));
        let view_proj = camera.proj(aspect) * camera.view(eye);
        let live = LodView { pos: eye, view_proj, proj_scale: camera.proj_scale(self.extent.height as f32) };
        FrameParams {
            view_proj,
            cam_pos: eye,
            lod_view: view.frozen.unwrap_or(live),
            tau: view.tau,
            update_lod: view.frozen.is_none(),
            wireframe: view.wireframe,
            lod_colors: view.lod_colors,
            fog: view.fog,
        }
    }
}

/// The view a frozen LOD would keep (call on the main thread with the
/// target's size).
pub fn lod_view(source: &TerrainSource, camera: &Camera, width: u32, height: u32) -> LodView {
    let mut camera = *camera;
    camera.ground = source.height_at(camera.target);
    let eye = camera.eye(|p| source.height_at(p));
    let view_proj = camera.proj(width as f32 / height.max(1) as f32) * camera.view(eye);
    LodView { pos: eye, view_proj, proj_scale: camera.proj_scale(height as f32) }
}

impl RenderApp for TerrainApp {
    type Pass = TerrainPass;

    fn create(&mut self, gpu: &Arc<GpuDevice>, format: vk::Format, frames_in_flight: usize) -> Result<()> {
        if let Some(mut old) = self.renderer.take() {
            old.destroy(gpu);
        }
        let color = if self.output == TerrainOutput::Readback { READBACK_FORMAT } else { format };
        let s = &self.source;
        self.renderer = Some(TerrainRenderer::new(gpu, &s.heightmap, &s.pyramid, s.info, s.materials.as_deref(), color, DEPTH_FORMAT, frames_in_flight).context("terrain renderer")?);
        Ok(())
    }

    fn declare(&mut self, target: TargetInfo) -> Result<(FrameGraph<TerrainPass>, Option<ResourceId>)> {
        let renderer = self.renderer.as_ref().context("terrain renderer not created")?;
        self.extent = target.extent;
        let (width, height) = (target.extent.width, target.extent.height);
        let mut graph = FrameGraph::new();
        let leaves = graph.buffer("terrain leaves", renderer.leaves_desc(), ResourceClass::External);
        let indirect = graph.buffer("terrain indirect", renderer.indirect_desc(), ResourceClass::External);
        let depth = graph.image("depth", ImageDesc::depth_2d(DEPTH_FORMAT, width, height, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT), ResourceClass::Transient);
        let (color, class) = match self.output {
            TerrainOutput::Present => (ImageDesc::color_2d(target.format, width, height, vk::ImageUsageFlags::COLOR_ATTACHMENT), ResourceClass::Backbuffer),
            TerrainOutput::Readback => (
                ImageDesc::color_2d(READBACK_FORMAT, width, height, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC),
                ResourceClass::External,
            ),
        };
        let target_id = graph.image("target", color, class);
        graph
            .pass("terrain lod", PassKind::Compute, TerrainPass::Lod)
            .access(leaves, Role::StorageBufferWrite(ShaderStage::Compute))
            .access(indirect, Role::StorageBufferWrite(ShaderStage::Compute))
            .id();
        graph
            .pass("terrain", PassKind::Graphics, TerrainPass::Draw)
            .access(leaves, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(indirect, Role::IndirectBuffer)
            .color(target_id, LoadOp::Clear)
            .depth(depth, LoadOp::Clear)
            .id();
        let mut ids = Ids { leaves, indirect, target: None, readback: None };
        let backbuffer = match self.output {
            TerrainOutput::Present => {
                graph.pass("present", PassKind::Present, TerrainPass::Present).present(target_id).root().id();
                Some(target_id)
            }
            TerrainOutput::Readback => {
                let size = width as u64 * height as u64 * 4;
                let readback = graph.buffer("readback", BufferDesc { size, usage: vk::BufferUsageFlags::TRANSFER_DST }, ResourceClass::External);
                graph.pass("copy", PassKind::Transfer, TerrainPass::Copy).access(target_id, Role::TransferSrc).discard(readback, Role::TransferDst).id();
                graph.pass("host read", PassKind::Transfer, TerrainPass::HostRead).access(readback, Role::HostRead).root().id();
                ids.target = Some(target_id);
                ids.readback = Some(readback);
                None
            }
        };
        self.ids = Some(ids);
        Ok((graph, backbuffer))
    }

    fn prepare(&mut self, slot: usize) -> Result<()> {
        let view = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        let params = self.params(&view);
        self.update_lod = params.update_lod;
        self.wireframe = params.wireframe;
        let renderer = self.renderer.as_mut().context("terrain renderer not created")?;
        let stats = renderer.read_stats(slot);
        *self.stats.lock().unwrap_or_else(|p| p.into_inner()) = stats;
        renderer.write_frame(slot, &params);
        Ok(())
    }

    fn external(&self, resource: ResourceId, _slot: usize) -> Option<Binding> {
        let (renderer, ids) = (self.renderer.as_ref()?, self.ids.as_ref()?);
        if resource == ids.leaves {
            return Some(Binding::Buffer(renderer.leaves()));
        }
        if resource == ids.indirect {
            return Some(Binding::Buffer(renderer.indirect()));
        }
        let offscreen = self.offscreen.as_ref()?;
        if Some(resource) == ids.target {
            return Some(Binding::Image(ImageBinding {
                image: offscreen.image.image,
                view: offscreen.image.view,
                format: READBACK_FORMAT,
                extent: offscreen.extent,
                range: offscreen.image.range(),
                identity: (0xAE7 << 40) | 10,
            }));
        }
        if Some(resource) == ids.readback {
            return Some(Binding::Buffer(BufferBinding { buffer: offscreen.buffer.buffer, offset: 0, size: offscreen.buffer.size, identity: (0xAE7 << 40) | 11 }));
        }
        None
    }

    fn recorder(&mut self) -> &mut dyn Recorder<TerrainPass> {
        self
    }

    fn status(&self) -> String {
        let stats = *self.stats.lock().unwrap_or_else(|p| p.into_inner());
        format!("terrain: {} leaves, {} drawn", stats.leaves, stats.drawn)
    }

    fn destroy(&mut self, gpu: &Arc<GpuDevice>) {
        if let Some(mut renderer) = self.renderer.take() {
            renderer.destroy(gpu);
        }
        if let Some(mut offscreen) = self.offscreen.take() {
            offscreen.image.destroy(gpu);
            offscreen.buffer.destroy(gpu);
        }
    }
}

impl TerrainApp {
    /// Create the offscreen target and readback buffer for `extent`
    /// (readback output only; call before `run_headless`).
    pub fn create_offscreen(&mut self, gpu: &GpuDevice, extent: vk::Extent2D) -> Result<()> {
        let desc = ImageDesc::color_2d(READBACK_FORMAT, extent.width, extent.height, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC);
        let image = OwnedImage::new(gpu, desc, "terrain offscreen")?;
        let size = extent.width as u64 * extent.height as u64 * 4;
        let buffer = OwnedBuffer::new(gpu, BufferDesc { size, usage: vk::BufferUsageFlags::TRANSFER_DST }, MemoryLocation::GpuToCpu, "terrain readback")?;
        self.offscreen = Some(Offscreen { image, buffer, extent });
        Ok(())
    }
}

impl Recorder<TerrainPass> for TerrainApp {
    fn clear_color(&self, _pass: &TerrainPass, _resource: ResourceId) -> [f32; 4] {
        SKY
    }

    fn clear_depth(&self, _pass: &TerrainPass, _resource: ResourceId) -> (f32, u32) {
        (0.0, 0) // reverse-Z
    }

    fn record(&mut self, pass: &TerrainPass, ctx: &PassContext<'_>) -> Result<()> {
        let renderer = self.renderer.as_ref().context("terrain renderer not created")?;
        match pass {
            TerrainPass::Lod => renderer.record_lod(ctx, self.update_lod),
            TerrainPass::Draw => renderer.record_draw(ctx, self.wireframe),
            TerrainPass::Copy => {
                let ids = self.ids.as_ref().context("not declared")?;
                let image = ctx.image(ids.target.context("no target")?)?;
                let buffer = ctx.buffer(ids.readback.context("no readback")?)?;
                crate::demo::record_readback(ctx, image, buffer);
            }
            TerrainPass::HostRead | TerrainPass::Present => {}
        }
        Ok(())
    }
}
