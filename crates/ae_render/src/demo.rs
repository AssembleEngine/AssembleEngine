//! The R2/R4 demo graph: upload → triangle → two compute chains → composite
//! → output.
//!
//! The baseline's demo ("blit halves → present") wrote the backbuffer from
//! two passes and chained compute writes in place, both of which the
//! single-writer rule forbids. Here each chain ping-pongs through distinct
//! transients and one composite pass writes the target:
//!
//! ```text
//! upload params (staging → params) ─────────────────────────┐ (uniform, every filter)
//! triangle ─ scene ─┬─ blur_h ─ a1 ─ blur_v ─ a2 ─┐         │
//!                   └─ invert ─ b1 ─ tint   ─ b2 ─┴─ composite ─ target ─ present | readback
//! ```
//!
//! The per-frame tint is written by the host into a per-slot staging buffer
//! and copied by the transfer pass, so the chain upload → compute → graphics
//! → present runs with no CPU wait on the GPU (R4).
//!
//! With aliasing, `b1` reuses `a1`'s allocation (equal description, disjoint
//! lifetimes).

use ae_framegraph::{
    BufferDesc, FrameGraph, ImageDesc, LoadOp, PassKind, ResourceClass, ResourceId, Role, ShaderStage,
};
use ash::vk;

pub const SCENE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemoPass {
    /// Copy the per-frame parameters from staging.
    Upload {
        src: ResourceId,
        dst: ResourceId,
    },
    Triangle,
    /// Filter `mode` from `src` to `dst` (see `demo_filter.comp`).
    Filter {
        mode: i32,
        src: ResourceId,
        dst: ResourceId,
        params: ResourceId,
    },
    Composite {
        left: ResourceId,
        right: ResourceId,
    },
    Readback {
        src: ResourceId,
        dst: ResourceId,
    },
    /// Terminal host-visibility pass for the readback buffer (records
    /// nothing; its barrier makes the copy visible to the host).
    HostRead,
    Present,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemoOutput {
    /// Composite into the swapchain image and present.
    Present { format: vk::Format },
    /// Composite into an external offscreen image and copy it into an
    /// external host-visible buffer (tests, golden hashes).
    Readback,
}

pub struct DemoResources {
    pub target: ResourceId,
    pub readback: Option<ResourceId>,
    /// External per-slot staging buffer holding the 16-byte tint.
    pub params_staging: ResourceId,
}

/// Bytes of the per-frame parameter block (`vec4 tint`).
pub const PARAMS_BYTES: u64 = 16;

/// The tint for a frame (std140 `vec4`).
pub fn params_bytes(tint: [f32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (index, value) in tint.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    out
}

/// The tint of frame `frame` (a slow colour cycle; additions and
/// multiplications only, so it is deterministic).
pub fn demo_tint(frame: u64) -> [f32; 4] {
    let t = (frame % 240) as f32 / 240.0;
    [1.0 - t * 0.5, 0.6 + t * 0.4, 0.3 + t * 0.2, 1.0]
}

pub fn build_demo_graph(extent: vk::Extent2D, output: DemoOutput) -> (FrameGraph<DemoPass>, DemoResources) {
    let mut graph = FrameGraph::new();
    let (width, height) = (extent.width, extent.height);
    let work_usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
    let scene = graph.image(
        "scene",
        ImageDesc::color_2d(
            SCENE_FORMAT,
            width,
            height,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | work_usage,
        ),
        ResourceClass::Transient,
    );
    let work = |graph: &mut FrameGraph<DemoPass>, name: &str| {
        graph.image(
            name,
            ImageDesc::color_2d(SCENE_FORMAT, width, height, work_usage),
            ResourceClass::Transient,
        )
    };
    let params_staging = graph.buffer(
        "params staging",
        BufferDesc {
            size: PARAMS_BYTES,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
        },
        ResourceClass::External,
    );
    let params = graph.buffer(
        "params",
        BufferDesc {
            size: PARAMS_BYTES,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        },
        ResourceClass::Transient,
    );
    let a1 = work(&mut graph, "a1");
    let a2 = work(&mut graph, "a2");
    let b1 = work(&mut graph, "b1");
    let b2 = work(&mut graph, "b2");
    let (target, readback) = match output {
        DemoOutput::Present { format } => (
            graph.image(
                "backbuffer",
                ImageDesc::color_2d(format, width, height, vk::ImageUsageFlags::COLOR_ATTACHMENT),
                ResourceClass::Backbuffer,
            ),
            None,
        ),
        DemoOutput::Readback => {
            let target = graph.image(
                "target",
                ImageDesc::color_2d(
                    SCENE_FORMAT,
                    width,
                    height,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                ),
                ResourceClass::External,
            );
            let readback = graph.buffer(
                "readback",
                BufferDesc {
                    size: width as u64 * height as u64 * 4,
                    usage: vk::BufferUsageFlags::TRANSFER_DST,
                },
                ResourceClass::External,
            );
            (target, Some(readback))
        }
    };

    graph
        .pass(
            "upload params",
            PassKind::Transfer,
            DemoPass::Upload {
                src: params_staging,
                dst: params,
            },
        )
        .access(params_staging, Role::TransferSrc)
        .discard(params, Role::TransferDst)
        .id();
    graph
        .pass("triangle", PassKind::Graphics, DemoPass::Triangle)
        .color(scene, LoadOp::Clear)
        .id();
    let filter = |graph: &mut FrameGraph<DemoPass>, name: &str, mode, src, dst| {
        graph
            .pass(name, PassKind::Compute, DemoPass::Filter { mode, src, dst, params })
            .access(src, Role::StorageImageRead(ShaderStage::Compute))
            .discard(dst, Role::StorageImageWrite(ShaderStage::Compute))
            .access(params, Role::StorageBufferRead(ShaderStage::Compute))
            .id();
    };
    filter(&mut graph, "blur_h", 0, scene, a1);
    filter(&mut graph, "blur_v", 1, a1, a2);
    filter(&mut graph, "invert", 2, scene, b1);
    filter(&mut graph, "tint", 3, b1, b2);
    graph
        .pass(
            "composite",
            PassKind::Graphics,
            DemoPass::Composite { left: a2, right: b2 },
        )
        .sampled(a2, ShaderStage::Fragment)
        .sampled(b2, ShaderStage::Fragment)
        .color(target, LoadOp::DontCare)
        .id();
    match readback {
        Some(buffer) => {
            graph
                .pass(
                    "readback",
                    PassKind::Transfer,
                    DemoPass::Readback {
                        src: target,
                        dst: buffer,
                    },
                )
                .access(target, Role::TransferSrc)
                .discard(buffer, Role::TransferDst)
                .id();
            graph
                .pass("host read", PassKind::Transfer, DemoPass::HostRead)
                .access(buffer, Role::HostRead)
                .root()
                .id();
        }
        None => {
            graph
                .pass("present", PassKind::Present, DemoPass::Present)
                .present(target)
                .root()
                .id();
        }
    }
    (
        graph,
        DemoResources {
            target,
            readback,
            params_staging,
        },
    )
}

pub use gpu::{DemoApp, DemoRenderer, TextZoom, record_readback};

mod gpu {
    use super::*;
    use crate::device::GpuDevice;
    use crate::executor::{PassContext, Recorder};
    use crate::pipeline::{self, Blend, GraphicsDesc, spirv};
    use crate::text::{TextBatch, screen_to_clip};
    use ae_text::parley::{FontStyle, FontWeight, style::StyleProperty};
    use ae_text::{Atlas, Shadow, TextSystem};
    use anyhow::Result;
    use glam::{Mat4, Vec3};
    use std::ops::Range;
    use std::sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    };

    /// Shares the nameplate zoom between SDL's event thread and the renderer.
    #[derive(Clone)]
    pub struct TextZoom(Arc<AtomicU32>);

    impl Default for TextZoom {
        fn default() -> Self {
            Self(Arc::new(AtomicU32::new(1.0_f32.to_bits())))
        }
    }

    impl TextZoom {
        pub fn scroll(&self, steps: f32) {
            if steps.is_finite() {
                let scale = (self.scale() * 1.25_f32.powf(steps)).clamp(0.35, 8.0);
                self.0.store(scale.to_bits(), Ordering::Relaxed);
            }
        }

        fn scale(&self) -> f32 {
            f32::from_bits(self.0.load(Ordering::Relaxed))
        }
    }

    /// Pipelines for the demo passes. Every descriptor is a global-set
    /// handle (M15 step E): each pass pushes the graph's indices and the
    /// demo's sampler index and binds the executor's set 0.
    pub struct DemoRenderer {
        gpu: Arc<GpuDevice>,
        sampler_index: u32,
        layout: vk::PipelineLayout,
        triangle: vk::Pipeline,
        filter: vk::Pipeline,
        composite: vk::Pipeline,
        composite_format: vk::Format,
        text: Option<DemoText>,
    }

    impl DemoRenderer {
        pub fn new(
            gpu: Arc<GpuDevice>,
            bindless: &mut crate::bindless::Bindless,
            target_format: vk::Format,
        ) -> Result<Self> {
            let device = &gpu.device;
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            let sampler_index = bindless.sampler(&sampler_info)?.index;
            let layout = bindless.pipeline_layout();
            let triangle = pipeline::graphics_pipeline(
                device,
                &GraphicsDesc {
                    vertex: spirv!("triangle.vert"),
                    fragment: spirv!("triangle.frag"),
                    layout,
                    color_formats: &[SCENE_FORMAT],
                    depth: None,
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    blend: Blend::Off,
                    glow_blend: Blend::Off,
                },
            )?;
            let filter = pipeline::compute_pipeline(device, spirv!("demo_filter.comp"), layout)?;
            let composite = pipeline::graphics_pipeline(
                device,
                &GraphicsDesc {
                    vertex: spirv!("fullscreen.vert"),
                    fragment: spirv!("composite.frag"),
                    layout,
                    color_formats: &[target_format],
                    depth: None,
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    blend: Blend::Off,
                    glow_blend: Blend::Off,
                },
            )?;
            Ok(Self {
                gpu,
                sampler_index,
                layout,
                triangle,
                filter,
                composite,
                composite_format: target_format,
                text: None,
            })
        }

        pub fn with_text(self) -> Result<Self> {
            self.with_text_zoom(TextZoom::default())
        }

        fn with_text_zoom(mut self, zoom: TextZoom) -> Result<Self> {
            self.text = Some(DemoText::new(self.gpu.clone(), self.composite_format, zoom)?);
            Ok(self)
        }

        pub fn target_format(&self) -> vk::Format {
            self.composite_format
        }

        /// Bind the global set and push the pass's block.
        fn bind(&self, ctx: &PassContext<'_>, point: vk::PipelineBindPoint, push: crate::bindless::Push) {
            let all = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
            unsafe {
                ctx.device
                    .cmd_bind_descriptor_sets(ctx.cmd, point, self.layout, 0, &[ctx.global_set], &[]);
                ctx.device
                    .cmd_push_constants(ctx.cmd, self.layout, all, 0, pipeline::push_bytes(&push));
            }
        }
    }

    impl Recorder<DemoPass> for DemoRenderer {
        fn clear_color(&self, _pass: &DemoPass, _resource: ResourceId) -> [f32; 4] {
            [0.05, 0.05, 0.08, 1.0]
        }

        fn record(&mut self, pass: &DemoPass, ctx: &PassContext<'_>) -> Result<()> {
            let device = ctx.device;
            match *pass {
                DemoPass::Triangle => unsafe {
                    device.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::GRAPHICS, self.triangle);
                    device.cmd_draw(ctx.cmd, 3, 1, 0, 0);
                },
                DemoPass::Upload { src, dst } => {
                    let (src, dst) = (ctx.buffer(src)?, ctx.buffer(dst)?);
                    let region = vk::BufferCopy {
                        src_offset: src.offset,
                        dst_offset: dst.offset,
                        size: PARAMS_BYTES,
                    };
                    unsafe { device.cmd_copy_buffer(ctx.cmd, src.buffer, dst.buffer, &[region]) };
                }
                DemoPass::Filter { mode, src, dst, params } => {
                    let (src, dst, params) = (ctx.image(src)?, ctx.image(dst)?, ctx.buffer(params)?);
                    // Words: the filter, then the two storage images and the
                    // parameter block, as global indices.
                    let push = crate::bindless::Push {
                        words: [
                            mode as u32,
                            src.storage.unwrap_or(0),
                            dst.storage.unwrap_or(0),
                            params.storage.unwrap_or(0),
                        ],
                        ..Default::default()
                    };
                    unsafe {
                        device.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.filter);
                    }
                    self.bind(ctx, vk::PipelineBindPoint::COMPUTE, push);
                    unsafe {
                        device.cmd_dispatch(ctx.cmd, dst.extent.width.div_ceil(8), dst.extent.height.div_ceil(8), 1);
                    }
                }
                DemoPass::Composite { left, right } => {
                    let (left, right) = (ctx.image(left)?, ctx.image(right)?);
                    let mut push = crate::bindless::Push::default();
                    push.words[0] = left.sampled.unwrap_or(0);
                    push.words[1] = right.sampled.unwrap_or(0);
                    push.sampler(0, self.sampler_index);
                    unsafe {
                        device.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::GRAPHICS, self.composite);
                    }
                    self.bind(ctx, vk::PipelineBindPoint::GRAPHICS, push);
                    unsafe {
                        device.cmd_draw(ctx.cmd, 3, 1, 0, 0);
                    }
                    if let Some(text) = &self.text {
                        text.draw(ctx)?;
                    }
                }
                DemoPass::Readback { src, dst } => {
                    let (image, buffer) = (ctx.image(src)?, ctx.buffer(dst)?);
                    record_readback(ctx, image, buffer);
                }
                DemoPass::HostRead | DemoPass::Present => {}
            }
            Ok(())
        }
    }

    struct DemoText {
        batch: TextBatch,
        hud: Range<u32>,
        plane: Range<u32>,
        plane_width: f32,
        zoom: TextZoom,
    }

    impl DemoText {
        fn new(gpu: Arc<GpuDevice>, format: vk::Format, zoom: TextZoom) -> Result<Self> {
            let mut text = TextSystem::new(true);
            let mut atlas = Atlas::new(512)?;
            let shadow = Shadow {
                offset: [1.0, 2.0],
                color: [0.0, 0.0, 0.0, 0.85],
            };

            let hud = "AssembleEngine\nMSDF  ·  مرحبا  ·  नमस्ते\nScroll to zoom the nameplate";
            let mut builder = text.builder(hud, true);
            builder.push_default(StyleProperty::FontSize(20.0));
            builder.push(StyleProperty::FontSize(28.0), .."AssembleEngine".len());
            builder.push(StyleProperty::FontWeight(FontWeight::BOLD), .."AssembleEngine".len());
            let mut layout = builder.build(hud);
            layout.break_all_lines(None);
            let hud = atlas.append(&layout, Some(shadow))?;

            let plane = "World-space text";
            let mut builder = text.builder(plane, false);
            builder.push_default(StyleProperty::FontSize(52.0));
            builder.push_default(StyleProperty::FontStyle(FontStyle::Italic));
            builder.push_default(StyleProperty::Underline(true));
            builder.push_default(StyleProperty::Brush([1.0, 0.85, 0.45, 1.0]));
            let mut layout = builder.build(plane);
            layout.break_all_lines(None);
            let plane_width = layout.width();
            let plane = atlas.append(&layout, Some(shadow))?;
            anyhow::ensure!(
                !hud.is_empty() && !plane.is_empty(),
                "the text demo needs installed sans-serif fonts"
            );

            let batch = TextBatch::new(gpu, &atlas, format, None)?;
            Ok(Self {
                batch,
                hud,
                plane,
                plane_width,
                zoom,
            })
        }

        fn draw(&self, ctx: &PassContext<'_>) -> Result<()> {
            let extent = ctx.render_area.extent;
            let aspect = extent.width as f32 / extent.height as f32;
            let projection = Mat4::perspective_rh(50.0_f32.to_radians(), aspect, 0.1, 100.0);
            let plane_to_view = Mat4::from_translation(Vec3::new(0.0, 0.45, -4.0))
                * Mat4::from_rotation_y(-0.62)
                * Mat4::from_rotation_x(0.33)
                * Mat4::from_scale(Vec3::splat(0.007 * self.zoom.scale()))
                * Mat4::from_translation(Vec3::new(-self.plane_width * 0.5, 0.0, 0.0));
            self.batch.draw(ctx, self.plane.clone(), projection * plane_to_view)?;
            self.batch
                .draw(ctx, self.hud.clone(), screen_to_clip(extent, [24.0, 20.0]))
        }
    }

    /// Copy a colour image (in `TRANSFER_SRC_OPTIMAL`) into a buffer.
    pub fn record_readback(
        ctx: &PassContext<'_>,
        image: crate::executor::ImageBinding,
        buffer: crate::executor::BufferBinding,
    ) {
        let region = vk::BufferImageCopy::default()
            .buffer_offset(buffer.offset)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: image.extent.width,
                height: image.extent.height,
                depth: 1,
            });
        unsafe {
            ctx.device.cmd_copy_image_to_buffer(
                ctx.cmd,
                image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.buffer,
                &[region],
            )
        };
    }

    /// The demo as a windowed `RenderApp` (composite into the swapchain).
    /// Owns the per-slot parameter staging buffers and animates the tint.
    #[derive(Default)]
    pub struct DemoApp {
        renderer: Option<DemoRenderer>,
        staging: Vec<crate::memory::OwnedBuffer>,
        params_staging: Option<ResourceId>,
        frame: u64,
        text: Option<TextZoom>,
    }

    impl DemoApp {
        pub fn with_text(zoom: TextZoom) -> Self {
            Self {
                text: Some(zoom),
                ..Self::default()
            }
        }
    }

    impl crate::thread::RenderApp for DemoApp {
        type Pass = DemoPass;

        fn create(
            &mut self,
            gpu: &Arc<GpuDevice>,
            bindless: &mut crate::bindless::Bindless,
            target: crate::thread::TargetInfo,
            frames_in_flight: usize,
        ) -> Result<()> {
            let format = target.format;
            let renderer = DemoRenderer::new(gpu.clone(), bindless, format)?;
            self.renderer = Some(match &self.text {
                Some(zoom) => renderer.with_text_zoom(zoom.clone())?,
                None => renderer,
            });
            self.staging = (0..frames_in_flight)
                .map(|slot| {
                    crate::memory::OwnedBuffer::new(
                        gpu,
                        BufferDesc {
                            size: PARAMS_BYTES,
                            usage: vk::BufferUsageFlags::TRANSFER_SRC,
                        },
                        gpu_allocator::MemoryLocation::CpuToGpu,
                        &format!("params staging [slot {slot}]"),
                    )
                })
                .collect::<Result<_>>()?;
            Ok(())
        }

        fn declare(
            &mut self,
            target: crate::thread::TargetInfo,
            _bindless: &mut crate::bindless::Bindless,
        ) -> Result<(FrameGraph<DemoPass>, Option<ResourceId>)> {
            let (graph, resources) = build_demo_graph(target.extent, DemoOutput::Present { format: target.format });
            self.params_staging = Some(resources.params_staging);
            Ok((graph, Some(resources.target)))
        }

        fn prepare(&mut self, _bindless: &mut crate::bindless::Bindless, slot: usize) -> Result<()> {
            self.frame += 1;
            let bytes = params_bytes(demo_tint(self.frame));
            let staging = self
                .staging
                .get_mut(slot)
                .and_then(|buffer| buffer.mapped())
                .ok_or_else(|| anyhow::anyhow!("params staging is not host visible"))?;
            staging[..bytes.len()].copy_from_slice(&bytes);
            Ok(())
        }

        fn external(&self, resource: ResourceId, slot: usize) -> Option<crate::executor::Binding> {
            if Some(resource) != self.params_staging {
                return None;
            }
            self.staging.get(slot).map(|buffer| {
                crate::executor::Binding::Buffer(crate::executor::BufferBinding {
                    buffer: buffer.buffer,
                    offset: 0,
                    size: PARAMS_BYTES,
                    identity: (7 << 40) | slot as u64,
                    storage: None,
                })
            })
        }

        fn recorder(&mut self) -> &mut dyn Recorder<DemoPass> {
            self.renderer.as_mut().expect("created before recording")
        }

        fn status(&self) -> String {
            if self.text.is_some() {
                "demo graph with MSDF text"
            } else {
                "demo graph"
            }
            .into()
        }

        fn destroy(&mut self, gpu: &Arc<GpuDevice>) {
            self.renderer = None;
            for mut buffer in self.staging.drain(..) {
                buffer.destroy(gpu);
            }
        }
    }

    impl Drop for DemoRenderer {
        fn drop(&mut self) {
            let device = &self.gpu.device;
            self.gpu.wait_idle();
            unsafe {
                device.destroy_pipeline(self.triangle, None);
                device.destroy_pipeline(self.filter, None);
                device.destroy_pipeline(self.composite, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_framegraph::{BakeOptions, bake};

    #[test]
    fn demo_graph_bakes_and_aliases() {
        let extent = vk::Extent2D { width: 64, height: 32 };
        for output in [
            DemoOutput::Readback,
            DemoOutput::Present {
                format: vk::Format::B8G8R8A8_SRGB,
            },
        ] {
            let (graph, _) = build_demo_graph(extent, output);
            let plain = bake(&graph, BakeOptions { aliasing: false }).expect("bakes");
            let aliased = bake(&graph, BakeOptions::default()).expect("bakes");
            let passes = if output == DemoOutput::Readback { 9 } else { 8 };
            assert_eq!(plain.passes.len(), passes);
            assert_eq!(plain.physical.len(), 6, "params, scene, a1, a2, b1, b2");
            assert!(aliased.physical.len() < plain.physical.len(), "{:?}", aliased.handoffs);
            assert_eq!(aliased.order().collect::<Vec<_>>(), plain.order().collect::<Vec<_>>());
        }
    }
}
