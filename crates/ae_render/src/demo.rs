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
    Upload { src: ResourceId, dst: ResourceId },
    Triangle,
    /// Filter `mode` from `src` to `dst` (see `demo_filter.comp`).
    Filter { mode: i32, src: ResourceId, dst: ResourceId, params: ResourceId },
    Composite { left: ResourceId, right: ResourceId },
    Readback { src: ResourceId, dst: ResourceId },
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
        ImageDesc::color_2d(SCENE_FORMAT, width, height, vk::ImageUsageFlags::COLOR_ATTACHMENT | work_usage),
        ResourceClass::Transient,
    );
    let work = |graph: &mut FrameGraph<DemoPass>, name: &str| {
        graph.image(name, ImageDesc::color_2d(SCENE_FORMAT, width, height, work_usage), ResourceClass::Transient)
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
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
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
        .pass("upload params", PassKind::Transfer, DemoPass::Upload { src: params_staging, dst: params })
        .access(params_staging, Role::TransferSrc)
        .discard(params, Role::TransferDst)
        .id();
    graph.pass("triangle", PassKind::Graphics, DemoPass::Triangle).color(scene, LoadOp::Clear).id();
    let filter = |graph: &mut FrameGraph<DemoPass>, name: &str, mode, src, dst| {
        graph
            .pass(name, PassKind::Compute, DemoPass::Filter { mode, src, dst, params })
            .access(src, Role::StorageImageRead(ShaderStage::Compute))
            .discard(dst, Role::StorageImageWrite(ShaderStage::Compute))
            .access(params, Role::UniformBuffer(ShaderStage::Compute))
            .id();
    };
    filter(&mut graph, "blur_h", 0, scene, a1);
    filter(&mut graph, "blur_v", 1, a1, a2);
    filter(&mut graph, "invert", 2, scene, b1);
    filter(&mut graph, "tint", 3, b1, b2);
    graph
        .pass("composite", PassKind::Graphics, DemoPass::Composite { left: a2, right: b2 })
        .sampled(a2, ShaderStage::Fragment)
        .sampled(b2, ShaderStage::Fragment)
        .color(target, LoadOp::DontCare)
        .id();
    match readback {
        Some(buffer) => {
            graph
                .pass("readback", PassKind::Transfer, DemoPass::Readback { src: target, dst: buffer })
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
            graph.pass("present", PassKind::Present, DemoPass::Present).present(target).root().id();
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

pub use gpu::{DemoApp, DemoRenderer, record_readback};

mod gpu {
    use super::*;
    use crate::device::GpuDevice;
    use crate::executor::{PassContext, Recorder};
    use ae_framegraph::PassId;
    use crate::pipeline::{self, Blend, GraphicsDesc, spirv};
    use anyhow::Result;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Pipelines and descriptor sets for the demo passes. Descriptor sets are
    /// per (slot, pass) and rewritten only when the bound views change,
    /// which happens after a rebake (device idle) or for a slot whose fence
    /// was just waited, so no set is updated while in flight.
    pub struct DemoRenderer {
        gpu: Arc<GpuDevice>,
        sampler: vk::Sampler,
        compute_set_layout: vk::DescriptorSetLayout,
        composite_set_layout: vk::DescriptorSetLayout,
        compute_layout: vk::PipelineLayout,
        composite_layout: vk::PipelineLayout,
        triangle_layout: vk::PipelineLayout,
        triangle: vk::Pipeline,
        filter: vk::Pipeline,
        composite: vk::Pipeline,
        composite_format: vk::Format,
        pool: vk::DescriptorPool,
        sets: BTreeMap<(usize, PassId), (vk::DescriptorSet, [vk::ImageView; 2], vk::Buffer)>,
    }

    impl DemoRenderer {
        pub fn new(gpu: Arc<GpuDevice>, target_format: vk::Format, frames_in_flight: usize) -> Result<Self> {
            let device = &gpu.device;
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            let sampler = unsafe { device.create_sampler(&sampler_info, None) }?;
            let compute_set_layout = pipeline::set_layout(
                device,
                &[
                    (vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE),
                    (vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE),
                    (vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::COMPUTE),
                ],
            )?;
            let composite_set_layout = pipeline::set_layout(
                device,
                &[
                    (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT),
                    (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT),
                ],
            )?;
            let compute_layout =
                pipeline::pipeline_layout(device, &[compute_set_layout], 4, vk::ShaderStageFlags::COMPUTE)?;
            let composite_layout =
                pipeline::pipeline_layout(device, &[composite_set_layout], 0, vk::ShaderStageFlags::empty())?;
            let triangle_layout = pipeline::pipeline_layout(device, &[], 0, vk::ShaderStageFlags::empty())?;
            let triangle = pipeline::graphics_pipeline(
                device,
                &GraphicsDesc {
                    vertex: spirv!("triangle.vert"),
                    fragment: spirv!("triangle.frag"),
                    layout: triangle_layout,
                    color_formats: &[SCENE_FORMAT],
                    depth: None,
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    blend: Blend::Off,
                    glow_blend: Blend::Off,
                },
            )?;
            let filter = pipeline::compute_pipeline(device, spirv!("demo_filter.comp"), compute_layout)?;
            let composite = pipeline::graphics_pipeline(
                device,
                &GraphicsDesc {
                    vertex: spirv!("fullscreen.vert"),
                    fragment: spirv!("composite.frag"),
                    layout: composite_layout,
                    color_formats: &[target_format],
                    depth: None,
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    blend: Blend::Off,
                    glow_blend: Blend::Off,
                },
            )?;
            let sets_needed = (frames_in_flight * 5) as u32;
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: sets_needed * 2,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: sets_needed * 2,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: sets_needed,
                },
            ];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(sets_needed)
                .pool_sizes(&sizes);
            let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;
            Ok(Self {
                gpu,
                sampler,
                compute_set_layout,
                composite_set_layout,
                compute_layout,
                composite_layout,
                triangle_layout,
                triangle,
                filter,
                composite,
                composite_format: target_format,
                pool,
                sets: BTreeMap::new(),
            })
        }

        pub fn target_format(&self) -> vk::Format {
            self.composite_format
        }

        /// Forget cached descriptor sets (after a rebake the views change;
        /// the device is idle then).
        pub fn reset_descriptors(&mut self) -> Result<()> {
            unsafe {
                self.gpu
                    .device
                    .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
            }?;
            self.sets.clear();
            Ok(())
        }

        fn set_for(
            &mut self,
            ctx: &PassContext<'_>,
            layout: vk::DescriptorSetLayout,
            views: [vk::ImageView; 2],
            kind: vk::DescriptorType,
            uniform: Option<crate::executor::BufferBinding>,
        ) -> Result<vk::DescriptorSet> {
            let device = &self.gpu.device;
            let key = (ctx.slot, ctx.pass);
            let uniform_handle = uniform.map(|binding| binding.buffer).unwrap_or_default();
            if let Some((set, cached, buffer)) = self.sets.get(&key)
                && *cached == views
                && *buffer == uniform_handle
            {
                return Ok(*set);
            }
            let set = match self.sets.get(&key) {
                Some((set, _, _)) => *set,
                None => {
                    let layouts = [layout];
                    let info = vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.pool)
                        .set_layouts(&layouts);
                    unsafe { device.allocate_descriptor_sets(&info) }?[0]
                }
            };
            let layout_for = if kind == vk::DescriptorType::STORAGE_IMAGE {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            };
            let infos: Vec<[vk::DescriptorImageInfo; 1]> = views
                .iter()
                .map(|view| {
                    [vk::DescriptorImageInfo::default()
                        .image_view(*view)
                        .image_layout(layout_for)
                        .sampler(self.sampler)]
                })
                .collect();
            let mut writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(binding, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding as u32)
                        .descriptor_type(kind)
                        .image_info(info)
                })
                .collect();
            let buffer_info = uniform.map(|binding| {
                [vk::DescriptorBufferInfo {
                    buffer: binding.buffer,
                    offset: binding.offset,
                    range: binding.size,
                }]
            });
            if let Some(info) = &buffer_info {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(info),
                );
            }
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            self.sets.insert(key, (set, views, uniform_handle));
            Ok(set)
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
                    let set = self.set_for(
                        ctx,
                        self.compute_set_layout,
                        [src.view, dst.view],
                        vk::DescriptorType::STORAGE_IMAGE,
                        Some(params),
                    )?;
                    unsafe {
                        device.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.filter);
                        device.cmd_bind_descriptor_sets(
                            ctx.cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.compute_layout,
                            0,
                            &[set],
                            &[],
                        );
                        device.cmd_push_constants(
                            ctx.cmd,
                            self.compute_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            &mode.to_le_bytes(),
                        );
                        device.cmd_dispatch(ctx.cmd, dst.extent.width.div_ceil(8), dst.extent.height.div_ceil(8), 1);
                    }
                }
                DemoPass::Composite { left, right } => {
                    let views = [ctx.image(left)?.view, ctx.image(right)?.view];
                    let set = self.set_for(
                        ctx,
                        self.composite_set_layout,
                        views,
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        None,
                    )?;
                    unsafe {
                        device.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::GRAPHICS, self.composite);
                        device.cmd_bind_descriptor_sets(
                            ctx.cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.composite_layout,
                            0,
                            &[set],
                            &[],
                        );
                        device.cmd_draw(ctx.cmd, 3, 1, 0, 0);
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
    }

    impl crate::thread::RenderApp for DemoApp {
        type Pass = DemoPass;

        fn create(&mut self, gpu: &Arc<GpuDevice>, target: crate::thread::TargetInfo, frames_in_flight: usize) -> Result<()> {
            let format = target.format;
            self.renderer = Some(DemoRenderer::new(gpu.clone(), format, frames_in_flight)?);
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
        ) -> Result<(FrameGraph<DemoPass>, Option<ResourceId>)> {
            let (graph, resources) = build_demo_graph(target.extent, DemoOutput::Present { format: target.format });
            self.params_staging = Some(resources.params_staging);
            Ok((graph, Some(resources.target)))
        }

        fn on_rebake(&mut self) -> Result<()> {
            match &mut self.renderer {
                Some(renderer) => renderer.reset_descriptors(),
                None => Ok(()),
            }
        }

        fn prepare(&mut self, slot: usize) -> Result<()> {
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
                })
            })
        }

        fn recorder(&mut self) -> &mut dyn Recorder<DemoPass> {
            self.renderer.as_mut().expect("created before recording")
        }

        fn status(&self) -> String {
            "demo graph".into()
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
                device.destroy_descriptor_pool(self.pool, None);
                device.destroy_pipeline(self.triangle, None);
                device.destroy_pipeline(self.filter, None);
                device.destroy_pipeline(self.composite, None);
                device.destroy_pipeline_layout(self.triangle_layout, None);
                device.destroy_pipeline_layout(self.compute_layout, None);
                device.destroy_pipeline_layout(self.composite_layout, None);
                device.destroy_descriptor_set_layout(self.compute_set_layout, None);
                device.destroy_descriptor_set_layout(self.composite_set_layout, None);
                device.destroy_sampler(self.sampler, None);
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
        for output in [DemoOutput::Readback, DemoOutput::Present { format: vk::Format::B8G8R8A8_SRGB }] {
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
