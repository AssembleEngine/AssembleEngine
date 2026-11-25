//! Vulkan drawing for backend-independent `ae_text` batches.

pub use ae_text;

use crate::device::GpuDevice;
use crate::executor::PassContext;
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::pipeline::{self, Blend, DepthState, GraphicsDesc, spirv};
use crate::upload;
use ae_text::{Atlas, Quad};
use anyhow::{Result, ensure};
use ash::vk;
use glam::{Mat4, Vec3};
use std::{ops::Range, sync::Arc};

const _: () = assert!(size_of::<Quad>() == 48);

/// Immutable uploaded glyph atlas and quads. Create on the render thread
/// and reuse across frames. The renderer supplies depth and clipping state.
pub struct TextBatch {
    gpu: Arc<GpuDevice>,
    atlas: Option<OwnedImage>,
    quads: Option<OwnedBuffer>,
    count: u32,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl TextBatch {
    /// Upload a linear RGBA8 MSDF atlas and build a one-color pipeline with
    /// optional read-only depth. Uploads complete before this returns.
    pub fn new(gpu: Arc<GpuDevice>, atlas: &Atlas, color: vk::Format, depth: Option<DepthState>) -> Result<Self> {
        ensure!(
            !depth.is_some_and(|depth| depth.write),
            "text depth attachment must be read-only"
        );
        let count = u32::try_from(atlas.quads().len())?;
        let mut batch = Self {
            gpu,
            atlas: None,
            quads: None,
            count,
            sampler: vk::Sampler::null(),
            pool: vk::DescriptorPool::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            set: vk::DescriptorSet::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        };
        if count != 0 {
            batch.create(atlas, color, depth)?;
        }
        Ok(batch)
    }

    fn create(&mut self, atlas: &Atlas, color: vk::Format, depth: Option<DepthState>) -> Result<()> {
        let gpu = &self.gpu;
        let device = &gpu.device;
        let [width, height] = atlas.size();
        self.atlas = Some(upload::image(
            gpu,
            vk::Format::R8G8B8A8_UNORM,
            width,
            height,
            &[atlas.rgba()],
            "MSDF text atlas",
        )?);
        self.quads = Some(upload::buffer(
            gpu,
            pipeline::slice_bytes(atlas.quads()),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "text quads",
        )?);

        let vertex = vk::ShaderStageFlags::VERTEX;
        let fragment = vk::ShaderStageFlags::FRAGMENT;
        let storage = vk::DescriptorType::STORAGE_BUFFER;
        let image = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
        self.set_layout = pipeline::set_layout(device, &[(storage, vertex), (image, fragment)])?;
        self.layout = pipeline::pipeline_layout(device, &[self.set_layout], size_of::<Mat4>() as u32, vertex)?;
        self.pipeline = pipeline::graphics_pipeline(
            device,
            &GraphicsDesc {
                vertex: spirv!("text.vert"),
                fragment: spirv!("text_msdf.frag"),
                layout: self.layout,
                color_formats: &[color],
                depth,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                blend: Blend::Premultiplied,
                glow_blend: Blend::Off,
            },
        )?;
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: storage,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: image,
                descriptor_count: 1,
            },
        ];
        unsafe {
            self.sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.0),
                None,
            )?;
            self.pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )?;
            self.set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool)
                    .set_layouts(&[self.set_layout]),
            )?[0];
            let buffer = self.quads.as_ref().expect("uploaded above");
            let buffer = [vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer)
                .range(buffer.size)];
            let texture = self.atlas.as_ref().expect("uploaded above");
            let texture = [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(texture.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.set)
                        .dst_binding(0)
                        .descriptor_type(storage)
                        .buffer_info(&buffer),
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.set)
                        .dst_binding(1)
                        .descriptor_type(image)
                        .image_info(&texture),
                ],
                &[],
            );
        }
        Ok(())
    }

    /// Records within the matching graphics pass. The transform maps local
    /// text pixels to clip space and may include a perspective projection.
    pub fn draw(&self, ctx: &PassContext<'_>, quads: Range<u32>, local_to_clip: Mat4) -> Result<()> {
        ensure!(
            quads.start <= quads.end && quads.end <= self.count,
            "invalid text quad range"
        );
        if quads.is_empty() {
            return Ok(());
        }
        let point = vk::PipelineBindPoint::GRAPHICS;
        unsafe {
            ctx.device.cmd_bind_pipeline(ctx.cmd, point, self.pipeline);
            ctx.device
                .cmd_bind_descriptor_sets(ctx.cmd, point, self.layout, 0, &[self.set], &[]);
            ctx.device.cmd_push_constants(
                ctx.cmd,
                self.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                pipeline::push_bytes(&local_to_clip.to_cols_array()),
            );
            ctx.device.cmd_draw(ctx.cmd, 6, quads.end - quads.start, 0, quads.start);
        }
        Ok(())
    }
}

impl Drop for TextBatch {
    fn drop(&mut self) {
        if self.atlas.is_none() {
            return;
        }
        self.gpu.wait_idle();
        let device = &self.gpu.device;
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
        if let Some(mut image) = self.atlas.take() {
            image.destroy(&self.gpu);
        }
        if let Some(mut buffer) = self.quads.take() {
            buffer.destroy(&self.gpu);
        }
    }
}

/// Position local text pixels in a positive-height Vulkan viewport.
pub fn screen_to_clip(extent: vk::Extent2D, origin: [f32; 2]) -> Mat4 {
    let [x, y] = origin;
    let sx = 2.0 / extent.width as f32;
    let sy = 2.0 / extent.height as f32;
    Mat4::from_translation(Vec3::new(x * sx - 1.0, y * sy - 1.0, 0.0)) * Mat4::from_scale(Vec3::new(sx, sy, 1.0))
}
