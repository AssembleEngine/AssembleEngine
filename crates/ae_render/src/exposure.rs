//! Automatic exposure (eye adaptation, M12): meter the HDR scene's log
//! luminance into a histogram, derive the EV100 that maps the trimmed mean
//! to [`crate::lighting::KEY_LUMINANCE`], ease toward it and hand it to the
//! tonemap, with no CPU readback.
//!
//! The scene passes store the scene at the camera's reference EV
//! ([`crate::tonemap::ToneParams::ev100`]); `post.bloom`'s downsample and
//! `post.tonemap` apply the difference between the eased EV and that
//! reference, read from the 1×1 exposure image. The bloom's clamp is
//! therefore exposure-relative: it stays at 4.0 in display-referred terms
//! whatever the adaptation does.
//!
//! The graph's three library passes ([`crate::library`]), after every scene
//! writer and before the bloom and the tonemap:
//! * `exposure.clear` (transfer) zeroes the histogram for this frame;
//! * `exposure.meter` (compute) samples the scene at a quarter resolution
//!   (four taps per 4×4 block, averaged), each tap with the glow added at
//!   the bloom's strength (the light the bloom shows, so an explosion's
//!   glow counts), and adds a centre-weighted count
//!   ([`crate::lighting::METER_CENTER_WEIGHT`]) to the block's log2
//!   luminance bin, binned per workgroup in shared memory first. With the
//!   depth sampled too and the terrain's vision planes in use, it divides
//!   each tap by the fog of war's dimming at its world position, so an
//!   unexplored frame meters as if it were in sight and the fog keeps its
//!   look. It does nothing when auto-exposure is off;
//! * `exposure.adapt` (compute, one invocation) reads the histogram, takes
//!   the mean of the bins between the two percentiles that
//!   [`crate::lighting::METER_LOW_PERCENTILE`] and
//!   [`crate::lighting::METER_HIGH_PERCENTILE`] trim, computes the target
//!   EV100 ([`crate::lighting::target_ev100`]), eases the previous frame's
//!   EV toward it exponentially in EV space
//!   ([`crate::lighting::adapt_ev100`]) and writes the exposure image. A
//!   new EV takes effect in the same frame's bloom and tonemap: both come
//!   after it in the graph.
//!
//! The exposure image and the histogram are host-owned and shared by every
//! frame slot: the executor derives the cross-frame barriers from their
//! access states, so a slot's adapt pass never races another slot's
//! tonemap. A fresh image's contents are undefined; `exposure.adapt`
//! replaces anything outside 0..64 with the reference EV, so a capture's
//! first frame is stable.

use crate::bindless::{Bindless, Sampled2D};
use crate::device::GpuDevice;
use crate::executor::{BufferBinding, ImageBinding, PassContext};
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::pipeline::{self, spirv};
use ae_framegraph::{BufferDesc, ImageDesc};
use anyhow::{Context as _, Result};
use ash::vk;
use glam::Mat4;
use gpu_allocator::MemoryLocation;

// `shaders/exposure_meter.comp` sizes its shared histogram to this.
const _: () = assert!(ae_render_abi::module::EXPOSURE_BINS == 128);

/// The exposure image's unique identity in the executor's external state.
/// An app rebuild makes a new image under it; the render thread then has the
/// executor forget the external states (`Executor::forget_external_states`).
const IDENTITY_EXPOSURE: u64 = (0xAE7 << 40) | 12;
const IDENTITY_HISTOGRAM: u64 = (0xAE7 << 40) | 13;

/// What one frame's adaptation is given (the rest is constants).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureFrame {
    /// Meter and adapt to the scene; off follows [`Self::scene_ev100`].
    pub auto: bool,
    /// The EV100 the scene's draws expose with (the reference the tonemap's
    /// scale is relative to).
    pub scene_ev100: f32,
    /// Seconds since the previous adaptation.
    pub dt: f32,
    /// The camera's view-projection, to rebuild the metered texels' world
    /// positions from the depth.
    pub view_proj: Mat4,
    /// The terrain's vision layout ([`crate::terrain::renderer::TerrainRenderer::vision_uniform`])
    /// when the frame shades by the fog of war: the meter undoes its
    /// dimming.
    pub vision: Option<[f32; 4]>,
    /// The bloom's strength this frame (0 with the bloom off): the meter
    /// adds the glow at it, so the light the bloom shows counts.
    pub bloom: f32,
    /// The scene's working space: the stored light's luminance row
    /// ([`crate::colorspace::WorkingSpace::luminance_row`]).
    pub working: crate::colorspace::WorkingSpace,
}

/// The per-frame uniform block, std140 (160 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct Settings {
    /// x: scene EV100, y: dt, z: minimum EV100, w: maximum EV100.
    params: [f32; 4],
    /// x: 1 when auto, y: the key luminance, z/w: the rates toward a
    /// brighter and a darker scene (1/s).
    rates: [f32; 4],
    /// x: the low percentile, y: the high percentile, z: the bin count.
    trim: [f32; 4],
    /// The terrain's vision layout (cells wide, high, cell size, u32 words
    /// per row).
    vision: [f32; 4],
    /// x: 1 when the vision planes apply.
    flags: [f32; 4],
    /// x: the bloom's strength, y: the centre weight, z: its sigma (screen
    /// heights).
    meter: [f32; 4],
    /// The inverse view-projection (column-major).
    inv_view_proj: [f32; 16],
    /// The working space's luminance row (Rec. 709 or BT.2020).
    luma: [f32; 4],
}

/// The meter's push constant: whether the graph binds the depth (with the
/// vision planes) and the glow.
pub struct Exposure {
    image: OwnedImage,
    histogram: OwnedBuffer,
    ubos: Vec<OwnedBuffer>,
    /// The settings block (slotted), the histogram and the exposure image
    /// as a storage image (exposure_meter.comp, exposure_adapt.comp).
    table: crate::bindless::HandleTable,
    layout: vk::PipelineLayout,
    sampler: vk::Sampler,
    sampler_index: u32,
    sampled: Sampled2D,
    meter: vk::Pipeline,
    adapt: vk::Pipeline,
}

impl Exposure {
    pub fn new(gpu: &GpuDevice, bindless: &mut Bindless, frames_in_flight: usize) -> Result<Self> {
        let device = &gpu.device;
        let image_desc = ImageDesc::color_2d(
            vk::Format::from_raw(ae_render_abi::module::EXPOSURE_FORMAT as i32),
            1,
            1,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
        );
        let image = OwnedImage::new(gpu, image_desc, "exposure").context("exposure image")?;
        let histogram = OwnedBuffer::new(
            gpu,
            BufferDesc {
                size: ae_render_abi::module::EXPOSURE_HISTOGRAM_BYTES,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            },
            MemoryLocation::GpuOnly,
            "exposure histogram",
        )
        .context("exposure histogram")?;
        let mut ubos = Vec::with_capacity(frames_in_flight);
        for slot in 0..frames_in_flight {
            ubos.push(OwnedBuffer::new(
                gpu,
                BufferDesc {
                    size: size_of::<Settings>() as u64,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                },
                MemoryLocation::CpuToGpu,
                &format!("exposure settings {slot}"),
            )?);
        }
        let layout = pipeline::pipeline_layout(
            device,
            &bindless.layouts(2),
            size_of::<crate::bindless::Push>() as u32,
            vk::ShaderStageFlags::COMPUTE,
        )?;
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        let handles = [
            bindless.per_slot_buffer(&ubos)?.index,
            bindless.whole_buffer(&histogram)?.index,
            bindless.storage_image(image.view, vk::ImageLayout::GENERAL)?.index,
        ];
        let table = crate::bindless::HandleTable::new(gpu, bindless, &handles, "exposure handles")?;
        let meter = pipeline::compute_pipeline(device, spirv!("exposure_meter.comp"), layout)
            .context("exposure meter pipeline")?;
        let adapt = pipeline::compute_pipeline(device, spirv!("exposure_adapt.comp"), layout)
            .context("exposure adapt pipeline")?;
        let sampler_index = bindless.sampler(sampler)?.index;
        let sampled = bindless.sampled_2d(image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        Ok(Self {
            image,
            histogram,
            ubos,
            table,
            layout,
            sampler,
            sampler_index,
            sampled,
            meter,
            adapt,
        })
    }

    /// The exposure image as the graph binds it.
    pub fn image(&self) -> ImageBinding {
        ImageBinding {
            image: self.image.image,
            view: self.image.view,
            format: self.image.desc.format,
            extent: vk::Extent2D { width: 1, height: 1 },
            range: self.image.range(),
            identity: IDENTITY_EXPOSURE,
            sampled: Some(self.sampled.0.index),
            storage: None,
        }
    }

    /// The histogram as the graph binds it.
    pub fn histogram(&self) -> BufferBinding {
        BufferBinding {
            buffer: self.histogram.buffer,
            offset: 0,
            size: self.histogram.size,
            identity: IDENTITY_HISTOGRAM,
            storage: None,
        }
    }

    pub fn histogram_desc(&self) -> BufferDesc {
        BufferDesc {
            size: self.histogram.size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        }
    }

    /// Write `slot`'s settings (its previous submission has completed).
    pub fn write_frame(&mut self, slot: usize, frame: &ExposureFrame) {
        let settings = Settings {
            params: [
                frame.scene_ev100,
                frame.dt.clamp(0.0, 0.1),
                crate::lighting::MIN_EV100,
                crate::lighting::MAX_EV100,
            ],
            rates: [
                if frame.auto { 1.0 } else { 0.0 },
                crate::lighting::KEY_LUMINANCE,
                crate::lighting::TOWARD_BRIGHTER_PER_SECOND,
                crate::lighting::TOWARD_DARKER_PER_SECOND,
            ],
            trim: [
                crate::lighting::METER_LOW_PERCENTILE,
                crate::lighting::METER_HIGH_PERCENTILE,
                ae_render_abi::module::EXPOSURE_BINS as f32,
                0.0,
            ],
            vision: frame.vision.unwrap_or([1.0, 1.0, 1.0, 1.0]),
            flags: [if frame.vision.is_some() { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            meter: [
                frame.bloom.max(0.0),
                crate::lighting::METER_CENTER_WEIGHT,
                crate::lighting::METER_CENTER_SIGMA,
                0.0,
            ],
            inv_view_proj: frame.view_proj.inverse().to_cols_array(),
            luma: {
                let l = frame.working.luminance_row();
                [l[0], l[1], l[2], 0.0]
            },
        };
        if let Some(bytes) = self.ubos.get_mut(slot).and_then(|u| u.mapped()) {
            bytes[..size_of::<Settings>()].copy_from_slice(pipeline::push_bytes(&settings));
        }
    }

    /// `exposure.clear`: zero the histogram (`cmd_fill_buffer`).
    pub fn record_clear(&self, ctx: &PassContext<'_>) {
        unsafe {
            ctx.device
                .cmd_fill_buffer(ctx.cmd, self.histogram.buffer, 0, vk::WHOLE_SIZE, 0)
        };
    }

    /// `exposure.meter`: the scene into the histogram at a quarter of the
    /// size, one centre-weighted count per 4×4 block. `depth` (when the
    /// graph has the meter sample it) and `vision` (the terrain's planes for
    /// this slot) let it undo the fog of war's dimming; `glow` adds the
    /// light the bloom shows. Without them it meters the scene as stored.
    pub fn record_meter(
        &self,
        ctx: &PassContext<'_>,
        scene: ImageBinding,
        depth: Option<ImageBinding>,
        glow: Option<ImageBinding>,
        vision: Option<u32>,
    ) -> Result<()> {
        // The vision planes' global index is the frame's (word 2); word 0
        // says they apply.
        let mut push = crate::bindless::Push::default();
        push.table(self.table.index);
        push.words[0] = u32::from(depth.is_some() && vision.is_some());
        push.words[1] = u32::from(glow.is_some());
        push.words[2] = vision.unwrap_or(0);
        push.image(0, scene.sampled.unwrap_or(0));
        push.image(1, depth.unwrap_or(scene).sampled.unwrap_or(0));
        push.image(2, glow.unwrap_or(scene).sampled.unwrap_or(0));
        push.sampler(0, self.sampler_index);
        unsafe {
            ctx.device.cmd_push_constants(
                ctx.cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pipeline::push_bytes(&push),
            );
            ctx.device
                .cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.meter);
            ctx.device.cmd_bind_descriptor_sets(
                ctx.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                2,
                &[ctx.global_set],
                &[],
            );
            let groups = (
                scene.extent.width.div_ceil(4).div_ceil(8),
                scene.extent.height.div_ceil(4).div_ceil(8),
            );
            ctx.device.cmd_dispatch(ctx.cmd, groups.0.max(1), groups.1.max(1), 1);
        }
        Ok(())
    }

    /// `exposure.adapt`: the histogram to the eased exposure, one thread.
    pub fn record_adapt(&self, ctx: &PassContext<'_>) {
        let mut push = crate::bindless::Push::default();
        push.table(self.table.index);
        unsafe {
            ctx.device
                .cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.adapt);
            ctx.device.cmd_bind_descriptor_sets(
                ctx.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                2,
                &[ctx.global_set],
                &[],
            );
            ctx.device.cmd_push_constants(
                ctx.cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pipeline::push_bytes(&push),
            );
            ctx.device.cmd_dispatch(ctx.cmd, 1, 1, 1);
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        let device = &gpu.device;
        unsafe {
            for pipeline in [self.meter, self.adapt] {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_sampler(self.sampler, None);
        }
        self.image.destroy(gpu);
        self.histogram.destroy(gpu);
        self.table.destroy(gpu);
        for ubo in &mut self.ubos {
            ubo.destroy(gpu);
        }
    }
}
