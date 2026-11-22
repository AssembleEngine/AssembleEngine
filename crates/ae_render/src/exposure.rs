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
#[repr(C)]
#[derive(Clone, Copy)]
struct MeterPush {
    depth: u32,
    glow: u32,
}

pub struct Exposure {
    image: OwnedImage,
    histogram: OwnedBuffer,
    ubos: Vec<OwnedBuffer>,
    /// Per slot: the meter's and the adapt's descriptor sets (one each, so
    /// two passes never rewrite one set in a frame).
    meter_sets: Vec<vk::DescriptorSet>,
    adapt_sets: Vec<vk::DescriptorSet>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    meter: vk::Pipeline,
    adapt: vk::Pipeline,
}

impl Exposure {
    pub fn new(gpu: &GpuDevice, frames_in_flight: usize) -> Result<Self> {
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
                    usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                },
                MemoryLocation::CpuToGpu,
                &format!("exposure settings {slot}"),
            )?);
        }
        let all = vk::ShaderStageFlags::COMPUTE;
        let (set_layout, layout, pool, meter_sets, adapt_sets, sampler) = unsafe {
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(all),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(all),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(all),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(all),
                // The meter's depth and the terrain's vision planes (bound
                // per frame; the adapt pass leaves them unwritten).
                vk::DescriptorSetLayoutBinding::default()
                    .binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(all),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(5)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(all),
                // The meter's glow (the light the bloom shows).
                vk::DescriptorSetLayoutBinding::default()
                    .binding(6)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(all),
            ];
            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let layout = pipeline::pipeline_layout(
                device,
                &[set_layout],
                size_of::<MeterPush>() as u32,
                vk::ShaderStageFlags::COMPUTE,
            )?;
            let n = frames_in_flight as u32;
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 2 * n,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 4 * n,
                },
                // Every set has the layout's three samplers (scene, depth,
                // glow), whether its pass reads them or not.
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 6 * n,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 2 * n,
                },
            ];
            let pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(2 * n)
                    .pool_sizes(&sizes),
                None,
            )?;
            let layouts = vec![set_layout; 2 * frames_in_flight];
            let sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )?;
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            let histogram_info = [vk::DescriptorBufferInfo {
                buffer: histogram.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            }];
            let image_info = [vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: image.view,
                image_layout: vk::ImageLayout::GENERAL,
            }];
            for slot in 0..frames_in_flight {
                let (meter, adapt) = (sets[2 * slot], sets[2 * slot + 1]);
                let ubo = [vk::DescriptorBufferInfo {
                    buffer: ubos[slot].buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                }];
                for (set, with_image) in [(meter, false), (adapt, true)] {
                    let mut writes = vec![
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&ubo),
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&histogram_info),
                    ];
                    if with_image {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(3)
                                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                .image_info(&image_info),
                        );
                    }
                    device.update_descriptor_sets(&writes, &[]);
                }
            }
            let meter_sets = (0..frames_in_flight).map(|slot| sets[2 * slot]).collect();
            let adapt_sets = (0..frames_in_flight)
                .map(|slot| sets[2 * slot + 1])
                .collect();
            (set_layout, layout, pool, meter_sets, adapt_sets, sampler)
        };
        let meter = pipeline::compute_pipeline(device, spirv!("exposure_meter.comp"), layout)
            .context("exposure meter pipeline")?;
        let adapt = pipeline::compute_pipeline(device, spirv!("exposure_adapt.comp"), layout)
            .context("exposure adapt pipeline")?;
        Ok(Self {
            image,
            histogram,
            ubos,
            meter_sets,
            adapt_sets,
            set_layout,
            layout,
            pool,
            sampler,
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
            extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            range: self.image.range(),
            identity: IDENTITY_EXPOSURE,
        }
    }

    /// The histogram as the graph binds it.
    pub fn histogram(&self) -> BufferBinding {
        BufferBinding {
            buffer: self.histogram.buffer,
            offset: 0,
            size: self.histogram.size,
            identity: IDENTITY_HISTOGRAM,
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
        vision: Option<vk::Buffer>,
    ) -> Result<()> {
        let set = self.meter_sets[ctx.slot];
        let image = |view| {
            [vk::DescriptorImageInfo {
                sampler: self.sampler,
                image_view: view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }]
        };
        // Stand-ins, unread: the scene for a missing depth, the histogram
        // for missing vision planes.
        let (scene_info, depth_info, glow_info) = (
            image(scene.view),
            image(depth.unwrap_or(scene).view),
            image(glow.unwrap_or(scene).view),
        );
        let vision_info = [vk::DescriptorBufferInfo {
            buffer: vision.unwrap_or(self.histogram.buffer),
            offset: 0,
            range: vk::WHOLE_SIZE,
        }];
        let push = MeterPush {
            depth: u32::from(depth.is_some() && vision.is_some()),
            glow: u32::from(glow.is_some()),
        };
        unsafe {
            ctx.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&scene_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(4)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&depth_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(5)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&vision_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(6)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&glow_info),
                ],
                &[],
            );
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
                0,
                &[set],
                &[],
            );
            let groups = (
                scene.extent.width.div_ceil(4).div_ceil(8),
                scene.extent.height.div_ceil(4).div_ceil(8),
            );
            ctx.device
                .cmd_dispatch(ctx.cmd, groups.0.max(1), groups.1.max(1), 1);
        }
        Ok(())
    }

    /// `exposure.adapt`: the histogram to the eased exposure, one thread.
    pub fn record_adapt(&self, ctx: &PassContext<'_>) {
        let set = self.adapt_sets[ctx.slot];
        unsafe {
            ctx.device
                .cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.adapt);
            ctx.device.cmd_bind_descriptor_sets(
                ctx.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[set],
                &[],
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
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
        self.image.destroy(gpu);
        self.histogram.destroy(gpu);
        for ubo in &mut self.ubos {
            ubo.destroy(gpu);
        }
    }
}
