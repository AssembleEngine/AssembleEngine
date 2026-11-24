//! The shore travel-time field (M16.5): for every water texel, the seconds a
//! swell crest takes to reach the waterline, zero *at* the waterline and
//! rising seaward. The rich tiers use it to time their shore foam and to make
//! breakers turn toward the beach; the M7 surface ignores it.
//!
//! The field is baked on the GPU from the ground height texture with a jump
//! flood (a nearest-waterline-seed pass, log2 steps of halved jumps, then a
//! resolve). It is made when the water renderer is built and rebuilt after a
//! terrain edit: [`ShoreField::mark_dirty`] is set by the app's edit hook, the
//! pass is skipped while the ground is unchanged, and no offline bake exists.
//!
//! The travel time is the distance to the waterline over the shallow-water
//! phase speed at the 1 m break depth (a constant): that rises monotonically
//! along a ray seaward. The prototype integrates the local speed (a Dijkstra
//! bake); a fast-marching pass would replace the resolve if the foam needs
//! depth-dependent timing later.

use crate::bindless::{Bindless, HandleTable, Push};
use crate::executor::PassContext;
use crate::memory::{OwnedBuffer, OwnedImage};
use crate::pipeline;
use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The field's format and what a graph declares for it: R16F, one texel per
/// ground height sample.
pub const FIELD_FORMAT: vk::Format = vk::Format::R16_SFLOAT;
/// The jump flood's nearest-seed scratch: integer texel coordinates (exact
/// at every map width), all-ones where there is no seed.
const SEED_FORMAT: vk::Format = vk::Format::R16G16_UINT;
const WORKGROUP: u32 = 8;

const MODE_SEED: u32 = 0;
const MODE_FLOOD: u32 = 1;
const MODE_FINAL: u32 = 2;

/// The GPU-side shore field, its seeds and its pipeline.
pub struct ShoreField {
    field: OwnedImage,
    /// `seeds[0]` and `seeds[1]`: the jump flood's ping-pong.
    seeds: [OwnedImage; 2],
    /// `(normalised water level, cell size (m), break depth (m), speed
    /// factor)`: the resolve's parameters.
    params: OwnedBuffer,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    table: HandleTable,
    width: u32,
    height: u32,
    /// The ground changed since the last bake (or the field was never made).
    dirty: AtomicBool,
    /// The seeds are in GENERAL (their first barrier has run).
    ready: AtomicBool,
    /// Bakes recorded (tests and reports).
    bakes: AtomicU64,
    /// The last edit's rectangle (cells), for reports; the bake is full.
    rect: std::sync::Mutex<Option<(u32, u32, u32, u32)>>,
    /// The library binding's identity (unique per allocation, so a rebuilt
    /// field never inherits the old image's tracked layout).
    identity: u64,
    /// The bindless handles, given back by [`Self::destroy`].
    group: Option<crate::bindless::GroupId>,
}

impl ShoreField {
    /// Make the field for a ground height texture of `width` x `height`.
    /// `norm_level` is the water level in the height texture's normalised
    /// units; `cell_size` is metres per texel.
    pub fn new(
        gpu: &crate::device::GpuDevice,
        bindless: &mut Bindless,
        width: u32,
        height: u32,
        norm_level: f32,
        cell_size: f32,
        identity: u64,
    ) -> Result<Self> {
        anyhow::ensure!(width > 0 && height > 0, "a shore field needs a size");
        let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC;
        let field = OwnedImage::new(
            gpu,
            ae_framegraph::ImageDesc::color_2d(FIELD_FORMAT, width, height, usage),
            "water shore",
        )
        .context("water shore field")?;
        let seed_a = match OwnedImage::new(
            gpu,
            ae_framegraph::ImageDesc::color_2d(SEED_FORMAT, width, height, usage),
            "water shore seeds a",
        ) {
            Ok(seed) => seed,
            Err(error) => {
                let mut field = field;
                field.destroy(gpu);
                return Err(error);
            }
        };
        let seed_b = match OwnedImage::new(
            gpu,
            ae_framegraph::ImageDesc::color_2d(SEED_FORMAT, width, height, usage),
            "water shore seeds b",
        ) {
            Ok(seed) => seed,
            Err(error) => {
                let (mut field, mut seed_a) = (field, seed_a);
                field.destroy(gpu);
                seed_a.destroy(gpu);
                return Err(error);
            }
        };
        let mut seeds = [seed_a, seed_b];
        let mut params = match OwnedBuffer::new(
            gpu,
            ae_framegraph::BufferDesc {
                size: 16,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            },
            MemoryLocation::CpuToGpu,
            "water shore params",
        ) {
            Ok(params) => params,
            Err(error) => {
                let (mut field, mut seeds) = (field, seeds);
                field.destroy(gpu);
                for seed in &mut seeds {
                    seed.destroy(gpu);
                }
                return Err(error);
            }
        };
        let values: Vec<u8> = [norm_level, cell_size, 1.0, 0.6]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let Some(mapped) = params.mapped() else {
            let (mut field, mut params) = (field, params);
            field.destroy(gpu);
            for seed in &mut seeds {
                seed.destroy(gpu);
            }
            params.destroy(gpu);
            return Err(anyhow::anyhow!("water shore params are not host-visible"));
        };
        mapped.copy_from_slice(&values);
        // The handles and the pipeline: a failure destroys what was made
        // here (the group rollback releases handles only) and the caller
        // destroys the images and the params.
        let built = bindless.group(|bindless| -> Result<(HandleTable, vk::Pipeline, vk::PipelineLayout)> {
            let handles = [
                bindless.storage_image(field.view, vk::ImageLayout::GENERAL)?.index,
                bindless.storage_image(seeds[0].view, vk::ImageLayout::GENERAL)?.index,
                bindless.storage_image(seeds[1].view, vk::ImageLayout::GENERAL)?.index,
                bindless.whole_buffer(&params)?.index,
            ];
            let table = HandleTable::new(gpu, bindless, &handles, "water shore handles")?;
            let layout = bindless.pipeline_layout();
            let pipeline = match pipeline::compute_pipeline(&gpu.device, pipeline::spirv!("water_shore.comp"), layout) {
                Ok(pipeline) => pipeline,
                Err(error) => {
                    let mut table = table;
                    table.destroy(gpu);
                    return Err(error.context("water shore pipeline"));
                }
            };
            Ok((table, pipeline, layout))
        });
        match built {
            Ok(((table, pipeline, layout), group)) => Ok(Self {
                field,
                seeds,
                params,
                pipeline,
                layout,
                table,
                width,
                height,
                dirty: AtomicBool::new(true),
                ready: AtomicBool::new(false),
                bakes: AtomicU64::new(0),
                rect: std::sync::Mutex::new(None),
                identity,
                group: Some(group),
            }),
            Err(error) => {
                let (mut field, mut params) = (field, params);
                field.destroy(gpu);
                for seed in &mut seeds {
                    seed.destroy(gpu);
                }
                params.destroy(gpu);
                Err(error)
            }
        }
    }

    /// The field's descriptor (what a graph declares: External, this size and
    /// format).
    pub fn desc(&self) -> ae_framegraph::ImageDesc {
        self.field.desc
    }

    pub fn view(&self) -> vk::ImageView {
        self.field.view
    }

    /// The field as the graph binds it.
    pub fn binding(&self) -> crate::executor::ImageBinding {
        crate::executor::ImageBinding {
            initial: vk::ImageLayout::UNDEFINED,
            image: self.field.image,
            view: self.field.view,
            format: self.field.desc.format,
            extent: vk::Extent2D {
                width: self.width,
                height: self.height,
            },
            range: self.field.range(),
            identity: self.identity,
            sampled: None,
            storage: None,
        }
    }

    /// Whether the ground changed since the last bake.
    pub fn dirty(&self) -> bool {
        self.dirty.load(Ordering::Relaxed)
    }

    /// The ground changed (a terrain edit landed); `rect` is the feed's
    /// changed cells, kept for reports.
    pub fn mark_dirty(&self, rect: Option<(u32, u32, u32, u32)>) {
        if let Some(r) = rect {
            *self.rect.lock().unwrap_or_else(|p| p.into_inner()) = Some(r);
        }
        self.dirty.store(true, Ordering::Relaxed);
    }

    /// The last edit's rectangle, if any.
    pub fn last_rect(&self) -> Option<(u32, u32, u32, u32)> {
        *self.rect.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Bakes recorded: one per frame the pass ran (the first, and each after
    /// a ground change).
    pub fn bakes(&self) -> u64 {
        self.bakes.load(Ordering::Relaxed)
    }

    /// `water.shore`: seed, jump flood, resolve. `ground` is the global
    /// sampled index of the ground height texture and `sampler` its sampler
    /// (the terrain renderer's).
    pub fn record(&self, ctx: &PassContext<'_>, ground: u32, sampler: u32) -> Result<()> {
        let device = ctx.device;
        if !self.ready.load(Ordering::Relaxed) {
            // The seeds start undefined; the field is the graph's External
            // image, which the executor transitions before the pass.
            let to_general = self.seeds.iter().map(|seed| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(seed.image)
                    .subresource_range(seed.range())
            });
            let barriers: Vec<vk::ImageMemoryBarrier2> = to_general.collect();
            unsafe {
                device.cmd_pipeline_barrier2(ctx.cmd, &vk::DependencyInfo::default().image_memory_barriers(&barriers));
            }
        }
        // Seed the waterline texels.
        self.dispatch(ctx, MODE_SEED, 1, 1, ground, sampler);
        self.after_write(ctx, &self.seeds[0]);
        // Jump flood: the largest halving jump down to one texel.
        let mut step = (self.width.max(self.height).next_power_of_two() / 2).max(1);
        let mut source = 0usize;
        while step >= 1 {
            let target = 1 - source;
            // The source is slot `source + 1` in the handle table; the
            // destination the other.
            self.dispatch(ctx, MODE_FLOOD, step, source as u32 + 1, ground, sampler);
            self.after_write(ctx, &self.seeds[target]);
            source = target;
            if step == 1 {
                break;
            }
            step /= 2;
        }
        // JFA+1: two more one-texel passes settle the seeds a plain halving
        // sequence can miss (Kang et al.'s remedy for the approximate flood).
        for _ in 0..2 {
            let target = 1 - source;
            self.dispatch(ctx, MODE_FLOOD, 1, source as u32 + 1, ground, sampler);
            self.after_write(ctx, &self.seeds[target]);
            source = target;
        }
        // Resolve the distance into a travel time.
        self.dispatch(ctx, MODE_FINAL, 1, source as u32 + 1, ground, sampler);
        self.dirty.store(false, Ordering::Relaxed);
        self.ready.store(true, Ordering::Relaxed);
        self.bakes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn dispatch(&self, ctx: &PassContext<'_>, mode: u32, step: u32, source: u32, ground: u32, sampler: u32) {
        let mut push = Push {
            words: [mode, step, self.width, self.height],
            ..Push::default()
        };
        push.words2[2] = source;
        push.table(self.table.index);
        push.image(0, ground);
        push.sampler(0, sampler);
        unsafe {
            ctx.device
                .cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            ctx.device.cmd_bind_descriptor_sets(
                ctx.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[ctx.global_set],
                &[],
            );
            ctx.device.cmd_push_constants(
                ctx.cmd,
                self.layout,
                crate::bindless::PUSH_STAGES,
                0,
                pipeline::push_bytes(&push),
            );
            ctx.device.cmd_dispatch(
                ctx.cmd,
                self.width.div_ceil(WORKGROUP),
                self.height.div_ceil(WORKGROUP),
                1,
            );
        }
    }

    /// The written image is visible to the next dispatch's reads.
    fn after_write(&self, ctx: &PassContext<'_>, image: &OwnedImage) {
        let barrier = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.image)
            .subresource_range(image.range())];
        unsafe {
            ctx.device
                .cmd_pipeline_barrier2(ctx.cmd, &vk::DependencyInfo::default().image_memory_barriers(&barrier));
        }
    }

    /// The field's first mip as f32 seconds (tests and tools), block by
    /// block; the image is left in GENERAL, where the pass keeps it.
    pub fn read(&self, gpu: &crate::device::GpuDevice) -> Result<Vec<f32>> {
        let bytes = crate::upload::read_image_from(gpu, &self.field, 2, vk::ImageLayout::GENERAL)?;
        Ok(bytes[0]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[lo, hi]| f16_to_f32(u16::from_le_bytes([lo, hi])))
            .collect())
    }

    /// Give the global-set handles back (before [`Self::destroy`]).
    pub fn retire(&mut self, bindless: &mut Bindless) {
        if let Some(group) = self.group.take() {
            bindless.retire_group(group);
        }
    }

    pub fn destroy(&mut self, gpu: &crate::device::GpuDevice) {
        self.field.destroy(gpu);
        for seed in &mut self.seeds {
            seed.destroy(gpu);
        }
        self.params.destroy(gpu);
        self.table.destroy(gpu);
        unsafe { gpu.device.destroy_pipeline(self.pipeline, None) };
    }
}

/// IEEE binary16 to f32 (the field's format). Values are finite and normal
/// in use; subnormals and infinities are still decoded.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = (h >> 10) & 0x1f;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: shift the mantissa up to a normal exponent.
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        _ => sign | (((exp as u32) + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats_decode() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert!((f16_to_f32(0x5640) - 100.0).abs() < 1e-3);
        // The largest finite half, and a subnormal.
        assert!((f16_to_f32(0x7bff) - 65504.0).abs() < 1.0);
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
    }
}
