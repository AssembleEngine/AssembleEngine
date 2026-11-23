//! The bloom (M12 step 4): the glow target at a quarter of the size, the
//! brightest 2×2 mean per 4×4 block, then an 11-tap Gaussian along x and
//! along y (RA3-graphics' three passes). `post.tonemap` adds the result to
//! the scene before the tone curve, so the curve rolls it off with the rest.
//!
//! The glow holds only light sources (glowing decals, the water's glints,
//! emissive models and particles), not lit surfaces: a white sail or
//! sunlit sand does not bloom however bright it is. The water's soft sheen
//! is kept out separately (`shaders/water.frag`), and each input is
//! clamped at [`crate::bloom::BLOOM_CLAMP`] display-referred light before it
//! is blurred: `bloom.downsample` applies the eased EV's scale (auto
//! exposure, [`crate::exposure`]) before the clamp, so the clamp is
//! exposure-relative and stays 4.0 of display light whatever the adaptation
//! does.

use crate::bindless::Bindless;
use crate::device::GpuDevice;
use crate::executor::{ImageBinding, PassContext};
use crate::fullscreen::BindlessProgram;
use crate::pipeline::spirv;
use anyhow::{Context as _, Result};
use ash::vk;

/// The most exposed light a glow texel may give the bloom (as RA3's
/// `BLOOM_CLAMP`): a sun glint many times the display's white would
/// otherwise bloom into a large blob.
pub const BLOOM_CLAMP: f32 = 4.0;

/// The three bloom passes, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BloomPass {
    Downsample,
    BlurX,
    BlurY,
}

pub struct Bloom {
    passes: [BindlessProgram; 3],
}

impl Bloom {
    pub fn new(gpu: &GpuDevice, bindless: &Bindless, frames_in_flight: usize) -> Result<Self> {
        let format = vk::Format::from_raw(ae_render_abi::module::BLOOM_FORMAT as i32);
        let make = |label: &str, shader: &[u8], uniforms: u32, sampled: usize| {
            let _ = sampled;
            BindlessProgram::new(gpu, bindless, label, shader, uniforms, &[format], frames_in_flight)
                .with_context(|| format!("{label} program"))
        };
        Ok(Self {
            passes: [
                // The downsample also takes the 1×1 exposure image (the
                // display scale it applies before the clamp).
                make("bloom.downsample", spirv!("bloom_downsample.frag"), 16, 2)?,
                make("bloom.blur_x", spirv!("bloom_blur_x.frag"), 0, 1)?,
                make("bloom.blur_y", spirv!("bloom_blur_y.frag"), 0, 1)?,
            ],
        })
    }

    /// Frame slot `slot`'s uniforms (before it records): the EV100 the
    /// scene was stored at, and whether the graph declares the exposure
    /// image.
    pub fn write(&mut self, slot: usize, scene_ev100: f32, exposure: bool) {
        let words: [f32; 4] = [scene_ev100, if exposure { 1.0 } else { 0.0 }, 0.0, 0.0];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.passes[BloomPass::Downsample as usize].write_uniforms(slot, &bytes);
    }

    /// Record `pass`: `input` sampled in the fragment stage into `output`
    /// (in `COLOR_ATTACHMENT_OPTIMAL`), and the exposure image for the
    /// downsample when the graph has one (the input stands in otherwise,
    /// unread).
    pub fn record(
        &self,
        ctx: &PassContext<'_>,
        pass: BloomPass,
        input: ImageBinding,
        _output: ImageBinding,
        exposure: Option<ImageBinding>,
        sampler: u32,
    ) -> Result<()> {
        let index = |b: ImageBinding| b.sampled.unwrap_or(0);
        let images = match pass {
            BloomPass::Downsample => [index(input), index(exposure.unwrap_or(input))],
            BloomPass::BlurX | BloomPass::BlurY => [index(input), index(input)],
        };
        let count = if pass == BloomPass::Downsample { 2 } else { 1 };
        self.passes[pass as usize].record(ctx, &images[..count], &[sampler])
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        for program in &mut self.passes {
            program.destroy(gpu);
        }
    }
}
