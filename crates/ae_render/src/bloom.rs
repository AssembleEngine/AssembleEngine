//! The bloom (M12 step 4): the glow target at a quarter of the size, the
//! brightest 2×2 mean per 4×4 block, then an 11-tap Gaussian along x and
//! along y (RA3-graphics' three passes). `post.tonemap` adds the result to
//! the scene before the tone curve, so ACES rolls it off with the rest.
//!
//! The glow holds only light sources (glowing decals, the water's glints,
//! emissive models and particles), not lit surfaces: a white sail or
//! sunlit sand does not bloom however bright it is. The water's soft sheen
//! is kept out separately (`shaders/water.frag`), and every input is
//! clamped at [`crate::bloom::BLOOM_CLAMP`] exposed light before it is
//! blurred.

use crate::device::GpuDevice;
use crate::executor::{ImageBinding, PassContext};
use crate::fullscreen::FullscreenProgram;
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
    passes: [FullscreenProgram; 3],
}

impl Bloom {
    pub fn new(gpu: &GpuDevice, frames_in_flight: usize) -> Result<Self> {
        let format = vk::Format::from_raw(ae_render_abi::module::BLOOM_FORMAT as i32);
        let make = |label: &str, shader: &[u8]| {
            FullscreenProgram::new(gpu, label, shader, 0, 1, &[format], frames_in_flight)
                .with_context(|| format!("{label} program"))
        };
        Ok(Self {
            passes: [
                make("bloom.downsample", spirv!("bloom_downsample.frag"))?,
                make("bloom.blur_x", spirv!("bloom_blur_x.frag"))?,
                make("bloom.blur_y", spirv!("bloom_blur_y.frag"))?,
            ],
        })
    }

    /// Record `pass`: `input` sampled in the fragment stage into `output`
    /// (in `COLOR_ATTACHMENT_OPTIMAL`).
    pub fn record(
        &self,
        ctx: &PassContext<'_>,
        pass: BloomPass,
        input: ImageBinding,
        _output: ImageBinding,
        sampler: vk::Sampler,
    ) -> Result<()> {
        self.passes[pass as usize].record(ctx, &[input], sampler)
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        for program in &mut self.passes {
            program.destroy(gpu);
        }
    }
}
