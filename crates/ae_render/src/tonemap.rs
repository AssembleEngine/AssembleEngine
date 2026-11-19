//! `post.tonemap`: the HDR scene ([`ae_render_abi::module::SCENE_FORMAT`],
//! linear light) to the output, by exposure and a tone curve
//! (`shaders/tonemap.frag`), as one fullscreen triangle.

use crate::device::GpuDevice;
use crate::executor::{ImageBinding, PassContext};
use crate::fullscreen::FullscreenProgram;
use crate::pipeline::spirv;
use anyhow::Result;
use ash::vk;

/// The tone curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Curve {
    /// Clamped at 1: what the 8-bit scene gave before the HDR target.
    #[default]
    Clamp,
    /// ACES (Narkowicz's fit).
    Aces,
}

/// Exposure and curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToneParams {
    pub exposure: f32,
    pub curve: Curve,
}

impl Default for ToneParams {
    fn default() -> Self {
        Self { exposure: 1.0, curve: Curve::Clamp }
    }
}

pub struct Tonemap {
    program: FullscreenProgram,
}

impl Tonemap {
    /// For an output of `format`.
    pub fn new(gpu: &GpuDevice, format: vk::Format, frames_in_flight: usize) -> Result<Self> {
        Ok(Self { program: FullscreenProgram::new(gpu, "post.tonemap", spirv!("tonemap.frag"), 16, 1, &[format], frames_in_flight)? })
    }

    /// Frame slot `slot`'s parameters (before it records).
    pub fn write(&mut self, slot: usize, params: ToneParams) {
        let words = [params.exposure, if params.curve == Curve::Aces { 1.0 } else { 0.0 }, 0.0, 0.0];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.program.write_uniforms(slot, &bytes);
    }

    pub fn record(&self, ctx: &PassContext<'_>, scene: ImageBinding, sampler: vk::Sampler) -> Result<()> {
        self.program.record(ctx, &[scene], sampler)
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        self.program.destroy(gpu);
    }
}
