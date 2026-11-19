//! `post.tonemap`: the HDR scene ([`ae_render_abi::module::SCENE_FORMAT`],
//! linear light, already exposed by the scene passes) to the output by a
//! tone curve (`shaders/tonemap.frag`), as one fullscreen triangle.

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

/// The camera exposure and the curve. The exposure is applied by the scene
/// passes before they store into the HDR buffer
/// ([`crate::lighting::exposure`]), so the tonemap itself only curves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToneParams {
    pub ev100: f32,
    pub curve: Curve,
    /// The sun's direct-normal illuminance (lux): a capture override for
    /// [`crate::lighting::SUN_ILLUMINANCE`].
    pub sun_illuminance: f32,
}

impl ToneParams {
    /// The frame's exposure factor.
    pub fn exposure(&self) -> f32 {
        crate::lighting::exposure(self.ev100)
    }
}

impl Default for ToneParams {
    fn default() -> Self {
        Self { ev100: crate::lighting::DEFAULT_EV100, curve: Curve::Clamp, sun_illuminance: crate::lighting::SUN_ILLUMINANCE }
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
        // Exposure was applied before the scene was stored; only the curve
        // is left here.
        let words: [f32; 4] = [1.0, if params.curve == Curve::Aces { 1.0 } else { 0.0 }, 0.0, 0.0];
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
