//! `post.tonemap`: the HDR scene ([`ae_render_abi::module::SCENE_FORMAT`],
//! linear light, stored at the camera's reference exposure) to the output by
//! a tone curve (`shaders/tonemap.frag`), as one fullscreen triangle. With
//! automatic exposure ([`crate::exposure`]) it first scales the stored scene
//! by the eased EV read from the 1×1 exposure image.

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
/// ([`crate::lighting::exposure`]), so the tonemap itself only curves; with
/// `auto` the display exposure is the metered, eased EV from the exposure
/// image and the tonemap scales the stored scene by the difference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToneParams {
    pub ev100: f32,
    pub curve: Curve,
    /// The sun's direct-normal illuminance (lux): a capture override for
    /// [`crate::lighting::SUN_ILLUMINANCE`].
    pub sun_illuminance: f32,
    /// How much of the graph's bloom joins the scene before the curve
    /// (RA3's 0.3; 0 turns it off without changing the graph).
    pub bloom: f32,
    /// Automatic exposure (eye adaptation, [`crate::exposure`]): the scene
    /// is stored at `ev100` and the tonemap applies the metered, eased EV
    /// instead. Off: `ev100` is the exposure the scene is stored and shown
    /// at.
    pub auto: bool,
}

impl ToneParams {
    /// The frame's exposure factor.
    pub fn exposure(&self) -> f32 {
        crate::lighting::exposure(self.ev100)
    }
}

impl Default for ToneParams {
    fn default() -> Self {
        Self { ev100: crate::lighting::DEFAULT_EV100, curve: Curve::Aces, sun_illuminance: crate::lighting::SUN_ILLUMINANCE, bloom: BLOOM_STRENGTH, auto: false }
    }
}

pub struct Tonemap {
    program: FullscreenProgram,
}

/// How much of the bloom's blurred light joins the scene (RA3's
/// `kBloomStrength`).
pub const BLOOM_STRENGTH: f32 = 0.3;

impl Tonemap {
    /// For an output of `format`.
    pub fn new(gpu: &GpuDevice, format: vk::Format, frames_in_flight: usize) -> Result<Self> {
        // Three samplers: the scene, the bloom and the 1×1 exposure image.
        // A graph without a bloom or an exposure pass binds the scene to
        // the missing one and the uniform switches it off.
        Ok(Self { program: FullscreenProgram::new(gpu, "post.tonemap", spirv!("tonemap.frag"), 32, 3, &[format], frames_in_flight)? })
    }

    /// Frame slot `slot`'s parameters (before it records): the graph's
    /// bloom, whether it declares the exposure image, and the EV100 the
    /// scene was stored at ([`ToneParams::ev100`]).
    pub fn write(&mut self, slot: usize, params: ToneParams, bloom: bool, exposure: bool) {
        // Exposure was applied before the scene was stored; the auto
        // exposure scales the stored scene by the image's EV against this
        // reference (1 when they agree).
        let words: [f32; 8] = [params.bloom, if params.curve == Curve::Aces { 1.0 } else { 0.0 }, if bloom { 1.0 } else { 0.0 }, if exposure { 1.0 } else { 0.0 }, params.ev100, 0.0, 0.0, 0.0];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.program.write_uniforms(slot, &bytes);
    }

    /// The scene, plus the bloom's blurred light and the exposure image when
    /// the graph has them (the scene stands in for either otherwise,
    /// unread).
    pub fn record(&self, ctx: &PassContext<'_>, scene: ImageBinding, bloom: Option<ImageBinding>, exposure: Option<ImageBinding>, sampler: vk::Sampler) -> Result<()> {
        self.program.record(ctx, &[scene, bloom.unwrap_or(scene), exposure.unwrap_or(scene)], sampler)
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        self.program.destroy(gpu);
    }
}
