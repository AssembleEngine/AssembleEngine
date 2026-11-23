//! `post.tonemap`: the HDR scene ([`ae_render_abi::module::SCENE_FORMAT`],
//! linear light, stored at the camera's reference exposure) to the output, as
//! one fullscreen triangle (`shaders/tonemap.frag`). There is no tone curve:
//! the pass applies the automatic exposure ([`crate::exposure`]: it scales
//! the stored scene by the eased EV read from the 1×1 exposure image), adds
//! the bloom, and maps the working space to the gamut it writes. The output
//! is the only limit on the light: an SDR target saturates at 1 (paper
//! white), and in HDR `post.encode` clips at the display's peak.
//!
//! The pass writes the target in SDR. In HDR it writes the display
//! intermediate instead ([`crate::encode::DISPLAY_FORMAT`], paper-white
//! units), which `post.encode` maps to the output.

use crate::bindless::Bindless;
use crate::colorspace::{self, Gamut, WorkingSpace};
use crate::device::GpuDevice;
use crate::executor::{ImageBinding, PassContext};
use crate::fullscreen::BindlessProgram;
use crate::pipeline::spirv;
use anyhow::Result;
use ash::vk;

/// The camera exposure, the bloom's strength and the sun. The exposure is
/// applied by the scene passes before they store into the HDR buffer
/// ([`crate::lighting::exposure`]); with `auto` the display exposure is the
/// metered, eased EV from the exposure image and the tonemap scales the
/// stored scene by the difference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToneParams {
    pub ev100: f32,
    /// The sun's direct-normal illuminance (lux): a capture override for
    /// [`crate::lighting::SUN_ILLUMINANCE`].
    pub sun_illuminance: f32,
    /// How much of the graph's bloom joins the scene (RA3's 0.3; 0 turns it
    /// off without changing the graph).
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
        Self {
            ev100: crate::lighting::DEFAULT_EV100,
            sun_illuminance: crate::lighting::SUN_ILLUMINANCE,
            bloom: BLOOM_STRENGTH,
            auto: false,
        }
    }
}

pub struct Tonemap {
    program: BindlessProgram,
}

/// How much of the bloom's blurred light joins the scene (RA3's
/// `kBloomStrength`).
pub const BLOOM_STRENGTH: f32 = 0.3;

/// The tonemap's uniform block: one vec4, a mat3 (columns padded to vec4)
/// and the output's luminance row (w: compress).
const UNIFORM_BYTES: u32 = 16 + 48 + 16;

impl Tonemap {
    /// For an output of `format`.
    pub fn new(gpu: &GpuDevice, bindless: &Bindless, format: vk::Format, frames_in_flight: usize) -> Result<Self> {
        // Three samplers: the scene, the bloom and the 1×1 exposure image.
        // A graph without a bloom or an exposure pass binds the scene to
        // the missing one and the uniform switches it off.
        Ok(Self {
            program: BindlessProgram::new(
                gpu,
                bindless,
                "post.tonemap",
                spirv!("tonemap.frag"),
                UNIFORM_BYTES,
                &[format],
                frames_in_flight,
            )?,
        })
    }

    /// Frame slot `slot`'s parameters (before it records): the graph's
    /// bloom, whether it declares the exposure image, and the EV100 the
    /// scene was stored at ([`ToneParams::ev100`]).
    pub fn write(
        &mut self,
        slot: usize,
        params: ToneParams,
        bloom: bool,
        exposure: bool,
        working: WorkingSpace,
        output: Gamut,
    ) {
        // Exposure was applied before the scene was stored; the auto
        // exposure scales the stored scene by the image's EV against this
        // reference (1 when they agree).
        let mut words: Vec<f32> = vec![
            params.bloom,
            if bloom { 1.0 } else { 0.0 },
            if exposure { 1.0 } else { 0.0 },
            params.ev100,
        ];
        // `output` is the gamut the tonemap writes in: SDR's BT.709 (the
        // scene, in the working space, is mapped to it and, when the working
        // space is wider, compressed into it), or in HDR the working space
        // itself (the display intermediate, which post.encode maps to the
        // output gamut once). Both matrices are the identity at the default.
        let to_output = colorspace::matrix(working.gamut(), output);
        for col in colorspace::columns(to_output) {
            words.extend_from_slice(&col);
            words.push(0.0);
        }
        // The compression's grey is the output's luminance (the colour is
        // in the output's primaries by then); w: 1 when the output gamut is
        // narrower than the working space.
        let out_luma = output.luminance_row();
        words.extend_from_slice(&[
            out_luma[0],
            out_luma[1],
            out_luma[2],
            if output.contains(working) { 0.0 } else { 1.0 },
        ]);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.program.write_uniforms(slot, &bytes);
    }

    /// The scene, plus the bloom's blurred light and the exposure image when
    /// the graph has them (the scene stands in for either otherwise,
    /// unread).
    pub fn record(
        &self,
        ctx: &PassContext<'_>,
        scene: ImageBinding,
        bloom: Option<ImageBinding>,
        exposure: Option<ImageBinding>,
        sampler: u32,
    ) -> Result<()> {
        let index = |b: ImageBinding| b.sampled.unwrap_or(0);
        self.program.record(
            ctx,
            &[
                index(scene),
                index(bloom.unwrap_or(scene)),
                index(exposure.unwrap_or(scene)),
            ],
            &[sampler],
        )
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        self.program.destroy(gpu);
    }
}
