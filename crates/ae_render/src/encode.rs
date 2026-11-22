//! `post.encode`: the display intermediate ([`DISPLAY_FORMAT`], linear light
//! in units of paper white, in the working space, up to peak / paper white)
//! to the output: HDR10 (ST.2084 / PQ in BT.2020 at `v * paper_white` nits)
//! or scRGB (BT.709-relative, `v * paper_white / 80`, negatives allowed), as
//! one fullscreen triangle (`shaders/encode.frag`).
//!
//! An SDR graph has no display intermediate and no encode pass: `post.tonemap`
//! writes the sRGB target directly, so SDR output is byte-identical to what
//! it was before HDR existed. An HDR graph declares the display between the
//! tonemap and the encode (a render module's grade and vignette write
//! intermediates from it; the UI is still to come), and the encode maps the
//! display's values to the output's encoding.
//!
//! The display is in the working space ([`WorkingSpace`]); the encode maps
//! it to the output gamut ([`Gamut`], P3 by default for HDR10), compresses
//! out-of-gamut colour toward the achromatic axis when the output gamut is
//! narrower than the working space ([`crate::colorspace::compress`]), and
//! then re-expresses it in the container's primaries (BT.2020 for HDR10,
//! BT.709-relative extended for scRGB). With the default BT.709 working
//! space the map is a pure re-expression; stage 3's wider space is where the
//! compression happens.
//!
//! There is no tone curve: the display intermediate carries the exposed
//! scene as is, and this pass clips it at the display's peak
//! ([`DisplayLevels::headroom`], `peak / paper white`), so a fireball's core
//! decodes above paper white up to what the display shows.

use crate::colorspace::{self, Gamut, WorkingSpace};
use crate::device::GpuDevice;
use crate::executor::{ImageBinding, PassContext};
use crate::fullscreen::FullscreenProgram;
use crate::pipeline::spirv;
use anyhow::Result;
use ash::vk;

/// The paper white the WSI's HDR10 description implies (M13 stage 5): the
/// NVIDIA Wayland WSI sends `st2084_pq` with no `set_luminances`, so the
/// protocol's default reference white for PQ, 203 nits, applies.
pub const HDR10_REFERENCE_NITS: f32 = 203.0;

/// The paper white the WSI's scRGB description carries (M13 stage 5): the
/// NVIDIA Wayland WSI sends `set_luminances(0, 80, 203)` — 1.0 in the
/// signal is 80 nits, and the reference white is 203 nits, so paper white
/// is 203 (`sc_rgb_scale` 2.5375). The protocol's default would be 80.
pub const SCRGB_REFERENCE_NITS: f32 = 203.0;

/// The display intermediate: linear light in paper-white units, sampled by
/// the encode pass (and, with the module ABI, by the grade and the UI).
pub const DISPLAY_FORMAT: vk::Format = vk::Format::from_raw(ae_render_abi::module::DISPLAY_FORMAT as i32);

/// The display image's name in a graph.
pub const DISPLAY_NAME: &str = ae_render_abi::module::DISPLAY_NAME;

/// How the output image encodes the display's light.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OutputEncoding {
    /// An sRGB attachment: `post.tonemap` writes the target directly, with
    /// no display intermediate and no encode pass.
    #[default]
    Sdr,
    /// HDR10: ST.2084 (PQ) in BT.2020 primaries, at `v * paper_white` nits,
    /// 10 bits per channel (`A2B10G10R10_UNORM_PACK32`).
    Hdr10Pq,
    /// scRGB: BT.709-relative linear light at `v * paper_white / 80` (1.0 =
    /// 80 nits), 16-bit float per channel (`R16G16B16A16_SFLOAT`). Values
    /// below 0 are kept: they are out of gamut rather than black.
    ScRgb,
}

impl OutputEncoding {
    /// Whether the graph keeps a display intermediate and ends in
    /// `post.encode`.
    pub fn uses_display(self) -> bool {
        !matches!(self, Self::Sdr)
    }

    /// The encoding as errors and logs name it.
    pub fn label(self) -> &'static str {
        match self {
            Self::Sdr => "SDR",
            Self::Hdr10Pq => "HDR10 (PQ)",
            Self::ScRgb => "scRGB",
        }
    }

    /// The readback image's format (the SDR capture is sRGB, as before).
    pub fn readback_format(self) -> vk::Format {
        match self {
            Self::Sdr => vk::Format::R8G8B8A8_SRGB,
            Self::Hdr10Pq => vk::Format::A2B10G10R10_UNORM_PACK32,
            Self::ScRgb => DISPLAY_FORMAT,
        }
    }

    /// The readback buffer's bytes per pixel.
    pub fn bytes_per_pixel(self) -> u64 {
        match self {
            Self::Sdr | Self::Hdr10Pq => 4,
            Self::ScRgb => 8,
        }
    }

    /// The output gamut when none is set: BT.709 for SDR, P3 for HDR10 (the
    /// reference display's, and the HDR10 default in M13), BT.2020 for
    /// scRGB (an extended 709-relative signal: its container can carry any
    /// primaries as negative components, so the widest gamut compresses
    /// nothing by default and negatives stay).
    pub fn default_gamut(self) -> Gamut {
        match self {
            Self::Sdr => Gamut::Bt709,
            Self::Hdr10Pq => Gamut::P3,
            Self::ScRgb => Gamut::Bt2020,
        }
    }

    /// The primaries the encoded signal is expressed in (the HDR10
    /// container's BT.2020, or BT.709 for scRGB and sRGB).
    pub fn container_gamut(self) -> Gamut {
        match self {
            Self::Hdr10Pq => Gamut::Bt2020,
            Self::Sdr | Self::ScRgb => Gamut::Bt709,
        }
    }

    /// The wire value a render module's declaration receives
    /// ([`ae_render_abi::module::OutputEncoding`]).
    pub fn to_wire(self) -> u32 {
        use ae_render_abi::module::OutputEncoding as Wire;
        match self {
            Self::Sdr => Wire::Sdr as u32,
            Self::Hdr10Pq => Wire::Hdr10Pq as u32,
            Self::ScRgb => Wire::ScRgb as u32,
        }
    }
}

/// The display's white and its peak, in nits.
///
/// Paper white is the display-linear 1.0 (`post.tonemap`'s output, and the
/// SDR white); the peak is the brightest the display shows, so the tone
/// curve's peak anchor displays at `peak / paper_white`. The defaults are
/// PQ's protocol reference (203 nits) and a 1000-nit peak; the client uses
/// the WSI's own reference (203 nits for PQ here, and for scRGB) and SDL's
/// display peak (441 nits here), with `--paper-white`/`--peak` overrides
/// (M13 stage 5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayLevels {
    /// Paper white (diffuse white) in nits.
    pub paper_white: f32,
    /// The display's peak in nits.
    pub peak: f32,
}

impl DisplayLevels {
    /// The levels that make an HDR frame carry the SDR one's values: paper
    /// white at the SDR white (80 nits, scRGB's own reference) and no
    /// headroom.
    pub const FLAT: Self = Self {
        paper_white: 80.0,
        peak: 80.0,
    };

    /// How far the peak is above paper white: the tone curve's peak anchor
    /// displays at this, in paper-white units.
    pub fn headroom(&self) -> f32 {
        self.peak / self.paper_white
    }

    /// scRGB's scale: display-linear 1.0 = 1.0 scRGB at 80 nits, so paper
    /// white is `paper_white / 80`.
    pub fn sc_rgb_scale(&self) -> f32 {
        self.paper_white / 80.0
    }
}

impl Default for DisplayLevels {
    fn default() -> Self {
        Self {
            paper_white: 203.0,
            peak: 1000.0,
        }
    }
}

/// The ST.2084 OETF: `nits` to a PQ code in 0..1, in f32 (the shader's
/// mirror; the sim's f32 rules bind the sim, not render shaders or the
/// host-side solve).
pub fn pq(nits: f32) -> f32 {
    const M1: f32 = 0.1593017578125;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;
    let n = (nits / 10_000.0).max(0.0);
    let n1 = n.powf(M1);
    ((C1 + C2 * n1) / (1.0 + C3 * n1)).powf(M2)
}

/// [`pq`] to the 10-bit code the HDR10 image stores (0..1023).
pub fn pq_code(nits: f32) -> u16 {
    (pq(nits) * 1023.0).round() as u16
}

/// The ST.2084 EOTF (the inverse of [`pq`]), for decoding a readback: a PQ
/// code in 0..1 to nits.
pub fn pq_eotf(code: f32) -> f32 {
    const M1: f32 = 0.1593017578125;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;
    let n = code.max(0.0).powf(1.0 / M2);
    let num = (n - C1).max(0.0);
    10_000.0 * (num / (C2 - C3 * n)).powf(1.0 / M1)
}

/// BT.709 to BT.2020 primaries (linear light, both D65): the HDR10
/// container's primaries. A pure re-expression (`BT.709` is inside
/// `BT.2020`), so in-gamut values stay in gamut and each channel is a convex
/// combination of the input's.
pub fn bt709_to_bt2020(c: [f32; 3]) -> [f32; 3] {
    colorspace::convert(Gamut::Bt709, Gamut::Bt2020, c)
}

/// The shader's whole mapping for one display-linear colour, in the working
/// space: clamp at the peak, map to the output gamut (compressing when it is
/// narrower than the working space), then PQ in BT.2020 or the scRGB scale
/// in BT.709-relative light, as the GPU pass does. The CPU mirror the tests
/// decode against.
#[allow(clippy::too_many_arguments)]
pub fn encode_rgb(
    encoding: OutputEncoding,
    levels: DisplayLevels,
    working: WorkingSpace,
    gamut: Gamut,
    c: [f32; 3],
) -> [f32; 3] {
    let peak = levels.headroom();
    let v = c.map(|x| x.min(peak));
    let mut out = colorspace::convert(working.gamut(), gamut, v);
    if !gamut.contains(working) {
        out = colorspace::compress(gamut, out);
    }
    let container = colorspace::convert(gamut, encoding.container_gamut(), out).map(|x| x.min(peak));
    match encoding {
        OutputEncoding::Sdr => container,
        OutputEncoding::ScRgb => container.map(|x| x * levels.sc_rgb_scale()),
        OutputEncoding::Hdr10Pq => container.map(|x| pq(x.max(0.0) * levels.paper_white)),
    }
}

/// The encode pass's uniform block: the levels, two mat3s (columns padded
/// to vec4) and the output gamut's luminance row.
const UNIFORM_BYTES: u32 = 16 + 2 * 48 + 16;

pub struct Encode {
    program: FullscreenProgram,
}

impl Encode {
    /// For an HDR output of `encoding` (not [`OutputEncoding::Sdr`]) whose
    /// image is `format`.
    pub fn new(gpu: &GpuDevice, encoding: OutputEncoding, format: vk::Format, frames_in_flight: usize) -> Result<Self> {
        anyhow::ensure!(encoding.uses_display(), "an SDR output has no post.encode pass");
        Ok(Self {
            program: FullscreenProgram::new(
                gpu,
                "post.encode",
                spirv!("encode.frag"),
                UNIFORM_BYTES,
                1,
                &[format],
                frames_in_flight,
            )?,
        })
    }

    /// Frame slot `slot`'s uniform: the levels, the encoding, the working
    /// space to the output gamut and the output gamut to the container, and
    /// whether the output gamut is narrower than the working space (so the
    /// pass compresses). None of this varies within a run in practice, but
    /// it is frame state like the levels (a module may redeclare).
    pub fn write(
        &mut self,
        slot: usize,
        encoding: OutputEncoding,
        levels: DisplayLevels,
        working: WorkingSpace,
        gamut: Gamut,
    ) {
        let compress = !gamut.contains(working);
        let words: [f32; 4] = [
            levels.paper_white,
            levels.peak,
            if encoding == OutputEncoding::ScRgb { 1.0 } else { 0.0 },
            if compress { 1.0 } else { 0.0 },
        ];
        let to_output = colorspace::matrix(working.gamut(), gamut);
        let to_container = colorspace::matrix(gamut, encoding.container_gamut());
        let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        // std140 stores a mat3 as three columns, each padded to a vec4.
        for m in [to_output, to_container] {
            for col in colorspace::columns(m) {
                for v in col {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes.extend_from_slice(&0f32.to_le_bytes());
            }
        }
        // The compression measures luminance in the output gamut (the
        // matrix keeps it there).
        let luma = gamut.luminance_row();
        for v in [luma[0], luma[1], luma[2], 0.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.program.write_uniforms(slot, &bytes);
    }

    /// Record the display image to the pass's colour attachment.
    pub fn record(&self, ctx: &PassContext<'_>, display: ImageBinding, sampler: vk::Sampler) -> Result<()> {
        self.program.record(ctx, &[display], sampler)
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        self.program.destroy(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_codes_match_st_2084() {
        // ST.2084's OETF at 100, 203 and 1000 nits is 0.508078, 0.580689 and
        // 0.751827; a 10-bit code is round(v * 1023).
        for (nits, code) in [(100.0, 520u16), (203.0, 594), (1000.0, 769), (10_000.0, 1023)] {
            assert_eq!(pq_code(nits), code, "{nits} nits");
        }
        // Just inside one code either side: the digits the gate measures in.
        assert!((pq_code(100.0) as i32 - 519).abs() == 1);
        // The EOTF inverts the OETF (in f32, to well inside a code).
        for nits in [0.0, 0.5, 1.0, 100.0, 203.0, 441.0, 1000.0, 10_000.0] {
            let back = pq_eotf(pq(nits));
            assert!((back - nits).abs() <= 0.01 * nits.max(1.0), "{nits} -> {back}");
        }
    }

    #[test]
    fn sc_rgb_is_relative_to_80_nits() {
        // 80-nit paper white: display 1.0 is scRGB 1.0 (the default gamut
        // carries it through BT.2020 and back, so within float rounding).
        let flat = encode_rgb(
            OutputEncoding::ScRgb,
            DisplayLevels::FLAT,
            WorkingSpace::Bt709,
            Gamut::Bt2020,
            [1.0, 0.5, 0.0],
        );
        for (got, want) in flat.iter().zip([1.0, 0.5, 0.0]) {
            assert!((got - want).abs() < 1e-6, "{flat:?}");
        }
        // 203 nits: the same light is 2.5375 scRGB.
        let levels = DisplayLevels {
            paper_white: 203.0,
            peak: 1000.0,
        };
        assert!((levels.sc_rgb_scale() - 2.5375).abs() < 1e-6);
        let v = encode_rgb(
            OutputEncoding::ScRgb,
            levels,
            WorkingSpace::Bt709,
            Gamut::Bt2020,
            [0.4, 0.2, 0.0],
        );
        assert!((v[0] - 1.015).abs() < 1e-5, "{v:?}");
        // Negatives pass (out of gamut, not black): the default scRGB gamut
        // is wider than the working space, so nothing is compressed.
        let neg = encode_rgb(
            OutputEncoding::ScRgb,
            levels,
            WorkingSpace::Bt709,
            Gamut::Bt2020,
            [-0.1, 0.0, 0.0],
        )[0];
        assert!((neg - -0.1 * levels.sc_rgb_scale()).abs() < 1e-5, "{neg}");
    }

    #[test]
    fn nothing_encodes_above_the_peak() {
        let levels = DisplayLevels {
            paper_white: 203.0,
            peak: 441.0,
        };
        // Display 8.0 is above the display's 2.172: it clamps at the peak.
        let top = pq(441.0);
        for c in [levels.headroom(), levels.headroom() * 4.0, 16.0] {
            let v = encode_rgb(OutputEncoding::Hdr10Pq, levels, WorkingSpace::Bt709, Gamut::P3, [c; 3]);
            assert!((v[0] - top).abs() < 1e-6, "{c}: {v:?} vs {top}");
        }
        // Paper white is PQ's 203 nits.
        let white = encode_rgb(
            OutputEncoding::Hdr10Pq,
            levels,
            WorkingSpace::Bt709,
            Gamut::P3,
            [1.0; 3],
        );
        assert!((white[0] - pq(203.0)).abs() < 1e-6, "{white:?}");
    }

    #[test]
    fn bt_2020_keeps_white_and_luminance() {
        // White stays white, and the luminance is the same in both spaces
        // (BT.2020's coefficients on the same light).
        let w = bt709_to_bt2020([1.0; 3]);
        assert!(w.iter().all(|v| (v - 1.0).abs() < 1e-6), "{w:?}");
        let c = [0.3, 0.5, 0.2];
        let b = bt709_to_bt2020(c);
        let y = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let y2020 = |c: [f32; 3]| 0.2627 * c[0] + 0.6780 * c[1] + 0.0593 * c[2];
        assert!((y(c) - y2020(b)).abs() < 1e-5, "{} vs {}", y(c), y2020(b));
        // In gamut stays in gamut, channel by channel.
        for c in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.2, 0.1]] {
            assert!(bt709_to_bt2020(c).iter().all(|v| *v >= 0.0 && *v <= 1.0), "{c:?}");
        }
    }
}
