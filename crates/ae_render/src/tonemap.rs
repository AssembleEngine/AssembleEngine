//! `post.tonemap`: the HDR scene ([`ae_render_abi::module::SCENE_FORMAT`],
//! linear light, stored at the camera's reference exposure) to the output by
//! a tone curve (`shaders/tonemap.frag`), as one fullscreen triangle. With
//! automatic exposure ([`crate::exposure`]) it first scales the stored scene
//! by the eased EV read from the 1×1 exposure image.
//!
//! The curve is Lottes 2016's (`x^a / ((x^a)^d · b + c)`, GDC "Advanced
//! Techniques and Optimization of HDR"), applied to the max channel so a
//! bright emitter keeps its hue, with a crosstalk step toward the
//! luminance-mapped grey above paper white so a fireball's core whitens while
//! its edge stays orange. `b` and `c` are solved on the CPU from two anchors:
//! the pre-curve key ([`crate::lighting::KEY_LUMINANCE`], the automatic
//! exposure's target) displaying at [`Lottes::MID_OUT`], and a peak scene
//! value displaying at [`Lottes::peak_out`] (1 in SDR; the display's
//! peak / paper white in HDR, [`crate::encode`]).
//!
//! The pass writes the target in SDR. In HDR it writes the display
//! intermediate instead ([`crate::encode::DISPLAY_FORMAT`], paper-white
//! units), which `post.encode` maps to the output.

use crate::colorspace::{self, Gamut, WorkingSpace};
use crate::device::GpuDevice;
use crate::executor::{ImageBinding, PassContext};
use crate::fullscreen::FullscreenProgram;
use crate::pipeline::spirv;
use anyhow::Result;
use ash::vk;

/// The tone curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Curve {
    /// No curve (the default): the exposed colour passes as is. The output
    /// limits it: an SDR target saturates at 1 (paper white), and in HDR
    /// `post.encode` clips at the display's peak.
    #[default]
    None,
    /// Lottes 2016 (see [`Lottes`]): a toe and a shoulder that roll the
    /// scene off into the output's peak.
    Lottes,
}

/// Lottes 2016's curve and its anchors.
///
/// `f(x) = x^a / ((x^a)^d · b + c)`: `a` (contrast) shapes the toe, `d`
/// (shoulder) how fast the shoulder compresses. `b` and `c` are solved so
/// that `mid` displays as `mid_out` and `peak` displays as `peak_out`.
///
/// The mid anchor is the metered key: the reference scene's sunlit ground at
/// [`crate::lighting::DEFAULT_EV100`] meters at
/// [`crate::lighting::KEY_LUMINANCE`], and it displays at [`Self::MID_OUT`]
/// (0.40 linear: the M13 look review's gate, from 0.596 with Narkowicz's ACES
/// fit). `a` and `d` are 1. With these anchors `d > 1` makes the curve fall
/// again past the peak (values above it would display darker than white), so
/// 1 is the largest monotone shoulder; `a` 1 keeps the toe a plain function
/// of x, which the solved denominator still lifts a little (its slope at zero
/// is `1 / c`, about 1.29).
///
/// [`Self::PEAK`] is in scene units and [`Self::peak_out`] is what it
/// displays at, in paper-white units: 1 in SDR, where the display's peak is
/// paper white, and `peak / paper white` in HDR ([`crate::encode`]). The mid
/// anchor keeps its place (0.40 of paper white) either way, so the metered
/// key displays where SDR shows it; the curve above it is deliberately
/// different in HDR, or the extra display range would go unused. Below it
/// the curves differ too: with `a = 1` the solved toe is flatter in HDR
/// (slope `1 / c` at zero: 1.29 in SDR, 0.95 at 441/203 nits), so the darks
/// display about 20% darker (docs/lighting.md).
///
/// Which anchor moves for HDR is a choice. The scene's peak (16) is a
/// property of the content, chosen in stage 1 as the top of its range (about
/// ten times the sunlit ground, above any emitter); it cannot change with
/// the display. Mapping it to the display's peak spends the display's whole
/// range on the content's whole range. The alternative — moving the scene's
/// peak to `16 × headroom` — would leave the content's brightest value below
/// the display's peak and reserve the top of the range for scene light the
/// content does not have; scaling both anchors with the headroom would
/// instead brighten every midtone. At `peak_out == 1` the curve is the SDR
/// one exactly (the GPU test
/// `hdr10_readback_matches_the_sdr_frame_at_paper_white_peak`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lottes {
    /// `a`: the toe's contrast.
    pub contrast: f32,
    /// `d`: the shoulder's compression.
    pub shoulder: f32,
    /// The scene value of the mid anchor (pre-curve display-linear).
    pub mid: f32,
    /// What the mid anchor displays as.
    pub mid_out: f32,
    /// The scene value that displays as [`Self::peak_out`].
    pub peak: f32,
    /// What [`Self::peak`] displays at, in paper-white units: 1 in SDR, the
    /// display's peak / paper white in HDR ([`crate::encode`]).
    pub peak_out: f32,
}

impl Lottes {
    /// Where the metered key displays, in display-linear units.
    pub const MID_OUT: f32 = 0.40;
    /// The scene value that displays as white: about ten times the sunlit
    /// ground's, so emitters above it still compress into a gradient (the
    /// fireball's core and its bloom) instead of a flat white disc.
    pub const PEAK: f32 = 16.0;
    /// How much of the luminance-mapped grey joins a bright emitter (0: none,
    /// 1: fully grey) once it is above paper white.
    pub const CROSSTALK: f32 = 0.7;
    /// Crosstalk is off at this scene value and complete at
    /// [`Self::CROSSTALK_PEAK`] (`smoothstep` between them).
    pub const CROSSTALK_START: f32 = 2.0;
    pub const CROSSTALK_PEAK: f32 = Self::PEAK;

    /// `b`, `c` solved from the anchors, or `None` when the anchors do not
    /// describe a monotone curve (`mid` at or above `peak`, `mid_out` at or
    /// above `peak_out`, a non-positive `b` or `c`).
    pub fn solve(&self) -> Option<(f32, f32)> {
        let a = self.contrast;
        let d = self.shoulder;
        if !(self.mid > 0.0 && self.mid < self.peak && self.mid_out > 0.0 && self.peak_out > self.mid_out) {
            return None;
        }
        let m = self.mid.powf(a);
        let p = self.peak.powf(a);
        let b = (p / self.peak_out - m / self.mid_out) / (p.powf(d) - m.powf(d));
        // From the peak anchor, not the mid one: algebraically the same, but
        // with `peak_out == 1` it is the SDR solver's expression bit for bit
        // (the mid anchor is then hit to within float rounding), so an SDR
        // capture stays byte-identical.
        let c = p / self.peak_out - p.powf(d) * b;
        (b > 0.0 && c > 0.0 && b.is_finite() && c.is_finite()).then_some((b, c))
    }

    /// The curve at `x` (display-linear in, display-linear out).
    pub fn map(&self, x: f32) -> f32 {
        let Some((b, c)) = self.solve() else { return x.max(0.0) };
        let z = x.max(0.0).powf(self.contrast);
        z / (z.powf(self.shoulder) * b + c)
    }

    /// The shader's colour mapping, for tests: the max channel decides, the
    /// other channels scale with it (hue kept), and above paper white a
    /// `smoothstep`-weighted share of the luminance-mapped grey joins in.
    pub fn map_rgb(&self, c: [f32; 3]) -> [f32; 3] {
        self.map_rgb_in(WorkingSpace::Bt709, c)
    }

    /// [`Self::map_rgb`] with the working space's luminance row (the
    /// shader's crosstalk uses the scene's row).
    pub fn map_rgb_in(&self, working: WorkingSpace, c: [f32; 3]) -> [f32; 3] {
        let y = working.gamut().luminance(c);
        let mx = c[0].max(c[1]).max(c[2]);
        let m = self.map(mx);
        let scale = if mx > 1e-6 { m / mx } else { 0.0 };
        let w = Self::CROSSTALK * smoothstep(Self::CROSSTALK_START, Self::CROSSTALK_PEAK, mx);
        let grey = self.map(y);
        [
            c[0] * scale * (1.0 - w) + grey * w,
            c[1] * scale * (1.0 - w) + grey * w,
            c[2] * scale * (1.0 - w) + grey * w,
        ]
    }
}

impl Default for Lottes {
    fn default() -> Self {
        Self {
            contrast: 1.0,
            shoulder: 1.0,
            mid: crate::lighting::KEY_LUMINANCE,
            mid_out: Self::MID_OUT,
            peak: Self::PEAK,
            peak_out: 1.0,
        }
    }
}

/// Hermite's smoothstep, as GLSL's.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
    /// The curve's shape and anchors (used by [`Curve::Lottes`]).
    pub lottes: Lottes,
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
    /// Auto-exposure's key for this curve: Lottes' mid anchor
    /// ([`crate::lighting::KEY_LUMINANCE`]), or with no curve the lower
    /// [`crate::lighting::KEY_LUMINANCE_NO_CURVE`], which leaves room above
    /// the mean before white.
    pub fn key(&self) -> f32 {
        match self.curve {
            Curve::None => crate::lighting::KEY_LUMINANCE_NO_CURVE,
            Curve::Lottes => crate::lighting::KEY_LUMINANCE,
        }
    }

    /// The frame's exposure factor.
    pub fn exposure(&self) -> f32 {
        crate::lighting::exposure(self.ev100)
    }
}

impl Default for ToneParams {
    fn default() -> Self {
        Self {
            ev100: crate::lighting::DEFAULT_EV100,
            curve: Curve::None,
            lottes: Lottes::default(),
            sun_illuminance: crate::lighting::SUN_ILLUMINANCE,
            bloom: BLOOM_STRENGTH,
            auto: false,
        }
    }
}

pub struct Tonemap {
    program: FullscreenProgram,
}

/// How much of the bloom's blurred light joins the scene (RA3's
/// `kBloomStrength`).
pub const BLOOM_STRENGTH: f32 = 0.3;

/// The tonemap's uniform block: three vec4s, a mat3 (columns padded to
/// vec4), the working space's luminance row and the output's (w: compress).
const UNIFORM_BYTES: u32 = 3 * 16 + 48 + 2 * 16;

impl Tonemap {
    /// For an output of `format`.
    pub fn new(gpu: &GpuDevice, format: vk::Format, frames_in_flight: usize) -> Result<Self> {
        // Three samplers: the scene, the bloom and the 1×1 exposure image.
        // A graph without a bloom or an exposure pass binds the scene to
        // the missing one and the uniform switches it off.
        Ok(Self { program: FullscreenProgram::new(gpu, "post.tonemap", spirv!("tonemap.frag"), UNIFORM_BYTES, 3, &[format], frames_in_flight)? })
    }

    /// Frame slot `slot`'s parameters (before it records): the graph's
    /// bloom, whether it declares the exposure image, and the EV100 the
    /// scene was stored at ([`ToneParams::ev100`]).
    pub fn write(&mut self, slot: usize, params: ToneParams, bloom: bool, exposure: bool, working: WorkingSpace, output: Gamut) {
        // Exposure was applied before the scene was stored; the auto
        // exposure scales the stored scene by the image's EV against this
        // reference (1 when they agree).
        //
        // Anchors that solve to no curve fall back to no curve, rather
        // than to a scaled identity.
        let solved = params.lottes.solve();
        let (lottes, curve) = match (params.curve, solved) {
            (Curve::Lottes, Some(bc)) => (bc, 1.0),
            _ => ((0.0, 1.0), 0.0),
        };
        let mut words: Vec<f32> = vec![
            params.bloom,
            curve,
            if bloom { 1.0 } else { 0.0 },
            if exposure { 1.0 } else { 0.0 },
            params.ev100,
            0.0,
            0.0,
            0.0,
            lottes.0,
            lottes.1,
            params.lottes.contrast,
            params.lottes.shoulder,
        ];
        // `output` is the gamut the tonemap writes in: SDR's BT.709 (the
        // curve's result, in the working space, is mapped to it and, when
        // the working space is wider, compressed into it), or in HDR the
        // working space itself (the display intermediate, which post.encode
        // maps to the output gamut once). Both matrices are the identity at
        // the default, so an SDR capture stays bit-identical.
        let to_output = colorspace::matrix(working.gamut(), output);
        for col in colorspace::columns(to_output) {
            words.extend_from_slice(&col);
            words.push(0.0);
        }
        // The crosstalk's grey is the working space's luminance.
        let luma = working.luminance_row();
        words.extend_from_slice(&[luma[0], luma[1], luma[2], 0.0]);
        // The compression's grey is the output's luminance (the colour is
        // in the output's primaries by then); w: 1 when the output gamut is
        // narrower than the working space.
        let out_luma = output.luminance_row();
        words.extend_from_slice(&[out_luma[0], out_luma[1], out_luma[2], if output.contains(working) { 0.0 } else { 1.0 }]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_anchors_are_hit() {
        let lottes = Lottes::default();
        assert!((lottes.map(lottes.mid) - lottes.mid_out).abs() < 1e-5, "{}", lottes.map(lottes.mid));
        assert!((lottes.map(lottes.peak) - 1.0).abs() < 1e-5, "{}", lottes.map(lottes.peak));
        // Below the mid anchor the curve is above the line (the toe lifts the
        // darks); between the anchors it is below it (the shoulder).
        let half = lottes.map(lottes.mid / 2.0);
        assert!(half > lottes.mid_out / 2.0 && half < lottes.mid / 2.0, "toe: {half}");
        assert!(lottes.map(lottes.mid * 2.0) < lottes.mid_out * 2.0, "shoulder");
        // The shadow lift, in absolute terms: a 5% grey stays below 7%.
        assert!(lottes.map(0.05) < 0.07, "shadows stay dark: {}", lottes.map(0.05));
    }

    #[test]
    fn the_headroom_scales_the_peak() {
        // HDR: the same scene's peak displays at the display's peak / paper
        // white, and the key stays at 0.40 of paper white.
        let lottes = Lottes { peak_out: 441.0 / 203.0, ..Lottes::default() };
        assert!((lottes.map(lottes.mid) - lottes.mid_out).abs() < 1e-5, "{}", lottes.map(lottes.mid));
        assert!((lottes.map(lottes.peak) - 441.0 / 203.0).abs() < 1e-5, "{}", lottes.map(lottes.peak));
        // An emitter between the key and the peak lands between them, above
        // paper white once it is bright enough.
        let emitter = lottes.map(4.2);
        assert!(emitter > 1.0 && emitter < 441.0 / 203.0, "{emitter}");
        // Monotone across the range the display can show.
        let mut last = -1.0;
        for i in 0..4000 {
            let y = lottes.map(i as f32 * 0.05);
            assert!(y >= last - 1e-6);
            last = y;
        }
    }

    #[test]
    fn the_curve_is_monotone() {
        // Up to the peak and far past it: no value displays darker than a
        // dimmer one (a fireball's core must stay the brightest part).
        let lottes = Lottes::default();
        let mut last = -1.0;
        for i in 0..20_000 {
            let x = i as f32 * 0.01;
            let y = lottes.map(x);
            assert!(y >= last - 1e-6, "map({x}) = {y} < {last}");
            last = y;
        }
    }

    #[test]
    fn an_emitter_keeps_its_hue_and_whitens_at_the_core() {
        let lottes = Lottes::default();
        // A fire's orange, from its edge to its core.
        let edge = lottes.map_rgb([0.6, 0.15, 0.02]);
        let mid = lottes.map_rgb([2.0, 0.5, 0.1]);
        let core = lottes.map_rgb([16.0, 4.0, 0.8]);
        for (name, c) in [("edge", edge), ("mid", mid), ("core", core)] {
            assert!(c.iter().all(|v| *v >= 0.0 && *v <= 1.0), "{name} in range: {c:?}");
        }
        // Hue kept above the edge: red stays above green above blue.
        assert!(mid[0] > mid[1] && mid[1] > mid[2], "{mid:?}");
        // The core whitens (green rises relative to red) rather than staying
        // a saturated orange.
        assert!(core[1] / core[0] > mid[1] / mid[0], "core whitens: {core:?} against {mid:?}");
        // And the gradient survives: the edge is dimmer than the core.
        assert!(edge[0] < mid[0] && mid[0] < core[0], "{edge:?} {mid:?} {core:?}");
    }

    #[test]
    fn bad_anchors_fall_back_to_no_curve() {
        let lottes = Lottes { mid: 20.0, ..Lottes::default() };
        assert_eq!(lottes.solve(), None);
        assert_eq!(lottes.map(0.5), 0.5);
        assert_eq!(lottes.map(2.0), 2.0);
    }
}
