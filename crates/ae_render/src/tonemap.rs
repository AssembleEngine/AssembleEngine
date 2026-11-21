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
//! value displaying at 1.

use crate::device::GpuDevice;
use crate::executor::{ImageBinding, PassContext};
use crate::fullscreen::FullscreenProgram;
use crate::pipeline::spirv;
use anyhow::Result;
use ash::vk;

/// The tone curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Curve {
    /// Clamped at 1: what the 8-bit scene gave before the HDR target, and
    /// what tests read linear values through.
    Clamp,
    /// Lottes 2016 (the default; see [`Lottes`]).
    #[default]
    Lottes,
}

/// Lottes 2016's curve and its anchors.
///
/// `f(x) = x^a / ((x^a)^d · b + c)`: `a` (contrast) shapes the toe, `d`
/// (shoulder) how fast the shoulder compresses. `b` and `c` are solved so
/// that `mid` displays as `mid_out` and `peak` displays as 1.
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
/// In SDR the display's peak is paper white, hence [`Self::PEAK`] in scene
/// units rather than a ratio; HDR output passes `peak / paper white`.
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
    /// The scene value that displays as white.
    pub peak: f32,
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
    /// above 1, a non-positive `b` or `c`).
    pub fn solve(&self) -> Option<(f32, f32)> {
        let a = self.contrast;
        let d = self.shoulder;
        if !(self.mid > 0.0 && self.mid < self.peak && self.mid_out > 0.0 && self.mid_out < 1.0) {
            return None;
        }
        let m = self.mid.powf(a);
        let p = self.peak.powf(a);
        let b = (m / self.mid_out - p) / (m.powf(d) - p.powf(d));
        let c = p - p.powf(d) * b;
        (b > 0.0 && c > 0.0 && b.is_finite() && c.is_finite()).then_some((b, c))
    }

    /// The curve at `x` (display-linear in, display-linear out).
    pub fn map(&self, x: f32) -> f32 {
        let Some((b, c)) = self.solve() else { return x.min(1.0) };
        let z = x.max(0.0).powf(self.contrast);
        z / (z.powf(self.shoulder) * b + c)
    }

    /// The shader's colour mapping, for tests: the max channel decides, the
    /// other channels scale with it (hue kept), and above paper white a
    /// `smoothstep`-weighted share of the luminance-mapped grey joins in.
    pub fn map_rgb(&self, c: [f32; 3]) -> [f32; 3] {
        let y = crate::lighting::luminance(c);
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
    /// The frame's exposure factor.
    pub fn exposure(&self) -> f32 {
        crate::lighting::exposure(self.ev100)
    }
}

impl Default for ToneParams {
    fn default() -> Self {
        Self {
            ev100: crate::lighting::DEFAULT_EV100,
            curve: Curve::Lottes,
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

impl Tonemap {
    /// For an output of `format`.
    pub fn new(gpu: &GpuDevice, format: vk::Format, frames_in_flight: usize) -> Result<Self> {
        // Three samplers: the scene, the bloom and the 1×1 exposure image.
        // A graph without a bloom or an exposure pass binds the scene to
        // the missing one and the uniform switches it off.
        Ok(Self { program: FullscreenProgram::new(gpu, "post.tonemap", spirv!("tonemap.frag"), 48, 3, &[format], frames_in_flight)? })
    }

    /// Frame slot `slot`'s parameters (before it records): the graph's
    /// bloom, whether it declares the exposure image, and the EV100 the
    /// scene was stored at ([`ToneParams::ev100`]).
    pub fn write(&mut self, slot: usize, params: ToneParams, bloom: bool, exposure: bool) {
        // Exposure was applied before the scene was stored; the auto
        // exposure scales the stored scene by the image's EV against this
        // reference (1 when they agree).
        //
        // Anchors that solve to no curve fall back to the clamp, as
        // `Lottes::map` does, rather than to a scaled identity.
        let solved = params.lottes.solve();
        let (lottes, curve) = match (params.curve, solved) {
            (Curve::Lottes, Some(bc)) => (bc, 1.0),
            _ => ((0.0, 1.0), 0.0),
        };
        let words: [f32; 12] = [
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
    fn bad_anchors_fall_back_to_the_clamp() {
        let lottes = Lottes { mid: 20.0, ..Lottes::default() };
        assert_eq!(lottes.solve(), None);
        assert_eq!(lottes.map(0.5), 0.5);
        assert_eq!(lottes.map(2.0), 1.0);
    }
}
