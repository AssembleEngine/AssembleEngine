//! Photometric lighting (M12): units, the scene's sun and sky, and exposure.
//!
//! The renderer works in physical units:
//! * one world unit is one metre ([`WORLD_UNIT_METRES`]);
//! * a point or spot light carries an intensity in candela (cd = lm/sr);
//! * the sun is a directional illuminance in lux (lm/m²);
//! * surface emission and luminance are cd/m², as linear RGB;
//! * a Lambertian surface's outgoing radiance is `albedo/pi * E` for the
//!   illuminance `E` it receives;
//! * the camera exposes with `1 / (1.2 * 2^EV100)` before the tone curve;
//!   automatic exposure ([`crate::exposure`]) meters the stored scene and
//!   scales it in the tonemap by the eased EV against that reference.
//!
//! The scene's sun and sky are fixed for now: 100 000 lx of direct normal
//! sun and 20 000 lx of sky on an upward-facing surface, with tints
//! normalised to unit luminance so the scalars are the illuminances. The
//! sky's radiance is [[`sky_radiance`]] (the clear colour, the fog and the
//! water's reflection).

/// One world unit is one metre (the renderer's terrain, units and lights).
pub const WORLD_UNIT_METRES: f32 = 1.0;

/// Direct-normal solar illuminance (lux).
pub const SUN_ILLUMINANCE: f32 = 100_000.0;
/// Sky irradiance on an upward-facing surface (lux).
pub const SKY_ILLUMINANCE: f32 = 20_000.0;
/// The sun's linear tint; normalised to unit luminance by [`sun_tint`].
pub const SUN_TINT: [f32; 3] = [1.0, 0.95, 0.85];
/// The sky's linear tint; normalised to unit luminance by [`sky_tint`].
pub const SKY_TINT: [f32; 3] = [0.45, 0.55, 0.70];

/// The default camera exposure (EV100): calibrated so the sunlit terrain
/// keeps the brightness of the old `1.6` sun coefficient.
pub const DEFAULT_EV100: f32 = 14.0;

/// The deepest illuminance a light is culled to, at exposure 1
/// (`E_cut = MIN_VISIBLE_ILLUMINANCE / exposure`): below one 8-bit code of
/// the tonemapped image before the tone curve.
pub const MIN_VISIBLE_ILLUMINANCE: f32 = 0.002;

/// Auto-exposure's key: the display-linear luminance (before the tone
/// curve) the metered scene's trimmed mean log luminance maps to. Set to
/// what the reference scene, untextured terrain under the full sun
/// ([`SUN_ILLUMINANCE`]), meters at [`DEFAULT_EV100`], so eye adaptation
/// settles on the calibrated look instead of moving it (the GPU test
/// `auto_exposure_holds_the_key_across_sunlight` pins auto against manual
/// there). The tone curve's mid anchor is this value: it displays at
/// [`crate::tonemap::Lottes::MID_OUT`]. A white Lambertian surface under the
/// full sun stores about 1.6 at that EV; the terrain's albedo is well below
/// it.
pub const KEY_LUMINANCE: f32 = 0.5;
/// Auto-exposure's key with no tone curve ([`crate::tonemap::Curve::None`]).
/// With no shoulder, white is only one stop above a 0.5 key: the default
/// map adapted about half an EV brighter than its start at
/// [`DEFAULT_EV100`] and its road and lights clipped (mean sRGB luma 150 at
/// 0.5 s, 186 at 10 s). At 0.32 the same map settles at its start's look
/// (153 at 10 s; `ae-client --capture`, crossing water map, 10,000 units,
/// default camera, 1280×720), and white is 1.6 stops above the key.
pub const KEY_LUMINANCE_NO_CURVE: f32 = 0.32;
/// The brightest auto-exposure (the lowest EV100), four EV above the
/// reference sun's: an overcast dusk still gets the display's range. Night
/// scenes are meant to stay dark, so the bound sits far above a real
/// night's EV (around -2 under a moon).
pub const MIN_EV100: f32 = 10.0;
/// The darkest auto-exposure (the highest EV100): a sunlit snowfield or a
/// blast does not crush the rest of the scene.
pub const MAX_EV100: f32 = 18.0;
/// How fast the exposure follows a brighter scene (a rising target EV), as
/// an exponential rate (1/s: the gap shrinks by `e` every 1/rate seconds,
/// wall-clock time): the eye constricts faster than it opens, and an
/// explosion visibly darkens the image over a second or so rather than
/// snapping.
pub const TOWARD_BRIGHTER_PER_SECOND: f32 = 1.5;
/// How fast it follows a darker scene (a falling target EV), 1/s: dark
/// adaptation is slower, as in the eye and in most engines (a fireball's
/// after-image fades over a few seconds).
pub const TOWARD_DARKER_PER_SECOND: f32 = 0.5;
/// Centre-weighted metering: a block at the centre of the screen counts this
/// many times one at the edge, so what the player looks at sets the
/// exposure.
pub const METER_CENTER_WEIGHT: f32 = 4.0;
/// The centre weight's falloff (a Gaussian's sigma), in screen heights: at
/// 0.25 the weight is halfway down about 0.3 heights from the centre.
pub const METER_CENTER_SIGMA: f32 = 0.25;
/// The fraction of the darkest scene pixels the metering skips.
pub const METER_LOW_PERCENTILE: f32 = 0.02;
/// The fraction of the brightest scene pixels the metering skips (a small
/// sun glint or the sky does not set the exposure).
pub const METER_HIGH_PERCENTILE: f32 = 0.02;

/// The EV100 that maps `mean_log2` (the metered mean log2 luminance of the
/// scene as stored, i.e. exposed at `scene_ev100`) to `key` (the curve's,
/// [`crate::tonemap::ToneParams::key`]).
///
/// The stored luminance is `R * 2^-scene_ev100 / 1.2`; the display's is
/// `R * 2^-ev100 / 1.2`, so the display is the stored scene times
/// `2^(scene_ev100 - ev100)`. Making its mean equal the key gives
/// `ev100 = scene_ev100 + (mean_log2 - log2(key))` (a brighter scene needs
/// less exposure, a higher EV).
pub fn target_ev100(scene_ev100: f32, mean_log2: f32, key: f32) -> f32 {
    scene_ev100 + (mean_log2 - key.log2())
}

/// Ease an EV100 toward `target` over `dt` seconds, exponentially in EV
/// space and clamped to [`MIN_EV100`]..[`MAX_EV100`]:
/// [`TOWARD_BRIGHTER_PER_SECOND`] when the scene brightens (the target EV
/// rises, the exposure falls), [`TOWARD_DARKER_PER_SECOND`] when it
/// darkens. Exponential rather than a capped step, so the metering's small
/// frame-to-frame noise is filtered instead of followed. Mirrored by
/// `shaders/exposure_adapt.comp`.
pub fn adapt_ev100(current: f32, target: f32, dt: f32) -> f32 {
    let target = target.clamp(MIN_EV100, MAX_EV100);
    let rate = if target > current { TOWARD_BRIGHTER_PER_SECOND } else { TOWARD_DARKER_PER_SECOND };
    let blend = 1.0 - (-rate * dt).exp();
    (current + (target - current) * blend).clamp(MIN_EV100, MAX_EV100)
}

/// Luminance weights (Rec. 709).
pub fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// `c` scaled to unit luminance (the chromaticity of a tint whose scalar is
/// an illuminance).
pub fn unit_tint(c: [f32; 3]) -> [f32; 3] {
    let y = luminance(c).max(1e-6);
    [c[0] / y, c[1] / y, c[2] / y]
}

/// The sun's tint in the working space (unit luminance there).
pub fn sun_tint(space: crate::colorspace::WorkingSpace) -> [f32; 3] {
    space.unit_tint(SUN_TINT)
}

/// The sky's tint in the working space (unit luminance there).
pub fn sky_tint(space: crate::colorspace::WorkingSpace) -> [f32; 3] {
    space.unit_tint(SKY_TINT)
}

/// The exposure factor of an EV100 setting: `1 / (1.2 * 2^EV100)`.
pub fn exposure(ev100: f32) -> f32 {
    1.0 / (1.2 * 2f32.powf(ev100))
}

/// The sky's radiance (cd/m²) as a uniform hemisphere: the illuminance over
/// pi, tinted and normalised to unit luminance in `space`. The scene's clear
/// colour, the distance fog and the water's reflection use it.
pub fn sky_radiance(space: crate::colorspace::WorkingSpace) -> [f32; 3] {
    let t = sky_tint(space);
    let k = SKY_ILLUMINANCE / std::f32::consts::PI;
    [t[0] * k, t[1] * k, t[2] * k]
}

/// The illuminance below which a light is not worth listing: one 8-bit code
/// of the exposed image before the tone curve.
pub fn cull_cutoff(exposure: f32) -> f32 {
    MIN_VISIBLE_ILLUMINANCE / exposure.max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tints_keep_their_luminance() {
        for space in [crate::colorspace::WorkingSpace::Bt709, crate::colorspace::WorkingSpace::Bt2020] {
            assert!((space.gamut().luminance(sun_tint(space)) - 1.0).abs() < 1e-5);
            assert!((space.gamut().luminance(sky_tint(space)) - 1.0).abs() < 1e-5);
            // The same physical colour: white and luminance kept.
            assert!((space.gamut().luminance(sky_radiance(space)) - super::SKY_ILLUMINANCE / std::f32::consts::PI).abs() < 1e-2);
        }
    }

    #[test]
    fn ev14_is_the_reference() {
        let e = exposure(DEFAULT_EV100);
        assert!((e - 1.0 / (1.2 * 16384.0)).abs() < 1e-12, "{e}");
        assert!((e * 1e5 / std::f32::consts::PI - 1.62).abs() < 0.01, "sunlit");
    }

    #[test]
    fn culling_follows_the_exposure() {
        // 1000 cd at EV14: sqrt(1000 / 39.3) metres.
        let cut = cull_cutoff(exposure(DEFAULT_EV100));
        assert!((cut - 39.3).abs() < 0.2, "{cut}");
        assert!((1000.0f32 / cut).sqrt() > 5.0 && (1000.0 / cut).sqrt() < 5.1);
        // Twice the exposure halves the illuminance floor.
        assert!((cull_cutoff(2.0 * exposure(DEFAULT_EV100)) - cut * 0.5).abs() < 0.2);
    }

    #[test]
    fn the_meter_targets_the_key() {
        let key = KEY_LUMINANCE.log2();
        // A scene stored at the key needs no change.
        assert!((target_ev100(14.0, key, KEY_LUMINANCE) - 14.0).abs() < 1e-6);
        // Four times the stored light needs two EV less exposure.
        assert!((target_ev100(14.0, key + 2.0, KEY_LUMINANCE) - 16.0).abs() < 1e-5);
        // The stored scene's own exposure cancels: the same physical scene
        // stored one EV down meters one EV lower and targets the same EV.
        assert!((target_ev100(15.0, key + 2.0 - 1.0, KEY_LUMINANCE) - 16.0).abs() < 1e-5);
    }

    #[test]
    fn adaptation_eases_at_its_two_rates_and_clamps() {
        let e = |x: f32| 1.0 - (-x).exp();
        // Brightening (the target EV rises): the gap closes at 1.5/s.
        assert!((adapt_ev100(14.0, 18.0, 0.5) - (14.0 + 4.0 * e(0.75))).abs() < 1e-5);
        // Darkening: at 0.5/s.
        assert!((adapt_ev100(14.0, 10.0, 1.0) - (14.0 - 4.0 * e(0.5))).abs() < 1e-5);
        // Never past the target.
        assert!(adapt_ev100(14.0, 14.1, 100.0) <= 14.1 + 1e-6);
        // Frame-to-frame noise is filtered: a target flickering one bin
        // (1/8 EV) either way at 60 Hz settles (the two rates bias it a
        // little toward the brighter side) and ripples by under 0.01 EV a
        // frame, where a capped 2 EV/s step followed it by 0.03.
        let mut ev = 14.0;
        let mut ripple = 0.0f32;
        for frame in 0..600 {
            let next = adapt_ev100(ev, if frame % 2 == 0 { 14.125 } else { 13.875 }, 1.0 / 60.0);
            if frame >= 300 {
                ripple = ripple.max((next - ev).abs());
            }
            ev = next;
        }
        assert!(ripple < 0.01, "ripple {ripple}");
        assert!((ev - 14.0).abs() < 0.125, "settled at {ev}");
        // The bounds hold.
        assert_eq!(adapt_ev100(14.0, 0.0, 100.0), MIN_EV100);
        assert_eq!(adapt_ev100(14.0, 40.0, 100.0), MAX_EV100);
    }
}
