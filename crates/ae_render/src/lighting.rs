//! Photometric lighting (M12): units, the scene's sun and sky, and exposure.
//!
//! The renderer works in physical units:
//! * one world unit is one metre ([`WORLD_UNIT_METRES`]);
//! * a point or spot light carries an intensity in candela (cd = lm/sr);
//! * the sun is a directional illuminance in lux (lm/m²);
//! * surface emission and luminance are cd/m², as linear RGB;
//! * a Lambertian surface's outgoing radiance is `albedo/pi * E` for the
//!   illuminance `E` it receives;
//! * the camera exposes with `1 / (1.2 * 2^EV100)` before the tone curve.
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

pub fn sun_tint() -> [f32; 3] {
    unit_tint(SUN_TINT)
}

pub fn sky_tint() -> [f32; 3] {
    unit_tint(SKY_TINT)
}

/// The exposure factor of an EV100 setting: `1 / (1.2 * 2^EV100)`.
pub fn exposure(ev100: f32) -> f32 {
    1.0 / (1.2 * 2f32.powf(ev100))
}

/// The reference exposure of the migration bridge (EV14).
pub fn reference_exposure() -> f32 {
    exposure(DEFAULT_EV100)
}

/// The sky's radiance (cd/m²) as a uniform hemisphere: the illuminance over
/// pi, tinted and normalised to unit luminance. The scene's clear colour,
/// the distance fog and the water's reflection use it.
pub fn sky_radiance() -> [f32; 3] {
    let t = sky_tint();
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
        assert!((luminance(sun_tint()) - 1.0).abs() < 1e-5);
        assert!((luminance(sky_tint()) - 1.0).abs() < 1e-5);
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
}
