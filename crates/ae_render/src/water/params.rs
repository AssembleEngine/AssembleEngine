//! The render side of a map's water look (M16.4): the values the rich tiers
//! read instead of the shaders' constants. Plain data, so the renderer takes
//! no dependency on the map package; the client converts `ae_map::WaterLook`
//! into this field by field.
//!
//! [`WaterParams::default`] is the M7 frame's look: each value is the
//! constant it replaces (the table test below names every source).

/// A map's water settings, in the shaders' units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterParams {
    /// Absorption per channel, inverse metres.
    pub absorption: [f32; 3],
    /// The water body's in-scattered radiance, shallow and deep (cd/m²).
    pub shallow: [f32; 3],
    pub deep: [f32; 3],
    /// The wind's direction in the ground plane (x, z; unit length) and its
    /// speed in m/s.
    pub wind: [f32; 2],
    pub wind_speed: f32,
    /// The dominant swell: amplitude and wavelength, metres.
    pub swell_amplitude: f32,
    pub swell_length: f32,
    /// Shore and impact foam, 0..1.
    pub foam: f32,
    /// The caustic web's scale, inverse metres.
    pub caustics: f32,
    /// Submerged units: tinted as if seen through `sub_tint` times the water
    /// above them, at most `sub_tint_max` metres, keeping `sub_keep` of
    /// their brightness; `outline` adds the soft hull outline aid.
    pub sub_tint: f32,
    pub sub_keep: f32,
    pub sub_tint_max: f32,
    pub outline: bool,
}

impl Default for WaterParams {
    /// The values the M7 shaders and renderer use today.
    fn default() -> Self {
        Self {
            // terrain.frag's tint uses 0.08 per metre on every channel.
            absorption: [0.08; 3],
            // terrain.frag's in-scattered radiances.
            shallow: [3146.0, 7471.0, 8258.0],
            deep: [786.0, 3146.0, 4719.0],
            // water.frag's first wave train: direction 0.55 rad, 48 m,
            // steepness 0.05; and its deep-water phase speed.
            wind: [0.55f32.cos(), 0.55f32.sin()],
            wind_speed: (9.81 * std::f32::consts::TAU / 48.0).sqrt(),
            swell_amplitude: 0.05 * 48.0 / std::f32::consts::TAU,
            swell_length: 48.0,
            // water.frag's foam clamp.
            foam: 0.9,
            // terrain.frag's caustic noise scale.
            caustics: 0.12,
            // The prototype's submerged-unit limits (its shaders/common.glsl);
            // M7 has no submerged units, so these are the M16 values.
            sub_tint: 4.0,
            sub_keep: 0.6,
            sub_tint_max: 24.0,
            outline: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults are the M7 constants, as a table: each row names the
    /// source the value came from, and the source still holds it (so a
    /// shader edit and a default cannot drift apart silently).
    #[test]
    fn defaults_are_the_m7_constants() {
        let water = include_str!("../../shaders/water.frag");
        let terrain = include_str!("../../shaders/terrain.frag");
        let renderer = include_str!("../terrain/renderer.rs");
        let p = WaterParams::default();
        let tau = std::f32::consts::TAU;

        // The shader constants the table below is copied from.
        assert!(terrain.contains("vec3 deep = aeToWorking(vec3(786.0, 3146.0, 4719.0)"));
        assert!(terrain.contains("vec3 shallow = aeToWorking(vec3(3146.0, 7471.0, 8258.0)"));
        assert!(terrain.contains("vec2 q = inWorld.xz * 0.12;"));
        assert!(water.contains("float[](48.0, 27.0, 15.0"));
        assert!(water.contains("float[](0.55, 1.15"));
        assert!(water.contains("float[](0.05, 0.06"));
        assert!(water.contains("0.0, 0.9);"), "the foam clamp");
        assert!(renderer.contains("Some(level) => [level, 1.0, 0.08, p.time]"));

        // The table.
        assert_eq!(p.absorption, [0.08; 3]);
        assert_eq!(p.deep, [786.0, 3146.0, 4719.0]);
        assert_eq!(p.shallow, [3146.0, 7471.0, 8258.0]);
        assert!((p.wind[0] - 0.55f32.cos()).abs() < 1e-6);
        assert!((p.wind[1] - 0.55f32.sin()).abs() < 1e-6);
        assert!((p.wind[0] * p.wind[0] + p.wind[1] * p.wind[1] - 1.0).abs() < 1e-6);
        assert!((p.wind_speed - (9.81 * tau / 48.0).sqrt()).abs() < 1e-6);
        assert!((p.swell_amplitude - 0.05 * 48.0 / tau).abs() < 1e-6);
        assert_eq!(p.swell_length, 48.0);
        assert_eq!(p.foam, 0.9);
        assert_eq!(p.caustics, 0.12);
        assert_eq!((p.sub_tint, p.sub_keep, p.sub_tint_max), (4.0, 0.6, 24.0));
        assert!(!p.outline);
    }
}
