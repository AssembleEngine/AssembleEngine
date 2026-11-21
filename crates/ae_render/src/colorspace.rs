//! Colour spaces and gamut mapping (M13 stage 3).
//!
//! The scene is lit in the *working space*, [`WorkingSpace`]: BT.709 (the
//! default, sRGB's primaries) or BT.2020. White is D65 in both, so a
//! conversion is a linear 3x3 matrix that keeps luminance and white. Colours
//! the engine authors (the sun and sky tints, light, particle and team
//! colours, the fog and the sky) are BT.709; when the working space is wider
//! they are converted on the host ([`WorkingSpace::from_709`]) and the
//! shaders convert the textures and literals they sample, so every colour
//! enters the lighting in the working space.
//!
//! The output gamut ([`Gamut`]) is separate from the range: the display's
//! primaries. `post.encode` maps the display intermediate from the working
//! space to the output gamut and, when the output gamut is narrower than the
//! working space (BT.2020 content on a P3 or BT.709 display), compresses the
//! colour toward the achromatic axis with [`compress`]: luminance and hue are
//! kept and in-gamut colours are untouched, so out-of-gamut light loses
//! chroma instead of clipping a channel. SDR output has no encode pass and
//! `post.tonemap` does the same mapping to BT.709 itself; in HDR the tonemap
//! leaves the display intermediate in the working space, so the map happens
//! once, in `post.encode`.
//!
//! P3 is Display P3: DCI-P3's primaries with D65 white (its transfer
//! function is sRGB's, which plays no part here: only the primaries matter,
//! [`Gamut::primaries`]).

use std::str::FromStr;

use anyhow::bail;

/// A set of RGB primaries with D65 white: BT.709/sRGB, Display P3 or
/// BT.2020.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Gamut {
    /// BT.709 / sRGB: the default working space and the SDR output's gamut.
    #[default]
    Bt709,
    /// Display P3: the default output gamut for HDR10 (M13 stage 3).
    P3,
    /// BT.2020: HDR10's container primaries and the wide working space.
    Bt2020,
}

/// The space the scene is lit in (M13 stage 3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WorkingSpace {
    /// BT.709 / sRGB primaries: the default, and what all captures used
    /// before stage 3.
    #[default]
    Bt709,
    /// BT.2020 primaries, D65 white.
    Bt2020,
}

impl FromStr for Gamut {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bt709" | "srgb" => Ok(Self::Bt709),
            "p3" | "display-p3" => Ok(Self::P3),
            "bt2020" | "rec2020" => Ok(Self::Bt2020),
            other => bail!("{other:?}: expected bt709, p3 or bt2020"),
        }
    }
}

impl FromStr for WorkingSpace {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bt709" | "srgb" => Ok(Self::Bt709),
            "bt2020" | "rec2020" => Ok(Self::Bt2020),
            other => bail!("{other:?}: expected bt709 or bt2020"),
        }
    }
}

/// `rgb_to_xyz` and `xyz_to_rgb` per gamut, row-major, D65.
const BT709_TO_XYZ: [[f32; 3]; 3] = [[0.412_390_8, 0.357_584_3, 0.180_480_8], [0.212_639_0, 0.715_168_7, 0.072_192_3], [0.019_330_8, 0.119_194_8, 0.950_532_1]];
const XYZ_TO_BT709: [[f32; 3]; 3] = [[3.240_969_9, -1.537_383_2, -0.498_610_8], [-0.969_243_6, 1.875_967_5, 0.041_555_1], [0.055_630_1, -0.203_977_0, 1.056_971_5]];
const BT2020_TO_XYZ: [[f32; 3]; 3] = [[0.636_958_1, 0.144_616_9, 0.168_881_0], [0.262_700_2, 0.677_998_1, 0.059_301_7], [0.0, 0.028_072_7, 1.060_985_1]];
const XYZ_TO_BT2020: [[f32; 3]; 3] = [[1.716_651_2, -0.355_670_8, -0.253_366_3], [-0.666_684_3, 1.616_481_2, 0.015_768_5], [0.017_639_9, -0.042_770_6, 0.942_103_1]];
const P3_TO_XYZ: [[f32; 3]; 3] = [[0.486_571_0, 0.265_667_7, 0.198_217_3], [0.228_974_6, 0.691_738_5, 0.079_286_9], [0.0, 0.045_113_4, 1.043_944_4]];
const XYZ_TO_P3: [[f32; 3]; 3] = [[2.493_496_9, -0.931_383_6, -0.402_710_8], [-0.829_489_0, 1.762_664_1, 0.023_624_7], [0.035_845_8, -0.076_172_4, 0.956_884_5]];

impl Gamut {
    /// The R, G and B primaries' chromaticities (`x`, `y`).
    pub fn primaries(self) -> [[f32; 2]; 3] {
        match self {
            Self::Bt709 => [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]],
            Self::P3 => [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
            Self::Bt2020 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        }
    }

    /// Linear RGB to CIE XYZ (D65), row-major.
    pub fn rgb_to_xyz(self) -> [[f32; 3]; 3] {
        match self {
            Self::Bt709 => BT709_TO_XYZ,
            Self::P3 => P3_TO_XYZ,
            Self::Bt2020 => BT2020_TO_XYZ,
        }
    }

    /// CIE XYZ (D65) to linear RGB, row-major.
    pub fn xyz_to_rgb(self) -> [[f32; 3]; 3] {
        match self {
            Self::Bt709 => XYZ_TO_BT709,
            Self::P3 => XYZ_TO_P3,
            Self::Bt2020 => XYZ_TO_BT2020,
        }
    }

    /// The luminance weights (the second row of [`Self::rgb_to_xyz`], at the
    /// conventional precision: Rec. 709's 0.2126/0.7152/0.0722 etc.).
    pub fn luminance_row(self) -> [f32; 3] {
        match self {
            Self::Bt709 => [0.2126, 0.7152, 0.0722],
            Self::P3 => [0.2290, 0.6917, 0.0793],
            Self::Bt2020 => [0.2627, 0.6780, 0.0593],
        }
    }

    /// Whether every colour in `space` is inside this gamut (BT.709 is
    /// inside P3, and both are inside BT.2020).
    pub fn contains(self, space: WorkingSpace) -> bool {
        matches!((self, space), (_, WorkingSpace::Bt709) | (Self::Bt2020, WorkingSpace::Bt2020))
    }

    /// Luminance of a linear RGB colour in this gamut.
    pub fn luminance(self, c: [f32; 3]) -> f32 {
        let w = self.luminance_row();
        w[0] * c[0] + w[1] * c[1] + w[2] * c[2]
    }
}

impl WorkingSpace {
    /// The primaries of the working space.
    pub fn gamut(self) -> Gamut {
        match self {
            Self::Bt709 => Gamut::Bt709,
            Self::Bt2020 => Gamut::Bt2020,
        }
    }

    /// The luminance weights of the scene's stored light.
    pub fn luminance_row(self) -> [f32; 3] {
        self.gamut().luminance_row()
    }

    /// Whether the space is wider than BT.709 (the shaders' working flag).
    pub fn is_wide(self) -> bool {
        self != Self::Bt709
    }

    /// A colour authored in BT.709 (every host colour before M13 stage 4)
    /// in this working space.
    pub fn from_709(self, c: [f32; 3]) -> [f32; 3] {
        convert(Gamut::Bt709, self.gamut(), c)
    }

    /// `c` scaled to unit luminance in this space (a tint whose scalar is
    /// an illuminance).
    pub fn unit_tint(self, c: [f32; 3]) -> [f32; 3] {
        let c = self.from_709(c);
        let y = self.gamut().luminance(c).max(1e-6);
        [c[0] / y, c[1] / y, c[2] / y]
    }
}

/// Multiply row-major 3x3 `a` by `b`.
fn mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (k, slot) in row.iter_mut().enumerate() {
            *slot = a[r][0] * b[0][k] + a[r][1] * b[1][k] + a[r][2] * b[2][k];
        }
    }
    out
}

/// A row-major matrix's columns (the layout a shader's `mat3` is uploaded
/// in, each padded to a `vec4` by std140).
pub fn columns(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|c| std::array::from_fn(|r| m[r][c]))
}

/// Apply a row-major 3x3 matrix to a column vector.
pub fn apply(m: [[f32; 3]; 3], c: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| m[r][0] * c[0] + m[r][1] * c[1] + m[r][2] * c[2])
}

/// The linear matrix from `from` to `to` (both D65, so white stays white).
pub fn matrix(from: Gamut, to: Gamut) -> [[f32; 3]; 3] {
    if from == to {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    mul(to.xyz_to_rgb(), from.rgb_to_xyz())
}

/// `c` from `from`'s primaries to `to`'s.
pub fn convert(from: Gamut, to: Gamut, c: [f32; 3]) -> [f32; 3] {
    if from == to { c } else { apply(matrix(from, to), c) }
}

/// Compress `c` (in `gamut`'s primaries) into the gamut by moving it toward
/// its own achromatic (grey) value, as far as the most negative channel
/// needs and no further: luminance is kept exactly, hue is kept (every
/// channel's distance from the grey scales by the same factor), and a colour
/// already inside the gamut is returned unchanged. This is not a clip: an
/// out-of-gamut primaries' light loses chroma but not brightness.
///
/// Only negative channels count as out of gamut: values above white are
/// brightness (an HDR emitter), not a chromaticity.
pub fn compress(gamut: Gamut, c: [f32; 3]) -> [f32; 3] {
    let lo = c[0].min(c[1]).min(c[2]);
    if lo >= 0.0 {
        return c;
    }
    let y = gamut.luminance(c);
    if y <= 0.0 || y.is_nan() {
        // Black, or a colour whose imaginary primaries have no light.
        return [0.0; 3];
    }
    let t = y / (y - lo);
    c.map(|v| y + t * (v - y))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Gamut; 3] = [Gamut::Bt709, Gamut::P3, Gamut::Bt2020];

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() <= tol)
    }

    #[test]
    fn white_stays_white_and_luminance_is_kept() {
        for from in ALL {
            for to in ALL {
                let white = convert(from, to, [1.0; 3]);
                assert!(close(white, [1.0; 3], 1e-5), "{from:?}->{to:?}: {white:?}");
                for c in [[0.2, 0.5, 0.1], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] {
                    let d = convert(from, to, c);
                    assert!((from.luminance(c) - to.luminance(d)).abs() < 2e-4, "{from:?}->{to:?}: {} vs {}", from.luminance(c), to.luminance(d));
                    let back = convert(to, from, d);
                    assert!(close(back, c, 1e-5), "{from:?}->{to:?}->{from:?}: {back:?} vs {c:?}");
                }
            }
        }
    }

    #[test]
    fn the_matrices_are_the_primaries_and_d65() {
        // Each column of rgb_to_xyz is a full-intensity primary: its
        // chromaticity is the gamut's primary. White has Y = 1 and D65's
        // chromaticity.
        for gamut in ALL {
            let m = gamut.rgb_to_xyz();
            for (k, [px, py]) in gamut.primaries().iter().enumerate() {
                let [x, y, z] = [m[0][k], m[1][k], m[2][k]];
                let sum = x + y + z;
                assert!((x / sum - px).abs() < 2e-4, "{gamut:?} primary {k}: {px} vs {}", x / sum);
                assert!((y / sum - py).abs() < 2e-4, "{gamut:?} primary {k}: {py} vs {}", y / sum);
            }
            let white = apply(m, [1.0; 3]);
            let sum = white[0] + white[1] + white[2];
            assert!((white[1] - 1.0).abs() < 1e-5, "{gamut:?} white Y: {white:?}");
            assert!((white[0] / sum - 0.3127).abs() < 2e-4);
            assert!((white[1] / sum - 0.3290).abs() < 2e-4);
        }
    }

    #[test]
    fn bt_709_to_bt_2020_is_the_container_matrix() {
        // The values M13 stage 2's shader used (BT.2087's matrix).
        let m = matrix(Gamut::Bt709, Gamut::Bt2020);
        assert!(close(m[0], [0.627_403_9, 0.329_283_0, 0.043_313_1], 2e-7));
        assert!(close(m[1], [0.069_097_3, 0.919_540_4, 0.011_362_3], 2e-7));
        assert!(close(m[2], [0.016_391_4, 0.088_013_3, 0.895_595_3], 2e-7));
        // The identity is exact, which keeps SDR output bit for bit.
        assert_eq!(matrix(Gamut::Bt709, Gamut::Bt709), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn p3_holds_bt_709_and_bt_2020_holds_both() {
        // 709 primaries are inside P3 (no negative 709 channel in P3), and
        // P3's are outside 709 (the blue coordinate of a P3 primary is
        // negative in 709).
        for c in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let in_p3 = convert(Gamut::Bt709, Gamut::P3, c);
            assert!(in_p3.iter().all(|v| *v >= -1e-6), "{c:?} in P3: {in_p3:?}");
        }
        let p3_green_in_709 = convert(Gamut::P3, Gamut::Bt709, [0.0, 1.0, 0.0]);
        assert!(p3_green_in_709.iter().any(|v| *v < -0.01), "P3 green is outside 709: {p3_green_in_709:?}");
        assert!(Gamut::P3.contains(WorkingSpace::Bt709));
        assert!(!Gamut::Bt709.contains(WorkingSpace::Bt2020));
        assert!(Gamut::Bt2020.contains(WorkingSpace::Bt2020));
    }

    #[test]
    fn compression_keeps_luminance_hue_and_leaves_gamut_alone() {
        // In gamut: untouched (exactly).
        for c in [[0.3, 0.5, 0.2], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]] {
            assert_eq!(compress(Gamut::P3, c), c);
        }
        // BT.2020 green through P3: outside, so compressed.
        let wide = convert(Gamut::Bt2020, Gamut::P3, [0.0, 1.0, 0.0]);
        assert!(wide.iter().any(|v| *v < -0.01), "starts out of gamut: {wide:?}");
        let squeezed = compress(Gamut::P3, wide);
        assert!(squeezed.iter().all(|v| *v >= -1e-6), "inside after: {squeezed:?}");
        assert!((squeezed.iter().fold(f32::INFINITY, |a, v| a.min(*v))).abs() < 1e-5, "one channel on the boundary: {squeezed:?}");
        // Luminance exactly, hue as the same scale on every channel's
        // distance from the grey.
        let (y, y2) = (Gamut::P3.luminance(wide), Gamut::P3.luminance(squeezed));
        assert!((y - y2).abs() < 1e-6, "{y} vs {y2}");
        let d = |c: [f32; 3], k: usize| c[k] - Gamut::P3.luminance(c);
        let t = d(squeezed, 0) / d(wide, 0);
        assert!(t > 0.0 && t < 1.0, "t {t}");
        for k in 0..3 {
            assert!((d(squeezed, k) - t * d(wide, k)).abs() < 1e-5, "channel {k} hue");
        }
        // Imaginary (all-negative) light has no chromaticity to keep.
        assert_eq!(compress(Gamut::P3, [-0.1, -0.1, -0.1]), [0.0; 3]);
    }

    #[test]
    fn the_working_space_converts_authored_colours() {
        // BT.709 is the identity; BT.2020 is the container matrix, white and
        // luminance kept.
        let c = [0.8, 0.4, 0.2];
        assert_eq!(WorkingSpace::Bt709.from_709(c), c);
        let wide = WorkingSpace::Bt2020.from_709(c);
        assert!(close(wide, convert(Gamut::Bt709, Gamut::Bt2020, c), 0.0));
        assert!((WorkingSpace::Bt2020.gamut().luminance(wide) - Gamut::Bt709.luminance(c)).abs() < 2e-4);
        // A tint normalised in either space has unit luminance there.
        for space in [WorkingSpace::Bt709, WorkingSpace::Bt2020] {
            let tint = space.unit_tint([1.0, 0.95, 0.85]);
            assert!((space.gamut().luminance(tint) - 1.0).abs() < 1e-5, "{space:?}: {tint:?}");
        }
    }
}
