//! The full tier's detail textures (M16 delivery 4), generated once per
//! process as the prototype generates them at startup
//! (`~/RedAlert-graphics/src/scene.cpp`: `makeNormalMap`,
//! `makeAnimatedNormalMap`, `makeFoamTexture`):
//!
//! * the static map: a tileable value-noise fBm, its normal in rgb and its
//!   height in alpha (RA3's `StaticDisplacementTexture`). The wave blur adds
//!   two scrolling copies of the height to the displacement; the surface
//!   reads its normals for the fine ripples and the churned water's facets;
//! * the wind ripples: a loop of [`RIPPLE_FRAMES`] frames, [`RIPPLE_LOOP`]
//!   seconds long, of travelling waves with whole cycles per tile and per
//!   loop (after 0 A.D.'s baked water normals), so the ripples change shape
//!   as they run instead of one pattern sliding along;
//! * the foam: bubble walls of three sizes (a periodic Worley noise's
//!   F2 - F1), dense in clumps and thinning to lace between them.
//!
//! The base levels are made on the CPU (in parallel) and kept for the
//! process, so a rebuilt graph does not make them again; their mip chains
//! are made on the GPU by [`upload::image_mipped`], averaged (not
//! renormalised: the shaders expect coarser normal mips to be flatter).

use crate::bindless::Bindless;
use crate::device::GpuDevice;
use crate::memory::OwnedImage;
use crate::upload::{self, MipFilter};
use anyhow::{Context, Result};
use ash::vk;
use rayon::prelude::*;
use std::sync::OnceLock;

/// The static map's side.
pub const STATIC_SIZE: u32 = 512;
/// The ripple loop's side and frames, and its length in seconds
/// (`WATER_ANIM_LOOP` in `water.frag`).
pub const RIPPLE_SIZE: u32 = 256;
pub const RIPPLE_FRAMES: u32 = 64;
pub const RIPPLE_LOOP: f32 = 12.0;
/// The foam texture's side.
pub const FOAM_SIZE: u32 = 512;
const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// Metres per ripple tile: one over the surface's octave scale.
const RIPPLE_TILE: f32 = 1.0 / 0.021;

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn unit(x: u32) -> f32 {
    (hash(x) & 0x00ff_ffff) as f32 / 16_777_216.0
}

fn periodic_hash(x: i32, y: i32, period: i32) -> f32 {
    let (x, y) = (x.rem_euclid(period) as u32, y.rem_euclid(period) as u32);
    unit(x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263) ^ (period as u32).wrapping_mul(1_442_695_041))
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn periodic_noise(u: f32, v: f32, period: i32) -> f32 {
    let (x, y) = (u * period as f32, v * period as f32);
    let (ix, iy) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - ix as f32, y - iy as f32);
    let (ux, uy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let a = periodic_hash(ix, iy, period);
    let b = periodic_hash(ix + 1, iy, period);
    let c = periodic_hash(ix, iy + 1, period);
    let d = periodic_hash(ix + 1, iy + 1, period);
    let low = a + (b - a) * ux;
    low + ((c + (d - c) * ux) - low) * uy
}

/// The tileable fBm both the static map and the foam clumps use (0..~1).
fn periodic_fbm(u: f32, v: f32) -> f32 {
    let (u, v) = (u - u.floor(), v - v.floor());
    let (mut sum, mut amp, mut period) = (0.0, 0.55, 5);
    for _ in 0..5 {
        sum += amp * periodic_noise(u, v, period);
        period *= 2;
        amp *= 0.55;
    }
    sum
}

fn encode(n: [f32; 3]) -> [u8; 3] {
    n.map(|c| ((c * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8)
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    v.map(|c| c / l)
}

/// The static map's texels and the rms of its slopes (the ripples are
/// scaled to it, so the water keeps the prototype's look).
fn static_map(size: u32) -> (Vec<u8>, f32) {
    let n = size as usize;
    let e = 1.0 / size as f32;
    let heights: Vec<f32> = (0..n * n)
        .into_par_iter()
        .map(|i| periodic_fbm(((i % n) as f32 + 0.5) * e, ((i / n) as f32 + 0.5) * e))
        .collect();
    let (lo, hi) = heights
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h)));
    let texels: Vec<([u8; 4], f32)> = (0..n * n)
        .into_par_iter()
        .map(|i| {
            let (u, v) = (((i % n) as f32 + 0.5) * e, ((i / n) as f32 + 0.5) * e);
            let hx = (periodic_fbm(u + e, v) - periodic_fbm(u - e, v)) / (2.0 * e);
            let hy = (periodic_fbm(u, v + e) - periodic_fbm(u, v - e)) / (2.0 * e);
            let normal = normalize([-hx * 0.035, -hy * 0.035, 1.0]);
            let [r, g, b] = encode(normal);
            let a = (((heights[i] - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0) * 255.0).round() as u8;
            let slope = (normal[0] / normal[2]).powi(2) + (normal[1] / normal[2]).powi(2);
            ([r, g, b, a], slope)
        })
        .collect();
    let rms = (texels.iter().map(|t| t.1 as f64).sum::<f64>() / texels.len() as f64).sqrt() as f32;
    (texels.into_iter().flat_map(|t| t.0).collect(), rms)
}

struct Ripple {
    kx: f32,
    ky: f32,
    amp: f32,
    cycles: f32,
    phase: f32,
}

/// The ripple loop: `frames` layers of `size`² texels, slopes scaled to
/// `target_rms`.
fn ripple_frames(size: u32, frames: u32, target_rms: f32) -> Vec<Vec<u8>> {
    const SPEED: f32 = 0.3; // of the deep-water speed: RTS-scale ripples
    let mut waves: Vec<Ripple> = Vec::new();
    let mut i = 0u32;
    while waves.len() < 64 && i < 4096 {
        let r = |k: u32| unit(i.wrapping_mul(4).wrapping_add(k));
        let (r0, r1, r2, r3) = (r(1), r(2), r(3), r(4));
        i += 1;
        let m = 4.0 * 16f32.powf(r0); // 4..64 cycles per tile, log-uniform
        let mut a = (r1 - 0.5) * 2.2; // around the wind
        if r3 < 0.12 {
            a += std::f32::consts::PI; // a few against it
        }
        let (kx, ky) = ((m * a.cos()).round(), (m * a.sin()).round());
        if (kx == 0.0 && ky == 0.0) || waves.iter().any(|w| w.kx == kx && w.ky == ky) {
            continue;
        }
        let k = (kx * kx + ky * ky).sqrt();
        let omega = SPEED * (9.81 * std::f32::consts::TAU * k / RIPPLE_TILE).sqrt();
        let cycles = (omega * RIPPLE_LOOP / std::f32::consts::TAU).round().max(1.0);
        waves.push(Ripple {
            kx,
            ky,
            amp: 1.0 / k,
            cycles,
            phase: r2 * std::f32::consts::TAU,
        });
    }
    let n = size as usize;
    let slopes: Vec<Vec<[f32; 2]>> = (0..frames)
        .into_par_iter()
        .map(|f| {
            let mut out = vec![[0.0f32; 2]; n * n];
            for w in &waves {
                let wt = std::f32::consts::TAU * w.cycles * f as f32 / frames as f32;
                let (dx, dy) = (
                    -w.amp * std::f32::consts::TAU * w.kx,
                    -w.amp * std::f32::consts::TAU * w.ky,
                );
                for (t, slope) in out.iter_mut().enumerate() {
                    let (x, y) = ((t % n) as f32, (t / n) as f32);
                    let s = std::f32::consts::TAU * (w.kx * x + w.ky * y) / size as f32 + w.phase - wt;
                    let v = s.sin();
                    slope[0] += dx * v;
                    slope[1] += dy * v;
                }
            }
            out
        })
        .collect();
    let sum: f64 = slopes
        .iter()
        .flatten()
        .map(|s| (s[0] * s[0] + s[1] * s[1]) as f64)
        .sum();
    let scale = target_rms / ((sum / (n * n * frames as usize) as f64).sqrt() as f32).max(1e-6);
    slopes
        .into_par_iter()
        .map(|frame| {
            frame
                .iter()
                .flat_map(|s| {
                    let [r, g, b] = encode(normalize([-s[0] * scale, -s[1] * scale, 1.0]));
                    [r, g, b, 255]
                })
                .collect()
        })
        .collect()
}

/// The foam texture: density in every channel.
fn foam(size: u32) -> Vec<u8> {
    let walls = |u: f32, v: f32, period: i32, seed: i32| {
        let (x, y) = (u * period as f32, v * period as f32);
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let (mut f1, mut f2) = (f32::MAX, f32::MAX);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy) = (ix + dx, iy + dy);
                let px = cx as f32 + periodic_hash(cx + seed, cy, period);
                let py = cy as f32 + periodic_hash(cx, cy + seed * 3, period);
                let d = ((px - x).powi(2) + (py - y).powi(2)).sqrt();
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
        1.0 - smooth(0.0, 0.18, f2 - f1) // 1 on a wall, 0 inside a bubble
    };
    let n = size as usize;
    (0..n * n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let (u, v) = (
                ((i % n) as f32 + 0.5) / size as f32,
                ((i / n) as f32 + 0.5) / size as f32,
            );
            let clump = smooth(0.35, 0.75, periodic_fbm(u, v));
            let w = walls(u, v, 48, 11) * 0.5 + walls(u, v, 22, 23) * 0.3 + walls(u, v, 9, 37) * 0.2;
            // Dense foam fills in between the walls; thin foam is only the walls.
            let (thin, dense) = (w * 0.8, 0.55 + w * 0.45);
            let f = ((thin + (dense - thin) * clump) * (0.35 + 0.65 * clump)).clamp(0.0, 1.0);
            let b = (f * 255.0).round() as u8;
            [b, b, b, 255]
        })
        .collect()
}

struct Texels {
    static_map: Vec<u8>,
    ripples: Vec<Vec<u8>>,
    foam: Vec<u8>,
}

fn texels() -> &'static Texels {
    static TEXELS: OnceLock<Texels> = OnceLock::new();
    TEXELS.get_or_init(|| {
        let (static_map, rms) = static_map(STATIC_SIZE);
        Texels {
            ripples: ripple_frames(RIPPLE_SIZE, RIPPLE_FRAMES, rms),
            static_map,
            foam: foam(FOAM_SIZE),
        }
    })
}

/// The three textures and their global sampled-image indices.
pub(crate) struct DetailTextures {
    images: [OwnedImage; 3],
    /// Sampled indices: the static map, the ripple loop (an array), the foam.
    pub static_map: u32,
    pub ripples: u32,
    pub foam: u32,
}

impl DetailTextures {
    pub fn new(gpu: &GpuDevice, bindless: &mut Bindless) -> Result<Self> {
        let t = texels();
        let image = |size: u32, layers: &[&[u8]], view: vk::ImageViewType, name: &str| {
            upload::image_mipped(gpu, FORMAT, size, size, layers, MipFilter::Average, view, name)
        };
        let mut made: Vec<OwnedImage> = Vec::with_capacity(3);
        let result = (|| -> Result<()> {
            made.push(image(
                STATIC_SIZE,
                &[&t.static_map],
                vk::ImageViewType::TYPE_2D,
                "water static map",
            )?);
            let ripples: Vec<&[u8]> = t.ripples.iter().map(Vec::as_slice).collect();
            made.push(image(
                RIPPLE_SIZE,
                &ripples,
                vk::ImageViewType::TYPE_2D_ARRAY,
                "water ripples",
            )?);
            made.push(image(FOAM_SIZE, &[&t.foam], vk::ImageViewType::TYPE_2D, "water foam")?);
            Ok(())
        })();
        let indices = result.and_then(|()| {
            let layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            Ok([
                bindless.sampled_2d(made[0].view, layout)?.0.index,
                bindless.sampled_2d_array(made[1].view, layout)?.0.index,
                bindless.sampled_2d(made[2].view, layout)?.0.index,
            ])
        });
        match indices {
            Ok([static_map, ripples, foam]) => {
                let images: [OwnedImage; 3] = made.try_into().ok().context("three detail textures")?;
                Ok(Self {
                    images,
                    static_map,
                    ripples,
                    foam,
                })
            }
            Err(error) => {
                for image in &mut made {
                    image.destroy(gpu);
                }
                Err(error)
            }
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        for image in &mut self.images {
            image.destroy(gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_detail_textures_tile() {
        // The fBm and the foam's cells are periodic in the tile: the first
        // and the last texel columns differ as neighbours do, not as a seam.
        let size = 64;
        let (map, _) = static_map(size);
        let alpha = |x: usize, y: usize| map[(y * size as usize + x) * 4 + 3] as i32;
        let n = size as usize;
        let seam: i32 = (0..n).map(|y| (alpha(0, y) - alpha(n - 1, y)).abs()).max().unwrap();
        let inner: i32 = (0..n).map(|y| (alpha(1, y) - alpha(0, y)).abs()).max().unwrap();
        assert!(seam <= inner * 2 + 2, "seam {seam} against neighbours {inner}");
    }

    #[test]
    fn the_ripples_loop_and_match_the_static_slopes() {
        let frames = ripple_frames(32, 8, 0.1);
        assert_eq!(frames.len(), 8);
        // Frame 8 would be frame 0 again: every wave has whole cycles per
        // loop, so frames 7 and 0 are neighbours.
        let decode = |b: u8| b as f32 / 255.0 * 2.0 - 1.0;
        let rms = |f: &[u8]| {
            let s: f32 = f
                .chunks(4)
                .map(|t| {
                    let (x, y, z) = (decode(t[0]), decode(t[1]), decode(t[2]));
                    (x / z).powi(2) + (y / z).powi(2)
                })
                .sum();
            (s / (f.len() / 4) as f32).sqrt()
        };
        for f in &frames {
            assert!((rms(f) - 0.1).abs() < 0.03, "frame rms {}", rms(f));
        }
    }
}
