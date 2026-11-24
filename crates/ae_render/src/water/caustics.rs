//! The caustic web (M16.5): one tileable, mipmapped texture made when the
//! water renderer is built. The refract tier's underwater pass projects it
//! along the sun (stage 3); the M7 terrain draws its own analytic caustics
//! and ignores this, so nothing changes until a graph samples it.
//!
//! The pattern is procedural and deterministic, with integer wave periods so
//! it tiles exactly at the texture's edges. Its mip chain is made on the GPU
//! by the engine's one mip path, [`upload::image_mipped`].

use crate::bindless::Bindless;
use crate::memory::OwnedImage;
use crate::upload::{self, MipFilter};
use anyhow::Result;
use ash::vk;

/// The texture's side (the prototype bakes 256²).
pub const SIZE: u32 = 256;
/// What the shaders sample: the web's intensity in r (and g, b), a 255.
pub const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// Feature cells across one tile of the web.
const CELLS: i32 = 7;

fn cell_hash(x: i32, y: i32, salt: u32) -> f32 {
    let mut h = (x.rem_euclid(CELLS) as u32).wrapping_mul(0x8da6_b343)
        ^ (y.rem_euclid(CELLS) as u32).wrapping_mul(0xd816_3841)
        ^ salt.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 16;
    (h & 0xffff) as f32 / 65535.0
}

/// The web at `(u, v)` (0..1 across one tile), 0..1: bright threads along
/// the edges of jittered cells (where neighbouring facets of the surface
/// focus the sun onto the same line), the cells' outlines bent by a warp so
/// no edge is straight. Light gathers into thin threads with dim cells
/// between. The cell lattice and the warp repeat over the tile, so `u = 0`
/// and `u = 1` meet.
///
/// An earlier web of crossed integer sines read as a woven lattice, from
/// close up and, beating between its two drifting copies, from afar.
pub fn pattern(u: f32, v: f32) -> f32 {
    use std::f32::consts::TAU;
    // Wrap first, so the seams meet exactly (not just up to f32 sine error).
    let (u, v) = (u - u.floor(), v - v.floor());
    let wu = u + 0.035 * (TAU * (2.0 * v + u)).sin() + 0.02 * (TAU * (3.0 * v - 2.0 * u) + 1.3).sin();
    let wv = v + 0.035 * (TAU * (2.0 * u - v) + 0.7).sin() + 0.02 * (TAU * (3.0 * u + 2.0 * v) + 2.1).sin();
    let (px, py) = (wu * CELLS as f32, wv * CELLS as f32);
    let (cx, cy) = (px.floor() as i32, py.floor() as i32);
    let (mut f1, mut f2) = (f32::MAX, f32::MAX);
    for dy in -2..=2 {
        for dx in -2..=2 {
            let (x, y) = (cx + dx, cy + dy);
            let fx = x as f32 + 0.1 + 0.8 * cell_hash(x, y, 1) - px;
            let fy = y as f32 + 0.1 + 0.8 * cell_hash(x, y, 2) - py;
            let d = (fx * fx + fy * fy).sqrt();
            if d < f1 {
                f2 = f1;
                f1 = d;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    // The edge distance; the threads thin out away from it.
    let edge = (f2 - f1).max(0.0);
    let thread = (1.0 - edge / 0.22).clamp(0.0, 1.0);
    (thread * thread * (3.0 - 2.0 * thread)).powf(1.6)
}

/// The base level's RGBA texels, row-major.
pub fn base(size: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let value = pattern(x as f32 / size as f32, y as f32 / size as f32);
            let c = (value * 255.0).round().clamp(0.0, 255.0) as u8;
            out.extend_from_slice(&[c, c, c, 255]);
        }
    }
    out
}

/// The texture and its bindless sampled handle.
pub struct Caustics {
    image: OwnedImage,
    /// The global sampled-image index (the handle table's slot is the
    /// [`super::WaterRenderer`]'s, not this module's).
    pub sampled: u32,
    /// The library binding's identity (unique per allocation).
    identity: u64,
}

impl Caustics {
    /// Make the texture: the deterministic base, its GPU mip chain, and its
    /// sampled handle (registered in the caller's group).
    pub fn new(gpu: &crate::device::GpuDevice, bindless: &mut Bindless, identity: u64) -> Result<Self> {
        let data = base(SIZE);
        let image = upload::image_mipped(
            gpu,
            FORMAT,
            SIZE,
            SIZE,
            &[&data],
            MipFilter::Average,
            vk::ImageViewType::TYPE_2D,
            "water caustics",
        )?;
        let sampled = match bindless.sampled_2d(image.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) {
            Ok(handle) => handle.0.index,
            Err(error) => {
                let mut image = image;
                image.destroy(gpu);
                return Err(error);
            }
        };
        Ok(Self {
            image,
            sampled,
            identity,
        })
    }

    pub fn desc(&self) -> ae_framegraph::ImageDesc {
        self.image.desc
    }

    pub fn view(&self) -> vk::ImageView {
        self.image.view
    }

    /// The texture as the graph binds it.
    pub fn binding(&self) -> crate::executor::ImageBinding {
        crate::executor::ImageBinding {
            initial: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image: self.image.image,
            view: self.image.view,
            format: self.image.desc.format,
            extent: vk::Extent2D {
                width: self.image.desc.width,
                height: self.image.desc.height,
            },
            range: self.image.range(),
            identity: self.identity,
            sampled: Some(self.sampled),
            storage: None,
        }
    }

    /// Every mip's texels (tests): level 0's 32-bit texels first.
    pub fn read(&self, gpu: &crate::device::GpuDevice) -> Result<Vec<Vec<u8>>> {
        upload::read_image(gpu, &self.image, 4)
    }

    pub fn destroy(&mut self, gpu: &crate::device::GpuDevice) {
        self.image.destroy(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pattern tiles: the web at u 0 meets u 1, and the same across v.
    #[test]
    fn the_web_tiles_and_is_not_flat() {
        for v in [0.0, 0.1, 0.37, 0.5] {
            assert!((pattern(0.0, v) - pattern(1.0, v)).abs() < 1e-6, "u seam at v {v}");
        }
        for u in [0.0, 0.2, 0.63, 0.9] {
            assert!((pattern(u, 0.0) - pattern(u, 1.0)).abs() < 1e-6, "v seam at u {u}");
        }
        let values: Vec<f32> = (0..64).map(|k| pattern(k as f32 / 64.0, 0.3)).collect();
        let (lo, hi) = values
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(hi - lo > 0.1, "the web has contrast: {lo}..{hi}");
        let b = base(SIZE);
        assert_eq!(b.len(), (SIZE * SIZE * 4) as usize);
        assert!(b.chunks(4).any(|t| t[0] > 128) && b.chunks(4).any(|t| t[0] < 32));
    }
}
