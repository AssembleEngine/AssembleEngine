//! Heightmap I/O (16-bit binary PGM), test-terrain generation and the error pyramid
//! used by the GPU LOD criterion.

use glam::Vec2;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

/// Corner heightmap: `width x height` samples (both odd: 2N+1 x 2M+1), row-major,
/// normalised to [0, 1].
pub struct Heightmap {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Heightmap {
    pub fn cells(&self) -> (usize, usize) {
        (self.width - 1, self.height - 1)
    }

    #[inline]
    pub fn at(&self, x: isize, y: isize) -> f32 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.data[y * self.width + x]
    }

    /// Bilinear sample at a (fractional) sample coordinate.
    pub fn sample(&self, p: Vec2) -> f32 {
        let f = p.floor();
        let t = p - f;
        let (x, y) = (f.x as isize, f.y as isize);
        let a = self.at(x, y) + (self.at(x + 1, y) - self.at(x, y)) * t.x;
        let b = self.at(x, y + 1) + (self.at(x + 1, y + 1) - self.at(x, y + 1)) * t.x;
        a + (b - a) * t.y
    }

    /// Loads a binary PGM (P5, 8 or 16 bit) and normalises it with its own min/max
    /// to the [0, 1] range.
    pub fn load_pgm(path: &Path) -> Result<Self, String> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let mut pos = 0;
        let mut token = || -> Result<String, String> {
            loop {
                while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if pos < bytes.len() && bytes[pos] == b'#' {
                    while pos < bytes.len() && bytes[pos] != b'\n' {
                        pos += 1;
                    }
                    continue;
                }
                break;
            }
            let start = pos;
            while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            Ok(String::from_utf8_lossy(&bytes[start..pos]).into_owned())
        };
        if token()? != "P5" {
            return Err("only binary PGM (P5) is supported".into());
        }
        let num = |s: String| s.parse::<usize>().map_err(|e| format!("bad PGM header: {e}"));
        let width = num(token()?)?;
        let height = num(token()?)?;
        let maxval = num(token()?)?;
        let body = &bytes[pos + 1..];
        let bpp = if maxval > 255 { 2 } else { 1 };
        if body.len() < width * height * bpp {
            return Err("truncated PGM".into());
        }
        if width % 2 == 0 || height % 2 == 0 || width < 3 || height < 3 {
            return Err(format!("heightmap must be (2N+1)x(2M+1) samples, got {width}x{height}"));
        }
        let raw: Vec<f32> = (0..width * height)
            .map(|i| {
                if bpp == 2 {
                    u16::from_be_bytes([body[2 * i], body[2 * i + 1]]) as f32
                } else {
                    body[i] as f32
                }
            })
            .collect();
        let (lo, hi) = raw.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let scale = if hi > lo { 1.0 / (hi - lo) } else { 0.0 };
        let data = raw.into_iter().map(|v| (v - lo) * scale).collect();
        Ok(Heightmap { width, height, data })
    }

    /// The corner heightmap of a map of `cells` cells whose cell heights
    /// `ground` gives (each corner the mean of the cells around it,
    /// [`Self::corner`]), normalised with `scale_offset`, or with its own
    /// range when `None`; returns it with the scale and offset (world =
    /// sample × scale + offset).
    pub fn from_cells(cells: (usize, usize), ground: impl Fn(usize, usize) -> f32, scale_offset: Option<(f32, f32)>) -> (Self, f32, f32) {
        let (width, height) = (cells.0 + 1, cells.1 + 1);
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                data.push(Self::corner(cells, &ground, x, y));
            }
        }
        let (scale, offset) = scale_offset.unwrap_or_else(|| {
            let (lo, hi) = data.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            ((hi - lo).max(1e-3), lo)
        });
        for v in &mut data {
            *v = (*v - offset) / scale;
        }
        (Self { width, height, data }, scale, offset)
    }

    /// Height of corner `(x, y)` of a map of `cells` cells whose cell
    /// heights `ground` gives: the mean of the (up to four) cells around it.
    pub fn corner(cells: (usize, usize), ground: impl Fn(usize, usize) -> f32, x: usize, y: usize) -> f32 {
        let (mut sum, mut n) = (0.0, 0.0);
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            if x >= dx && y >= dy && x - dx < cells.0 && y - dy < cells.1 {
                sum += ground(x - dx, y - dy);
                n += 1.0;
            }
        }
        sum / n
    }

    pub fn save_pgm(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut w = BufWriter::new(std::fs::File::create(path)?);
        write!(w, "P5\n{} {}\n65535\n", self.width, self.height)?;
        for &v in &self.data {
            w.write_all(&((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes())?;
        }
        w.flush()
    }
}

// ---------------------------------------------------------------------------------------
// Test terrain generation
// ---------------------------------------------------------------------------------------

struct Noise {
    perm: [u8; 512],
}

impl Noise {
    fn new(seed: u64) -> Self {
        let mut p: Vec<u8> = (0..=255).collect();
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        for i in (1..256).rev() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            p.swap(i, (s % (i as u64 + 1)) as usize);
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Noise { perm }
    }

    /// 2D Perlin gradient noise, roughly in [-1, 1].
    fn perlin(&self, p: Vec2) -> f32 {
        let f = p.floor();
        let (xi, yi) = ((f.x as i32 & 255) as usize, (f.y as i32 & 255) as usize);
        let t = p - f;
        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let grad = |h: u8, x: f32, y: f32| {
            const G: [(f32, f32); 8] = [
                (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071),
            ];
            let g = G[(h & 7) as usize];
            g.0 * x + g.1 * y
        };
        let pp = &self.perm;
        let h = |x: usize, y: usize| pp[pp[x] as usize + y];
        let n00 = grad(h(xi, yi), t.x, t.y);
        let n10 = grad(h(xi + 1, yi), t.x - 1.0, t.y);
        let n01 = grad(h(xi, yi + 1), t.x, t.y - 1.0);
        let n11 = grad(h(xi + 1, yi + 1), t.x - 1.0, t.y - 1.0);
        let (u, v) = (fade(t.x), fade(t.y));
        let a = n00 + (n10 - n00) * u;
        let b = n01 + (n11 - n01) * u;
        (a + (b - a) * v) * 1.4
    }

    fn fbm(&self, p: Vec2, octaves: u32) -> f32 {
        let (mut sum, mut amp, mut q) = (0.0, 0.5, p);
        for i in 0..octaves {
            sum += amp * self.perlin(q + Vec2::splat(i as f32 * 17.13));
            q = Vec2::new(q.x * 1.6 - q.y * 1.2, q.x * 1.2 + q.y * 1.6); // rotate + scale x2
            amp *= 0.5;
        }
        sum
    }

    /// Ridged multifractal (Musgrave): sharp crests, smooth valleys, in ~[0, 1].
    fn ridged(&self, p: Vec2, octaves: u32) -> f32 {
        let (mut sum, mut amp, mut q, mut weight) = (0.0, 0.5, p, 1.0f32);
        for i in 0..octaves {
            let n = 1.0 - self.perlin(q + Vec2::splat(i as f32 * 31.7)).abs();
            let n = n * n;
            sum += n * amp * weight;
            weight = (n * 1.8).clamp(0.0, 1.0);
            q = Vec2::new(q.x * 1.62 - q.y * 1.18, q.x * 1.18 + q.y * 1.62);
            amp *= 0.5;
        }
        sum
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

struct Plateau {
    center: Vec2,
    radius: f32,
    height: f32,
}

/// Mountain ranges along all four borders with strongly varying peak heights, a mostly
/// flat central plain and a few steep-sided flat-topped plateau hills.
pub fn generate(width: usize, height: usize, seed: u64) -> Heightmap {
    assert!(width % 2 == 1 && height % 2 == 1);
    let noise = Noise::new(seed);
    let mut rng = seed ^ 0xA5A5_5A5A_1234_5678;
    let mut rand = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / (1u64 << 53) as f32
    };
    let extent = (width.max(height) - 1) as f32;
    let aspect = Vec2::new((width - 1) as f32 / extent, (height - 1) as f32 / extent);
    let plateaus: Vec<Plateau> = (0..14)
        .map(|_| Plateau {
            center: Vec2::new(0.3 + 0.4 * rand(), 0.3 + 0.4 * rand()) * aspect,
            radius: 0.015 + 0.035 * rand(),
            height: 0.04 + 0.12 * rand(),
        })
        .collect();

    let eval = |x: usize, y: usize| -> f32 {
        let p = Vec2::new(x as f32, y as f32) / extent; // isotropic, [0, 1] on the long side
        let uv = Vec2::new(x as f32 / (width - 1) as f32, y as f32 / (height - 1) as f32);
        let edge = uv.x.min(1.0 - uv.x).min(uv.y).min(1.0 - uv.y);

        let warp = Vec2::new(noise.fbm(p * 3.0 + 11.0, 4), noise.fbm(p * 3.0 + 37.0, 4)) * 0.05;
        let e = edge + noise.fbm(p * 4.0 + 5.0, 4) * 0.07;
        let mask = 1.0 - smoothstep(0.06, 0.27, e);

        // mountains: ridged noise, with the overall range height itself varying a lot
        let q = p + warp;
        let ridges = noise.ridged(q * 7.0, 10);
        let range_height = 0.25 + 1.1 * smoothstep(-0.5, 0.6, noise.fbm(q * 2.2 + 3.0, 3));
        let peaks = ridges.powf(1.6) * range_height + 0.12 * noise.fbm(q * 40.0, 3).abs() * ridges;

        // central plain: gentle undulation + fine grain
        let plain = 0.05 + 0.012 * noise.fbm(p * 5.0, 5) + 0.002 * noise.fbm(p * 60.0, 3);

        let mut plat = 0.0f32;
        for (i, pl) in plateaus.iter().enumerate() {
            let wobble = 1.0 + 0.3 * noise.fbm(p * 14.0 + i as f32 * 7.7, 4);
            let d = (p - pl.center).length() / pl.radius * wobble;
            let s = smoothstep(1.0, 0.82, d);
            let top = pl.height * (1.0 + 0.05 * noise.fbm(p * 50.0 + i as f32, 3));
            plat = plat.max(s * top);
        }

        plain + plat * (1.0 - mask) + mask.powf(1.3) * peaks
    };

    let mut data = vec![0.0f32; width * height];
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let rows_per = height.div_ceil(threads);
    std::thread::scope(|s| {
        for (chunk_i, chunk) in data.chunks_mut(rows_per * width).enumerate() {
            let eval = &eval;
            s.spawn(move || {
                for (i, v) in chunk.iter_mut().enumerate() {
                    let y = chunk_i * rows_per + i / width;
                    *v = eval(i % width, y);
                }
            });
        }
    });
    let (lo, hi) = data.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    for v in &mut data {
        *v = (*v - lo) / (hi - lo);
    }
    Heightmap { width, height, data }
}

// ---------------------------------------------------------------------------------------
// Error pyramid
// ---------------------------------------------------------------------------------------

/// Lowest pyramid level stored on the GPU (level L = blocks of 2^L x 2^L cells).
pub const PYRAMID_BASE_LEVEL: u32 = 2;

/// Per-block statistics over the map padded to a power-of-two square of `root_cells`.
/// Level L block = (min height, max height, E_L, max E_{L-2} over the block), where E_L is
/// the max deviation of the true heights from either triangulation of the block
/// corners, i.e. the geometric error of a mesh with vertex spacing 2^L.
/// `levels[i]` is level `PYRAMID_BASE_LEVEL + i`, `(root_cells >> L)^2` texels.
pub struct Pyramid {
    pub root_cells: usize,
    pub levels: Vec<Vec<[f32; 4]>>,
    /// `errors[i]` = E_l for l = i + 1 (blocks of 2^l cells, `(root_cells >> l)^2` values):
    /// the exact error of a mesh with vertex spacing 2^l inside each block. Lets the LOD
    /// test look only at the blocks a triangle actually overlaps.
    pub errors: Vec<Vec<f32>>,
}

/// An inclusive rectangle `(x0, y0, x1, y1)` of samples, texels or cells.
pub type Rect = (usize, usize, usize, usize);

/// The texels [`Pyramid::update`] rewrote: per entry of `levels` and of
/// `errors`, inclusive texel rectangles (they may overlap; empty:
/// unchanged).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PyramidChange {
    pub levels: Vec<Vec<Rect>>,
    pub errors: Vec<Vec<Rect>>,
}

/// Blocks `b` (of `n`, `s` cells a side, on an axis of `cells` cells)
/// whose stats read a corner in `d0..=d1`: block `b` reads corners
/// `b·s ..= b·s + s`, clamped to the map (corners beyond it are copies of
/// the edge).
fn blocks_reading(d0: usize, d1: usize, cells: usize, n: usize, s: usize) -> Option<(usize, usize)> {
    let reads = |b: usize| (b * s).min(cells) <= d1 && (b * s + s).min(cells) >= d0;
    let first = (0..n).find(|&b| reads(b))?;
    let last = (first..n).take_while(|&b| reads(b)).last()?;
    Some((first, last))
}

impl Pyramid {
    /// Recompute every block that reads a corner in any of `corners`
    /// (inclusive rectangles, in samples) after the heightmap changed
    /// there; the result equals [`build_pyramid`] of the new heightmap.
    /// The error of a block needs every sample in it, so the cost is about
    /// one pass over the map per level whose blocks the rectangles all
    /// reach (the top ones cover it), not over the rectangles; a block
    /// several rectangles reach is computed once.
    pub fn update(&mut self, hm: &Heightmap, height_scale: f32, height_offset: f32, corners: &[Rect]) -> PyramidChange {
        let cells = hm.cells();
        let k = self.root_cells.trailing_zeros();
        let h = |x: usize, y: usize| hm.at(x as isize, y as isize) * height_scale + height_offset;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let mut change = PyramidChange { levels: vec![Vec::new(); self.levels.len()], errors: vec![Vec::new(); self.errors.len()] };
        for l in 1..=k {
            let (n, s) = (self.root_cells >> l, 1usize << l);
            let mut rects = Vec::new();
            let mut blocks: Vec<(usize, usize)> = Vec::new();
            for &(x0, y0, x1, y1) in corners {
                let (Some((bx0, bx1)), Some((by0, by1))) = (blocks_reading(x0, x1, cells.0, n, s), blocks_reading(y0, y1, cells.1, n, s)) else { continue };
                rects.push((bx0, by0, bx1, by1));
                blocks.extend((by0..=by1).flat_map(|by| (bx0..=bx1).map(move |bx| (by, bx))));
            }
            blocks.sort_unstable();
            blocks.dedup();
            let mut stats = vec![[0.0f32; 3]; blocks.len()];
            let per = blocks.len().div_ceil(threads).max(1);
            let h = &h;
            std::thread::scope(|sc| {
                for (out, part) in stats.chunks_mut(per).zip(blocks.chunks(per)) {
                    sc.spawn(move || {
                        for (o, &(by, bx)) in out.iter_mut().zip(part) {
                            *o = block_stats(h, cells, l, bx, by);
                        }
                    });
                }
            });
            for (&(by, bx), st) in blocks.iter().zip(&stats) {
                self.errors[l as usize - 1][by * n + bx] = st[2];
            }
            if l >= PYRAMID_BASE_LEVEL {
                // E_{l-2} of the sub-blocks is current: their levels ran
                // first (E_0 is not stored, so it is recomputed).
                let level = (l - PYRAMID_BASE_LEVEL) as usize;
                for (&(by, bx), &st) in blocks.iter().zip(&stats) {
                    let fine = |x: usize, y: usize| if l == 2 { block_stats(h, cells, 0, x, y)[2] } else { self.errors[l as usize - 3][y * n * 4 + x] };
                    self.levels[level][by * n + bx] = level_entry(st, fine, n, by * n + bx);
                }
                change.levels[level] = rects.clone();
            }
            change.errors[l as usize - 1] = rects;
        }
        change
    }
}

/// `(min, max, E_l)` of the level-`l` block `(bx, by)` (2^l cells a side)
/// of a map of `cells` cells, with world heights from `h` (which clamps
/// beyond the map). The one definition both the full build and the
/// incremental update use, so they agree bit for bit.
fn block_stats(h: &impl Fn(usize, usize) -> f32, cells: (usize, usize), l: u32, bx: usize, by: usize) -> [f32; 3] {
    let (cx, cy) = cells;
    let s = 1usize << l;
    let (x0, y0) = (bx * s, by * s);
    let c = [h(x0, y0), h(x0 + s, y0), h(x0, y0 + s), h(x0 + s, y0 + s)];
    let (mut lo, mut hi, mut err) = (f32::MAX, f32::MIN, 0.0f32);
    // nodes beyond the map are clamped copies: skip them
    let xe = (x0 + s).min(cx.max(x0));
    let ye = (y0 + s).min(cy.max(y0));
    for y in y0..=ye {
        let ty = (y - y0) as f32 / s as f32;
        for x in x0..=xe {
            let tx = (x - x0) as f32 / s as f32;
            let v = h(x, y);
            // The mesh is triangles, not bilinear patches: measure against
            // both diagonal splits of the block (whichever LEB uses), which
            // across sharp features deviates up to ~2x more than bilinear.
            let [c00, c10, c01, c11] = c;
            let diag_a = if tx >= ty {
                c00 + (c10 - c00) * tx + (c11 - c10) * ty
            } else {
                c00 + (c11 - c01) * tx + (c01 - c00) * ty
            };
            let diag_b = if tx + ty <= 1.0 {
                c00 + (c10 - c00) * tx + (c01 - c00) * ty
            } else {
                c11 + (c11 - c01) * (tx - 1.0) + (c11 - c10) * (ty - 1.0)
            };
            err = err.max((v - diag_a).abs()).max((v - diag_b).abs());
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    [lo, hi, err]
}

/// Level-`l` pyramid entry of block `i` (row-major, `n` a side): its own
/// stats and the largest `E_{l-2}` among its 4x4 sub-blocks.
fn level_entry(stats: [f32; 3], fine_error: impl Fn(usize, usize) -> f32, n: usize, i: usize) -> [f32; 4] {
    let (bx, by) = (i % n, i / n);
    let mut e2 = 0.0f32;
    for y in 0..4 {
        for x in 0..4 {
            e2 = e2.max(fine_error(bx * 4 + x, by * 4 + y));
        }
    }
    let [lo, hi, e] = stats;
    [lo, hi, e, e2]
}

pub fn build_pyramid(hm: &Heightmap, height_scale: f32, height_offset: f32) -> Pyramid {
    let cells = hm.cells();
    let root_cells = cells.0.max(cells.1).next_power_of_two().max(1 << (PYRAMID_BASE_LEVEL + 2));
    let k = root_cells.trailing_zeros();
    let h = |x: usize, y: usize| hm.at(x as isize, y as isize) * height_scale + height_offset;
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());

    // stats[L] for L in 0..=k as (min, max, E_L)
    let mut stats: Vec<Vec<[f32; 3]>> = Vec::new();
    for l in 0..=k {
        let n = root_cells >> l;
        let mut out = vec![[0.0f32; 3]; n * n];
        let rows_per = n.div_ceil(threads);
        let h = &h;
        std::thread::scope(|sc| {
            for (ci, chunk) in out.chunks_mut(rows_per * n).enumerate() {
                sc.spawn(move || {
                    for (i, o) in chunk.iter_mut().enumerate() {
                        *o = block_stats(h, cells, l, i % n, ci * rows_per + i / n);
                    }
                });
            }
        });
        stats.push(out);
    }

    let levels = (PYRAMID_BASE_LEVEL..=k)
        .map(|l| {
            let n = root_cells >> l;
            let fine = &stats[(l - 2) as usize];
            (0..n * n).map(|i| level_entry(stats[l as usize][i], |x, y| fine[y * n * 4 + x][2], n, i)).collect()
        })
        .collect();
    let errors = stats[1..].iter().map(|l| l.iter().map(|s| s[2]).collect()).collect();
    Pyramid { root_cells, levels, errors }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(p: &Pyramid) -> (Vec<Vec<[u32; 4]>>, Vec<Vec<u32>>) {
        (p.levels.iter().map(|l| l.iter().map(|v| v.map(f32::to_bits)).collect()).collect(), p.errors.iter().map(|l| l.iter().map(|v| v.to_bits()).collect()).collect())
    }

    fn inside(rects: &[Rect], n: usize, i: usize) -> bool {
        rects.iter().any(|&(x0, y0, x1, y1)| (x0..=x1).contains(&(i % n)) && (y0..=y1).contains(&(i / n)))
    }

    /// Updating the pyramid after edits gives exactly the pyramid built
    /// from scratch, and every texel that changed is in the reported
    /// rectangles: edits at the map's edges and corners, across block
    /// boundaries, one corner, and heights outside the original [0, 1].
    #[test]
    fn pyramid_updates_equal_a_rebuild() {
        // 100 x 70 cells: not square, not a power of two (the pyramid pads
        // to 128 and clamps beyond the map).
        let mut hm = generate(101, 71, 3);
        let (scale, offset) = (420.0, -35.0);
        let mut pyramid = build_pyramid(&hm, scale, offset);
        // Each update takes one or more rectangles (several at once share
        // the blocks they both reach).
        let edits: [&[(Rect, f32)]; 6] = [
            &[((10, 10, 20, 14), 1.6)],
            &[((0, 0, 3, 2), -0.4), ((95, 60, 100, 70), 0.9)],
            &[((31, 31, 33, 33), 0.0), ((34, 31, 36, 33), 0.5), ((64, 0, 64, 70), 2.5)],
            &[((50, 40, 50, 40), -1.0)],
            &[((0, 66, 100, 70), 0.25)],
            &[((5, 5, 6, 6), 3.0), ((90, 5, 91, 6), -3.0), ((5, 60, 6, 61), 1.0), ((90, 60, 91, 61), 0.1)],
        ];
        for (n, edit) in edits.into_iter().enumerate() {
            for &((x0, y0, x1, y1), value) in edit {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        hm.data[y * hm.width + x] = value + 0.001 * (x * 7 + y * 3) as f32;
                    }
                }
            }
            let rects: Vec<Rect> = edit.iter().map(|e| e.0).collect();
            let before = bits(&pyramid);
            let change = pyramid.update(&hm, scale, offset, &rects);
            let after = bits(&pyramid);
            let rebuilt = build_pyramid(&hm, scale, offset);
            assert_eq!(after, bits(&rebuilt), "edit {n}: the update equals the rebuild");
            let mut changed = 0;
            for (l, (old, new)) in before.0.iter().zip(&after.0).enumerate() {
                let side = (old.len() as f64).sqrt() as usize;
                for (i, (a, b)) in old.iter().zip(new).enumerate() {
                    if a != b {
                        changed += 1;
                        assert!(inside(&change.levels[l], side, i), "edit {n}: level {l} texel {i} changed outside {:?}", change.levels[l]);
                    }
                }
            }
            for (l, (old, new)) in before.1.iter().zip(&after.1).enumerate() {
                let side = (old.len() as f64).sqrt() as usize;
                for (i, (a, b)) in old.iter().zip(new).enumerate() {
                    if a != b {
                        assert!(inside(&change.errors[l], side, i), "edit {n}: error level {l} texel {i} changed outside {:?}", change.errors[l]);
                    }
                }
            }
            assert!(changed > 0, "edit {n} changed something");
            // The top level is a single block; it always reads the edit.
            assert!(change.levels.last().unwrap().iter().all(|&r| r == (0, 0, 0, 0)) && !change.levels.last().unwrap().is_empty());
            assert_eq!(change.levels[0].len(), rects.len(), "one region per rectangle at the finest level");
        }
    }

    #[test]
    fn corners_average_the_cells_around_them() {
        let cells = (3, 2);
        let ground = |x: usize, y: usize| (y * 3 + x) as f32;
        assert_eq!(Heightmap::corner(cells, ground, 0, 0), 0.0);
        assert_eq!(Heightmap::corner(cells, ground, 1, 1), (0.0 + 1.0 + 3.0 + 4.0) / 4.0);
        assert_eq!(Heightmap::corner(cells, ground, 3, 2), 5.0);
        assert_eq!(Heightmap::corner(cells, ground, 3, 1), (2.0 + 5.0) / 2.0);
    }
}
