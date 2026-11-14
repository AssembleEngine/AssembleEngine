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

pub fn build_pyramid(hm: &Heightmap, height_scale: f32, height_offset: f32) -> Pyramid {
    let (cx, cy) = hm.cells();
    let root_cells = cx.max(cy).next_power_of_two().max(1 << (PYRAMID_BASE_LEVEL + 2));
    let k = root_cells.trailing_zeros();
    let h = |x: usize, y: usize| hm.at(x as isize, y as isize) * height_scale + height_offset;
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());

    // stats[L] for L in 0..=k as (min, max, E_L)
    let mut stats: Vec<Vec<[f32; 3]>> = Vec::new();
    for l in 0..=k {
        let n = root_cells >> l;
        let s = 1usize << l;
        let mut out = vec![[0.0f32; 3]; n * n];
        let rows_per = n.div_ceil(threads);
        std::thread::scope(|sc| {
            for (ci, chunk) in out.chunks_mut(rows_per * n).enumerate() {
                sc.spawn(move || {
                    for (i, o) in chunk.iter_mut().enumerate() {
                        let (bx, by) = (i % n, ci * rows_per + i / n);
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
                        *o = [lo, hi, err];
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
            let fn_ = n * 4;
            (0..n * n)
                .map(|i| {
                    let (bx, by) = (i % n, i / n);
                    let mut e2 = 0.0f32;
                    for y in 0..4 {
                        for x in 0..4 {
                            e2 = e2.max(fine[(by * 4 + y) * fn_ + bx * 4 + x][2]);
                        }
                    }
                    let [lo, hi, e] = stats[l as usize][i];
                    [lo, hi, e, e2]
                })
                .collect()
        })
        .collect();
    let errors = stats[1..].iter().map(|l| l.iter().map(|s| s[2]).collect()).collect();
    Pyramid { root_cells, levels, errors }
}
