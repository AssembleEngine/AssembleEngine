//! The sun and the terrain's shadow from it (M7).
//!
//! The sun is fixed (one direction for every pass). The hills' shadows are
//! a heightfield, not a shadow map: for each corner of the heightmap, the
//! *shadow height* is the highest point above it that terrain toward the
//! sun still shades, `S(p) = max over t >= 0 of H(p + t d) - t tan(e)` (`d`
//! the sun's horizontal direction, `e` its elevation). A point at height
//! `y` over `p` is in the hills' shadow when `y < S(p)`; on the ground
//! `S(p) = H(p)` where the sun reaches it. One texture answers for the
//! terrain, the units and the water surface at any distance, with no
//! depth bias, no cascades and no view dependence (hills off screen cast
//! into it), and it changes only when the terrain does.
//!
//! It is computed on the CPU, exactly: from each corner a ray steps one
//! cell at a time toward the sun over the bilinear heightfield, and stops
//! once even the map's highest point could no longer rise above what it
//! has found, or it leaves the map. Rows are split over threads. An edit
//! recomputes only the corners its shadow can reach ([`update`]).

use super::heightmap::Heightmap;
use glam::Vec3;

/// Direction to the sun (unit length).
pub fn direction() -> Vec3 {
    Vec3::new(-0.5, 0.55, 0.35).normalize()
}

/// Inclusive corner rectangle `(x0, y0, x1, y1)`.
pub type Rect = (usize, usize, usize, usize);

/// The sun as the march sees it: a step of one cell toward it, in cells
/// on the map and in height gained.
#[derive(Clone, Copy)]
struct March {
    dx: f32,
    dy: f32,
    rise: f32,
}

fn march(cell: f32, sun: Vec3) -> Option<March> {
    let flat = (sun.x * sun.x + sun.z * sun.z).sqrt();
    (flat > 1e-6 && sun.y > 0.0).then(|| March {
        dx: sun.x / flat,
        dy: sun.z / flat,
        rise: cell * sun.y / flat,
    })
}

/// The heightmap in world units, read without a copy.
#[derive(Clone, Copy)]
struct Ground<'a> {
    data: &'a [f32],
    scale: f32,
    offset: f32,
}

impl Ground<'_> {
    fn at(&self, i: usize) -> f32 {
        self.data[i] * self.scale + self.offset
    }

    /// Lowest and highest world height.
    fn range(&self) -> (f32, f32) {
        let (lo, hi) = self
            .data
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let (a, b) = (lo * self.scale + self.offset, hi * self.scale + self.offset);
        (a.min(b), a.max(b))
    }

    /// Copy the ground heights of `rect` into `shadow` (rows `w` long).
    fn reset(&self, shadow: &mut [f32], w: usize, (x0, y0, x1, y1): Rect) {
        for y in y0..=y1 {
            for x in x0..=x1 {
                shadow[y * w + x] = self.at(y * w + x);
            }
        }
    }
}

/// World shadow height per corner of `heightmap` (row-major like its data)
/// for corners `cell` apart, world height = sample × `scale` + `offset`,
/// with the sun in direction `sun`. A sun overhead or below the horizon
/// casts nothing (the ground heights).
pub fn shadow_heights(heightmap: &Heightmap, cell: f32, scale: f32, offset: f32, sun: Vec3) -> Vec<f32> {
    let ground = Ground {
        data: &heightmap.data,
        scale,
        offset,
    };
    let mut s: Vec<f32> = (0..heightmap.data.len()).map(|i| ground.at(i)).collect();
    if let Some(m) = march(cell, sun) {
        fill(
            &mut s,
            ground,
            heightmap.width,
            heightmap.height,
            m,
            (0, 0, heightmap.width - 1, heightmap.height - 1),
        );
    }
    s
}

/// After the corners in `changed` (inclusive rectangles) moved: recompute
/// `shadow` (from [`shadow_heights`] with the same `cell`, `scale`,
/// `offset` and `sun`) where it can differ, the rectangles and the ground
/// their shadows (old or new) can reach. Returns the rectangles rewritten.
pub fn update(
    shadow: &mut [f32],
    heightmap: &Heightmap,
    cell: f32,
    scale: f32,
    offset: f32,
    sun: Vec3,
    changed: &[Rect],
) -> Vec<Rect> {
    let (w, h) = (heightmap.width, heightmap.height);
    let ground = Ground {
        data: &heightmap.data,
        scale,
        offset,
    };
    let Some(m) = march(cell, sun) else {
        for &rect in changed {
            ground.reset(shadow, w, rect);
        }
        return changed.to_vec();
    };
    // The longest shadow anything can cast (the old shadows included), in cells.
    let (lo, hi) = ground.range();
    let hi = shadow.iter().copied().fold(hi, f32::max);
    let reach = ((hi - lo) / m.rise).ceil() + 1.0;
    let mut out = Vec::new();
    for &(x0, y0, x1, y1) in changed {
        // The rectangle and its copy moved `reach` away from the sun.
        let (ex, ey) = (-m.dx * reach, -m.dy * reach);
        let clip = |v: f32, n: usize| v.clamp(0.0, (n - 1) as f32) as usize;
        let rect = (
            clip((x0 as f32 + ex.min(0.0)).floor(), w),
            clip((y0 as f32 + ey.min(0.0)).floor(), h),
            clip((x1 as f32 + ex.max(0.0)).ceil(), w),
            clip((y1 as f32 + ey.max(0.0)).ceil(), h),
        );
        ground.reset(shadow, w, rect);
        fill(shadow, ground, w, h, m, rect);
        out.push(rect);
    }
    out
}

/// March every corner of `rect` (whose `shadow` holds the ground heights).
fn fill(shadow: &mut [f32], ground: Ground<'_>, w: usize, h: usize, m: March, rect: Rect) {
    if w < 2 || h < 2 {
        return;
    }
    let top = ground.range().1;
    let (x0, y0, x1, y1) = rect;
    let (maxx, maxy) = ((w - 1) as f32, (h - 1) as f32);
    let sample = |x: f32, y: f32| {
        let (ix, iy) = ((x as usize).min(w - 2), (y as usize).min(h - 2));
        let (fx, fy) = (x - ix as f32, y - iy as f32);
        let i = iy * w + ix;
        let (g00, g10, g01, g11) = (ground.at(i), ground.at(i + 1), ground.at(i + w), ground.at(i + w + 1));
        let a = g00 + (g10 - g00) * fx;
        let b = g01 + (g11 - g01) * fx;
        a + (b - a) * fy
    };
    let row = |y: usize, out: &mut [f32]| {
        for (x, v) in (x0..=x1).zip(out.iter_mut()) {
            let mut best = *v;
            let mut t = 1.0f32;
            while top - t * m.rise > best {
                let (px, py) = (x as f32 + m.dx * t, y as f32 + m.dy * t);
                if !(0.0..=maxx).contains(&px) || !(0.0..=maxy).contains(&py) {
                    break; // nothing beyond the map casts
                }
                best = best.max(sample(px, py) - t * m.rise);
                t += 1.0;
            }
            *v = best;
        }
    };
    let rows: Vec<(usize, &mut [f32])> = shadow
        .chunks_mut(w)
        .enumerate()
        .skip(y0)
        .take(y1 - y0 + 1)
        .map(|(y, r)| (y, &mut r[x0..=x1]))
        .collect();
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(rows.len().div_ceil(16))
        .max(1);
    let per = rows.len().div_ceil(threads);
    let mut rows = rows;
    std::thread::scope(|scope| {
        while !rows.is_empty() {
            let rest = rows.split_off(per.min(rows.len()));
            let chunk = std::mem::replace(&mut rows, rest);
            let row = &row;
            scope.spawn(move || {
                for (y, out) in chunk {
                    row(y, out);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Heightmap {
        Heightmap {
            width: w,
            height: h,
            data: (0..w * h).map(|i| f(i % w, i / w)).collect(),
        }
    }

    /// Flat ground is lit everywhere (shadow height = ground height, exactly).
    #[test]
    fn flat_ground_casts_nothing() {
        let hm = map(33, 17, |_, _| 0.25);
        let s = shadow_heights(&hm, 8.0, 100.0, -5.0, direction());
        assert!(s.iter().all(|&v| v == 20.0));
    }

    /// A tower shades the ground away from the sun out to its height over
    /// tan(elevation), and nothing toward the sun.
    #[test]
    fn a_tower_shades_away_from_the_sun() {
        let (n, cell, tall) = (65usize, 4.0, 40.0);
        let c = 32;
        let hm = map(n, n, |x, y| {
            if (c - 1..=c + 1).contains(&x) && (c - 1..=c + 1).contains(&y) {
                1.0
            } else {
                0.0
            }
        });
        for sun in [
            direction(),
            Vec3::new(0.3, 0.6, -0.7).normalize(),
            Vec3::new(0.9, 0.5, 0.1).normalize(),
            Vec3::new(-0.1, 0.4, 0.95).normalize(),
        ] {
            let s = shadow_heights(&hm, cell, tall, 0.0, sun);
            let flat = (sun.x * sun.x + sun.z * sun.z).sqrt();
            let (dx, dz, rise) = (sun.x / flat, sun.z / flat, sun.y / flat);
            let length = tall / rise; // world units
            let at = |t: f32| {
                let x = (c as f32 - dx * t / cell).round() as usize;
                let y = (c as f32 - dz * t / cell).round() as usize;
                s[y * n + x]
            };
            // Away from the sun: shaded to most of the shadow's length, with
            // the shadow height falling as it goes.
            for t in [8.0, length * 0.5, length * 0.8] {
                let expect = tall - t * rise;
                assert!(
                    (at(t) - expect).abs() < tall * 0.1,
                    "{sun:?} t {t}: {} vs {expect}",
                    at(t)
                );
            }
            assert!(at(length + 3.0 * cell) == 0.0, "{sun:?}: lit past the shadow's end");
            // Toward the sun: lit.
            for t in [8.0, 20.0, length] {
                let x = (c as f32 + dx * t / cell).round() as usize;
                let y = (c as f32 + dz * t / cell).round() as usize;
                assert_eq!(s[y * n + x], 0.0, "{sun:?}: lit toward the sun at {t}");
            }
        }
    }

    /// An edit's update equals a recompute from scratch, raised or lowered.
    #[test]
    fn updates_equal_a_full_recompute() {
        let (n, cell) = (97usize, 4.0);
        let mut hm = map(n, n, |x, y| {
            ((x * 7 + y * 13) % 11) as f32 * 0.02 + (x as f32 * 0.05).sin() * 0.1
        });
        let sun = direction();
        let mut s = shadow_heights(&hm, cell, 80.0, 3.0, sun);
        for (rect, v) in [
            ((40, 40, 44, 46), 0.9),
            ((40, 40, 44, 46), 0.0),
            ((0, 90, 6, 96), 1.0),
            ((90, 0, 96, 3), 0.7),
        ] {
            let (x0, y0, x1, y1) = rect;
            for y in y0..=y1 {
                for x in x0..=x1 {
                    hm.data[y * n + x] = v;
                }
            }
            let written = update(&mut s, &hm, cell, 80.0, 3.0, sun, &[rect]);
            assert!(written.iter().all(|r| r.0 <= x0 && r.1 <= y0 && r.2 >= x1 && r.3 >= y1));
            assert_eq!(s, shadow_heights(&hm, cell, 80.0, 3.0, sun), "after {rect:?} = {v}");
        }
    }

    /// No sun above the horizon, no shadows.
    #[test]
    fn overhead_or_set_sun_casts_nothing() {
        let hm = map(9, 9, |x, _| x as f32);
        let ground: Vec<f32> = hm.data.iter().map(|v| v * 2.0).collect();
        assert_eq!(shadow_heights(&hm, 1.0, 2.0, 0.0, Vec3::Y), ground);
        assert_eq!(shadow_heights(&hm, 1.0, 2.0, 0.0, Vec3::new(1.0, -0.2, 0.0)), ground);
    }
}
