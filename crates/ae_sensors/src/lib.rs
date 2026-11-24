//! Sensor coverage: which cells each allyteam sees (line of sight) and
//! detects (radar), as one bit per cell, stamped from unit columns.
//!
//! Everything after the position-to-cell step is integer math, so coverage
//! is the same on every machine and could move into the sim as a native
//! service unchanged:
//! * a position becomes the cell `floor(x / cell_size)`, clamped to the
//!   grid (a unit off the map sees from the edge cell);
//! * a world radius becomes `floor(r / cell_size)` cells;
//! * a unit at cell `(ux, uy)` with radius `r` covers every cell `(x, y)`
//!   with `(x - ux)² + (y - uy)² <= r²`: per row `dy` the span
//!   `|x - ux| <= isqrt(r² - dy²)`, clipped to the grid.
//!
//! No hash-map iteration; allyteams are stamped in parallel (each into its
//! own grids), which gives the same bits as serial stamping.

use rayon::prelude::*;

/// The cell layout coverage is stamped on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensorGrid {
    pub width: u32,
    pub height: u32,
    /// World units per cell side.
    pub cell_size: f32,
}

impl SensorGrid {
    /// Cells covering `extent` world units (rounded up).
    pub fn new(extent_x: f32, extent_y: f32, cell_size: f32) -> Self {
        assert!(
            cell_size > 0.0 && extent_x > 0.0 && extent_y > 0.0,
            "a grid needs a positive extent and cell size"
        );
        Self {
            width: (extent_x / cell_size).ceil().max(1.0) as u32,
            height: (extent_y / cell_size).ceil().max(1.0) as u32,
            cell_size,
        }
    }

    /// The cell of a world position, clamped to the grid.
    pub fn cell(&self, x: f32, y: f32) -> (u32, u32) {
        let c = |v: f32, n: u32| {
            let f = (v / self.cell_size).floor();
            if f.is_nan() || f < 0.0 {
                0
            } else {
                (f as u32).min(n - 1)
            }
        };
        (c(x, self.width), c(y, self.height))
    }

    /// A world radius in whole cells (rounded down; negative or NaN is 0).
    pub fn radius(&self, r: f32) -> u32 {
        let f = (r / self.cell_size).floor();
        if f.is_nan() || f < 0.0 {
            0
        } else {
            f.min(u16::MAX as f32) as u32
        }
    }
}

/// One bit per cell, rows padded to whole 64-bit words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bits {
    pub width: u32,
    pub height: u32,
    /// Words per row.
    pub stride: usize,
    pub words: Vec<u64>,
}

impl Bits {
    pub fn new(width: u32, height: u32) -> Self {
        let stride = (width as usize).div_ceil(64);
        Self {
            width,
            height,
            stride,
            words: vec![0; stride * height as usize],
        }
    }

    pub fn get(&self, x: u32, y: u32) -> bool {
        x < self.width
            && y < self.height
            && self.words[y as usize * self.stride + (x / 64) as usize] >> (x % 64) & 1 != 0
    }

    pub fn set(&mut self, x: u32, y: u32) {
        self.set_span(y, x, x);
    }

    /// Set cells `x0..=x1` of row `y` (`x0 <= x1 < width`).
    pub fn set_span(&mut self, y: u32, x0: u32, x1: u32) {
        debug_assert!(x0 <= x1 && x1 < self.width && y < self.height);
        let row = &mut self.words[y as usize * self.stride..][..self.stride];
        let (w0, w1) = ((x0 / 64) as usize, (x1 / 64) as usize);
        let low = !0u64 << (x0 % 64);
        let high = !0u64 >> (63 - x1 % 64);
        if w0 == w1 {
            row[w0] |= low & high;
        } else {
            row[w0] |= low;
            for w in &mut row[w0 + 1..w1] {
                *w = !0;
            }
            row[w1] |= high;
        }
    }

    /// Cells set.
    pub fn count(&self) -> u64 {
        self.words.iter().map(|w| w.count_ones() as u64).sum()
    }

    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// `self |= other` (same shape).
    pub fn or(&mut self, other: &Bits) {
        assert_eq!((self.width, self.height), (other.width, other.height));
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= b;
        }
    }
}

/// Integer square root (floor).
pub fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    // A float estimate, corrected exactly (u128: no overflow near 2^64).
    let mut x = (n as f64).sqrt() as u128;
    let n = n as u128;
    while x * x > n {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= n {
        x += 1;
    }
    x as u64
}

/// Half widths of a disc of radius `r` cells, per row offset `0..=r`.
fn disc(r: u32) -> Vec<u32> {
    let r2 = r as u64 * r as u64;
    (0..=r as u64).map(|dy| isqrt(r2 - dy * dy) as u32).collect()
}

/// The units that carry sensors, as columns (one row per unit).
#[derive(Clone, Copy, Debug)]
pub struct Sensors<'a> {
    pub x: &'a [f32],
    pub y: &'a [f32],
    /// Allyteam index, `< allies`; rows with a larger index are ignored.
    pub ally: &'a [u32],
    /// Line-of-sight radius, world units (0: none).
    pub los: &'a [f32],
    /// Radar radius, world units (0: none).
    pub radar: &'a [f32],
    /// Sonar radius, when supplied by the game.
    pub sonar: Option<&'a [f32]>,
}

impl Sensors<'_> {
    fn len(&self) -> usize {
        let n = self.x.len();
        assert!(
            self.y.len() == n
                && self.ally.len() == n
                && self.los.len() == n
                && self.radar.len() == n
                && self.sonar.is_none_or(|s| s.len() == n),
            "sensor columns differ in length"
        );
        n
    }
}

/// Coverage per allyteam.
#[derive(Clone, Debug, PartialEq)]
pub struct Coverage {
    pub grid: SensorGrid,
    pub los: Vec<Bits>,
    pub radar: Vec<Bits>,
    pub sonar: Vec<Bits>,
    /// Cells written while stamping (spans summed, overlaps counted
    /// again): the work done, deterministic.
    pub stamped: u64,
}

impl Coverage {
    pub fn new(grid: SensorGrid, allies: usize) -> Self {
        let bits = || (0..allies).map(|_| Bits::new(grid.width, grid.height)).collect();
        Self {
            grid,
            los: bits(),
            radar: bits(),
            sonar: bits(),
            stamped: 0,
        }
    }

    pub fn allies(&self) -> usize {
        self.los.len()
    }

    /// Clear and stamp every unit's discs (allyteams in parallel when
    /// `parallel`; the bits are the same either way).
    pub fn stamp(&mut self, units: &Sensors<'_>, parallel: bool) {
        let n = units.len();
        let allies = self.allies();
        // Rows per allyteam, in row order (a counting sort: deterministic).
        let mut by_ally: Vec<Vec<u32>> = vec![Vec::new(); allies];
        for (row, &a) in units.ally.iter().enumerate().take(n) {
            if let Some(list) = by_ally.get_mut(a as usize) {
                list.push(row as u32);
            }
        }
        let grid = self.grid;
        let stamp_ally = |(rows, ((los, radar), sonar)): (&Vec<u32>, ((&mut Bits, &mut Bits), &mut Bits))| -> u64 {
            los.clear();
            radar.clear();
            sonar.clear();
            let mut discs: Vec<(u32, Vec<u32>)> = Vec::new();
            let mut stamped = 0;
            for &row in rows {
                let r = row as usize;
                let (cx, cy) = grid.cell(units.x[r], units.y[r]);
                for (bits, radius) in [
                    (&mut *los, units.los[r]),
                    (&mut *radar, units.radar[r]),
                    (&mut *sonar, units.sonar.map_or(0.0, |s| s[r])),
                ] {
                    if radius.is_nan() || radius <= 0.0 {
                        continue;
                    }
                    let rc = grid.radius(radius);
                    let at = match discs.iter().position(|(d, _)| *d == rc) {
                        Some(i) => i,
                        None => {
                            discs.push((rc, disc(rc)));
                            discs.len() - 1
                        }
                    };
                    stamped += stamp_disc(bits, cx, cy, &discs[at].1);
                }
            }
            stamped
        };
        let pairs = self.los.iter_mut().zip(&mut self.radar).zip(&mut self.sonar);
        self.stamped = if parallel {
            by_ally.par_iter().zip(pairs.collect::<Vec<_>>()).map(stamp_ally).sum()
        } else {
            by_ally.iter().zip(pairs).map(stamp_ally).sum()
        };
    }

    /// Whether `ally` has line of sight to / radar on the cell of a world
    /// position.
    pub fn sees(&self, ally: usize, x: f32, y: f32) -> (bool, bool) {
        let (cx, cy) = self.grid.cell(x, y);
        (self.los[ally].get(cx, cy), self.radar[ally].get(cx, cy))
    }
}

/// Set a disc's cells; the number of cells written.
fn stamp_disc(bits: &mut Bits, cx: u32, cy: u32, half: &[u32]) -> u64 {
    let r = half.len() as i64 - 1;
    let (w, h) = (bits.width as i64, bits.height as i64);
    let mut stamped = 0;
    for dy in -r..=r {
        let y = cy as i64 + dy;
        if y < 0 || y >= h {
            continue;
        }
        let hw = half[dy.unsigned_abs() as usize] as i64;
        let x0 = (cx as i64 - hw).max(0);
        let x1 = (cx as i64 + hw).min(w - 1);
        bits.set_span(y as u32, x0 as u32, x1 as u32);
        stamped += (x1 - x0 + 1) as u64;
    }
    stamped
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self, max: f32) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * max
        }
    }

    struct World {
        x: Vec<f32>,
        y: Vec<f32>,
        ally: Vec<u32>,
        los: Vec<f32>,
        radar: Vec<f32>,
    }

    fn world(seed: u64, n: usize, allies: u32, extent: f32) -> World {
        let mut rng = Rng(seed);
        let mut w = World {
            x: vec![],
            y: vec![],
            ally: vec![],
            los: vec![],
            radar: vec![],
        };
        for i in 0..n {
            // Some off the map, some with no radar, one NaN radius.
            w.x.push(rng.f(extent * 1.1) - extent * 0.05);
            w.y.push(rng.f(extent * 1.1) - extent * 0.05);
            w.ally.push((rng.next() % (allies as u64 + 1)) as u32); // allies: ignored
            w.los.push(rng.f(300.0));
            w.radar.push(if i % 3 == 0 {
                0.0
            } else if i == 5 {
                f32::NAN
            } else {
                rng.f(900.0)
            });
        }
        w
    }

    fn sensors(w: &World) -> Sensors<'_> {
        Sensors {
            x: &w.x,
            y: &w.y,
            ally: &w.ally,
            los: &w.los,
            radar: &w.radar,
            sonar: None,
        }
    }

    /// Brute force: a cell is covered if some unit of the allyteam is within
    /// its radius in cells (the documented predicate).
    fn oracle(grid: &SensorGrid, w: &World, ally: u32, radar: bool) -> Bits {
        let mut bits = Bits::new(grid.width, grid.height);
        for y in 0..grid.height {
            for x in 0..grid.width {
                let covered = (0..w.x.len()).any(|i| {
                    let radius = if radar { w.radar[i] } else { w.los[i] };
                    if w.ally[i] != ally || radius.is_nan() || radius <= 0.0 {
                        return false;
                    }
                    let (ux, uy) = grid.cell(w.x[i], w.y[i]);
                    let r = grid.radius(radius) as i64;
                    let (dx, dy) = (x as i64 - ux as i64, y as i64 - uy as i64);
                    dx * dx + dy * dy <= r * r
                });
                if covered {
                    bits.set(x, y);
                }
            }
        }
        bits
    }

    #[test]
    fn sonar_stamps_the_same_discs_as_radar() {
        let grid = SensorGrid::new(256.0, 256.0, 32.0);
        let mut coverage = Coverage::new(grid, 2);
        let world = world(739, 64, 2, 256.0);
        let units = sensors(&world);
        coverage.stamp(
            &Sensors {
                sonar: Some(units.radar),
                ..units
            },
            true,
        );
        assert_eq!(coverage.sonar, coverage.radar);
        assert!(coverage.sonar.iter().any(|bits| bits.count() != 0));
        coverage.stamp(&units, false);
        assert!(coverage.sonar.iter().all(|bits| bits.count() == 0));
    }

    #[test]
    fn stamping_matches_the_disc_predicate() {
        let grid = SensorGrid::new(3000.0, 2000.0, 32.0);
        assert_eq!((grid.width, grid.height), (94, 63));
        for seed in [1, 2, 3] {
            let w = world(seed * 0x9e37_79b9, 60, 3, 3000.0);
            let mut c = Coverage::new(grid, 3);
            c.stamp(&sensors(&w), false);
            for a in 0..3 {
                assert_eq!(c.los[a], oracle(&grid, &w, a as u32, false), "seed {seed} ally {a} los");
                assert_eq!(
                    c.radar[a],
                    oracle(&grid, &w, a as u32, true),
                    "seed {seed} ally {a} radar"
                );
                assert!(c.los[a].count() > 0 && c.radar[a].count() > c.los[a].count() / 4);
            }
        }
    }

    #[test]
    fn parallel_stamping_gives_the_same_bits() {
        let grid = SensorGrid::new(20480.0, 20480.0, 32.0);
        let w = world(77, 10_000, 4, 20480.0);
        let (mut serial, mut parallel) = (Coverage::new(grid, 4), Coverage::new(grid, 4));
        serial.stamp(&sensors(&w), false);
        parallel.stamp(&sensors(&w), true);
        assert_eq!(serial, parallel);
        // Stamping again replaces, not accumulates.
        let before = parallel.clone();
        parallel.stamp(&sensors(&w), true);
        assert_eq!(parallel, before);
    }

    #[test]
    fn spans_cells_and_radii() {
        let mut b = Bits::new(130, 2);
        b.set_span(1, 3, 3);
        b.set_span(0, 60, 129);
        assert_eq!(b.count(), 1 + 70);
        assert!(
            b.get(3, 1)
                && !b.get(2, 1)
                && !b.get(4, 1)
                && b.get(64, 0)
                && b.get(129, 0)
                && !b.get(59, 0)
                && !b.get(130, 0)
        );
        let g = SensorGrid::new(100.0, 100.0, 10.0);
        assert_eq!(g.cell(-5.0, 1e9), (0, 9));
        assert_eq!(g.cell(f32::NAN, 55.0), (0, 5));
        assert_eq!((g.radius(19.9), g.radius(-1.0), g.radius(f32::NAN)), (1, 0, 0));
        for n in [
            0u64,
            1,
            2,
            3,
            4,
            15,
            16,
            17,
            1 << 40,
            (1 << 40) - 1,
            u32::MAX as u64 * u32::MAX as u64,
        ] {
            let r = isqrt(n) as u128;
            assert!(r * r <= n as u128 && (r + 1) * (r + 1) > n as u128, "{n}");
        }
        // Radius 0 cells: just the unit's cell.
        let mut c = Coverage::new(g, 1);
        c.stamp(
            &Sensors {
                x: &[55.0],
                y: &[55.0],
                ally: &[0],
                los: &[5.0],
                radar: &[0.0],
                sonar: None,
            },
            false,
        );
        assert_eq!((c.los[0].count(), c.los[0].get(5, 5), c.radar[0].count()), (1, true, 0));
    }
}
