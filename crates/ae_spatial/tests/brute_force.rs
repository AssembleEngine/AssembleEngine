//! Every query kind against a brute-force scan over all items, on random
//! items (including items outside the grid bounds and on cell edges), and
//! the same output at 1 and 8 threads.

use ae_spatial::{Aabb, Bounds, Circle, Grid, Hits, Nearest, Segment};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + (hi - lo) * self.f() }
}

const B: Bounds = Bounds { min_x: 0.0, min_y: 0.0, max_x: 1000.0, max_y: 600.0 };

struct World {
    x: Vec<f32>,
    y: Vec<f32>,
    r: Vec<f32>,
}

fn world(n: usize, seed: u64) -> World {
    let mut rng = Rng(seed);
    let mut w = World { x: Vec::new(), y: Vec::new(), r: Vec::new() };
    for i in 0..n {
        // Mostly inside, some outside the bounds, some on exact cell edges.
        let (x, y) = match i % 10 {
            0 => (rng.range(-200.0, 1200.0), rng.range(-150.0, 750.0)),
            1 => ((rng.range(0.0, 40.0) as i32) as f32 * 25.0, (rng.range(0.0, 24.0) as i32) as f32 * 25.0),
            _ => (rng.range(0.0, 1000.0), rng.range(0.0, 600.0)),
        };
        w.x.push(x);
        w.y.push(y);
        w.r.push(rng.range(0.5, 6.0));
    }
    w
}

fn grid(w: &World, cell: f32) -> Grid {
    let mut g = Grid::new(B, cell);
    g.rebuild(&w.x, &w.y, &w.r);
    g
}

fn in_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
}

#[test]
fn radius_and_box_match_brute_force() {
    let w = world(5000, 1);
    let mut rng = Rng(2);
    let circles: Vec<Circle> = (0..700).map(|_| Circle { x: rng.range(-100.0, 1100.0), y: rng.range(-100.0, 700.0), r: rng.range(0.0, 90.0) }).collect();
    let boxes: Vec<Aabb> = (0..700)
        .map(|_| {
            let (x, y) = (rng.range(-100.0, 1100.0), rng.range(-100.0, 700.0));
            Aabb { min_x: x, min_y: y, max_x: x + rng.range(0.0, 150.0), max_y: y + rng.range(0.0, 150.0) }
        })
        .collect();
    for cell in [7.0, 25.0, 64.0] {
        let g = grid(&w, cell);
        let mut hits = Hits::default();
        g.radius(&circles, &mut hits);
        for (q, c) in circles.iter().enumerate() {
            let want: Vec<u32> = (0..w.x.len() as u32)
                .filter(|&i| {
                    let (dx, dy) = (w.x[i as usize] - c.x, w.y[i as usize] - c.y);
                    dx * dx + dy * dy <= c.r * c.r
                })
                .collect();
            assert_eq!(hits.of(q), &want[..], "circle {q} cell {cell}");
        }
        g.boxes(&boxes, &mut hits);
        for (q, b) in boxes.iter().enumerate() {
            let want: Vec<u32> = (0..w.x.len() as u32)
                .filter(|&i| {
                    let (x, y) = (w.x[i as usize], w.y[i as usize]);
                    x >= b.min_x && x <= b.max_x && y >= b.min_y && y <= b.max_y
                })
                .collect();
            assert_eq!(hits.of(q), &want[..], "box {q} cell {cell}");
        }
    }
}

#[test]
fn nearest_matches_brute_force() {
    let w = world(4000, 3);
    let mut rng = Rng(4);
    let queries: Vec<Nearest> = (0..600)
        .map(|i| Nearest { x: rng.range(-150.0, 1150.0), y: rng.range(-150.0, 750.0), max_r: if i % 3 == 0 { f32::INFINITY } else { rng.range(5.0, 200.0) }, k: (i % 13) as u32 })
        .collect();
    for cell in [9.0, 40.0] {
        let g = grid(&w, cell);
        let mut hits = Hits::default();
        g.nearest(&queries, &mut hits);
        for (q, n) in queries.iter().enumerate() {
            let mut all: Vec<(f32, u32)> = (0..w.x.len() as u32)
                .map(|i| {
                    let (dx, dy) = (w.x[i as usize] - n.x, w.y[i as usize] - n.y);
                    (dx * dx + dy * dy, i)
                })
                .filter(|&(d2, _)| d2 <= n.max_r * n.max_r)
                .collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            all.truncate(n.k as usize);
            let want: Vec<u32> = all.iter().map(|a| a.1).collect();
            assert_eq!(hits.of(q), &want[..], "nearest {q} cell {cell} {n:?}");
            assert_eq!(hits.values_of(q), &all.iter().map(|a| a.0).collect::<Vec<_>>()[..]);
        }
    }
}

#[test]
fn sweep_matches_brute_force() {
    let w = world(4000, 5);
    let mut rng = Rng(6);
    let segs: Vec<Segment> = (0..600)
        .map(|i| {
            let (x0, y0) = (rng.range(-150.0, 1150.0), rng.range(-150.0, 750.0));
            let (len, ang) = (if i % 5 == 0 { 0.0 } else { rng.range(0.0, 600.0) }, rng.range(0.0, 6.283));
            Segment { x0, y0, x1: x0 + len * ang.cos(), y1: y0 + len * ang.sin(), pad: if i % 2 == 0 { 0.0 } else { rng.range(0.0, 8.0) } }
        })
        .collect();
    for cell in [10.0, 50.0] {
        let g = grid(&w, cell);
        let mut hits = Hits::default();
        g.sweep(&segs, &mut hits);
        for (q, s) in segs.iter().enumerate() {
            let (dx, dy) = (s.x1 - s.x0, s.y1 - s.y0);
            let len2 = dx * dx + dy * dy;
            // Brute force by closest approach; tangent cases are skipped.
            for i in 0..w.x.len() as u32 {
                let (px, py, r) = (w.x[i as usize], w.y[i as usize], s.pad + w.r[i as usize]);
                let t = if len2 == 0.0 { 0.0 } else { (((px - s.x0) * dx + (py - s.y0) * dy) / len2).clamp(0.0, 1.0) };
                let (cx, cy) = (s.x0 + dx * t - px, s.y0 + dy * t - py);
                let d2 = cx * cx + cy * cy;
                if (d2 - r * r).abs() <= 1e-2 * r * r.max(1.0) { continue; }
                let found = hits.of(q).contains(&i);
                assert_eq!(found, d2 < r * r, "segment {q} item {i} cell {cell}: d2 {d2} r2 {}", r * r);
            }
            let v = hits.values_of(q);
            let items = hits.of(q);
            for k in 1..v.len() {
                assert!(v[k - 1] < v[k] || (v[k - 1] == v[k] && items[k - 1] < items[k]), "segment {q}: hits out of order");
            }
            assert!(v.iter().all(|t| (0.0..=1.0).contains(t)));
        }
    }
}

#[test]
fn same_hits_at_any_thread_count() {
    let w = world(20_000, 7);
    let mut rng = Rng(8);
    let circles: Vec<Circle> = (0..3000).map(|_| Circle { x: rng.range(0.0, 1000.0), y: rng.range(0.0, 600.0), r: 30.0 }).collect();
    let near: Vec<Nearest> = (0..3000).map(|_| Nearest { x: rng.range(0.0, 1000.0), y: rng.range(0.0, 600.0), max_r: 100.0, k: 8 }).collect();
    let segs: Vec<Segment> = (0..3000)
        .map(|_| {
            let (x, y) = (rng.range(0.0, 1000.0), rng.range(0.0, 600.0));
            Segment { x0: x, y0: y, x1: x + rng.range(-200.0, 200.0), y1: y + rng.range(-200.0, 200.0), pad: 1.0 }
        })
        .collect();
    let run = |threads| {
        in_pool(threads, || {
            let g = grid(&w, 32.0);
            let (mut a, mut b, mut c) = (Hits::default(), Hits::default(), Hits::default());
            g.radius(&circles, &mut a);
            g.nearest(&near, &mut b);
            g.sweep(&segs, &mut c);
            (a, b, c)
        })
    };
    let one = run(1);
    assert!(one.0.len() > 10_000 && one.1.len() > 10_000 && one.2.len() > 1_000);
    assert_eq!(one, run(8));
}
