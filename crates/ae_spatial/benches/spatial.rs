use ae_spatial::{Bounds, Circle, Grid, Hits, Nearest, Segment};
use criterion::{Criterion, criterion_group, criterion_main};

fn points(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut f = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 * 20480.0
    };
    (0..n).map(|_| (f(), f())).unzip()
}

fn bench(c: &mut Criterion) {
    let b = Bounds {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 20480.0,
        max_y: 20480.0,
    };
    let (x, y) = points(50_000);
    let mut g = Grid::new(b, 64.0);
    c.bench_function("rebuild_50k", |bn| bn.iter(|| g.rebuild(&x, &y, &[])));
    g.rebuild(&x, &y, &[]);
    let circles: Vec<Circle> = x
        .iter()
        .zip(&y)
        .take(10_000)
        .map(|(&x, &y)| Circle { x, y, r: 160.0 })
        .collect();
    let near: Vec<Nearest> = x
        .iter()
        .zip(&y)
        .take(10_000)
        .map(|(&x, &y)| Nearest {
            x,
            y,
            max_r: 400.0,
            k: 8,
        })
        .collect();
    let segs: Vec<Segment> = x
        .iter()
        .zip(&y)
        .take(10_000)
        .map(|(&x, &y)| Segment {
            x0: x,
            y0: y,
            x1: x + 300.0,
            y1: y - 200.0,
            pad: 2.0,
        })
        .collect();
    let mut hits = Hits::default();
    c.bench_function("radius_10k_r160", |bn| bn.iter(|| g.radius(&circles, &mut hits)));
    c.bench_function("nearest8_10k", |bn| bn.iter(|| g.nearest(&near, &mut hits)));
    c.bench_function("sweep_10k_len360", |bn| bn.iter(|| g.sweep(&segs, &mut hits)));
}

criterion_group!(benches, bench);
criterion_main!(benches);
