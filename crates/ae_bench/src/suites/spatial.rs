//! Spatial query service: grid rebuild and batch queries over a synthetic
//! army (clustered like units on a map, not uniform noise), at 1 and N
//! threads. Hit counts are exact counters; the hit streams must be equal at
//! every thread count, or the suite fails.

use super::Options;
use crate::measure::time;
use crate::record::Report;
use ae_spatial::{Bounds, Circle, Grid, Hits, Nearest, Segment};
use anyhow::{Result, bail};

const MAP: f32 = 20_480.0;

/// `n` units in 40 squads scattered over the map, each squad a dense disc.
fn army(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut s = 0x005E_ED0F_5A7A_u64;
    let mut f = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let squads: Vec<(f32, f32)> = (0..40).map(|_| (f() * MAP, f() * MAP)).collect();
    let (mut x, mut y, mut r) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let (cx, cy) = squads[i % squads.len()];
        let (a, d) = (f() * std::f32::consts::TAU, f().sqrt() * 600.0);
        x.push(cx + d * a.cos());
        y.push(cy + d * a.sin());
        r.push(3.0 + f() * 3.0);
    }
    (x, y, r)
}

fn pool(threads: usize) -> Result<rayon::ThreadPool> {
    Ok(rayon::ThreadPoolBuilder::new().num_threads(threads).build()?)
}

pub fn run(options: &Options) -> Result<Report> {
    let mut report = Report::new("spatial");
    let n = if options.small { 10_000 } else { 50_000 };
    let q = n / 5;
    let tag = format!("units{n}_queries{q}");
    let (x, y, r) = army(n);
    let bounds = Bounds {
        min_x: 0.0,
        min_y: 0.0,
        max_x: MAP,
        max_y: MAP,
    };
    // Weapon-range queries from the first q units; sweeps as shots.
    let circles: Vec<Circle> = (0..q)
        .map(|i| Circle {
            x: x[i],
            y: y[i],
            r: 160.0,
        })
        .collect();
    let nearest: Vec<Nearest> = (0..q)
        .map(|i| Nearest {
            x: x[i],
            y: y[i],
            max_r: 400.0,
            k: 8,
        })
        .collect();
    let shots: Vec<Segment> = (0..q)
        .map(|i| Segment {
            x0: x[i],
            y0: y[i],
            x1: x[i] + 300.0,
            y1: y[i] - 200.0,
            pad: 1.0,
        })
        .collect();

    // Rebuild is a serial counting sort: timed once.
    let mut grid = Grid::new(bounds, 64.0);
    report.time(
        &format!("{tag}/rebuild"),
        time(options.budget, || grid.rebuild(&x, &y, &r)),
    );
    grid.rebuild(&x, &y, &r);
    let mut threads = vec![1];
    if options.threads > 1 {
        threads.push(options.threads);
    }
    let mut first: Option<(usize, [Hits; 3])> = None;
    for &t in &threads {
        let p = pool(t)?;
        let mut out = [Hits::default(), Hits::default(), Hits::default()];
        p.install(|| {
            report.time(
                &format!("{tag}/radius160_{t}t"),
                time(options.budget, || grid.radius(&circles, &mut out[0])),
            );
            report.time(
                &format!("{tag}/nearest8_{t}t"),
                time(options.budget, || grid.nearest(&nearest, &mut out[1])),
            );
            report.time(
                &format!("{tag}/sweep360_{t}t"),
                time(options.budget, || grid.sweep(&shots, &mut out[2])),
            );
        });
        match &first {
            None => {
                report.counter(&format!("{tag}/radius_hits"), out[0].len() as u64, "");
                report.counter(&format!("{tag}/nearest_hits"), out[1].len() as u64, "");
                report.counter(&format!("{tag}/sweep_hits"), out[2].len() as u64, "");
                first = Some((t, out));
            }
            Some((t0, base)) => {
                if base != &out {
                    bail!("spatial hits differ between {t0} and {t} thread(s)");
                }
            }
        }
    }
    Ok(report)
}
