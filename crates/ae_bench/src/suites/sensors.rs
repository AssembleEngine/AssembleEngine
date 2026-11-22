//! Sensor coverage (`ae_sensors`): line-of-sight and radar discs of a
//! synthetic army (squads like `spatial`'s, split between allyteams) on the
//! 20,480 m crossing map at 32 m cells, serial and with allyteams in
//! parallel. Stamped cells and covered cells are exact counters; the bits
//! must be equal serial and parallel, or the suite fails.

use super::Options;
use crate::measure::time;
use crate::record::Report;
use ae_sensors::{Coverage, SensorGrid, Sensors};
use anyhow::{Result, bail};

const MAP: f32 = 20_480.0;
const CELL: f32 = 32.0;

struct Army {
    x: Vec<f32>,
    y: Vec<f32>,
    ally: Vec<u32>,
    los: Vec<f32>,
    radar: Vec<f32>,
}

/// `n` units in 40 squads; squad `s` belongs to allyteam `s % allies`.
/// Line of sight 250..650 m; one unit in 23 has a 2,000 m radar (23: coprime with the 40 squads, so every allyteam gets some).
fn army(n: usize, allies: u32) -> Army {
    let mut s = 0x5E45_0125_u64;
    let mut f = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let squads: Vec<(f32, f32)> = (0..40).map(|_| (f() * MAP, f() * MAP)).collect();
    let mut a = Army {
        x: vec![],
        y: vec![],
        ally: vec![],
        los: vec![],
        radar: vec![],
    };
    for i in 0..n {
        let squad = i % squads.len();
        let (cx, cy) = squads[squad];
        let (t, d) = (f() * std::f32::consts::TAU, f().sqrt() * 600.0);
        a.x.push(cx + d * t.cos());
        a.y.push(cy + d * t.sin());
        a.ally.push(squad as u32 % allies);
        a.los.push(250.0 + f() * 400.0);
        a.radar.push(if i % 23 == 0 { 2000.0 } else { 0.0 });
    }
    a
}

pub fn run(options: &Options) -> Result<Report> {
    let mut report = Report::new("sensors");
    let grid = SensorGrid::new(MAP, MAP, CELL);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.threads.max(1))
        .build()?;
    for (n, allies) in [(10_000, 2), (10_000, 8), (50_000, 8)] {
        if options.small && n > 10_000 {
            continue;
        }
        let tag = format!("units{n}_allies{allies}");
        let a = army(n, allies);
        let units = Sensors {
            x: &a.x,
            y: &a.y,
            ally: &a.ally,
            los: &a.los,
            radar: &a.radar,
        };
        let mut serial = Coverage::new(grid, allies as usize);
        report.time(
            &format!("{tag}/stamp_1t"),
            time(options.budget, || serial.stamp(&units, false)),
        );
        serial.stamp(&units, false);
        let mut parallel = Coverage::new(grid, allies as usize);
        pool.install(|| {
            report.time(
                &format!("{tag}/stamp_{}t", options.threads),
                time(options.budget, || parallel.stamp(&units, true)),
            )
        });
        pool.install(|| parallel.stamp(&units, true));
        if serial != parallel {
            bail!("{tag}: coverage differs serial and parallel");
        }
        report.counter(&format!("{tag}/stamped_cells"), serial.stamped, "");
        report.info(
            &format!("{tag}/los_cells"),
            serial.los.iter().map(|b| b.count()).sum(),
            "",
        );
        report.info(
            &format!("{tag}/radar_cells"),
            serial.radar.iter().map(|b| b.count()).sum(),
            "",
        );
    }
    report.info("grid_cells", grid.width as u64 * grid.height as u64, "");
    Ok(report)
}
