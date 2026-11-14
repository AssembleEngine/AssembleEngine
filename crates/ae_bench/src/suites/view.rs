//! Copy layer rule executor (`ae_view`): view tables for observers of a
//! synthetic 10k / 50k army (the `sensors` suite's squads, 8 fields of every
//! class) on the 20,480 m map. Times the frame preparation (rows by
//! entity, once per tick), one observer's build, and a tick for two
//! observers (preparation + both builds; the M4 exit asks under 2 ms at
//! 10k), with the army moving between ticks so ids, blips and ghosts are
//! in steady churn. The `_contact` case mixes the allyteams inside every
//! squad, so nearly every enemy is in sight, as on the crossing map. Rows
//! per state are exact counters.

use super::Options;
use crate::measure::time;
use crate::record::Report;
use ae_sensors::{Coverage, SensorGrid, Sensors};
use ae_view::{Frame, Observer, ViewClass, ViewSpec};
use anyhow::Result;

const MAP: f32 = 20_480.0;

struct Army {
    entities: Vec<u64>,
    columns: Vec<Vec<u32>>,
    ally: Vec<u32>,
    los: Vec<f32>,
    radar: Vec<f32>,
}

/// `mixed`: allyteams alternate unit by unit inside every squad (armies
/// in contact: most enemies in sight); otherwise squad by squad.
fn army(n: usize, allies: u32, mixed: bool) -> Army {
    let mut s = 0x0F1E_3D7C_u64;
    let mut f = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let squads: Vec<(f32, f32)> = (0..40).map(|_| (f() * MAP, f() * MAP)).collect();
    let mut a = Army { entities: vec![], columns: vec![vec![]; 8], ally: vec![], los: vec![], radar: vec![] };
    for i in 0..n {
        let squad = i % squads.len();
        let (cx, cy) = squads[squad];
        let (t, d) = (f() * std::f32::consts::TAU, f().sqrt() * 900.0);
        let row = [(cx + d * t.cos()).to_bits(), (cy + d * t.sin()).to_bits(), 0.5f32.to_bits(), 0.25f32.to_bits(), i as u32, 100, 7, 3];
        for (c, w) in a.columns.iter_mut().zip(row) {
            c.push(w);
        }
        a.entities.push((1u64 << 32) | (0x1000 + i as u64 * 3));
        // (i / 40: the unit's place in its squad; i % allies would equal squad % allies.)
        a.ally.push(if mixed { (i / squads.len()) as u32 % allies } else { squad as u32 % allies });
        a.los.push(250.0 + f() * 400.0);
        a.radar.push(if i % 23 == 0 { 2000.0 } else { 0.0 });
    }
    a
}

/// Move everyone a little (deterministic): x += vx, y += vy.
fn step(a: &mut Army) {
    for r in 0..a.entities.len() {
        let (vx, vy) = (f32::from_bits(a.columns[2][r]), f32::from_bits(a.columns[3][r]));
        a.columns[0][r] = (f32::from_bits(a.columns[0][r]) + vx * 8.0).rem_euclid(MAP).to_bits();
        a.columns[1][r] = (f32::from_bits(a.columns[1][r]) + vy * 8.0).rem_euclid(MAP).to_bits();
    }
}

fn spec(ghosting: bool) -> ViewSpec {
    let c = |name: &str, class| (name.to_string(), class);
    ViewSpec {
        fields: vec![
            c("x", ViewClass::RadarDegraded),
            c("y", ViewClass::RadarDegraded),
            c("vx", ViewClass::Los),
            c("vy", ViewClass::Los),
            c("kind", if ghosting { ViewClass::Ghost } else { ViewClass::Los }),
            c("hp", ViewClass::Los),
            c("orders", ViewClass::OwnerOnly),
            c("secret", ViewClass::Never),
        ],
        x: 0,
        y: 1,
        wobble: 60.0,
        wobble_ticks: 30,
    }
}

pub fn run(options: &Options) -> Result<Report> {
    let mut report = Report::new("view");
    let grid = SensorGrid::new(MAP, MAP, 32.0);
    for (n, allies, ghosting, mixed) in [(10_000, 2, false, false), (10_000, 2, true, false), (10_000, 2, false, true), (50_000, 8, false, false)] {
        if options.small && n > 10_000 {
            continue;
        }
        let tag = format!("units{n}_allies{allies}{}{}", if ghosting { "_ghosts" } else { "" }, if mixed { "_contact" } else { "" });
        let spec = spec(ghosting);
        let mut a = army(n, allies, mixed);
        let mut coverage = Coverage::new(grid, allies as usize);
        let mut observers = [Observer::new(0, 11), Observer::new(1, 22)];
        let mut tick = 0u64;
        let mut counts = [0u64; 4];
        // Warm up: ids and ghosts in steady state.
        for _ in 0..30 {
            step(&mut a);
            tick += 1;
            coverage.stamp(&Sensors { x: &col(&a, 0), y: &col(&a, 1), ally: &a.ally, los: &a.los, radar: &a.radar }, true);
            let refs: Vec<&[u32]> = a.columns.iter().map(Vec::as_slice).collect();
            let frame = Frame::new(tick, &a.entities, &refs, &a.ally, &coverage);
            for o in &mut observers {
                let table = o.build(&spec, &frame);
                counts = [0; 4];
                for s in &table.state {
                    counts[*s as usize] += 1;
                }
            }
        }
        let refs: Vec<&[u32]> = a.columns.iter().map(Vec::as_slice).collect();
        report.time(&format!("{tag}/frame_prepare"), time(options.budget, || Frame::new(tick, &a.entities, &refs, &a.ally, &coverage).tick));
        let frame = Frame::new(tick, &a.entities, &refs, &a.ally, &coverage);
        let mut one = observers[0].clone();
        report.time(&format!("{tag}/build_1_observer"), time(options.budget, || one.build(&spec, &frame).len()));
        let mut two = observers.clone();
        report.time(
            &format!("{tag}/tick_2_observers"),
            time(options.budget, || {
                let frame = Frame::new(tick, &a.entities, &refs, &a.ally, &coverage);
                two[0].build(&spec, &frame).len() + two[1].build(&spec, &frame).len()
            }),
        );
        for (state, count) in ["own", "seen", "blip", "ghost"].iter().zip(counts) {
            report.counter(&format!("{tag}/rows_{state}_observer1"), count, "");
        }

    }
    Ok(report)
}

fn col(a: &Army, c: usize) -> Vec<f32> {
    a.columns[c].iter().map(|w| f32::from_bits(*w)).collect()
}
