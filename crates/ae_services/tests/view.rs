//! The copy layer on the crossing game with teams: two observers' views of
//! the live sim, checked against the capture they came from.
//!
//! * Viewing changes nothing: the digest equals a run without it, and
//!   with teams the world is the same at 1 and 2 workers.
//! * Each observer's `Own` rows are exactly its allyteam's units (all
//!   fields, `NavAgent.mode` too); every other row is an enemy unit, `Seen`
//!   only where its cell is in the observer's line of sight and `Blip`
//!   only on radar, without `Los` and `OwnerOnly` fields.
//! * The observers see different worlds, and no word of any table lies in
//!   the range of entity indices (the viewed fields are small enums and
//!   normal floats, so one there would be a leaked id).
//!
//! The column arena is process-wide, so this file holds one test.

use ae_services::nav::{Map, NavPlugin};
use ae_services::spatial::SpatialPlugin;
use ae_services::view::ViewPipeline;
use ae_sim::runtime::{Runtime, RuntimeConfig};
use ae_view::{Observer, RowState};
use anyhow::{Result, ensure};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

const UNITS: u64 = 3000;
const MAP_CELLS: i32 = 512;
const TICKS: u64 = 90;

#[test]
fn observers_see_their_own_and_what_their_sensors_reach() -> Result<()> {
    let wasm = ae_sim::build::game("crossing")?;
    let nav = NavPlugin { teams: 2, ..NavPlugin::new(Map::WaterCrossing, MAP_CELLS) };
    let spatial = SpatialPlugin::square(nav.extent(), 64.0);
    let run = |workers: usize, view: bool| -> Result<(u128, Vec<String>, [usize; 4])> {
        let cfg = RuntimeConfig { max_workers: workers, range_rows: 128, parallel_min_rows: 256, ..RuntimeConfig::default() };
        let mut rt = Runtime::with_hosts(&wasm, cfg, vec![Arc::new(nav.clone()), Arc::new(spatial)])?;
        let mut world = rt.world_for(UNITS, workers)?;
        let mut pipeline = ViewPipeline::new(&rt.schema, &["NavAgent"], 2, nav.extent())?;
        let mut observers = [Observer::new(0, 0xA11CE), Observer::new(1, 0xB0B)];
        let mut notes = Vec::new();
        let mut totals = [0usize; 4];
        for _ in 0..TICKS {
            rt.tick(&mut world, workers)?;
            if !view || world.tick % 15 != 0 {
                continue;
            }
            pipeline.capture(&rt, &world)?;
            let started = Instant::now();
            let tables = pipeline.observe(&mut observers);
            let built = started.elapsed();
            let capture = pipeline.captured();
            ensure!(capture.len() == UNITS as usize, "every unit has Body, Team, Sensor, NavAgent");
            // Entity indices lie in [16, 2^23); the viewed fields are small
            // enums (< 16) or normal floats, never in that range, so a word
            // there would be an id that leaked.
            const IDS: std::ops::Range<u32> = 16..1 << 23;
            ensure!(capture.entities.iter().all(|e| IDS.contains(&(*e as u32))), "entity indices outside the range this check assumes");
            let (x, y, ally) = (pipeline.spec.x, pipeline.spec.y, pipeline.field("Team", "ally").unwrap());
            let mode = pipeline.field("NavAgent", "mode").unwrap();
            let profile = pipeline.field("NavAgent", "profile").unwrap();
            let mut counts = Vec::new();
            for (o, table) in tables.iter().enumerate() {
                let own: BTreeSet<(u32, u32, u32)> = (0..capture.len()).filter(|&r| capture.columns[ally][r] == o as u32).map(|r| (capture.columns[x][r], capture.columns[y][r], capture.columns[mode][r])).collect();
                let enemy: BTreeSet<(u32, u32)> = (0..capture.len()).filter(|&r| capture.columns[ally][r] != o as u32).map(|r| (capture.columns[x][r], capture.columns[y][r])).collect();
                let (cx, cy, cm, cp) = (table.column(x).unwrap(), table.column(y).unwrap(), table.column(mode).unwrap(), table.column(profile).unwrap());
                let mut own_rows = BTreeSet::new();
                let mut n = [0usize; 4];
                for r in 0..table.len() {
                    n[table.state[r] as usize] += 1;
                    match table.state[r] {
                        RowState::Own => {
                            own_rows.insert((cx[r], cy[r], cm[r]));
                        }
                        RowState::Seen => {
                            ensure!(enemy.contains(&(cx[r], cy[r])), "tick {}: a Seen row is no enemy unit", world.tick);
                            ensure!(pipeline.coverage.sees(o, f32::from_bits(cx[r]), f32::from_bits(cy[r])).0, "Seen outside line of sight");
                            ensure!(cm[r] == 0, "an enemy's mode (owner only) leaked");
                        }
                        RowState::Blip => ensure!(cm[r] == 0 && cp[r] == 0, "a blip shows line-of-sight or owner fields"),
                        RowState::Ghost => anyhow::bail!("no ghosting fields in this table"),
                    }
                }
                ensure!(own_rows == own, "observer {o}: Own rows are not its allyteam's units");
                for col in &table.columns {
                    ensure!(col.iter().all(|w| !IDS.contains(w)), "a word in the entity id range reached a view");
                }
                for (t, c) in totals.iter_mut().zip(n) {
                    *t += c;
                }
                counts.push(n);
            }
            ensure!(tables[0].len() != tables[1].len() || tables[0].columns != tables[1].columns, "the two observers see the same world");
            notes.push(format!("tick {}: own/seen/blip {:?} | {:?}, two views built in {:.3} ms", world.tick, &counts[0][..3], &counts[1][..3], built.as_secs_f64() * 1e3));
        }
        Ok((rt.digest(&world)?.total, notes, totals))
    };
    let (plain, ..) = run(2, false)?;
    let (viewed, notes, totals) = run(2, true)?;
    let (one, ..) = run(1, false)?;
    ensure!(plain == viewed, "viewing changed the world");
    ensure!(plain == one, "teams: 1 and 2 workers differ");
    for n in &notes {
        println!("{n}");
    }
    ensure!(totals[1] > 0 && totals[2] > 0, "rows own/seen/blip/ghost over the run: {totals:?}");
    Ok(())
}
