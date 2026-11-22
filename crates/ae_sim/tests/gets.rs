//! Other entities' components read from a system (`get`, M14 step 6, C) on
//! the `gets_guest` module: `advance` writes every unit's `Position` and
//! `read` gets its target's `Position`, which the scheduler puts in a later
//! layer, so `read` sees this tick's writes.
//!
//! Checked at 8, 1, 2 and 4 workers:
//! * every reading row's `Out.x` is its target's `Position.x` of the same
//!   tick (the value `advance` wrote, not the tick's start);
//! * targets despawned by `vanish` give -1 and its `none` flag; live ones 0;
//! * the world digests are equal.
//!
//! Then a module whose system gets a component it writes is rejected when
//! it loads (`gets_bad_guest`).
//!
//! One runtime at a time: the column arena is process-wide.

use ae_sim::runtime::{Runtime, RuntimeConfig};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;
use std::path::Path;

const UNITS: u64 = 64;
const TICKS: u64 = 12;

fn decode<T: Copy>(bytes: &[u8]) -> T {
    assert_eq!(bytes.len(), size_of::<T>());
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}

#[derive(Clone, Copy, Debug)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug)]
struct Link {
    target: u64,
}

#[derive(Clone, Copy, Debug)]
struct Out {
    x: f32,
    none: u32,
}

fn build(name: &str) -> Result<Vec<u8>> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/{name}/Cargo.toml"));
    ae_sim::build::build_guest(&manifest, &format!("ae_sim_tests_{name}"))
}

#[test]
fn systems_read_other_entities_components_the_same_at_any_worker_count() -> Result<()> {
    let wasm = build("gets_guest")?;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let max = cores.clamp(1, 8);
    let cfg = RuntimeConfig {
        max_workers: max,
        range_rows: 16,
        parallel_min_rows: 1,
        ..RuntimeConfig::default()
    };
    let mut rt = Runtime::new(&wasm, cfg)?;
    let mut order = vec![max];
    order.extend([1, 2, 4, 8].into_iter().filter(|&w| w < max));
    let mut base: Option<(usize, u128)> = None;
    for &w in &order {
        let mut world = rt.world_for(UNITS, w)?;
        let mut none_seen = 0;
        for t in 1..=TICKS {
            rt.tick(&mut world, w)?;
            ensure!(world.tick == t);
            let rows = |name: &str| -> Result<Vec<(u64, Vec<u8>)>> { rt.rows(&world, name) };
            let positions: BTreeMap<u64, Position> =
                rows("Position")?.into_iter().map(|(e, b)| (e, decode(&b))).collect();
            let links: BTreeMap<u64, Link> = rows("Link")?.into_iter().map(|(e, b)| (e, decode(&b))).collect();
            let outs: BTreeMap<u64, Out> = rows("Out")?.into_iter().map(|(e, b)| (e, decode(&b))).collect();
            ensure!(outs.len() == links.len() && links.len() == positions.len());
            // `link` first sees the spawns at run tick 1 (the world's tick
            // 2): before that every target is 0, which is no entity.
            if t == 1 {
                continue;
            }
            for (&entity, link) in &links {
                let out = &outs[&entity];
                match positions.get(&link.target) {
                    Some(p) => {
                        ensure!(
                            p.y == 0.0 && out.none == 0 && out.x == p.x,
                            "{w} workers, tick {t}: {entity} reads {:?} for {}, its target has {p:?}",
                            out.x,
                            link.target
                        );
                    }
                    None => {
                        ensure!(
                            out.none == 1 && out.x == -1.0,
                            "{w} workers, tick {t}: {entity} gets {} (gone) as {out:?}",
                            link.target
                        );
                        none_seen += 1;
                    }
                }
            }
            // The values are the ones `advance` wrote this tick, not the
            // tick's start (a reader that ran first would be one behind).
        }
        ensure!(none_seen > 0, "{w} workers: no despawned target was read");
        let digest = rt.digest(&world)?.total;
        match &base {
            None => base = Some((w, digest)),
            Some((w0, d0)) => ensure!(
                digest == *d0,
                "{w} workers: digest {digest:016x}, {w0} workers {d0:016x}"
            ),
        }
        eprintln!("{w} worker(s): digest {digest:016x}, {none_seen} reads of despawned units");
    }
    Ok(())
}

#[test]
fn a_get_of_a_written_component_is_rejected_at_load() -> Result<()> {
    let wasm = build("gets_bad_guest")?;
    let err = Runtime::new(&wasm, RuntimeConfig::default())
        .err()
        .expect("the module must be rejected");
    let text = format!("{err:#}");
    ensure!(text.contains("which it writes"), "{text}");
    Ok(())
}
