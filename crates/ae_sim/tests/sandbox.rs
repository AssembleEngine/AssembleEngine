//! The sandbox game (`games/sandbox`) through the whole runtime: schema
//! section, schedule layers, worker crew, commands, events and digest.
//!
//! The column arena is process-wide, so this file holds one test that owns
//! the process's runtime and runs every world on it.

use ae_sim::runtime::{Runtime, RuntimeConfig};
use anyhow::{Result, ensure};

const UNITS: u64 = 4000;
const TICKS: u64 = 240;
/// Stress churn (units live about 6 s), so deaths, spawns and events
/// happen every wear tick.
const STRESS: u64 = 500 << 32;

/// Final digest (low 64 bits) of the stress run, identical in dev and
/// release builds and at every worker count. Regenerate deliberately: run
/// with `--nocapture` and copy the printed value.
const EXPECTED: u64 = 0x55e3_89f5_6787_ecdd;

struct Outcome {
    digest: u128,
    units: usize,
    deaths: u64,
    events: u64,
    commands: u64,
    held: usize,
}

fn run(rt: &mut Runtime, workers: usize, param: u64) -> Result<Outcome> {
    let mut world = rt.world(param)?;
    let (mut events, mut commands) = (0, 0);
    for _ in 0..TICKS {
        let s = rt.tick(&mut world, workers)?;
        events += s.events;
        commands += s.commands;
    }
    let census = rt.rows(&world, "Census")?;
    ensure!(census.len() == 1, "one census singleton, found {}", census.len());
    let deaths = u64::from_le_bytes(census[0].1[..8].try_into().unwrap());
    Ok(Outcome { digest: rt.digest(&world)?.total, units: rt.count(&world, "Position")?, deaths, events, commands, held: rt.events_held(&world) })
}

#[test]
fn sandbox_is_identical_at_every_worker_count() -> Result<()> {
    let wasm = ae_sim::build::game("sandbox")?;
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 8);
    // Small ranges so even 4k units spread over many jobs.
    let cfg = RuntimeConfig { max_workers: workers, range_rows: 256, parallel_min_rows: 512, ..RuntimeConfig::default() };
    let mut rt = Runtime::new(&wasm, cfg)?;

    let names = |layer: &Vec<usize>| layer.iter().map(|&i| rt.schema.systems[i].name.clone()).collect::<Vec<_>>();
    let layers: Vec<Vec<String>> = rt.schedule.layers.iter().map(names).collect();
    assert_eq!(layers, [vec!["setup", "think"], vec!["integrate"], vec!["wear"], vec!["census"]]);

    let base = run(&mut rt, 1, UNITS | STRESS)?;
    eprintln!(
        "1 worker: digest {:032x} ({:016x}), {} units, {} deaths, {} events, {} commands",
        base.digest, base.digest as u64, base.units, base.deaths, base.events, base.commands
    );
    ensure!(base.units == UNITS as usize, "every death is replaced: {} units", base.units);
    ensure!(base.deaths > 500, "the stress run should kill units, {} died", base.deaths);
    ensure!(base.deaths == base.events, "census counted {} deaths from {} events", base.deaths, base.events);
    ensure!(base.held == 0, "census reads every event the tick it is emitted; {} held", base.held);
    ensure!(base.commands >= 2 * base.deaths, "each death despawns and spawns");
    ensure!(base.digest as u64 == EXPECTED, "digest {:016x} differs from the pinned {EXPECTED:016x}", base.digest as u64);

    for w in [2, 3, 4, 8].into_iter().filter(|&w| w <= workers) {
        let r = run(&mut rt, w, UNITS | STRESS)?;
        ensure!(r.digest == base.digest, "{w} workers: digest {:032x}, 1 worker {:032x}", r.digest, base.digest);
        ensure!((r.deaths, r.events, r.commands) == (base.deaths, base.events, base.commands), "{w} workers: different counts");
    }

    // The digest sees the parameter: default churn gives another world.
    let calm = run(&mut rt, workers, UNITS)?;
    ensure!(calm.digest != base.digest && calm.deaths < base.deaths / 4, "default churn should differ from stress");
    Ok(())
}
