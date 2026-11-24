//! Headless sim: runs a game module and reports digests and timings.
//!
//! ```text
//! ae-headless --game sandbox --units 50000 --ticks 3000 --workers 1,2,4,8,16 [--digest] [--every N] [--fuel] [--param-hi N]
//! ae-headless --game crossing --nav water [--map-cells 2560] --units 10000 --ticks 1200 --workers 1,8 [--compare]
//! ```
//! The game parameter each system sees is `units | param_hi << 32`.
//! `AE_TICK_DUMP=1` adds per-run tick statistics (odd vs even ticks, outliers).
//! `--game` is a directory under `games/` (built for wasm32-wasip1-threads)
//! or a `.wasm` path. Every worker count runs the same world from tick 0; the
//! final digests must be equal, or the exit status is 1.
//!
//! `--nav water|land` provides the engine's host systems (`ae_services`:
//! `nav_spawn`, `nav`, `spatial`) on that map, `--map-cells` cells square;
//! the nav service's pool gets as many threads as the run's sim workers.
//! `--teams N` gives the map's units allyteams and sensors
//! (`NavPlugin::teams`); `--observers K` then builds K allyteams' views after
//! every tick (`ae_services::view`) and prints the median capture +
//! coverage time and view build time (all K together and per observer).
//!
//! `--record FILE` writes a replay of one run (with `--script orders`,
//! synthetic player commands; with `--script bot`, the skirmish game's
//! bots play both allyteams (`ae_bot`); a checkpoint digest every
//! `--checkpoint-every` ticks, default 30); `--replay FILE --verify` plays
//! it back from the file alone and checks every checkpoint (see
//! `replay.rs`). `--bisect --inject TICK:Component.field:ROW` flips a bit
//! of one entity's field in one run and reports where that run first
//! differs from a clean one: tick, digest parts, entities and fields.
//! `AE_DIGEST_COST=1` times the world digest.
//!
//! `--content DIR` mounts a game's content: its catalogue (`DIR/defs/`)
//! spawns the game's `SpawnRequest`s, and its hash goes into replay and
//! lockstep headers. `--package SPEC|PATH` (with `--packages DIR`
//! libraries) mounts packages and their dependencies: a package's
//! `module.wasm` is the game and a map package's `map.json` the map, so
//! `--game` and `--nav` are not needed (see `setup.rs`).
//!
//! `--connect ADDR --player P` plays in lockstep with `ae-relay` (see
//! `client.rs`).
//!
//! `--compare` also runs the same plan through the standalone nav service
//! (`ae_nav::scenario`) at each worker count: its state hash must equal the
//! sim's nav digest, and both tick times are printed.

mod client;
mod replay;
mod setup;

use ae_nav::scenario::Scenario;
use ae_services::nav::{Map, NavPlugin};
use ae_sim::runtime::{Runtime, RuntimeConfig, TickStats};
use ae_wasm::Metering;
use anyhow::{Context, Result, bail};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: ae-headless --game NAME|PATH.wasm|--package SPEC|PATH... [--packages DIR] [--content DIR] [--units N] [--ticks N] [--workers 1,2,4,8] [--digest] [--every N] [--fuel] [--range-rows N] [--param-hi N] [--nav water|land] [--map-cells N] [--compare] [--overlap on|off] [--teams N] [--observers K]
       ae-headless --game NAME --nav water ... --record FILE.aerep [--script orders|bot] [--checkpoint-every N]
       ae-headless --game NAME --replay FILE.aerep [--verify] [--workers N]
       ae-headless --game NAME --replay FILE.aerep --bisect --inject TICK:Component.field:ROW
       ae-headless --game NAME --nav water ... --connect RELAY_ADDR --player P [--script orders|bot] [--latency MS] [--jitter MS] [--loss PCT]";

/// Who sends player commands (`--script`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Script {
    None,
    /// Synthetic move orders ([`replay::orders`]).
    Orders,
    /// The skirmish bots ([`ae_bot::SkirmishBot`]).
    Bot,
}

struct Opts {
    game: String,
    units: u64,
    ticks: u64,
    workers: Vec<usize>,
    digest: bool,
    every: u64,
    fuel: bool,
    teams: u32,
    observers: u32,
    range_rows: usize,
    /// Game-specific bits 32..64 of the game parameter (sandbox: wear percent).
    param_hi: u32,
    nav: Option<Map>,
    map_cells: i32,
    compare: bool,
    /// Run host systems beside the layer's module systems.
    overlap: bool,
    record: Option<String>,
    replay: Option<String>,
    verify: bool,
    /// `--script orders|bot`: player commands while recording or playing.
    script: Script,
    checkpoint_every: u64,
    bisect: bool,
    inject: Option<replay::Inject>,
    connect: Option<String>,
    /// A content directory: its catalogue spawns defs (`SpawnRequest`).
    /// `--content DIR` (repeatable): directories mounted over the packages.
    content: Vec<String>,
    /// `--package SPEC|PATH` (repeatable) and `--packages DIR` (libraries).
    packages: Vec<String>,
    package_dirs: Vec<String>,
    player: u32,
    latency: u64,
    jitter: u64,
    loss: f64,
    seed: u64,
}

fn parse() -> Result<Opts> {
    let mut o = Opts {
        game: String::new(),
        units: 10_000,
        ticks: 300,
        workers: vec![1],
        digest: false,
        every: 0,
        fuel: false,
        range_rows: 1024,
        param_hi: 0,
        nav: None,
        map_cells: 2560,
        compare: false,
        overlap: true,
        teams: 0,
        observers: 0,
        record: None,
        replay: None,
        verify: false,
        script: Script::None,
        checkpoint_every: 30,
        bisect: false,
        inject: None,
        connect: None,
        content: Vec::new(),
        packages: Vec::new(),
        package_dirs: Vec::new(),
        player: 0,
        latency: 0,
        jitter: 0,
        loss: 0.0,
        seed: 1,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--game" => o.game = value()?,
            "--units" => o.units = value()?.parse()?,
            "--ticks" => o.ticks = value()?.parse()?,
            "--workers" => {
                o.workers = value()?
                    .split(',')
                    .map(|w| w.trim().parse())
                    .collect::<Result<_, _>>()?
            }
            "--every" => o.every = value()?.parse()?,
            "--range-rows" => o.range_rows = value()?.parse()?,
            "--param-hi" => o.param_hi = value()?.parse()?,
            "--nav" => {
                o.nav = Some(match value()?.as_str() {
                    "water" => Map::WaterCrossing,
                    "land" => Map::LandCrossing,
                    "coast" => Map::Coast,
                    m => bail!("unknown map {m} (water, land or coast)"),
                })
            }
            "--map-cells" => o.map_cells = value()?.parse()?,
            "--compare" => o.compare = true,
            "--teams" => o.teams = value()?.parse()?,
            "--observers" => o.observers = value()?.parse()?,
            "--overlap" => {
                o.overlap = match value()?.as_str() {
                    "on" => true,
                    "off" => false,
                    v => bail!("--overlap on|off, not {v}"),
                }
            }
            "--record" => o.record = Some(value()?),
            "--replay" => o.replay = Some(value()?),
            "--verify" => o.verify = true,
            "--bisect" => o.bisect = true,
            "--connect" => o.connect = Some(value()?),
            "--content" => o.content.push(value()?),
            "--package" => o.packages.push(value()?),
            "--packages" => o.package_dirs.push(value()?),
            "--player" => o.player = value()?.parse()?,
            "--latency" => o.latency = value()?.parse()?,
            "--jitter" => o.jitter = value()?.parse()?,
            "--loss" => o.loss = value()?.parse::<f64>()? / 100.0,
            "--seed" => o.seed = value()?.parse()?,
            "--inject" => o.inject = Some(replay::Inject::parse(&value()?)?),
            "--checkpoint-every" => o.checkpoint_every = value()?.parse()?,
            "--script" => match value()?.as_str() {
                "orders" => o.script = Script::Orders,
                "bot" => o.script = Script::Bot,
                s => bail!("unknown script {s} (orders, bot)"),
            },
            "--digest" => o.digest = true,
            "--fuel" => o.fuel = true,
            _ => bail!("unknown argument {a}\n{USAGE}"),
        }
    }
    if (o.game.is_empty() && o.packages.is_empty() && o.replay.is_none())
        || o.workers.is_empty()
        || o.workers.contains(&0)
        || (o.compare && o.nav.is_none())
        || o.observers > o.teams
        || (o.teams > 0 && o.nav.is_none() && o.packages.is_empty() && o.content.is_empty())
    {
        bail!("{USAGE}");
    }
    Ok(o)
}

struct RunResult {
    workers: usize,
    digest: u128,
    wall_ms: f64,
    tick_ms: Vec<f64>,
    totals: TickStats,
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool> {
    let o = parse()?;
    if let Some(relay) = &o.connect {
        return client::run(&o, relay, o.player);
    }
    if let Some(path) = &o.record {
        return replay::record(&o, path);
    }
    if let Some(path) = &o.replay {
        if o.bisect {
            let inject = o
                .inject
                .as_ref()
                .context("--bisect compares a run with --inject TICK:Component.field:ROW against a clean one")?;
            return replay::bisect(&o, path, inject);
        }
        return replay::replay(&o, path, o.verify);
    }
    ae_profile::start();
    let setup = setup::Setup::new(&o, None)?;
    let wasm = &setup.wasm;
    let max = *o.workers.iter().max().unwrap();
    let cfg = RuntimeConfig {
        max_workers: max,
        range_rows: o.range_rows,
        metering: if o.fuel { Metering::Fuel } else { Metering::None },
        overlap_hosts: o.overlap,
        ..RuntimeConfig::default()
    };
    if let Some(c) = &setup.content {
        let packages = if c.packages.is_empty() {
            String::new()
        } else {
            format!(" (packages {})", c.packages.join(", "))
        };
        println!(
            "content{packages}: {} defs, hash {:032x}",
            c.catalog.as_ref().map_or(0, |c| c.defs.len()),
            c.hash
        );
    }
    if let Some(n) = setup.nav.as_ref().filter(|n| n.map == Map::Loaded) {
        println!("map from a package: {} cells a side", n.map_cells);
    }
    let extent = setup.nav.as_ref().map_or(0.0, |n| n.extent());
    let mut rt = Runtime::with_hosts(wasm, cfg, setup.plugins())?;
    println!(
        "game {}: {} components, {} systems in {} layers: {}",
        o.game,
        rt.schema.components.len(),
        rt.system_count(),
        rt.schedule.layers.len(),
        rt.schedule
            .layers
            .iter()
            .map(|l| l.iter().map(|&i| rt.system_name(i)).collect::<Vec<_>>().join(" "))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    let mut results: Vec<RunResult> = Vec::new();
    let mut same = true;
    for &w in &o.workers {
        let built = Instant::now();
        let mut world = rt.world_for(o.units | (o.param_hi as u64) << 32, w)?;
        if setup.nav.is_some() {
            println!(
                "  {w}w: world with nav service built in {:.2} s",
                built.elapsed().as_secs_f64()
            );
        }
        let mut totals = TickStats::default();
        let mut tick_ms = Vec::with_capacity(o.ticks as usize);
        let mut host_ms = Vec::with_capacity(o.ticks as usize);
        let nav_index = (0..rt.system_count()).find(|&i| rt.is_host_system(i) && rt.system_name(i) == "nav");
        let mut view = if o.observers > 0 {
            let pipeline = ae_services::view::ViewPipeline::new(&rt.schema, &["NavAgent"], o.teams as usize, extent)?;
            let observers: Vec<ae_view::Observer> = (0..o.observers)
                .map(|a| ae_view::Observer::new(a, 0x5EED + a as u64))
                .collect();
            Some((pipeline, observers, Vec::new(), Vec::new(), [0usize; 4]))
        } else {
            None
        };
        let mut view_ms = 0.0;
        let started = Instant::now();
        for t in 0..o.ticks {
            let t0 = Instant::now();
            let s = rt.tick(&mut world, w)?;
            tick_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            if let Some((pipeline, observers, capture_ms, build_ms, rows)) = view.as_mut() {
                let v0 = Instant::now();
                pipeline.capture(&rt, &world)?;
                let v1 = Instant::now();
                let tables = pipeline.observe(observers);
                let v2 = Instant::now();
                capture_ms.push((v1 - v0).as_secs_f64() * 1e3);
                build_ms.push((v2 - v1).as_secs_f64() * 1e3);
                *rows = [0; 4];
                for table in &tables {
                    for st in &table.state {
                        rows[*st as usize] += 1;
                    }
                }
                view_ms += (v2 - v0).as_secs_f64() * 1e3;
            }
            if let Some(i) = nav_index {
                host_ms.push(s.host_system_ns[i] as f64 / 1e6);
            }
            accumulate(&mut totals, &s);
            if o.every > 0 && (t + 1) % o.every == 0 {
                println!("  {w}w tick {:>6}: digest {:016x}", t + 1, rt.digest(&world)?.short());
            }
        }
        let wall_ms = started.elapsed().as_secs_f64() * 1e3 - view_ms;
        if std::env::var_os("AE_DIGEST_COST").is_some() {
            let t = Instant::now();
            for _ in 0..10 {
                std::hint::black_box(rt.digest(&world)?);
            }
            println!("  digest: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 10.0);
        }
        if let Some((_, observers, capture_ms, build_ms, rows)) = view.as_mut() {
            let median = |v: &mut Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v.get(v.len() / 2).copied().unwrap_or(0.0)
            };
            // Steady state: the second half of the run.
            let half = capture_ms.len() / 2;
            let (mut c, mut b) = (capture_ms.split_off(half), build_ms.split_off(half));
            let n = observers.len();
            println!(
                "  {w}w views ({} teams, {n} observers, last tick rows own/seen/blip/ghost {:?} summed): capture + coverage median {:.3} ms, build {:.3} ms ({:.3} ms per observer)",
                o.teams,
                rows,
                median(&mut c),
                median(&mut b),
                median(&mut b) / n as f64
            );
        }
        let digest = rt.digest(&world)?;
        let units = rt
            .count(&world, "Position")
            .or_else(|_| rt.count(&world, "Body"))
            .unwrap_or(0);
        let ticks = o.ticks.max(1) as f64;
        let mut sorted = tick_ms.clone();
        sorted.sort_by(f64::total_cmp);
        let pct = |p: f64| {
            sorted
                .get(((sorted.len().saturating_sub(1)) as f64 * p).round() as usize)
                .copied()
                .unwrap_or(0.0)
        };
        println!(
            "{w:>2} worker(s): digest {:032x}  {units} units  tick median {:.3} ms p99 {:.3} max {:.3}  systems {:.3} host {:.3} apply {:.3} prepare {:.3} ms/tick  commands {} events {}",
            digest.total,
            pct(0.5),
            pct(0.99),
            pct(1.0),
            totals.systems_ns as f64 / ticks / 1e6,
            totals.host_ns as f64 / ticks / 1e6,
            totals.apply_ns as f64 / ticks / 1e6,
            totals.prepare_ns as f64 / ticks / 1e6,
            totals.commands,
            totals.events,
        );
        if o.digest {
            for (name, hash, rows) in &digest.parts {
                println!("    {name:<12} {rows:>8} rows  {:032x}", hash);
            }
        }
        for i in (0..rt.system_count()).filter(|&i| rt.is_host_system(i)) {
            println!(
                "    host {:<10} {:>8.3} ms/tick",
                rt.system_name(i),
                totals.host_system_ns[i] as f64 / ticks / 1e6
            );
        }
        if o.fuel {
            for i in 0..rt.system_count() {
                println!("    fuel {:<12} {:>14}", rt.system_name(i), totals.fuel[i]);
            }
        }
        if std::env::var_os("AE_TICK_DUMP").is_some() {
            let even: Vec<f64> = tick_ms.iter().skip(1).step_by(2).copied().collect();
            let odd: Vec<f64> = tick_ms.iter().skip(2).step_by(2).copied().collect();
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
            let over: usize = tick_ms.iter().skip(1).filter(|&&t| t > 2.0 * pct(0.5)).count();
            println!(
                "    ticks 1,3,5..: mean {:.3}  ticks 2,4,..: mean {:.3}  p90 {:.3}  ticks over 2x median: {over}  sum of those {:.1} ms",
                mean(&even),
                mean(&odd),
                pct(0.9),
                tick_ms.iter().skip(1).filter(|&&t| t > 2.0 * pct(0.5)).sum::<f64>()
            );
        }
        if o.compare {
            let nav = rt.host_digest(&world, "nav").map(|h| h as u64);
            drop(world);
            let (same_nav, standalone) = compare(&o, w, nav, &rt.schema)?;
            same &= same_nav;
            let med = |v: &[f64]| {
                let mut s = v[1.min(v.len())..].to_vec();
                s.sort_by(f64::total_cmp);
                s.get(s.len() / 2).copied().unwrap_or(0.0)
            };
            println!(
                "    vs standalone nav service: tick median {:.3} ms in the sim (of which the nav system {:.3}), {:.3} ms standalone; nav state {}",
                med(&tick_ms),
                med(&host_ms),
                med(&standalone),
                if same_nav { "equal" } else { "DIFFERS" }
            );
        }
        results.push(RunResult {
            workers: w,
            digest: digest.total,
            wall_ms,
            tick_ms,
            totals,
        });
    }
    let base = &results[0];
    for r in &results[1..] {
        if r.digest != base.digest {
            same = false;
            println!(
                "DIGEST MISMATCH: {} worker(s) {:032x} vs {} worker(s) {:032x}",
                r.workers, r.digest, base.workers, base.digest
            );
        }
    }
    if results.len() > 1 {
        let one = results.iter().find(|r| r.workers == 1).unwrap_or(base);
        for r in &results {
            let sys = one.totals.systems_ns as f64 / r.totals.systems_ns.max(1) as f64;
            let med = |v: &[f64]| {
                let mut s = v.to_vec();
                s.sort_by(f64::total_cmp);
                s[s.len() / 2]
            };
            let steady = |v: &[f64]| v[1.min(v.len() - 1)..].iter().sum::<f64>() / (v.len().max(2) - 1) as f64;
            println!(
                "  {:>2}w: speedup vs 1w: whole run {:.2}x, mean tick after setup {:.2}x ({:.3} ms), median tick {:.2}x, system phase {:.2}x; setup tick {:.1} ms",
                r.workers,
                one.wall_ms / r.wall_ms,
                steady(&one.tick_ms) / steady(&r.tick_ms),
                steady(&r.tick_ms),
                med(&one.tick_ms) / med(&r.tick_ms),
                sys,
                r.tick_ms[0]
            );
        }
        println!(
            "{}",
            if same {
                "digests equal at every worker count"
            } else {
                "DIGESTS DIFFER"
            }
        );
    }
    Ok(same)
}

/// The same plan through the standalone nav service at `w` threads: whether
/// its final state hash equals the sim's nav digest, and its tick times (ms).
fn compare(o: &Opts, w: usize, sim_nav: Option<u64>, schema: &ae_sim::schema::Schema) -> Result<(bool, Vec<f64>)> {
    let plugin = NavPlugin::new(o.nav.unwrap(), o.map_cells);
    let (nav, plan) = plugin.plan(o.units as u32, w, plugin.profiles(schema)?);
    let mut s = Scenario::from_plan(nav, &plan);
    let mut ms = Vec::with_capacity(o.ticks as usize);
    // The sim's first tick is the scenario's first step only when orders
    // are given at tick 0.
    if o.param_hi != 0 {
        bail!("--compare runs the map's plan as is: orders at tick 0, nothing else (--param-hi 0)");
    }
    let mut last = Instant::now();
    s.run(o.ticks, |_, _, _| {
        let now = Instant::now();
        ms.push((now - last).as_secs_f64() * 1e3);
        last = now;
    });
    let hash = s.nav.state_hash();
    if sim_nav != Some(hash) {
        println!("NAV MISMATCH: standalone {hash:016x}, sim {sim_nav:x?}");
    }
    Ok((sim_nav == Some(hash), ms))
}

fn accumulate(t: &mut TickStats, s: &TickStats) {
    t.systems_ns += s.systems_ns;
    t.host_ns += s.host_ns;
    t.apply_ns += s.apply_ns;
    t.prepare_ns += s.prepare_ns;
    t.calls += s.calls;
    t.commands += s.commands;
    t.events += s.events;
    if t.fuel.len() < s.fuel.len() {
        t.fuel.resize(s.fuel.len(), 0);
    }
    for (a, b) in t.fuel.iter_mut().zip(&s.fuel) {
        *a += b;
    }
    if t.host_system_ns.len() < s.host_system_ns.len() {
        t.host_system_ns.resize(s.host_system_ns.len(), 0);
    }
    for (a, b) in t.host_system_ns.iter_mut().zip(&s.host_system_ns) {
        *a += b;
    }
}
