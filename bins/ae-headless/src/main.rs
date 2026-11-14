//! Headless sim: runs a game module and reports digests and timings.
//!
//! ```text
//! ae-headless --game sandbox --units 50000 --ticks 3000 --workers 1,2,4,8,16 [--digest] [--every N] [--fuel] [--param-hi N]
//! ```
//! The game parameter each system sees is `units | param_hi << 32`.
//! `AE_TICK_DUMP=1` adds per-run tick statistics (odd vs even ticks, outliers).
//! `--game` is a directory under `games/` (built for wasm32-wasip1-threads)
//! or a `.wasm` path. Every worker count runs the same world from tick 0; the
//! final digests must be equal, or the exit status is 1.

use ae_sim::runtime::{Runtime, RuntimeConfig, TickStats};
use ae_wasm::Metering;
use anyhow::{Context, Result, bail};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: ae-headless --game NAME|PATH.wasm [--units N] [--ticks N] [--workers 1,2,4,8] [--digest] [--every N] [--fuel] [--range-rows N] [--param-hi N]";

struct Opts {
    game: String,
    units: u64,
    ticks: u64,
    workers: Vec<usize>,
    digest: bool,
    every: u64,
    fuel: bool,
    range_rows: usize,
    /// Game-specific bits 32..64 of the game parameter (sandbox: wear percent).
    param_hi: u32,
}

fn parse() -> Result<Opts> {
    let mut o = Opts { game: String::new(), units: 10_000, ticks: 300, workers: vec![1], digest: false, every: 0, fuel: false, range_rows: 1024, param_hi: 0 };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--game" => o.game = value()?,
            "--units" => o.units = value()?.parse()?,
            "--ticks" => o.ticks = value()?.parse()?,
            "--workers" => o.workers = value()?.split(',').map(|w| w.trim().parse()).collect::<Result<_, _>>()?,
            "--every" => o.every = value()?.parse()?,
            "--range-rows" => o.range_rows = value()?.parse()?,
            "--param-hi" => o.param_hi = value()?.parse()?,
            "--digest" => o.digest = true,
            "--fuel" => o.fuel = true,
            _ => bail!("unknown argument {a}\n{USAGE}"),
        }
    }
    if o.game.is_empty() || o.workers.is_empty() || o.workers.contains(&0) {
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
    ae_profile::start();
    let wasm = ae_sim::build::game(&o.game)?;
    let max = *o.workers.iter().max().unwrap();
    let cfg = RuntimeConfig {
        max_workers: max,
        range_rows: o.range_rows,
        metering: if o.fuel { Metering::Fuel } else { Metering::None },
        ..RuntimeConfig::default()
    };
    let mut rt = Runtime::new(&wasm, cfg)?;
    println!(
        "game {}: {} components, {} systems in {} layers: {}",
        o.game,
        rt.schema.components.len(),
        rt.schema.systems.len(),
        rt.schedule.layers.len(),
        rt.schedule.layers.iter().map(|l| l.iter().map(|&i| rt.schema.systems[i].name.as_str()).collect::<Vec<_>>().join(" ")).collect::<Vec<_>>().join(" | ")
    );
    let mut results: Vec<RunResult> = Vec::new();
    for &w in &o.workers {
        let mut world = rt.world(o.units | (o.param_hi as u64) << 32)?;
        let mut totals = TickStats::default();
        let mut tick_ms = Vec::with_capacity(o.ticks as usize);
        let started = Instant::now();
        for t in 0..o.ticks {
            let t0 = Instant::now();
            let s = rt.tick(&mut world, w)?;
            tick_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            accumulate(&mut totals, &s);
            if o.every > 0 && (t + 1) % o.every == 0 {
                println!("  {w}w tick {:>6}: digest {:016x}", t + 1, rt.digest(&world)?.short());
            }
        }
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let digest = rt.digest(&world)?;
        let units = rt.count(&world, "Position").unwrap_or(0);
        let ticks = o.ticks.max(1) as f64;
        let mut sorted = tick_ms.clone();
        sorted.sort_by(f64::total_cmp);
        let pct = |p: f64| sorted.get(((sorted.len().saturating_sub(1)) as f64 * p).round() as usize).copied().unwrap_or(0.0);
        println!(
            "{w:>2} worker(s): digest {:032x}  {units} units  tick median {:.3} ms p99 {:.3} max {:.3}  systems {:.3} apply {:.3} prepare {:.3} ms/tick  commands {} events {}",
            digest.total,
            pct(0.5),
            pct(0.99),
            pct(1.0),
            totals.systems_ns as f64 / ticks / 1e6,
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
        if o.fuel {
            for (i, s) in rt.schema.systems.iter().enumerate() {
                println!("    fuel {:<12} {:>14}", s.name, totals.fuel[i]);
            }
        }
        if std::env::var_os("AE_TICK_DUMP").is_some() {
            let even: Vec<f64> = tick_ms.iter().skip(1).step_by(2).copied().collect();
            let odd: Vec<f64> = tick_ms.iter().skip(2).step_by(2).copied().collect();
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
            let over: usize = tick_ms.iter().skip(1).filter(|&&t| t > 2.0 * pct(0.5)).count();
            println!("    ticks 1,3,5..: mean {:.3}  ticks 2,4,..: mean {:.3}  p90 {:.3}  ticks over 2x median: {over}  sum of those {:.1} ms", mean(&even), mean(&odd), pct(0.9),
                tick_ms.iter().skip(1).filter(|&&t| t > 2.0 * pct(0.5)).sum::<f64>());
        }
        results.push(RunResult { workers: w, digest: digest.total, wall_ms, tick_ms, totals });
    }
    let base = &results[0];
    let mut same = true;
    for r in &results[1..] {
        if r.digest != base.digest {
            same = false;
            println!("DIGEST MISMATCH: {} worker(s) {:032x} vs {} worker(s) {:032x}", r.workers, r.digest, base.workers, base.digest);
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
        println!("{}", if same { "digests equal at every worker count" } else { "DIGESTS DIFFER" });
    }
    Ok(same)
}

fn accumulate(t: &mut TickStats, s: &TickStats) {
    t.systems_ns += s.systems_ns;
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
}
