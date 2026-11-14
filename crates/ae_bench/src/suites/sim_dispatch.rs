//! Whole sim tick of the sandbox game (`games/sandbox`) through the runtime:
//! schedule layers on the worker crew, command and event merge, apply.
//! Two churn settings: `default` (units live about 30 s) and `stress`
//! (about 6 s: deaths, events and structural commands every wear tick).
//! Every worker count runs the same world from tick 0; final digests and
//! command/event counts must be equal, or the suite fails.
//!
//! The column arena is process-wide, so the suite builds one runtime and
//! must be the only sim runtime in the process.

use super::Options;
use crate::measure::{Sample, summarise};
use crate::record::Report;
use ae_sim::runtime::{Runtime, RuntimeConfig};
use anyhow::{Result, bail};
use std::time::Instant;

fn exact(ns: f64, samples: usize) -> Sample {
    Sample {
        median_ns: ns,
        min_ns: ns,
        spread: 0.0,
        samples,
    }
}

pub fn run(options: &Options) -> Result<Report> {
    let mut report = Report::new("sim-dispatch");
    let wasm = ae_sim::build::game("sandbox")?;
    let (units, ticks) = if options.small {
        (10_000u64, 150u64)
    } else {
        (50_000, 600)
    };
    let mut workers = vec![1usize];
    for w in [2, 4, 8, 16] {
        if w <= options.threads {
            workers.push(w);
        }
    }
    if options.threads > 1 && !workers.contains(&options.threads) {
        workers.push(options.threads);
    }
    let max = *workers.iter().max().unwrap();
    let mut rt = Runtime::new(
        &wasm,
        RuntimeConfig {
            max_workers: max,
            ..RuntimeConfig::default()
        },
    )?;
    for (name, wear) in [("default", 0u64), ("stress", 500)] {
        let tag = format!("sandbox_{name}_units{units}_ticks{ticks}");
        let mut first: Option<(usize, u128, [u64; 3])> = None;
        for &w in &workers {
            let mut world = rt.world(units | wear << 32)?;
            let mut tick_ns = Vec::with_capacity(ticks as usize);
            let (mut systems, mut apply, mut counts) = (0u64, 0u64, [0u64; 3]);
            for _ in 0..ticks {
                let t0 = Instant::now();
                let s = rt.tick(&mut world, w)?;
                tick_ns.push(t0.elapsed().as_nanos() as f64);
                systems += s.systems_ns;
                apply += s.apply_ns;
                counts[0] += s.commands;
                counts[1] += s.events;
                counts[2] += s.calls;
            }
            let digest = rt.digest(&world)?.total;
            // Tick 0 is the serial setup (every unit spawned); report it apart.
            report.time(&format!("{tag}/setup_tick_{w}t"), exact(tick_ns[0], 1));
            let mut steady = tick_ns[1..].to_vec();
            let mean = steady.iter().sum::<f64>() / steady.len() as f64;
            report.time(&format!("{tag}/tick_{w}t"), summarise(&mut steady));
            report.time(&format!("{tag}/tick_mean_{w}t"), exact(mean, steady.len()));
            report.time(
                &format!("{tag}/systems_{w}t"),
                exact(systems as f64 / ticks as f64, ticks as usize),
            );
            report.time(
                &format!("{tag}/apply_{w}t"),
                exact(apply as f64 / ticks as f64, ticks as usize),
            );
            match &first {
                None => {
                    report.counter(&format!("{tag}/commands"), counts[0], "");
                    report.counter(&format!("{tag}/events"), counts[1], "");
                    report.info(
                        &format!("{tag}/digest_low32"),
                        digest as u64 & 0xFFFF_FFFF,
                        "",
                    );
                    first = Some((w, digest, counts));
                }
                Some((w0, d0, c0)) => {
                    if digest != *d0 {
                        bail!(
                            "sandbox digest differs: {d0:#034x} at {w0} worker(s), {digest:#034x} at {w}"
                        );
                    }
                    if counts[..2] != c0[..2] {
                        bail!("sandbox command/event counts differ between {w0} and {w} worker(s)");
                    }
                }
            }
        }
    }
    Ok(report)
}
