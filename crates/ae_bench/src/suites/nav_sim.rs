//! Whole navigation tick: the columnar `NavService` driven through body
//! columns (`ae_nav::scenario`), at several worker counts.
//!
//! * `land`: navbench's `scale_10000_one_point` (pathsim units from the spawn
//!   plaza to cell (1020, 1900) on the dry map). At 10k units and 600 ticks
//!   its state hash equals pathsim's own navbench run (`cb3e58c816c30008`).
//! * `water`: the same map with water (`WaterMode::Surface`: lakes and a
//!   river) and all five movement classes: land units cross the river on
//!   the causeway, hovercraft and amphibious units go into the lake, ships
//!   sail up the river.
//!
//! Each worker count runs the same ticks; the final state hashes and the
//! navigation work counters must be equal, or the suite fails.

use super::Options;
use crate::measure::{Sample, summarise};
use crate::record::Report;
use ae_nav::scenario::{Scenario, land_crossing, water_crossing};
use anyhow::{Result, bail};
use std::time::Instant;

struct Run {
    tick_ns: Vec<f64>,
    hash: u64,
    phases: [f64; 4],
    work: [u64; 4],
    fallback_slots: u64,
    arrived: usize,
    unreachable: usize,
}

fn run_once(mut s: Scenario, ticks: u64) -> Run {
    let mut tick_ns = Vec::with_capacity(ticks as usize);
    let mut phases = [0.0f64; 4];
    let mut last = Instant::now();
    s.run(ticks, |_, nav, _| {
        let now = Instant::now();
        tick_ns.push((now - last).as_nanos() as f64);
        let p = &nav.stats().phases;
        for (acc, v) in phases
            .iter_mut()
            .zip([p.steer_ms, p.avoidance_ms, p.physics_ms, p.jobs_ms])
        {
            *acc += v;
        }
        last = Instant::now();
    });
    let stats = s.nav.stats();
    Run {
        tick_ns,
        hash: s.nav.state_hash(),
        phases: phases.map(|v| v / ticks as f64 * 1e6),
        work: [
            stats.nav.fields_built,
            stats.nav.field_expansions,
            stats.nav.abstract_expansions,
            stats.nav.context_misses,
        ],
        fallback_slots: stats.nav.fallback_slots,
        arrived: stats.arrived,
        unreachable: stats.unreachable,
    }
}

fn exact(ns: f64, samples: usize) -> Sample {
    Sample {
        median_ns: ns,
        min_ns: ns,
        spread: 0.0,
        samples,
    }
}

pub fn run(options: &Options) -> Result<Report> {
    let mut report = Report::new("nav-sim");
    let (units, map, ticks) = if options.small {
        (2_000, 768, 150)
    } else {
        (10_000, 2560, 600)
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
    let scenarios: [(&str, fn(u32, i32, usize) -> Scenario); 2] = [("land", land_crossing), ("water", water_crossing)];
    for (name, build) in scenarios {
        let tag = format!("{name}_units{units}_map{map}_ticks{ticks}");
        let mut first: Option<(usize, Run)> = None;
        for &w in &workers {
            let r = run_once(build(units, map, w), ticks);
            let mut samples = r.tick_ns.clone();
            report.time(&format!("{tag}/tick_{w}t"), summarise(&mut samples));
            let p95 = samples[((samples.len() - 1) as f64 * 0.95).round() as usize];
            report.time(&format!("{tag}/tick_p95_{w}t"), exact(p95, samples.len()));
            if w == *workers.last().unwrap() {
                for (phase, ns) in ["steer", "avoidance", "physics", "jobs_wait"].iter().zip(r.phases) {
                    report.time(&format!("{tag}/phase_{phase}_{w}t"), exact(ns, ticks as usize));
                }
            }
            match &first {
                None => {
                    for (counter, v) in [
                        "fields_built",
                        "field_expansions",
                        "abstract_expansions",
                        "contexts_built",
                    ]
                    .iter()
                    .zip(r.work)
                    {
                        report.counter(&format!("{tag}/{counter}"), v, "");
                    }
                    report.counter(&format!("{tag}/fallback_slots"), r.fallback_slots, "");
                    report.info(&format!("{tag}/arrived_at_end"), r.arrived as u64, "");
                    report.info(&format!("{tag}/unreachable_at_end"), r.unreachable as u64, "");
                    report.info(&format!("{tag}/state_hash_low32"), r.hash & 0xFFFF_FFFF, "");
                    first = Some((w, r));
                }
                Some((w0, r0)) => {
                    if r.hash != r0.hash {
                        bail!(
                            "{name}: state hash differs: {:#018x} at {w0} worker(s), {:#018x} at {w}",
                            r0.hash,
                            r.hash
                        );
                    }
                    if r.work != r0.work {
                        bail!("{name}: navigation work differs between {w0} and {w} worker(s)");
                    }
                }
            }
        }
    }
    Ok(report)
}
