//! Standalone subsystem bench suites.

use crate::measure::Budget;
use crate::record::Report;
use anyhow::Result;

pub mod anim;
pub mod ecs_structural;
pub mod framegraph_bake;
pub mod nav;
pub mod nav_sim;
pub mod sensors;
pub mod sim_dispatch;
pub mod spatial;
pub mod view;

/// Options shared by all suites.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub budget: Budget,
    /// Use the small workload (seconds, for PR checks) instead of the full one.
    pub small: bool,
    /// Worker threads for the parallel variants.
    pub threads: usize,
}

pub struct Suite {
    pub name: &'static str,
    pub about: &'static str,
    pub run: fn(&Options) -> Result<Report>,
}

pub const SUITES: &[Suite] = &[
    Suite {
        name: "framegraph-bake",
        about: "frame graph declare, signature, bake and barrier planning",
        run: framegraph_bake::run,
    },
    Suite {
        name: "ecs-structural",
        about: "spawn, despawn, add and remove commands on the ECS backend; id repeatability",
        run: ecs_structural::run,
    },
    Suite {
        name: "nav",
        about: "terrain, nav hierarchy, destination context, flow field, ORCA avoidance",
        run: nav::run,
    },
    Suite {
        name: "nav-sim",
        about: "whole navigation tick (NavService over body columns) at 1..N workers; equal digests required",
        run: nav_sim::run,
    },
    Suite {
        name: "sensors",
        about: "line-of-sight and radar coverage per allyteam (ae_sensors), serial and parallel; equal bits required",
        run: sensors::run,
    },
    Suite {
        name: "sim-dispatch",
        about: "sandbox game tick: layered schedule on the worker crew, merge, apply; equal digests at 1..N workers",
        run: sim_dispatch::run,
    },
    Suite {
        name: "view",
        about: "copy layer view tables per observer (ae_view): frame preparation, one build, a two-observer tick",
        run: view::run,
    },
    Suite {
        name: "anim",
        about: "animation ticker rows (ae_services::anim) on seeded channels at 1 and N threads; equal digests required",
        run: anim::run,
    },
    Suite {
        name: "spatial",
        about: "grid rebuild and batch radius / nearest-k / sweep queries; equal hits at 1 and N threads",
        run: spatial::run,
    },
];

pub fn find(name: &str) -> Option<&'static Suite> {
    SUITES.iter().find(|s| s.name == name)
}
