//! Headless deterministic replay log and divergence finder.
//!
//! A replay is a fixed simulation configuration plus timestamped commands and
//! terrain edits. Replaying it at different worker counts must produce the same
//! state digest at every sampled tick. Worker timing never enters the digest
//! because navigation results are only applied at fixed tick boundaries.
use crate::reference::commands::CommandPayload;
use crate::reference::sim::{SimConfig, Simulation, StateHashParts};
use crate::terrain::TerrainPatch;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReplayEvent {
    Command(CommandPayload),
    Terrain(TerrainPatch),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayEventRecord {
    /// Tick at which the input becomes visible. Events are applied before the
    /// movement step for this tick.
    pub tick: u64,
    pub event: ReplayEvent,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayLog {
    pub config: SimConfig,
    pub events: Vec<ReplayEventRecord>,
    pub ticks: u64,
    pub sample_every: u64,
}

#[derive(Clone, Debug)]
pub struct ReplayDigest {
    pub tick: u64,
    pub state_hash: u64,
    pub parts: StateHashParts,
    pub arrived: usize,
    pub unreachable: usize,
    pub terrain_version: u32,
}

#[derive(Clone, Debug)]
pub struct Divergence {
    pub sample_index: usize,
    pub tick: u64,
    pub left_hash: u64,
    pub right_hash: u64,
    /// Which state parts differ (bodies, unit_nav, commands, terrain, nav).
    pub differing_parts: Vec<&'static str>,
}

impl Divergence {
    pub fn describe(&self) -> String {
        format!("first divergence at tick {} (sample {}): {:016x} vs {:016x}; differing: {}",
            self.tick, self.sample_index, self.left_hash, self.right_hash,
            if self.differing_parts.is_empty() { "-".to_string() } else { self.differing_parts.join(", ") })
    }
}

fn differing(a: &StateHashParts, b: &StateHashParts) -> Vec<&'static str> {
    let mut v = Vec::new();
    if a.bodies != b.bodies { v.push("bodies"); }
    if a.unit_nav != b.unit_nav { v.push("unit_nav"); }
    if a.commands != b.commands { v.push("commands"); }
    if a.terrain != b.terrain { v.push("terrain"); }
    if a.nav_version != b.nav_version { v.push("nav_version"); }
    v
}

impl ReplayLog {
    pub fn sorted_events(&self) -> Vec<ReplayEventRecord> {
        let mut events = self.events.clone();
        events.sort_by_key(|e| e.tick);
        events
    }

    /// Run the replay. Sampling cadence is independent of authoritative state.
    pub fn run(&self) -> Vec<ReplayDigest> {
        self.run_with_config(&self.config)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, serde_json::to_string(self).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?)
    }

    pub fn run_with_config(&self, config: &SimConfig) -> Vec<ReplayDigest> {
        let mut sim = Simulation::new(config.clone());
        let events = self.sorted_events();
        let mut next = 0usize;
        let mut out = Vec::new();
        for tick in 1..=self.ticks {
            while next < events.len() && events[next].tick <= tick {
                match &events[next].event {
                    ReplayEvent::Command(c) => sim.queue_command_at(c.clone()),
                    ReplayEvent::Terrain(p) => sim.queue_terrain_patch(p.clone()),
                }
                next += 1;
            }
            sim.advance_tick();
            if tick % self.sample_every.max(1) == 0 {
                let snap = sim.snapshot().expect("simulation publishes a snapshot");
                out.push(ReplayDigest {
                    tick: snap.tick,
                    state_hash: snap.state_hash,
                    parts: sim.state_hash_parts(),
                    arrived: snap.stats.arrived,
                    unreachable: snap.stats.unreachable,
                    terrain_version: snap.terrain.version,
                });
            }
        }
        out
    }
}

/// Compare two equal-history runs and report the first divergent sample.
pub fn first_divergence(left: &[ReplayDigest], right: &[ReplayDigest]) -> Option<Divergence> {
    let n = left.len().min(right.len());
    for i in 0..n {
        if left[i].state_hash != right[i].state_hash {
            return Some(Divergence {
                sample_index: i,
                tick: left[i].tick.min(right[i].tick),
                left_hash: left[i].state_hash,
                right_hash: right[i].state_hash,
                differing_parts: differing(&left[i].parts, &right[i].parts),
            });
        }
    }
    if left.len() != right.len() {
        let i = n;
        let l = left.get(i);
        let r = right.get(i);
        return Some(Divergence {
            sample_index: i,
            tick: l.or(r).map(|d| d.tick).unwrap_or(0),
            left_hash: l.map(|d| d.state_hash).unwrap_or(0),
            right_hash: r.map(|d| d.state_hash).unwrap_or(0),
            differing_parts: vec!["length"],
        });
    }
    None
}
