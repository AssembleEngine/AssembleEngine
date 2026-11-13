//! Result records and their on-disk form.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Wall time; lower is better; compared with a tolerance.
    Time,
    /// Deterministic amount of work; lower is better; any increase is a regression.
    Counter,
    /// Deterministic fact about the workload (sizes, counts). A change means
    /// the workload itself changed, so times are not comparable.
    Info,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub metric: String,
    pub kind: Kind,
    pub value: f64,
    pub unit: String,
    /// Relative spread of the samples ((p90 - p10) / median); 0 for exact values.
    pub spread: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub suite: String,
    pub commit: String,
    pub machine: String,
    pub records: Vec<Record>,
}

impl Report {
    pub fn new(suite: &str) -> Self {
        Self { suite: suite.to_string(), commit: current_commit(), machine: machine_id(), records: Vec::new() }
    }

    pub fn time(&mut self, metric: &str, sample: crate::measure::Sample) {
        self.records.push(Record {
            metric: metric.to_string(),
            kind: Kind::Time,
            value: sample.median_ns,
            unit: "ns".to_string(),
            spread: sample.spread,
        });
    }

    pub fn counter(&mut self, metric: &str, value: u64, unit: &str) {
        self.push_exact(metric, Kind::Counter, value, unit);
    }

    pub fn info(&mut self, metric: &str, value: u64, unit: &str) {
        self.push_exact(metric, Kind::Info, value, unit);
    }

    fn push_exact(&mut self, metric: &str, kind: Kind, value: u64, unit: &str) {
        self.records.push(Record { metric: metric.to_string(), kind, value: value as f64, unit: unit.to_string(), spread: 0.0 });
    }

    /// Write to `<dir>/<label>/<suite>.json` and return the path.
    pub fn save(&self, dir: &Path, label: &str) -> Result<PathBuf> {
        let dir = dir.join(label);
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let path = dir.join(format!("{}.json", self.suite));
        std::fs::write(&path, serde_json::to_string_pretty(self)? + "\n").with_context(|| format!("write {}", path.display()))?;
        Ok(path)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
    }

    /// Load one report file, or every `*.json` directly inside a directory.
    pub fn load_all(path: &Path) -> Result<Vec<Self>> {
        if path.is_file() {
            return Ok(vec![Self::load(path)?]);
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)
            .with_context(|| format!("read {}", path.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
            .collect();
        files.sort();
        files.iter().map(|p| Self::load(p)).collect()
    }

    pub fn print(&self) {
        println!("suite {}  commit {}  machine {}", self.suite, self.commit, self.machine);
        for r in &self.records {
            let value = format_value(r.value, &r.unit);
            match r.kind {
                Kind::Time => println!("  {:<44} {:>14}  ±{:.0}%", r.metric, value, r.spread * 100.0),
                Kind::Counter => println!("  {:<44} {:>14}  (exact)", r.metric, value),
                Kind::Info => println!("  {:<44} {:>14}  (workload)", r.metric, value),
            }
        }
    }
}

pub fn format_value(value: f64, unit: &str) -> String {
    if unit != "ns" {
        return format!("{value:.0} {unit}").trim_end().to_string();
    }
    match value {
        v if v >= 1e9 => format!("{:.3} s", v / 1e9),
        v if v >= 1e6 => format!("{:.3} ms", v / 1e6),
        v if v >= 1e3 => format!("{:.3} µs", v / 1e3),
        v => format!("{v:.1} ns"),
    }
}

/// Short commit hash, with `-dirty` when the tree has uncommitted changes.
fn current_commit() -> String {
    let git = |args: &[&str]| {
        Command::new("git").args(args).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    match git(&["rev-parse", "--short", "HEAD"]) {
        Some(hash) if !hash.is_empty() => {
            let dirty = git(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty());
            if dirty { format!("{hash}-dirty") } else { hash }
        }
        _ => "uncommitted".to_string(),
    }
}

/// Set `AE_BENCH_MACHINE` on the pinned benchmark machine so wall times from
/// different machines are never compared by accident.
fn machine_id() -> String {
    std::env::var("AE_BENCH_MACHINE").unwrap_or_else(|_| "local".to_string())
}
