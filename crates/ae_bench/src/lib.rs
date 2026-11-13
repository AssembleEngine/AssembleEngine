//! Benchmark infrastructure.
//!
//! * [`record`]: the JSON result format every bench writes.
//! * [`measure`]: wall-time sampling.
//! * [`diff`]: regression detection between two result sets.
//! * [`suites`]: subsystem benches that run one engine crate on a synthetic
//!   workload, with no window, GPU or network. `ae-tool bench <suite>`.
//!
//! Every suite reports two kinds of numbers: wall times (noisy, compared
//! with a tolerance) and deterministic work counters (exact, compared
//! strictly). Counters make regressions detectable on any machine.

pub mod diff;
pub mod measure;
pub mod record;
pub mod suites;

pub use record::{Kind, Record, Report};
