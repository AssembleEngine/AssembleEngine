//! Wall-time sampling for subsystem benches.

use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub median_ns: f64,
    pub min_ns: f64,
    /// (p90 - p10) / median.
    pub spread: f64,
    pub samples: usize,
}

/// How long to keep sampling one metric.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub min_samples: usize,
    pub max_samples: usize,
    pub target: Duration,
}

impl Budget {
    pub const QUICK: Budget = Budget { min_samples: 5, max_samples: 200, target: Duration::from_millis(300) };
    pub const FULL: Budget = Budget { min_samples: 15, max_samples: 2000, target: Duration::from_secs(2) };
}

/// Time `f` repeatedly (after one warm-up call) and summarise the samples.
/// The result of `f` goes through `black_box` so the work is not optimised away.
pub fn time<R>(budget: Budget, mut f: impl FnMut() -> R) -> Sample {
    std::hint::black_box(f());
    let mut samples = Vec::with_capacity(budget.min_samples);
    let started = Instant::now();
    while samples.len() < budget.min_samples || (samples.len() < budget.max_samples && started.elapsed() < budget.target) {
        let t = Instant::now();
        std::hint::black_box(f());
        samples.push(t.elapsed().as_nanos() as f64);
    }
    summarise(&mut samples)
}

fn summarise(samples: &mut [f64]) -> Sample {
    samples.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    let median = at(0.5);
    Sample { median_ns: median, min_ns: samples[0], spread: if median > 0.0 { (at(0.9) - at(0.1)) / median } else { 0.0 }, samples: samples.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_uses_order_statistics() {
        let mut s: Vec<f64> = (1..=11).rev().map(|v| v as f64).collect();
        let sample = summarise(&mut s);
        assert_eq!(sample.median_ns, 6.0);
        assert_eq!(sample.min_ns, 1.0);
        assert!((sample.spread - 8.0 / 6.0).abs() < 1e-9);
    }
}
