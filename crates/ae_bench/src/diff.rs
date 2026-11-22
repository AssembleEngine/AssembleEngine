//! Regression detection between two sets of benchmark reports.

use crate::record::{Kind, Report, format_value};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    /// Allowed relative wall-time increase (0.10 = 10%).
    pub time: f64,
    /// Allowed relative counter increase. Counters are exact, so 0 by default.
    pub counter: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            time: 0.10,
            counter: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Same,
    Improved,
    Regressed,
    /// An `Info` value differs: the workload changed, times are not comparable.
    WorkloadChanged,
    /// Present on one side only.
    Added,
    Removed,
    /// Wall times from different machines; reported but never judged.
    NotComparable,
}

#[derive(Clone, Debug)]
pub struct Row {
    pub suite: String,
    pub metric: String,
    pub unit: String,
    pub base: Option<f64>,
    pub head: Option<f64>,
    pub verdict: Verdict,
}

impl Row {
    pub fn change(&self) -> Option<f64> {
        match (self.base, self.head) {
            (Some(b), Some(h)) if b != 0.0 => Some(h / b - 1.0),
            _ => None,
        }
    }
}

pub struct Diff {
    pub rows: Vec<Row>,
}

impl Diff {
    pub fn regressions(&self) -> usize {
        self.rows.iter().filter(|r| r.verdict == Verdict::Regressed).count()
    }

    pub fn print(&self, show_unchanged: bool) {
        println!(
            "{:<58} {:>14} {:>14} {:>9}  verdict",
            "metric", "base", "head", "change"
        );
        for row in &self.rows {
            if row.verdict == Verdict::Same && !show_unchanged {
                continue;
            }
            let value = |v: Option<f64>| v.map(|v| format_value(v, &row.unit)).unwrap_or_else(|| "-".to_string());
            let change = row
                .change()
                .map(|c| format!("{:+.1}%", c * 100.0))
                .unwrap_or_else(|| "-".to_string());
            let verdict = match row.verdict {
                Verdict::Same => "same",
                Verdict::Improved => "improved",
                Verdict::Regressed => "REGRESSED",
                Verdict::WorkloadChanged => "workload changed",
                Verdict::Added => "added",
                Verdict::Removed => "removed",
                Verdict::NotComparable => "other machine",
            };
            println!(
                "{:<58} {:>14} {:>14} {:>9}  {}",
                format!("{}/{}", row.suite, row.metric),
                value(row.base),
                value(row.head),
                change,
                verdict
            );
        }
        let same = self.rows.iter().filter(|r| r.verdict == Verdict::Same).count();
        println!(
            "{} metrics, {} unchanged, {} regressed",
            self.rows.len(),
            same,
            self.regressions()
        );
    }
}

pub fn diff(base: &[Report], head: &[Report], thresholds: Thresholds) -> Diff {
    type Key = (String, String);
    struct Entry {
        kind: Kind,
        unit: String,
        value: f64,
        machine: String,
        tolerance: Option<f64>,
    }
    let index = |reports: &[Report]| -> BTreeMap<Key, Entry> {
        let mut map = BTreeMap::new();
        for report in reports {
            for r in &report.records {
                map.insert(
                    (report.suite.clone(), r.metric.clone()),
                    Entry {
                        kind: r.kind,
                        unit: r.unit.clone(),
                        value: r.value,
                        machine: report.machine.clone(),
                        tolerance: r.tolerance,
                    },
                );
            }
        }
        map
    };
    let (base, head) = (index(base), index(head));

    let mut keys: Vec<&Key> = base.keys().chain(head.keys()).collect();
    keys.sort();
    keys.dedup();

    let rows = keys
        .into_iter()
        .map(|key| {
            let (b, h) = (base.get(key), head.get(key));
            let verdict = match (b, h) {
                (None, Some(_)) => Verdict::Added,
                (Some(_), None) => Verdict::Removed,
                (Some(b), Some(h)) => {
                    // A record's own tolerance widens the global one (either side's).
                    let own = b
                        .tolerance
                        .into_iter()
                        .chain(h.tolerance)
                        .fold(thresholds.time, f64::max);
                    judge(
                        b.kind,
                        b.value,
                        h.value,
                        b.machine == h.machine,
                        Thresholds {
                            time: own,
                            ..thresholds
                        },
                    )
                }
                (None, None) => unreachable!(),
            };
            let unit = h.or(b).map(|e| e.unit.clone()).unwrap_or_default();
            Row {
                suite: key.0.clone(),
                metric: key.1.clone(),
                unit,
                base: b.map(|e| e.value),
                head: h.map(|e| e.value),
                verdict,
            }
        })
        .collect();
    Diff { rows }
}

fn judge(kind: Kind, base: f64, head: f64, same_machine: bool, thresholds: Thresholds) -> Verdict {
    let ratio = if base == 0.0 {
        if head == 0.0 { 1.0 } else { f64::INFINITY }
    } else {
        head / base
    };
    match kind {
        Kind::Info if head != base => Verdict::WorkloadChanged,
        Kind::Info => Verdict::Same,
        Kind::Time if !same_machine => Verdict::NotComparable,
        Kind::Time if ratio > 1.0 + thresholds.time => Verdict::Regressed,
        Kind::Time if ratio < 1.0 - thresholds.time => Verdict::Improved,
        Kind::Time => Verdict::Same,
        Kind::Counter if ratio > 1.0 + thresholds.counter => Verdict::Regressed,
        Kind::Counter if head < base => Verdict::Improved,
        Kind::Counter => Verdict::Same,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Record;

    fn report(machine: &str, records: &[(&str, Kind, f64)]) -> Report {
        Report {
            suite: "s".into(),
            commit: "c".into(),
            machine: machine.into(),
            records: records
                .iter()
                .map(|(m, kind, value)| Record {
                    metric: m.to_string(),
                    kind: *kind,
                    value: *value,
                    unit: "ns".into(),
                    spread: 0.0,
                    tolerance: None,
                })
                .collect(),
        }
    }

    fn verdicts(base: Report, head: Report) -> Vec<(String, Verdict)> {
        diff(&[base], &[head], Thresholds::default())
            .rows
            .into_iter()
            .map(|r| (r.metric, r.verdict))
            .collect()
    }

    #[test]
    fn times_use_a_tolerance_and_counters_do_not() {
        let base = report(
            "m",
            &[
                ("t_noise", Kind::Time, 100.0),
                ("t_slow", Kind::Time, 100.0),
                ("t_fast", Kind::Time, 100.0),
                ("c_up", Kind::Counter, 1000.0),
                ("c_down", Kind::Counter, 1000.0),
                ("c_same", Kind::Counter, 5.0),
            ],
        );
        let head = report(
            "m",
            &[
                ("t_noise", Kind::Time, 108.0),
                ("t_slow", Kind::Time, 120.0),
                ("t_fast", Kind::Time, 70.0),
                ("c_up", Kind::Counter, 1001.0),
                ("c_down", Kind::Counter, 900.0),
                ("c_same", Kind::Counter, 5.0),
            ],
        );
        let v = verdicts(base, head);
        let get = |name: &str| v.iter().find(|(m, _)| m == name).unwrap().1;
        assert_eq!(get("t_noise"), Verdict::Same);
        assert_eq!(get("t_slow"), Verdict::Regressed);
        assert_eq!(get("t_fast"), Verdict::Improved);
        assert_eq!(get("c_up"), Verdict::Regressed);
        assert_eq!(get("c_down"), Verdict::Improved);
        assert_eq!(get("c_same"), Verdict::Same);
    }

    #[test]
    fn a_record_tolerance_widens_the_global_one() {
        let mut base = report("m", &[("cpu", Kind::Time, 100.0), ("gpu", Kind::Time, 100.0)]);
        base.records[0].tolerance = Some(0.25);
        let head = report("m", &[("cpu", Kind::Time, 120.0), ("gpu", Kind::Time, 120.0)]);
        let v = verdicts(base.clone(), head.clone());
        let get = |v: &[(String, Verdict)], name: &str| v.iter().find(|(m, _)| m == name).unwrap().1;
        assert_eq!(get(&v, "cpu"), Verdict::Same);
        assert_eq!(get(&v, "gpu"), Verdict::Regressed);
        let mut slower = head;
        slower.records[0].value = 130.0;
        assert_eq!(get(&verdicts(base, slower), "cpu"), Verdict::Regressed);
    }

    #[test]
    fn times_from_another_machine_are_not_judged_but_counters_are() {
        let base = report("a", &[("t", Kind::Time, 100.0), ("c", Kind::Counter, 10.0)]);
        let head = report("b", &[("t", Kind::Time, 500.0), ("c", Kind::Counter, 11.0)]);
        let result = diff(&[base], &[head], Thresholds::default());
        assert_eq!(
            result.rows.iter().find(|r| r.metric == "t").unwrap().verdict,
            Verdict::NotComparable
        );
        assert_eq!(result.regressions(), 1);
    }

    #[test]
    fn added_removed_and_workload_changes_are_reported() {
        let base = report("m", &[("gone", Kind::Time, 1.0), ("size", Kind::Info, 64.0)]);
        let head = report("m", &[("new", Kind::Time, 1.0), ("size", Kind::Info, 128.0)]);
        let v = verdicts(base, head);
        assert!(v.contains(&("gone".to_string(), Verdict::Removed)));
        assert!(v.contains(&("new".to_string(), Verdict::Added)));
        assert!(v.contains(&("size".to_string(), Verdict::WorkloadChanged)));
    }
}
