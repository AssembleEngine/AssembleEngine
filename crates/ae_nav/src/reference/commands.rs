//! Command payloads, curve sampling/simplification, and deterministic arrival
//! slots. The requested point is kept distinct from generated slots.
use crate::math::{CellPos, WorldPos};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CommandKind {
    MoveTo,
    /// Route guide: units progress along the curve in order and gather at
    /// its endpoint (Alt + RMB drag).
    FollowCurve {
        guide_points: Vec<WorldPos>,
    },
    /// Formation: each unit's destination is a slot on the curve, spread
    /// evenly along it (RMB drag).
    CurveFormation {
        guide_points: Vec<WorldPos>,
    },
    /// Stop: units leave their command and stand where they come to rest
    /// (S). `requested_point` is ignored.
    Stop,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommandPayload {
    pub id: u64,
    /// Stable unit IDs, already deduplicated and sorted by the UI.
    pub unit_ids: Vec<u32>,
    pub kind: CommandKind,
    /// Destination requested by the user. For a curve command this equals the
    /// final guide point, but is rendered separately from arrival slots.
    pub requested_point: WorldPos,
    pub corridor_tolerance_world: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArrivalSlot {
    pub unit_id: u32,
    pub cell: CellPos,
    pub position: WorldPos,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandStatus {
    Planning,
    Executing,
    Completed,
    PartlyUnreachable,
    Unreachable,
    TruncatedCurve,
    Repairing,
    Interrupted,
}

impl CommandStatus {
    pub fn label(self) -> &'static str {
        match self {
            CommandStatus::Planning => "planning",
            CommandStatus::Executing => "executing",
            CommandStatus::Completed => "completed",
            CommandStatus::PartlyUnreachable => "partly unreachable",
            CommandStatus::Unreachable => "unreachable",
            CommandStatus::TruncatedCurve => "curve truncated",
            CommandStatus::Repairing => "repairing",
            CommandStatus::Interrupted => "interrupted",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CommandRecord {
    pub payload: CommandPayload,
    pub commit_tick: u64,
    pub requested_tick: u64,
    pub slots: Vec<ArrivalSlot>,
    pub status: CommandStatus,
    pub unreachable_units: u32,
    pub arrived_units: u32,
    pub total_units: u32,
    pub active_units: u32,
    pub replaced_units: u32,
    pub first_move_tick: Option<u64>,
    pub first_move_wall_ms: Option<f64>,
    pub committed_wall_ms: f64,
    pub skipped_guides: u32,
    pub truncated: bool,
    pub cohort_ids: Vec<u32>,
    pub route_polylines: Vec<Vec<WorldPos>>,
    pub repair_rerouted_units: u32,
    pub repair_invalidated_units: u32,
}

/// Distance-based sampling followed by deterministic Ramer-Douglas-Peucker
/// simplification. Used by the UI before sending a curve command; simulation
/// never receives every mouse event.
pub fn sample_and_simplify(raw: &[WorldPos], min_spacing: f32, epsilon: f32) -> Vec<WorldPos> {
    if raw.len() < 2 {
        return raw.to_vec();
    }
    let spacing = min_spacing.max(0.01);
    let mut sample = Vec::with_capacity(raw.len());
    sample.push(raw[0]);
    let mut last = raw[0];
    for p in raw.iter().skip(1) {
        if p.dist(last) >= spacing {
            // Also drop samples that would recreate the prior segment.
            if sample
                .last()
                .map(|q: &WorldPos| q.dist(*p) >= spacing * 0.5)
                .unwrap_or(true)
            {
                sample.push(*p);
                last = *p;
            }
        }
    }
    let end = *raw.last().unwrap();
    if sample.last().map(|p| p.dist(end)).unwrap_or(1.0) > 1e-4 {
        sample.push(end);
    }
    if sample.len() <= 2 {
        return sample;
    }
    let mut keep = vec![false; sample.len()];
    keep[0] = true;
    keep[sample.len() - 1] = true;
    rdp(&sample, 0, sample.len() - 1, epsilon.max(0.01), &mut keep);
    sample
        .iter()
        .zip(keep)
        .filter_map(|(p, k)| if k { Some(*p) } else { None })
        .collect()
}

fn point_line_distance(p: WorldPos, a: WorldPos, b: WorldPos) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len2 = dx * dx + dy * dy;
    if len2 <= 1e-9 {
        return p.dist(a);
    }
    let t = (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0);
    let q = WorldPos::new(a.x + t * dx, a.y + t * dy);
    p.dist(q)
}

fn rdp(points: &[WorldPos], a: usize, b: usize, epsilon: f32, keep: &mut [bool]) {
    if b <= a + 1 {
        return;
    }
    let mut max_d = -1.0;
    let mut idx = a;
    for i in a + 1..b {
        let d = point_line_distance(points[i], points[a], points[b]);
        if d > max_d {
            max_d = d;
            idx = i;
        }
    }
    if max_d > epsilon {
        keep[idx] = true;
        rdp(points, a, idx, epsilon, keep);
        rdp(points, idx, b, epsilon, keep);
    }
}

/// Resample a polyline by distance. Handy for tests and for converting a
/// smoothed gesture into ordered navigation checkpoints.
pub fn resample_by_distance(points: &[WorldPos], spacing: f32) -> Vec<WorldPos> {
    if points.len() < 2 || spacing <= 0.0 {
        return points.to_vec();
    }
    let mut out = vec![points[0]];
    let mut carry = 0.0f32;
    for w in points.windows(2) {
        let a = w[0];
        let b = w[1];
        let seg_len = a.dist(b);
        if seg_len <= 1e-6 {
            continue;
        }
        let dir = b.sub(a).scaled(1.0 / seg_len);
        let mut t = spacing - carry;
        while t <= seg_len {
            out.push(a.add(dir.scaled(t)));
            t += spacing;
        }
        carry = seg_len - (t - spacing);
    }
    if out.last().map(|p| p.dist(*points.last().unwrap())).unwrap_or(0.0) > 1e-4 {
        out.push(*points.last().unwrap());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_simplification_preserves_corners_and_is_order_stable() {
        let raw: Vec<WorldPos> = (0..=100).map(|i| WorldPos::new(i as f32, 0.0)).collect();
        let mut with_turn = raw.clone();
        with_turn.extend((1..=50).map(|i| WorldPos::new(100.0, i as f32)));
        let s = sample_and_simplify(&with_turn, 1.5, 0.5);
        assert_eq!(s.first(), Some(&WorldPos::new(0.0, 0.0)));
        assert_eq!(s.last(), Some(&WorldPos::new(100.0, 50.0)));
        assert!(s.len() < with_turn.len());
        assert!(s.iter().any(|p| (*p - WorldPos::new(100.0, 0.0)).length() < 1.0));
    }
}
