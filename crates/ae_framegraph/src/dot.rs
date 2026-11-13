//! Graphviz export of a baked plan: passes in execution order (labelled with
//! their index and kind), dependency edges labelled with the resource, and
//! alias groups. Deterministic for a given graph and plan.

use super::{BakedGraph, FrameGraph, PassKind};
use std::fmt::Write as _;

fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

pub fn to_dot<R>(graph: &FrameGraph<R>, baked: &BakedGraph) -> String {
    to_dot_with_times(graph, baked, &[])
}

/// As `to_dot`, annotating pass `i` (baked order) with `times_ms[i]` when
/// present (GPU timestamp averages).
pub fn to_dot_with_times<R>(graph: &FrameGraph<R>, baked: &BakedGraph, times_ms: &[f32]) -> String {
    let mut out = String::from("digraph frame_graph {\n  rankdir=LR;\n  node [shape=box, style=filled];\n");
    for (index, pass) in baked.passes.iter().enumerate() {
        let color = match pass.kind {
            PassKind::Graphics => "#cfe2ff",
            PassKind::Compute => "#d1e7dd",
            PassKind::Transfer => "#fff3cd",
            PassKind::Present => "#e2e3e5",
        };
        let time = times_ms
            .get(index)
            .map(|ms| format!("\\n{ms:.3} ms"))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "  p{} [label=\"{}: {}\\n{:?}{}{}\", fillcolor=\"{}\"];",
            pass.pass.0,
            index,
            escape(graph.pass_name(pass.pass)),
            pass.kind,
            if pass.conditional { " (conditional)" } else { "" },
            time,
            color
        );
    }
    for edge in &baked.edges {
        let _ = writeln!(
            out,
            "  p{} -> p{} [label=\"{}\"];",
            edge.from.0,
            edge.to.0,
            escape(graph.resource_name(edge.resource))
        );
    }
    for (index, physical) in baked.physical.iter().enumerate() {
        let members: Vec<String> = physical
            .members
            .iter()
            .map(|member| escape(graph.resource_name(*member)))
            .collect();
        let _ = writeln!(out, "  // physical {index} {:?}: {}", physical.class, members.join(" -> "));
    }
    for handoff in &baked.handoffs {
        let _ = writeln!(
            out,
            "  // handoff physical {}: {} -> {} at pass {}",
            handoff.physical,
            escape(graph.resource_name(handoff.from)),
            escape(graph.resource_name(handoff.to)),
            handoff.at_pass
        );
    }
    out.push_str("}\n");
    out
}
