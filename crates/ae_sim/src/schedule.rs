//! Deterministic system schedule with parallel layers.
//!
//! Lifted from `ecs_sim/src/engine/schedule.rs`, with one rule reversed: the
//! prototype only reported read/write conflicts, here they order systems.
//!
//! 1. Order: registration order, refined by explicit `after` edges. Kahn's
//!    algorithm takes the lowest registration index first, so the order is a
//!    pure function of the declarations.
//! 2. Edges: explicit `after` edges, plus one edge for every pair of systems
//!    that conflict on a resource (write/write, write/read, read/write),
//!    pointing from the system earlier in the order to the later one.
//!    Components and event streams are both resources: emitting into a
//!    stream counts as a write, reading it as a read.
//! 3. Layers: a system's layer is one more than the deepest layer of any
//!    system it has an edge from. Systems of one layer touch no common
//!    resource and run at the same time; structural commands and events of a
//!    layer are merged and applied when the whole layer is done.
//!
//! Layers only depend on the declarations, never on worker count or timing,
//! so which layer's commands a system sees is part of the game's semantics.

use ae_ecs::{Access, ComponentId};
use anyhow::{Result, bail, ensure};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

/// Something systems read or write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Resource {
    Component(ComponentId),
    /// An event stream, by its stable id.
    Stream(u64),
}

/// How a system's rows are spread over workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    /// Work ranges are fanned out over workers.
    #[default]
    Parallel,
    /// All of the system's ranges run on one worker, in order.
    Serial,
}

/// When a system runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Cadence {
    #[default]
    EveryTick,
    /// On ticks where `tick % n == phase`.
    Every { n: u32, phase: u32 },
    /// Every tick, but only on rows whose entity index satisfies
    /// `index % n == tick % n`, so each row is visited once every `n` ticks.
    Staggered { n: u32 },
    /// Tick 0 only (setup).
    Once,
}

impl Cadence {
    pub fn runs_on(self, tick: u64) -> bool {
        match self {
            Cadence::EveryTick | Cadence::Staggered { .. } => true,
            Cadence::Every { n, phase } => tick % n as u64 == phase as u64,
            Cadence::Once => tick == 0,
        }
    }

    /// `(n, phase)` row filter for this tick, if any.
    pub fn stagger(self, tick: u64) -> Option<(u32, u32)> {
        match self {
            Cadence::Staggered { n } => Some((n, (tick % n as u64) as u32)),
            _ => None,
        }
    }

    pub fn validate(self) -> Result<()> {
        match self {
            Cadence::Every { n, phase } => ensure!(
                n > 0 && phase < n,
                "cadence every({n}, {phase}) needs n > 0 and phase < n"
            ),
            Cadence::Staggered { n } => ensure!(n > 0, "cadence staggered({n}) needs n > 0"),
            _ => {}
        }
        Ok(())
    }
}

/// What the scheduler needs to know about one system.
#[derive(Clone, Debug, Default)]
pub struct SystemDecl {
    pub name: String,
    /// Query terms; may be empty (the system runs once per tick, no rows).
    pub terms: Vec<ae_ecs::Term>,
    pub emits: Vec<u64>,
    pub reads_events: Vec<u64>,
    /// Components read of other entities during the run (`get(T)`): reads
    /// for layering, not query terms (they do not restrict the rows).
    pub gets: Vec<u64>,
    /// Names of systems that must run before this one.
    pub after: Vec<String>,
    pub mode: Mode,
    pub cadence: Cadence,
}

impl SystemDecl {
    /// Read and write sets. A resource may appear once per system.
    pub fn access(&self) -> Result<(BTreeSet<Resource>, BTreeSet<Resource>)> {
        let mut reads = BTreeSet::new();
        let mut writes = BTreeSet::new();
        let mut seen = BTreeSet::new();
        {
            let mut note = |r: Resource| -> Result<()> {
                ensure!(seen.insert(r), "system `{}` lists {r:?} twice", self.name);
                Ok(())
            };
            for term in &self.terms {
                let r = Resource::Component(term.component);
                note(r)?;
                match term.access {
                    Access::Read => reads.insert(r),
                    Access::Write => writes.insert(r),
                };
            }
            for &s in &self.emits {
                note(Resource::Stream(s))?;
                writes.insert(Resource::Stream(s));
            }
            for &s in &self.reads_events {
                note(Resource::Stream(s))?;
                reads.insert(Resource::Stream(s));
            }
        }
        for &id in &self.gets {
            // A `get` of a component also read as a term is one read; of a
            // written one is rejected when the module loads.
            let r = Resource::Component(ae_ecs::ComponentId(id));
            if seen.insert(r) {
                reads.insert(r);
            }
        }
        Ok((reads, writes))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeReason {
    After,
    Conflict(Resource),
}

#[derive(Clone, Debug, Default)]
pub struct Schedule {
    /// System indices (registration order) in execution order.
    pub order: Vec<usize>,
    /// Position of each system in `order`. Command keys use this.
    pub position: Vec<u32>,
    /// Layer of each system.
    pub layer_of: Vec<u32>,
    /// Systems per layer, in execution order.
    pub layers: Vec<Vec<usize>>,
    /// `(from, to, why)` with `from` before `to`; one entry per pair, the
    /// explicit reason first.
    pub edges: Vec<(usize, usize, EdgeReason)>,
}

pub fn compute(systems: &[SystemDecl]) -> Result<Schedule> {
    let count = systems.len();
    let mut by_name = BTreeMap::new();
    for (index, s) in systems.iter().enumerate() {
        ensure!(
            by_name.insert(s.name.as_str(), index).is_none(),
            "two systems are named `{}`",
            s.name
        );
        s.cadence
            .validate()
            .map_err(|e| anyhow::anyhow!("system `{}`: {e}", s.name))?;
    }
    let access: Vec<_> = systems.iter().map(SystemDecl::access).collect::<Result<_>>()?;

    // 1. Order from explicit edges, lowest registration index first.
    let mut explicit: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); count];
    let mut indegree = vec![0usize; count];
    for (index, s) in systems.iter().enumerate() {
        for dep in &s.after {
            let Some(&d) = by_name.get(dep.as_str()) else {
                bail!("system `{}` runs after unknown system `{dep}`", s.name)
            };
            ensure!(d != index, "system `{}` runs after itself", s.name);
            if explicit[d].insert(index) {
                indegree[index] += 1;
            }
        }
    }
    let mut ready: BinaryHeap<Reverse<usize>> = (0..count).filter(|&i| indegree[i] == 0).map(Reverse).collect();
    let mut order = Vec::with_capacity(count);
    while let Some(Reverse(index)) = ready.pop() {
        order.push(index);
        for &next in &explicit[index] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                ready.push(Reverse(next));
            }
        }
    }
    if order.len() != count {
        let blocked: Vec<&str> = (0..count)
            .filter(|i| indegree[*i] > 0)
            .map(|i| systems[i].name.as_str())
            .collect();
        bail!("`after` cycle between systems: {}", blocked.join(", "));
    }
    let mut position = vec![0u32; count];
    for (p, &i) in order.iter().enumerate() {
        position[i] = p as u32;
    }

    // 2. Edges, always from earlier to later in the order.
    let mut edges = Vec::new();
    for (pa, &a) in order.iter().enumerate() {
        for &b in &order[pa + 1..] {
            if explicit[a].contains(&b) {
                edges.push((a, b, EdgeReason::After));
                continue;
            }
            let ((ra, wa), (rb, wb)) = (&access[a], &access[b]);
            let conflict = wa
                .intersection(wb)
                .chain(wa.intersection(rb))
                .chain(ra.intersection(wb))
                .min()
                .copied();
            if let Some(r) = conflict {
                edges.push((a, b, EdgeReason::Conflict(r)));
            }
        }
    }

    // 3. Layers: longest path from a source, visited in order.
    let mut layer_of = vec![0u32; count];
    for &(a, b, _) in &edges {
        // Edges are listed with `a` in order before `b`, and grouped by `a`
        // in order, so every predecessor's layer is final when used.
        layer_of[b] = layer_of[b].max(layer_of[a] + 1);
    }
    let depth = layer_of.iter().copied().max().map_or(0, |d| d as usize + 1);
    let mut layers = vec![Vec::new(); depth];
    for &i in &order {
        layers[layer_of[i] as usize].push(i);
    }
    Ok(Schedule {
        order,
        position,
        layer_of,
        layers,
        edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_ecs::Term;

    fn sys(name: &str, reads: &[u64], writes: &[u64]) -> SystemDecl {
        let mut terms: Vec<Term> = reads
            .iter()
            .map(|&c| Term {
                component: ComponentId(c),
                access: Access::Read,
            })
            .collect();
        terms.extend(writes.iter().map(|&c| Term {
            component: ComponentId(c),
            access: Access::Write,
        }));
        SystemDecl {
            name: name.into(),
            terms,
            ..Default::default()
        }
    }

    fn after(mut s: SystemDecl, deps: &[&str]) -> SystemDecl {
        s.after = deps.iter().map(|d| d.to_string()).collect();
        s
    }

    fn names(systems: &[SystemDecl], s: &Schedule) -> Vec<Vec<String>> {
        s.layers
            .iter()
            .map(|l| l.iter().map(|&i| systems[i].name.clone()).collect())
            .collect()
    }

    #[test]
    fn readers_share_a_layer_writers_split() {
        let systems = [
            sys("a", &[1], &[]),
            sys("b", &[1], &[]),
            sys("w", &[], &[1]),
            sys("c", &[1], &[]),
        ];
        let s = compute(&systems).unwrap();
        assert_eq!(names(&systems, &s), [vec!["a", "b"], vec!["w"], vec!["c"]]);
    }

    #[test]
    fn disjoint_writers_run_together() {
        let systems = [sys("move", &[2], &[1]), sys("heal", &[], &[3]), sys("aim", &[1], &[2])];
        let s = compute(&systems).unwrap();
        // aim reads Position (1) that move writes, and writes Velocity (2) that move reads.
        assert_eq!(names(&systems, &s), [vec!["move", "heal"], vec!["aim"]]);
    }

    #[test]
    fn conflicting_systems_never_share_a_layer() {
        // Every pair with a common written resource or a write/read overlap.
        let mut systems = Vec::new();
        for i in 0..24u64 {
            let r = [i % 5, (i * 7) % 5 + 5];
            let w = [(i * 3) % 6];
            systems.push(sys(&format!("s{i}"), &r[..(i % 3) as usize], &w[..(i % 2) as usize]));
        }
        // Remove duplicate resources within a system.
        for s in &mut systems {
            let mut seen = BTreeSet::new();
            s.terms.retain(|t| seen.insert(t.component));
        }
        let s = compute(&systems).unwrap();
        for layer in &s.layers {
            for (k, &a) in layer.iter().enumerate() {
                for &b in &layer[k + 1..] {
                    let ((ra, wa), (rb, wb)) = (systems[a].access().unwrap(), systems[b].access().unwrap());
                    assert!(
                        wa.is_disjoint(&wb) && wa.is_disjoint(&rb) && ra.is_disjoint(&wb),
                        "{} and {} share a layer",
                        systems[a].name,
                        systems[b].name
                    );
                }
            }
        }
    }

    #[test]
    fn explicit_after_reorders_and_adds_a_layer() {
        let systems = [
            after(sys("late", &[], &[1]), &["early"]),
            sys("early", &[], &[2]),
            sys("free", &[], &[3]),
        ];
        let s = compute(&systems).unwrap();
        assert_eq!(s.order, [1, 0, 2]);
        assert_eq!(names(&systems, &s), [vec!["early", "free"], vec!["late"]]);
        // Conflict direction follows the order, not registration.
        let systems = [after(sys("x", &[1], &[]), &["y"]), sys("y", &[], &[1])];
        let s = compute(&systems).unwrap();
        assert_eq!(names(&systems, &s), [vec!["y"], vec!["x"]]);
        assert_eq!(s.edges, [(1, 0, EdgeReason::After)]);
    }

    #[test]
    fn events_order_emitters_and_readers() {
        let mut emit = sys("die", &[], &[4]);
        emit.emits = vec![900];
        let mut read = sys("count", &[], &[5]);
        read.reads_events = vec![900];
        let systems = [emit, read];
        let s = compute(&systems).unwrap();
        assert_eq!(s.edges, [(0, 1, EdgeReason::Conflict(Resource::Stream(900)))]);
        assert_eq!(s.layers.len(), 2);
    }

    #[test]
    fn cycles_and_bad_declarations_are_rejected() {
        let systems = [after(sys("a", &[], &[]), &["b"]), after(sys("b", &[], &[]), &["a"])];
        assert!(compute(&systems).unwrap_err().to_string().contains("cycle"));
        assert!(compute(&[after(sys("a", &[], &[]), &["nope"])]).is_err());
        assert!(
            compute(&[sys("a", &[], &[]), sys("a", &[], &[])]).is_err(),
            "duplicate name"
        );
        assert!(
            compute(&[sys("a", &[1], &[1])]).is_err(),
            "read and write of one component"
        );
        let mut s = sys("a", &[], &[]);
        s.cadence = Cadence::Every { n: 4, phase: 4 };
        assert!(compute(&[s]).is_err());
    }

    #[test]
    fn layers_are_a_function_of_the_declarations() {
        let systems: Vec<SystemDecl> = (0..12u64)
            .map(|i| sys(&format!("s{i}"), &[i % 4], &[4 + i % 3]))
            .collect();
        let a = compute(&systems).unwrap();
        let b = compute(&systems.clone()).unwrap();
        assert_eq!(a.layers, b.layers);
        assert_eq!(a.position, b.position);
        // Systems with no resources at all form one layer.
        let empty: Vec<SystemDecl> = (0..5).map(|i| sys(&format!("e{i}"), &[], &[])).collect();
        assert_eq!(compute(&empty).unwrap().layers.len(), 1);
    }

    #[test]
    fn cadences() {
        assert!(Cadence::Once.runs_on(0) && !Cadence::Once.runs_on(1));
        let c = Cadence::Every { n: 3, phase: 1 };
        assert_eq!((0..7).filter(|&t| c.runs_on(t)).collect::<Vec<_>>(), [1, 4]);
        assert_eq!(Cadence::Staggered { n: 4 }.stagger(6), Some((4, 2)));
        assert_eq!(Cadence::EveryTick.stagger(6), None);
    }
}
