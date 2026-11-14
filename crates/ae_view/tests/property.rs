//! Property test of the rule executor against an oracle written from the
//! documented rules, not from the executor's code: random worlds over 60
//! ticks with movement, sensors, churn (despawned entities replaced by
//! recycled ids: same index, new generation) and two observers, for a
//! plain table and a ghosting one. Every tick, for every observer:
//! * `Never` fields have no column;
//! * every row's state is what the observer's coverage says of its entity
//!   (own, line of sight, radar), and every entity in that state has
//!   exactly one row;
//! * a field holds a value exactly where the class table allows it, hidden
//!   values are 0; shown values equal the sim's (blip positions within the
//!   wobble; ghost values, the position included, equal the entity's last
//!   sighting in line of sight, and no ghost sits on a cell in sight);
//! * no word in any column equals either half of any sim entity id;
//! * view ids are unique and ascending, kept while an entity stays in view,
//!   never reused, and new for a recycled id or a return after absence.

use ae_sensors::{Coverage, SensorGrid, Sensors};
use ae_view::{Frame, Observer, RowState, ViewClass, ViewSpec};
use std::collections::{BTreeMap, BTreeSet};

const EXTENT: f32 = 2000.0;
const CELL: f32 = 20.0;
const WOBBLE: f32 = 30.0;

/// Field order of the test tables.
const X: usize = 0;
const Y: usize = 1;
const TAG: usize = 2;
const FIELDS: [(&str, ViewClass); 7] = [
    ("x", ViewClass::RadarDegraded),
    ("y", ViewClass::RadarDegraded),
    ("tag", ViewClass::Always),
    ("hp", ViewClass::Los),
    ("secret", ViewClass::OwnerOnly),
    ("plan", ViewClass::Never),
    ("kind", ViewClass::Los), // ViewClass::Ghost in the ghosting table
];

/// The documented class table (independent of `ViewClass::shows`).
fn allowed(class: ViewClass, state: RowState) -> bool {
    use RowState as S;
    use ViewClass as C;
    match class {
        C::Never => false,
        C::OwnerOnly => state == S::Own,
        C::Los => matches!(state, S::Own | S::Seen),
        C::Ghost => matches!(state, S::Own | S::Seen | S::Ghost),
        C::RadarDegraded => matches!(state, S::Own | S::Seen | S::Blip | S::Ghost),
        C::Always => true,
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self, max: f32) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * max
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[derive(Clone)]
struct Unit {
    id: u64,
    x: f32,
    y: f32,
    ally: u32,
    los: f32,
    radar: f32,
    /// Fields 2.. (tag, hp, secret, plan, kind).
    rest: [u32; 5],
    /// Moves (false: static, like a building).
    mobile: bool,
}

/// Entity ids: index in the high-bit range, generation in the high word,
/// so neither half can collide with the small test values or float bits.
fn entity(index: u32, generation: u32) -> u64 {
    (0x9000_0000u64 + generation as u64) << 32 | (0x8000_0000u64 + index as u64)
}

struct World {
    units: Vec<Unit>,
    rng: Rng,
    serial: u32,
    generation: BTreeMap<u32, u32>,
    free: Vec<u32>,
}

impl World {
    fn new(seed: u64, n: usize) -> Self {
        let mut w = World { units: Vec::new(), rng: Rng(seed), serial: 0, generation: BTreeMap::new(), free: Vec::new() };
        for i in 0..n as u32 {
            w.free.push(i);
        }
        for _ in 0..n {
            w.spawn();
        }
        w
    }

    fn spawn(&mut self) {
        let index = self.free.remove(0);
        let generation = *self.generation.entry(index).and_modify(|g| *g += 1).or_insert(0);
        self.serial += 1;
        let r = &mut self.rng;
        let ally = r.below(3) as u32;
        let unit = Unit {
            id: entity(index, generation),
            x: r.f(EXTENT),
            y: r.f(EXTENT),
            ally,
            los: 60.0 + r.f(200.0),
            radar: if r.below(4) == 0 { 400.0 + r.f(400.0) } else { 0.0 },
            rest: [self.serial, r.below(100) as u32, 7000 + r.below(100) as u32, 9000 + r.below(100) as u32, r.below(10) as u32],
            mobile: r.below(3) != 0,
        };
        self.units.push(unit);
    }

    fn step(&mut self, tick: u64) {
        for u in &mut self.units {
            if u.mobile {
                u.x = (u.x + self.rng.f(120.0) - 60.0).clamp(0.0, EXTENT - 1.0);
                u.y = (u.y + self.rng.f(120.0) - 60.0).clamp(0.0, EXTENT - 1.0);
            }
            u.rest[1] = self.rng.below(100) as u32;
        }
        // Churn: a few despawned, as many spawned (recycling their indices).
        if tick.is_multiple_of(5) {
            for _ in 0..4 {
                let i = self.rng.below(self.units.len() as u64) as usize;
                let gone = self.units.swap_remove(i);
                self.free.push((gone.id & 0xFFFF_FFFF) as u32 - 0x8000_0000);
            }
            for _ in 0..4 {
                self.spawn();
            }
        }
        // Table order changes too.
        let n = self.units.len();
        for i in 0..n {
            let j = self.rng.below(n as u64) as usize;
            self.units.swap(i, j);
        }
    }
}

/// What the oracle remembers per observer.
#[derive(Default)]
struct Memory {
    /// Last sighting in line of sight, by tag: values of every field.
    sighted: BTreeMap<u32, [u32; 7]>,
    /// Entity → view id while continuously in view (non-ghost rows).
    ids: BTreeMap<u64, u32>,
    used: BTreeSet<u32>,
    /// Tag → id of last tick's ghost rows (a returning entity takes it back).
    ghost_ids: BTreeMap<u32, u32>,
}

fn words(u: &Unit) -> [u32; 7] {
    [u.x.to_bits(), u.y.to_bits(), u.rest[0], u.rest[1], u.rest[2], u.rest[3], u.rest[4]]
}

fn run(seed: u64, ghosting: bool) -> (usize, usize, usize) {
    let mut fields: Vec<(String, ViewClass)> = FIELDS.iter().map(|(n, c)| (n.to_string(), *c)).collect();
    if ghosting {
        fields[6].1 = ViewClass::Ghost;
    }
    let spec = ViewSpec { fields, x: X, y: Y, wobble: WOBBLE, wobble_ticks: 10 };
    let grid = SensorGrid::new(EXTENT, EXTENT, CELL);
    let mut world = World::new(seed, 300);
    let mut observers = [Observer::new(0, seed ^ 0xA), Observer::new(1, seed ^ 0xB)];
    let mut memory = [Memory::default(), Memory::default()];
    let (mut ghosts_seen, mut blips_seen, mut different) = (0, 0, 0);
    for tick in 1..=60u64 {
        world.step(tick);
        let u = &world.units;
        let (xs, ys): (Vec<f32>, Vec<f32>) = u.iter().map(|u| (u.x, u.y)).unzip();
        let allies: Vec<u32> = u.iter().map(|u| u.ally).collect();
        let (los, radar): (Vec<f32>, Vec<f32>) = u.iter().map(|u| (u.los, u.radar)).unzip();
        let mut coverage = Coverage::new(grid, 3);
        coverage.stamp(&Sensors { x: &xs, y: &ys, ally: &allies, los: &los, radar: &radar }, false);
        let entities: Vec<u64> = u.iter().map(|u| u.id).collect();
        let columns: Vec<Vec<u32>> = (0..7).map(|f| u.iter().map(|u| words(u)[f]).collect()).collect();
        let column_refs: Vec<&[u32]> = columns.iter().map(Vec::as_slice).collect();
        let frame = Frame::new(tick, &entities, &column_refs, &allies, &coverage);
        let id_halves: BTreeSet<u32> = entities.iter().flat_map(|e| [*e as u32, (*e >> 32) as u32]).collect();
        let by_tag: BTreeMap<u32, &Unit> = u.iter().map(|u| (u.rest[0], u)).collect();

        let mut lens = Vec::new();
        for (o, (observer, mem)) in observers.iter_mut().zip(memory.iter_mut()).enumerate() {
            let table = observer.build(&spec, &frame);
            let at = |what: &str| format!("seed {seed} ghosting {ghosting} tick {tick} observer {o}: {what}");
            lens.push(table.len());
            // Layout: every field but `Never`, in order.
            let want_fields: Vec<usize> = (0..7).filter(|&f| spec.fields[f].1 != ViewClass::Never).collect();
            assert_eq!(table.fields, want_fields, "{}", at("layout"));
            assert!(table.column(5).is_none(), "{}", at("a Never column exists"));
            // Ids unique, ascending.
            assert!(table.ids.windows(2).all(|w| w[0] < w[1]), "{}", at("ids ascending and unique"));
            // No sim id half anywhere.
            for col in &table.columns {
                assert!(col.iter().all(|w| !id_halves.contains(w)), "{}", at("a sim id reached a column"));
            }

            let expected_state = |u: &Unit| -> Option<RowState> {
                if u.ally == o as u32 {
                    return Some(RowState::Own);
                }
                match coverage.sees(o, u.x, u.y) {
                    (true, _) => Some(RowState::Seen),
                    (false, true) => Some(RowState::Blip),
                    _ => None,
                }
            };
            let tag_col = table.column(TAG).unwrap();
            let mut rows_by_tag: BTreeMap<u32, usize> = BTreeMap::new();
            let mut ids_now: BTreeMap<u64, u32> = BTreeMap::new();
            let mut ghost_ids: BTreeMap<u32, u32> = BTreeMap::new();
            for r in 0..table.len() {
                let state = table.state[r];
                let tag = tag_col[r];
                assert!(table.mask[r] & (1 << want_fields.iter().position(|&f| f == TAG).unwrap()) != 0, "{}", at("tag (Always) is shown"));
                assert!(rows_by_tag.insert(tag, r).is_none() || state == RowState::Ghost, "{}", at("one row per entity"));
                let value = |f: usize| table.column(f).map(|c| c[r]);
                let shown = |f: usize| {
                    let c = want_fields.iter().position(|&w| w == f).unwrap();
                    table.mask[r] & (1 << c) != 0
                };
                if state == RowState::Ghost {
                    ghosts_seen += 1;
                    ghost_ids.insert(tag, table.ids[r]);
                    assert!(ghosting, "{}", at("a ghost in a table without Ghost fields"));
                    let last = mem.sighted.get(&tag).unwrap_or_else(|| panic!("{}", at("a ghost of something never seen")));
                    if let Some(u) = by_tag.get(&tag) {
                        assert!(expected_state(u).is_none(), "{}", at("a ghost of an entity in view"));
                    }
                    // The ghost's own position (from the table) is the last sighting's.
                    let (gx, gy) = (value(X).unwrap(), value(Y).unwrap());
                    assert!(shown(X) && shown(Y) && (gx, gy) == (last[X], last[Y]), "{}", at("a ghost without its last-seen position"));
                    assert!(!coverage.sees(o, f32::from_bits(gx), f32::from_bits(gy)).0, "{}", at("a ghost on a cell in line of sight"));
                    for &f in &want_fields {
                        let class = spec.fields[f].1;
                        assert_eq!(shown(f), allowed(class, state), "{}", at(&format!("field {f} shown in a ghost")));
                        assert_eq!(value(f).unwrap(), if allowed(class, state) { last[f] } else { 0 }, "{}", at(&format!("ghost field {f}")));
                    }
                    continue;
                }
                let u = by_tag.get(&tag).unwrap_or_else(|| panic!("{}", at("a row for an entity not in the frame")));
                assert_eq!(Some(state), expected_state(u), "{}", at("row state"));
                let truth = words(u);
                for &f in &want_fields {
                    let class = spec.fields[f].1;
                    assert_eq!(shown(f), allowed(class, state), "{}", at(&format!("field {f} ({class:?}) shown in {state:?}")));
                    let v = value(f).unwrap();
                    if !allowed(class, state) {
                        assert_eq!(v, 0, "{}", at("a hidden value is not 0"));
                    } else if state == RowState::Blip && (f == X || f == Y) {
                        blips_seen += 1;
                        let d = (f32::from_bits(v) - f32::from_bits(truth[f])).abs();
                        assert!(d <= WOBBLE * 1.0001, "{}", at(&format!("blip off by {d}")));
                    } else {
                        assert_eq!(v, truth[f], "{}", at(&format!("field {f}")));
                    }
                }
                // Ids: kept while in view, never reused otherwise.
                let id = table.ids[r];
                match mem.ids.get(&u.id) {
                    Some(&old) => assert_eq!(id, old, "{}", at("an entity in view changed id")),
                    None => assert!(!mem.used.contains(&id) || mem.ghost_ids.get(&tag) == Some(&id), "{}", at("a fresh row reused an id that was not its ghost's")),
                }
                ids_now.insert(u.id, id);
                if state == RowState::Seen || state == RowState::Own {
                    mem.sighted.insert(tag, truth);
                }
            }
            // Completeness: every entity in view has a row.
            for u in u.iter() {
                if expected_state(u).is_some() {
                    assert!(rows_by_tag.contains_key(&u.rest[0]), "{}", at("an entity in view has no row"));
                }
            }
            mem.used.extend(table.ids.iter().copied());
            mem.ids = ids_now;
            mem.ghost_ids = ghost_ids;
        }
        different += (lens[0] != lens[1]) as usize;
    }
    (ghosts_seen, blips_seen, different)
}

#[test]
fn views_reveal_exactly_what_the_rules_allow() {
    let mut totals = (0, 0, 0);
    for seed in [0x1234_5678, 0xDEAD_BEEF, 0x0BAD_F00D, 0x5EED] {
        for ghosting in [false, true] {
            let (g, b, d) = run(seed, ghosting);
            totals = (totals.0 + g, totals.1 + b, totals.2 + d);
        }
    }
    // The checks were exercised: ghosts, blips, observers that differ.
    assert!(totals.0 > 50 && totals.1 > 50 && totals.2 > 50, "ghost rows {}, blip positions {}, ticks where the observers differ {}", totals.0, totals.1, totals.2);
}
