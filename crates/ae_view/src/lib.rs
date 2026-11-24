//! The copy layer's rule executor: turns one end-of-tick capture of sim
//! columns into a view table per observer, keeping only what that
//! observer may know.
//!
//! **Rows.** For an observer of allyteam `a`, a captured entity is
//! * [`RowState::Own`] if its allyteam is `a`;
//! * else [`RowState::Seen`] if `a` has line of sight to its cell;
//! * else [`RowState::Blip`] if `a` has radar on its cell;
//! * else absent, unless the table ghosts (it has a [`ViewClass::Ghost`]
//!   field): an entity last seen in line of sight stays as a
//!   [`RowState::Ghost`] row with its last-seen values (the exact position
//!   included) until `a` has line
//!   of sight to that cell again and does not see it there (it moved, died
//!   or is back as a `Seen` row, which takes the ghost's id back).
//!
//! **Fields** (in a row, by class):
//!
//! | Class | Own | Seen | Blip | Ghost |
//! |---|---|---|---|---|
//! | `Never` | - | - | - | - |
//! | `OwnerOnly` | yes | - | - | - |
//! | `Los` | yes | yes | - | - |
//! | `Ghost` | yes | yes | - | last seen |
//! | `RadarDegraded` | yes | yes | yes, position wobbled | last seen |
//! | `Always` | yes | yes | yes | last seen |
//!
//! `Never` fields have no column at all. A hidden field reads 0 and its bit
//! in the row's `mask` is clear.
//!
//! **Ids.** Rows carry [`ViewId`]s, never sim entity ids. An observer gives
//! an entity a new id when it comes into view (in an order shuffled by the
//! observer's seed, so ids do not follow entity order) and keeps it while
//! it stays in view; a recycled entity id (a new generation) or an entity
//! that leaves view and comes back gets a new one, so ids carry no identity
//! beyond what the observer saw continuously. Rows come in id order.
//!
//! **Radar wobble.** A blip's position is moved by up to `wobble` world
//! units in x and y, by a hash of (observer seed, view id, tick bucket),
//! blended linearly across `wobble_ticks`, so blips drift smoothly instead
//! of jittering every tick. It is keyed on the view id, not the entity.

use ae_sensors::Coverage;

/// What an observer may see of a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ViewClass {
    /// Never copied (the default for fields nobody classified).
    Never = 0,
    Always = 1,
    OwnerOnly = 2,
    Los = 3,
    RadarDegraded = 4,
    Ghost = 5,
}

impl ViewClass {
    pub fn from_wire(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Never,
            1 => Self::Always,
            2 => Self::OwnerOnly,
            3 => Self::Los,
            4 => Self::RadarDegraded,
            5 => Self::Ghost,
            _ => return None,
        })
    }

    /// Whether the field shows in a row of this state.
    pub fn shows(self, state: RowState) -> bool {
        use RowState as S;
        use ViewClass as C;
        match (self, state) {
            (C::Never, _) => false,
            (_, S::Own) => true,
            (C::OwnerOnly, _) => false,
            (C::Los | C::Ghost | C::RadarDegraded | C::Always, S::Seen) => true,
            (C::RadarDegraded | C::Always, S::Blip) => true,
            (C::Ghost | C::RadarDegraded | C::Always, S::Ghost) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum RowState {
    Own = 0,
    Seen = 1,
    Blip = 2,
    Ghost = 3,
}

/// An observer's name for an entity.
pub type ViewId = u32;

/// One view table's layout: the captured fields, their classes, and which
/// two are the position.
#[derive(Clone, Debug)]
pub struct ViewSpec {
    pub fields: Vec<(String, ViewClass)>,
    /// Indices into `fields` of the world x and y (`f32` bits).
    pub x: usize,
    pub y: usize,
    /// Optional depth field (positive metres below the surface).
    pub submerged: Option<usize>,
    /// Radar wobble amplitude, world units, and the ticks it drifts over.
    pub wobble: f32,
    pub wobble_ticks: u32,
}

impl ViewSpec {
    /// The table ghosts when it has a `Ghost` field.
    pub fn ghosts(&self) -> bool {
        self.fields.iter().any(|(_, c)| *c == ViewClass::Ghost)
    }

    /// The fields that get a column: all but `Never`, in order.
    pub fn columns(&self) -> Vec<usize> {
        (0..self.fields.len())
            .filter(|&f| self.fields[f].1 != ViewClass::Never)
            .collect()
    }

    fn check(&self) {
        assert!(
            self.fields.len() <= 32,
            "a view table has at most 32 fields (row masks are u32)"
        );
        assert!(
            self.x < self.fields.len() && self.y < self.fields.len() && self.x != self.y,
            "position fields"
        );
        assert!(self.wobble_ticks > 0, "wobble_ticks must be positive");
    }
}

/// One tick's capture of a table, as the observers read it.
pub struct Frame<'a> {
    pub tick: u64,
    pub entities: &'a [u64],
    /// One column per spec field (raw 4-byte words), row `r` of each
    /// belonging to `entities[r]`.
    pub columns: &'a [&'a [u32]],
    /// Allyteam of each row.
    pub ally: &'a [u32],
    pub coverage: &'a Coverage,
    /// Rows by entity id (ascending), made once for every observer.
    by_entity: Vec<u32>,
}

impl<'a> Frame<'a> {
    pub fn new(
        tick: u64,
        entities: &'a [u64],
        columns: &'a [&'a [u32]],
        ally: &'a [u32],
        coverage: &'a Coverage,
    ) -> Self {
        let n = entities.len();
        assert!(
            columns.iter().all(|c| c.len() == n) && ally.len() == n,
            "frame columns differ in length"
        );
        let mut by_entity: Vec<u32> = (0..n as u32).collect();
        by_entity.sort_unstable_by_key(|&r| entities[r as usize]);
        assert!(
            by_entity
                .windows(2)
                .all(|w| entities[w[0] as usize] != entities[w[1] as usize]),
            "an entity appears twice in a frame"
        );
        Self {
            tick,
            entities,
            columns,
            ally,
            coverage,
            by_entity,
        }
    }
}

/// An observer's view of one table this tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViewTable {
    pub tick: u64,
    pub ids: Vec<ViewId>,
    pub state: Vec<RowState>,
    /// Per row, bit `c` set when column `c` holds a value.
    pub mask: Vec<u32>,
    /// One column per non-`Never` spec field ([`ViewSpec::columns`]).
    pub columns: Vec<Vec<u32>>,
    /// Spec field index of each column.
    pub fields: Vec<usize>,
}

impl ViewTable {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The column of spec field `field`, if it has one.
    pub fn column(&self, field: usize) -> Option<&[u32]> {
        self.fields
            .iter()
            .position(|&f| f == field)
            .map(|c| self.columns[c].as_slice())
    }
}

#[derive(Clone, Debug)]
struct Ghost {
    id: ViewId,
    /// The entity the ghost stands for, while it may still be in the
    /// frame (to take the id back if it shows again).
    entity: u64,
    cell: (u32, u32),
    /// Last-seen words of every spec field (only Ghost/Always are shown).
    values: Vec<u32>,
}

/// One observer of one table: its allyteam and what it remembers between
/// ticks (ids of entities in view, ghosts).
#[derive(Clone, Debug)]
pub struct Observer {
    pub ally: u32,
    seed: u64,
    next_id: ViewId,
    /// Entities in view last tick, ascending by entity: (entity, id).
    known: Vec<(u64, ViewId)>,
    /// Ghosting tables: the rows in line of sight last tick (ascending by
    /// entity), with their values; the ghosts now.
    last_seen: Vec<Ghost>,
    ghosts: Vec<Ghost>,
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A hash to `[-1, 1)`.
fn unit(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 23) as f32 - 1.0
}

impl Observer {
    /// `seed` keys the id shuffle and the radar wobble: give each observer
    /// its own (it is local, never synced).
    pub fn new(ally: u32, seed: u64) -> Self {
        Self {
            ally,
            seed,
            next_id: 1,
            known: Vec::new(),
            last_seen: Vec::new(),
            ghosts: Vec::new(),
        }
    }

    /// The sim entity behind view id `id`, while it is in view (not for
    /// ghosts). For trusted engine code turning a player's choice (made on
    /// view ids) into commands, which name sim entities; never hand it to
    /// untrusted code.
    pub fn entity_of(&self, id: ViewId) -> Option<u64> {
        self.known.iter().find(|(_, v)| *v == id).map(|(e, _)| *e)
    }

    /// The wobble of a blip: (dx, dy) world units.
    pub fn wobble(&self, spec: &ViewSpec, id: ViewId, tick: u64) -> (f32, f32) {
        let period = spec.wobble_ticks as u64;
        let (bucket, t) = (tick / period, (tick % period) as f32 / period as f32);
        let at = |b: u64, axis: u64| unit(mix(self.seed ^ mix((id as u64) << 32 | axis) ^ mix(b)));
        let lerp = |axis| at(bucket, axis) * (1.0 - t) + at(bucket + 1, axis) * t;
        (lerp(0) * spec.wobble, lerp(1) * spec.wobble)
    }

    /// Build this observer's table for a frame.
    pub fn build(&mut self, spec: &ViewSpec, frame: &Frame<'_>) -> ViewTable {
        spec.check();
        assert_eq!(frame.columns.len(), spec.fields.len(), "frame columns vs spec fields");
        let ally = self.ally as usize;
        let cov = frame.coverage;
        let submerged = |r| {
            spec.submerged
                .is_some_and(|f| f32::from_bits(frame.columns[f][r]) > 0.0)
        };
        let state_of = |r: usize| -> Option<RowState> {
            if frame.ally[r] == self.ally {
                return Some(RowState::Own);
            }
            let (x, y) = (
                f32::from_bits(frame.columns[spec.x][r]),
                f32::from_bits(frame.columns[spec.y][r]),
            );
            if submerged(r) {
                return cov
                    .sonar
                    .get(ally)
                    .filter(|bits| {
                        let (cx, cy) = cov.grid.cell(x, y);
                        bits.get(cx, cy)
                    })
                    .map(|_| RowState::Seen);
            }
            let (los, radar) = if ally < cov.allies() {
                cov.sees(ally, x, y)
            } else {
                (false, false)
            };
            if los {
                Some(RowState::Seen)
            } else if radar {
                Some(RowState::Blip)
            } else {
                None
            }
        };

        // Visible rows, merged against last tick's ids (both by entity).
        let mut visible: Vec<(u32, RowState, Option<ViewId>)> = Vec::with_capacity(frame.entities.len());
        let mut k = 0;
        for &r in &frame.by_entity {
            let Some(state) = state_of(r as usize) else { continue };
            let e = frame.entities[r as usize];
            while k < self.known.len() && self.known[k].0 < e {
                k += 1;
            }
            let id = (k < self.known.len() && self.known[k].0 == e).then(|| self.known[k].1);
            visible.push((r, state, id));
        }

        let entity_of = |v: &(u32, RowState, Option<ViewId>)| frame.entities[v.0 as usize];
        let find =
            |visible: &[(u32, RowState, Option<ViewId>)], e: u64| visible.binary_search_by_key(&e, entity_of).ok();
        let los = |cell: (u32, u32)| ally < cov.allies() && cov.los[ally].get(cell.0, cell.1);
        let ghosts = spec.ghosts();

        // Ghosts: back in view (the row takes the ghost's id), confirmed
        // gone (line of sight on its cell), or kept.
        let mut kept = Vec::new();
        for g in std::mem::take(&mut self.ghosts) {
            match find(&visible, g.entity) {
                Some(i) if visible[i].2.is_none() => visible[i].2 = Some(g.id),
                _ if los(g.cell) => {}
                _ => kept.push(g),
            }
        }

        // New ids for rows coming into view, in a seed-shuffled order.
        let mut fresh: Vec<usize> = (0..visible.len()).filter(|&i| visible[i].2.is_none()).collect();
        fresh.sort_unstable_by_key(|&i| mix(self.seed ^ entity_of(&visible[i])));
        for i in fresh {
            visible[i].2 = Some(self.next_id);
            self.next_id = self.next_id.wrapping_add(1).max(1);
        }

        // Rows in line of sight last tick and out of view now become ghosts,
        // unless the observer sees their cell (it saw them go).
        if ghosts {
            for g in std::mem::take(&mut self.last_seen) {
                if find(&visible, g.entity).is_none() && !los(g.cell) {
                    kept.push(g);
                }
            }
            self.last_seen = visible
                .iter()
                .filter(|v| v.1 == RowState::Seen && !submerged(v.0 as usize))
                .map(|v| {
                    let r = v.0 as usize;
                    let (x, y) = (
                        f32::from_bits(frame.columns[spec.x][r]),
                        f32::from_bits(frame.columns[spec.y][r]),
                    );
                    Ghost {
                        id: v.2.unwrap(),
                        entity: frame.entities[r],
                        cell: cov.grid.cell(x, y),
                        values: frame.columns.iter().map(|c| c[r]).collect(),
                    }
                })
                .collect();
        }
        self.ghosts = kept;
        self.known = visible.iter().map(|v| (entity_of(v), v.2.unwrap())).collect();

        // The table, in id order.
        let fields = spec.columns();
        let rows = visible.len() + self.ghosts.len();
        let mut out = ViewTable {
            tick: frame.tick,
            ids: Vec::with_capacity(rows),
            state: Vec::with_capacity(rows),
            mask: Vec::with_capacity(rows),
            columns: vec![Vec::with_capacity(rows); fields.len()],
            fields: fields.clone(),
        };
        #[derive(Clone, Copy)]
        enum Source<'g> {
            Row(u32, RowState),
            Ghost(&'g Ghost),
        }
        let mut order: Vec<(ViewId, Source)> = visible
            .iter()
            .map(|v| (v.2.unwrap(), Source::Row(v.0, v.1)))
            .chain(self.ghosts.iter().map(|g| (g.id, Source::Ghost(g))))
            .collect();
        order.sort_unstable_by_key(|o| o.0);
        for (id, source) in order {
            let state = match source {
                Source::Row(_, state) => state,
                Source::Ghost(_) => RowState::Ghost,
            };
            let word = |f: usize| match source {
                Source::Row(r, _) => frame.columns[f][r as usize],
                Source::Ghost(g) => g.values[f],
            };
            let wobble = (state == RowState::Blip).then(|| self.wobble(spec, id, frame.tick));
            let mut mask = 0u32;
            for (c, &f) in fields.iter().enumerate() {
                let class = spec.fields[f].1;
                let value = if class.shows(state) {
                    mask |= 1 << c;
                    match wobble {
                        Some((dx, _)) if f == spec.x => (f32::from_bits(word(f)) + dx).to_bits(),
                        Some((_, dy)) if f == spec.y => (f32::from_bits(word(f)) + dy).to_bits(),
                        _ => word(f),
                    }
                } else {
                    0
                };
                out.columns[c].push(value);
            }
            out.ids.push(id);
            out.state.push(state);
            out.mask.push(mask);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submerged_rows_require_sonar_and_leave_no_ghost() {
        use ae_sensors::SensorGrid;

        let spec = ViewSpec {
            fields: vec![
                ("x".into(), ViewClass::RadarDegraded),
                ("y".into(), ViewClass::RadarDegraded),
                ("depth".into(), ViewClass::Los),
                ("silhouette".into(), ViewClass::Ghost),
            ],
            x: 0,
            y: 1,
            submerged: Some(2),
            wobble: 1.0,
            wobble_ticks: 1,
        };
        let mut cov = Coverage::new(SensorGrid::new(64.0, 64.0, 32.0), 2);
        let mut observer = Observer::new(0, 0);
        let build = |observer: &mut Observer, cov: &Coverage, depth: f32| {
            let xy = [1.0f32.to_bits(); 2];
            let depths = [depth.to_bits(); 2];
            let columns: [&[u32]; 4] = [&xy, &xy, &depths, &[1, 1]];
            let frame = Frame::new(0, &[10, 20], &columns, &[0, 1], cov);
            let mut states = observer.build(&spec, &frame).state;
            states.sort_by_key(|s| *s as u8);
            states
        };
        cov.los[0].set(0, 0);
        assert_eq!(build(&mut observer, &cov, 0.0), [RowState::Own, RowState::Seen]);
        cov.los[0].clear();
        assert_eq!(build(&mut observer, &cov, 0.0), [RowState::Own, RowState::Ghost]);
        cov.los[0].set(0, 0);
        cov.radar[0].set(0, 0);
        assert_eq!(build(&mut observer, &cov, 4.0), [RowState::Own]);
        cov.los[0].clear();
        cov.sonar[0].set(0, 0);
        assert_eq!(build(&mut observer, &cov, 4.0), [RowState::Own, RowState::Seen]);
        cov.sonar[0].clear();
        assert_eq!(build(&mut observer, &cov, 4.0), [RowState::Own]);
    }

    #[test]
    fn wobble_is_bounded_smooth_and_keyed_by_observer_and_id() {
        let spec = ViewSpec {
            fields: vec![
                ("x".into(), ViewClass::RadarDegraded),
                ("y".into(), ViewClass::RadarDegraded),
            ],
            x: 0,
            y: 1,
            submerged: None,
            wobble: 40.0,
            wobble_ticks: 30,
        };
        let (a, b) = (Observer::new(0, 1), Observer::new(0, 2));
        let mut moved = 0.0f32;
        for tick in 0..300 {
            let (w0, w1) = (a.wobble(&spec, 7, tick), a.wobble(&spec, 7, tick + 1));
            assert!(w0.0.abs() <= 40.0 && w0.1.abs() <= 40.0);
            // At most 2 × amplitude per period, spread over its ticks.
            assert!(
                (w1.0 - w0.0).abs() <= 80.0 / 30.0 + 1e-3 && (w1.1 - w0.1).abs() <= 80.0 / 30.0 + 1e-3,
                "tick {tick}"
            );
            moved += (w1.0 - w0.0).abs();
        }
        assert!(moved > 40.0, "it drifts");
        assert_ne!(a.wobble(&spec, 7, 5), b.wobble(&spec, 7, 5), "observers differ");
        assert_ne!(a.wobble(&spec, 7, 5), a.wobble(&spec, 8, 5), "ids differ");
    }
}
