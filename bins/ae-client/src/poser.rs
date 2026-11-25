//! Animation records for the renderer: each posed unit's `Anim` channels,
//! as the sim left them after a tick, packed into what the GPU's pose pass
//! needs to evaluate its model's clips and manual joints itself
//! (`ae_render::units::RECORD_CELLS`): writes to the units' animation table
//! (`ae_render::units::AnimWrites`, kept by the `UnitFeed`). Nothing is
//! posed on the CPU: the pose is presentation, a function of the channels,
//! and the sim answers its own joint questions (`Rig::dir_in`).
//!
//! A posed unit has a slot in the table, the same while it stays posed:
//! two records, each stamped with its tick. The GPU draws the unit between
//! their poses at the drawn time, a tick behind as positions are (held at
//! a record before or past them). Each tick the unit's record is compared
//! with the newer one: the same (a unit standing still, a clip held), and
//! nothing is written; otherwise it goes over the older record when the
//! newer is the last tick's (one record: only its cells that differ), or,
//! when the newer is older than that (nothing changed since), the newer is
//! stamped the last tick and the new one goes over the older. A unit new
//! to the table gets both records at its channels (no motion). A unit is
//! posed when its class is drawn with a model and its `Anim` names that
//! model. Units are packed in parallel (rayon), by slot, each record made
//! straight from the unit's `Anim` in the sim's column (borrowed, not
//! copied: `ae_sim::runtime::Runtime::chunks`).

use ae_render::units::{AnimWrites, Cell, NO_POSE, POSE_CELLS, POSE_MANUAL, RECORD_CELLS};
use ae_sdk::anim::{Anim, LAYERS, MANUAL, manual_flags, manual_kind};
use rayon::prelude::*;

/// A unit posed at the last stream.
#[derive(Clone, Copy)]
struct Unit {
    key: u64,
    model: u64,
    slot: u32,
}

/// A slot's work in a stream.
#[derive(Clone, Copy)]
struct Job<'a> {
    anim: &'a Anim,
    /// New to its slot: both records at its channels.
    fresh: bool,
}

const ALL_CELLS: u64 = (1 << POSE_CELLS) - 1;

pub struct Poser {
    /// Per stream class: the model id of the model it is drawn with (one
    /// the sim has a rig for).
    classes: Vec<Option<u64>>,
    /// Units posed at the last stream, ascending by key.
    units: Vec<Unit>,
    /// Slots in use or free; free ones.
    slots: usize,
    free: Vec<u32>,
    /// Per slot: its two records as written (`POSE_CELLS`).
    table: Vec<Cell>,
    /// Per slot: a bit per cell written this stream.
    written: Vec<u64>,
    /// Per slot: either record shows a manual joint ([`POSE_MANUAL`]).
    manual: Vec<bool>,
    /// A stream was made (the first one writes the whole table).
    started: bool,
    /// The next stream writes the whole table ([`Self::write_whole`]).
    whole_next: bool,
    /// Scratch, kept from stream to stream: per unit its model if posed and
    /// its kept slot; per slot its unit (`u32::MAX`: none) and freshness;
    /// per slot the cells it wrote.
    models: Vec<Option<u64>>,
    kept: Vec<Option<u32>>,
    job_of: Vec<(u32, bool)>,
    /// Per block of [`BLOCK`] slots: the cells it wrote (places, values).
    blocks: Vec<(Vec<u32>, Vec<Cell>)>,
    /// The last stream's units, for their memory.
    spare_units: Vec<Unit>,
}

/// Slots a thread packs at a time.
const BLOCK: usize = 512;

impl Poser {
    /// `classes`: each stream class's model id (`None`: not posed).
    pub fn new(classes: Vec<Option<u64>>) -> Self {
        Self {
            classes,
            units: Vec::new(),
            slots: 0,
            free: Vec::new(),
            table: Vec::new(),
            written: Vec::new(),
            manual: Vec::new(),
            started: false,
            whole_next: false,
            models: Vec::new(),
            kept: Vec::new(),
            job_of: Vec::new(),
            blocks: Vec::new(),
            spare_units: Vec::new(),
        }
    }

    /// Pack `units` (key, class, channels; ascending by key) at `tick`:
    /// each one's `pose` attribute into `poses` (cleared first;
    /// [`NO_POSE`] when it is not posed; [`POSE_MANUAL`] set when either
    /// of its records shows a manual joint), the cells that changed into
    /// `writes` (replacing what it held; its memory is reused).
    pub fn pose(&mut self, tick: u64, units: &[(u64, u32, &Anim)], poses: &mut Vec<u32>, writes: &mut AnimWrites) {
        let now = tick as u32;
        let whole = !self.started || self.whole_next;
        writes.reset = whole;
        (self.started, self.whole_next) = (true, false);
        // Each unit's model, if posed.
        let mut models = std::mem::take(&mut self.models);
        models.clear();
        models.extend(
            units
                .iter()
                .map(|&(_, class, a)| match self.classes.get(class as usize) {
                    Some(&Some(model)) if a.model == model => Some(model),
                    _ => None,
                }),
        );
        // Units that stay keep their slots; the others' slots are freed
        // before new units take slots.
        let mut kept = std::mem::take(&mut self.kept);
        kept.clear();
        kept.resize(units.len(), None);
        let mut at = 0;
        let mut last = std::mem::replace(&mut self.units, std::mem::take(&mut self.spare_units));
        self.units.clear();
        for &unit in &last {
            while at < units.len() && units[at].0 < unit.key {
                at += 1;
            }
            match units.get(at) {
                Some(u) if u.0 == unit.key && models[at] == Some(unit.model) => kept[at] = Some(unit.slot),
                _ => self.free.push(unit.slot),
            }
        }
        last.clear();
        self.spare_units = last;
        let mut job_of = std::mem::take(&mut self.job_of);
        poses.clear();
        for (i, &(key_of, _, _)) in units.iter().enumerate() {
            let Some(model) = models[i] else {
                poses.push(NO_POSE);
                continue;
            };
            let (slot, fresh) = match kept[i] {
                Some(slot) => (slot, false),
                None => (self.take_slot(), true),
            };
            if job_of.len() < self.slots {
                job_of.resize(self.slots, (u32::MAX, false));
            }
            job_of[slot as usize] = (i as u32, fresh);
            self.units.push(Unit {
                key: key_of,
                model,
                slot,
            });
            poses.push((slot as usize * POSE_CELLS) as u32);
        }
        job_of.resize(self.slots, (u32::MAX, false));
        // Each block of slots packs its units and lists the cells it wrote,
        // in slot order (in parallel; the lists keep their memory).
        let mut blocks = std::mem::take(&mut self.blocks);
        blocks.resize_with(self.slots.div_ceil(BLOCK), Default::default);
        self.table
            .par_chunks_mut(POSE_CELLS * BLOCK)
            .zip(self.written.par_chunks_mut(BLOCK))
            .zip(self.manual.par_chunks_mut(BLOCK))
            .zip(job_of.par_chunks_mut(BLOCK))
            .zip(blocks.par_iter_mut())
            .enumerate()
            .for_each(|(b, ((((cells, written), manual), jobs), (at, out)))| {
                at.clear();
                out.clear();
                for (s, (((cells, w), m), job)) in cells
                    .chunks_mut(POSE_CELLS)
                    .zip(written)
                    .zip(manual)
                    .zip(jobs)
                    .enumerate()
                {
                    let (i, fresh) = std::mem::replace(job, (u32::MAX, false));
                    *w = match units.get(i as usize) {
                        Some(&(_, _, anim)) => pack_slot(&Job { anim, fresh }, now, cells),
                        None => 0,
                    };
                    *m = cells[0][1].max(cells[RECORD_CELLS][1]) > 0;
                    let base = (b * BLOCK + s) * POSE_CELLS;
                    let mut bits = *w;
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        at.push((base + k) as u32);
                        out.push(cells[k]);
                        bits &= bits - 1;
                    }
                }
            });
        // Their lists in block order (`writes` keeps its memory from
        // stream to stream, see `UnitFeed::publish_reusing`).
        writes.at.clear();
        writes.cells.clear();
        for (at, cells) in &blocks {
            writes.at.extend_from_slice(at);
            writes.cells.extend_from_slice(cells);
        }
        (self.models, self.kept, self.job_of, self.blocks) = (models, kept, job_of, blocks);
        // The cull's sphere for manual joints from the records just packed.
        for pose in poses.iter_mut().filter(|p| **p != NO_POSE) {
            if self.manual[*pose as usize / POSE_CELLS] {
                *pose |= POSE_MANUAL;
            }
        }
        if whole {
            writes.at.clear();
            writes.at.extend(0..self.table.len() as u32);
            writes.cells.clear();
            writes.cells.extend_from_slice(&self.table);
        }
    }

    /// The next stream writes the whole table, not only what changed (its
    /// reader missed writes: `UnitFeed::wants_whole`).
    pub fn write_whole(&mut self) {
        self.whole_next = true;
    }

    /// A free slot, or a new one.
    fn take_slot(&mut self) -> u32 {
        if let Some(slot) = self.free.pop() {
            return slot;
        }
        self.slots += 1;
        self.table.resize(self.slots * POSE_CELLS, [0; 4]);
        self.written.resize(self.slots, 0);
        self.manual.resize(self.slots, false);
        (self.slots - 1) as u32
    }
}

/// `a` at `tick` as a record (`ae_render::units::RECORD_CELLS`).
pub fn record(a: &Anim, tick: u32) -> [Cell; RECORD_CELLS] {
    let mut r = [[0; 4]; RECORD_CELLS];
    for i in 0..LAYERS {
        if a.clip[i] != 0 {
            r[1 + i] = [
                a.clip[i] as u32 | (a.mask[i] as u32) << 16,
                a.mode[i] as u32 | (a.kind[i] as u32) << 8,
                a.weight[i].to_bits(),
                a.phase[i].to_bits(),
            ];
        }
    }
    let mut n = 0;
    let mut instant = 0;
    for k in 0..MANUAL {
        if a.manual_kind[k] != manual_kind::NONE && a.manual_flags[k] & manual_flags::FRESH == 0 {
            r[5 + n / 4][n % 4] = a.bone[k] as u32 | (a.manual_kind[k] as u32) << 16;
            r[7 + n] = a.value[k].map(f32::to_bits);
            if matches!(
                a.manual_kind[k],
                manual_kind::TURN | manual_kind::MOVE | manual_kind::SCALE
            ) && a.top_speed[k] <= 0.0
                && a.manual_flags[k] & manual_flags::PENDING == 0
            {
                instant |= 1 << n;
            }
            n += 1;
        }
    }
    r[0] = [tick, n as u32, instant, 0];
    r
}

/// One unit's tick into its slot's `cells` (two records); a bit per cell
/// written.
fn pack_slot(job: &Job, now: u32, cells: &mut [Cell]) -> u64 {
    let then = now.wrapping_sub(1);
    let mut rec = record(job.anim, now);
    let (a, b) = cells.split_at_mut(RECORD_CELLS);
    if job.fresh {
        b.copy_from_slice(&rec);
        rec[0][0] = then;
        a.copy_from_slice(&rec);
        return ALL_CELLS;
    }
    // Which of the two is the newer (ticks wrap).
    let b_newer = b[0][0].wrapping_sub(a[0][0]) as i32 >= 0;
    let (older, newer, older_at, newer_at) = match b_newer {
        true => (a, b, 0, RECORD_CELLS),
        false => (b, a, RECORD_CELLS, 0),
    };
    let same = newer[0][1..] == rec[0][1..] && newer[1..] == rec[1..];
    if same {
        return 0;
    }
    let mut written = 0u64;
    // Written over the newer when it is this tick's already; over the
    // older otherwise, the newer stamped the last tick when it is older
    // than that (what it holds was so then).
    let (target, at) = match newer[0][0] {
        t if t == now => (newer, newer_at),
        t => {
            if t != then {
                newer[0][0] = then;
                written |= 1 << newer_at;
            }
            (older, older_at)
        }
    };
    for (k, (cell, new)) in target.iter_mut().zip(&rec).enumerate() {
        if cell != new {
            *cell = *new;
            written |= 1 << (at + k);
        }
    }
    written
}
