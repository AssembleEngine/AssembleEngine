//! Death's edge cases (M14 step 6, D): a kill before the death script's
//! first poll, a kill twice, and a kill after `life` returned. Three units;
//! their deaths keep the corpse (`Fate::Corpse`), so the test can read the
//! counters and `Dying`.
//!
//! Systems, in schedule order:
//! * `start` (tick 0): the three units.
//! * `fate`: unit 0 killed at tick 0 (before its first poll); unit 1 at
//!   ticks 1 and 3 (the second must do nothing); unit 2 at tick 2 (its
//!   `life` returned at tick 0).
//! * `scripts`: `resume_with_death`.

use ae_sdk::life::{self, Dying, Fate};
use ae_sdk::script::{self, Callins, Script, ScriptAlloc, Unit};
use ae_sdk::{Component, Run, Spawn, schedule, system};

pub const SLOTS: u32 = 4096;
ae_sdk::script_slots!(SLOTS, with_death(unit, death));

#[global_allocator]
static ALLOC: ScriptAlloc = ScriptAlloc;

#[repr(C)]
#[derive(Clone, Copy, Default, Component)]
pub struct Counters {
    pub deaths: u32,
    pub polls: u32,
}

/// Which of the three cases a unit is (its row index changes as units gain
/// `Dying`, so the systems must not key on it).
#[repr(C)]
#[derive(Clone, Copy, Default, Component)]
pub struct Kind {
    pub role: u32,
    pub reserved: u32,
}

#[system()]
fn start(run: &Run) {
    if run.tick() != 0 {
        return;
    }
    for role in 0..3 {
        Spawn::new()
            .with(&Counters::default())
            .with(&Kind { role, reserved: 0 })
            .with(&Script::default())
            .with(&Callins::default())
            .submit();
    }
}

#[system(read(Kind), write(Callins), get(Dying), after(start))]
fn fate(run: &Run) {
    let (kinds, posts, ids) = (run.read::<Kind>(), run.write::<Callins>(), run.entities());
    for i in run.rows() {
        let kill = match (kinds[i].role, run.tick()) {
            (0, 0) => Some(0),
            (1, 1 | 3) => Some(run.tick()),
            (2, 2) => Some(run.tick()),
            _ => None,
        };
        if let Some(at) = kill {
            life::kill(ids[i], &mut posts[i], at, kinds[i].role);
        }
    }
}

#[system(read(Kind), write(Script, Callins, Counters))]
fn scripts(run: &Run) {
    script::resume_with_death(run, unit, death, 5000);
}

async fn unit(u: Unit) {
    u.update(|c: &mut Counters| c.polls += 1);
    if u.get::<Kind>().role == 2 {
        // Life ends on its own at once; the composite then waits for the
        // kill.
        return;
    }
    u.sleep_ticks(1 << 40).await;
}

async fn death(u: Unit) -> Fate {
    u.update(|c: &mut Counters| c.deaths += 1);
    Fate::Corpse
}

schedule!(start, fate, scripts);
