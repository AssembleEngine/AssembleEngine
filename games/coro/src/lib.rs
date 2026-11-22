//! Coro: the unit-coroutine spike (M14 step 4). Every unit runs a plain
//! Rust `async fn` ([`unit`]) for a few rounds: sleeps of varied length,
//! a nested `async fn` waiting tick by tick, heap use inside a poll, a
//! spawn from inside a resumed poll, and locals (a running value, a
//! histogram) kept across every `await` in the future's slot. Its results
//! go to its `Counter`, so the world digest covers what the scripts did.
//!
//! Game parameter: bits 0..32 the unit count (a quarter more join at tick
//! [`WAVE`]); bit 32 [`BAD_READ`], bit 33 [`BAD_DROP`]: break the heap rule
//! (a `Vec` kept across an `await`, then read and forgotten, or dropped).
//!
//! Systems, in schedule order:
//! * `start`: the units (tick 0 and [`WAVE`]), each with a `Script` and a
//!   `Counter`.
//! * `scripts`: `ae_sdk::script::resume` over the units.
//! * `fade`: sparks (spawned by scripts) despawned [`SPARK_TICKS`] after
//!   they were made.

use ae_sdk::script::{self, Script, ScriptAlloc, Unit};
use ae_sdk::{Component, Entity, Run, Spawn, schedule, system};

/// Script slots (entity indices below it), each the size of [`unit`]'s
/// future.
pub const SLOTS: u32 = 16384;
ae_sdk::script_slots!(SLOTS, unit);

#[global_allocator]
static ALLOC: ScriptAlloc = ScriptAlloc;

/// The tick more units join.
pub const WAVE: u64 = 40;
pub const SPARK_TICKS: u64 = 4;
pub const BAD_READ: u64 = 1 << 32;
pub const BAD_DROP: u64 = 1 << 33;

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Counter {
    pub sum: u64,
    /// Ticks slept, by round.
    pub hist: [u32; 8],
    pub rounds: u32,
    /// Heap blocks kept across an `await` that read back wrong.
    pub bad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Spark {
    pub from: Entity,
    pub born: u64,
    pub value: u64,
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[system()]
fn start(run: &Run) {
    let n = run.param() as u32;
    let count = match run.tick() {
        0 => n,
        WAVE => n / 4,
        _ => return,
    };
    for _ in 0..count {
        Spawn::new()
            .with(&Script::default())
            .with(&Counter {
                sum: 0,
                hist: [0; 8],
                rounds: 0,
                bad: 0,
            })
            .submit();
    }
}

/// `n` ticks, one at a time, folding each into the unit's sum.
async fn pulse(u: Unit, seed: u64, n: u64) {
    let mut x = seed;
    for _ in 0..n {
        u.next_tick().await;
        x = mix(x ^ u.tick());
        u.update(|c: &mut Counter| c.sum ^= x);
    }
}

/// A unit's script.
async fn unit(u: Unit) {
    let (me, param) = (u.entity(), u.param());
    let mut acc = mix(me.0);
    let mut hist = [0u32; 8];
    let rounds = 4 + (acc % 6) as u32;
    for k in 0..rounds {
        let wait = 1 + acc % 9;
        u.sleep_ticks(wait).await;
        hist[k as usize % 8] += wait as u32;
        acc = mix(acc ^ u.tick() ^ (k as u64) << 48);
        // Heap inside one poll: made and dropped before the next `await`.
        let folded = {
            let v: Vec<u64> = (0..acc % 50).map(|j| mix(acc.wrapping_add(j))).collect();
            v.iter().fold(0u64, |a, &x| a.wrapping_add(x))
        };
        pulse(u, folded, 1 + acc % 3).await;
        if k % 2 == 0 {
            Spawn::new()
                .with(&Spark {
                    from: me,
                    born: u.tick(),
                    value: acc,
                })
                .submit();
        }
        u.update(|c: &mut Counter| {
            c.rounds += 1;
            c.hist = hist;
            c.sum = c.sum.wrapping_add(acc);
        });
        if k == 1 && param & (BAD_READ | BAD_DROP) != 0 {
            // Against the rule: a block kept across an `await`.
            let kept = vec![7u64; 4];
            u.next_tick().await;
            let wrong = kept.iter().any(|&x| x != 7) as u32;
            u.update(|c: &mut Counter| c.bad += wrong);
            if param & BAD_DROP == 0 {
                core::mem::forget(kept);
            }
        }
    }
}

#[system(write(Script, Counter))]
fn scripts(run: &Run) {
    script::resume(run, unit);
}

#[system(read(Spark))]
fn fade(run: &Run) {
    let (sparks, ids) = (run.read::<Spark>(), run.entities());
    for i in run.rows() {
        if run.tick() >= sparks[i].born + SPARK_TICKS {
            ae_sdk::despawn(ids[i]);
        }
    }
}

schedule!(start, scripts, fade);
