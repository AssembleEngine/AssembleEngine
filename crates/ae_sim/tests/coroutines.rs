//! Unit coroutines (`games/coro`, M14 step 4): a plain Rust `async fn` per
//! unit, its future in a host-owned slot (`ae_sdk::script`), resumed by an
//! ordinary parallel system in small work ranges, so a future made on one
//! worker instance is resumed on others.
//!
//! Checked at 1, 2, 4 and 8 workers (most first):
//! * the world digest, and a hash of the whole slot arena twice mid-run
//!   (the suspended futures' bytes: what a save would store) are the same;
//! * with 2 or more workers, futures were resumed on another instance than
//!   the poll before (`Script::moved`);
//! * every script ran all its rounds and finished; no heap block came back
//!   wrong.
//!
//! Then the heap rule, broken on purpose: a `Vec` kept across an `await`
//! reads back as poison (`script-checks`), and dropping it traps.
//!
//! Run on its own: `cargo test -p ae_sim --test coroutines -- --nocapture`.
//! The column arena is process-wide, so this file holds one test.

use ae_sim::runtime::{Runtime, RuntimeConfig, World};
use anyhow::{Result, ensure};
use xxhash_rust::xxh3::xxh3_64;

const UNITS: u64 = 3000;
const TICKS: u64 = 200;
/// Ticks the slot arena is hashed after.
const SNAPSHOTS: [u64; 2] = [30, 80];
const BAD_READ: u64 = 1 << 32;
const BAD_DROP: u64 = 1 << 33;

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

#[derive(Debug)]
struct Outcome {
    digest: u128,
    slots: [u64; 2],
    units: usize,
    done: usize,
    polls: u64,
    moved: u64,
    rounds: u64,
    bad: u64,
    /// Bytes of a future (the largest).
    size: u32,
}

fn inspect(rt: &Runtime, w: &World) -> Result<Outcome> {
    let scripts = rt.rows(w, "Script")?;
    let counters = rt.rows(w, "Counter")?;
    ensure!(scripts.len() == counters.len(), "every unit has a Script and a Counter");
    // Script { state, polls: u32, due: u64, size, moved, last, reserved: u32 };
    // Counter { sum: u64, hist: [u32; 8], rounds, bad: u32 }.
    Ok(Outcome {
        digest: 0,
        slots: [0; 2],
        units: scripts.len(),
        done: scripts.iter().filter(|(_, b)| u32_at(b, 0) == 2).count(),
        polls: scripts.iter().map(|(_, b)| u32_at(b, 4) as u64).sum(),
        moved: scripts.iter().map(|(_, b)| u32_at(b, 20) as u64).sum(),
        size: scripts.iter().map(|(_, b)| u32_at(b, 16)).max().unwrap_or(0),
        rounds: counters.iter().map(|(_, b)| u32_at(b, 40) as u64).sum(),
        bad: counters.iter().map(|(_, b)| u32_at(b, 44) as u64).sum(),
    })
}

fn run(rt: &mut Runtime, workers: usize, param: u64, ticks: u64) -> Result<Outcome> {
    let mut world = rt.world(param)?;
    let mut slots = [0; 2];
    for t in 1..=ticks {
        rt.tick(&mut world, workers)?;
        if let Some(k) = SNAPSHOTS.iter().position(|&s| s == t) {
            slots[k] = xxh3_64(rt.script_slots());
        }
    }
    let mut out = inspect(rt, &world)?;
    out.digest = rt.digest(&world)?.total;
    out.slots = slots;
    Ok(out)
}

#[test]
fn coroutines_resume_anywhere_with_the_same_result() -> Result<()> {
    let wasm = ae_sim::build::game("coro")?;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let max = cores.clamp(1, 8);
    // Small ranges: about 30 jobs per tick for the scripts, so ranges change
    // workers from tick to tick.
    let cfg = RuntimeConfig {
        max_workers: max,
        range_rows: 128,
        parallel_min_rows: 256,
        ..RuntimeConfig::default()
    };
    let mut rt = Runtime::new(&wasm, cfg)?;

    let mut order = vec![max];
    order.extend([1, 2, 4, 8].into_iter().filter(|&w| w < max));
    let mut base: Option<(usize, Outcome)> = None;
    for &w in &order {
        let started = std::time::Instant::now();
        let r = run(&mut rt, w, UNITS, TICKS)?;
        eprintln!(
            "{w} worker(s): digest {:016x}, slots {:016x} {:016x}; {} units, {} done, {} polls, {} resumed on another instance, {} rounds, future {} bytes, {:.2} s",
            r.digest as u64,
            r.slots[0],
            r.slots[1],
            r.units,
            r.done,
            r.polls,
            r.moved,
            r.rounds,
            r.size,
            started.elapsed().as_secs_f64()
        );
        ensure!(
            r.units == (UNITS + UNITS / 4) as usize,
            "{w} workers: {} units",
            r.units
        );
        ensure!(
            r.done == r.units,
            "{w} workers: {} of {} scripts finished",
            r.done,
            r.units
        );
        ensure!(r.bad == 0, "{w} workers: {} heap blocks read back wrong", r.bad);
        ensure!(
            r.slots[0] != r.slots[1],
            "{w} workers: the slot arena did not change between snapshots"
        );
        ensure!(
            w == 1 || r.moved > 0,
            "{w} workers: no future was resumed on another instance"
        );
        match &base {
            None => base = Some((w, r)),
            Some((bw, b)) => {
                ensure!(
                    r.digest == b.digest,
                    "{w} workers: digest {:032x}, {bw} workers {:032x}",
                    r.digest,
                    b.digest
                );
                ensure!(
                    r.slots == b.slots,
                    "{w} workers: slot arena {:x?}, {bw} workers {:x?}",
                    r.slots,
                    b.slots
                );
                ensure!(
                    (r.polls, r.rounds) == (b.polls, b.rounds),
                    "{w} workers: polls or rounds differ from {bw} workers"
                );
            }
        }
    }

    // A block kept across an `await` reads back as poison...
    let r = run(&mut rt, max, UNITS | BAD_READ, 60)?;
    eprintln!("kept across an await: {} of {} read back wrong", r.bad, r.units);
    ensure!(
        r.bad > 0,
        "no block kept across an await read back wrong (is the arena poisoned?)"
    );
    // ...and freeing it traps (last: the trapped instance is not reused).
    let e = run(&mut rt, max, UNITS | BAD_DROP, 60)
        .err()
        .ok_or_else(|| anyhow::anyhow!("dropping a block kept across an await did not trap"))?;
    let text = format!("{e:?}");
    eprintln!(
        "dropped across an await: {}",
        text.lines()
            .find(|l| l.contains("heap_survived"))
            .unwrap_or(text.lines().next().unwrap_or(""))
    );
    ensure!(
        text.contains("heap_survived_an_await"),
        "the trap is not the heap check: {text}"
    );
    Ok(())
}
