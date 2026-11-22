//! Guest heap allocation from parallel systems (`games/alloc`).
//!
//! Two parallel systems in one layer allocate on every row: growing vectors,
//! sorts, B-tree maps, strings, and blocks that live until the row's next
//! run, so most are freed by a different worker instance than the one that
//! allocated them; one block in 128 is 64-256 KiB, so the heap grows from
//! several instances at once. Each guest computation is checked two ways in
//! the guest (heap and stack), each held block's contents before it is
//! freed, and the world digest at 1, 2, 4 and 8 workers. The guest memory size
//! shows that the heap really grew under parallel load and then stopped
//! growing; the count of `sched_yield` calls shows instances really waited
//! for wasi-libc's heap lock.
//!
//! The guest heap is not reset between worlds, so each world after the
//! first starts with the previous world's held blocks still allocated
//! (about 14 MiB here): a leak of this probe, not of the runtime.
//!
//! Run on its own any time: `cargo test -p ae_sim --test guest_alloc -- --nocapture`
//! (add `--release` for a faster, longer-running variant via `AE_ALLOC_TICKS`).
//!
//! The column arena is process-wide, so this file holds one test.

use ae_sim::runtime::{Runtime, RuntimeConfig, World};
use anyhow::{Result, ensure};

const ROWS: u64 = 4000;
const MIB: f64 = (1 << 20) as f64;

fn ticks() -> u64 {
    std::env::var("AE_ALLOC_TICKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(90)
}

struct Outcome {
    digest: u128,
    bad: u64,
    rows: usize,
    /// Rows holding a block of 64 KiB or more at the end.
    big: usize,
    /// Guest memory after tick 0, at the middle and at the end, in bytes.
    memory: [usize; 3],
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn inspect(rt: &Runtime, world: &World) -> Result<(u64, usize, usize)> {
    let lists = rt.rows(world, "Lists")?;
    let blocks = rt.rows(world, "Blocks")?;
    ensure!(lists.len() == blocks.len(), "every probe row has both components");
    // Lists { checksum: u64, len: u32, bad: u32 }; Blocks { checksum: u64,
    // held: u32, held_len: u32, held_seed: u32, bad: u32 }.
    let bad = lists.iter().map(|(_, b)| u32_at(b, 12) as u64).sum::<u64>()
        + blocks.iter().map(|(_, b)| u32_at(b, 20) as u64).sum::<u64>();
    let big = blocks.iter().filter(|(_, b)| u32_at(b, 12) >= 64 << 10).count();
    ensure!(
        blocks.iter().all(|(_, b)| u32_at(b, 8) != 0),
        "every row holds a block after it ran"
    );
    Ok((bad, lists.len(), big))
}

fn run(rt: &mut Runtime, workers: usize, ticks: u64) -> Result<Outcome> {
    let mut world = rt.world(ROWS)?;
    let mut memory = [0; 3];
    for t in 0..ticks {
        rt.tick(&mut world, workers)?;
        if t == 0 {
            memory[0] = rt.memory_bytes();
        } else if t == ticks / 2 {
            memory[1] = rt.memory_bytes();
        }
    }
    memory[2] = rt.memory_bytes();
    let (bad, rows, big) = inspect(rt, &world)?;
    Ok(Outcome {
        digest: rt.digest(&world)?.total,
        bad,
        rows,
        big,
        memory,
    })
}

#[test]
fn guest_heap_is_safe_from_parallel_systems() -> Result<()> {
    let wasm = ae_sim::build::game("alloc")?;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let max = cores.clamp(1, 8);
    // Small ranges: about 30 jobs per system per tick, so every layer runs
    // both systems on every worker at once.
    let cfg = RuntimeConfig {
        max_workers: max,
        range_rows: 128,
        parallel_min_rows: 256,
        ..RuntimeConfig::default()
    };
    let mut rt = Runtime::new(&wasm, cfg)?;
    let ticks = ticks();
    let start = rt.memory_bytes();

    // Most workers first: the heap grows from empty under parallel load.
    let mut order = vec![max];
    order.extend([1, 2, 4, 8].into_iter().filter(|&w| w < max));
    let mut base: Option<(usize, Outcome)> = None;
    for (k, &w) in order.iter().enumerate() {
        let yields = ae_sim::imports::sched_yields();
        let started = std::time::Instant::now();
        let r = run(&mut rt, w, ticks)?;
        let secs = started.elapsed().as_secs_f64();
        let yields = ae_sim::imports::sched_yields() - yields;
        let [after0, mid, end] = r.memory;
        eprintln!(
            "{w} worker(s): digest {:016x}, {} rows, {} bad, {} big blocks held, {yields} lock waits, {:.2} s; guest memory {:.1} MiB after tick 0, {:.1} at tick {}, {:.1} at the end",
            r.digest as u64,
            r.rows,
            r.bad,
            r.big,
            secs,
            after0 as f64 / MIB,
            mid as f64 / MIB,
            ticks / 2,
            end as f64 / MIB
        );
        ensure!(
            r.bad == 0,
            "{w} workers: {} guest checks failed (heap corruption or a lost allocation)",
            r.bad
        );
        ensure!(r.rows == ROWS as usize, "{w} workers: {} probe rows", r.rows);
        ensure!(
            r.big > 0,
            "{w} workers: no big block held; the probe does not reach the grow path"
        );
        if k == 0 {
            ensure!(
                w == 1 || yields > 0,
                "{w} workers never contended for the heap lock; the probe is not parallel enough"
            );
            // The heap grew while the parallel systems ran (tick 0 only spawns)...
            ensure!(
                end > after0,
                "{w} workers: guest memory did not grow under parallel allocation ({after0} -> {end} bytes)"
            );
            // ...and stops growing once it covers the working set: the second
            // half of the run adds at most a quarter of what the first did.
            let (first, second) = (mid.saturating_sub(after0), end.saturating_sub(mid));
            ensure!(
                second <= first / 4 + (1 << 20),
                "{w} workers: guest memory keeps growing ({first} bytes in the first half, {second} in the second)"
            );
        }
        match &base {
            None => base = Some((w, r)),
            Some((bw, b)) => ensure!(
                r.digest == b.digest,
                "{w} workers: digest {:032x}, {bw} workers {:032x}",
                r.digest,
                b.digest
            ),
        }
    }
    eprintln!(
        "guest memory {:.1} MiB at start, {:.1} MiB after {} worlds",
        start as f64 / MIB,
        rt.memory_bytes() as f64 / MIB,
        order.len()
    );
    Ok(())
}
