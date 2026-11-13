//! M0 risk spike: can N wasmtime instances of one Rust guest run system
//! kernels concurrently, in place, over one shared linear memory?
//!
//! Checks, in order:
//! 1. Isolation: each instance gets its own shadow stack and TLS block
//!    (with a negative control showing what happens without that setup).
//! 2. Correctness: the column bytes after T ticks are identical for every
//!    worker count, and identical to a native Rust run of the same kernel.
//! 3. Cost: host -> guest call overhead, per-row cost by chunk size, scaling
//!    with worker count, and the price of fuel and epoch interruption.
//!
//! `cargo run -p spike_parallel_guest --release`

use anyhow::{Context, Result, anyhow, ensure};
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use spike_parallel_guest::{GUEST_WASM, Guest, Metering, STACK_BYTES, engine, memory_base, spawn_guests};
use wasmtime::{Engine, Module, SharedMemory};

const ROWS: usize = 1 << 20;
const TICKS: usize = 30;
const RANGE_ROWS: usize = 1024;
const DT: f32 = 1.0 / 30.0;

/// Position and velocity columns inside guest memory, as offsets.
#[derive(Clone, Copy)]
struct Columns {
    pos: u32,
    vel: u32,
}

fn seed_value(row: usize, salt: u32) -> f32 {
    let mut h = (row as u32).wrapping_mul(0x9E37_79B9) ^ salt;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    (h % 20_001) as f32 / 10.0 - 1000.0
}

fn fill_native(pos: &mut [[f32; 2]], vel: &mut [[f32; 2]]) {
    for row in 0..pos.len() {
        pos[row] = [seed_value(row, 1), seed_value(row, 2)];
        vel[row] = [seed_value(row, 3) * 0.05, seed_value(row, 4) * 0.05];
    }
}

fn alloc_columns(memory: &SharedMemory, allocator: &mut Guest) -> Result<Columns> {
    let bytes = (ROWS * 8) as u32;
    let columns = Columns { pos: allocator.alloc(bytes)?, vel: allocator.alloc(bytes)? };
    reset_columns(memory, columns);
    Ok(columns)
}

fn column<'a>(memory: &SharedMemory, offset: u32) -> &'a mut [[f32; 2]] {
    // The spike is the only user of this memory and never runs guests while
    // the host holds one of these slices.
    unsafe { std::slice::from_raw_parts_mut(memory_base(memory).add(offset as usize).cast(), ROWS) }
}

fn reset_columns(memory: &SharedMemory, columns: Columns) {
    fill_native(column(memory, columns.pos), column(memory, columns.vel));
}

fn digest(rows: &[[f32; 2]]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for value in rows.iter().flatten() {
        hash = (hash ^ value.to_bits() as u64).wrapping_mul(0x100_0000_01B3);
    }
    hash
}

/// The guest's `integrate`, natively: the reference result.
fn integrate_native(pos: &mut [[f32; 2]], vel: &[[f32; 2]], dt: f32) {
    for (p, v) in pos.iter_mut().zip(vel) {
        p[0] += v[0] * dt;
        p[1] += v[1] * dt;
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        if r > 1000.0 {
            let k = 1000.0 / r;
            p[0] *= k;
            p[1] *= k;
        }
    }
}

/// Run `ticks` ticks over all rows. Workers claim fixed-size row ranges from
/// a shared counter; the split never depends on the worker count.
fn run_ticks(guests: &mut [Guest], columns: Columns, ticks: usize, range_rows: usize) -> Result<f64> {
    let ranges = ROWS.div_ceil(range_rows);
    let next = AtomicUsize::new(usize::MAX);
    let (start, done) = (Barrier::new(guests.len() + 1), Barrier::new(guests.len() + 1));
    let mut elapsed = 0.0;
    std::thread::scope(|scope| -> Result<()> {
        let handles: Vec<_> = guests
            .iter_mut()
            .map(|guest| {
                let (next, start, done) = (&next, &start, &done);
                scope.spawn(move || -> Result<()> {
                    for _ in 0..ticks {
                        start.wait();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            if index >= ranges {
                                break;
                            }
                            let first = index * range_rows;
                            let count = range_rows.min(ROWS - first) as u32;
                            let byte = (first * 8) as u32;
                            guest.integrate.call(&mut guest.store, (columns.pos + byte, columns.vel + byte, count, DT))?;
                        }
                        done.wait();
                    }
                    Ok(())
                })
            })
            .collect();
        let started = Instant::now();
        for _ in 0..ticks {
            next.store(0, Ordering::Relaxed);
            start.wait();
            done.wait();
        }
        elapsed = started.elapsed().as_secs_f64();
        for handle in handles {
            handle.join().map_err(|_| anyhow!("worker panicked"))??;
        }
        Ok(())
    })?;
    Ok(elapsed)
}

fn isolation_check(engine: &Engine, module: &Module) -> Result<()> {
    println!("== 1. stack and TLS isolation");

    // Negative control: two instances with no per-instance setup.
    let (_memory, mut shared) = spawn_guests(engine, module, 2, Metering::None, false)?;
    let (a, b) = shared.split_at_mut(1);
    let same_stack = a[0].stack_address.call(&mut a[0].store, ())? == b[0].stack_address.call(&mut b[0].store, ())?;
    let (base_a, base_b) = (a[0].global_i32("__tls_base")?, b[0].global_i32("__tls_base")?);
    println!("   without setup: same shadow stack = {same_stack}; __tls_base = {base_a:#x} (1st instance), {base_b:#x} (2nd instance)");
    ensure!(same_stack, "negative control did not show a shared stack; the check below proves nothing");
    ensure!(base_b == 0 || base_b == base_a, "2nd instance unexpectedly has its own TLS block");

    const WORKERS: usize = 8;
    const CALLS: u32 = 2000;
    let (_memory, mut guests) = spawn_guests(engine, module, WORKERS, Metering::None, true)?;
    let mut addresses = Vec::new();
    for guest in guests.iter_mut() {
        addresses.push(guest.stack_address.call(&mut guest.store, ())?);
    }
    let mut tls_bases = Vec::new();
    for guest in guests.iter_mut() {
        tls_bases.push(guest.global_i32("__tls_base")?);
    }
    ensure!(tls_bases.iter().all(|base| *base != 0), "an instance has no TLS block: {tls_bases:x?}");
    let mut distinct = tls_bases.clone();
    distinct.sort_unstable();
    distinct.dedup();
    ensure!(distinct.len() == tls_bases.len(), "instances share a TLS block: {tls_bases:x?}");
    let mut sorted = addresses.clone();
    sorted.sort_unstable();
    let min_gap = sorted.windows(2).map(|w| w[1] - w[0]).min().unwrap();
    ensure!(min_gap >= STACK_BYTES, "shadow stacks overlap (closest pair {min_gap} bytes apart)");

    let start = Barrier::new(WORKERS);
    let results: Vec<Result<(u32, u32)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = guests
            .iter_mut()
            .enumerate()
            .map(|(index, guest)| {
                let start = &start;
                scope.spawn(move || -> Result<(u32, u32)> {
                    start.wait();
                    let (mut bad, mut last) = (0, 0);
                    for _ in 0..CALLS {
                        bad += guest.stack_probe.call(&mut guest.store, (index as u32 + 1, 20_000))?;
                        last = guest.tls_bump.call(&mut guest.store, ())?;
                    }
                    Ok((bad, last))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    });
    for (index, result) in results.into_iter().enumerate() {
        let (bad, last) = result?;
        ensure!(bad == 0, "instance {index}: {bad} corrupted stack bytes");
        ensure!(last == CALLS, "instance {index}: TLS counter is {last}, expected {CALLS}");
    }
    println!("   with setup: {WORKERS} instances ran {CALLS} concurrent calls each; 0 corrupted stack bytes, every TLS counter = {CALLS}");
    println!("   stacks are at least {min_gap} bytes apart");
    Ok(())
}

fn correctness_and_scaling(engine: &Engine, module: &Module) -> Result<()> {
    println!("== 2. result equality and scaling ({ROWS} rows, {TICKS} ticks, {RANGE_ROWS}-row ranges)");
    let (mut pos, mut vel) = (vec![[0f32; 2]; ROWS], vec![[0f32; 2]; ROWS]);
    fill_native(&mut pos, &mut vel);
    let started = Instant::now();
    for _ in 0..TICKS {
        integrate_native(&mut pos, &vel, DT);
    }
    let native = started.elapsed().as_secs_f64();
    let reference = digest(&pos);
    println!("   native Rust, 1 thread:  {:>7.2} ms/tick  {:>5.2} ns/row  digest {reference:016x}", native * 1e3 / TICKS as f64, native * 1e9 / (TICKS * ROWS) as f64);

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut single = 0.0;
    for workers in [1usize, 2, 4, 8, 16].into_iter().filter(|w| *w <= cores.max(1)) {
        let (memory, mut guests) = spawn_guests(engine, module, workers, Metering::None, true)?;
        let columns = alloc_columns(&memory, &mut guests[0])?;
        // Best of three: the machine is not idle.
        let mut best = f64::INFINITY;
        let mut result = 0;
        for _ in 0..3 {
            reset_columns(&memory, columns);
            best = best.min(run_ticks(&mut guests, columns, TICKS, RANGE_ROWS)?);
            result = digest(column(&memory, columns.pos));
            ensure!(result == reference, "{workers} workers: digest {result:016x} differs from native {reference:016x}");
        }
        if workers == 1 {
            single = best;
        }
        println!(
            "   wasm, {workers:>2} worker(s):     {:>7.2} ms/tick  {:>5.2} ns/row  digest {result:016x}  speedup {:.2}x",
            best * 1e3 / TICKS as f64,
            best * 1e9 / (TICKS * ROWS) as f64,
            single / best
        );
    }
    println!("   every worker count matches the native digest bit for bit");
    Ok(())
}

fn call_overhead(module_bytes: &[u8]) -> Result<()> {
    println!("== 3. call overhead and metering cost (1 worker)");
    for (label, metering) in [("no metering", Metering::None), ("fuel", Metering::Fuel), ("epoch interruption", Metering::Epoch), ("raw NaNs (unsafe)", Metering::RawNans)] {
        let engine = engine(metering)?;
        let module = Module::new(&engine, module_bytes)?;
        let (memory, mut guests) = spawn_guests(&engine, &module, 1, metering, true)?;
        let columns = alloc_columns(&memory, &mut guests[0])?;
        let guest = &mut guests[0];

        const CALLS: usize = 2_000_000;
        let started = Instant::now();
        for _ in 0..CALLS {
            guest.noop.call(&mut guest.store, ())?;
        }
        let noop = started.elapsed().as_secs_f64() * 1e9 / CALLS as f64;
        print!("   {label:<19} empty call {noop:>5.1} ns | ns/row by range size:");
        for range_rows in [64usize, 256, 1024, 4096] {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                reset_columns(&memory, columns);
                best = best.min(run_ticks(std::slice::from_mut(guest), columns, 4, range_rows)?);
            }
            print!("  {range_rows}: {:.2}", best * 1e9 / (4 * ROWS) as f64);
        }
        println!();
    }
    Ok(())
}

fn main() -> Result<()> {
    let module_bytes = std::fs::read(GUEST_WASM).with_context(|| format!("read {GUEST_WASM}"))?;
    let engine = engine(Metering::None)?;
    let module = Module::new(&engine, &module_bytes)?;
    println!("guest: {} ({} bytes)", GUEST_WASM.rsplit('/').next().unwrap_or(GUEST_WASM), module_bytes.len());

    isolation_check(&engine, &module)?;
    correctness_and_scaling(&engine, &module)?;
    call_overhead(&module_bytes)?;
    println!("spike passed");
    Ok(())
}
