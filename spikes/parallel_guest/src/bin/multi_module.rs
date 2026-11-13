//! M0 spike: two independently linked sim modules sharing one linear memory.
//!
//! Module 1 is the usual guest (its own dlmalloc heap, stack and data at the
//! bottom of memory). Module 2 is linked with `--global-base=64 MiB`, gets
//! its heap from the host (`host_alloc`), and every instance of it gets a
//! host-assigned shadow stack and TLS block before its first call.
//!
//! Layout of the shared memory:
//!
//! | Range | Owner |
//! |---|---|
//! | 0 .. 64 MiB | module 1: data, stack, dlmalloc heap (grows from its initial size) |
//! | 64 .. 80 MiB | module 2: static data (link-time base) |
//! | 80 .. 96 MiB | module 2: heap, stacks and TLS blocks, allocated by the host |
//! | 96 MiB .. | module 1's dlmalloc, once it grows past its initial pages |
//!
//! Checks: every address each module uses lies in its own range; both
//! modules run kernels concurrently on disjoint halves of one column and the
//! result equals native Rust; module 1's heap data and both modules'
//! thread-locals are untouched by the other.
//!
//! `cargo run -p spike_parallel_guest --release --bin multi_module`

use anyhow::{Context, Result, anyhow, ensure};
use spike_parallel_guest::{GUEST_WASM, Metering, engine, memory_base, spawn_guests_with};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use wasmtime::{Engine, Instance, Linker, Module, SharedMemory, Store, TypedFunc, Val};

const GUEST2_WASM: &str = env!("GUEST2_WASM");
const MIB: u32 = 1 << 20;
const MODULE2_DATA: (u32, u32) = (64 * MIB, 80 * MIB);
const MODULE2_HEAP: (u32, u32) = (80 * MIB, 96 * MIB);
const PAGE: u64 = 65536;
const ROWS: usize = 1 << 19;
const HALF: usize = ROWS / 2;
const RANGE_ROWS: usize = 1024;
const TICKS: usize = 30;
const DT: f32 = 1.0 / 30.0;
const STACK_BYTES: u32 = 256 << 10;

// ---------------------------------------------------------------------------
// Host allocator for module 2: power-of-two blocks in MODULE2_HEAP, sizes
// kept on the host side so `free` and `realloc` need no header.
// ---------------------------------------------------------------------------

struct HostHeap {
    top: u32,
    free: BTreeMap<u32, Vec<u32>>,
    live: BTreeMap<u32, (u32, u32)>, // address -> (class bytes, requested bytes)
    allocations: u64,
}

static HEAP: Mutex<HostHeap> = Mutex::new(HostHeap { top: MODULE2_HEAP.0, free: BTreeMap::new(), live: BTreeMap::new(), allocations: 0 });

fn heap_alloc(size: u32, align: u32) -> u32 {
    let class = size.max(align).max(16).next_power_of_two();
    let mut heap = HEAP.lock().unwrap();
    heap.allocations += 1;
    let address = match heap.free.get_mut(&class).and_then(Vec::pop) {
        Some(address) => address,
        None => {
            let address = heap.top.next_multiple_of(class.min(4096));
            if address + class > MODULE2_HEAP.1 {
                return 0;
            }
            heap.top = address + class;
            address
        }
    };
    heap.live.insert(address, (class, size));
    address
}

fn heap_free(address: u32) {
    let mut heap = HEAP.lock().unwrap();
    if let Some((class, _)) = heap.live.remove(&address) {
        heap.free.entry(class).or_default().push(address);
    }
}

fn heap_block_size(address: u32) -> u32 {
    HEAP.lock().unwrap().live.get(&address).map(|&(_, size)| size).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Module 2 instances
// ---------------------------------------------------------------------------

struct Guest2 {
    store: Store<()>,
    damp: TypedFunc<(u32, u32, u32), ()>,
    tls_bump: TypedFunc<(), u32>,
    static_address: TypedFunc<(), u32>,
    heap_probe: TypedFunc<(), u32>,
}

fn global_i32(store: &mut Store<()>, instance: &Instance, name: &str) -> Result<i32> {
    let global = instance.get_global(&mut *store, name).with_context(|| format!("module 2 does not export {name}"))?;
    global.get(&mut *store).i32().with_context(|| format!("{name} is not i32"))
}

fn instantiate_module2(engine: &Engine, module: &Module, memory: &SharedMemory, first: bool) -> Result<Guest2> {
    let mut store = Store::new(engine, ());
    let mut linker = Linker::new(engine);
    linker.define(&store, "env", "memory", memory.clone())?;
    linker.func_wrap("env", "host_alloc", heap_alloc)?;
    linker.func_wrap("env", "host_free", heap_free)?;
    linker.func_wrap("env", "host_block_size", heap_block_size)?;
    linker.define_unknown_imports_as_traps(module)?;
    // The start function copies data segments into module 2's region once.
    let instance = linker.instantiate(&mut store, module)?;

    // The linker put module 2's default stack at the bottom of memory, on
    // top of module 1's. It must be replaced before the first call.
    let default_stack = global_i32(&mut store, &instance, "__stack_pointer")? as u32;
    ensure!(default_stack < MODULE2_DATA.0, "expected the linker's default stack below the module's base");
    let stack = heap_alloc(STACK_BYTES, 16);
    ensure!(stack != 0, "module 2 heap exhausted");
    instance.get_global(&mut store, "__stack_pointer").unwrap().set(&mut store, Val::I32((stack + STACK_BYTES) as i32))?;

    let size = global_i32(&mut store, &instance, "__tls_size")? as u32;
    let align = (global_i32(&mut store, &instance, "__tls_align")? as u32).max(1);
    let block = heap_alloc(size.max(1), align);
    let init_tls: TypedFunc<u32, ()> = instance.get_typed_func(&mut store, "__wasm_init_tls")?;
    init_tls.call(&mut store, block)?;

    if first {
        let ctors: TypedFunc<(), ()> = instance.get_typed_func(&mut store, "__wasm_call_ctors")?;
        ctors.call(&mut store, ())?;
        let heap_base = global_i32(&mut store, &instance, "__heap_base")? as u32;
        ensure!(heap_base <= MODULE2_DATA.1, "module 2 data ({heap_base:#x}) overflows its region");
        println!("module 2: data {:#x}..{heap_base:#x}, default stack pointer {default_stack:#x} (replaced), TLS {size} bytes", MODULE2_DATA.0);
    }
    Ok(Guest2 {
        damp: instance.get_typed_func(&mut store, "damp")?,
        tls_bump: instance.get_typed_func(&mut store, "tls_bump")?,
        static_address: instance.get_typed_func(&mut store, "static_address")?,
        heap_probe: instance.get_typed_func(&mut store, "heap_probe")?,
        store,
    })
}

// ---------------------------------------------------------------------------
// Reference kernels
// ---------------------------------------------------------------------------

fn weight(row: usize) -> f32 {
    (200 + ((row & 255) as u32 * 37) % 56) as f32 * (1.0 / 256.0)
}

fn integrate_native(pos: &mut [[f32; 2]], vel: &[[f32; 2]]) {
    for (p, v) in pos.iter_mut().zip(vel) {
        p[0] += v[0] * DT;
        p[1] += v[1] * DT;
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        if r > 1000.0 {
            let k = 1000.0 / r;
            p[0] *= k;
            p[1] *= k;
        }
    }
}

fn damp_native(pos: &mut [[f32; 2]], first_row: usize) {
    for (i, p) in pos.iter_mut().enumerate() {
        let w = weight(first_row + i);
        p[0] *= w;
        p[1] *= w;
    }
}

fn seed(row: usize, salt: u32) -> f32 {
    let mut h = (row as u32).wrapping_mul(0x9E37_79B9) ^ salt;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    (h % 20_001) as f32 / 10.0 - 1000.0
}

fn fill(pos: &mut [[f32; 2]], vel: &mut [[f32; 2]]) {
    for row in 0..pos.len() {
        pos[row] = [seed(row, 1), seed(row, 2)];
        vel[row] = [seed(row, 3) * 0.05, seed(row, 4) * 0.05];
    }
}

fn digest(rows: &[[f32; 2]]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for value in rows.iter().flatten() {
        hash = (hash ^ value.to_bits() as u64).wrapping_mul(0x100_0000_01B3);
    }
    hash
}

fn in_range(address: u32, (lo, hi): (u32, u32)) -> bool {
    (lo..hi).contains(&address)
}

fn main() -> Result<()> {
    let engine = engine(Metering::None)?;
    let module1 = Module::new(&engine, std::fs::read(GUEST_WASM)?)?;
    let module2 = Module::new(&engine, std::fs::read(GUEST2_WASM)?)?;

    // Reserve module 2's ranges right after module 1's constructors, before
    // module 1 allocates anything big.
    let (memory, mut guests1) = spawn_guests_with(&engine, &module1, 2, Metering::None, true, |memory, _| {
        let target_pages = MODULE2_HEAP.1 as u64 / PAGE;
        let current = memory.data_size() as u64 / PAGE;
        ensure!(current < MODULE2_DATA.0 as u64 / PAGE, "module 1 already reaches module 2's base");
        memory.grow(target_pages - current)?;
        Ok(())
    })?;
    let mut guests2 = vec![instantiate_module2(&engine, &module2, &memory, true)?, instantiate_module2(&engine, &module2, &memory, false)?];

    // Address checks.
    for g in guests2.iter_mut() {
        let statics = g.static_address.call(&mut g.store, ())?;
        let heap = g.heap_probe.call(&mut g.store, ())?;
        ensure!(in_range(statics, MODULE2_DATA), "module 2 static data at {statics:#x}");
        ensure!(in_range(heap, MODULE2_HEAP), "module 2 heap allocation at {heap:#x}");
    }
    let base = memory_base(&memory);
    let column_bytes = (ROWS * 8) as u32;
    let (pos, vel) = (guests1[0].alloc(column_bytes)?, guests1[0].alloc(column_bytes)?);
    let sentinel = guests1[0].alloc(64 << 10)?;
    for g in guests1.iter_mut() {
        let stack = g.stack_address.call(&mut g.store, ())?;
        ensure!(!in_range(stack, (MODULE2_DATA.0, MODULE2_HEAP.1)), "module 1 stack at {stack:#x} is in module 2's ranges");
    }
    for (name, address, len) in [("position column", pos, column_bytes), ("velocity column", vel, column_bytes), ("sentinel", sentinel, 64 << 10)] {
        let overlaps = address < MODULE2_HEAP.1 && address + len > MODULE2_DATA.0;
        ensure!(!overlaps, "module 1 {name} at {address:#x} overlaps module 2's ranges");
    }
    println!("module 1: columns at {pos:#x} and {vel:#x}, sentinel at {sentinel:#x}; module 2 addresses all inside 64..96 MiB");

    let column = |offset: u32| unsafe { std::slice::from_raw_parts_mut(base.add(offset as usize).cast::<[f32; 2]>(), ROWS) };
    fill(column(pos), column(vel));
    unsafe { std::ptr::write_bytes(base.add(sentinel as usize), 0xA5, 64 << 10) };

    // Native reference: module 1's kernel on the first half, module 2's on the second.
    let (mut ref_pos, mut ref_vel) = (vec![[0f32; 2]; ROWS], vec![[0f32; 2]; ROWS]);
    fill(&mut ref_pos, &mut ref_vel);
    for _ in 0..TICKS {
        integrate_native(&mut ref_pos[..HALF], &ref_vel[..HALF]);
        for first in (HALF..ROWS).step_by(RANGE_ROWS) {
            damp_native(&mut ref_pos[first..first + RANGE_ROWS], first);
        }
    }
    let reference = digest(&ref_pos);

    // Both modules at once: two module 1 workers on the first half, two
    // module 2 workers on the second, claiming ranges from separate counters.
    let ranges = HALF / RANGE_ROWS;
    let (next1, next2) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let (start, done) = (Barrier::new(5), Barrier::new(5));
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::new();
        for g in guests1.iter_mut() {
            let (next, start, done) = (&next1, &start, &done);
            handles.push(scope.spawn(move || -> Result<()> {
                for _ in 0..TICKS {
                    start.wait();
                    let mut result = Ok(());
                    while let Some(index) = Some(next.fetch_add(1, Ordering::Relaxed)).filter(|i| *i < ranges) {
                        let byte = (index * RANGE_ROWS * 8) as u32;
                        if let Err(e) = g.integrate.call(&mut g.store, (pos + byte, vel + byte, RANGE_ROWS as u32, DT)) {
                            result = Err(anyhow!("module 1 trapped: {e}"));
                        }
                    }
                    done.wait();
                    result?;
                }
                Ok(())
            }));
        }
        for g in guests2.iter_mut() {
            let (next, start, done) = (&next2, &start, &done);
            handles.push(scope.spawn(move || -> Result<()> {
                for _ in 0..TICKS {
                    start.wait();
                    let mut result = Ok(());
                    while let Some(index) = Some(next.fetch_add(1, Ordering::Relaxed)).filter(|i| *i < ranges) {
                        let first = HALF + index * RANGE_ROWS;
                        if let Err(e) = g.damp.call(&mut g.store, (pos + (first * 8) as u32, RANGE_ROWS as u32, first as u32)) {
                            result = Err(anyhow!("module 2 trapped: {e}"));
                        }
                    }
                    done.wait();
                    result?;
                }
                Ok(())
            }));
        }
        for _ in 0..TICKS {
            next1.store(0, Ordering::Relaxed);
            next2.store(0, Ordering::Relaxed);
            start.wait();
            done.wait();
        }
        for h in handles {
            h.join().map_err(|_| anyhow!("worker panicked"))??;
        }
        Ok(())
    })?;
    let result = digest(column(pos));
    ensure!(result == reference, "digest {result:016x} differs from native {reference:016x}");
    println!("{TICKS} ticks, both modules concurrently on one column: digest {result:016x} equals native");

    let intact = unsafe { std::slice::from_raw_parts(base.add(sentinel as usize), 64 << 10) }.iter().all(|b| *b == 0xA5);
    ensure!(intact, "module 1's sentinel buffer was overwritten");
    for (index, g) in guests1.iter_mut().enumerate() {
        let before = g.tls_bump.call(&mut g.store, ())?;
        for _ in 0..99 {
            g.tls_bump.call(&mut g.store, ())?;
        }
        ensure!(g.tls_bump.call(&mut g.store, ())? == before + 100, "module 1 instance {index}: TLS disturbed");
    }
    for (index, g) in guests2.iter_mut().enumerate() {
        for call in 1..=100 {
            ensure!(g.tls_bump.call(&mut g.store, ())? == call, "module 2 instance {index}: TLS shared or disturbed");
        }
    }
    let heap = HEAP.lock().unwrap();
    println!(
        "module 1 sentinel intact; every instance's TLS counter independent; module 2 host heap: {} allocations, {} live, high-water {} KiB",
        heap.allocations,
        heap.live.len(),
        (heap.top - MODULE2_HEAP.0) >> 10
    );
    println!("spike passed");
    Ok(())
}
