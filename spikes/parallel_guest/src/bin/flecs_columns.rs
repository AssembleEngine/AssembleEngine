//! M1 core spike: real flecs tables whose columns live in a host-owned arena
//! inside the guest's shared linear memory, processed in place by N guest
//! instances in parallel, with structural changes between ticks.
//!
//! Checks:
//! * every column flecs allocates lies inside the arena, which the host
//!   claimed by growing the shared memory itself; the guest allocator grows
//!   around it (no guest allocation or shadow stack inside the arena);
//! * after T ticks of system work plus spawn/despawn/add/remove churn, the
//!   world digest is identical for native Rust and for 1..16 guest workers;
//! * cost of the system dispatch and of the serial structural phase.
//!
//! `cargo run -p spike_parallel_guest --release --bin flecs_columns`

use ae_ecs::flecs_backend as fb;
use ae_ecs::{Access, ComponentId, EntityId, Term};
use anyhow::{Context, Result, anyhow, ensure};
use spike_parallel_guest::{GUEST_WASM, Guest, Metering, engine, memory_base, spawn_guests_with};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::time::Instant;
use wasmtime::{Module, SharedMemory};

const ENTITIES: usize = 1 << 20;
const TICKS: usize = 30;
const RANGE_ROWS: usize = 1024;
const CHURN: usize = 10_000;
const DT: f32 = 1.0 / 30.0;
const ARENA_PAGES: u64 = 8192; // 512 MiB
const PAGE: usize = 65536;

// ---------------------------------------------------------------------------
// Column arena: a deterministic power-of-two allocator over a fixed range.
// The flecs column allocator hook is process-wide, so its state is global.
// ---------------------------------------------------------------------------

const HEADER: usize = 16;
const MIN_CLASS: u32 = 6; // 64-byte blocks
const CLASSES: usize = 40;

struct Arena {
    base: usize,
    len: usize,
    top: usize,
    free: [Vec<usize>; CLASSES],
    live_bytes: usize,
    peak_bytes: usize,
    calls: usize,
}

static ARENA: Mutex<Option<Arena>> = Mutex::new(None);

fn class_of(size: usize) -> u32 {
    (size + HEADER).next_power_of_two().trailing_zeros().max(MIN_CLASS)
}

fn arena_alloc(arena: &mut Arena, size: usize) -> *mut c_void {
    arena.calls += 1;
    let class = class_of(size);
    let block = match arena.free[class as usize].pop() {
        Some(block) => block,
        None => {
            let block = arena.top;
            arena.top += 1 << class;
            assert!(arena.top <= arena.len, "column arena exhausted");
            block
        }
    };
    arena.live_bytes += 1 << class;
    arena.peak_bytes = arena.peak_bytes.max(arena.live_bytes);
    let address = arena.base + block;
    unsafe {
        (address as *mut u32).write(class);
        (address + HEADER) as *mut c_void
    }
}

fn arena_release(arena: &mut Arena, ptr: *mut c_void) -> u32 {
    let address = ptr as usize - HEADER;
    let class = unsafe { (address as *const u32).read() };
    arena.free[class as usize].push(address - arena.base);
    arena.live_bytes -= 1 << class;
    class
}

unsafe extern "C-unwind" fn column_malloc(size: i32) -> *mut c_void {
    arena_alloc(ARENA.lock().unwrap().as_mut().expect("arena"), size as usize)
}

unsafe extern "C-unwind" fn column_realloc(ptr: *mut c_void, size: i32) -> *mut c_void {
    let mut guard = ARENA.lock().unwrap();
    let arena = guard.as_mut().expect("arena");
    if ptr.is_null() {
        return arena_alloc(arena, size as usize);
    }
    let old_class = unsafe { ((ptr as usize - HEADER) as *const u32).read() };
    if class_of(size as usize) == old_class {
        return ptr;
    }
    let new = arena_alloc(arena, size as usize);
    let old_capacity = (1usize << old_class) - HEADER;
    unsafe { std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), new.cast::<u8>(), old_capacity.min(size as usize)) };
    arena_release(arena, ptr);
    new
}

unsafe extern "C-unwind" fn column_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        arena_release(ARENA.lock().unwrap().as_mut().expect("arena"), ptr);
    }
}

fn arena_range() -> (usize, usize) {
    let guard = ARENA.lock().unwrap();
    let arena = guard.as_ref().expect("arena");
    (arena.base, arena.base + arena.len)
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n as u64) as usize
    }
}

fn bytes<T: Copy>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast(), size_of::<T>()) }
}

fn seed(row: u32, salt: u32) -> f32 {
    let mut h = row.wrapping_mul(0x9E37_79B9) ^ salt;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    (h % 20_001) as f32 / 10.0 - 1000.0
}

/// One unit of system work: a row range of one table, as offsets into guest memory.
#[derive(Clone, Copy)]
struct Work {
    pos: u32,
    vel: u32,
    count: u32,
}

struct World {
    world: fb::World,
    position: ComponentId,
    velocity: ComponentId,
    health: ComponentId,
    stunned: ComponentId,
    query: fb::Query,
    terms: [Term; 2],
    alive: Vec<EntityId>,
    rng: Rng,
    serial: u32,
    /// Spawn targets: entities are created directly in their final table.
    plain: fb::Table,
    with_health: fb::Table,
    /// Per structural operation kind: (effective operations, seconds).
    ops: [(u64, f64); 4],
}

const OP_NAMES: [&str; 4] = ["despawn", "spawn", "add Stunned", "remove Stunned"];

impl World {
    fn new() -> Result<Self> {
        unsafe {
            let world = fb::init()?;
            let position = fb::register_component(world, "Position", 8, 4)?;
            let velocity = fb::register_component(world, "Velocity", 8, 4)?;
            let health = fb::register_component(world, "Health", 4, 4)?;
            let stunned = fb::register_component(world, "Stunned", 4, 4)?;
            let terms = [
                Term {
                    component: position,
                    access: Access::Write,
                },
                Term {
                    component: velocity,
                    access: Access::Read,
                },
            ];
            let query = fb::create_query(world, &terms)?;
            let plain = fb::find_table(world, &[position, velocity])?;
            let with_health = fb::find_table(world, &[position, velocity, health])?;
            let mut this = Self {
                world,
                position,
                velocity,
                health,
                stunned,
                query,
                terms,
                alive: Vec::with_capacity(ENTITIES),
                rng: Rng(0xF1EC5),
                serial: 0,
                plain,
                with_health,
                ops: [(0, 0.0); 4],
            };
            for _ in 0..ENTITIES {
                this.spawn()?;
            }
            Ok(this)
        }
    }

    fn spawn(&mut self) -> Result<()> {
        let n = self.serial;
        self.serial += 1;
        unsafe {
            let e = fb::spawn_in_table(self.world, if n % 3 == 0 { self.with_health } else { self.plain });
            // The entity already has these components: the sets write in place.
            fb::set_component(self.world, e, self.position, bytes(&[seed(n, 1), seed(n, 2)]))?;
            fb::set_component(
                self.world,
                e,
                self.velocity,
                bytes(&[seed(n, 3) * 0.05, seed(n, 4) * 0.05]),
            )?;
            if n % 3 == 0 {
                fb::set_component(self.world, e, self.health, bytes(&n))?;
            }
            self.alive.push(e);
        }
        Ok(())
    }

    /// The serial structural phase between ticks. Toggles that would not
    /// change the entity's table are skipped, as the engine's command merge
    /// will skip them, so every counted operation is one real table change.
    fn churn(&mut self) -> Result<()> {
        unsafe {
            let started = Instant::now();
            for _ in 0..CHURN / 5 {
                let e = self.alive.swap_remove(self.rng.below(self.alive.len()));
                ensure!(fb::delete_entity(self.world, e), "despawn of a dead entity");
            }
            self.ops[0].0 += (CHURN / 5) as u64;
            self.ops[0].1 += started.elapsed().as_secs_f64();

            let started = Instant::now();
            for _ in 0..CHURN / 5 {
                self.spawn()?;
            }
            self.ops[1].0 += (CHURN / 5) as u64;
            self.ops[1].1 += started.elapsed().as_secs_f64();

            let started = Instant::now();
            for _ in 0..CHURN * 3 / 10 {
                let e = self.alive[self.rng.below(self.alive.len())];
                if !fb::has_component(self.world, e, self.stunned) {
                    fb::set_component(self.world, e, self.stunned, bytes(&1u32))?;
                    self.ops[2].0 += 1;
                }
            }
            self.ops[2].1 += started.elapsed().as_secs_f64();

            let started = Instant::now();
            for _ in 0..CHURN * 3 / 10 {
                let e = self.alive[self.rng.below(self.alive.len())];
                if fb::has_component(self.world, e, self.stunned) {
                    fb::remove_component(self.world, e, self.stunned);
                    self.ops[3].0 += 1;
                }
            }
            self.ops[3].1 += started.elapsed().as_secs_f64();
        }
        Ok(())
    }

    /// Split every matched table into fixed-size row ranges, in flecs order.
    fn work(&self, memory_base: usize) -> Result<Vec<Work>> {
        let (lo, hi) = arena_range();
        let mut out = Vec::new();
        unsafe {
            fb::for_each_chunk(self.world, self.query, &self.terms, |chunk| {
                let (pos, vel) = (chunk.fields[0] as usize, chunk.fields[1] as usize);
                ensure!(
                    (lo..hi).contains(&pos) && (lo..hi).contains(&vel),
                    "a column is outside the arena"
                );
                let mut first = 0;
                while first < chunk.count {
                    let count = RANGE_ROWS.min(chunk.count - first);
                    let byte = first * 8;
                    out.push(Work {
                        pos: (pos + byte - memory_base) as u32,
                        vel: (vel + byte - memory_base) as u32,
                        count: count as u32,
                    });
                    first += count;
                }
                Ok(())
            })?;
        }
        Ok(out)
    }

    /// Order-independent digest over (entity id, position bits); also the row and table count.
    fn digest(&self) -> Result<(u64, usize, usize)> {
        let (mut sum, mut rows, mut tables) = (0u64, 0usize, 0usize);
        unsafe {
            fb::for_each_chunk(self.world, self.query, &self.terms, |chunk| {
                let ids = std::slice::from_raw_parts(chunk.entities, chunk.count);
                let pos = std::slice::from_raw_parts(chunk.fields[0].cast::<[f32; 2]>(), chunk.count);
                for (id, p) in ids.iter().zip(pos) {
                    let mut h = id ^ ((p[0].to_bits() as u64) << 32 | p[1].to_bits() as u64);
                    h = (h ^ (h >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                    sum = sum.wrapping_add(h ^ (h >> 29));
                }
                rows += chunk.count;
                tables += 1;
                Ok(())
            })?;
        }
        Ok((sum, rows, tables))
    }
}

impl Drop for World {
    fn drop(&mut self) {
        unsafe {
            fb::fini_query(self.query);
            fb::fini(self.world);
        }
    }
}

/// The guest's `integrate` kernel, natively.
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

struct Timings {
    system: f64,
    churn: f64,
}

fn run_native(memory: &SharedMemory) -> Result<(u64, Timings)> {
    let base = memory_base(memory) as usize;
    let mut world = World::new()?;
    let mut t = Timings {
        system: 0.0,
        churn: 0.0,
    };
    for _ in 0..TICKS {
        let started = Instant::now();
        for w in world.work(base)? {
            let pos =
                unsafe { std::slice::from_raw_parts_mut((base + w.pos as usize) as *mut [f32; 2], w.count as usize) };
            let vel =
                unsafe { std::slice::from_raw_parts((base + w.vel as usize) as *const [f32; 2], w.count as usize) };
            integrate_native(pos, vel, DT);
        }
        t.system += started.elapsed().as_secs_f64();
        let started = Instant::now();
        world.churn()?;
        t.churn += started.elapsed().as_secs_f64();
    }
    let total: u64 = world.ops.iter().map(|(n, _)| n).sum();
    println!(
        "structural phase: {} table changes per tick on average",
        total / TICKS as u64
    );
    for (name, (count, secs)) in OP_NAMES.iter().zip(world.ops) {
        println!(
            "  {name:<15} {:>6} per tick  {:>6.0} ns each",
            count / TICKS as u64,
            secs * 1e9 / count.max(1) as f64
        );
    }
    Ok((world.digest()?.0, t))
}

fn run_guests(memory: &SharedMemory, guests: &mut [Guest]) -> Result<(u64, Timings)> {
    let base = memory_base(memory) as usize;
    let mut world = World::new()?;
    let work: RwLock<Arc<Vec<Work>>> = RwLock::new(Arc::new(Vec::new()));
    let next = AtomicUsize::new(0);
    let (start, done) = (Barrier::new(guests.len() + 1), Barrier::new(guests.len() + 1));
    let mut t = Timings {
        system: 0.0,
        churn: 0.0,
    };
    std::thread::scope(|scope| -> Result<()> {
        let handles: Vec<_> = guests
            .iter_mut()
            .map(|guest| {
                let (work, next, start, done) = (&work, &next, &start, &done);
                scope.spawn(move || -> Result<()> {
                    for _ in 0..TICKS {
                        start.wait();
                        let items = work.read().unwrap().clone();
                        let mut failure = None;
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(w) = items.get(index) else { break };
                            if let Err(e) = guest.integrate.call(&mut guest.store, (w.pos, w.vel, w.count, DT)) {
                                failure = Some(e);
                            }
                        }
                        done.wait();
                        if let Some(e) = failure {
                            return Err(anyhow!("guest trapped: {e}"));
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        for _ in 0..TICKS {
            let started = Instant::now();
            *work.write().unwrap() = Arc::new(world.work(base)?);
            next.store(0, Ordering::Relaxed);
            start.wait();
            done.wait();
            t.system += started.elapsed().as_secs_f64();
            let started = Instant::now();
            world.churn()?;
            t.churn += started.elapsed().as_secs_f64();
        }
        for handle in handles {
            handle.join().map_err(|_| anyhow!("worker panicked"))??;
        }
        Ok(())
    })?;
    Ok((world.digest()?.0, t))
}

fn main() -> Result<()> {
    let module_bytes = std::fs::read(GUEST_WASM).with_context(|| format!("read {GUEST_WASM}"))?;
    let engine = engine(Metering::None)?;
    let module = Module::new(&engine, &module_bytes)?;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let max_workers = 16.min(cores);

    // The arena is claimed once the first instance is up, before the other
    // instances allocate their shadow stacks, so the guest allocator has to
    // grow past it.
    let (memory, mut guests) =
        spawn_guests_with(&engine, &module, max_workers, Metering::None, true, |memory, _first| {
            let first_page = memory.grow(ARENA_PAGES)?;
            let base = memory_base(memory) as usize + first_page as usize * PAGE;
            *ARENA.lock().unwrap() = Some(Arena {
                base,
                len: ARENA_PAGES as usize * PAGE,
                top: 0,
                free: std::array::from_fn(|_| Vec::new()),
                live_bytes: 0,
                peak_bytes: 0,
                calls: 0,
            });
            Ok(())
        })?;
    let (lo, hi) = arena_range();
    let base = memory_base(&memory) as usize;
    println!(
        "shared memory {} MiB; column arena at guest offsets [{:#x}, {:#x})",
        memory.data_size() >> 20,
        lo - base,
        hi - base
    );

    // Guest allocations made after the arena was claimed must lie outside it.
    for (index, guest) in guests.iter_mut().enumerate() {
        let stack = guest.stack_address.call(&mut guest.store, ())? as usize + base;
        ensure!(
            !(lo..hi).contains(&stack),
            "instance {index} has its shadow stack inside the arena"
        );
    }
    let probe = guests[0].alloc(1 << 20)? as usize + base;
    ensure!(
        !(lo..hi).contains(&probe),
        "guest malloc returned memory inside the arena"
    );
    println!("guest stacks and a 1 MiB guest malloc after the claim all lie outside the arena");

    unsafe { fb::install_column_allocator(column_malloc, column_realloc, column_free) };

    println!(
        "{ENTITIES} entities, 3 archetypes plus Stunned toggling; {TICKS} ticks of integrate over {RANGE_ROWS}-row ranges, {CHURN} structural commands per tick"
    );
    let (reference, t) = run_native(&memory)?;
    {
        let world = World::new()?;
        let (_, rows, tables) = world.digest()?;
        println!(
            "freshly spawned world: {tables} tables matched by the query, {rows} rows (churn adds the Stunned variants)"
        );
    }
    {
        let arena = ARENA.lock().unwrap();
        let arena = arena.as_ref().unwrap();
        println!(
            "arena: {} allocator calls so far, peak live {} MiB, high-water {} MiB",
            arena.calls,
            arena.peak_bytes >> 20,
            arena.top >> 20
        );
    }
    let per = |s: f64| s * 1e3 / TICKS as f64;
    println!(
        "native Rust:      system {:>6.2} ms/tick  churn {:>5.2} ms/tick  digest {reference:016x}",
        per(t.system),
        per(t.churn)
    );

    let mut single = 0.0;
    for workers in [1usize, 2, 4, 8, 16].into_iter().filter(|w| *w <= max_workers) {
        let (digest, t) = run_guests(&memory, &mut guests[..workers])?;
        ensure!(
            digest == reference,
            "{workers} workers: digest {digest:016x} differs from native {reference:016x}"
        );
        if workers == 1 {
            single = t.system;
        }
        println!(
            "wasm {workers:>2} worker(s): system {:>6.2} ms/tick  churn {:>5.2} ms/tick  digest {digest:016x}  speedup {:.2}x",
            per(t.system),
            per(t.churn),
            single / t.system
        );
    }
    println!("every worker count matches the native digest bit for bit; spike passed");
    Ok(())
}
