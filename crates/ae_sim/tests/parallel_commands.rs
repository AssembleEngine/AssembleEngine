//! M1 exit-criterion test: structural commands emitted by guest systems
//! running on several workers must produce the same world (entity ids,
//! component values, table membership) at every worker count.
//!
//! Earlier spikes only ran row-local kernels, where equal results at any
//! worker count are nearly automatic. Here every row may despawn itself,
//! spawn a child, or gain or lose a component. The commands are recorded
//! under `{system, range, seq}` keys on whichever worker ran the range, then
//! merged by key and applied serially, so the order of application, and with
//! it every new entity id, is independent of scheduling.
//!
//! The column arena and flecs' column allocator are process-wide, so this
//! file holds one test that owns the process's sim memory.

include!("common/lifecycle.rs");

use ae_ecs::flecs_backend as fb;
use ae_ecs::{Access, ComponentId, EntityId, Term};
use ae_sim::command::{Applier, ApplyStats, Command, CommandBuffer, CommandKey, merge};
use ae_sim::imports::{self, WorkerCtx};
use ae_sim::work::{WorkRange, ranges};
use ae_wasm::{DEFAULT_STACK_BYTES, GuestMemory, InstanceSet, InstanceSetBuilder, Metering, SimModule, claim_pages};
use anyhow::{Context, Result, anyhow, ensure};
use std::path::PathBuf;
use std::process::Command as Process;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use wasmtime::TypedFunc;

const ENTITIES: usize = 100_000;
const TICKS: u32 = 30;
const RANGE_ROWS: usize = 1024;
const ARENA_PAGES: u64 = 4096; // 256 MiB
const MAX_WORKERS: usize = 8;

/// World digest after the run, identical in dev and release builds. It pins
/// the whole pipeline (kernel, merge order, apply rules, flecs id
/// allocation) across builds and platforms. A deliberate change to any of
/// those means regenerating it: run with `--nocapture` and copy the native
/// digest.
const EXPECTED_DIGEST: u64 = 0x1c4e_ef97_2d45_6341;

type LifecycleFn = TypedFunc<(u32, u32, u32, u32, u32, u64, u64, u64), ()>;

fn build_guest() -> Result<Vec<u8>> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_dir = crate_dir.join("../../target/test-guests");
    let status = Process::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args([
            "build",
            "--release",
            "--quiet",
            "--target",
            "wasm32-wasip1-threads",
            "--manifest-path",
        ])
        .arg(crate_dir.join("tests/guest/Cargo.toml"))
        .arg("--target-dir")
        .arg(&target_dir)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .status()
        .context("run cargo for the test guest")?;
    ensure!(status.success(), "test guest build failed");
    let wasm = target_dir.join("wasm32-wasip1-threads/release/ae_sim_test_guest.wasm");
    std::fs::read(&wasm).with_context(|| format!("read {}", wasm.display()))
}

fn bytes<T: Copy>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast(), size_of::<T>()) }
}

struct World {
    world: fb::World,
    position: ComponentId,
    velocity: ComponentId,
    stunned: ComponentId,
    query: fb::Query,
    terms: [Term; 2],
    stunned_query: fb::Query,
    stunned_terms: [Term; 1],
    applier: Applier,
    stats: ApplyStats,
}

impl World {
    fn new() -> Result<Self> {
        unsafe {
            let world = fb::init()?;
            let position = fb::register_component(world, "Position", 8, 4)?;
            let velocity = fb::register_component(world, "Velocity", 8, 4)?;
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
            let stunned_terms = [Term {
                component: stunned,
                access: Access::Read,
            }];
            let stunned_query = fb::create_query(world, &stunned_terms)?;
            let table = fb::find_table(world, &[position, velocity])?;
            for i in 0..ENTITIES as u32 {
                let e = fb::spawn_in_table(world, table);
                let angle = (i % 360) as f32;
                fb::set_component(world, e, position, bytes(&[(i % 1000) as f32, (i / 1000) as f32]))?;
                fb::set_component(world, e, velocity, bytes(&[angle * 0.01 - 1.8, 1.8 - angle * 0.01]))?;
            }
            Ok(Self {
                world,
                position,
                velocity,
                stunned,
                query,
                terms,
                stunned_query,
                stunned_terms,
                applier: Applier::default(),
                stats: ApplyStats::default(),
            })
        }
    }

    fn ranges(&self, memory: GuestMemory) -> Result<Vec<WorkRange>> {
        unsafe { ranges(self.world, self.query, &self.terms, RANGE_ROWS, memory) }
    }

    fn apply(&mut self, commands: &[(CommandKey, Command)]) -> Result<()> {
        let s = unsafe { self.applier.apply(self.world, commands)? };
        let t = &mut self.stats;
        t.spawned += s.spawned;
        t.despawned += s.despawned;
        t.added += s.added;
        t.overwritten += s.overwritten;
        t.removed += s.removed;
        t.ignored += s.ignored;
        Ok(())
    }

    /// Order-independent digest of every live entity's id and position, and
    /// of every Stunned entity's id and value.
    fn digest(&self) -> Result<(u64, usize, usize)> {
        let fold = |sum: &mut u64, a: u64, b: u64| *sum = sum.wrapping_add(mix(a ^ mix(b)));
        let (mut sum, mut rows, mut stunned) = (0u64, 0usize, 0usize);
        unsafe {
            fb::for_each_chunk(self.world, self.query, &self.terms, |chunk| {
                let ids = std::slice::from_raw_parts(chunk.entities, chunk.count);
                let pos = std::slice::from_raw_parts(chunk.fields[0].cast::<[f32; 2]>(), chunk.count);
                for (id, p) in ids.iter().zip(pos) {
                    fold(&mut sum, *id, (p[0].to_bits() as u64) << 32 | p[1].to_bits() as u64);
                }
                rows += chunk.count;
                Ok(())
            })?;
            fb::for_each_chunk(self.world, self.stunned_query, &self.stunned_terms, |chunk| {
                let ids = std::slice::from_raw_parts(chunk.entities, chunk.count);
                let values = std::slice::from_raw_parts(chunk.fields[0].cast::<u32>(), chunk.count);
                for (id, v) in ids.iter().zip(values) {
                    fold(&mut sum, !*id, *v as u64);
                }
                stunned += chunk.count;
                Ok(())
            })?;
        }
        Ok((sum, rows, stunned))
    }
}

impl Drop for World {
    fn drop(&mut self) {
        unsafe {
            fb::fini_query(self.query);
            fb::fini_query(self.stunned_query);
            fb::fini(self.world);
        }
    }
}

/// The reference: the same row logic in native Rust, one buffer, ranges in order.
struct NativeEmit<'a> {
    buffer: &'a mut CommandBuffer,
    position: ComponentId,
    velocity: ComponentId,
    stunned: ComponentId,
}

impl Emit for NativeEmit<'_> {
    fn despawn(&mut self, entity: u64) {
        self.buffer.push(Command::Despawn(EntityId(entity))).unwrap();
    }
    fn spawn(&mut self, position: [f32; 2], velocity: [f32; 2]) {
        let mut components = vec![
            (self.position, bytes(&position).to_vec()),
            (self.velocity, bytes(&velocity).to_vec()),
        ];
        components.sort_by_key(|(c, _)| *c);
        self.buffer.push(Command::Spawn(components)).unwrap();
    }
    fn add_stunned(&mut self, entity: u64, value: u32) {
        self.buffer
            .push(Command::Add(
                EntityId(entity),
                self.stunned,
                value.to_le_bytes().to_vec(),
            ))
            .unwrap();
    }
    fn remove_stunned(&mut self, entity: u64) {
        self.buffer
            .push(Command::Remove(EntityId(entity), self.stunned))
            .unwrap();
    }
}

fn run_native(memory: GuestMemory) -> Result<(u64, usize, usize, ApplyStats)> {
    let mut world = World::new()?;
    let mut buffer = CommandBuffer::default();
    for tick in 0..TICKS {
        for range in world.ranges(memory)? {
            buffer.begin(0, range.index);
            let n = range.count as usize;
            let positions =
                unsafe { std::slice::from_raw_parts_mut(memory.ptr(range.columns[0]).cast::<[f32; 2]>(), n) };
            let velocities = unsafe { std::slice::from_raw_parts(memory.ptr(range.columns[1]).cast::<[f32; 2]>(), n) };
            let entities = unsafe { std::slice::from_raw_parts(range.entities, n) };
            let mut emit = NativeEmit {
                buffer: &mut buffer,
                position: world.position,
                velocity: world.velocity,
                stunned: world.stunned,
            };
            for i in 0..n {
                lifecycle_row(entities[i], tick, &mut positions[i], velocities[i], &mut emit);
            }
        }
        let commands = merge([buffer.take()])?;
        world.apply(&commands)?;
    }
    let (digest, rows, stunned) = world.digest()?;
    Ok((digest, rows, stunned, world.stats))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MergeOrder {
    /// The engine's rule: sort by command key.
    ByKey,
    /// Negative control: worker 0's commands, then worker 1's, and so on.
    ByWorker,
}

struct GuestRun {
    digest: u64,
    rows: usize,
    stunned: usize,
    system_ms: f64,
    apply_ms: f64,
}

fn run_guests(
    set: &mut InstanceSet<WorkerCtx>,
    scratch: &[u32],
    workers: usize,
    order: MergeOrder,
) -> Result<GuestRun> {
    let memory = set.guest_memory();
    let mut world = World::new()?;
    let (cp, cv, cs) = (world.position.0, world.velocity.0, world.stunned.0);
    let mut functions: Vec<LifecycleFn> = Vec::new();
    for instance in set.instances.iter_mut().take(workers) {
        functions.push(instance.func("lifecycle")?);
    }
    let (mut system_s, mut apply_s) = (0.0, 0.0);
    for tick in 0..TICKS {
        let started = Instant::now();
        let work = world.ranges(memory)?;
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| -> Result<()> {
            let handles: Vec<_> = set
                .instances
                .iter_mut()
                .take(workers)
                .zip(&functions)
                .zip(scratch)
                .map(|((instance, lifecycle), &ids)| {
                    let (work, next) = (&work, &next);
                    scope.spawn(move || -> Result<()> {
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(range) = work.get(index) else { return Ok(()) };
                            // flecs keeps entity ids on the host heap; the guest gets a copy.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    range.entities,
                                    memory.ptr(ids).cast::<u64>(),
                                    range.count as usize,
                                )
                            };
                            instance.store.data_mut().commands.begin(0, range.index);
                            let args = (range.columns[0], range.columns[1], ids, range.count, tick, cp, cv, cs);
                            lifecycle
                                .call(&mut instance.store, args)
                                .map_err(|e| anyhow!("guest trapped: {e}"))?;
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().map_err(|_| anyhow!("worker panicked"))??;
            }
            Ok(())
        })?;
        system_s += started.elapsed().as_secs_f64();

        let started = Instant::now();
        let buffers: Vec<_> = set
            .instances
            .iter_mut()
            .take(workers)
            .map(|i| i.store.data_mut().commands.take())
            .collect();
        let commands = match order {
            MergeOrder::ByKey => merge(buffers)?,
            MergeOrder::ByWorker => buffers.into_iter().flatten().collect(),
        };
        world.apply(&commands)?;
        apply_s += started.elapsed().as_secs_f64();
    }
    let (digest, rows, stunned) = world.digest()?;
    Ok(GuestRun {
        digest,
        rows,
        stunned,
        system_ms: system_s * 1e3 / TICKS as f64,
        apply_ms: apply_s * 1e3 / TICKS as f64,
    })
}

#[test]
fn structural_commands_are_independent_of_worker_count() -> Result<()> {
    let wasm = build_guest()?;
    let module = SimModule::new(&wasm, Metering::None)?;
    let memory = module.new_memory()?;
    let ctx_memory = memory.clone();
    let workers = MAX_WORKERS.min(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let mut set = InstanceSetBuilder {
        module: &module,
        count: workers,
        stack_bytes: DEFAULT_STACK_BYTES,
        link: &imports::link,
        data: &|_| WorkerCtx::new(ctx_memory.clone()),
    }
    .build(memory, |memory, _owner| {
        let offset = claim_pages(memory, ARENA_PAGES)?;
        let base = GuestMemory::of(memory).ptr(offset);
        unsafe { ae_ecs::arena::install(base, (ARENA_PAGES * ae_wasm::PAGE_BYTES) as usize).map_err(|e| anyhow!(e))? };
        Ok(())
    })?;
    let mut scratch = Vec::new();
    for _ in 0..workers {
        scratch.push(set.malloc((RANGE_ROWS * 8) as u32)?);
    }

    let (reference, rows, stunned, stats) = run_native(set.guest_memory())?;
    eprintln!(
        "native: digest {reference:016x}, {rows} live, {stunned} stunned after {TICKS} ticks; spawned {} despawned {} added {} overwritten {} removed {} ignored {}",
        stats.spawned, stats.despawned, stats.added, stats.overwritten, stats.removed, stats.ignored
    );
    ensure!(
        stats.spawned > 1000 && stats.despawned > 1000 && stats.added > 1000 && stats.removed > 100,
        "the workload must exercise every command kind"
    );
    ensure!(
        stats.ignored > 0,
        "the workload should include commands on already-despawned entities"
    );
    ensure!(
        reference == EXPECTED_DIGEST,
        "native digest {reference:016x} differs from the pinned {EXPECTED_DIGEST:016x}"
    );

    for w in [1usize, 2, 4, 8].into_iter().filter(|w| *w <= workers) {
        let run = run_guests(&mut set, &scratch, w, MergeOrder::ByKey)?;
        eprintln!(
            "{w} worker(s): digest {:016x}, system {:.2} ms/tick, merge+apply {:.2} ms/tick",
            run.digest, run.system_ms, run.apply_ms
        );
        ensure!(
            (run.rows, run.stunned) == (rows, stunned),
            "{w} workers: {} live / {} stunned, native {rows} / {stunned}",
            run.rows,
            run.stunned
        );
        ensure!(
            run.digest == reference,
            "{w} workers: digest {:016x} differs from native {reference:016x}",
            run.digest
        );
    }

    // Negative control: without the key sort, multi-worker results depend on
    // which worker happened to run which range.
    if workers >= 4 {
        let distinct: std::collections::BTreeSet<u64> = (0..3)
            .map(|_| run_guests(&mut set, &scratch, workers, MergeOrder::ByWorker).map(|r| r.digest))
            .collect::<Result<_>>()?;
        eprintln!(
            "negative control, merged by worker instead of key: {} distinct digest(s) in 3 runs, {} the reference",
            distinct.len(),
            if distinct.contains(&reference) {
                "including"
            } else {
                "none equal to"
            }
        );
    }
    Ok(())
}
