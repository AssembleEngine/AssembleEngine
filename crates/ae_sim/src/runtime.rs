//! A sim module running a world: schema -> components, queries and
//! schedule; each tick runs the schedule layer by layer on the worker crew,
//! merging and applying commands and events at every layer end.
//!
//! One [`Runtime`] per process: the column arena and flecs' column allocator
//! are process-wide. A runtime can run several worlds one after another
//! (e.g. the same game at 1, 2, 4, 8 and 16 workers); each [`World`] starts
//! from the same empty state.

use crate::command::{Applier, ApplyStats, Command, CommandKey, Emitted, merge};
use crate::crew::{Batch, BatchSystem, Crew, Job, Worker, WorkerOut};
use crate::digest::StateHash;
use crate::events::EventStore;
use crate::imports::{self, IdMap, WorkerCtx};
use crate::schedule::{self, Cadence, Mode, Schedule, SystemDecl};
use crate::schema::{ComponentDecl, Schema};
use crate::work::{WorkRange, ranges};
use ae_abi::{EventSpan, MAX_EVENT_READS, MAX_TERMS, RUN_DESC_BYTES};
use ae_ecs::flecs_backend as fb;
use ae_ecs::{Access, ComponentId, Term};
use ae_wasm::{GuestMemory, InstanceSetBuilder, Metering, PAGE_BYTES, SimModule, claim_pages};
use anyhow::{Context, Result, anyhow, ensure};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct RuntimeConfig {
    /// Instances (and so the most workers a tick may use).
    pub max_workers: usize,
    /// Rows per work range.
    pub range_rows: usize,
    /// A parallel system with fewer rows than this in total runs as one
    /// job. Row counts are world state, so the choice is deterministic; it
    /// never changes results, only cost.
    pub parallel_min_rows: usize,
    /// Column arena size, in 64 KiB pages.
    pub arena_pages: u64,
    /// Per-layer event staging area in guest memory, in pages.
    pub event_pages: u64,
    pub stack_bytes: u32,
    pub metering: Metering,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_workers: 1,
            range_rows: 1024,
            parallel_min_rows: 2048,
            arena_pages: 4096,
            event_pages: 64,
            stack_bytes: ae_wasm::DEFAULT_STACK_BYTES,
            metering: Metering::None,
        }
    }
}

struct SystemRt {
    name: Arc<str>,
    decl: SystemDecl,
    /// Stable component ids of the terms, and which are written.
    term_ids: Vec<u64>,
    write_mask: u32,
}

pub struct Runtime {
    pub schema: Schema,
    pub schedule: Schedule,
    systems: Vec<SystemRt>,
    cfg: RuntimeConfig,
    memory: GuestMemory,
    crew: Crew,
    events_at: u32,
    events_len: usize,
    /// Profiler zone name per layer.
    layer_labels: Vec<String>,
}

/// Timing and counters of one tick.
#[derive(Clone, Debug, Default)]
pub struct TickStats {
    /// Running systems (guest calls), summed over layers.
    pub systems_ns: u64,
    /// Merging and applying commands and events.
    pub apply_ns: u64,
    /// Building work ranges and staging events.
    pub prepare_ns: u64,
    pub calls: u64,
    pub commands: u64,
    pub events: u64,
    pub apply: ApplyStats,
    /// Guest fuel per system (schema order), with fuel metering.
    pub fuel: Vec<u64>,
}

pub struct World {
    world: fb::World,
    ids: Arc<IdMap>,
    /// ECS id per schema component (events excluded: `None`).
    ecs_ids: Vec<Option<ComponentId>>,
    /// Query and terms per system (`None`: no query).
    queries: Vec<Option<(fb::Query, Vec<Term>)>>,
    /// One single-term query per hashed component.
    hash_queries: Vec<(usize, fb::Query, [Term; 1])>,
    applier: Applier,
    events: EventStore,
    pub tick: u64,
    pub param: u64,
}

impl Drop for World {
    fn drop(&mut self) {
        unsafe {
            for (q, _) in self.queries.iter().flatten() {
                fb::fini_query(*q);
            }
            for (_, q, _) in &self.hash_queries {
                fb::fini_query(*q);
            }
            fb::fini(self.world);
        }
    }
}

/// Digest of a world: total and per component (by name), plus held events.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Digest {
    pub total: u128,
    pub parts: Vec<(String, u128, u64)>,
}

impl Digest {
    /// Low 64 bits, for printing and pinning.
    pub fn short(&self) -> u64 {
        self.total as u64
    }
}

impl Runtime {
    pub fn new(wasm: &[u8], cfg: RuntimeConfig) -> Result<Self> {
        let schema = Schema::from_module(wasm)?;
        ensure!(cfg.max_workers >= 1 && cfg.range_rows >= 1, "bad runtime config");
        // The schedule only needs resource identity, so it is computed on
        // stable ids, once.
        let mut systems = Vec::new();
        let mut decls = Vec::new();
        for s in &schema.systems {
            let terms: Vec<Term> = s.terms.iter().map(|&(id, access)| Term { component: ComponentId(id), access }).collect();
            let write_mask = s.terms.iter().enumerate().filter(|(_, (_, a))| *a == Access::Write).fold(0u32, |m, (i, _)| m | 1 << i);
            let decl = SystemDecl {
                name: s.name.clone(),
                terms,
                emits: s.emits.clone(),
                reads_events: s.reads_events.clone(),
                after: s.after.clone(),
                mode: s.mode,
                cadence: s.cadence,
            };
            decls.push(decl.clone());
            systems.push(SystemRt { name: s.name.as_str().into(), decl, term_ids: s.terms.iter().map(|t| t.0).collect(), write_mask });
        }
        let schedule = schedule::compute(&decls)?;

        let module = SimModule::new(wasm, cfg.metering)?;
        let memory = module.new_memory()?;
        let ctx_memory = memory.clone();
        let mut events_at = 0;
        let mut set = InstanceSetBuilder {
            module: &module,
            count: cfg.max_workers,
            stack_bytes: cfg.stack_bytes,
            link: &imports::link,
            data: &|_| WorkerCtx::new(ctx_memory.clone()),
        }
        .build(memory, |memory, _| {
            let offset = claim_pages(memory, cfg.arena_pages)?;
            let base = GuestMemory::of(memory).ptr(offset);
            unsafe { ae_ecs::arena::install(base, (cfg.arena_pages * PAGE_BYTES) as usize).map_err(|e| anyhow!(e))? };
            events_at = claim_pages(memory, cfg.event_pages)?;
            Ok(())
        })?;
        let guest = set.guest_memory();
        let ids_cap = cfg.range_rows as u32;
        let mut scratch = Vec::new();
        for _ in 0..cfg.max_workers {
            let desc = set.malloc(RUN_DESC_BYTES as u32 + 8)?.next_multiple_of(8);
            let ids_buf = set.malloc(ids_cap * 8 + 8)?.next_multiple_of(8);
            scratch.push((desc, ids_buf));
        }
        let mut workers = Vec::new();
        for (mut instance, (desc, ids_buf)) in std::mem::take(&mut set.instances).into_iter().zip(scratch) {
            let funcs = schema.systems.iter().map(|s| instance.func::<u32, ()>(&s.export())).collect::<Result<_>>()?;
            workers.push(Worker { instance, funcs, desc, ids_buf, ids_cap, metered: cfg.metering == Metering::Fuel });
        }
        let crew = Crew::new(workers)?;
        let layer_labels = schedule.layers.iter().enumerate().map(|(k, l)| format!("layer {k}: {}", l.iter().map(|&i| schema.systems[i].name.as_str()).collect::<Vec<_>>().join(" "))).collect();
        Ok(Self { schema, schedule, systems, cfg, memory: guest, crew, events_at, events_len: (cfg.event_pages * PAGE_BYTES) as usize, layer_labels })
    }

    pub fn max_workers(&self) -> usize {
        self.crew.size()
    }

    /// A fresh world at tick 0. `param` reaches every system call.
    pub fn world(&self, param: u64) -> Result<World> {
        let world = unsafe { fb::init()? };
        let mut ids = IdMap::default();
        let mut ecs_ids = Vec::new();
        // Registration in stable-id order: flecs ids come out the same in every world.
        for c in &self.schema.components {
            if c.event {
                ids.streams.insert(c.id, c.size as usize);
                ecs_ids.push(None);
            } else {
                let id = unsafe { fb::register_component(world, &c.name, c.size, c.align)? };
                ids.components.insert(c.id, (id, c.size as usize));
                ecs_ids.push(Some(id));
            }
        }
        let mut emits = vec![Vec::new(); self.systems.len()];
        for (i, s) in self.systems.iter().enumerate() {
            emits[self.schedule.position[i] as usize] = s.decl.emits.clone();
        }
        ids.emits = emits;
        let ecs = |stable: u64| ids.components[&stable].0;
        let mut queries = Vec::new();
        for s in &self.systems {
            if s.decl.terms.is_empty() {
                queries.push(None);
                continue;
            }
            let terms: Vec<Term> = s.decl.terms.iter().map(|t| Term { component: ecs(t.component.0), access: t.access }).collect();
            queries.push(Some((unsafe { fb::create_query(world, &terms)? }, terms)));
        }
        let mut hash_queries = Vec::new();
        for (i, c) in self.schema.components.iter().enumerate() {
            if let Some(id) = ecs_ids[i]
                && c.fields.iter().any(|f| f.hashed())
            {
                let terms = [Term { component: id, access: Access::Read }];
                hash_queries.push((i, unsafe { fb::create_query(world, &terms)? }, terms));
            }
        }
        let mut events = EventStore::default();
        for c in self.schema.components.iter().filter(|c| c.event) {
            events.declare(c.id, c.size as usize);
        }
        for (i, s) in self.systems.iter().enumerate() {
            for &stream in &s.decl.reads_events {
                events.add_reader(stream, i)?;
            }
        }
        Ok(World { world, ids: Arc::new(ids), ecs_ids, queries, hash_queries, applier: Applier::default(), events, tick: 0, param })
    }

    /// Run one tick on `workers` workers.
    pub fn tick(&mut self, w: &mut World, workers: usize) -> Result<TickStats> {
        ae_profile::zone!("sim_tick");
        let tick = w.tick;
        let mut stats = TickStats { fuel: vec![0; self.systems.len()], ..Default::default() };
        if tick == 1 {
            for (i, s) in self.systems.iter().enumerate() {
                if s.decl.cadence == Cadence::Once {
                    w.events.retire(i);
                }
            }
        }
        for layer in 0..self.schedule.layers.len() {
            let _layer = ae_profile::zone_dyn(&self.layer_labels[layer]);
            let started = Instant::now();
            let batch = {
                ae_profile::zone!("prepare");
                self.prepare(w, layer, tick)?
            };
            stats.prepare_ns += started.elapsed().as_nanos() as u64;
            let Some(batch) = batch else { continue };
            let index_of: Vec<usize> = batch.systems.iter().map(|s| s.index).collect();
            let started = Instant::now();
            let outs = self.crew.run(Arc::new(batch), workers);
            stats.systems_ns += started.elapsed().as_nanos() as u64;

            ae_profile::zone!("merge_apply");
            let started = Instant::now();
            let mut commands = Vec::with_capacity(outs.len());
            let mut events: Vec<Emitted> = Vec::new();
            for out in outs {
                let WorkerOut { commands: c, events: e, fuel, calls, error } = out;
                if let Some(error) = error {
                    return Err(error.context(format!("tick {tick}, layer {layer}")));
                }
                commands.push(c);
                events.extend(e);
                stats.calls += calls;
                for (k, f) in fuel.into_iter().enumerate() {
                    stats.fuel[index_of[k]] += f;
                }
            }
            let commands: Vec<(CommandKey, Command)> = merge(commands)?;
            events.sort_unstable_by_key(|(key, stream, _)| (*key, *stream));
            ensure!(events.windows(2).all(|p| p[0].0 != p[1].0), "duplicate event key: a work range ran twice");
            stats.commands += commands.len() as u64;
            stats.events += events.len() as u64;
            let applied = unsafe { w.applier.apply(w.world, &commands)? };
            add_stats(&mut stats.apply, &applied);
            w.events.append(&events)?;
            stats.apply_ns += started.elapsed().as_nanos() as u64;
        }
        w.events.prune();
        w.tick += 1;
        ae_profile::plot!("sim commands", stats.commands);
        ae_profile::plot!("sim events", stats.events);
        ae_profile::frame_mark!("sim");
        Ok(stats)
    }

    /// Work ranges, staged events and jobs of one layer; `None` if no
    /// system of the layer runs this tick.
    fn prepare(&self, w: &mut World, layer: usize, tick: u64) -> Result<Option<Batch>> {
        let mut systems = Vec::new();
        let mut jobs = Vec::new();
        let mut staged = 0usize;
        for &i in &self.schedule.layers[layer] {
            let s = &self.systems[i];
            if !s.decl.cadence.runs_on(tick) {
                continue;
            }
            let ranges: Vec<WorkRange> = match &w.queries[i] {
                Some((q, terms)) => unsafe { ranges(w.world, *q, terms, self.cfg.range_rows, self.memory)? },
                None => vec![WorkRange { index: 0, entities: std::ptr::null(), count: 0, columns: [0; MAX_TERMS] }],
            };
            let mut events = [EventSpan::default(); MAX_EVENT_READS];
            for (k, &stream) in s.decl.reads_events.iter().enumerate() {
                let (bytes, count) = w.events.unseen(stream, i);
                ensure!(staged + bytes.len() <= self.events_len, "events of one layer exceed the {}-byte staging area", self.events_len);
                let at = self.events_at + staged as u32;
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.memory.ptr(at), bytes.len()) };
                staged = (staged + bytes.len()).next_multiple_of(16);
                events[k] = EventSpan { stream, ptr: at, count };
                w.events.mark_seen(stream, i);
            }
            let rows: usize = ranges.iter().map(|r| r.count as usize).sum();
            let index = systems.len();
            if s.decl.mode == Mode::Serial || rows < self.cfg.parallel_min_rows {
                if !ranges.is_empty() {
                    jobs.push(Job { system: index, first: 0, end: ranges.len() });
                }
            } else {
                jobs.extend((0..ranges.len()).map(|r| Job { system: index, first: r, end: r + 1 }));
            }
            let mut term_ids = [0u64; MAX_TERMS];
            term_ids[..s.term_ids.len()].copy_from_slice(&s.term_ids);
            systems.push(BatchSystem {
                name: s.name.clone(),
                index: i,
                position: self.schedule.position[i],
                term_count: s.term_ids.len() as u32,
                term_ids,
                write_mask: s.write_mask,
                stagger: s.decl.cadence.stagger(tick),
                events,
                event_count: s.decl.reads_events.len() as u32,
                ranges,
            });
        }
        if systems.is_empty() {
            return Ok(None);
        }
        Ok(Some(Batch::new(systems, jobs, tick, w.param, w.ids.clone(), self.memory)))
    }

    /// Order-independent digest of every hashed field of every live row,
    /// per component, plus the events still held.
    pub fn digest(&self, w: &World) -> Result<Digest> {
        let mut total = StateHash::new();
        let mut parts = Vec::new();
        let mut row = Vec::new();
        for (ci, q, terms) in &w.hash_queries {
            let c: &ComponentDecl = &self.schema.components[*ci];
            let mut fields: Vec<_> = c.fields.iter().filter(|f| f.hashed()).collect();
            fields.sort_by_key(|f| f.id);
            let mut part = StateHash::new();
            unsafe {
                fb::for_each_chunk(w.world, *q, terms, |chunk| {
                    let ids = std::slice::from_raw_parts(chunk.entities, chunk.count);
                    let data = std::slice::from_raw_parts(chunk.fields[0], chunk.count * c.size as usize);
                    for (r, id) in ids.iter().enumerate() {
                        let value = &data[r * c.size as usize..(r + 1) * c.size as usize];
                        row.clear();
                        for f in &fields {
                            row.extend_from_slice(&value[f.offset as usize..(f.offset + f.size) as usize]);
                        }
                        part.add_row(*id, c.id, &row);
                    }
                    Ok(())
                })?;
            }
            total.merge(&part);
            parts.push((c.name.clone(), part.finish(), part.rows()));
        }
        let events = w.events.digest();
        total.merge(&events);
        parts.push(("(events)".into(), events.finish(), w.events.held() as u64));
        Ok(Digest { total: total.finish(), parts })
    }

    /// Live rows of a component, by name.
    pub fn count(&self, w: &World, component: &str) -> Result<usize> {
        let i = self.schema.components.iter().position(|c| c.name == component).with_context(|| format!("no component `{component}`"))?;
        let id = w.ecs_ids[i].with_context(|| format!("`{component}` is an event stream"))?;
        let terms = [Term { component: id, access: Access::Read }];
        let mut n = 0;
        unsafe {
            let q = fb::create_query(w.world, &terms)?;
            let r = fb::for_each_chunk(w.world, q, &terms, |chunk| {
                n += chunk.count;
                Ok(())
            });
            fb::fini_query(q);
            r?;
        }
        Ok(n)
    }

    /// Every live row of a component, by name: `(entity, value bytes)`,
    /// sorted by entity.
    pub fn rows(&self, w: &World, component: &str) -> Result<Vec<(u64, Vec<u8>)>> {
        let i = self.schema.components.iter().position(|c| c.name == component).with_context(|| format!("no component `{component}`"))?;
        let id = w.ecs_ids[i].with_context(|| format!("`{component}` is an event stream"))?;
        let size = self.schema.components[i].size as usize;
        let terms = [Term { component: id, access: Access::Read }];
        let mut out = Vec::new();
        unsafe {
            let q = fb::create_query(w.world, &terms)?;
            let r = fb::for_each_chunk(w.world, q, &terms, |chunk| {
                let ids = std::slice::from_raw_parts(chunk.entities, chunk.count);
                let data = std::slice::from_raw_parts(chunk.fields[0], chunk.count * size);
                for (r, id) in ids.iter().enumerate() {
                    out.push((*id, data[r * size..(r + 1) * size].to_vec()));
                }
                Ok(())
            });
            fb::fini_query(q);
            r?;
        }
        out.sort_by_key(|(e, _)| *e);
        Ok(out)
    }

    pub fn events_held(&self, w: &World) -> usize {
        w.events.held()
    }
}

fn add_stats(t: &mut ApplyStats, s: &ApplyStats) {
    t.spawned += s.spawned;
    t.despawned += s.despawned;
    t.added += s.added;
    t.overwritten += s.overwritten;
    t.removed += s.removed;
    t.ignored += s.ignored;
}
