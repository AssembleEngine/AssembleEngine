//! A sim module running a world: schema -> components, queries and
//! schedule; each tick runs the schedule layer by layer on the worker crew,
//! merging and applying commands and events at every layer end. Host
//! systems ([`crate::host`]) the module schedules run in their layer after
//! the module's systems, on the calling thread.
//!
//! One [`Runtime`] per process: the column arena and flecs' column allocator
//! are process-wide. A runtime can run several worlds one after another
//! (e.g. the same game at 1, 2, 4, 8 and 16 workers); each [`World`] starts
//! from the same empty state.

use crate::capture::{Capture, CaptureSpec};
use crate::command::{Applier, ApplyStats, Command, CommandKey, Emitted, merge};
use crate::crew::{Batch, BatchSystem, Crew, Job, Worker, WorkerOut};
use crate::digest::StateHash;
use crate::events::EventStore;
use crate::host::{HostCtx, HostDecl, HostInit, HostPlugin, HostSystem, SpatialIndex};
use crate::imports::{self, IdMap, WorkerCtx, WorldPtr};
use crate::schedule::{self, Cadence, Mode, Schedule, SystemDecl};
use crate::schema::{ComponentDecl, Schema, Slot};
use crate::work::{WorkRange, ranges};
use ae_abi::{EventSpan, MAX_EVENT_READS, MAX_TERMS, RUN_DESC_BYTES};
use ae_ecs::flecs_backend as fb;
use ae_ecs::{Access, ComponentId, Term};
use ae_wasm::{GuestMemory, InstanceSetBuilder, Metering, PAGE_BYTES, SimModule, claim_pages};
use anyhow::{Context, Result, anyhow, ensure};
use std::sync::Arc;
use std::time::Instant;

/// Where a module's coroutine slots are (`ae_abi::script`).
#[derive(Clone, Copy, Debug)]
struct ScriptArena {
    offset: u32,
    slot_bytes: u32,
    slots: u32,
}

impl ScriptArena {
    fn len(&self) -> usize {
        self.slot_bytes as usize * self.slots as usize
    }
}

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
    /// Run a layer's host systems on the coordinating thread while the
    /// crew's helpers start on its module systems, then join them
    /// (otherwise: module systems first, then host systems). Never changes
    /// results: commands and events are keyed by schedule position.
    pub overlap_hosts: bool,
    /// Ticks a second. Published to modules (`ae_abi::sim::TICK_HZ`) and
    /// host plugins ([`HostInit::tick_hz`]); a world's results depend on it.
    pub tick_hz: u32,
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
            overlap_hosts: true,
            tick_hz: 30,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `schema.systems[i]`.
    Module(usize),
    /// `hosts[i]`.
    Host(usize),
}

struct SystemRt {
    name: Arc<str>,
    decl: SystemDecl,
    /// Stable component ids of the query terms, and which are written.
    term_ids: Vec<u64>,
    write_mask: u32,
    kind: Kind,
}

/// A host system the module schedules, and the plugin that provides it.
struct HostSlot {
    plugin: usize,
    decl: HostDecl,
}

pub struct Runtime {
    pub schema: Schema,
    /// Over all systems, module and host, in `schedule!` order (see
    /// [`Self::system_name`]).
    pub schedule: Schedule,
    systems: Vec<SystemRt>,
    hosts: Vec<HostSlot>,
    plugins: Vec<Arc<dyn HostPlugin>>,
    cfg: RuntimeConfig,
    memory: GuestMemory,
    shared: wasmtime::SharedMemory,
    crew: Crew,
    events_at: u32,
    /// The module's coroutine slot arena (`ae_abi::script`), if it declares one.
    scripts: Option<ScriptArena>,
    events_len: usize,
    /// Profiler zone name per layer.
    layer_labels: Vec<String>,
    /// Where this runtime's column arena starts (released on drop, so a
    /// later runtime in the process can install its own).
    arena_base: usize,
}

impl Drop for Runtime {
    /// Worlds must be dropped first (their columns live in this memory).
    fn drop(&mut self) {
        ae_ecs::arena::release(self.arena_base as *mut u8);
    }
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
    /// Running host systems, summed.
    pub host_ns: u64,
    /// Guest fuel per system (runtime order), with fuel metering.
    pub fuel: Vec<u64>,
    /// Wall time per host system (runtime order; zero for module systems).
    pub host_system_ns: Vec<u64>,
    /// Commands of the tick's bundle appended to their streams, and dropped
    /// (unknown or non-command stream, wrong size).
    pub commands_in: u64,
    pub commands_dropped: u64,
}

pub struct World {
    world: fb::World,
    ids: Arc<IdMap>,
    /// ECS id per schema component (events excluded: `None`).
    ecs_ids: Vec<Option<ComponentId>>,
    /// Query and terms per system (`None`: no query).
    queries: Vec<Option<(fb::Query, Vec<Term>)>>,
    /// State of each host system (by host slot).
    hosts: Vec<Box<dyn HostSystem>>,
    spatial: Option<Arc<SpatialIndex>>,
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
        Self::with_hosts(wasm, cfg, Vec::new())
    }

    /// A runtime whose module may schedule host systems from `plugins`
    /// (the first plugin providing a name wins).
    pub fn with_hosts(wasm: &[u8], cfg: RuntimeConfig, plugins: Vec<Arc<dyn HostPlugin>>) -> Result<Self> {
        let schema = Schema::from_module(wasm)?;
        ensure!(
            cfg.max_workers >= 1 && cfg.range_rows >= 1 && cfg.tick_hz >= 1,
            "bad runtime config"
        );
        let offered: Vec<Vec<HostDecl>> = plugins.iter().map(|p| p.systems()).collect();
        // The schedule only needs resource identity, so it is computed on
        // stable ids, once.
        let mut systems = Vec::new();
        let mut hosts = Vec::new();
        for slot in &schema.order {
            match slot {
                Slot::Module(i) => {
                    let s = &schema.systems[*i];
                    let terms: Vec<Term> = s
                        .terms
                        .iter()
                        .map(|&(id, access)| Term {
                            component: ComponentId(id),
                            access,
                        })
                        .collect();
                    let decl = SystemDecl {
                        name: s.name.clone(),
                        terms,
                        emits: s.emits.clone(),
                        reads_events: s.reads_events.clone(),
                        gets: s.gets.clone(),
                        after: s.after.clone(),
                        mode: s.mode,
                        cadence: s.cadence,
                    };
                    systems.push(SystemRt {
                        name: s.name.as_str().into(),
                        decl,
                        term_ids: s.terms.iter().map(|t| t.0).collect(),
                        write_mask: write_mask(&s.terms),
                        kind: Kind::Module(*i),
                    });
                }
                Slot::Host(name) => {
                    let (plugin, decl) = offered
                        .iter()
                        .enumerate()
                        .find_map(|(p, decls)| decls.iter().find(|d| d.name == *name).map(|d| (p, d.clone())))
                        .with_context(|| {
                            format!("the module schedules host system `{name}`, which no host plugin provides")
                        })?;
                    plugins[plugin]
                        .check(name, &schema)
                        .with_context(|| format!("host system `{name}`"))?;
                    decl.cadence.validate()?;
                    // The row filter is the SDK's (`run.rows()`); a host system sees whole tables.
                    ensure!(
                        !matches!(decl.cadence, Cadence::Staggered { .. }),
                        "host system `{name}`: staggered cadence is for module systems"
                    );
                    ensure!(
                        decl.query.len() <= MAX_TERMS && decl.reads_events.len() <= MAX_EVENT_READS,
                        "host system `{name}`: too many terms or streams"
                    );
                    for &(id, _) in &decl.query {
                        ensure!(
                            schema.component(id).is_some_and(|c| !c.event),
                            "host system `{name}` queries component {id:#x}, which the module does not declare"
                        );
                    }
                    for &id in decl.emits.iter().chain(&decl.reads_events) {
                        ensure!(
                            schema.component(id).is_some_and(|c| c.event),
                            "host system `{name}` uses event stream {id:#x}, which the module does not declare"
                        );
                    }
                    let mut terms: Vec<Term> = decl
                        .query
                        .iter()
                        .map(|&(id, access)| Term {
                            component: ComponentId(id),
                            access,
                        })
                        .collect();
                    // Created components order like writes (no query term).
                    terms.extend(
                        decl.creates
                            .iter()
                            .filter(|id| !decl.query.iter().any(|q| q.0 == **id))
                            .map(|&id| Term {
                                component: ComponentId(id),
                                access: Access::Write,
                            }),
                    );
                    let sd = SystemDecl {
                        name: name.clone(),
                        terms,
                        emits: decl.emits.clone(),
                        reads_events: decl.reads_events.clone(),
                        gets: Vec::new(),
                        after: decl.after.clone(),
                        mode: Mode::Serial,
                        cadence: decl.cadence,
                    };
                    systems.push(SystemRt {
                        name: name.as_str().into(),
                        decl: sd,
                        term_ids: decl.query.iter().map(|t| t.0).collect(),
                        write_mask: write_mask(&decl.query),
                        kind: Kind::Host(hosts.len()),
                    });
                    hosts.push(HostSlot { plugin, decl });
                }
            }
        }
        let decls: Vec<SystemDecl> = systems.iter().map(|s| s.decl.clone()).collect();
        let schedule = schedule::compute(&decls)?;

        let module = SimModule::new(wasm, cfg.metering)?;
        let memory = module.new_memory()?;
        let ctx_memory = memory.clone();
        let shared = memory.clone();
        let mut events_at = 0;
        let mut arena_base = 0usize;
        let script_decl = match schema.data.get(ae_abi::script::DATA) {
            Some(d) => {
                ensure!(d.len() >= 8, "`{}` record is {} bytes", ae_abi::script::DATA, d.len());
                let (slot_bytes, slots) = (ae_abi::get_u32(d, 0), ae_abi::get_u32(d, 4));
                ensure!(
                    slot_bytes > 0 && slot_bytes.is_multiple_of(ae_abi::script::SLOT_ALIGN) && slots > 0,
                    "bad script slot arena: {slots} slots of {slot_bytes} bytes"
                );
                Some((slot_bytes, slots))
            }
            None => None,
        };
        let mut scripts = None;
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
            arena_base = base as usize;
            events_at = claim_pages(memory, cfg.event_pages)?;
            if let Some((slot_bytes, slots)) = script_decl {
                let len = slot_bytes as u64 * slots as u64;
                let offset = claim_pages(memory, len.div_ceil(PAGE_BYTES))?;
                scripts = Some(ScriptArena {
                    offset,
                    slot_bytes,
                    slots,
                });
            }
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
            let funcs = schema
                .systems
                .iter()
                .map(|s| instance.func::<u32, ()>(&s.export()))
                .collect::<Result<_>>()?;
            workers.push(Worker {
                instance,
                funcs,
                desc,
                ids_buf,
                ids_cap,
                metered: cfg.metering == Metering::Fuel,
            });
        }
        let crew = Crew::new(workers)?;
        let layer_labels = schedule
            .layers
            .iter()
            .enumerate()
            .map(|(k, l)| {
                format!(
                    "layer {k}: {}",
                    l.iter().map(|&i| &*systems[i].name).collect::<Vec<_>>().join(" ")
                )
            })
            .collect();
        Ok(Self {
            schema,
            schedule,
            systems,
            hosts,
            plugins,
            cfg,
            memory: guest,
            shared,
            crew,
            events_at,
            scripts,
            events_len: (cfg.event_pages * PAGE_BYTES) as usize,
            layer_labels,
            arena_base,
        })
    }

    /// Current size of the shared guest memory, in bytes (column arena,
    /// event staging and the guest heap).
    /// Switch [`RuntimeConfig::overlap_hosts`] (results never change).
    pub fn set_overlap_hosts(&mut self, on: bool) {
        self.cfg.overlap_hosts = on;
    }

    pub fn memory_bytes(&self) -> usize {
        self.shared.data_size()
    }

    /// The coroutine slot arena's bytes (`ae_abi::script`; empty when the
    /// module declares none). It belongs to the newest world: each new
    /// world zeroes it. Between ticks only: workers write it during one.
    pub fn script_slots(&self) -> &[u8] {
        match &self.scripts {
            Some(a) => unsafe { std::slice::from_raw_parts(self.memory.ptr(a.offset), a.len()) },
            None => &[],
        }
    }

    pub fn max_workers(&self) -> usize {
        self.crew.size()
    }

    /// Systems, module and host, in `schedule!` order: the indices of
    /// [`Self::schedule`] and of [`TickStats::fuel`].
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    pub fn system_name(&self, i: usize) -> &str {
        &self.systems[i].name
    }

    pub fn is_host_system(&self, i: usize) -> bool {
        matches!(self.systems[i].kind, Kind::Host(_))
    }

    /// A fresh world at tick 0. `param` reaches every system call. Host
    /// systems size their own thread pools for the most workers.
    pub fn world(&self, param: u64) -> Result<World> {
        self.world_for(param, self.max_workers())
    }

    /// A fresh world whose host systems size their thread pools for
    /// `workers` (results never depend on it).
    pub fn world_for(&self, param: u64, workers: usize) -> Result<World> {
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
        ids.names
            .insert((ae_abi::sim::SCOPE, ae_abi::sim::TICK_HZ), self.cfg.tick_hz);
        if let Some(a) = &self.scripts {
            // Every world starts with empty slots.
            unsafe { std::ptr::write_bytes(self.memory.ptr(a.offset), 0, a.len()) };
            use ae_abi::script::{BASE, SCOPE, SLOT_BYTES, SLOTS};
            ids.names.extend([
                ((SCOPE, BASE), a.offset),
                ((SCOPE, SLOT_BYTES), a.slot_bytes),
                ((SCOPE, SLOTS), a.slots),
            ]);
        }
        for plugin in &self.plugins {
            for (scope, name, value) in plugin.names() {
                ids.names.insert((scope, name), value);
            }
            for op in plugin.queries() {
                ids.queries.0.insert(op, plugin.clone());
            }
        }
        let ecs = |stable: u64| ids.components[&stable].0;
        let mut queries = Vec::new();
        for s in &self.systems {
            // Query terms come first; a host system's `creates` follow them
            // for scheduling only.
            let query = &s.decl.terms[..s.term_ids.len()];
            if query.is_empty() {
                queries.push(None);
                continue;
            }
            let terms: Vec<Term> = query
                .iter()
                .map(|t| Term {
                    component: ecs(t.component.0),
                    access: t.access,
                })
                .collect();
            queries.push(Some((unsafe { fb::create_query(world, &terms)? }, terms)));
        }
        let mut hash_queries = Vec::new();
        for (i, c) in self.schema.components.iter().enumerate() {
            if let Some(id) = ecs_ids[i]
                && c.fields.iter().any(|f| f.hashed())
            {
                let terms = [Term {
                    component: id,
                    access: Access::Read,
                }];
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
        let init = HostInit {
            param,
            workers,
            tick_hz: self.cfg.tick_hz,
        };
        let mut hosts: Vec<Option<Box<dyn HostSystem>>> = (0..self.hosts.len()).map(|_| None).collect();
        for (p, plugin) in self.plugins.iter().enumerate() {
            let mine: Vec<usize> = (0..self.hosts.len()).filter(|&h| self.hosts[h].plugin == p).collect();
            if mine.is_empty() {
                continue;
            }
            let names: Vec<&str> = mine.iter().map(|&h| self.hosts[h].decl.name.as_str()).collect();
            let made = plugin.instantiate(&init, &self.schema, &names)?;
            ensure!(
                made.len() == mine.len(),
                "host plugin made {} systems for {} names",
                made.len(),
                mine.len()
            );
            for (h, system) in mine.into_iter().zip(made) {
                hosts[h] = Some(system);
            }
        }
        let hosts = hosts
            .into_iter()
            .map(|h| h.expect("every host slot has a plugin"))
            .collect();
        Ok(World {
            world,
            ids: Arc::new(ids),
            ecs_ids,
            queries,
            hash_queries,
            hosts,
            spatial: None,
            applier: Applier::default(),
            events,
            tick: 0,
            param,
        })
    }

    /// Run one tick on `workers` workers.
    pub fn tick(&mut self, w: &mut World, workers: usize) -> Result<TickStats> {
        let tick = w.tick;
        self.tick_with(w, workers, &crate::input::TickBundle::new(tick))
    }

    /// Whether players may send commands on `stream`.
    pub fn is_command_stream(&self, stream: u64) -> bool {
        self.schema.component(stream).is_some_and(|c| c.event && c.command)
    }

    /// One tick with the players' commands for it (`bundle.tick` must be the
    /// world's tick): valid commands are appended to their event streams
    /// before the first layer, in bundle order; invalid ones are dropped and
    /// counted, identically on every peer.
    pub fn tick_with(&mut self, w: &mut World, workers: usize, bundle: &crate::input::TickBundle) -> Result<TickStats> {
        ae_profile::zone!("sim_tick");
        let tick = w.tick;
        ensure!(bundle.tick == tick, "a bundle for tick {} at tick {tick}", bundle.tick);
        let mut stats = TickStats {
            fuel: vec![0; self.systems.len()],
            host_system_ns: vec![0; self.systems.len()],
            ..Default::default()
        };
        if !bundle.commands.is_empty() {
            let mut accepted = Vec::with_capacity(bundle.commands.len());
            for (seq, c) in bundle.commands.iter().enumerate() {
                let ok = self
                    .schema
                    .component(c.stream)
                    .is_some_and(|d| d.event && d.command && d.size as usize == c.bytes.len());
                if ok {
                    accepted.push((
                        crate::command::CommandKey {
                            system: u32::MAX,
                            range: c.player,
                            seq: seq as u32,
                        },
                        c.stream,
                        c.bytes.clone(),
                    ));
                } else {
                    stats.commands_dropped += 1;
                }
            }
            stats.commands_in = accepted.len() as u64;
            w.events.append(&accepted)?;
        }
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
            let hosts: Vec<usize> = self.schedule.layers[layer]
                .iter()
                .copied()
                .filter(|&i| {
                    matches!(self.systems[i].kind, Kind::Host(_)) && self.systems[i].decl.cadence.runs_on(tick)
                })
                .collect();
            if batch.is_none() && hosts.is_empty() {
                continue;
            }
            let mut commands = Vec::new();
            let mut events: Vec<Emitted> = Vec::new();
            // Host systems' queries are read here, on the coordinating
            // thread; their runs then touch only their own rows, which no
            // module system of the layer writes.
            let jobs: Vec<HostJob> = hosts.iter().map(|&i| self.host_job(w, i)).collect::<Result<_>>()?;
            let (crew_outs, host_outs) = {
                let World {
                    hosts: states,
                    events: store,
                    ids,
                    spatial,
                    param,
                    ..
                } = &mut *w;
                let (memory, param) = (self.memory, *param);
                let run = move || run_hosts(jobs, memory, tick, param, states, store, ids, spatial);
                match batch {
                    None => (None, run()),
                    Some(batch) => {
                        let index_of: Vec<usize> = batch.systems.iter().map(|s| s.index).collect();
                        let started = Instant::now();
                        let (outs, host_outs) = if self.cfg.overlap_hosts {
                            // The helpers start the batch while this thread
                            // runs the host systems, then joins them.
                            let (outs, host_outs) = self.crew.run_beside(Arc::new(batch), workers, run);
                            // Module-system time beyond the host systems'.
                            let host: u64 = host_outs.as_ref().map(|h| h.iter().map(|o| o.2).sum()).unwrap_or(0);
                            stats.systems_ns += (started.elapsed().as_nanos() as u64).saturating_sub(host);
                            (outs, host_outs)
                        } else {
                            let outs = self.crew.run(Arc::new(batch), workers);
                            stats.systems_ns += started.elapsed().as_nanos() as u64;
                            (outs, run())
                        };
                        (Some((index_of, outs)), host_outs)
                    }
                }
            };
            if let Some((index_of, outs)) = crew_outs {
                for out in outs {
                    let WorkerOut {
                        commands: c,
                        events: e,
                        fuel,
                        calls,
                        error,
                    } = out;
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
            }
            for (i, (c, e), ns) in host_outs.with_context(|| format!("tick {tick}, layer {layer}"))? {
                stats.host_ns += ns;
                stats.host_system_ns[i] += ns;
                stats.calls += 1;
                commands.push(c);
                events.extend(e);
            }

            ae_profile::zone!("merge_apply");
            let started = Instant::now();
            let commands: Vec<(CommandKey, Command)> = merge(commands)?;
            events.sort_unstable_by_key(|(key, stream, _)| (*key, *stream));
            ensure!(
                events.windows(2).all(|p| p[0].0 != p[1].0),
                "duplicate event key: a work range ran twice"
            );
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
            let Kind::Module(module) = s.kind else { continue };
            if !s.decl.cadence.runs_on(tick) {
                continue;
            }
            let ranges: Vec<WorkRange> = match &w.queries[i] {
                Some((q, terms)) => unsafe { ranges(w.world, *q, terms, self.cfg.range_rows, self.memory)? },
                None => vec![WorkRange {
                    index: 0,
                    entities: std::ptr::null(),
                    count: 0,
                    columns: [0; MAX_TERMS],
                }],
            };
            let mut events = [EventSpan::default(); MAX_EVENT_READS];
            for (k, &stream) in s.decl.reads_events.iter().enumerate() {
                let (bytes, count) = w.events.unseen(stream, i);
                ensure!(
                    staged + bytes.len() <= self.events_len,
                    "events of one layer exceed the {}-byte staging area",
                    self.events_len
                );
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
                    jobs.push(Job {
                        system: index,
                        first: 0,
                        end: ranges.len(),
                    });
                }
            } else {
                jobs.extend((0..ranges.len()).map(|r| Job {
                    system: index,
                    first: r,
                    end: r + 1,
                }));
            }
            let mut term_ids = [0u64; MAX_TERMS];
            term_ids[..s.term_ids.len()].copy_from_slice(&s.term_ids);
            systems.push(BatchSystem {
                name: s.name.clone(),
                index: i,
                module,
                position: self.schedule.position[i],
                term_count: s.term_ids.len() as u32,
                term_ids,
                write_mask: s.write_mask,
                stagger: s.decl.cadence.stagger(tick),
                events,
                event_count: s.decl.reads_events.len() as u32,
                gets: {
                    let mut gets = [0; ae_abi::MAX_GETS];
                    gets[..s.decl.gets.len()].copy_from_slice(&s.decl.gets);
                    gets
                },
                get_count: s.decl.gets.len() as u32,
                ranges,
            });
        }
        if systems.is_empty() {
            return Ok(None);
        }
        Ok(Some(Batch::new(
            systems,
            jobs,
            tick,
            w.param,
            w.ids.clone(),
            self.memory,
            w.spatial.clone(),
            WorldPtr(w.world),
        )))
    }

    /// Host system `i`'s (runtime order) inputs for this run.
    fn host_job(&self, w: &World, i: usize) -> Result<HostJob> {
        let s = &self.systems[i];
        let Kind::Host(h) = s.kind else { unreachable!() };
        let decl = &self.hosts[h].decl;
        let (ranges, sizes) = match &w.queries[i] {
            Some((q, terms)) => {
                let sizes = terms
                    .iter()
                    .map(|t| unsafe { fb::component_size(w.world, t.component) })
                    .collect::<Result<Vec<_>>>()?;
                (unsafe { ranges(w.world, *q, terms, usize::MAX, self.memory)? }, sizes)
            }
            None => (Vec::new(), Vec::new()),
        };
        Ok(HostJob {
            index: i,
            slot: h,
            name: s.name.clone(),
            position: self.schedule.position[i],
            ranges,
            sizes,
            reads: decl.reads_events.clone(),
            emits: decl.emits.clone(),
        })
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
        for (i, s) in self.systems.iter().enumerate() {
            let Kind::Host(h) = s.kind else { continue };
            if let Some(hash) = w.hosts[h].digest() {
                let mut part = StateHash::new();
                part.add_row(u64::MAX - 1, ae_abi::name_id(&s.name), &hash.to_le_bytes());
                total.merge(&part);
                parts.push((format!("(host {})", s.name), part.finish(), i as u64));
            }
        }
        Ok(Digest {
            total: total.finish(),
            parts,
        })
    }

    /// Live rows of a component, by name.
    pub fn count(&self, w: &World, component: &str) -> Result<usize> {
        let i = self
            .schema
            .components
            .iter()
            .position(|c| c.name == component)
            .with_context(|| format!("no component `{component}`"))?;
        let id = w.ecs_ids[i].with_context(|| format!("`{component}` is an event stream"))?;
        let terms = [Term {
            component: id,
            access: Access::Read,
        }];
        let mut n = 0;
        unsafe {
            let q = fb::create_query_uncached(w.world, &terms)?;
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
        let mut out = Vec::new();
        self.each_row(w, component, |e, bytes| out.push((e, bytes.to_vec())))?;
        out.sort_by_key(|(e, _)| *e);
        Ok(out)
    }

    /// `f(entity, value bytes)` for every live row of a component, by name,
    /// in table order (no copies).
    pub fn each_row(&self, w: &World, component: &str, mut f: impl FnMut(u64, &[u8])) -> Result<()> {
        self.each_chunk(w, component, |ids, data| {
            let size = data.len() / ids.len().max(1);
            for (r, id) in ids.iter().enumerate() {
                f(*id, &data[r * size..(r + 1) * size]);
            }
        })
    }

    /// Every chunk of entities with `component`, borrowed from the world:
    /// their ids and the column's bytes (as [`Self::each_chunk`]). The
    /// slices stay valid while `w` is borrowed: nothing changes a world's
    /// tables without `&mut World`.
    pub fn chunks<'w>(&self, w: &'w World, component: &str) -> Result<Vec<(&'w [u64], &'w [u8])>> {
        let mut out = Vec::new();
        self.each_chunk(w, component, |ids, data| {
            // SAFETY: the tables' memory outlives the query, until the
            // world is next changed (which needs `&mut World`).
            out.push(unsafe {
                (
                    std::slice::from_raw_parts(ids.as_ptr(), ids.len()),
                    std::slice::from_raw_parts(data.as_ptr(), data.len()),
                )
            });
        })?;
        Ok(out)
    }

    /// Every chunk of entities with `component` (between ticks): their ids
    /// and the column's bytes, a row of the component's size per entity.
    pub fn each_chunk(&self, w: &World, component: &str, mut f: impl FnMut(&[u64], &[u8])) -> Result<()> {
        let i = self
            .schema
            .components
            .iter()
            .position(|c| c.name == component)
            .with_context(|| format!("no component `{component}`"))?;
        let id = w.ecs_ids[i].with_context(|| format!("`{component}` is an event stream"))?;
        let size = self.schema.components[i].size as usize;
        let terms = [Term {
            component: id,
            access: Access::Read,
        }];
        unsafe {
            let q = fb::create_query_uncached(w.world, &terms)?;
            let r = fb::for_each_chunk(w.world, q, &terms, |chunk| {
                let ids = std::slice::from_raw_parts(chunk.entities, chunk.count);
                let data = std::slice::from_raw_parts(chunk.fields[0], chunk.count * size);
                f(ids, data);
                Ok(())
            });
            fb::fini_query(q);
            r
        }
    }

    /// Overwrite 4-byte field `component.field` of `entity` with `bits`
    /// (between ticks). For fault injection (desync tests and tools): the
    /// world no longer follows from its commands. Whether the entity had
    /// the component.
    pub fn poke(&self, w: &mut World, entity: u64, component: &str, field: &str, bits: u32) -> Result<bool> {
        let i = self
            .schema
            .components
            .iter()
            .position(|c| c.name == component)
            .with_context(|| format!("no component `{component}`"))?;
        let c = &self.schema.components[i];
        let f = c
            .fields
            .iter()
            .find(|f| f.name == field)
            .with_context(|| format!("`{component}` has no field `{field}`"))?;
        ensure!(f.size == 4, "`{component}.{field}` is {} bytes; poke writes 4", f.size);
        let id = w.ecs_ids[i].with_context(|| format!("`{component}` is an event stream"))?;
        let (size, offset) = (c.size as usize, f.offset as usize);
        let terms = [Term {
            component: id,
            access: Access::Write,
        }];
        let mut found = false;
        unsafe {
            let q = fb::create_query_uncached(w.world, &terms)?;
            let r = fb::for_each_chunk(w.world, q, &terms, |chunk| {
                let ids = std::slice::from_raw_parts(chunk.entities, chunk.count);
                if let Some(r) = ids.iter().position(|&e| e == entity) {
                    chunk.fields[0]
                        .add(r * size + offset)
                        .cast::<u32>()
                        .write_unaligned(bits);
                    found = true;
                }
                Ok(())
            });
            fb::fini_query(q);
            r?;
        }
        Ok(found)
    }

    /// Copy the fields of `spec` from every entity that has all its
    /// components into `out` (between ticks; see [`crate::capture`]).
    pub fn capture(&self, w: &World, spec: &CaptureSpec, out: &mut Capture) -> Result<()> {
        let mut terms = Vec::with_capacity(spec.components.len());
        for &ci in &spec.components {
            let id =
                w.ecs_ids[ci].with_context(|| format!("`{}` is an event stream", self.schema.components[ci].name))?;
            terms.push(Term {
                component: id,
                access: Access::Read,
            });
        }
        out.tick = w.tick;
        out.entities.clear();
        out.columns.resize_with(spec.fields.len(), Vec::new);
        for column in &mut out.columns {
            column.clear();
        }
        unsafe {
            let q = fb::create_query_uncached(w.world, &terms)?;
            let r = fb::for_each_chunk(w.world, q, &terms, |chunk| {
                out.entities
                    .extend_from_slice(std::slice::from_raw_parts(chunk.entities, chunk.count));
                for (column, &(term, offset)) in out.columns.iter_mut().zip(&spec.fields) {
                    let base = chunk.fields[term].cast_const();
                    let stride = spec.sizes[term];
                    column
                        .extend((0..chunk.count).map(|r| base.add(r * stride + offset).cast::<u32>().read_unaligned()));
                }
                Ok(())
            });
            fb::fini_query(q);
            r?;
        }
        Ok(())
    }

    /// [`HostSystem::digest`] of the host system named `name`.
    pub fn host_digest(&self, w: &World, name: &str) -> Option<u128> {
        self.systems.iter().find_map(|s| match s.kind {
            Kind::Host(h) if &*s.name == name => w.hosts[h].digest(),
            _ => None,
        })
    }

    pub fn events_held(&self, w: &World) -> usize {
        w.events.held()
    }
}

/// A host system run's commands and events.
type HostOut = (Vec<(CommandKey, Command)>, Vec<Emitted>);

/// One host system's run, prepared on the coordinating thread.
struct HostJob {
    /// Runtime order, host slot.
    index: usize,
    slot: usize,
    name: Arc<str>,
    position: u32,
    ranges: Vec<WorkRange>,
    sizes: Vec<usize>,
    reads: Vec<u64>,
    emits: Vec<u64>,
}

/// Run a layer's host systems in schedule order: per system, its runtime
/// index, commands and events, and run time (ns).
#[allow(clippy::too_many_arguments)]
fn run_hosts(
    jobs: Vec<HostJob>,
    memory: GuestMemory,
    tick: u64,
    param: u64,
    states: &mut [Box<dyn HostSystem>],
    events: &mut EventStore,
    ids: &IdMap,
    spatial: &mut Option<Arc<SpatialIndex>>,
) -> Result<Vec<(usize, HostOut, u64)>> {
    let mut out = Vec::with_capacity(jobs.len());
    for job in jobs {
        let _zone = ae_profile::zone_dyn(&job.name);
        let started = Instant::now();
        let mut buffer = crate::command::CommandBuffer::default();
        buffer.begin(job.position, 0);
        let mut ctx = HostCtx {
            tick,
            param,
            ranges: &job.ranges,
            sizes: &job.sizes,
            memory,
            events,
            reader: job.index,
            reads: &job.reads,
            emits: &job.emits,
            commands: &mut buffer,
            ids,
            spatial,
        };
        states[job.slot]
            .run(&mut ctx)
            .with_context(|| format!("host system `{}`", job.name))?;
        let e = buffer.take_events();
        out.push((job.index, (buffer.take(), e), started.elapsed().as_nanos() as u64));
    }
    Ok(out)
}

fn write_mask(terms: &[(u64, Access)]) -> u32 {
    terms
        .iter()
        .enumerate()
        .filter(|(_, (_, a))| *a == Access::Write)
        .fold(0u32, |m, (i, _)| m | 1 << i)
}

fn add_stats(t: &mut ApplyStats, s: &ApplyStats) {
    t.spawned += s.spawned;
    t.despawned += s.despawned;
    t.added += s.added;
    t.overwritten += s.overwritten;
    t.removed += s.removed;
    t.ignored += s.ignored;
}
