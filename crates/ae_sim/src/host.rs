//! Host systems: native code that runs in the sim schedule beside the
//! module's systems.
//!
//! A module places them with `schedule!(.., host(nav), ..)`; a
//! [`HostPlugin`] given to the runtime provides them. Each declares what it
//! reads and writes exactly as a module system does, so the scheduler orders
//! it by the same conflict rules, and its commands and events are keyed by
//! its schedule position and merged with everyone else's. A host system runs
//! on the coordinating thread with the whole query at once, while the crew's
//! helpers start on the layer's module systems (which cannot conflict with
//! it); the thread then joins them. It may use its own threads inside.
//! `RuntimeConfig::overlap_hosts = false` runs it after them instead; results
//! are the same either way.
//!
//! Host state that is not in components (the nav service's bookkeeping)
//! enters the world digest through [`HostSystem::digest`].

use crate::command::{Command, CommandBuffer};
use crate::events::EventStore;
use crate::imports::IdMap;
use crate::schedule::Cadence;
use crate::schema::{ComponentDecl, Schema, component_record};
use crate::work::WorkRange;
use ae_ecs::{Access, ComponentId, MAX_QUERY_TERMS};
use ae_wasm::GuestMemory;
use anyhow::{Context, Result, ensure};
use std::sync::Arc;

/// What a host system reads, writes and when it runs.
#[derive(Clone, Debug)]
pub struct HostDecl {
    pub name: String,
    /// Query terms (stable component ids). The system sees every matching
    /// row, in table order, each time it runs.
    pub query: Vec<(u64, Access)>,
    /// Components the system only creates (spawn or add commands), for
    /// ordering: they count as writes.
    pub creates: Vec<u64>,
    pub emits: Vec<u64>,
    pub reads_events: Vec<u64>,
    /// Systems (module or host) that must run before this one.
    pub after: Vec<String>,
    /// `EveryTick`, `Every` or `Once` (a host system sees whole tables, so
    /// `Staggered` is rejected).
    pub cadence: Cadence,
}

impl HostDecl {
    pub fn new(name: &str) -> Self {
        Self { name: name.into(), query: Vec::new(), creates: Vec::new(), emits: Vec::new(), reads_events: Vec::new(), after: Vec::new(), cadence: Cadence::EveryTick }
    }
}

/// Per-world inputs to [`HostPlugin::instantiate`].
#[derive(Clone, Copy, Debug)]
pub struct HostInit {
    /// The world's game parameter.
    pub param: u64,
    /// Threads the world's ticks will use; host systems with their own
    /// thread pools size them by this. Results must not depend on it.
    pub workers: usize,
    /// The sim's ticks a second (`RuntimeConfig::tick_hz`).
    pub tick_hz: u32,
}

/// Provides host systems.
pub trait HostPlugin: Send + Sync {
    /// Every system this plugin can provide. Only those the module
    /// schedules run.
    fn systems(&self) -> Vec<HostDecl>;
    /// Check a scheduled system against the module's schema (layouts of the
    /// components it touches; see [`expect_component`]).
    fn check(&self, name: &str, schema: &Schema) -> Result<()>;
    /// Fresh state for one world: one system per name, in `names` order
    /// (the subset of [`Self::systems`] the module schedules). `schema` is
    /// the module's, with the data it declares for services
    /// ([`Schema::data`]).
    fn instantiate(&self, init: &HostInit, schema: &Schema, names: &[&str]) -> Result<Vec<Box<dyn HostSystem>>>;
    /// Names module systems can look up with `ae_lookup(scope, name)`:
    /// `(scope, name, value)` (see `ae_abi::lookup`). Content, so the same
    /// on every peer. Every plugin's names are published, whether the
    /// module schedules its systems or not.
    fn names(&self) -> Vec<(u64, u64, u32)> {
        Vec::new()
    }
    /// Ops module systems can call with `ae_query` (`ae_abi::query`): ids
    /// this plugin answers in [`Self::query`].
    fn queries(&self) -> Vec<u64> {
        Vec::new()
    }
    /// Answer op `op` on `input` into `out`; returns the bytes written. Must
    /// be a pure function of the content and the input (it runs on any
    /// worker, mid-system), and must not panic on bad input: return an
    /// error (the module traps).
    fn query(&self, op: u64, _input: &[u8], _out: &mut [u8]) -> Result<usize> {
        anyhow::bail!("query {op:#x} is not answered here")
    }
}

pub trait HostSystem: Send {
    fn run(&mut self, ctx: &mut HostCtx<'_>) -> Result<()>;
    /// Hash of state kept outside components, if any; part of the world
    /// digest.
    fn digest(&self) -> Option<u128> {
        None
    }
}

/// Positions of bodies, rebuilt by a host system, queried by module systems
/// through `ae_spatial_radius`. Item `i` of the grid is `entities[i]`.
pub struct SpatialIndex {
    pub grid: ae_spatial::Grid,
    pub entities: Vec<u64>,
}

/// One table's rows of a host system's query.
pub struct Chunk<'a> {
    pub entities: &'a [u64],
    columns: [*mut u8; MAX_QUERY_TERMS],
    sizes: &'a [usize],
}

impl Chunk<'_> {
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Column of query term `term` as `T` (a `#[repr(C)]` mirror of the
    /// component, e.g. an `ae_sdk` type compiled natively).
    ///
    /// # Safety
    /// `T` must have the component's layout, and the caller must not hold
    /// two mutable views of one column at once. Writing a term declared
    /// `Read` breaks the schedule's guarantees.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn column<T: Copy>(&self, term: usize) -> &mut [T] {
        assert_eq!(size_of::<T>(), self.sizes[term], "column type does not match the component size");
        assert!((self.columns[term] as usize).is_multiple_of(align_of::<T>()), "misaligned column");
        unsafe { std::slice::from_raw_parts_mut(self.columns[term].cast::<T>(), self.entities.len()) }
    }
}

/// What a host system can see and do during one run.
pub struct HostCtx<'a> {
    pub tick: u64,
    pub param: u64,
    pub(crate) ranges: &'a [WorkRange],
    pub(crate) sizes: &'a [usize],
    pub(crate) memory: GuestMemory,
    pub(crate) events: &'a mut EventStore,
    pub(crate) reader: usize,
    pub(crate) reads: &'a [u64],
    pub(crate) emits: &'a [u64],
    pub(crate) commands: &'a mut CommandBuffer,
    pub(crate) ids: &'a IdMap,
    pub(crate) spatial: &'a mut Option<Arc<SpatialIndex>>,
}

impl HostCtx<'_> {
    /// The query's rows, table by table.
    pub fn chunks(&self) -> impl Iterator<Item = Chunk<'_>> + '_ {
        self.ranges.iter().map(|r| {
            let mut columns = [std::ptr::null_mut(); MAX_QUERY_TERMS];
            for (t, c) in columns.iter_mut().enumerate().take(self.sizes.len()) {
                *c = self.memory.ptr(r.columns[t]);
            }
            let entities = if r.count == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(r.entities, r.count as usize) } };
            Chunk { entities, columns, sizes: self.sizes }
        })
    }

    /// Rows the query matches.
    pub fn rows(&self) -> usize {
        self.ranges.iter().map(|r| r.count as usize).sum()
    }

    /// Every event of `stream` emitted since this system last ran, in key
    /// order. The stream must be in the system's `reads_events`.
    pub fn events<E: Copy>(&mut self, stream: u64) -> Result<Vec<E>> {
        ensure!(self.reads.contains(&stream), "host system reads stream {stream:#x} without declaring it");
        let size = self.ids.streams.get(&stream).copied().context("unknown event stream")?;
        ensure!(size == size_of::<E>(), "event type of {} bytes for a stream of {size}", size_of::<E>());
        let (bytes, count) = self.events.unseen(stream, self.reader);
        let out = (0..count as usize).map(|k| unsafe { bytes.as_ptr().add(k * size).cast::<E>().read_unaligned() }).collect();
        self.events.mark_seen(stream, self.reader);
        Ok(out)
    }

    /// Spawn an entity with these components (stable id, value bytes).
    pub fn spawn(&mut self, components: &[(u64, &[u8])]) -> Result<()> {
        let mut c: Vec<(ComponentId, Vec<u8>)> = components.iter().map(|&(id, bytes)| Ok((self.ids.component(id, Some(bytes.len()))?, bytes.to_vec()))).collect::<Result<_>>()?;
        c.sort_by_key(|(id, _)| *id);
        ensure!(c.windows(2).all(|w| w[0].0 != w[1].0), "spawn lists a component twice");
        self.commands.push(Command::Spawn(c))
    }

    /// Emit an event (the stream must be in the system's `emits`).
    pub fn emit(&mut self, stream: u64, bytes: &[u8]) -> Result<()> {
        ensure!(self.emits.contains(&stream), "host system emits stream {stream:#x} without declaring it");
        ensure!(self.ids.streams.get(&stream) == Some(&bytes.len()), "event of {} bytes for stream {stream:#x}", bytes.len());
        self.commands.push_event(stream, bytes.to_vec())
    }

    /// Replace the spatial index module systems query from now on.
    pub fn publish_spatial(&mut self, index: Arc<SpatialIndex>) {
        *self.spatial = Some(index);
    }

    /// The current spatial index; take it back to reuse its allocations
    /// (`Arc::try_unwrap` succeeds when no batch holds it).
    pub fn take_spatial(&mut self) -> Option<Arc<SpatialIndex>> {
        self.spatial.take()
    }
}

/// The bytes of a plain value (for [`HostCtx::spawn`] and [`HostCtx::emit`]).
pub fn bytes_of<T: Copy>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
}

/// Check that the module declares component (or event) `id` exactly as the
/// engine's own record says (`ae_sdk::Component::SCHEMA` compiled natively):
/// same name, size, alignment and fields.
pub fn expect_component(schema: &Schema, record: &[u8]) -> Result<ComponentDecl> {
    let ours = component_record(record)?;
    let theirs = schema.component(ours.id).with_context(|| format!("the module does not declare `{}` (id {:#x})", ours.name, ours.id))?;
    ensure!(*theirs == ours, "the module's `{}` differs from the engine's: {theirs:?} vs {ours:?}", ours.name);
    Ok(ours)
}
