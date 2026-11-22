//! Host functions a sim module imports from `env` to emit structural
//! commands while a system runs:
//!
//! | Import | Effect |
//! |---|---|
//! | `ae_spawn(record: u32, len: u32)` | spawn an entity from a spawn record |
//! | `ae_despawn(entity: u64)` | despawn |
//! | `ae_add(entity: u64, component: u64, data: u32, len: u32)` | add or overwrite a component |
//! | `ae_remove(entity: u64, component: u64)` | remove a component |
//! | `ae_emit(stream: u64, data: u32, len: u32)` | emit an event |
//! | `ae_get(entity: u64, component: u64, out: u32, len: u32) -> u32` | copy a component of another entity (1), or 0 when absent |
//! | `ae_spatial_radius(x: f32, y: f32, r: f32, out: u32, cap: u32) -> u32` | entities near a point, from the host's spatial index |
//! | `ae_lookup(scope: u64, name: u64) -> u32` | a name host plugins published (`HostPlugin::names`), `u32::MAX` when missing |
//! | `ae_query(op: u64, input: u32, input_len: u32, out: u32, out_len: u32) -> u32` | a pure function a host plugin answers (`HostPlugin::query`): bytes written, `u32::MAX` for an unknown op |
//! | `wasi_snapshot_preview1::sched_yield()` | yield the thread (wasi-libc's malloc lock waits with it) |
//!
//! Spawn record (little-endian): `u32 count, u32 reserved`, then `count`
//! entries of `u64 component, u32 len, u32 reserved, len bytes padded to 8`.
//! Every command is recorded under the current work range's key and takes
//! effect when the system's commands are merged.
//!
//! With an [`IdMap`] (modules built with the SDK), component and stream ids
//! are the schema's stable ids; the host maps them to ECS ids and checks
//! every value's size. Without one, component ids are ECS ids, passed through.

use crate::command::{Command, CommandBuffer};
use crate::host::{HostPlugin, SpatialIndex};
use ae_ecs::flecs_backend as fb;
use ae_ecs::{ComponentId, EntityId};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::sync::Arc;
use wasmtime::{Caller, Linker, SharedMemory};

pub const MAX_SPAWN_COMPONENTS: usize = 32;
pub const MAX_COMPONENT_BYTES: usize = 4096;

/// Host plugins by the query ops they answer (`ae_query`).
#[derive(Clone, Default)]
pub struct Queries(pub BTreeMap<u64, Arc<dyn HostPlugin>>);

impl std::fmt::Debug for Queries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.0.keys().map(|k| format!("{k:#x}"))).finish()
    }
}

/// Query input and output limit, bytes.
pub const MAX_QUERY_BYTES: usize = 4096;

/// Stable schema ids -> ECS component ids and value sizes.
#[derive(Clone, Debug, Default)]
pub struct IdMap {
    pub components: BTreeMap<u64, (ComponentId, usize)>,
    /// Event stream id -> event size.
    pub streams: BTreeMap<u64, usize>,
    /// Streams each system (by schedule position) declared in `emits`.
    pub emits: Vec<Vec<u64>>,
    /// `(scope, name)` -> value, from the host plugins (`ae_lookup`).
    pub names: BTreeMap<(u64, u64), u32>,
    pub queries: Queries,
}

impl IdMap {
    pub(crate) fn component(&self, id: u64, len: Option<usize>) -> Result<ComponentId> {
        let &(component, size) = self
            .components
            .get(&id)
            .with_context(|| format!("unknown component id {id:#x}"))?;
        if let Some(len) = len {
            ensure!(len == size, "component {id:#x} is {size} bytes, got {len}");
        }
        Ok(component)
    }
}

/// A world pointer carried into the workers: read only between the sync
/// points, by workers of one batch (see `WorkerCtx::world`).
#[derive(Clone, Copy, Debug)]
pub struct WorldPtr(pub fb::World);

// SAFETY: the pointer is valid for the batch's whole run, and every use is a
// read (`ecs_get_id`) between sync points; no worker mutates the world.
unsafe impl Send for WorldPtr {}
unsafe impl Sync for WorldPtr {}

impl WorldPtr {
    fn get(self) -> fb::World {
        self.0
    }
}

/// Store data of one sim worker instance.
pub struct WorkerCtx {
    pub memory: SharedMemory,
    pub commands: CommandBuffer,
    pub ids: Option<Arc<IdMap>>,
    /// The spatial index of the current batch, if a host system built one.
    pub spatial: Option<Arc<SpatialIndex>>,
    /// The world of the current batch (read-only: structural changes wait
    /// for the layer's sync point), for `ae_get`.
    pub world: Option<WorldPtr>,
    /// The running system's `get` list (stable component ids).
    pub gets: [u64; ae_abi::MAX_GETS],
    pub get_count: u32,
    hits: Vec<u32>,
}

impl WorkerCtx {
    pub fn new(memory: SharedMemory) -> Self {
        Self {
            memory,
            commands: CommandBuffer::default(),
            ids: None,
            spatial: None,
            world: None,
            gets: [0; ae_abi::MAX_GETS],
            get_count: 0,
            hits: Vec::new(),
        }
    }

    pub fn with_ids(memory: SharedMemory, ids: Arc<IdMap>) -> Self {
        Self {
            ids: Some(ids),
            ..Self::new(memory)
        }
    }

    fn component(&self, id: u64, len: Option<usize>) -> Result<ComponentId> {
        match &self.ids {
            Some(map) => map.component(id, len),
            None => Ok(ComponentId(id)),
        }
    }
}

/// Copy `len` bytes at guest offset `ptr`, bounds-checked.
pub fn read_guest(memory: &SharedMemory, ptr: u32, len: usize) -> Result<Vec<u8>> {
    let end = ptr as usize + len;
    ensure!(end <= memory.data_size(), "guest range {ptr:#x}+{len} is out of bounds");
    let mut out = vec![0u8; len];
    // Shared memory may be written by other instances concurrently; a system
    // only reads its own record, which nothing else writes.
    unsafe {
        std::ptr::copy_nonoverlapping(
            (memory.data().as_ptr() as *const u8).add(ptr as usize),
            out.as_mut_ptr(),
            len,
        )
    };
    Ok(out)
}

pub fn parse_spawn(record: &[u8]) -> Result<Vec<(ComponentId, Vec<u8>)>> {
    let word = |at: usize| -> Result<u32> {
        let bytes = record
            .get(at..at + 4)
            .ok_or_else(|| anyhow::anyhow!("spawn record truncated"))?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    };
    let count = word(0)? as usize;
    ensure!(
        count > 0 && count <= MAX_SPAWN_COMPONENTS,
        "spawn record has {count} components (1..={MAX_SPAWN_COMPONENTS})"
    );
    let mut at = 8;
    let mut components = Vec::with_capacity(count);
    for _ in 0..count {
        let component = word(at)? as u64 | (word(at + 4)? as u64) << 32;
        let len = word(at + 8)? as usize;
        ensure!(len <= MAX_COMPONENT_BYTES, "spawn component of {len} bytes");
        let data = record
            .get(at + 16..at + 16 + len)
            .ok_or_else(|| anyhow::anyhow!("spawn record truncated"))?;
        components.push((ComponentId(component), data.to_vec()));
        at += 16 + len.next_multiple_of(8);
    }
    ensure!(
        at == record.len(),
        "spawn record has {} trailing bytes",
        record.len().saturating_sub(at)
    );
    components.sort_by_key(|(c, _)| *c);
    if components.windows(2).any(|w| w[0].0 == w[1].0) {
        bail!("spawn record lists a component twice");
    }
    Ok(components)
}

fn trap(error: anyhow::Error) -> wasmtime::Error {
    wasmtime::Error::msg(format!("{error:#}"))
}

static YIELDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Times any guest instance in this process waited on a lock with
/// `sched_yield` (wasi-libc's malloc lock under contention).
pub fn sched_yields() -> u64 {
    YIELDS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Define the command imports.
pub fn link(linker: &mut Linker<WorkerCtx>) -> Result<()> {
    linker.func_wrap(
        "env",
        "ae_spawn",
        |mut caller: Caller<'_, WorkerCtx>, ptr: u32, len: u32| -> wasmtime::Result<()> {
            let mut run = || -> Result<()> {
                ensure!(
                    len as usize <= 16 + MAX_SPAWN_COMPONENTS * (16 + MAX_COMPONENT_BYTES),
                    "spawn record too large"
                );
                let record = read_guest(&caller.data().memory, ptr, len as usize)?;
                let mut components = parse_spawn(&record)?;
                if caller.data().ids.is_some() {
                    for (component, data) in &mut components {
                        *component = caller.data().component(component.0, Some(data.len()))?;
                    }
                    components.sort_by_key(|(c, _)| *c);
                }
                caller.data_mut().commands.push(Command::Spawn(components))
            };
            run().map_err(trap)
        },
    )?;
    linker.func_wrap(
        "env",
        "ae_despawn",
        |mut caller: Caller<'_, WorkerCtx>, entity: u64| -> wasmtime::Result<()> {
            caller
                .data_mut()
                .commands
                .push(Command::Despawn(EntityId(entity)))
                .map_err(trap)
        },
    )?;
    linker.func_wrap(
        "env",
        "ae_add",
        |mut caller: Caller<'_, WorkerCtx>, entity: u64, component: u64, ptr: u32, len: u32| -> wasmtime::Result<()> {
            let mut run = || -> Result<()> {
                ensure!(len as usize <= MAX_COMPONENT_BYTES, "component of {len} bytes");
                let data = read_guest(&caller.data().memory, ptr, len as usize)?;
                let component = caller.data().component(component, Some(data.len()))?;
                caller
                    .data_mut()
                    .commands
                    .push(Command::Add(EntityId(entity), component, data))
            };
            run().map_err(trap)
        },
    )?;
    linker.func_wrap(
        "env",
        "ae_remove",
        |mut caller: Caller<'_, WorkerCtx>, entity: u64, component: u64| -> wasmtime::Result<()> {
            let mut run = || -> Result<()> {
                let component = caller.data().component(component, None)?;
                caller
                    .data_mut()
                    .commands
                    .push(Command::Remove(EntityId(entity), component))
            };
            run().map_err(trap)
        },
    )?;
    linker.func_wrap(
        "env",
        "ae_emit",
        |mut caller: Caller<'_, WorkerCtx>, stream: u64, ptr: u32, len: u32| -> wasmtime::Result<()> {
            let mut run = || -> Result<()> {
                let size = caller.data().ids.as_ref().and_then(|m| m.streams.get(&stream).copied());
                let size = size.with_context(|| format!("unknown event stream {stream:#x}"))?;
                ensure!(
                    len as usize == size,
                    "event of stream {stream:#x} is {size} bytes, got {len}"
                );
                let system = caller
                    .data()
                    .commands
                    .current_system()
                    .context("event emitted outside a system call")?;
                let declared = caller
                    .data()
                    .ids
                    .as_ref()
                    .is_some_and(|m| m.emits.get(system as usize).is_some_and(|e| e.contains(&stream)));
                ensure!(
                    declared,
                    "system at position {system} emits stream {stream:#x} without declaring it in `emits`"
                );
                let data = read_guest(&caller.data().memory, ptr, len as usize)?;
                caller.data_mut().commands.push_event(stream, data)
            };
            run().map_err(trap)
        },
    )?;
    linker.func_wrap(
        "env",
        "ae_spatial_radius",
        |mut caller: Caller<'_, WorkerCtx>, x: f32, y: f32, r: f32, out: u32, cap: u32| -> wasmtime::Result<u32> {
            let mut run = || -> Result<u32> {
                let ctx = caller.data_mut();
                let index = ctx
                    .spatial
                    .clone()
                    .context("no spatial index: schedule `host(spatial)` before querying it")?;
                ensure!(
                    r.is_finite() && r >= 0.0 && x.is_finite() && y.is_finite(),
                    "bad spatial query ({x}, {y}, r {r})"
                );
                index.grid.radius_one(ae_spatial::Circle { x, y, r }, &mut ctx.hits);
                let n = ctx.hits.len().min(cap as usize);
                let end = out as usize + n * 8;
                ensure!(
                    end <= ctx.memory.data_size(),
                    "spatial query output {out:#x}+{} is out of bounds",
                    n * 8
                );
                // The output buffer is the calling system's own memory.
                let dst = unsafe { (ctx.memory.data().as_ptr() as *mut u8).add(out as usize) };
                for (k, &item) in ctx.hits[..n].iter().enumerate() {
                    unsafe {
                        dst.add(k * 8)
                            .cast::<u64>()
                            .write_unaligned(index.entities[item as usize])
                    };
                }
                Ok(ctx.hits.len() as u32)
            };
            run().map_err(trap)
        },
    )?;
    linker.func_wrap(
        "env",
        "ae_lookup",
        |caller: Caller<'_, WorkerCtx>, scope: u64, name: u64| -> u32 {
            caller
                .data()
                .ids
                .as_ref()
                .and_then(|m| m.names.get(&(scope, name)).copied())
                .unwrap_or(u32::MAX)
        },
    )?;
    linker.func_wrap(
        "env",
        "ae_get",
        |caller: Caller<'_, WorkerCtx>, entity: u64, component: u64, out: u32, len: u32| -> wasmtime::Result<u32> {
            let run = || -> Result<u32> {
                let ctx = caller.data();
                let id = ctx.component(component, Some(len as usize))?;
                ensure!(
                    ctx.gets[..ctx.get_count as usize].contains(&component),
                    "system gets component {component:#x} without declaring it in `get`"
                );
                let world = ctx.world.map(WorldPtr::get).context("`ae_get` outside a system call")?;
                let Some(src) = (unsafe { fb::get_component(world, EntityId(entity), id) }) else {
                    return Ok(0);
                };
                let end = out as usize + len as usize;
                ensure!(
                    end <= ctx.memory.data_size(),
                    "get output {out:#x}+{len} is out of bounds"
                );
                // The output buffer is the calling system's own memory; the
                // source is its column (guest memory), which nothing writes
                // structurally until the layer's sync point.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src,
                        (ctx.memory.data().as_ptr() as *mut u8).add(out as usize),
                        len as usize,
                    )
                };
                Ok(1)
            };
            run().map_err(trap)
        },
    )?;
    linker.func_wrap(
        "env",
        "ae_query",
        |caller: Caller<'_, WorkerCtx>,
         op: u64,
         input: u32,
         input_len: u32,
         out: u32,
         out_len: u32|
         -> wasmtime::Result<u32> {
            let run = || -> Result<u32> {
                let ctx = caller.data();
                let Some(plugin) = ctx.ids.as_ref().and_then(|m| m.queries.0.get(&op)) else {
                    return Ok(ae_abi::query::MISSING);
                };
                ensure!(
                    input_len as usize <= MAX_QUERY_BYTES && out_len as usize <= MAX_QUERY_BYTES,
                    "query of {input_len} -> {out_len} bytes"
                );
                let data = read_guest(&ctx.memory, input, input_len as usize)?;
                let mut result = vec![0u8; out_len as usize];
                let n = plugin.query(op, &data, &mut result)?;
                ensure!(n <= result.len(), "query {op:#x} wrote past its output");
                let end = out as usize + n;
                ensure!(
                    end <= ctx.memory.data_size(),
                    "query output {out:#x}+{n} is out of bounds"
                );
                // The output buffer is the calling system's own memory.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        result.as_ptr(),
                        (ctx.memory.data().as_ptr() as *mut u8).add(out as usize),
                        n,
                    )
                };
                Ok(n as u32)
            };
            run().map_err(trap)
        },
    )?;
    // wasi-libc's malloc lock calls this while it waits for another
    // instance to release the heap. Yielding has no effect on results, only
    // on how the wait is spent. (Every other WASI import traps.)
    linker.func_wrap("wasi_snapshot_preview1", "sched_yield", || -> i32 {
        YIELDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::thread::yield_now();
        0
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(entries: &[(u64, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for (component, data) in entries {
            out.extend_from_slice(&component.to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(data);
            out.resize(out.len().next_multiple_of(8), 0);
        }
        out
    }

    #[test]
    fn spawn_records_parse_and_sort() {
        let parsed = parse_spawn(&record(&[(9, &[1, 2, 3]), (4, &[7; 8])])).unwrap();
        assert_eq!(
            parsed,
            vec![(ComponentId(4), vec![7; 8]), (ComponentId(9), vec![1, 2, 3])]
        );
    }

    #[test]
    fn malformed_spawn_records_are_rejected() {
        let good = record(&[(9, &[1, 2, 3])]);
        assert!(parse_spawn(&good[..good.len() - 1]).is_err(), "truncated");
        let mut long = good.clone();
        long.extend_from_slice(&[0; 8]);
        assert!(parse_spawn(&long).is_err(), "trailing bytes");
        assert!(
            parse_spawn(&record(&[(9, &[1]), (9, &[2])])).is_err(),
            "duplicate component"
        );
        assert!(parse_spawn(&record(&[])).is_err(), "no components");
    }
}
