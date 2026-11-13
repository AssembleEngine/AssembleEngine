//! Host functions a sim module imports from `env` to emit structural
//! commands while a system runs:
//!
//! | Import | Effect |
//! |---|---|
//! | `ae_spawn(record: u32, len: u32)` | spawn an entity from a spawn record |
//! | `ae_despawn(entity: u64)` | despawn |
//! | `ae_add(entity: u64, component: u64, data: u32, len: u32)` | add or overwrite a component |
//! | `ae_remove(entity: u64, component: u64)` | remove a component |
//!
//! Spawn record (little-endian): `u32 count, u32 reserved`, then `count`
//! entries of `u64 component, u32 len, u32 reserved, len bytes padded to 8`.
//! Every command is recorded under the current work range's key and takes
//! effect when the system's commands are merged.

use crate::command::{Command, CommandBuffer};
use ae_ecs::{ComponentId, EntityId};
use anyhow::{Result, bail, ensure};
use wasmtime::{Caller, Linker, SharedMemory};

pub const MAX_SPAWN_COMPONENTS: usize = 32;
pub const MAX_COMPONENT_BYTES: usize = 4096;

/// Store data of one sim worker instance.
pub struct WorkerCtx {
    pub memory: SharedMemory,
    pub commands: CommandBuffer,
}

impl WorkerCtx {
    pub fn new(memory: SharedMemory) -> Self {
        Self { memory, commands: CommandBuffer::default() }
    }
}

/// Copy `len` bytes at guest offset `ptr`, bounds-checked.
pub fn read_guest(memory: &SharedMemory, ptr: u32, len: usize) -> Result<Vec<u8>> {
    let end = ptr as usize + len;
    ensure!(end <= memory.data_size(), "guest range {ptr:#x}+{len} is out of bounds");
    let mut out = vec![0u8; len];
    // Shared memory may be written by other instances concurrently; a system
    // only reads its own record, which nothing else writes.
    unsafe { std::ptr::copy_nonoverlapping((memory.data().as_ptr() as *const u8).add(ptr as usize), out.as_mut_ptr(), len) };
    Ok(out)
}

pub fn parse_spawn(record: &[u8]) -> Result<Vec<(ComponentId, Vec<u8>)>> {
    let word = |at: usize| -> Result<u32> {
        let bytes = record.get(at..at + 4).ok_or_else(|| anyhow::anyhow!("spawn record truncated"))?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    };
    let count = word(0)? as usize;
    ensure!(count > 0 && count <= MAX_SPAWN_COMPONENTS, "spawn record has {count} components (1..={MAX_SPAWN_COMPONENTS})");
    let mut at = 8;
    let mut components = Vec::with_capacity(count);
    for _ in 0..count {
        let component = word(at)? as u64 | (word(at + 4)? as u64) << 32;
        let len = word(at + 8)? as usize;
        ensure!(len <= MAX_COMPONENT_BYTES, "spawn component of {len} bytes");
        let data = record.get(at + 16..at + 16 + len).ok_or_else(|| anyhow::anyhow!("spawn record truncated"))?;
        components.push((ComponentId(component), data.to_vec()));
        at += 16 + len.next_multiple_of(8);
    }
    ensure!(at == record.len(), "spawn record has {} trailing bytes", record.len().saturating_sub(at));
    components.sort_by_key(|(c, _)| *c);
    if components.windows(2).any(|w| w[0].0 == w[1].0) {
        bail!("spawn record lists a component twice");
    }
    Ok(components)
}

fn trap(error: anyhow::Error) -> wasmtime::Error {
    wasmtime::Error::msg(format!("{error:#}"))
}

/// Define the command imports.
pub fn link(linker: &mut Linker<WorkerCtx>) -> Result<()> {
    linker.func_wrap("env", "ae_spawn", |mut caller: Caller<'_, WorkerCtx>, ptr: u32, len: u32| -> wasmtime::Result<()> {
        let mut run = || -> Result<()> {
            ensure!(len as usize <= 16 + MAX_SPAWN_COMPONENTS * (16 + MAX_COMPONENT_BYTES), "spawn record too large");
            let record = read_guest(&caller.data().memory, ptr, len as usize)?;
            let components = parse_spawn(&record)?;
            caller.data_mut().commands.push(Command::Spawn(components))
        };
        run().map_err(trap)
    })?;
    linker.func_wrap("env", "ae_despawn", |mut caller: Caller<'_, WorkerCtx>, entity: u64| -> wasmtime::Result<()> {
        caller.data_mut().commands.push(Command::Despawn(EntityId(entity))).map_err(trap)
    })?;
    linker.func_wrap(
        "env",
        "ae_add",
        |mut caller: Caller<'_, WorkerCtx>, entity: u64, component: u64, ptr: u32, len: u32| -> wasmtime::Result<()> {
            let mut run = || -> Result<()> {
                ensure!(len as usize <= MAX_COMPONENT_BYTES, "component of {len} bytes");
                let data = read_guest(&caller.data().memory, ptr, len as usize)?;
                caller.data_mut().commands.push(Command::Add(EntityId(entity), ComponentId(component), data))
            };
            run().map_err(trap)
        },
    )?;
    linker.func_wrap("env", "ae_remove", |mut caller: Caller<'_, WorkerCtx>, entity: u64, component: u64| -> wasmtime::Result<()> {
        caller.data_mut().commands.push(Command::Remove(EntityId(entity), ComponentId(component))).map_err(trap)
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
        assert_eq!(parsed, vec![(ComponentId(4), vec![7; 8]), (ComponentId(9), vec![1, 2, 3])]);
    }

    #[test]
    fn malformed_spawn_records_are_rejected() {
        let good = record(&[(9, &[1, 2, 3])]);
        assert!(parse_spawn(&good[..good.len() - 1]).is_err(), "truncated");
        let mut long = good.clone();
        long.extend_from_slice(&[0; 8]);
        assert!(parse_spawn(&long).is_err(), "trailing bytes");
        assert!(parse_spawn(&record(&[(9, &[1]), (9, &[2])])).is_err(), "duplicate component");
        assert!(parse_spawn(&record(&[])).is_err(), "no components");
    }
}
