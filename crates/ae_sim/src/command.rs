//! Deferred structural commands.

use ae_ecs::flecs_backend as fb;
use ae_ecs::{ComponentId, EntityId};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;

/// Where a command came from. Commands are applied in key order, which is
/// independent of which worker ran which range and when.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommandKey {
    /// Position of the system in the schedule.
    pub system: u32,
    /// Index of the work range within the system's ranges (table order, then
    /// row order within the table).
    pub range: u32,
    /// Order of the command within that range's call.
    pub seq: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// New entity with these components (sorted by id, no duplicates).
    Spawn(Vec<(ComponentId, Vec<u8>)>),
    Despawn(EntityId),
    /// Add the component, or overwrite it if present.
    Add(EntityId, ComponentId, Vec<u8>),
    Remove(EntityId, ComponentId),
}

/// An event emitted by a system: stream id and the event's bytes.
pub type Emitted = (CommandKey, u64, Vec<u8>);

/// One worker's commands (and emitted events) for the current layer.
#[derive(Default)]
pub struct CommandBuffer {
    current: Option<(u32, u32)>,
    seq: u32,
    event_seq: u32,
    entries: Vec<(CommandKey, Command)>,
    events: Vec<Emitted>,
}

impl CommandBuffer {
    /// Start recording for a work range. Must precede every guest call.
    pub fn begin(&mut self, system: u32, range: u32) {
        self.current = Some((system, range));
        self.seq = 0;
        self.event_seq = 0;
    }

    /// Record an event under the current range's key. Events have their own
    /// sequence, so emitting one never shifts command keys.
    pub fn push_event(&mut self, stream: u64, bytes: Vec<u8>) -> Result<()> {
        let (system, range) = self
            .current
            .ok_or_else(|| anyhow::anyhow!("event emitted outside a system call"))?;
        self.events.push((
            CommandKey {
                system,
                range,
                seq: self.event_seq,
            },
            stream,
            bytes,
        ));
        self.event_seq += 1;
        Ok(())
    }

    pub fn take_events(&mut self) -> Vec<Emitted> {
        std::mem::take(&mut self.events)
    }

    pub fn push(&mut self, command: Command) -> Result<()> {
        let (system, range) = self
            .current
            .ok_or_else(|| anyhow::anyhow!("command emitted outside a system call"))?;
        self.entries.push((
            CommandKey {
                system,
                range,
                seq: self.seq,
            },
            command,
        ));
        self.seq += 1;
        Ok(())
    }

    /// Schedule position of the system being recorded.
    pub fn current_system(&self) -> Option<u32> {
        self.current.map(|(system, _)| system)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn take(&mut self) -> Vec<(CommandKey, Command)> {
        self.current = None;
        std::mem::take(&mut self.entries)
    }
}

/// Concatenate per-worker buffers and sort by key. Keys are unique by
/// construction; a duplicate means two workers claimed the same range.
pub fn merge(buffers: impl IntoIterator<Item = Vec<(CommandKey, Command)>>) -> Result<Vec<(CommandKey, Command)>> {
    let mut all: Vec<(CommandKey, Command)> = buffers.into_iter().flatten().collect();
    all.sort_unstable_by_key(|(key, _)| *key);
    ensure!(
        all.windows(2).all(|w| w[0].0 != w[1].0),
        "duplicate command key: a work range ran twice"
    );
    Ok(all)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ApplyStats {
    pub spawned: u64,
    pub despawned: u64,
    pub added: u64,
    pub overwritten: u64,
    pub removed: u64,
    /// Commands on entities that were already gone, or removes of absent
    /// components: deterministic no-ops.
    pub ignored: u64,
}

/// Applies merged commands to a world. Caches spawn tables.
#[derive(Default)]
pub struct Applier {
    tables: BTreeMap<Vec<ComponentId>, fb::Table>,
}

impl Applier {
    /// Apply in order. Every structural change happens here, on one thread.
    ///
    /// # Safety
    /// `world` must be a live world and no column pointer from it may be in
    /// use while this runs.
    pub unsafe fn apply(&mut self, world: fb::World, commands: &[(CommandKey, Command)]) -> Result<ApplyStats> {
        let mut stats = ApplyStats::default();
        for (_, command) in commands {
            match command {
                Command::Spawn(components) => {
                    let ids: Vec<ComponentId> = components.iter().map(|(c, _)| *c).collect();
                    let table = match self.tables.get(&ids) {
                        Some(table) => *table,
                        None => {
                            let table = unsafe { fb::find_table(world, &ids)? };
                            self.tables.insert(ids, table);
                            table
                        }
                    };
                    let entity = unsafe { fb::spawn_in_table(world, table) };
                    for (component, bytes) in components {
                        unsafe { fb::set_component(world, entity, *component, bytes)? };
                    }
                    stats.spawned += 1;
                }
                Command::Despawn(entity) => {
                    if unsafe { fb::delete_entity(world, *entity) } {
                        stats.despawned += 1;
                    } else {
                        stats.ignored += 1;
                    }
                }
                Command::Add(entity, component, bytes) => {
                    if !unsafe { fb::entity_alive(world, *entity) } {
                        stats.ignored += 1;
                        continue;
                    }
                    if unsafe { fb::has_component(world, *entity, *component) } {
                        stats.overwritten += 1;
                    } else {
                        stats.added += 1;
                    }
                    unsafe { fb::set_component(world, *entity, *component, bytes)? };
                }
                Command::Remove(entity, component) => {
                    if unsafe { fb::entity_alive(world, *entity) && fb::has_component(world, *entity, *component) } {
                        unsafe { fb::remove_component(world, *entity, *component) };
                        stats.removed += 1;
                    } else {
                        stats.ignored += 1;
                    }
                }
            }
        }
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_orders_by_key_regardless_of_buffer_order() {
        let mut a = CommandBuffer::default();
        let mut b = CommandBuffer::default();
        a.begin(0, 3);
        a.push(Command::Despawn(EntityId(30))).unwrap();
        a.push(Command::Despawn(EntityId(31))).unwrap();
        b.begin(0, 1);
        b.push(Command::Despawn(EntityId(10))).unwrap();
        a.begin(1, 0);
        a.push(Command::Despawn(EntityId(100))).unwrap();
        let order = |buffers: Vec<Vec<(CommandKey, Command)>>| -> Vec<Command> {
            merge(buffers).unwrap().into_iter().map(|(_, c)| c).collect()
        };
        let (ta, tb) = (a.take(), b.take());
        let expected: Vec<Command> = [10, 30, 31, 100]
            .iter()
            .map(|&e| Command::Despawn(EntityId(e)))
            .collect();
        assert_eq!(order(vec![ta.clone(), tb.clone()]), expected);
        assert_eq!(order(vec![tb, ta]), expected);
    }

    #[test]
    fn duplicate_keys_are_rejected() {
        let mut a = CommandBuffer::default();
        let mut b = CommandBuffer::default();
        a.begin(0, 0);
        b.begin(0, 0);
        a.push(Command::Despawn(EntityId(1))).unwrap();
        b.push(Command::Despawn(EntityId(2))).unwrap();
        assert!(merge(vec![a.take(), b.take()]).is_err());
    }

    /// Commands naming entities that do not exist (made up, despawned, or
    /// an old generation of a recycled slot) must be no-ops, never create
    /// an entity.
    #[test]
    fn commands_on_missing_entities_are_ignored() {
        unsafe {
            let world = fb::init().unwrap();
            let c = fb::register_component(world, "C", 4, 4).unwrap();
            let terms = [ae_ecs::Term {
                component: c,
                access: ae_ecs::Access::Read,
            }];
            let query = fb::create_query(world, &terms).unwrap();
            let count = || {
                let mut n = 0;
                fb::for_each_chunk(world, query, &terms, |chunk| {
                    n += chunk.count;
                    Ok(())
                })
                .unwrap();
                n
            };
            let mut applier = Applier::default();
            let key = |seq| CommandKey {
                system: 0,
                range: 0,
                seq,
            };
            let spawned = applier
                .apply(world, &[(key(0), Command::Spawn(vec![(c, vec![1, 0, 0, 0])]))])
                .unwrap();
            assert_eq!(spawned.spawned, 1);
            let live = {
                let mut id = 0;
                fb::for_each_chunk(world, query, &terms, |chunk| {
                    id = *chunk.entities;
                    Ok(())
                })
                .unwrap();
                EntityId(id)
            };
            // Despawn it, then spawn again so the slot is recycled under a new generation.
            applier.apply(world, &[(key(0), Command::Despawn(live))]).unwrap();
            applier
                .apply(world, &[(key(0), Command::Spawn(vec![(c, vec![2, 0, 0, 0])]))])
                .unwrap();
            assert_eq!(count(), 1);

            let missing = [
                EntityId(live.0),
                EntityId(123_456_789),
                EntityId(0xFFFF_FFFF),
                EntityId(live.0 | 1 << 40),
            ];
            for entity in missing {
                let commands = [
                    (key(0), Command::Add(entity, c, vec![9, 0, 0, 0])),
                    (key(1), Command::Remove(entity, c)),
                    (key(2), Command::Despawn(entity)),
                ];
                let stats = applier.apply(world, &commands).unwrap();
                assert_eq!(
                    stats,
                    ApplyStats {
                        ignored: 3,
                        ..ApplyStats::default()
                    },
                    "entity {:#x}",
                    entity.0
                );
                assert!(
                    !fb::entity_alive(world, entity),
                    "entity {:#x} came into existence",
                    entity.0
                );
            }
            assert_eq!(
                count(),
                1,
                "no entity may be created or deleted by commands on missing ids"
            );
            fb::fini_query(query);
            fb::fini(world);
        }
    }

    #[test]
    fn push_outside_a_call_is_an_error() {
        assert!(CommandBuffer::default().push(Command::Despawn(EntityId(1))).is_err());
    }
}
