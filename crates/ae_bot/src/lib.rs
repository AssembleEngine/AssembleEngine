//! Host-side bots: command generators that play an allyteam the way a
//! player does, through tick commands. A bot runs outside the sim, reads
//! only what its allyteam owns (and what every player knows, such as where
//! the bases are), and its commands go into the tick bundle, so replays
//! record them and lockstep peers need not run the bot to stay in sync.
//! Everything a bot decides follows from the world, so the same world gives
//! the same commands on every machine.
//!
//! [`SkirmishBot`] plays `games/skirmish`: it waits until `wave` of its
//! units stand idle (the factory's new units gather at the rally point)
//! and then sends all of them to attack-move the enemy base (from the
//! game's `Match`, which every player sees).

use ae_sim::input::Command;
use ae_sim::runtime::{Runtime, World};
use anyhow::{Context, Result};

/// The skirmish bot of allyteam `ally`.
#[derive(Clone, Debug)]
pub struct SkirmishBot {
    pub ally: u32,
    /// Idle units it waits for before it attacks.
    pub wave: usize,
    /// Ticks between decisions (each bot decides at its own phase).
    pub every: u64,
    stream: u64,
    team_ally: usize,
    queue_len: usize,
    queue_state: usize,
}

/// `games/skirmish`'s `command` kinds and `state::NEW`.
pub const MOVE: u32 = 1;
pub const ATTACK: u32 = 2;
pub const STOP: u32 = 3;
const NEW: u32 = 0;

/// A skirmish `UnitCommand { entity, kind, append, x, y }` from `player`
/// on `stream` (the game's `UnitCommand` id).
pub fn unit_command(player: u32, stream: u64, entity: u64, kind: u32, append: bool, x: f32, y: f32) -> Command {
    let mut bytes = Vec::with_capacity(24);
    bytes.extend_from_slice(&entity.to_le_bytes());
    for w in [kind, append as u32, x.to_bits(), y.to_bits()] {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    Command { player, stream, bytes }
}

fn field(rt: &Runtime, component: &str, name: &str) -> Result<usize> {
    let c = rt
        .schema
        .components
        .iter()
        .find(|c| c.name == component)
        .with_context(|| format!("no component `{component}`: not the skirmish game?"))?;
    Ok(c.fields
        .iter()
        .find(|f| f.name == name)
        .with_context(|| format!("no field `{component}.{name}`"))?
        .offset as usize)
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn f32_at(bytes: &[u8], at: usize) -> f32 {
    f32::from_bits(u32_at(bytes, at))
}

/// The skirmish game's bases, from its `Match` (every player sees it);
/// `None` before the game's first tick.
pub fn bases(rt: &Runtime, world: &World) -> Result<Option<[(f32, f32); 2]>> {
    let (bx, by) = (field(rt, "Match", "base_x")?, field(rt, "Match", "base_y")?);
    Ok(rt
        .rows(world, "Match")?
        .first()
        .map(|(_, m)| [(f32_at(m, bx), f32_at(m, by)), (f32_at(m, bx + 4), f32_at(m, by + 4))]))
}

impl SkirmishBot {
    pub fn new(rt: &Runtime, ally: u32, wave: usize) -> Result<Self> {
        let stream = rt
            .schema
            .components
            .iter()
            .find(|c| c.name == "UnitCommand")
            .context("no `UnitCommand` stream: not the skirmish game?")?
            .id;
        Ok(Self {
            ally,
            wave,
            every: 30,
            stream,
            team_ally: field(rt, "Team", "ally")?,
            queue_len: field(rt, "Queue", "len")?,
            queue_state: field(rt, "Queue", "state")?,
        })
    }

    /// The standard pair: allyteam 0 attacks in waves of 8, allyteam 1 in
    /// waves of 16 (`ae-headless --script bot`, the client's opponent, the
    /// golden match).
    pub fn pair(rt: &Runtime) -> Result<[Self; 2]> {
        Ok([Self::new(rt, 0, 8)?, Self::new(rt, 1, 16)?])
    }

    /// This tick's commands, appended to `out` (player = the allyteam).
    pub fn commands(&self, rt: &Runtime, world: &World, out: &mut Vec<Command>) -> Result<()> {
        if world.tick % self.every != (7 + 11 * self.ally as u64) % self.every {
            return Ok(());
        }
        let mut teams: Vec<(u64, u32)> = rt
            .rows(world, "Team")?
            .into_iter()
            .map(|(e, b)| (e, u32_at(&b, self.team_ally)))
            .collect();
        teams.sort_unstable();
        let mut idle: Vec<u64> = rt
            .rows(world, "Queue")?
            .into_iter()
            .filter(|(e, b)| {
                u32_at(b, self.queue_len) == 0
                    && u32_at(b, self.queue_state) != NEW
                    && teams
                        .binary_search_by_key(e, |t| t.0)
                        .is_ok_and(|k| teams[k].1 == self.ally)
            })
            .map(|(e, _)| e)
            .collect();
        if idle.len() < self.wave {
            return Ok(());
        }
        idle.sort_unstable();
        let Some(bases) = bases(rt, world)? else { return Ok(()) };
        let (x, y) = bases[(self.ally as usize + 1) % 2];
        out.extend(
            idle.into_iter()
                .map(|e| unit_command(self.ally, self.stream, e, ATTACK, false, x, y)),
        );
        Ok(())
    }
}
