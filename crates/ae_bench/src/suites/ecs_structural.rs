//! Structural-change bench for the ECS backend: a steady stream of spawn,
//! despawn, add-component and remove-component commands applied in a fixed
//! order, as the sim will do at its sync points.
//!
//! Besides cost, this checks that entity id allocation is repeatable: two
//! worlds fed the same commands must end with the same digest, and the digest
//! is recorded so a change between runs or builds shows up in `bench-diff`.

use super::Options;
use crate::measure::time;
use crate::record::Report;
use ae_ecs::flecs_backend as fb;
use ae_ecs::{Access, ComponentId, EntityId, Term};
use anyhow::{Result, ensure};

const TICKS: usize = 20;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct World {
    world: fb::World,
    position: ComponentId,
    velocity: ComponentId,
    health: ComponentId,
    stunned: ComponentId,
    query: fb::Query,
    terms: [Term; 1],
    alive: Vec<EntityId>,
    rng: Rng,
    serial: u32,
}

impl World {
    fn new(entities: usize) -> Result<Self> {
        unsafe {
            let world = fb::init()?;
            let position = fb::register_component(world, "Position", 8, 4)?;
            let velocity = fb::register_component(world, "Velocity", 8, 4)?;
            let health = fb::register_component(world, "Health", 4, 4)?;
            let stunned = fb::register_component(world, "Stunned", 4, 4)?;
            let terms = [Term { component: position, access: Access::Read }];
            let query = fb::create_query(world, &terms)?;
            let mut this = Self { world, position, velocity, health, stunned, query, terms, alive: Vec::with_capacity(entities * 2), rng: Rng(0xAE), serial: 0 };
            for _ in 0..entities {
                this.spawn()?;
            }
            Ok(this)
        }
    }

    /// Spawn into one of four archetypes, chosen by a running serial number.
    fn spawn(&mut self) -> Result<()> {
        let serial = self.serial;
        self.serial += 1;
        unsafe {
            let entity = fb::create_entity(self.world);
            let position = [serial as f32, -(serial as f32)];
            fb::set_component(self.world, entity, self.position, bytes(&position))?;
            if serial % 2 == 0 {
                fb::set_component(self.world, entity, self.velocity, bytes(&[1.0f32, 0.5]))?;
            }
            if serial % 3 == 0 {
                fb::set_component(self.world, entity, self.health, bytes(&serial))?;
            }
            self.alive.push(entity);
        }
        Ok(())
    }

    /// One tick's structural commands: 20% despawn, 20% spawn, 30% add, 30% remove.
    fn apply_commands(&mut self, commands: usize) -> Result<()> {
        let each = commands / 10;
        unsafe {
            for _ in 0..each * 2 {
                let index = self.rng.below(self.alive.len());
                let entity = self.alive.swap_remove(index);
                ensure!(fb::delete_entity(self.world, entity), "despawn of a dead entity");
            }
            for _ in 0..each * 2 {
                self.spawn()?;
            }
            for _ in 0..each * 3 {
                let entity = self.alive[self.rng.below(self.alive.len())];
                fb::set_component(self.world, entity, self.stunned, bytes(&7u32))?;
            }
            for _ in 0..each * 3 {
                let entity = self.alive[self.rng.below(self.alive.len())];
                fb::remove_component(self.world, entity, self.stunned);
            }
        }
        Ok(())
    }

    /// Order-independent digest over (entity id, position) of every live row.
    fn digest(&self) -> Result<(u64, usize, usize)> {
        let (mut sum, mut rows, mut chunks) = (0u64, 0usize, 0usize);
        unsafe {
            fb::for_each_chunk(self.world, self.query, &self.terms, |chunk| {
                let ids = std::slice::from_raw_parts(chunk.entities, chunk.count);
                let positions = std::slice::from_raw_parts(chunk.fields[0].cast::<[f32; 2]>(), chunk.count);
                for (id, p) in ids.iter().zip(positions) {
                    let mut h = id ^ ((p[0].to_bits() as u64) << 32 | p[1].to_bits() as u64);
                    h = (h ^ (h >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                    sum = sum.wrapping_add(h ^ (h >> 29));
                }
                rows += chunk.count;
                chunks += 1;
                Ok(())
            })?;
        }
        Ok((sum, rows, chunks))
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

fn bytes<T: Copy>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast(), size_of::<T>()) }
}

fn run_world(entities: usize, commands: usize) -> Result<(u64, usize, usize)> {
    let mut world = World::new(entities)?;
    for _ in 0..TICKS {
        world.apply_commands(commands)?;
    }
    world.digest()
}

pub fn run(options: &Options) -> Result<Report> {
    let mut report = Report::new("ecs-structural");
    let (entities, commands) = if options.small { (10_000, 1_000) } else { (50_000, 5_000) };
    let tag = format!("{}k_{}cmd", entities / 1000, commands);

    let (digest, rows, chunks) = run_world(entities, commands)?;
    let (again, ..) = run_world(entities, commands)?;
    ensure!(digest == again, "two worlds fed the same commands ended with different digests");
    ensure!(rows == entities, "live rows {rows} != {entities}: spawns and despawns should balance");
    report.info(&format!("{tag}/digest_low32"), digest & 0xFFFF_FFFF, "");
    report.info(&format!("{tag}/live_rows"), rows as u64, "");
    report.counter(&format!("{tag}/chunks"), chunks as u64, "");

    report.time(&format!("{tag}/populate"), time(options.budget, || World::new(entities).map(|w| w.alive.len())));
    let mut world = World::new(entities)?;
    report.time(&format!("{tag}/apply_tick_commands"), time(options.budget, || world.apply_commands(commands)));
    report.time(&format!("{tag}/digest_scan"), time(options.budget, || world.digest()));
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_commands_give_same_digest() {
        let a = run_world(2_000, 200).unwrap();
        let b = run_world(2_000, 200).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.1, 2_000);
    }
}
