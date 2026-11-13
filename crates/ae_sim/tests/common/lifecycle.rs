// Row logic of the lifecycle test system, shared by the test guest and the
// native reference (both `include!` this file).
//
// Every row integrates its position, then, depending on a hash of
// (entity id, tick): despawns itself, spawns a child, gains the Stunned
// component, or loses it. Every structural change goes through `Emit`, so the
// guest turns them into deferred host commands.

pub const DT: f32 = 1.0 / 30.0;

pub trait Emit {
    fn despawn(&mut self, entity: u64);
    fn spawn(&mut self, position: [f32; 2], velocity: [f32; 2]);
    fn add_stunned(&mut self, entity: u64, value: u32);
    fn remove_stunned(&mut self, entity: u64);
}

#[inline]
pub fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
pub fn lifecycle_row(entity: u64, tick: u32, position: &mut [f32; 2], velocity: [f32; 2], emit: &mut impl Emit) {
    position[0] += velocity[0] * DT;
    position[1] += velocity[1] * DT;
    match mix(entity ^ ((tick as u64) << 40)) % 1000 {
        0..=3 => emit.despawn(entity),
        4..=7 => emit.spawn(*position, [-velocity[1], velocity[0]]),
        8..=27 => emit.add_stunned(entity, tick),
        28..=47 => emit.remove_stunned(entity),
        _ => {}
    }
}
