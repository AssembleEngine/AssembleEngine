//! Death (feature `life`, M14 step 6, D): the [`Dying`] component and
//! [`kill`], which posts `KILLED` and adds `Dying`.
//!
//! Opt-in like `aim`: every component a module compiles takes an entity id
//! in it. A dying unit stays an entity (its script can animate a fall), but
//! it is excluded by a `without(Dying)` term: the spatial service does not
//! find it and nav drops it like a despawn. With
//! [`resume_with_death`](crate::script::resume_with_death) the death script
//! is cut by its cap and the unit despawned at its end (or kept as a
//! corpse). The engine has no backstop: a game that handles `KILLED`
//! itself and never ends its death keeps the unit dying forever.

use crate::script::{Callins, callin};
use crate::{Component, Entity, abi};

/// A unit killed, its death script running (or its corpse kept): excluded
/// from spatial queries and nav until it is despawned or the game removes
/// the component.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Component)]
#[component(id = 0xAE00_0000_0000_0015)]
pub struct Dying {
    /// The tick of the kill (the sim's `run.tick()`).
    pub killed_at: u64,
    /// What the game says killed it.
    pub cause: u32,
    pub flags: u32,
}

const _: () = assert!(<Dying as Component>::ID == abi::engine::DYING);

/// What a death script's end does with the unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fate {
    /// Despawn it (the usual).
    Remove,
    /// Keep it: the entity stays, its script `DONE` and `Dying` kept, so it
    /// stays excluded and its last frame held by `Anim`.
    Corpse,
}

/// Kill `entity`: post `KILLED` to `callins` (its row's) and add [`Dying`]
/// by command, so the unit is excluded from the next sync point on. Does
/// nothing when it is already `Dying`.
///
/// The calling system declares `get(Dying)` (the presence test) and
/// `write(Callins)`; a unit without a `Script` is the game's to despawn as
/// it does now (the engine keeps it dying).
pub fn kill(entity: Entity, callins: &mut Callins, tick: u64, cause: u32) {
    if is_dying(entity) {
        return;
    }
    callins.post(callin::KILLED);
    crate::add(
        entity,
        &Dying {
            killed_at: tick,
            cause,
            flags: 0,
        },
    );
}

/// Whether `entity` has [`Dying`], through `ae_get` (the system's `get`
/// list must hold it).
fn is_dying(entity: Entity) -> bool {
    let mut out = core::mem::MaybeUninit::<Dying>::uninit();
    let n = unsafe {
        crate::host::ae_get(
            entity.0,
            <Dying as Component>::ID,
            out.as_mut_ptr() as usize as u32,
            core::mem::size_of::<Dying>() as u32,
        )
    };
    n != 0
}
