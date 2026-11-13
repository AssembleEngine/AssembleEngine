//! Engine storage and query layer.
//!
//! The rest of the engine talks to this crate in engine terms (`EntityId`,
//! `ComponentId`, chunks of column pointers). Flecs types stay inside
//! [`flecs_backend`], so the ECS can be replaced without touching the guest
//! ABI or any other crate.

pub mod flecs_backend;

/// Engine entity id. Opaque to guests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct EntityId(pub u64);

/// Engine component id. Opaque to guests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ComponentId(pub u64);

/// How a query term accesses its component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

/// One AND term of a query: a component and how it is accessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Term {
    pub component: ComponentId,
    pub access: Access,
}

/// Most terms a single query may declare.
pub const MAX_QUERY_TERMS: usize = 16;
