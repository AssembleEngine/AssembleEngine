//! Rust SDK for AssembleEngine sim modules.
//!
//! ```ignore
//! use ae_sdk::{Component, Run, system, schedule};
//!
//! #[repr(C)]
//! #[derive(Clone, Copy, Component)]
//! pub struct Position { pub x: f32, pub y: f32 }
//!
//! #[system(write(Position), read(Velocity))]
//! fn integrate(run: &Run) {
//!     let (p, v) = (run.write::<Position>(), run.read::<Velocity>());
//!     for i in run.rows() { p[i].x += v[i].x; }
//! }
//!
//! schedule!(integrate);
//! ```
//!
//! Rules a module must follow (the engine checks what it can):
//! * No sim state outside components: statics and thread-locals are not
//!   saved, hashed or shared between workers.
//! * A system only touches the columns of its own range, and only writes
//!   the terms it declared `write`.
//! * Structural changes and events go through [`despawn`], [`add`],
//!   [`remove`], [`Spawn`] and [`emit`]; they take effect when the system's
//!   layer is done.

pub use ae_abi as abi;
pub use ae_sdk_macros::{Component, schedule, system};

// The derive names `::ae_sdk`; this lets the SDK use it on its own types.
extern crate self as ae_sdk;

#[cfg(feature = "nav")]
pub mod nav;

#[cfg(feature = "catalog")]
pub mod catalog;

#[cfg(feature = "anim")]
pub mod anim;

#[cfg(feature = "script")]
pub mod script;

use abi::RunDesc;

/// Engine entity id.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Entity(pub u64);

impl Entity {
    /// Index part of the id (stable while the entity lives).
    pub fn index(self) -> u32 {
        self.0 as u32
    }
}

/// A component type. Implemented by `#[derive(Component)]`.
///
/// # Safety
/// The type must be `#[repr(C)]`, `Copy`, and its schema record must match
/// its layout. Only the derive should implement it.
pub unsafe trait Component: Copy + 'static {
    const ID: u64;
    const NAME: &'static str;
    const SCHEMA: &'static [u8];
}

/// An event stream type. Implemented by `#[derive(Component)]` with
/// `#[component(event)]`.
///
/// # Safety
/// As for [`Component`].
pub unsafe trait Event: Copy + 'static {
    const ID: u64;
    const NAME: &'static str;
    const SCHEMA: &'static [u8];
}

/// A field value type the schema can describe.
///
/// # Safety
/// `KIND` must describe the type's bytes.
pub unsafe trait Field {
    const KIND: u32;
}

macro_rules! field_kinds {
    ($($t:ty => $k:ident),* $(,)?) => { $(unsafe impl Field for $t { const KIND: u32 = abi::$k; })* };
}
field_kinds!(u8 => FIELD_U8, i8 => FIELD_I8, u16 => FIELD_U16, i16 => FIELD_I16, u32 => FIELD_U32, i32 => FIELD_I32,
    u64 => FIELD_U64, i64 => FIELD_I64, f32 => FIELD_F32, f64 => FIELD_F64, bool => FIELD_BOOL, Entity => FIELD_ENTITY);

unsafe impl<T: Field, const N: usize> Field for [T; N] {
    const KIND: u32 = T::KIND | abi::FIELD_ARRAY;
}

/// One system call: a range of rows and the columns of the system's terms.
pub struct Run {
    desc: &'static RunDesc,
}

impl Run {
    /// # Safety
    /// `desc` must be the descriptor offset the host passed to the system export.
    #[doc(hidden)]
    pub unsafe fn from_desc(desc: u32) -> Self {
        Self { desc: unsafe { &*(desc as usize as *const RunDesc) } }
    }

    pub fn tick(&self) -> u64 {
        self.desc.tick
    }

    /// The host's game parameter.
    pub fn param(&self) -> u64 {
        self.desc.param
    }

    /// Rows in this range.
    pub fn len(&self) -> usize {
        self.desc.count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.desc.count == 0
    }

    pub fn entities(&self) -> &[Entity] {
        if self.desc.count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(self.desc.entities as usize as *const Entity, self.len()) }
    }

    /// Row indices this call should visit: every row, or for a staggered
    /// system the rows whose turn it is.
    pub fn rows(&self) -> impl Iterator<Item = usize> + '_ {
        let (n, phase) = (self.desc.stagger_n, self.desc.stagger_phase);
        let entities = self.entities();
        (0..self.len()).filter(move |&i| n == 0 || entities[i].index() % n == phase)
    }

    fn term<T: Component>(&self) -> usize {
        let terms = &self.desc.term_ids[..self.desc.term_count as usize];
        match terms.iter().position(|&id| id == T::ID) {
            Some(i) => i,
            None => panic!("system has no query term for {}", T::NAME),
        }
    }

    /// The system has a query term for `T`.
    pub fn has<T: Component>(&self) -> bool {
        self.desc.term_ids[..self.desc.term_count as usize].contains(&T::ID)
    }

    /// The first row of a term's column (`write`: a term declared `write`).
    /// For the SDK's own row access (`script::Unit`).
    #[doc(hidden)]
    pub fn column_ptr<T: Component>(&self, write: bool) -> *mut T {
        let i = self.term::<T>();
        assert!(!write || self.desc.write_mask & (1 << i) != 0, "{} is a read-only term", T::NAME);
        self.desc.columns[i] as usize as *mut T
    }

    /// The column of a term declared `read` or `write`.
    pub fn read<T: Component>(&self) -> &[T] {
        let i = self.term::<T>();
        if self.desc.count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(self.desc.columns[i] as usize as *const T, self.len()) }
    }

    /// The column of a term declared `write`. Call once per term per run:
    /// two calls for the same type alias.
    #[allow(clippy::mut_from_ref)]
    pub fn write<T: Component>(&self) -> &mut [T] {
        let i = self.term::<T>();
        assert!(self.desc.write_mask & (1 << i) != 0, "{} is a read-only term", T::NAME);
        if self.desc.count == 0 {
            return &mut [];
        }
        unsafe { core::slice::from_raw_parts_mut(self.desc.columns[i] as usize as *mut T, self.len()) }
    }

    /// Events of a stream this system declared in `reads_events`: every
    /// event emitted since this system last ran (see `ae_sim::events`).
    pub fn events<E: Event>(&self) -> &[E] {
        let spans = &self.desc.events[..self.desc.event_count as usize];
        let Some(span) = spans.iter().find(|s| s.stream == E::ID) else { panic!("system does not read events of {}", E::NAME) };
        if span.count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(span.ptr as usize as *const E, span.count as usize) }
    }
}

#[cfg(target_family = "wasm")]
mod host {
    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        pub fn ae_spawn(record: u32, len: u32);
        pub fn ae_despawn(entity: u64);
        pub fn ae_add(entity: u64, component: u64, data: u32, len: u32);
        pub fn ae_remove(entity: u64, component: u64);
        pub fn ae_emit(stream: u64, data: u32, len: u32);
        pub fn ae_spatial_radius(x: f32, y: f32, r: f32, out: u32, cap: u32) -> u32;
        #[allow(dead_code)]
        pub fn ae_lookup(scope: u64, name: u64) -> u32;
        #[allow(dead_code)]
        pub fn ae_query(op: u64, input: u32, input_len: u32, out: u32, out_len: u32) -> u32;
    }
}

#[cfg(not(target_family = "wasm"))]
#[allow(clippy::missing_safety_doc)]
mod host {
    fn native() -> ! {
        panic!("sim module imports exist only inside the engine")
    }
    pub unsafe fn ae_spawn(_: u32, _: u32) {
        native()
    }
    pub unsafe fn ae_despawn(_: u64) {
        native()
    }
    pub unsafe fn ae_add(_: u64, _: u64, _: u32, _: u32) {
        native()
    }
    pub unsafe fn ae_remove(_: u64, _: u64) {
        native()
    }
    pub unsafe fn ae_emit(_: u64, _: u32, _: u32) {
        native()
    }
    pub unsafe fn ae_spatial_radius(_: f32, _: f32, _: f32, _: u32, _: u32) -> u32 {
        native()
    }
    #[allow(dead_code)]
    pub unsafe fn ae_lookup(_: u64, _: u64) -> u32 {
        native()
    }
    #[allow(dead_code)]
    pub unsafe fn ae_query(_: u64, _: u32, _: u32, _: u32, _: u32) -> u32 {
        native()
    }
}

fn bytes_of<T: Copy>(value: &T) -> (u32, u32) {
    (value as *const T as usize as u32, size_of::<T>() as u32)
}

pub fn despawn(entity: Entity) {
    unsafe { host::ae_despawn(entity.0) }
}

/// Add `value` to `entity`, or overwrite it.
pub fn add<C: Component>(entity: Entity, value: &C) {
    let (ptr, len) = bytes_of(value);
    unsafe { host::ae_add(entity.0, C::ID, ptr, len) }
}

pub fn remove<C: Component>(entity: Entity) {
    unsafe { host::ae_remove(entity.0, C::ID) }
}

pub fn emit<E: Event>(event: &E) {
    let (ptr, len) = bytes_of(event);
    unsafe { host::ae_emit(E::ID, ptr, len) }
}

/// Entities whose centre lies within `r` of `(x, y)`, from the spatial
/// index the host's `spatial` system built last (a module that calls this
/// lists `host(spatial)` in its `schedule!`, and runs `after(spatial)` to see
/// this tick's positions). Fills `out` with up to `out.len()` ids, in the
/// index's fixed order, and returns the total count, which may be larger.
pub fn spatial_radius(x: f32, y: f32, r: f32, out: &mut [Entity]) -> usize {
    let (ptr, cap) = (out.as_mut_ptr() as usize as u32, out.len() as u32);
    unsafe { host::ae_spatial_radius(x, y, r, ptr, cap) as usize }
}

/// Builds a spawn record: `Spawn::new().with(&a).with(&b).submit()`.
pub struct Spawn {
    buf: [u64; Spawn::WORDS],
    len: usize,
    count: u32,
}

impl Spawn {
    const WORDS: usize = 128;

    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self { buf: [0; Self::WORDS], len: 8, count: 0 }
    }

    fn bytes(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.buf.as_mut_ptr().cast::<u8>(), Self::WORDS * 8) }
    }

    pub fn with<C: Component>(mut self, value: &C) -> Self {
        let size = size_of::<C>();
        let at = self.len;
        assert!(at + 16 + size.next_multiple_of(8) <= Self::WORDS * 8, "spawn record too large");
        let src = unsafe { core::slice::from_raw_parts((value as *const C).cast::<u8>(), size) };
        let b = self.bytes();
        abi::put_u64(b, at, C::ID);
        abi::put_u32(b, at + 8, size as u32);
        b[at + 16..at + 16 + size].copy_from_slice(src);
        self.len = at + 16 + size.next_multiple_of(8);
        self.count += 1;
        self
    }

    pub fn submit(mut self) {
        let count = self.count;
        abi::put_u32(self.bytes(), 0, count);
        let (ptr, len) = (self.buf.as_ptr() as usize as u32, self.len as u32);
        unsafe { host::ae_spawn(ptr, len) }
    }
}

/// Declare components or events the module uses without a `#[system]`
/// naming them (e.g. ones only engine services create, like
/// `ae_sdk::catalog::Def`): puts copies of their schema records in this
/// crate, so they are in the module's schema however the linker treats
/// the crates that define them.
#[macro_export]
macro_rules! declare {
    ($($ty:ty),+ $(,)?) => {
        $(
            const _: () = {
                #[used]
                #[cfg_attr(target_family = "wasm", unsafe(link_section = "ae.schema"))]
                static RECORD: [u8; <$ty>::__AE_SCHEMA.len()] = <$ty>::__AE_SCHEMA;
            };
        )+
    };
}
