//! The sim module ABI, shared by the host (`ae_sim`) and the guest SDK
//! (`ae-sdk`). `no_std`, no dependencies, so it builds for every target.
//!
//! # Schema section
//!
//! A sim module describes its components, event streams and systems in a
//! wasm custom section named [`SCHEMA_SECTION`]. The section is a
//! concatenation of self-delimiting records. The linker concatenates them in
//! an order nobody controls, so the host never relies on record order:
//! components are keyed by stable id, systems by name, and system order comes
//! from the one [`KIND_ORDER`] record.
//!
//! Every record starts with `u32 kind, u32 len` (`len` counts the whole
//! record, header included, and is a multiple of 4). All numbers are
//! little-endian. Strings are `u32 len` + bytes padded to 4, except where a
//! record header already carries the length.
//!
//! | Kind | Body after the header |
//! |---|---|
//! | [`KIND_COMPONENT`] | `u64 id, u32 size, u32 align, u32 flags, u32 field_count, u32 name_len, u32 0`, name (pad 4), then per field: `u64 id, u32 offset, u32 size, u32 kind, u32 flags, u32 name_len, u32 0`, name (pad 4) |
//! | [`KIND_SYSTEM`] | `u32 mode, u32 cadence, u32 cadence_n, u32 cadence_phase, u32 terms, u32 emits, u32 reads, u32 after, u32 name_len, u32 0`, name (pad 4), terms `{u64 id, u32 access, u32 0}`, emitted stream ids `u64`, read stream ids `u64`, `after` names as strings |
//! | [`KIND_ORDER`] | `u32 count, u32 0`, system names as strings |
//! | [`KIND_DATA`] | `u32 name_len, u32 0`, name (pad 4), payload (the rest of the record): named data for the engine's services, e.g. `nav.classes` |
//!
//! # Systems
//!
//! A system named `n` is the export `ae_sys_n(desc: u32)`. The host writes a
//! [`RunDesc`] into the calling worker's scratch memory and passes its
//! offset. The call covers one work range: `count` rows of the system's
//! query, columns in term order.

#![no_std]

pub const SCHEMA_SECTION: &str = "ae.schema";
pub const SYSTEM_EXPORT_PREFIX: &str = "ae_sys_";
/// A `schedule!` entry naming a host (native) system rather than one of the
/// module's own: `host:<name>`. The host must provide a system of that name.
pub const HOST_SYSTEM_PREFIX: &str = "host:";

pub const KIND_COMPONENT: u32 = 1;
pub const KIND_SYSTEM: u32 = 2;
pub const KIND_ORDER: u32 = 3;
pub const KIND_DATA: u32 = 4;

/// Component flag: the type is an event stream, not a component.
pub const COMPONENT_EVENT: u32 = 1 << 0;

/// Field flags.
pub const FIELD_HASHED: u32 = 1 << 0;
pub const FIELD_SYNCED: u32 = 1 << 1;
pub const FIELD_SAVED: u32 = 1 << 2;
pub const FIELD_DEFAULT: u32 = FIELD_HASHED | FIELD_SYNCED | FIELD_SAVED;

/// Field flags bits 8..12: the field's view class, what an observer may see
/// of it through the copy layer (`ae_view`). 0 is `VIEW_NEVER`, so a field
/// nobody classified never reaches a view.
pub const FIELD_VIEW_SHIFT: u32 = 8;
pub const FIELD_VIEW_MASK: u32 = 0xF << FIELD_VIEW_SHIFT;
pub const VIEW_NEVER: u32 = 0;
pub const VIEW_ALWAYS: u32 = 1;
pub const VIEW_OWNER_ONLY: u32 = 2;
pub const VIEW_LOS: u32 = 3;
pub const VIEW_RADAR_DEGRADED: u32 = 4;
pub const VIEW_GHOST: u32 = 5;

/// A view class by its attribute name (`#[field(view = "...")]`).
pub fn view_class(name: &str) -> Option<u32> {
    Some(match name {
        "never" => VIEW_NEVER,
        "always" => VIEW_ALWAYS,
        "owner" => VIEW_OWNER_ONLY,
        "los" => VIEW_LOS,
        "radar" => VIEW_RADAR_DEGRADED,
        "ghost" => VIEW_GHOST,
        _ => return None,
    })
}

/// Field value kinds. `KIND_ARRAY` is or-ed onto the element kind.
pub const FIELD_BYTES: u32 = 0;
pub const FIELD_U8: u32 = 1;
pub const FIELD_I8: u32 = 2;
pub const FIELD_U16: u32 = 3;
pub const FIELD_I16: u32 = 4;
pub const FIELD_U32: u32 = 5;
pub const FIELD_I32: u32 = 6;
pub const FIELD_U64: u32 = 7;
pub const FIELD_I64: u32 = 8;
pub const FIELD_F32: u32 = 9;
pub const FIELD_F64: u32 = 10;
pub const FIELD_BOOL: u32 = 11;
pub const FIELD_ENTITY: u32 = 12;
pub const FIELD_ARRAY: u32 = 1 << 8;

pub const ACCESS_READ: u32 = 0;
pub const ACCESS_WRITE: u32 = 1;

pub const MODE_PARALLEL: u32 = 0;
pub const MODE_SERIAL: u32 = 1;

pub const CADENCE_EVERY_TICK: u32 = 0;
pub const CADENCE_EVERY: u32 = 1;
pub const CADENCE_STAGGERED: u32 = 2;
pub const CADENCE_ONCE: u32 = 3;

pub const MAX_TERMS: usize = 16;
pub const MAX_EVENT_READS: usize = 8;

/// Byte offsets inside a component record.
pub mod component {
    pub const ID: usize = 8;
    pub const SIZE: usize = 16;
    pub const ALIGN: usize = 20;
    pub const FLAGS: usize = 24;
    pub const FIELD_COUNT: usize = 28;
    pub const NAME_LEN: usize = 32;
    pub const NAME: usize = 40;
    /// Field entry header length (name follows, padded to 4).
    pub const FIELD_HEADER: usize = 32;
    pub const FIELD_ID: usize = 0;
    pub const FIELD_OFFSET: usize = 8;
    pub const FIELD_SIZE: usize = 12;
    pub const FIELD_KIND: usize = 16;
    pub const FIELD_FLAGS: usize = 20;
    pub const FIELD_NAME_LEN: usize = 24;
}

/// Byte offsets inside a system record.
pub mod system {
    pub const MODE: usize = 8;
    pub const CADENCE: usize = 12;
    pub const CADENCE_N: usize = 16;
    pub const CADENCE_PHASE: usize = 20;
    pub const TERMS: usize = 24;
    pub const EMITS: usize = 28;
    pub const READS: usize = 32;
    pub const AFTER: usize = 36;
    pub const NAME_LEN: usize = 40;
    pub const NAME: usize = 48;
    pub const TERM_LEN: usize = 16;
}

pub const fn pad4(n: usize) -> usize {
    (n + 3) & !3
}

/// Encoded length of a string with a `u32` length prefix.
pub const fn string_len(n: usize) -> usize {
    4 + pad4(n)
}

pub const fn put_u32(buf: &mut [u8], at: usize, v: u32) {
    let b = v.to_le_bytes();
    let mut i = 0;
    while i < 4 {
        buf[at + i] = b[i];
        i += 1;
    }
}

pub const fn put_u64(buf: &mut [u8], at: usize, v: u64) {
    put_u32(buf, at, v as u32);
    put_u32(buf, at + 4, (v >> 32) as u32);
}

pub const fn put_bytes(buf: &mut [u8], at: usize, bytes: &[u8]) {
    let mut i = 0;
    while i < bytes.len() {
        buf[at + i] = bytes[i];
        i += 1;
    }
}

/// Bytes of a [`KIND_DATA`] record before its payload.
pub const fn data_header_len(name: &str) -> usize {
    16 + pad4(name.len())
}

/// Write the header of a [`KIND_DATA`] record that fills `buf`; returns
/// the payload's offset.
pub const fn put_data_header(buf: &mut [u8], name: &str) -> usize {
    assert!(buf.len().is_multiple_of(4), "a data record's length is a multiple of 4");
    put_u32(buf, 0, KIND_DATA);
    put_u32(buf, 4, buf.len() as u32);
    put_u32(buf, 8, name.len() as u32);
    put_u32(buf, 12, 0);
    put_bytes(buf, 16, name.as_bytes());
    data_header_len(name)
}

pub const fn get_u32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]])
}

pub const fn get_u64(buf: &[u8], at: usize) -> u64 {
    get_u32(buf, at) as u64 | (get_u32(buf, at + 4) as u64) << 32
}

/// FNV-1a 64: default stable id of a component, stream or field, from its name.
pub const fn name_id(name: &str) -> u64 {
    let bytes = name.as_bytes();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        i += 1;
    }
    h
}

/// Rows of one event stream a system reads this call.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EventSpan {
    pub stream: u64,
    /// Guest offset of the first event; events are packed at the stream's size.
    pub ptr: u32,
    pub count: u32,
}

/// One system call. Written by the host into worker scratch memory.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RunDesc {
    pub tick: u64,
    /// Game parameter set by the host (e.g. initial unit count).
    pub param: u64,
    /// Position of the system in the schedule.
    pub system: u32,
    /// Index of this work range in the system's range list.
    pub range: u32,
    pub count: u32,
    /// Guest offset of `count` entity ids (`u64`).
    pub entities: u32,
    /// Row filter for staggered systems: visit rows whose entity index
    /// (low 32 bits of the id) `% stagger_n == stagger_phase`. 0 = all rows.
    pub stagger_n: u32,
    pub stagger_phase: u32,
    pub term_count: u32,
    pub event_count: u32,
    /// Bit `i` set: term `i` is written.
    pub write_mask: u32,
    pub reserved: u32,
    /// Guest offset of each term's column at the range's first row.
    pub columns: [u32; MAX_TERMS],
    /// Stable component id of each term, for SDK type checks.
    pub term_ids: [u64; MAX_TERMS],
    pub events: [EventSpan; MAX_EVENT_READS],
}

pub const RUN_DESC_BYTES: usize = 56 +4 * MAX_TERMS + 8 * MAX_TERMS + 16 * MAX_EVENT_READS;
const _: () = assert!(core::mem::size_of::<RunDesc>() == RUN_DESC_BYTES);
const _: () = assert!(core::mem::align_of::<RunDesc>() == 8);

impl RunDesc {
    pub const fn zeroed() -> Self {
        Self {
            tick: 0,
            param: 0,
            system: 0,
            range: 0,
            count: 0,
            entities: 0,
            stagger_n: 0,
            stagger_phase: 0,
            term_count: 0,
            event_count: 0,
            write_mask: 0,
            reserved: 0,
            columns: [0; MAX_TERMS],
            term_ids: [0; MAX_TERMS],
            events: [EventSpan { stream: 0, ptr: 0, count: 0 }; MAX_EVENT_READS],
        }
    }
}

/// Host imports (module `env`) a sim module may call. Component and stream
/// ids are the stable ids from the schema.
pub mod imports {
    /// `ae_spawn(record: u32, len: u32)`; record layout in `ae_sim::imports`.
    pub const SPAWN: &str = "ae_spawn";
    /// `ae_despawn(entity: u64)`.
    pub const DESPAWN: &str = "ae_despawn";
    /// `ae_add(entity: u64, component: u64, data: u32, len: u32)`.
    pub const ADD: &str = "ae_add";
    /// `ae_remove(entity: u64, component: u64)`.
    pub const REMOVE: &str = "ae_remove";
    /// `ae_emit(stream: u64, data: u32, len: u32)`.
    pub const EMIT: &str = "ae_emit";
    /// `ae_spatial_radius(x: f32, y: f32, r: f32, out: u32, cap: u32) -> u32`:
    /// entities whose centre is within `r` of `(x, y)` in the spatial index
    /// the host rebuilt last; writes up to `cap` entity ids (`u64`) at `out`
    /// and returns the total count.
    pub const SPATIAL_RADIUS: &str = "ae_spatial_radius";
}

/// Stable ids of the components engine services read and write (defined in
/// `ae_sdk::nav`; host services check the module's layout against them).
pub mod engine {
    /// `Body { x, y, vx, vy, radius: f32 }`: a moving body.
    pub const BODY: u64 = 0xAE00_0000_0000_0001;
    /// `NavAgent { max_speed: f32, profile: u32, mode: u32 }`.
    pub const NAV_AGENT: u64 = 0xAE00_0000_0000_0002;
    /// `Goal { x, y: f32, group: u64 }`: a destination the map gave the unit.
    pub const GOAL: u64 = 0xAE00_0000_0000_0003;
    /// Event `MoveOrder { entity, x, y, group, kind, reserved }`.
    pub const MOVE_ORDER: u64 = 0xAE00_0000_0000_0004;
    /// Event `TerrainEdit { x0, y0, x1, y1, set, blocked, height, cost }`: a
    /// game's edit of a cell rectangle.
    pub const TERRAIN_EDIT: u64 = 0xAE00_0000_0000_0005;
    /// Event `TerrainChanged { version, x0, y0, x1, y1, flags }`: a terrain
    /// version became authoritative.
    pub const TERRAIN_CHANGED: u64 = 0xAE00_0000_0000_0006;
    /// Event `OrderPoint { group, index, x, y, reserved }`: a guide point of
    /// a route or formation order.
    pub const ORDER_POINT: u64 = 0xAE00_0000_0000_0007;
    /// Event `CostOverlay { id, x0, y0, x1, y1, cost, op }`: add or remove a
    /// removable traversal cost overlay.
    pub const COST_OVERLAY: u64 = 0xAE00_0000_0000_0008;
    /// `Team { ally: u32 }`: the allyteam that owns the entity.
    pub const TEAM: u64 = 0xAE00_0000_0000_0009;
    /// `Sensor { los, radar: f32 }`: line-of-sight and radar radii.
    pub const SENSOR: u64 = 0xAE00_0000_0000_000A;
}
