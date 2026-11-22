//! Minimal wrappers over the Flecs C API (lifted from the ecs_sim prototype).
//!
//! Safety model: all Flecs calls for one world happen on one thread at a
//! time. Structural changes only occur at system boundaries, never while
//! column pointers from a chunk are in use.

use crate::{Access, ComponentId, EntityId, MAX_QUERY_TERMS, Term};
use anyhow::{Result, bail, ensure};
use flecs_ecs_sys::*;
use std::ffi::{CString, c_void};
use std::mem::MaybeUninit;
use std::ptr;

pub type World = *mut ecs_world_t;
pub type Query = *mut ecs_query_t;
pub type Table = *mut ecs_table_t;

// Values of flecs' `ecs_inout_kind_t` (EcsInOutNone, EcsInOut, EcsIn).
const FLECS_INOUT_NONE: i16 = 1;
const FLECS_INOUT: i16 = 3;
const FLECS_IN: i16 = 4;

/// One table span matched by a query.
pub struct ChunkView {
    pub entities: *const u64,
    pub count: usize,
    /// Data column pointer per declared term (length == terms.len()).
    pub fields: [*mut u8; MAX_QUERY_TERMS],
}

pub type ColumnMalloc = unsafe extern "C-unwind" fn(ecs_size_t) -> *mut c_void;
pub type ColumnRealloc = unsafe extern "C-unwind" fn(*mut c_void, ecs_size_t) -> *mut c_void;
pub type ColumnFree = unsafe extern "C-unwind" fn(*mut c_void);

/// Route component column storage to a custom allocator (patched Flecs,
/// `vendor/patches/0001-column-allocator.patch`). Everything else Flecs
/// allocates (world, tables, queries, entity arrays) stays on the host heap.
/// Process-wide; must run before [`init`].
///
/// # Safety
///
/// The three functions must be a real allocator: every pointer Flecs hands back
/// must come from them, and they must stay valid for the process' lifetime. Call
/// this before any other Flecs call, [`init`] in particular.
pub unsafe fn install_column_allocator(malloc: ColumnMalloc, realloc: ColumnRealloc, free: ColumnFree) {
    unsafe { ecs_os_set_api_defaults() };
    let allocator = ecs_column_allocator_t {
        malloc_: Some(malloc),
        realloc_: Some(realloc),
        free_: Some(free),
    };
    unsafe { ecs_column_allocator_set(&allocator) };
}
///
/// # Safety
///
/// Flecs allows one live world, and every call for it must happen on one thread
/// at a time. The world must be passed to [`fini`] exactly once.
pub unsafe fn init() -> Result<World> {
    let world = unsafe { ecs_init() };
    if world.is_null() {
        bail!("ecs_init failed");
    }
    Ok(world)
}
///
/// # Safety
///
/// `world` must come from [`init`] and must not be touched afterwards: no Flecs
/// call, no live query, no chunk pointer taken from it.
pub unsafe fn fini(world: World) {
    if !world.is_null() {
        unsafe { ecs_fini(world) };
    }
}

/// Register a component with a storage layout `{name, size, align}`.
///
/// # Safety
///
/// `world` must be a live world, and `size`/`align` the component's real layout:
/// Flecs sizes its columns from them, so a wrong size writes past column memory.
pub unsafe fn register_component(world: World, name: &str, size: u32, align: u32) -> Result<ComponentId> {
    let cname = CString::new(name).map_err(|_| anyhow::anyhow!("component name has NUL"))?;
    let mut entity_desc = unsafe { MaybeUninit::<ecs_entity_desc_t>::zeroed().assume_init() };
    entity_desc.name = cname.as_ptr();
    entity_desc.use_low_id = true;
    let entity = unsafe { ecs_entity_init(world, &entity_desc) };
    if entity == 0 {
        bail!("ecs_entity_init failed for component {name}");
    }

    let mut desc = unsafe { MaybeUninit::<ecs_component_desc_t>::zeroed().assume_init() };
    desc.entity = entity;
    desc.type_.size = size as ecs_size_t;
    desc.type_.alignment = align.max(8) as ecs_size_t;
    let component = unsafe { ecs_component_init(world, &desc) };
    if component == 0 {
        bail!("ecs_component_init failed for component {name}");
    }
    Ok(ComponentId(component))
}
///
/// # Safety
///
/// `world` must be a live world.
pub unsafe fn create_entity(world: World) -> EntityId {
    let desc = unsafe { MaybeUninit::<ecs_entity_desc_t>::zeroed().assume_init() };
    EntityId(unsafe { ecs_entity_init(world, &desc) })
}

/// The table (archetype) holding exactly `components`, created if needed.
///
/// # Safety
///
/// `world` must be a live world and every entry of `components` a component
/// registered in it.
pub unsafe fn find_table(world: World, components: &[ComponentId]) -> Result<Table> {
    let mut ids: Vec<u64> = components.iter().map(|c| c.0).collect();
    ids.sort_unstable();
    ids.dedup();
    let table = unsafe { ecs_table_find(world, ids.as_ptr(), ids.len() as i32) };
    ensure!(!table.is_null(), "ecs_table_find failed");
    Ok(table)
}

/// Create an entity directly in `table`. Setting its components afterwards
/// writes in place: no table move.
///
/// # Safety
///
/// `world` must be a live world and `table` one of its tables (from
/// [`find_table`] on the same world).
pub unsafe fn spawn_in_table(world: World, table: Table) -> EntityId {
    EntityId(unsafe { ecs_new_w_table(world, table) })
}

/// Byte size of a component's values (0 for a tag).
///
/// # Safety
///
/// `world` must be a live world and `component` registered in it.
pub unsafe fn component_size(world: World, component: ComponentId) -> Result<usize> {
    let info = unsafe { ecs_get_type_info(world, component.0) };
    ensure!(!info.is_null(), "component {} has no type info", component.0);
    Ok(unsafe { (*info).size } as usize)
}
///
/// # Safety
///
/// `world` must be a live world; `entity` may be any id, `0` reads as absent.
pub unsafe fn has_component(world: World, entity: EntityId, component: ComponentId) -> bool {
    unsafe { ecs_has_id(world, entity.0, component.0) }
}

/// Assign a component value to an entity (structural add when absent).
///
/// # Safety
///
/// `world` must be a live world and `bytes` exactly the component's size: Flecs
/// copies that many bytes into the column.
pub unsafe fn set_component(world: World, entity: EntityId, component: ComponentId, bytes: &[u8]) -> Result<()> {
    ensure!(unsafe { ecs_is_valid(world, entity.0) }, "invalid entity {}", entity.0);
    unsafe { ecs_set_id(world, entity.0, component.0, bytes.len(), bytes.as_ptr().cast()) };
    Ok(())
}

/// Remove a component from an entity (structural).
///
/// # Safety
///
/// `world` must be a live world, and no chunk pointer from an earlier
/// [`for_each_chunk`] may be in use, since this is a structural change.
pub unsafe fn remove_component(world: World, entity: EntityId, component: ComponentId) {
    unsafe { ecs_remove_id(world, entity.0, component.0) };
}
///
/// # Safety
///
/// `world` must be a live world, and no chunk pointer from an earlier
/// [`for_each_chunk`] may be in use, since this is a structural change.
pub unsafe fn delete_entity(world: World, entity: EntityId) -> bool {
    if entity.0 == 0 || !unsafe { ecs_is_valid(world, entity.0) } {
        return false;
    }
    unsafe { ecs_delete(world, entity.0) };
    true
}
///
/// # Safety
///
/// `world` must be a live world; `entity` may be any id.
pub unsafe fn entity_alive(world: World, entity: EntityId) -> bool {
    entity.0 != 0 && unsafe { ecs_is_valid(world, entity.0) }
}

/// Read `component` of `entity` in place: a pointer into the column arena,
/// or `None` when the entity is not alive or does not have it. The pointer
/// stays valid until the next structural change; within a schedule layer,
/// where structural changes wait for the sync point, it holds for the whole
/// layer.
///
/// # Safety
///
/// `world` must be a live world and `component` registered in it.
pub unsafe fn get_component(world: World, entity: EntityId, component: ComponentId) -> Option<*const u8> {
    if !unsafe { entity_alive(world, entity) } {
        return None;
    }
    let p = unsafe { ecs_get_id(world, entity.0, component.0) };
    (!p.is_null()).then(|| p.cast::<u8>())
}

/// Create a cached query from AND terms.
///
/// # Safety
///
/// `world` must be a live world and every term's component registered in it. The
/// query must be passed to [`fini_query`] before the world is dropped.
pub unsafe fn create_query(world: World, terms: &[Term]) -> Result<Query> {
    unsafe { query_with_cache(world, terms, ecs_query_cache_kind_t_EcsQueryCacheAll) }
}

/// A query without a cache. Flecs gives every cached query an entity, so
/// creating one in the middle of a run changes the ids later spawns get;
/// an uncached query creates no entity and leaves the world as it was.
/// For inspection (counts, end-of-tick captures) between ticks.
///
/// # Safety
///
/// As [`create_query`]: a live world, registered components, and [`fini_query`]
/// before the world is dropped.
pub unsafe fn create_query_uncached(world: World, terms: &[Term]) -> Result<Query> {
    unsafe { query_with_cache(world, terms, ecs_query_cache_kind_t_EcsQueryCacheNone) }
}

unsafe fn query_with_cache(world: World, terms: &[Term], cache: ecs_query_cache_kind_t) -> Result<Query> {
    ensure!(!terms.is_empty() && terms.len() <= MAX_QUERY_TERMS, "bad term count");
    let mut desc = unsafe { MaybeUninit::<ecs_query_desc_t>::zeroed().assume_init() };
    for (slot, term) in desc.terms.iter_mut().zip(terms.iter()) {
        slot.id = term.component.0;
        match term.access {
            Access::Read => slot.inout = FLECS_IN,
            Access::Write => slot.inout = FLECS_INOUT,
            Access::Without => {
                slot.inout = FLECS_INOUT_NONE;
                slot.oper = ecs_oper_kind_t_EcsNot as i16;
            }
        }
    }
    desc.cache_kind = cache;
    let query = unsafe { ecs_query_init(world, &desc) };
    if query.is_null() {
        bail!("ecs_query_init failed");
    }
    Ok(query)
}
///
/// # Safety
///
/// `query` must come from [`create_query`] or [`create_query_uncached`] on a
/// world that is still alive, and must not be used afterwards.
pub unsafe fn fini_query(query: Query) {
    if !query.is_null() {
        unsafe { ecs_query_fini(query) };
    }
}

/// Iterate a query, yielding one chunk (table span) at a time.
///
/// The pointers in a chunk are valid only until the next structural change.
///
/// # Safety
///
/// `world` and `query` must be live, and `terms` must be the query's own terms in
/// the same order. The pointers in a [`ChunkView`] are valid only until the next
/// structural change, so `f` must not make one.
pub unsafe fn for_each_chunk(
    world: World,
    query: Query,
    terms: &[Term],
    mut f: impl FnMut(ChunkView) -> Result<()>,
) -> Result<()> {
    ensure!(terms.len() <= MAX_QUERY_TERMS, "too many terms");
    let mut iter = unsafe { ecs_query_iter(world, query) };
    while unsafe { ecs_query_next(&mut iter) } {
        if iter.count <= 0 {
            continue;
        }
        let mut view = ChunkView {
            entities: iter.entities,
            count: iter.count as usize,
            fields: [ptr::null_mut(); MAX_QUERY_TERMS],
        };
        for (index, term) in terms.iter().enumerate() {
            if term.access == Access::Without {
                // A NOT term has no column; `fields` stays null for it.
                continue;
            }
            let column = unsafe { ecs_table_get_id(world, iter.table, term.component.0, 0) };
            if column.is_null() {
                bail!("query component {} has no table column", term.component.0);
            }
            view.fields[index] = column.cast();
        }
        f(view)?;
    }
    Ok(())
}
