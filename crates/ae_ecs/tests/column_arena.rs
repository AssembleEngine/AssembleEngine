//! Checks the patched-flecs column allocator hook: component columns land in
//! a caller-owned arena, entity id arrays do not, and data survives growth.
//!
//! The column allocator is process-wide, so this file holds a single test.

use ae_ecs::flecs_backend as fb;
use ae_ecs::{Access, Term};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

const ARENA_BYTES: usize = 64 << 20;
const HEADER: usize = 16;

static ARENA_BASE: AtomicUsize = AtomicUsize::new(0);
static ARENA_TOP: AtomicUsize = AtomicUsize::new(0);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// Bump allocator with a size header; frees are dropped. Enough for a test.
unsafe extern "C-unwind" fn arena_malloc(size: i32) -> *mut c_void {
    let size = (size as usize + 15) & !15;
    let offset = ARENA_TOP.fetch_add(size + HEADER, Ordering::Relaxed);
    assert!(offset + size + HEADER <= ARENA_BYTES, "test arena exhausted");
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    let base = ARENA_BASE.load(Ordering::Relaxed) as *mut u8;
    unsafe {
        base.add(offset).cast::<usize>().write(size);
        base.add(offset + HEADER).cast()
    }
}

unsafe extern "C-unwind" fn arena_realloc(ptr: *mut c_void, size: i32) -> *mut c_void {
    let new = unsafe { arena_malloc(size) };
    if !ptr.is_null() {
        let old_size = unsafe { ptr.cast::<u8>().sub(HEADER).cast::<usize>().read() };
        unsafe { std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), new.cast::<u8>(), old_size.min(size as usize)) };
    }
    new
}

unsafe extern "C-unwind" fn arena_free(_ptr: *mut c_void) {}

fn in_arena(ptr: *const u8) -> bool {
    let base = ARENA_BASE.load(Ordering::Relaxed);
    (base..base + ARENA_BYTES).contains(&(ptr as usize))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Health(u32);

fn bytes_of<T: Copy>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((value as *const T).cast(), size_of::<T>()) }
}

#[test]
fn columns_live_in_the_arena_and_entity_ids_do_not() {
    let arena = vec![0u64; ARENA_BYTES / 8].leak();
    ARENA_BASE.store(arena.as_mut_ptr() as usize, Ordering::Relaxed);

    const N: usize = 10_000;
    unsafe {
        fb::install_column_allocator(arena_malloc, arena_realloc, arena_free);
        let world = fb::init().unwrap();
        let position = fb::register_component(world, "Position", 8, 4).unwrap();
        let health = fb::register_component(world, "Health", 4, 4).unwrap();

        let mut entities = Vec::with_capacity(N);
        for i in 0..N {
            let e = fb::create_entity(world);
            fb::set_component(
                world,
                e,
                position,
                bytes_of(&Position {
                    x: i as f32,
                    y: -(i as f32),
                }),
            )
            .unwrap();
            // Two archetypes: every third entity also has Health.
            if i % 3 == 0 {
                fb::set_component(world, e, health, bytes_of(&Health(i as u32))).unwrap();
            }
            entities.push(e);
        }
        assert!(ALLOCS.load(Ordering::Relaxed) > 0, "column allocator was never called");

        let terms = [Term {
            component: position,
            access: Access::Write,
        }];
        let query = fb::create_query(world, &terms).unwrap();
        let (mut rows, mut chunks, mut sum) = (0usize, 0usize, 0f64);
        fb::for_each_chunk(world, query, &terms, |chunk| {
            assert!(in_arena(chunk.fields[0]), "Position column is outside the arena");
            assert!(!in_arena(chunk.entities.cast()), "entity id array is inside the arena");
            let column = std::slice::from_raw_parts_mut(chunk.fields[0].cast::<Position>(), chunk.count);
            for p in column.iter_mut() {
                sum += p.x as f64;
                p.y += 1.0;
            }
            rows += chunk.count;
            chunks += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(rows, N);
        assert_eq!(chunks, 2, "expected one chunk per archetype");
        assert_eq!(sum, (N * (N - 1) / 2) as f64, "column data was lost while growing");

        // Structural changes: delete some, move others between archetypes.
        for (i, &e) in entities.iter().enumerate() {
            match i % 5 {
                0 => assert!(fb::delete_entity(world, e)),
                1 => fb::remove_component(world, e, health),
                _ => {}
            }
        }
        let mut alive = 0;
        fb::for_each_chunk(world, query, &terms, |chunk| {
            assert!(in_arena(chunk.fields[0]));
            alive += chunk.count;
            Ok(())
        })
        .unwrap();
        assert_eq!(alive, N - N.div_ceil(5));
        assert!(!fb::entity_alive(world, entities[0]));
        assert!(fb::entity_alive(world, entities[1]));

        fb::fini_query(query);
        fb::fini(world);
    }
}
