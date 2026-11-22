//! Lifecycle test system as a sim module.

include!("../../common/lifecycle.rs");

#[link(wasm_import_module = "env")]
extern "C" {
    fn ae_spawn(record: u32, len: u32);
    fn ae_despawn(entity: u64);
    fn ae_add(entity: u64, component: u64, data: u32, len: u32);
    fn ae_remove(entity: u64, component: u64);
}

struct GuestEmit {
    position: u64,
    velocity: u64,
    stunned: u64,
}

fn put(record: &mut [u8], at: usize, bytes: &[u8]) {
    record[at..at + bytes.len()].copy_from_slice(bytes);
}

impl Emit for GuestEmit {
    fn despawn(&mut self, entity: u64) {
        unsafe { ae_despawn(entity) }
    }

    fn spawn(&mut self, position: [f32; 2], velocity: [f32; 2]) {
        // Header, then two entries of 16 header bytes plus 8 data bytes.
        let mut record = [0u8; 8 + 2 * 24];
        put(&mut record, 0, &2u32.to_le_bytes());
        for (index, (component, value)) in [(self.position, position), (self.velocity, velocity)]
            .into_iter()
            .enumerate()
        {
            let at = 8 + index * 24;
            put(&mut record, at, &component.to_le_bytes());
            put(&mut record, at + 8, &8u32.to_le_bytes());
            put(&mut record, at + 16, &value[0].to_le_bytes());
            put(&mut record, at + 20, &value[1].to_le_bytes());
        }
        unsafe { ae_spawn(record.as_ptr() as u32, record.len() as u32) }
    }

    fn add_stunned(&mut self, entity: u64, value: u32) {
        let bytes = value.to_le_bytes();
        unsafe { ae_add(entity, self.stunned, bytes.as_ptr() as u32, 4) }
    }

    fn remove_stunned(&mut self, entity: u64) {
        unsafe { ae_remove(entity, self.stunned) }
    }
}

/// One work range of the lifecycle system. Query terms: Position (write),
/// Velocity (read). `entities` is a copy of the range's entity ids.
#[no_mangle]
pub unsafe extern "C" fn lifecycle(
    positions: *mut [f32; 2],
    velocities: *const [f32; 2],
    entities: *const u64,
    count: u32,
    tick: u32,
    position_component: u64,
    velocity_component: u64,
    stunned_component: u64,
) {
    let n = count as usize;
    let positions = core::slice::from_raw_parts_mut(positions, n);
    let velocities = core::slice::from_raw_parts(velocities, n);
    let entities = core::slice::from_raw_parts(entities, n);
    let mut emit = GuestEmit {
        position: position_component,
        velocity: velocity_component,
        stunned: stunned_component,
    };
    for i in 0..n {
        lifecycle_row(entities[i], tick, &mut positions[i], velocities[i], &mut emit);
    }
}
