//! Work ranges: the unit of parallel system execution.

use ae_ecs::flecs_backend as fb;
use ae_ecs::{Access, MAX_QUERY_TERMS, Term};
use ae_wasm::GuestMemory;
use anyhow::{Result, ensure};

/// A slice of one table matched by a query, at most `range_rows` long.
#[derive(Clone, Copy, Debug)]
pub struct WorkRange {
    /// Position in the system's range list: table order, then row order.
    pub index: u32,
    /// Host pointer to the range's entity ids (flecs keeps them on the host heap).
    pub entities: *const u64,
    pub count: u32,
    /// Guest offset of each term's column at the first row of the range.
    pub columns: [u32; MAX_QUERY_TERMS],
}

unsafe impl Send for WorkRange {}
unsafe impl Sync for WorkRange {}

/// Every range of a query, in a deterministic order.
///
/// # Safety
/// `world` and `query` must be live; the ranges are valid until the next
/// structural change. Column storage must be in `memory` (the column arena).
pub unsafe fn ranges(
    world: fb::World,
    query: fb::Query,
    terms: &[Term],
    range_rows: usize,
    memory: GuestMemory,
) -> Result<Vec<WorkRange>> {
    ensure!(range_rows > 0, "range_rows must be positive");
    // A NOT term has no column: the readable terms keep their relative
    // order (the guest's `RunDesc` and a host chunk index that list).
    let readable: Vec<(usize, usize)> = {
        let mut sizes = Vec::new();
        let mut skipped = 0;
        for (index, t) in terms.iter().enumerate() {
            if t.access == Access::Without {
                skipped += 1;
                continue;
            }
            sizes.push((index, unsafe { fb::component_size(world, t.component) }?));
        }
        let _ = skipped;
        sizes
    };
    let mut out = Vec::new();
    unsafe {
        fb::for_each_chunk(world, query, terms, |chunk| {
            for &(term, _) in &readable {
                ensure!(
                    ae_ecs::arena::contains(chunk.fields[term]),
                    "a column is outside the column arena"
                );
            }
            let mut first = 0;
            while first < chunk.count {
                let count = range_rows.min(chunk.count - first);
                let mut columns = [0u32; MAX_QUERY_TERMS];
                for (slot, &(term, size)) in readable.iter().enumerate() {
                    columns[slot] = memory.offset(chunk.fields[term].add(first * size));
                }
                out.push(WorkRange {
                    index: out.len() as u32,
                    entities: chunk.entities.add(first),
                    count: count as u32,
                    columns,
                });
                first += count;
            }
            Ok(())
        })?;
    }
    Ok(out)
}
