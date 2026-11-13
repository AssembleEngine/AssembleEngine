//! Sim core.
//!
//! * [`work`]: cut query results into fixed-size row ranges. The cut
//!   depends only on table contents, never on the worker count.
//! * [`command`]: structural changes (spawn, despawn, add, remove) are
//!   deferred into per-worker buffers under a [`command::CommandKey`], merged
//!   by key and applied serially, so entity ids and table layout come out the
//!   same at any worker count.
//! * [`imports`]: the host functions sim modules call to emit commands.

pub mod command;
pub mod imports;
pub mod work;
