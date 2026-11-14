//! Native engine services wired into the sim schedule as host systems
//! (`ae_sim::host`). A module uses them by listing them in its `schedule!`
//! (`host(nav_spawn)`, `host(nav)`, `host(spatial)`) and by declaring the
//! engine components they read and write (`ae_sdk::nav`, feature `nav`).
//!
//! * [`nav::NavPlugin`]: `nav_spawn` spawns a map's units once; `nav` runs
//!   the navigation service over every entity with a `Body` and a
//!   `NavAgent`, taking `MoveOrder` events as group commands.
//! * [`spatial::SpatialPlugin`]: `spatial` rebuilds the spatial index from
//!   every `Body`, for module systems' `ae_sdk::spatial_radius` queries.
//! * [`catalog::CatalogPlugin`]: `catalog_spawn` spawns catalogue defs
//!   from `SpawnRequest` events, with their allyteam and position.
//! * [`view::ViewPipeline`]: the copy layer over the sim: captures the
//!   viewed fields, stamps sensor coverage, builds observers' view tables.

pub mod catalog;
pub mod nav;
pub mod spatial;
pub mod view;
