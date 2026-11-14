//! Adaptive terrain: a concurrent binary tree (CBT) of longest-edge
//! bisection (LEB) triangles refined on the GPU against a screen-space
//! error target, lifted from `~/map-rendering`.
//!
//! - `leb`: CPU reference of the CBT and LEB (mirrored by
//!   `shaders/terrain_common.glsl`), unit-tested;
//! - `heightmap`: corner heightmaps, PGM files, the test-terrain generator,
//!   and the error pyramid behind the LOD metric;
//! - `camera`: the RTS orbit camera (reverse-Z infinite projection);
//! - `renderer`: GPU resources, the LOD compute sequence and the draw;
//! - `app`: the terrain as a frame graph [`RenderApp`](crate::thread::RenderApp).

pub mod app;
pub mod camera;
pub mod heightmap;
pub mod leb;
pub mod renderer;
