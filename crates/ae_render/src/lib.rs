//! Vulkan side of the renderer, lifted from the ecs_sim prototype
//! (`~/ecs_sim/src/render/`) with its GPU tests.
//!
//! - `window`: SDL3 window and event pump on the main thread, `SurfaceSource`;
//! - `device`: instance and device, validation counters;
//! - `swapchain`, `memory`, `pipeline`: Vulkan plumbing;
//! - `executor`: realizes a baked [`ae_framegraph`] graph, records and submits;
//! - `thread`: the GPU render thread and the `RenderApp` interface;
//! - `demo`: the prototype's demo graph and recorder (a test fixture);
//! - `module`, `graph_file`: render modules (WASM) and frame graphs from data files;
//! - `upload`: one-shot uploads of static buffers and images;
//! - `terrain`: CBT/LEB adaptive terrain from map-rendering.
//!
//! Shaders in `shaders/` are compiled to SPIR-V by `build.rs` with `glslc`.

pub mod decals;
pub mod demo;
pub mod device;
pub mod executor;
pub mod fullscreen;
pub mod graph_file;
pub mod library;
pub mod memory;
pub mod models;
pub mod particles;
pub mod module;
pub mod pipeline;
pub mod shadow;
pub mod swapchain;
pub mod terrain;
pub mod thread;
pub mod tonemap;
pub mod units;
pub mod upload;
pub mod window;
