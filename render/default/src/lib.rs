//! The default render module: the terrain, then the units over it (sharing
//! its colour target and depth buffer), the water surface when the scene
//! has it, then present or readback, all
//! from the engine's pass library. It declares the same graph the engine
//! declares natively (`ae_render::terrain::app::native_graph`), which the
//! host's tests check pass for pass.

use ae_render_sdk::{Class, Declaration, DeclareEnv, GraphBuilder, ImageSpec, LoadOp, PassKind, PassWriter, ProgramTable, RenderModule, ResourceRef, Role, PARTICLE_SHADOW_NAME, SHADOW_MAP_NAME, ShaderStage, particle_shadow_spec, render_module, shadow_map_spec, vk};

#[derive(Default)]
struct Scene;

/// A library buffer: `External`, size 0 (the host sizes it).
fn library(graph: &mut GraphBuilder, name: &str) -> ResourceRef {
    graph.buffer(name, 0, 0, Class::External)
}

impl RenderModule for Scene {
    fn declare(&mut self, env: &DeclareEnv) -> Option<Declaration> {
        let (w, h) = (env.width, env.height);
        let mut g = GraphBuilder::new();
        let mut p = ProgramTable::new();
        let leaves = library(&mut g, "terrain.leaves");
        let indirect = library(&mut g, "terrain.indirect");
        let depth = g.image("depth", ImageSpec::depth_2d(vk::FORMAT_D32_SFLOAT, w, h, vk::USAGE_DEPTH_STENCIL_ATTACHMENT | vk::USAGE_SAMPLED), Class::Transient);
        let target = if env.readback {
            g.image("readback.target", ImageSpec::color_2d(env.color_format, w, h, vk::USAGE_COLOR_ATTACHMENT | vk::USAGE_TRANSFER_SRC), Class::External)
        } else {
            g.image("target", ImageSpec::color_2d(env.color_format, w, h, vk::USAGE_COLOR_ATTACHMENT), Class::Backbuffer)
        };
        // The unit buffers, then the units' shadow map.
        let units = env.units.then(|| ["units.instances", "units.visible", "units.indirect", "units.counts"].map(|name| library(&mut g, name)));
        let map = (env.units && env.shadows).then(|| g.image(SHADOW_MAP_NAME, shadow_map_spec(), Class::External));
        let particles = env.particles.then(|| ["particles.state", "particles.keys", "particles.indirect"].map(|name| library(&mut g, name)));
        // The particles' coloured shadows, with the map.
        let tint = (map.is_some() && particles.is_some()).then(|| g.image(PARTICLE_SHADOW_NAME, particle_shadow_spec(), Class::External));
        let maps = [map, tint];
        let lod = p.library("terrain.lod");
        g.pass("terrain lod", PassKind::Compute, lod)
            .access(leaves, Role::StorageBufferWrite(ShaderStage::Compute))
            .access(indirect, Role::StorageBufferWrite(ShaderStage::Compute))
            .id();
        // The units' shadows: every unit drawn from the sun, before the
        // passes that sample the map.
        if let (Some([instances, ..]), Some(map)) = (units, map) {
            let shadows = p.library("shadows.draw");
            g.pass("shadows", PassKind::Graphics, shadows).access(instances, Role::StorageBufferRead(ShaderStage::Vertex)).depth(map, LoadOp::Clear).id();
        }
        // The particles' coloured shadows: this frame's particles (the
        // simulation, declared later, is their state's only writer).
        if let (Some([state, keys, draws]), Some(tint)) = (particles, tint) {
            let shadows = p.library("shadows.particles");
            g.pass("particle shadows", PassKind::Graphics, shadows)
                .access(state, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(keys, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(draws, Role::IndirectBuffer)
                .color(tint, LoadOp::Clear)
                .id();
        }
        let draw = p.library("terrain.draw");
        receive(
            g.pass("terrain", PassKind::Graphics, draw)
                .access(leaves, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(indirect, Role::IndirectBuffer)
                .color(target, LoadOp::Clear)
                .depth(depth, LoadOp::Clear),
            maps,
        )
        .id();
        if let Some([instances, visible, draws, counts]) = units {
            let reset = p.library("units.reset");
            g.pass("units reset", PassKind::Transfer, reset).access(draws, Role::TransferDst).id();
            let cull = p.library("units.cull");
            g.pass("units cull", PassKind::Compute, cull)
                .access(instances, Role::StorageBufferRead(ShaderStage::Compute))
                .access(visible, Role::StorageBufferWrite(ShaderStage::Compute))
                .access(draws, Role::StorageBufferWrite(ShaderStage::Compute))
                .id();
            let units = p.library("units.draw");
            receive(
                g.pass("units", PassKind::Graphics, units)
                    .access(instances, Role::StorageBufferRead(ShaderStage::Vertex))
                    .access(visible, Role::StorageBufferRead(ShaderStage::Vertex))
                    .access(draws, Role::IndirectBuffer)
                    .color(target, LoadOp::Load)
                    .depth(depth, LoadOp::Load),
                maps,
            )
            .id();
            let count = p.library("units.count");
            g.pass("units count", PassKind::Transfer, count).access(draws, Role::TransferSrc).discard(counts, Role::TransferDst).id();
            let read = p.library("host.read");
            g.pass("units count read", PassKind::Transfer, read).access(counts, Role::HostRead).root().id();
        }
        if env.water {
            let water = p.library("water.draw");
            receive(g.pass("water", PassKind::Graphics, water).color(target, LoadOp::Load).depth_read(depth), maps).id();
        }
        // The particles over everything, sampling the depth the scene left.
        if let Some([state, keys, draws]) = particles {
            let sim = p.library("particles.sim");
            let write = Role::StorageBufferWrite(ShaderStage::Compute);
            g.pass("particles sim", PassKind::Compute, sim).access(state, write).access(keys, write).access(draws, write).id();
            let draw = p.library("particles.draw");
            g.pass("particles", PassKind::Graphics, draw)
                .access(state, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(keys, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(draws, Role::IndirectBuffer)
                .color(target, LoadOp::Load)
                .sampled(depth, ShaderStage::Fragment)
                .id();
        }
        if env.readback {
            let buffer = library(&mut g, "readback.buffer");
            let copy = p.library("readback.copy");
            g.pass("copy", PassKind::Transfer, copy).access(target, Role::TransferSrc).discard(buffer, Role::TransferDst).id();
            let read = p.library("host.read");
            g.pass("host read", PassKind::Transfer, read).access(buffer, Role::HostRead).root().id();
        } else {
            let present = p.library("present");
            g.pass("present", PassKind::Present, present).present(target).root().id();
        }
        Some(Declaration { graph: g, programs: p })
    }
}

/// A receiver samples the shadow maps the graph has.
fn receive(mut pass: PassWriter<'_>, maps: [Option<ResourceRef>; 2]) -> PassWriter<'_> {
    for map in maps.into_iter().flatten() {
        pass = pass.sampled(map, ShaderStage::Fragment);
    }
    pass
}

render_module!(Scene);
