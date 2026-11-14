//! The default render module: the terrain, then the units over it (sharing
//! its colour target and depth buffer), then present or readback, all
//! from the engine's pass library. It declares the same graph the engine
//! declares natively (`ae_render::terrain::app::native_graph`), which the
//! host's tests check pass for pass.

use ae_render_sdk::{Class, Declaration, DeclareEnv, GraphBuilder, ImageSpec, LoadOp, PassKind, ProgramTable, RenderModule, ResourceRef, Role, ShaderStage, render_module, vk};

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
        let depth = g.image("depth", ImageSpec::depth_2d(vk::FORMAT_D32_SFLOAT, w, h, vk::USAGE_DEPTH_STENCIL_ATTACHMENT), Class::Transient);
        let target = if env.readback {
            g.image("readback.target", ImageSpec::color_2d(env.color_format, w, h, vk::USAGE_COLOR_ATTACHMENT | vk::USAGE_TRANSFER_SRC), Class::External)
        } else {
            g.image("target", ImageSpec::color_2d(env.color_format, w, h, vk::USAGE_COLOR_ATTACHMENT), Class::Backbuffer)
        };
        let lod = p.library("terrain.lod");
        g.pass("terrain lod", PassKind::Compute, lod)
            .access(leaves, Role::StorageBufferWrite(ShaderStage::Compute))
            .access(indirect, Role::StorageBufferWrite(ShaderStage::Compute))
            .id();
        let draw = p.library("terrain.draw");
        g.pass("terrain", PassKind::Graphics, draw)
            .access(leaves, Role::StorageBufferRead(ShaderStage::Vertex))
            .access(indirect, Role::IndirectBuffer)
            .color(target, LoadOp::Clear)
            .depth(depth, LoadOp::Clear)
            .id();
        if env.units {
            let instances = library(&mut g, "units.instances");
            let visible = library(&mut g, "units.visible");
            let draws = library(&mut g, "units.indirect");
            let reset = p.library("units.reset");
            g.pass("units reset", PassKind::Transfer, reset).access(draws, Role::TransferDst).id();
            let cull = p.library("units.cull");
            g.pass("units cull", PassKind::Compute, cull)
                .access(instances, Role::StorageBufferRead(ShaderStage::Compute))
                .access(visible, Role::StorageBufferWrite(ShaderStage::Compute))
                .access(draws, Role::StorageBufferWrite(ShaderStage::Compute))
                .id();
            let units = p.library("units.draw");
            g.pass("units", PassKind::Graphics, units)
                .access(instances, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(visible, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(draws, Role::IndirectBuffer)
                .color(target, LoadOp::Load)
                .depth(depth, LoadOp::Load)
                .id();
            let counts = library(&mut g, "units.counts");
            let count = p.library("units.count");
            g.pass("units count", PassKind::Transfer, count).access(draws, Role::TransferSrc).discard(counts, Role::TransferDst).id();
            let read = p.library("host.read");
            g.pass("units count read", PassKind::Transfer, read).access(counts, Role::HostRead).root().id();
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

render_module!(Scene);
