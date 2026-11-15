//! A render module with post-processing: the terrain and units (engine
//! library passes) draw into an intermediate `scene` image; a fullscreen
//! program of this module grades it into the target; a second one, a
//! vignette, blends over the target when the host sets flag bit 0 (it is a
//! conditional pass the module disables otherwise). Then present or
//! readback, as in the default module.

use ae_render_sdk::{Class, Declaration, DeclareEnv, FrameInput, FrameOutput, GraphBuilder, ImageSpec, LoadOp, PassKind, PassWriter, Program, ProgramTable, RenderModule, ResourceRef, Role, SHADOW_MAP_NAME, ShaderStage, render_module, shadow_map_spec, vk};

const GRADE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/grade.frag.spv"));
const VIGNETTE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vignette.frag.spv"));

/// Flag bit: draw the vignette.
pub const FLAG_VIGNETTE: u32 = 1;

#[derive(Default)]
struct Post {
    /// Payload of the vignette program.
    vignette: u32,
}

fn library(graph: &mut GraphBuilder, name: &str) -> ResourceRef {
    graph.buffer(name, 0, 0, Class::External)
}

fn floats(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

impl RenderModule for Post {
    fn declare(&mut self, env: &DeclareEnv) -> Option<Declaration> {
        let (w, h) = (env.width, env.height);
        let mut g = GraphBuilder::new();
        let mut p = ProgramTable::new();
        let leaves = library(&mut g, "terrain.leaves");
        let indirect = library(&mut g, "terrain.indirect");
        let depth = g.image("depth", ImageSpec::depth_2d(vk::FORMAT_D32_SFLOAT, w, h, vk::USAGE_DEPTH_STENCIL_ATTACHMENT | vk::USAGE_SAMPLED), Class::Transient);
        let scene = g.image("scene", ImageSpec::color_2d(env.color_format, w, h, vk::USAGE_COLOR_ATTACHMENT | vk::USAGE_SAMPLED), Class::Transient);
        let target = if env.readback {
            g.image("readback.target", ImageSpec::color_2d(env.color_format, w, h, vk::USAGE_COLOR_ATTACHMENT | vk::USAGE_TRANSFER_SRC), Class::External)
        } else {
            g.image("target", ImageSpec::color_2d(env.color_format, w, h, vk::USAGE_COLOR_ATTACHMENT), Class::Backbuffer)
        };
        // The unit buffers, then the units' shadow map.
        let units = env.units.then(|| ["units.instances", "units.visible", "units.indirect", "units.counts"].map(|name| library(&mut g, name)));
        let map = (env.units && env.shadows).then(|| g.image(SHADOW_MAP_NAME, shadow_map_spec(), Class::External));
        let particles = env.particles.then(|| ["particles.state", "particles.keys", "particles.indirect"].map(|name| library(&mut g, name)));
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
        let draw = p.library("terrain.draw");
        receive(
            g.pass("terrain", PassKind::Graphics, draw)
                .access(leaves, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(indirect, Role::IndirectBuffer)
                .color(scene, LoadOp::Clear)
                .depth(depth, LoadOp::Clear),
            map,
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
                    .color(scene, LoadOp::Load)
                    .depth(depth, LoadOp::Load),
                map,
            )
            .id();
            let count = p.library("units.count");
            g.pass("units count", PassKind::Transfer, count).access(draws, Role::TransferSrc).discard(counts, Role::TransferDst).id();
            let read = p.library("host.read");
            g.pass("units count read", PassKind::Transfer, read).access(counts, Role::HostRead).root().id();
        }
        if env.water {
            let water = p.library("water.draw");
            receive(g.pass("water", PassKind::Graphics, water).color(scene, LoadOp::Load).depth_read(depth), map).id();
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
                .color(scene, LoadOp::Load)
                .sampled(depth, ShaderStage::Fragment)
                .id();
        }
        let grade = p.push(Program::Fullscreen { label: "grade", spirv: GRADE, uniform_bytes: 32 });
        g.pass("grade", PassKind::Graphics, grade).sampled(scene, ShaderStage::Fragment).color(target, LoadOp::DontCare).id();
        self.vignette = p.push(Program::Fullscreen { label: "vignette", spirv: VIGNETTE, uniform_bytes: 16 });
        g.pass("vignette", PassKind::Graphics, self.vignette).color(target, LoadOp::Load).conditional().id();
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

    fn frame(&mut self, input: &FrameInput, out: &mut FrameOutput) {
        // grade: tint, then contrast and saturation.
        floats(&mut out.uniforms, &[1.04, 1.0, 0.94, 1.0, 1.12, 1.15, 0.0, 0.0]);
        // vignette: strength, radius.
        floats(&mut out.uniforms, &[0.85, 0.35, 0.0, 0.0]);
        if input.flags & FLAG_VIGNETTE == 0 {
            out.disable(self.vignette);
        }
    }
}

/// A receiver samples the units' shadow map when the graph has it.
fn receive(pass: PassWriter<'_>, map: Option<ResourceRef>) -> PassWriter<'_> {
    match map {
        Some(map) => pass.sampled(map, ShaderStage::Fragment),
        None => pass,
    }
}

render_module!(Post);
