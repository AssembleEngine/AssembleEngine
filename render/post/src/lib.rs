//! A render module with post-processing: the terrain and units (engine
//! library passes) draw into the HDR `scene` image; the engine's
//! `post.tonemap` turns it into the display intermediate when the output is
//! HDR (or into `ldr`, the target's format, in SDR); a fullscreen program of
//! this module grades it; a second one adds a vignette when the host sets
//! flag bit 0. Then `post.encode` writes the target in HDR, or present /
//! readback, as in the default module.
//!
//! The grade runs on the display, before the encode, so it works above
//! paper white: its values are not clamped at 1 (the display is linear light
//! up to the peak) and the encode maps them once, at the end. The HDR
//! vignette multiplies the display, which is range-safe (paper white and the
//! peak scale together); the SDR one blends black over the target, where
//! there is no display intermediate.

use ae_render_sdk::{
    BLOOM_NAME, BLOOM_SCRATCH_NAME, Class, DISPLAY_NAME, Declaration, DeclareEnv, FrameInput, FrameOutput, GLOW_NAME,
    GraphBuilder, ImageSpec, LoadOp, PARTICLE_SHADOW_NAME, PassKind, PassWriter, Program, ProgramTable, RenderModule,
    ResourceRef, Role, SCENE_NAME, SHADOW_MAP_NAME, ShaderStage, bloom_spec, display_spec, glow_spec,
    particle_shadow_spec, render_module, scene_spec, shadow_map_spec, vk,
};

const GRADE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/grade.frag.spv"));
const VIGNETTE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vignette.frag.spv"));
const VIGNETTE_DISPLAY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vignette_display.frag.spv"));

/// Flag bit: draw the vignette.
pub const FLAG_VIGNETTE: u32 = 1;

#[derive(Default)]
struct Post {
    /// Payload of the vignette program.
    vignette: u32,
    /// The output is HDR: the grade and the vignette run on the display and
    /// `post.encode` writes the target.
    hdr: bool,
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
        let depth = g.image(
            "depth",
            ImageSpec::depth_2d(
                vk::FORMAT_D32_SFLOAT,
                w,
                h,
                vk::USAGE_DEPTH_STENCIL_ATTACHMENT | vk::USAGE_SAMPLED,
            ),
            Class::Transient,
        );
        let scene = g.image(SCENE_NAME, scene_spec(w, h), Class::Transient);
        // The glow (the bloom's input) and the quarter-size bloom buffers.
        let glow = g.image(GLOW_NAME, glow_spec(w, h), Class::Transient);
        let bloom0 = g.image(BLOOM_NAME, bloom_spec(w, h), Class::Transient);
        let bloom1 = g.image(BLOOM_SCRATCH_NAME, bloom_spec(w, h), Class::Transient);
        // The output's tail: HDR keeps the display intermediate (linear
        // light in paper-white units, working space) and maps it once in
        // `post.encode`; SDR grades `ldr` straight into the target.
        let hdr = env.uses_display();
        let display = hdr.then(|| g.image(DISPLAY_NAME, display_spec(w, h), Class::Transient));
        let graded = hdr.then(|| g.image("display.graded", display_spec(w, h), Class::Transient));
        let vignetted = hdr.then(|| g.image("display.vignetted", display_spec(w, h), Class::Transient));
        let ldr = (!hdr).then(|| {
            g.image(
                "ldr",
                ImageSpec::color_2d(env.color_format, w, h, vk::USAGE_COLOR_ATTACHMENT | vk::USAGE_SAMPLED),
                Class::Transient,
            )
        });
        let target = if env.readback {
            g.image(
                "readback.target",
                ImageSpec::color_2d(
                    env.color_format,
                    w,
                    h,
                    vk::USAGE_COLOR_ATTACHMENT | vk::USAGE_TRANSFER_SRC,
                ),
                Class::External,
            )
        } else {
            g.image(
                "target",
                ImageSpec::color_2d(env.color_format, w, h, vk::USAGE_COLOR_ATTACHMENT),
                Class::Backbuffer,
            )
        };
        // The unit buffers, then the units' shadow map.
        let units = env.units.then(|| {
            [
                "units.instances",
                "units.visible",
                "units.indirect",
                "units.counts",
                "units.palette",
            ]
            .map(|name| library(&mut g, name))
        });
        let map = (env.units && env.shadows).then(|| g.image(SHADOW_MAP_NAME, shadow_map_spec(), Class::External));
        let particles = env
            .particles
            .then(|| ["particles.state", "particles.keys", "particles.indirect"].map(|name| library(&mut g, name)));
        // The particles' coloured shadows, with the map.
        let tint = (map.is_some() && particles.is_some())
            .then(|| g.image(PARTICLE_SHADOW_NAME, particle_shadow_spec(), Class::External));
        let maps = [map, tint];
        // The clustered lights: the list the host writes and each cluster's.
        let lights = [library(&mut g, "lights.list"), library(&mut g, "lights.clusters")];
        let lod = p.library("terrain.lod");
        g.pass("terrain lod", PassKind::Compute, lod)
            .access(leaves, Role::StorageBufferWrite(ShaderStage::Compute))
            .access(indirect, Role::StorageBufferWrite(ShaderStage::Compute))
            .id();
        // The units' poses and cull, before anything draws them (the
        // shadows read the cull's lists per cascade).
        if let Some([instances, visible, draws, _, palette]) = units {
            let pose = p.library("units.pose");
            g.pass("units pose", PassKind::Compute, pose)
                .access(instances, Role::StorageBufferRead(ShaderStage::Compute))
                .access(palette, Role::StorageBufferWrite(ShaderStage::Compute))
                .id();
            let reset = p.library("units.reset");
            g.pass("units reset", PassKind::Transfer, reset)
                .access(draws, Role::TransferDst)
                .id();
            let cull = p.library("units.cull");
            g.pass("units cull", PassKind::Compute, cull)
                .access(instances, Role::StorageBufferRead(ShaderStage::Compute))
                .access(visible, Role::StorageBufferWrite(ShaderStage::Compute))
                .access(draws, Role::StorageBufferWrite(ShaderStage::Compute))
                .id();
        }
        // The lights' clusters, before anything lit.
        let cluster = p.library("lights.cluster");
        let write = Role::StorageBufferWrite(ShaderStage::Compute);
        g.pass("lights", PassKind::Compute, cluster)
            .access(lights[0], write)
            .access(lights[1], write)
            .id();
        // The units' shadows: the units drawn from the sun, before the
        // passes that sample the map.
        if let (Some([instances, visible, draws, _, palette]), Some(map)) = (units, map) {
            let shadows = p.library("shadows.draw");
            g.pass("shadows", PassKind::Graphics, shadows)
                .access(instances, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(visible, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(draws, Role::IndirectBuffer)
                .access(palette, Role::StorageBufferRead(ShaderStage::Vertex))
                .depth(map, LoadOp::Clear)
                .id();
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
                .color(scene, LoadOp::Clear)
                .color(glow, LoadOp::Clear)
                .depth(depth, LoadOp::Clear),
            maps,
            lights,
        )
        .id();
        if let Some([instances, visible, draws, counts, palette]) = units {
            let units = p.library("units.draw");
            receive(
                g.pass("units", PassKind::Graphics, units)
                    .access(instances, Role::StorageBufferRead(ShaderStage::Vertex))
                    .access(visible, Role::StorageBufferRead(ShaderStage::Vertex))
                    .access(draws, Role::IndirectBuffer)
                    .access(palette, Role::StorageBufferRead(ShaderStage::Vertex))
                    .color(scene, LoadOp::Load)
                    .color(glow, LoadOp::Load)
                    .depth(depth, LoadOp::Load),
                maps,
                lights,
            )
            .id();
            let count = p.library("units.count");
            g.pass("units count", PassKind::Transfer, count)
                .access(draws, Role::TransferSrc)
                .discard(counts, Role::TransferDst)
                .id();
            let read = p.library("host.read");
            g.pass("units count read", PassKind::Transfer, read)
                .access(counts, Role::HostRead)
                .root()
                .id();
        }
        if env.water {
            let water = p.library("water.draw");
            receive(
                g.pass("water", PassKind::Graphics, water)
                    .color(scene, LoadOp::Load)
                    .color(glow, LoadOp::Load)
                    .depth_read(depth),
                maps,
                lights,
            )
            .id();
        }
        // The particles over everything, sampling the depth the scene left.
        if let Some([state, keys, draws]) = particles {
            let sim = p.library("particles.sim");
            lit(
                g.pass("particles sim", PassKind::Compute, sim)
                    .access(state, write)
                    .access(keys, write)
                    .access(draws, write),
                lights,
                ShaderStage::Compute,
            )
            .id();
            let draw = p.library("particles.draw");
            g.pass("particles", PassKind::Graphics, draw)
                .access(state, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(keys, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(draws, Role::IndirectBuffer)
                .color(scene, LoadOp::Load)
                .color(glow, LoadOp::Load)
                .sampled(depth, ShaderStage::Fragment)
                .id();
        }
        let downsample = p.library("bloom.downsample");
        g.pass("bloom downsample", PassKind::Graphics, downsample)
            .sampled(glow, ShaderStage::Fragment)
            .color(bloom0, LoadOp::DontCare)
            .id();
        let blur_x = p.library("bloom.blur_x");
        g.pass("bloom blur x", PassKind::Graphics, blur_x)
            .sampled(bloom0, ShaderStage::Fragment)
            .color(bloom1, LoadOp::DontCare)
            .id();
        let blur_y = p.library("bloom.blur_y");
        g.pass("bloom blur y", PassKind::Graphics, blur_y)
            .sampled(bloom1, ShaderStage::Fragment)
            .color(bloom0, LoadOp::DontCare)
            .id();
        let tonemap = p.library("post.tonemap");
        let tonemap_out = display.or(ldr).expect("an HDR or an SDR tail");
        g.pass("tonemap", PassKind::Graphics, tonemap)
            .sampled(scene, ShaderStage::Fragment)
            .sampled(bloom0, ShaderStage::Fragment)
            .color(tonemap_out, LoadOp::DontCare)
            .id();
        let grade = p.push(Program::Fullscreen {
            label: "grade",
            spirv: GRADE,
            uniform_bytes: 32,
        });
        match (display, graded, vignetted) {
            (Some(display), Some(graded), Some(vignetted)) => {
                // The grade reads the display and writes the graded display;
                // the vignette multiplies that into a third intermediate, and
                // the encode is the only pass that writes the target.
                g.pass("grade", PassKind::Graphics, grade)
                    .sampled(display, ShaderStage::Fragment)
                    .color(graded, LoadOp::DontCare)
                    .id();
                self.vignette = p.push(Program::Fullscreen {
                    label: "vignette_display",
                    spirv: VIGNETTE_DISPLAY,
                    uniform_bytes: 16,
                });
                g.pass("vignette", PassKind::Graphics, self.vignette)
                    .sampled(graded, ShaderStage::Fragment)
                    .color(vignetted, LoadOp::DontCare)
                    .id();
                let encode = p.library("post.encode");
                g.pass("encode", PassKind::Graphics, encode)
                    .sampled(vignetted, ShaderStage::Fragment)
                    .color(target, LoadOp::DontCare)
                    .id();
            }
            _ => {
                let ldr = ldr.expect("an SDR tail has its intermediate");
                g.pass("grade", PassKind::Graphics, grade)
                    .sampled(ldr, ShaderStage::Fragment)
                    .color(target, LoadOp::DontCare)
                    .id();
                self.vignette = p.push(Program::Fullscreen {
                    label: "vignette",
                    spirv: VIGNETTE,
                    uniform_bytes: 16,
                });
                g.pass("vignette", PassKind::Graphics, self.vignette)
                    .color(target, LoadOp::Load)
                    .conditional()
                    .id();
            }
        }
        self.hdr = hdr;
        if env.readback {
            let buffer = library(&mut g, "readback.buffer");
            let copy = p.library("readback.copy");
            g.pass("copy", PassKind::Transfer, copy)
                .access(target, Role::TransferSrc)
                .discard(buffer, Role::TransferDst)
                .id();
            let read = p.library("host.read");
            g.pass("host read", PassKind::Transfer, read)
                .access(buffer, Role::HostRead)
                .root()
                .id();
        } else {
            let present = p.library("present");
            g.pass("present", PassKind::Present, present)
                .present(target)
                .root()
                .id();
        }
        Some(Declaration { graph: g, programs: p })
    }

    fn frame(&mut self, input: &FrameInput, out: &mut FrameOutput) {
        // grade: tint, then contrast and saturation (above 1 too: the grade
        // works on the display, in paper-white units, before the encode).
        floats(&mut out.uniforms, &[1.04, 1.0, 0.94, 1.0, 1.12, 1.15, 0.0, 0.0]);
        // vignette: strength, radius. The HDR vignette multiplies the display
        // (range-safe), so it always runs and a zero strength is the
        // identity; the SDR one blends black over the target and is skipped
        // when off.
        let on = input.flags & FLAG_VIGNETTE != 0;
        let strength = if self.hdr && !on { 0.0 } else { 0.85 };
        floats(&mut out.uniforms, &[strength, 0.35, 0.0, 0.0]);
        if !self.hdr && !on {
            out.disable(self.vignette);
        }
    }
}

/// A receiver samples the shadow maps the graph has, and reads the lights.
fn receive(mut pass: PassWriter<'_>, maps: [Option<ResourceRef>; 2], lights: [ResourceRef; 2]) -> PassWriter<'_> {
    for map in maps.into_iter().flatten() {
        pass = pass.sampled(map, ShaderStage::Fragment);
    }
    lit(pass, lights, ShaderStage::Fragment)
}

/// A lit pass reads the light list and the clusters.
fn lit(pass: PassWriter<'_>, lights: [ResourceRef; 2], stage: ShaderStage) -> PassWriter<'_> {
    pass.access(lights[0], Role::StorageBufferRead(stage))
        .access(lights[1], Role::StorageBufferRead(stage))
}

render_module!(Post);
