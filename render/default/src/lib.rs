//! The default render module: the terrain, then the units over it (sharing
//! its colour target and depth buffer), the water surface when the scene
//! has it, the automatic exposure's meter (the scene's log luminance into a
//! histogram, then the eased EV into a 1×1 image), the bloom, then present
//! or readback, all from the engine's pass library; the draws write the HDR
//! scene image and the glow (their light sources only), the bloom reads the
//! glow at a quarter size, and `post.tonemap` turns both into the target
//! with the eased EV, or into the display intermediate when the output is
//! HDR (`post.encode` maps that to the target). It declares the same graph
//! the engine declares natively (`ae_render::terrain::app::native_graph`),
//! which the host's tests check pass for pass.

use ae_render_sdk::{
    BLOOM_NAME, BLOOM_SCRATCH_NAME, Class, DISPLAY_NAME, Declaration, DeclareEnv, EXPOSURE_HISTOGRAM_NAME,
    EXPOSURE_NAME, GLOW_NAME, GraphBuilder, ImageSpec, LoadOp, PARTICLE_SHADOW_NAME, PassKind, PassWriter,
    ProgramTable, RenderModule, ResourceRef, Role, SCENE_NAME, SHADOW_MAP_NAME, ShaderStage, WATER_REFRACTION_NAME,
    WATER_VOLUME_OUT_NAME, bloom_spec, display_spec, exposure_spec, glow_spec, particle_shadow_spec, render_module,
    scene_spec, shadow_map_spec, vk,
};

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
        // The HDR scene the draws write; `post.tonemap` turns it into the target.
        let scene = g.image(SCENE_NAME, scene_spec(w, h), Class::Transient);
        // An HDR output keeps the display intermediate (paper-white units,
        // working space): the tonemap writes it and `post.encode` maps it to
        // the target. SDR has neither and the tonemap writes the target.
        let display = env
            .uses_display()
            .then(|| g.image(DISPLAY_NAME, display_spec(w, h), Class::Transient));
        // The glow (the bloom's input) and the quarter-size bloom buffers.
        let glow = g.image(GLOW_NAME, glow_spec(w, h), Class::Transient);
        let bloom0 = g.image(BLOOM_NAME, bloom_spec(w, h), Class::Transient);
        let bloom1 = g.image(BLOOM_SCRATCH_NAME, bloom_spec(w, h), Class::Transient);
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
        // The eased EV and its metering histogram (host-owned, shared by the
        // slots: the executor derives the cross-frame barriers).
        let exposure = g.image(EXPOSURE_NAME, exposure_spec(), Class::External);
        let histogram = library(&mut g, EXPOSURE_HISTOGRAM_NAME);
        let reflection = (env.water && env.water_reflect).then(|| {
            use ae_render_sdk::{
                TERRAIN_INDIRECT_REFLECT_NAME, TERRAIN_LEAVES_REFLECT_NAME, UNITS_INDIRECT_REFLECT_NAME,
                UNITS_VISIBLE_REFLECT_NAME, WATER_REFLECTION_DEPTH_NAME, WATER_REFLECTION_NAME,
            };
            let (w, h) = (w.div_ceil(2), h.div_ceil(2));
            let color = g.image(
                WATER_REFLECTION_NAME,
                ImageSpec::color_2d(
                    vk::FORMAT_R16G16B16A16_SFLOAT,
                    w,
                    h,
                    vk::USAGE_COLOR_ATTACHMENT | vk::USAGE_SAMPLED | vk::USAGE_TRANSFER_SRC,
                ),
                Class::Transient,
            );
            let depth = g.image(
                WATER_REFLECTION_DEPTH_NAME,
                ImageSpec::depth_2d(vk::FORMAT_D32_SFLOAT, w, h, vk::USAGE_DEPTH_STENCIL_ATTACHMENT),
                Class::Transient,
            );
            let [leaves, indirect] = [TERRAIN_LEAVES_REFLECT_NAME, TERRAIN_INDIRECT_REFLECT_NAME]
                .map(|name| g.buffer(name, 0, 0, Class::Transient));
            let unit_buffers = env.units.then(|| {
                [UNITS_VISIBLE_REFLECT_NAME, UNITS_INDIRECT_REFLECT_NAME]
                    .map(|name| g.buffer(name, 0, 0, Class::Transient))
            });
            (color, depth, leaves, indirect, unit_buffers)
        });

        let refraction = (env.water && env.water_refract).then(|| {
            let color = g.image(
                WATER_REFRACTION_NAME,
                ImageSpec::color_2d(
                    vk::FORMAT_R16G16B16A16_SFLOAT,
                    w,
                    h,
                    vk::USAGE_COLOR_ATTACHMENT | vk::USAGE_SAMPLED | vk::USAGE_TRANSFER_SRC,
                ),
                Class::Transient,
            );
            (color, library(&mut g, "water.caustics"))
        });

        let volume = (refraction.is_some() && env.water_volume).then(|| {
            g.image(
                WATER_VOLUME_OUT_NAME,
                ImageSpec::color_2d(
                    vk::FORMAT_R16G16B16A16_SFLOAT,
                    w,
                    h,
                    vk::USAGE_COLOR_ATTACHMENT | vk::USAGE_SAMPLED | vk::USAGE_TRANSFER_SRC,
                ),
                Class::Transient,
            )
        });

        let waves = (env.water && env.water_waves).then(|| {
            let [raw, churn, waves, particles, leaves, indirect] = [
                "water.raw",
                "water.churn",
                "water.waves",
                "water.particles",
                "water.mesh.leaves",
                "water.mesh.indirect",
            ]
            .map(|name| g.buffer(name, 0, 0, Class::Transient));
            let emit = p.library("water.emit");
            g.pass("water emit", PassKind::Compute, emit)
                .discard(particles, Role::StorageBufferWrite(ShaderStage::Compute))
                .id();
            for (name, key, target) in [
                ("water splat", "water.splat", raw),
                ("water churn", "water.churn", churn),
            ] {
                let pass = p.library(key);
                g.pass(name, PassKind::Graphics, pass)
                    .access(particles, Role::StorageBufferRead(ShaderStage::Vertex))
                    .color(target, LoadOp::Clear)
                    .id();
            }
            let blur = p.library("water.blur");
            g.pass("water blur", PassKind::Compute, blur)
                .sampled(raw, ShaderStage::Compute)
                .sampled(churn, ShaderStage::Compute)
                .discard(waves, Role::StorageImageWrite(ShaderStage::Compute))
                .id();
            let lod = p.library("water.lod");
            g.pass("water lod", PassKind::Compute, lod)
                .sampled(waves, ShaderStage::Compute)
                .discard(leaves, Role::StorageBufferWrite(ShaderStage::Compute))
                .discard(indirect, Role::StorageBufferWrite(ShaderStage::Compute))
                .id();
            // The surface times its breakers by the shore field; the host
            // bakes it once and again after a ground edit.
            let shore = library(&mut g, "water.shore");
            let bake = p.library("water.shore");
            g.pass("shore bake", PassKind::Compute, bake)
                .access(shore, Role::StorageImageWrite(ShaderStage::Compute))
                .root();
            (waves, leaves, indirect, churn, shore)
        });
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
            // One cull fills every view's lists: the camera's and the
            // shadow cascades', and the reflection's when it draws.
            let mirror = reflection.and_then(|(.., unit_buffers)| unit_buffers);
            if let Some([_, mirror_draws]) = mirror {
                g.pass(
                    "units reset reflect",
                    PassKind::Transfer,
                    p.library("units.reset_reflect"),
                )
                .discard(mirror_draws, Role::TransferDst)
                .id();
            }
            let cull = p.library("units.cull");
            let mut cull = g
                .pass("units cull", PassKind::Compute, cull)
                .access(instances, Role::StorageBufferRead(ShaderStage::Compute))
                .access(visible, Role::StorageBufferWrite(ShaderStage::Compute))
                .access(draws, Role::StorageBufferWrite(ShaderStage::Compute));
            if let Some([mirror_visible, mirror_draws]) = mirror {
                cull = cull
                    .discard(mirror_visible, Role::StorageBufferWrite(ShaderStage::Compute))
                    .access(mirror_draws, Role::StorageBufferWrite(ShaderStage::Compute));
            }
            cull.id();
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
            // Floating hulls cast from where they ride the waves.
            let mut shadows = g
                .pass("shadows", PassKind::Graphics, shadows)
                .access(instances, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(visible, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(draws, Role::IndirectBuffer)
                .access(palette, Role::StorageBufferRead(ShaderStage::Vertex));
            if let Some((waves, ..)) = waves {
                shadows = shadows.sampled(waves, ShaderStage::Vertex);
            }
            shadows.depth(map, LoadOp::Clear).id();
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
        if let Some((color, depth, leaves, reflect_indirect, unit_buffers)) = reflection {
            use ShaderStage::{Compute, Fragment, Vertex};
            g.pass(
                "terrain cull reflect",
                PassKind::Compute,
                p.library("terrain.cull_reflect"),
            )
            .access(indirect, Role::StorageBufferRead(Compute))
            .discard(leaves, Role::StorageBufferWrite(Compute))
            .discard(reflect_indirect, Role::StorageBufferWrite(Compute))
            .id();
            let mut terrain = g
                .pass("terrain reflect", PassKind::Graphics, p.library("terrain.reflect"))
                .access(leaves, Role::StorageBufferRead(Vertex))
                .access(reflect_indirect, Role::IndirectBuffer)
                .color(color, LoadOp::Clear)
                .depth(depth, LoadOp::Clear);
            for map in maps.into_iter().flatten() {
                terrain = terrain.sampled(map, Fragment);
            }
            terrain.id();
            if let (Some([instances, _, _, _, palette]), Some([visible, draws])) = (units, unit_buffers) {
                let mut units = g
                    .pass("units reflect", PassKind::Graphics, p.library("units.reflect"))
                    .access(instances, Role::StorageBufferRead(Vertex))
                    .access(visible, Role::StorageBufferRead(Vertex))
                    .access(draws, Role::IndirectBuffer)
                    .access(palette, Role::StorageBufferRead(Vertex));
                // Floating hulls are mirrored where they ride the waves.
                if let Some((waves, ..)) = waves {
                    units = units.sampled(waves, Vertex);
                }
                let mut units = units.color(color, LoadOp::Load).depth(depth, LoadOp::Load);
                for map in maps.into_iter().flatten() {
                    units = units.sampled(map, Fragment);
                }
                units.id();
            }
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
            let mut draw = g
                .pass("units", PassKind::Graphics, units)
                .access(instances, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(visible, Role::StorageBufferRead(ShaderStage::Vertex))
                .access(draws, Role::IndirectBuffer)
                .access(palette, Role::StorageBufferRead(ShaderStage::Vertex))
                .color(scene, LoadOp::Load)
                .color(glow, LoadOp::Load)
                .depth(depth, LoadOp::Load);
            if let Some((waves, ..)) = waves {
                draw = draw.sampled(waves, ShaderStage::Vertex);
            }
            receive(draw, maps, lights).id();
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
        let particle_sim = |g: &mut GraphBuilder, p: &mut ProgramTable, [state, keys, draws]: [ResourceRef; 3]| {
            let sim = p.library("particles.sim");
            let mut pass = g
                .pass("particles sim", PassKind::Compute, sim)
                .access(state, write)
                .access(keys, write)
                .access(draws, write);
            if let (Some(_), Some((waves, ..))) = (refraction, waves) {
                pass = pass.sampled(waves, ShaderStage::Compute);
            }
            lit(pass, lights, ShaderStage::Compute).id();
        };
        if let Some((color, caustics)) = refraction {
            if let Some(particles) = particles {
                particle_sim(&mut g, &mut p, particles);
            }
            let underwater = p.library("water.underwater");
            let mut underwater = g
                .pass("water underwater", PassKind::Graphics, underwater)
                .color(color, LoadOp::Clear)
                .sampled(scene, ShaderStage::Fragment)
                .sampled(depth, ShaderStage::Fragment)
                .sampled(caustics, ShaderStage::Fragment);
            for map in maps.into_iter().flatten() {
                underwater = underwater.sampled(map, ShaderStage::Fragment);
            }
            if let Some((waves, ..)) = waves {
                underwater = underwater.sampled(waves, ShaderStage::Fragment);
            }
            underwater.id();
            if let (Some([state, keys, indirect]), Some((waves, ..))) = (particles, waves) {
                let pass = p.library("particles.underwater");
                g.pass("particles underwater", PassKind::Graphics, pass)
                    .access(state, Role::StorageBufferRead(ShaderStage::Vertex))
                    .access(keys, Role::StorageBufferRead(ShaderStage::Vertex))
                    .access(indirect, Role::IndirectBuffer)
                    .color(color, LoadOp::Load)
                    .sampled(depth, ShaderStage::Fragment)
                    .sampled(waves, ShaderStage::Fragment)
                    .id();
            }
        }
        if env.water {
            let water = p.library("water.draw");
            let mut water = g
                .pass("water", PassKind::Graphics, water)
                .color(scene, LoadOp::Load)
                .color(glow, LoadOp::Load)
                .depth_read(depth);
            if let Some((color, ..)) = reflection {
                water = water.sampled(color, ShaderStage::Fragment);
            }
            if let Some((color, _)) = refraction {
                water = water.sampled(color, ShaderStage::Fragment);
            }
            if let Some((waves, leaves, indirect, churn, shore)) = waves {
                water = water
                    .sampled(waves, ShaderStage::Graphics)
                    .access(leaves, Role::StorageBufferRead(ShaderStage::Vertex))
                    .access(indirect, Role::IndirectBuffer)
                    .sampled(churn, ShaderStage::Fragment)
                    .sampled(shore, ShaderStage::Fragment);
            }
            receive(water, maps, lights).id();
        }
        if let (Some(_), Some([instances, visible, draws, _, palette])) = (refraction, units) {
            let overlay = p.library("units.water_overlay");
            receive(
                g.pass("units water overlay", PassKind::Graphics, overlay)
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
        }
        // The particles over everything, sampling the depth the scene left.
        if let Some([state, keys, draws]) = particles {
            if refraction.is_none() {
                particle_sim(&mut g, &mut p, [state, keys, draws]);
            }
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
        if let Some(output) = volume {
            let volume = p.library("water.volume");
            let mut pass = g
                .pass("water volume", PassKind::Graphics, volume)
                .sampled(scene, ShaderStage::Fragment)
                .sampled(depth, ShaderStage::Fragment)
                .sampled(refraction.unwrap().0, ShaderStage::Fragment);
            if let Some((wave_image, ..)) = waves {
                pass = pass.sampled(wave_image, ShaderStage::Fragment);
            }
            pass.color(output, LoadOp::DontCare).id();
        }
        let post_scene = volume.unwrap_or(scene);
        // The meter (the scene, the glow the bloom shows, and the depth to
        // undo the fog of war's dimming), after every scene writer and before
        // the bloom and the tonemap.
        let clear = p.library("exposure.clear");
        g.pass("exposure clear", PassKind::Transfer, clear)
            .access(histogram, Role::TransferDst)
            .id();
        let meter = p.library("exposure.meter");
        g.pass("exposure meter", PassKind::Compute, meter)
            .sampled(post_scene, ShaderStage::Compute)
            .sampled(depth, ShaderStage::Compute)
            .sampled(glow, ShaderStage::Compute)
            .access(histogram, Role::StorageBufferReadWrite(ShaderStage::Compute))
            .id();
        let adapt = p.library("exposure.adapt");
        g.pass("exposure adapt", PassKind::Compute, adapt)
            .access(histogram, Role::StorageBufferRead(ShaderStage::Compute))
            .access(exposure, Role::StorageImageReadWrite(ShaderStage::Compute))
            .id();
        let downsample = p.library("bloom.downsample");
        g.pass("bloom downsample", PassKind::Graphics, downsample)
            .sampled(glow, ShaderStage::Fragment)
            .sampled(exposure, ShaderStage::Fragment)
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
        g.pass("tonemap", PassKind::Graphics, tonemap)
            .sampled(post_scene, ShaderStage::Fragment)
            .sampled(bloom0, ShaderStage::Fragment)
            .sampled(exposure, ShaderStage::Fragment)
            .color(display.unwrap_or(target), LoadOp::DontCare)
            .id();
        if let Some(display) = display {
            // The display's light, in the output's encoding (PQ or scRGB).
            let encode = p.library("post.encode");
            g.pass("encode", PassKind::Graphics, encode)
                .sampled(display, ShaderStage::Fragment)
                .color(target, LoadOp::DontCare)
                .id();
        }
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

render_module!(Scene);
