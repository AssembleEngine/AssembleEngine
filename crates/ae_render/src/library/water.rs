use super::{LibraryEnv, LibraryError, error};
use ae_framegraph::{FrameGraph, ResourceId, Role, ShaderStage};
use ae_render_abi::module::{
    DECLARE_WATER_REFLECT, DECLARE_WATER_REFRACT, DECLARE_WATER_VOLUME, DECLARE_WATER_WAVES,
    TERRAIN_INDIRECT_REFLECT_NAME, TERRAIN_LEAVES_REFLECT_NAME, UNITS_INDIRECT_REFLECT_NAME,
    UNITS_VISIBLE_REFLECT_NAME, WATER_CHURN_NAME, WATER_PARTICLES_NAME, WATER_RAW_NAME, WATER_REFLECTION_DEPTH_NAME,
    WATER_REFLECTION_NAME, WATER_REFRACTION_NAME, WATER_VOLUME_OUT_NAME, WATER_WAVES_NAME,
};

/// The optional water tiers. An installed graph's set describes which
/// passes own the water effects.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaterTiers {
    pub reflect: bool,
    pub refract: bool,
    pub waves: bool,
    pub volume: bool,
}

impl WaterTiers {
    /// Encode a request for a graph declaration. Volume requires refraction.
    pub(crate) fn declare_flags(self, water: bool) -> u32 {
        if !water {
            return 0;
        }
        (u32::from(self.reflect) * DECLARE_WATER_REFLECT)
            | (u32::from(self.refract || self.volume) * DECLARE_WATER_REFRACT)
            | (u32::from(self.waves) * DECLARE_WATER_WAVES)
            | (u32::from(self.volume) * DECLARE_WATER_VOLUME)
    }
}

pub(super) fn check<'a, R>(
    graph: &FrameGraph<R>,
    env: &LibraryEnv,
    libs: impl Fn(usize) -> Option<&'a str>,
) -> Result<WaterTiers, LibraryError> {
    let reflect_passes: &[&str] = if env.units {
        &[
            "terrain.cull_reflect",
            "terrain.reflect",
            "units.reset_reflect",
            "units.cull_reflect",
            "units.reflect",
        ]
    } else {
        &["terrain.cull_reflect", "terrain.reflect"]
    };
    let reflect_buffers: &[&str] = if env.units {
        &[
            TERRAIN_LEAVES_REFLECT_NAME,
            TERRAIN_INDIRECT_REFLECT_NAME,
            UNITS_VISIBLE_REFLECT_NAME,
            UNITS_INDIRECT_REFLECT_NAME,
        ]
    } else {
        &[TERRAIN_LEAVES_REFLECT_NAME, TERRAIN_INDIRECT_REFLECT_NAME]
    };
    if !env.units
        && ((0..graph.passes.len()).any(|i| {
            matches!(
                libs(i),
                Some("units.reset_reflect" | "units.cull_reflect" | "units.reflect")
            )
        }) || graph.resources.iter().any(|r| {
            matches!(
                r.name.as_str(),
                UNITS_INDIRECT_REFLECT_NAME | UNITS_VISIBLE_REFLECT_NAME
            )
        }))
    {
        return Err(error(format!(
            "passes `units.cull_reflect` and `units.reflect` for `{WATER_REFLECTION_NAME}` require units"
        )));
    }

    let tier = |passes: &[&str], images: &[&str], buffers: &[&str], producer: &str, consumer: &str| {
        check_tier(graph, env.water, &libs, passes, images, buffers, producer, consumer)
    };
    let reflect = tier(
        reflect_passes,
        &[WATER_REFLECTION_NAME, WATER_REFLECTION_DEPTH_NAME],
        reflect_buffers,
        "terrain.reflect",
        "water.draw",
    )?;
    let refract = tier(
        &["water.underwater"],
        &[WATER_REFRACTION_NAME],
        &[],
        "water.underwater",
        "water.draw",
    )?;
    let waves = tier(
        &["water.emit", "water.splat", "water.churn", "water.blur", "water.lod"],
        &[WATER_WAVES_NAME, WATER_RAW_NAME, WATER_CHURN_NAME],
        &[
            WATER_PARTICLES_NAME,
            ae_render_abi::module::WATER_MESH_LEAVES_NAME,
            ae_render_abi::module::WATER_MESH_INDIRECT_NAME,
        ],
        "water.blur",
        "water.draw",
    )?;
    let volume = tier(
        &["water.volume"],
        &[WATER_VOLUME_OUT_NAME],
        &[],
        "water.volume",
        "post.tonemap",
    )?;
    if volume && !refract {
        return Err(error(format!(
            "pass `water.volume` needs `{WATER_REFRACTION_NAME}` from `water.underwater`"
        )));
    }
    let before = |first, second, resource| {
        let position = |name| (0..graph.passes.len()).position(|i| libs(i) == Some(name));
        if !position(first).zip(position(second)).is_some_and(|(a, b)| a < b) {
            return Err(error(format!(
                "pass `{first}` must run before `{second}` for `{resource}`"
            )));
        }
        Ok(())
    };
    if reflect {
        before("terrain.lod", "terrain.cull_reflect", WATER_REFLECTION_NAME)?;
        before("terrain.cull_reflect", "terrain.reflect", WATER_REFLECTION_NAME)?;
        if env.units {
            before("units.reset_reflect", "units.cull_reflect", WATER_REFLECTION_NAME)?;
            before("units.cull_reflect", "units.reflect", WATER_REFLECTION_NAME)?;
            before("terrain.reflect", "units.reflect", WATER_REFLECTION_NAME)?;
        }
    }
    let position = |name| (0..graph.passes.len()).position(|i| libs(i) == Some(name));
    if refract {
        check_refraction(graph, env, &libs)?;
    } else if position("units.water_overlay").is_some() {
        return Err(error(format!(
            "`units.water_overlay` requires `{WATER_REFRACTION_NAME}`"
        )));
    }
    if waves {
        before("water.emit", "water.splat", WATER_PARTICLES_NAME)?;
        before("water.emit", "water.churn", WATER_PARTICLES_NAME)?;
        before("water.splat", "water.blur", WATER_RAW_NAME)?;
        before("water.churn", "water.blur", WATER_CHURN_NAME)?;
        before("water.blur", "water.lod", WATER_WAVES_NAME)?;
        let draw = &graph.passes[position("water.draw").expect("checked water.draw")];
        let reads = |name, role| {
            draw.intents
                .iter()
                .any(|i| graph.resources[i.resource.index()].name == name && i.role == role)
        };
        for (name, role) in [
            (
                ae_render_abi::module::WATER_MESH_LEAVES_NAME,
                Role::StorageBufferRead(ShaderStage::Vertex),
            ),
            (ae_render_abi::module::WATER_MESH_INDIRECT_NAME, Role::IndirectBuffer),
            // The churned water's facets and whitewater, and the breakers'
            // timing.
            (
                ae_render_abi::module::WATER_CHURN_NAME,
                Role::Sampled(ShaderStage::Fragment),
            ),
            ("water.shore", Role::Sampled(ShaderStage::Fragment)),
        ] {
            if !reads(name, role) {
                return Err(error(format!("pass `water.draw` must read `{name}` as {role:?}")));
            }
        }
        if env.units {
            let units = position("units.draw")
                .ok_or_else(|| error("waves require `units.draw` in scenes with units".into()))?;
            before("water.blur", "units.draw", WATER_WAVES_NAME)?;
            if !graph.passes[units].intents.iter().any(|i| {
                graph.resources[i.resource.index()].name == WATER_WAVES_NAME
                    && i.role == Role::Sampled(ShaderStage::Vertex)
            }) {
                return Err(error(format!(
                    "pass `units.draw` must sample `{WATER_WAVES_NAME}` in the vertex shader"
                )));
            }
        }
    }
    if position("particles.underwater").is_some() && !(env.particles && waves && refract) {
        return Err(error(
            "`particles.underwater` requires particles, waves and refraction".into(),
        ));
    }
    if waves && refract {
        let underwater = position("water.underwater").expect("checked refraction tier");
        if !graph.passes[underwater].intents.iter().any(|i| {
            graph.resources[i.resource.index()].name == WATER_WAVES_NAME
                && i.role == Role::Sampled(ShaderStage::Fragment)
        }) {
            return Err(error(format!(
                "`water.underwater` must sample displaced `{WATER_WAVES_NAME}`"
            )));
        }
    }
    if env.particles && waves && refract {
        before("particles.sim", "particles.underwater", WATER_REFRACTION_NAME)?;
        before("water.underwater", "particles.underwater", WATER_REFRACTION_NAME)?;
        before("particles.underwater", "water.draw", WATER_REFRACTION_NAME)?;
        before("water.draw", "particles.draw", WATER_REFRACTION_NAME)?;
        for (pass, role) in [
            ("particles.sim", Role::Sampled(ShaderStage::Compute)),
            ("particles.underwater", Role::Sampled(ShaderStage::Fragment)),
        ] {
            let index = position(pass).expect("checked particle pass");
            if !graph.passes[index]
                .intents
                .iter()
                .any(|i| graph.resources[i.resource.index()].name == WATER_WAVES_NAME && i.role == role)
            {
                return Err(error(format!("`{pass}` must use `{WATER_WAVES_NAME}` as {role:?}")));
            }
        }
    }
    if volume {
        before("water.draw", "water.volume", WATER_VOLUME_OUT_NAME)?;
        let volume_index = position("water.volume").expect("checked volume pass");
        let surface = position("water.draw").expect("checked water draw");
        let depth = graph.passes[surface].intents.iter().find(|i| i.role == Role::DepthRead);
        if !depth.is_some_and(|d| {
            graph.passes[volume_index]
                .intents
                .iter()
                .any(|i| i.resource == d.resource && i.role == Role::Sampled(ShaderStage::Fragment))
        }) {
            return Err(error("`water.volume` must sample the water draw's scene depth".into()));
        }

        let accesses = |index, name: &str, role| {
            let pass: &ae_framegraph::Pass<R> = &graph.passes[index];
            pass.intents
                .iter()
                .any(|i| graph.resources[i.resource.index()].name == name && i.role == role)
        };
        if waves && !accesses(volume_index, WATER_WAVES_NAME, Role::Sampled(ShaderStage::Fragment)) {
            return Err(error(format!("`water.volume` must sample `{WATER_WAVES_NAME}`")));
        }
        for (pass, stage) in [
            ("exposure.meter", ShaderStage::Compute),
            ("post.tonemap", ShaderStage::Fragment),
        ] {
            if let Some(index) = position(pass) {
                before("water.volume", pass, WATER_VOLUME_OUT_NAME)?;
                if !accesses(index, WATER_VOLUME_OUT_NAME, Role::Sampled(stage)) {
                    return Err(error(format!("`{pass}` must sample `{WATER_VOLUME_OUT_NAME}`")));
                }
            }
        }
        for writer in ["units.water_overlay", "particles.draw"] {
            if position(writer).is_some() {
                before(writer, "water.volume", WATER_VOLUME_OUT_NAME)?;
            }
        }
    }
    Ok(WaterTiers {
        reflect,
        refract,
        waves,
        volume,
    })
}

/// The pre-water scene alpha is an internal unit marker consumed by underwater.
fn check_refraction<'a, R>(
    graph: &FrameGraph<R>,
    env: &LibraryEnv,
    libs: &impl Fn(usize) -> Option<&'a str>,
) -> Result<(), LibraryError> {
    let pass = |name| {
        let mut found = (0..graph.passes.len()).filter(|&i| libs(i) == Some(name));
        let first = found.next();
        first
            .filter(|&i| found.next().is_none() && !graph.passes[i].conditional)
            .ok_or_else(|| error(format!("`{WATER_REFRACTION_NAME}` requires one unconditional `{name}`")))
    };
    let underwater = pass("water.underwater")?;
    let water = pass("water.draw")?;
    let particles = (0..graph.passes.len())
        .any(|i| libs(i) == Some("particles.underwater"))
        .then(|| pass("particles.underwater"))
        .transpose()?;
    let output = graph
        .resources
        .iter()
        .position(|r| r.name == WATER_REFRACTION_NAME)
        .unwrap();
    if !graph.passes[underwater]
        .intents
        .iter()
        .any(|i| i.resource.index() == output && i.role == Role::ColorAttachment(ae_framegraph::LoadOp::Clear))
        || graph.passes.iter().enumerate().any(|(at, pass)| {
            at != underwater
                && Some(at) != particles
                && pass
                    .intents
                    .iter()
                    .any(|i| i.resource.index() == output && i.role.usage().write)
        })
    {
        return Err(error(format!(
            "`water.underwater` must clear `{WATER_REFRACTION_NAME}` and only `particles.underwater` may draw over it"
        )));
    }
    let terrain = pass("terrain.draw")?;
    let scene = graph.passes[underwater]
        .intents
        .iter()
        .find(|i| {
            graph.resources[i.resource.index()].name == ae_render_abi::module::SCENE_NAME
                && i.role == Role::Sampled(ShaderStage::Fragment)
        })
        .map(|i| i.resource)
        .ok_or_else(|| error("`water.underwater` must sample `scene`".into()))?;
    let depth = graph.passes[water]
        .intents
        .iter()
        .find(|i| i.role == Role::DepthRead)
        .map(|i| i.resource);
    if particles.is_some_and(|index| {
        let uses = |resource, role| {
            graph.passes[index]
                .intents
                .iter()
                .any(|i| Some(i.resource) == resource && i.role == role)
        };
        !uses(
            Some(ResourceId(output as u32)),
            Role::ColorAttachment(ae_framegraph::LoadOp::Load),
        ) || !uses(depth, Role::Sampled(ShaderStage::Fragment))
    }) {
        return Err(error(format!(
            "`particles.underwater` must load `{WATER_REFRACTION_NAME}` and sample the scene depth"
        )));
    }
    let writes_scene = |index: usize| {
        graph.passes[index]
            .intents
            .iter()
            .any(|i| i.resource == scene && matches!(i.role, Role::ColorAttachment(_)))
    };
    let writes_depth = |index: usize| {
        graph.passes[index]
            .intents
            .iter()
            .any(|i| Some(i.resource) == depth && matches!(i.role, Role::DepthAttachment(_)))
    };
    if terrain >= underwater
        || !writes_scene(terrain)
        || !writes_depth(terrain)
        || !writes_scene(water)
        || !graph.passes[underwater]
            .intents
            .iter()
            .any(|i| Some(i.resource) == depth && i.role == Role::Sampled(ShaderStage::Fragment))
    {
        return Err(error(
            "terrain, water and `water.underwater` must share the scene and depth".into(),
        ));
    }
    if env.particles && pass("particles.sim")? >= water {
        return Err(error("`particles.sim` must run before refractive water".into()));
    }
    if env.units {
        let units = pass("units.draw")?;
        let overlay = pass("units.water_overlay")?;
        if !(terrain < units
            && units < underwater
            && water < overlay
            && writes_scene(units)
            && writes_scene(overlay)
            && writes_depth(units)
            && writes_depth(overlay))
        {
            return Err(error(
                "refractive water needs `units.draw` before the copy and `units.water_overlay` after water".into(),
            ));
        }
        if graph.passes[units + 1..underwater]
            .iter()
            .any(|pass| pass.intents.iter().any(|i| i.resource == scene))
        {
            return Err(error(
                "`scene` alpha is reserved between `units.draw` and `water.underwater`".into(),
            ));
        }
    }
    Ok(())
}

/// A tier's resources and library programs form a single contract. Pass
/// labels are display names. Only a library program can own an effect.
#[allow(clippy::too_many_arguments)]
fn check_tier<'a, R>(
    graph: &FrameGraph<R>,
    water: bool,
    libs: &impl Fn(usize) -> Option<&'a str>,
    passes: &[&str],
    images: &[&str],
    buffers: &[&str],
    producer: &str,
    consumer: &str,
) -> Result<bool, LibraryError> {
    let output = images[0];
    let resources = images
        .iter()
        .map(|&name| (name, true))
        .chain(buffers.iter().map(|&name| (name, false)));
    if !(0..graph.passes.len()).any(|i| libs(i).is_some_and(|name| passes.contains(&name)))
        && !resources
            .clone()
            .any(|(name, _)| graph.resources.iter().any(|r| r.name == name))
    {
        return Ok(false);
    }
    if !water {
        return Err(error(format!(
            "pass `{producer}` and resource `{output}` require a scene with water"
        )));
    }

    let find_pass = |name: &str| {
        let mut matches = (0..graph.passes.len()).filter(|&i| libs(i) == Some(name));
        let i = matches
            .next()
            .ok_or_else(|| error(format!("`{output}` requires library pass `{name}`")))?;
        if matches.next().is_some() || graph.passes[i].conditional {
            return Err(error(format!("`{output}` requires one unconditional `{name}` pass")));
        }
        Ok(i)
    };
    let find_resource = |name: &str, image: bool| {
        let mut matches = graph.resources.iter().enumerate().filter(|(_, r)| r.name == name);
        let (i, r) = matches
            .next()
            .ok_or_else(|| error(format!("pass `{producer}` requires resource `{name}`")))?;
        if matches.next().is_some() || r.desc.is_image() != image {
            let kind = if image { "image" } else { "buffer" };
            return Err(error(format!("pass `{producer}` requires one `{name}` {kind}")));
        }
        Ok(ResourceId(i as u32))
    };

    for (name, image) in resources {
        find_resource(name, image)?;
    }
    let consume = find_pass(consumer)?;
    for &name in passes {
        if find_pass(name)? >= consume {
            return Err(error(format!(
                "pass `{name}` for `{output}` must run before `{consumer}`"
            )));
        }
    }

    let output_id = find_resource(output, true)?;
    let write = find_pass(producer)?;
    if !graph.passes[write]
        .intents
        .iter()
        .any(|i| i.resource == output_id && i.role.usage().write)
    {
        return Err(error(format!("pass `{producer}` must write `{output}`")));
    }
    // The displacement is read from the vertex shader. The colour copies
    // are read from the fragment shader.
    let stage = if output == WATER_WAVES_NAME {
        ShaderStage::Graphics
    } else {
        ShaderStage::Fragment
    };
    if !graph.passes[consume].intents.iter().any(|i| {
        i.resource == output_id && matches!(i.role, Role::Sampled(s) if s == stage || s == ShaderStage::Graphics)
    }) {
        return Err(error(format!(
            "pass `{consumer}` must sample `{output}` in the {stage:?} stage"
        )));
    }
    Ok(true)
}
