//! Golden frame graph `.dot` output: the baked plans of the engine's own
//! graphs (the terrain scene as the client runs it, with and without
//! units, water and the units' shadows, presenting and reading back; the
//! HDR10 and scRGB tails natively and from both in-repo render modules; the
//! frame graph demo) are compared with the files in `tests/golden/`. A
//! change to the bake (order, edges, aliasing) or to a graph shows up as a
//! diff in review.
//!
//! No GPU needed. After an intended change, regenerate with
//! `AE_UPDATE_GOLDEN=1 cargo test -p ae_render --test golden`.

use ae_framegraph::{BakeOptions, BufferDesc, FrameGraph, bake, dot};
use ae_render::demo::{DemoOutput, build_demo_graph};
use ae_render::encode::{DisplayLevels, OutputEncoding};
use ae_render::library::{LibraryEnv, LibraryPass, LibraryResource, resolve};
use ae_render::module::{DeclareTarget, RenderModule, build, engine};
use ae_render::terrain::app::{TerrainOutput, native_graph};
use ae_render_abi::module::{DECLARE_PARTICLES, DECLARE_SHADOWS, DECLARE_UNITS, DECLARE_WATER, Program, ProgramTable};
use ash::vk;
use std::path::PathBuf;

fn dot_of<R>(graph: &FrameGraph<R>) -> String {
    let baked = bake(graph, BakeOptions::default()).expect("the graph bakes");
    dot::to_dot(graph, &baked)
}

fn scene(units: bool, water: bool, shadows: bool, output: TerrainOutput) -> String {
    scene_with(units, water, shadows, false, output)
}

const EXTENT: vk::Extent2D = vk::Extent2D { width: 1280, height: 720 };

/// The levels the client sends (M13 stage 5): 203 nit paper white, 361 nit
/// peak.
const LEVELS: DisplayLevels = DisplayLevels { paper_white: 203.0, peak: 361.0 };

fn env_with(units: bool, water: bool, shadows: bool, particles: bool, encoding: OutputEncoding) -> LibraryEnv {
    LibraryEnv {
        color_format: vk::Format::B8G8R8A8_SRGB,
        scene_format: ae_render::terrain::app::SCENE_FORMAT,
        depth_format: vk::Format::D32_SFLOAT,
        extent: EXTENT,
        units,
        water,
        shadows,
        particles,
        encoding,
        levels: LEVELS,
        working: Default::default(),
        gamut: encoding.default_gamut(),
    }
}

/// Fixed library sizes (the renderers' real ones depend on the map), with
/// every usage a library buffer may need.
fn resolve_libraries<R>(graph: &mut FrameGraph<R>, env: &LibraryEnv) {
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    resolve(graph, env, |r: LibraryResource| Some(BufferDesc { size: 1024 * (1 + r as u64), usage })).expect("library resources resolve");
}

fn scene_with(units: bool, water: bool, shadows: bool, particles: bool, output: TerrainOutput) -> String {
    let env = env_with(units, water, shadows, particles, OutputEncoding::Sdr);
    let mut graph = native_graph(&env, output);
    resolve_libraries(&mut graph, &env);
    dot_of(&graph)
}

/// The native graph with the full scene (units, water, shadows, particles),
/// presenting, for an HDR encoding.
fn scene_hdr(encoding: OutputEncoding) -> String {
    let env = env_with(true, true, true, true, encoding);
    let mut graph = native_graph(&env, TerrainOutput::Present);
    resolve_libraries(&mut graph, &env);
    dot_of(&graph)
}

/// An in-repo module's graph with the full scene, presenting, for an HDR
/// encoding: the same plan the client bakes with `--render-module`.
fn module_hdr(name: &str, encoding: OutputEncoding) -> String {
    let env = env_with(true, true, true, true, encoding);
    let flags = DECLARE_UNITS | DECLARE_WATER | DECLARE_SHADOWS | DECLARE_PARTICLES;
    let target = DeclareTarget { width: EXTENT.width, height: EXTENT.height, color_format: env.color_format, flags, encoding, levels: LEVELS, working: env.working, gamut: env.gamut };
    let wasm = build(name).unwrap_or_else(|e| panic!("build render/{name}: {e:#}"));
    let mut module = RenderModule::load(&engine(), &wasm).expect("load the module");
    let declared = module.declare(&target).expect("declare").expect("the module renders this target");
    let programs = ProgramTable::decode(&declared.programs).expect("program table");
    let graph = ae_framegraph::descriptor::decode(&declared.graph).expect("graph descriptor");
    let mut graph = FrameGraph {
        passes: graph
            .passes
            .into_iter()
            .map(|p| ae_framegraph::Pass {
                record: match programs[p.record as usize] {
                    Program::Library(name) => Some(LibraryPass::from_name(name).unwrap_or_else(|| panic!("library pass `{name}`"))),
                    Program::Fullscreen { .. } => None,
                },
                name: p.name,
                kind: p.kind,
                scope: p.scope,
                intents: p.intents,
                conditional: p.conditional,
            })
            .collect(),
        resources: graph.resources,
        roots: graph.roots,
        plan_version: graph.plan_version,
    };
    resolve_libraries(&mut graph, &env);
    dot_of(&graph)
}

/// Compare with (or, with `AE_UPDATE_GOLDEN=1`, write) `tests/golden/<name>.dot`.
fn check(name: &str, dot: &str) -> Option<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(format!("{name}.dot"));
    if std::env::var("AE_UPDATE_GOLDEN").is_ok_and(|v| v == "1") {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, dot).unwrap();
        return None;
    }
    match std::fs::read_to_string(&path) {
        Ok(want) if want == dot => None,
        Ok(want) => {
            let first = want.lines().zip(dot.lines()).position(|(a, b)| a != b).unwrap_or(want.lines().count().min(dot.lines().count()));
            Some(format!(
                "{name}: differs from {} at line {}:\n  golden: {}\n  now:    {}",
                path.display(),
                first + 1,
                want.lines().nth(first).unwrap_or("<end>"),
                dot.lines().nth(first).unwrap_or("<end>")
            ))
        }
        Err(_) => Some(format!("{name}: {} is missing", path.display())),
    }
}

#[test]
fn baked_plans_match_the_golden_dot_files() {
    let cases: [(&str, fn() -> String); 16] = [
        ("scene_units_present", || scene(true, false, false, TerrainOutput::Present)),
        ("scene_units_readback", || scene(true, false, false, TerrainOutput::Readback)),
        ("scene_present", || scene(false, false, false, TerrainOutput::Present)),
        ("scene_units_water_present", || scene(true, true, false, TerrainOutput::Present)),
        ("scene_water_readback", || scene(false, true, false, TerrainOutput::Readback)),
        ("scene_units_shadows_present", || scene(true, false, true, TerrainOutput::Present)),
        ("scene_units_water_shadows_readback", || scene(true, true, true, TerrainOutput::Readback)),
        ("scene_units_water_shadows_particles_present", || scene_with(true, true, true, true, TerrainOutput::Present)),
        ("demo_present", || dot_of(&build_demo_graph(vk::Extent2D { width: 1280, height: 720 }, DemoOutput::Present { format: vk::Format::B8G8R8A8_SRGB }).0)),
        ("demo_readback", || dot_of(&build_demo_graph(vk::Extent2D { width: 256, height: 256 }, DemoOutput::Readback).0)),
        ("scene_hdr10_present", || scene_hdr(OutputEncoding::Hdr10Pq)),
        ("scene_scrgb_present", || scene_hdr(OutputEncoding::ScRgb)),
        ("default_hdr10_present", || module_hdr("default", OutputEncoding::Hdr10Pq)),
        ("default_scrgb_present", || module_hdr("default", OutputEncoding::ScRgb)),
        ("post_hdr10_present", || module_hdr("post", OutputEncoding::Hdr10Pq)),
        ("post_scrgb_present", || module_hdr("post", OutputEncoding::ScRgb)),
    ];
    let mut failures = Vec::new();
    for (name, make) in cases {
        let dot = make();
        assert!(dot.starts_with("digraph frame_graph {") && dot.contains(" -> "), "{name}: a plan with edges");
        assert_eq!(dot, make(), "{name}: building and exporting again gives the same text");
        failures.extend(check(name, &dot));
    }
    assert!(failures.is_empty(), "{}\n(after an intended change: AE_UPDATE_GOLDEN=1 cargo test -p ae_render --test golden)", failures.join("\n"));
}
