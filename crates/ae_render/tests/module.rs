//! Render modules without a GPU: the in-repo default module declares the
//! same graph as the engine's native one, in SDR and in both HDR encodings;
//! broken modules (including one built for the old ABI) are rejected with
//! errors that say what is wrong, and the display rule is enforced on
//! module graphs.

use ae_framegraph::{BakeOptions, BufferDesc, FrameGraph, bake};
use ae_render::library::{LibraryEnv, LibraryPass, LibraryResource, check, resolve};
use ae_render::encode::{DisplayLevels, OutputEncoding};
use ae_render::module::{DeclareTarget, RenderModule, build, engine};
use ae_render::terrain::app::{TerrainOutput, native_graph};
use ae_render_abi::module::{DECLARE_PARTICLES, DECLARE_READBACK, DECLARE_SHADOWS, DECLARE_UNITS, DECLARE_WATER, FrameInput, Program, ProgramTable};
use ash::vk;

fn size(r: LibraryResource) -> Option<BufferDesc> {
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    Some(BufferDesc { size: 256 + 16 * r as u64, usage })
}

/// Resolve sizes and check every library pass; payloads become library
/// passes through `program`.
fn wire<R>(mut graph: FrameGraph<R>, env: &LibraryEnv, program: impl Fn(&R) -> LibraryPass) -> FrameGraph<LibraryPass> {
    let map = resolve(&mut graph, env, size).unwrap();
    let graph = FrameGraph {
        passes: graph.passes.into_iter().map(|p| ae_framegraph::Pass { record: program(&p.record), name: p.name, kind: p.kind, scope: p.scope, intents: p.intents, conditional: p.conditional }).collect(),
        resources: graph.resources,
        roots: graph.roots,
        plan_version: graph.plan_version,
    };
    for (i, p) in graph.passes.iter().enumerate() {
        check(&graph, i, p.record, &map, env).unwrap_or_else(|e| panic!("{e}"));
    }
    graph
}

/// The encode target's format and colour space stand-in for `encoding`.
fn target_for(width: u32, height: u32, units: bool, readback: bool, water: bool, shadows: bool, particles: bool, encoding: OutputEncoding, levels: DisplayLevels) -> DeclareTarget {
    let flags = if units { DECLARE_UNITS } else { 0 }
        | if readback { DECLARE_READBACK } else { 0 }
        | if water { DECLARE_WATER } else { 0 }
        | if shadows { DECLARE_SHADOWS } else { 0 }
        | if particles { DECLARE_PARTICLES } else { 0 };
    DeclareTarget { width, height, color_format: vk::Format::B8G8R8A8_SRGB, flags, encoding, levels, working: Default::default(), gamut: encoding.default_gamut() }
}

#[test]
fn the_default_module_declares_the_native_graph() {
    let wasm = build("default").expect("build render/default");
    let engine = engine();
    let mut module = RenderModule::load(&engine, &wasm).unwrap();
    // (units, shadows): shadows only with units.
    let scenes = [(true, false), (true, true), (false, false)]
        .into_iter()
        .flat_map(|(units, shadows)| [true, false].into_iter().flat_map(move |water| [TerrainOutput::Present, TerrainOutput::Readback].map(|output| (units, shadows, water, output))))
        .flat_map(|scene| [false, true].map(|particles| (scene, particles)));
    let levels = DisplayLevels { paper_white: 203.0, peak: 361.0 };
    for encoding in [OutputEncoding::Sdr, OutputEncoding::Hdr10Pq, OutputEncoding::ScRgb] {
        for ((units, shadows, water, output), particles) in scenes.clone() {
            let env = LibraryEnv { color_format: vk::Format::B8G8R8A8_SRGB, scene_format: ae_render::terrain::app::SCENE_FORMAT, depth_format: vk::Format::D32_SFLOAT, extent: vk::Extent2D { width: 1280, height: 720 }, units, water, shadows, particles, encoding, levels, working: Default::default(), gamut: encoding.default_gamut() };
            let target = target_for(1280, 720, units, output == TerrainOutput::Readback, water, shadows, particles, encoding, levels);
            let declared = module.declare(&target).unwrap().expect("the module renders this target");
            let programs = ProgramTable::decode(&declared.programs).unwrap();
            let graph = ae_framegraph::descriptor::decode(&declared.graph).unwrap();
            let from_module = wire(graph, &env, |&payload| match programs[payload as usize] {
                Program::Library(name) => LibraryPass::from_name(name).unwrap(),
                other => panic!("{other:?}"),
            });
            let native = wire(native_graph(&env, output), &env, |&p| p);
            assert_eq!(from_module.signature(), native.signature(), "{encoding:?}: units {units}, shadows {shadows}, water {water}, particles {particles}, {output:?}: same topology");
            assert_eq!(bake(&from_module, BakeOptions::default()).unwrap(), bake(&native, BakeOptions::default()).unwrap());
            let libs = |g: &FrameGraph<LibraryPass>| g.passes.iter().map(|p| p.record).collect::<Vec<_>>();
            assert_eq!(libs(&from_module), libs(&native), "{encoding:?}: the same library pass in every position");
        }
    }
    // The per-frame call runs (the default module asks for nothing).
    let frame = module.frame(&FrameInput { width: 1280, height: 720, ..FrameInput::default() }).unwrap();
    assert!(frame.disabled.is_empty() && frame.uniforms.is_empty());
}

/// A module in WAT with the four exports; `declare_body` and `frame_body`
/// are the functions' bodies (each returns an i32).
fn wat(version: u32, declare_body: &str, frame_body: &str, extra: &str) -> String {
    format!(
        r#"(module
            {extra}
            (memory (export "memory") 1)
            (func (export "ae_render_version") (result i32) i32.const {version})
            (func (export "ae_render_declare_input") (result i32) i32.const 512)
            (func (export "ae_render_declare") (result i32) {declare_body})
            (func (export "ae_render_input") (result i32) i32.const 1024)
            (func (export "ae_render_frame") (result i32) {frame_body}))"#
    )
}

/// The v1 contract: `declare` took the four scalars and there was no
/// declaration input.
fn wat_v1() -> String {
    r#"(module
        (memory (export "memory") 1)
        (func (export "ae_render_version") (result i32) i32.const 1)
        (func (export "ae_render_declare") (param i32 i32 i32 i32) (result i32) i32.const 0)
        (func (export "ae_render_input") (result i32) i32.const 1024)
        (func (export "ae_render_frame") (result i32) i32.const 0))"#
        .to_string()
}

#[test]
fn broken_modules_are_rejected() {
    let engine = engine();
    let load = |text: &str| RenderModule::load(&engine, text.as_bytes()).map(|_| ()).map_err(|e| format!("{e:#}"));
    assert!(load("not wasm").unwrap_err().contains("compiling"));
    let e = load(&wat(1, "i32.const 0", "i32.const 0", r#"(import "env" "f" (func))"#)).unwrap_err();
    assert!(e.contains("import nothing") && e.contains("env::f"), "{e}");
    // An old-ABI module is refused by number, before any other export is
    // read (its `declare` has the old signature).
    let e = load(&wat_v1()).unwrap_err();
    assert_eq!(e, "render module ABI version 1; this host runs 2 (rebuild the module)");
    let e = load(&wat(7, "i32.const 0", "i32.const 0", "")).unwrap_err();
    assert!(e.contains("version 7") && e.contains("runs 2"), "{e}");
    let e = load(r#"(module (memory (export "memory") 1) (func (export "ae_render_version") (result i32) i32.const 2))"#).unwrap_err();
    assert!(e.contains("ae_render_declare_input"), "{e}");

    let target = |width, height, format| DeclareTarget { width, height, color_format: format, flags: 0, encoding: OutputEncoding::Sdr, levels: DisplayLevels::default(), working: Default::default(), gamut: Default::default() };
    // Loads, but hands back pointers outside its memory, or traps.
    let mut m = RenderModule::load(&engine, wat(2, "i32.const 70000", "unreachable", "").as_bytes()).unwrap();
    let e = format!("{:#}", m.declare(&target(8, 8, vk::Format::B8G8R8A8_SRGB)).unwrap_err());
    assert!(e.contains("outside the module's memory"), "{e}");
    let e = format!("{:#}", m.frame(&FrameInput::default()).unwrap_err());
    assert!(e.contains("ae_render_frame") || e.contains("unreachable"), "{e}");
    // `declare` returning 0 declines the target.
    let mut m = RenderModule::load(&engine, wat(2, "i32.const 0", "i32.const 0", "").as_bytes()).unwrap();
    assert_eq!(m.declare(&target(8, 8, vk::Format::B8G8R8A8_SRGB)).unwrap(), None);
    // A graph pointer in bounds but garbage behind it reaches the decoder,
    // which rejects it (not this layer's job; checked here end to end).
    let mut m = RenderModule::load(&engine, wat(2, "i32.const 16", "i32.const 0", r#"(data (i32.const 16) "\40\00\00\00\08\00\00\00\40\00\00\00\08\00\00\00")"#).as_bytes()).unwrap();
    let declared = m.declare(&target(8, 8, vk::Format::B8G8R8A8_SRGB)).unwrap().unwrap();
    assert!(ae_framegraph::descriptor::decode(&declared.graph).is_err());
    assert!(ProgramTable::decode(&declared.programs).is_err());
}

/// The display rule is enforced on module graphs: `post.encode` exactly
/// when the output is HDR, named with the encoding and the pass.
#[test]
fn the_display_rule_is_enforced_on_module_graphs() {
    use ae_render::library::{LibraryEnv, check_graph};
    use ae_framegraph::{FrameGraph, ImageDesc, PassKind, ResourceClass};
    use ash::vk;

    let env = |encoding| LibraryEnv {
        color_format: vk::Format::B8G8R8A8_SRGB,
        scene_format: ae_render::terrain::app::SCENE_FORMAT,
        depth_format: vk::Format::D32_SFLOAT,
        extent: vk::Extent2D { width: 64, height: 64 },
        units: false,
        water: false,
        shadows: false,
        particles: false,
        encoding,
        levels: DisplayLevels::default(),
        working: Default::default(),
        gamut: Default::default(),
    };
    let mut g = FrameGraph::<()>::new();
    let target = g.image("target", ImageDesc::color_2d(vk::Format::B8G8R8A8_SRGB, 64, 64, vk::ImageUsageFlags::COLOR_ATTACHMENT), ResourceClass::Backbuffer);
    g.pass("present", PassKind::Present, ()).present(target).root().id();
    let map = Vec::new();
    let e = check_graph(&g, &map, &env(OutputEncoding::Hdr10Pq), |_| None).unwrap_err().to_string();
    assert_eq!(e, "the output is HDR10 (PQ) but the graph has no `post.encode` pass: the tonemap writes the display, not the target");
    let e = check_graph(&g, &map, &env(OutputEncoding::ScRgb), |_| None).unwrap_err().to_string();
    assert!(e.starts_with("the output is scRGB but the graph has no `post.encode` pass"), "{e}");
    let e = check_graph(&g, &map, &env(OutputEncoding::Sdr), |_| Some(LibraryPass::PostEncode)).unwrap_err().to_string();
    assert_eq!(e, "the output is SDR but the graph has an extra `post.encode` pass: the tonemap writes the target");

}
