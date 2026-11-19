//! Render modules without a GPU: the in-repo default module declares the
//! same graph as the engine's native one; broken modules are rejected with
//! errors that say what is wrong.

use ae_framegraph::{BakeOptions, BufferDesc, FrameGraph, bake};
use ae_render::library::{LibraryEnv, LibraryPass, LibraryResource, check, resolve};
use ae_render::module::{RenderModule, build, engine};
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
    for ((units, shadows, water, output), particles) in scenes {
        let env = LibraryEnv { color_format: vk::Format::B8G8R8A8_SRGB, scene_format: ae_render::terrain::app::SCENE_FORMAT, depth_format: vk::Format::D32_SFLOAT, extent: vk::Extent2D { width: 1280, height: 720 }, units, water, shadows, particles };
        let flags = if units { DECLARE_UNITS } else { 0 }
            | if output == TerrainOutput::Readback { DECLARE_READBACK } else { 0 }
            | if water { DECLARE_WATER } else { 0 }
            | if shadows { DECLARE_SHADOWS } else { 0 }
            | if particles { DECLARE_PARTICLES } else { 0 };
        let declared = module.declare(1280, 720, env.color_format.as_raw() as u32, flags).unwrap().expect("the module renders this target");
        let programs = ProgramTable::decode(&declared.programs).unwrap();
        let graph = ae_framegraph::descriptor::decode(&declared.graph).unwrap();
        let from_module = wire(graph, &env, |&payload| match programs[payload as usize] {
            Program::Library(name) => LibraryPass::from_name(name).unwrap(),
            other => panic!("{other:?}"),
        });
        let native = wire(native_graph(&env, output), &env, |&p| p);
        assert_eq!(from_module.signature(), native.signature(), "units {units}, shadows {shadows}, water {water}, particles {particles}, {output:?}: same topology");
        assert_eq!(bake(&from_module, BakeOptions::default()).unwrap(), bake(&native, BakeOptions::default()).unwrap());
        let libs = |g: &FrameGraph<LibraryPass>| g.passes.iter().map(|p| p.record).collect::<Vec<_>>();
        assert_eq!(libs(&from_module), libs(&native), "the same library pass in every position");
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
            (func (export "ae_render_declare") (param i32 i32 i32 i32) (result i32) {declare_body})
            (func (export "ae_render_input") (result i32) i32.const 1024)
            (func (export "ae_render_frame") (result i32) {frame_body}))"#
    )
}

#[test]
fn broken_modules_are_rejected() {
    let engine = engine();
    let load = |text: &str| RenderModule::load(&engine, text.as_bytes()).map(|_| ()).map_err(|e| format!("{e:#}"));
    assert!(load("not wasm").unwrap_err().contains("compiling"));
    let e = load(&wat(1, "i32.const 0", "i32.const 0", r#"(import "env" "f" (func))"#)).unwrap_err();
    assert!(e.contains("import nothing") && e.contains("env::f"), "{e}");
    let e = load(&wat(7, "i32.const 0", "i32.const 0", "")).unwrap_err();
    assert!(e.contains("version 7"), "{e}");
    let e = load(r#"(module (memory (export "memory") 1) (func (export "ae_render_version") (result i32) i32.const 1))"#).unwrap_err();
    assert!(e.contains("ae_render_declare"), "{e}");

    // Loads, but hands back pointers outside its memory, or traps.
    let mut m = RenderModule::load(&engine, wat(1, "i32.const 70000", "unreachable", "").as_bytes()).unwrap();
    let e = format!("{:#}", m.declare(8, 8, 50, 0).unwrap_err());
    assert!(e.contains("outside the module's memory"), "{e}");
    let e = format!("{:#}", m.frame(&FrameInput::default()).unwrap_err());
    assert!(e.contains("ae_render_frame") || e.contains("unreachable"), "{e}");
    // `declare` returning 0 declines the target.
    let mut m = RenderModule::load(&engine, wat(1, "i32.const 0", "i32.const 0", "").as_bytes()).unwrap();
    assert_eq!(m.declare(8, 8, 50, 0).unwrap(), None);
    // A graph pointer in bounds but garbage behind it reaches the decoder,
    // which rejects it (not this layer's job; checked here end to end).
    let mut m = RenderModule::load(&engine, wat(1, "i32.const 16", "i32.const 0", r#"(data (i32.const 16) "\40\00\00\00\08\00\00\00\40\00\00\00\08\00\00\00")"#).as_bytes()).unwrap();
    let declared = m.declare(8, 8, 50, 0).unwrap().unwrap();
    assert!(ae_framegraph::descriptor::decode(&declared.graph).is_err());
    assert!(ProgramTable::decode(&declared.programs).is_err());
}
