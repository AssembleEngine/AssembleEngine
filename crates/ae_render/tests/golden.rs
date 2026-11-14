//! Golden frame graph `.dot` output: the baked plans of the engine's own
//! graphs (the terrain scene as the client runs it, with and without
//! units, presenting and reading back; the frame graph demo) are compared
//! with the files in `tests/golden/`. A change to the bake (order, edges,
//! aliasing) or to a native graph shows up as a diff in review.
//!
//! No GPU needed. After an intended change, regenerate with
//! `AE_UPDATE_GOLDEN=1 cargo test -p ae_render --test golden`.

use ae_framegraph::{BakeOptions, BufferDesc, FrameGraph, bake, dot};
use ae_render::demo::{DemoOutput, build_demo_graph};
use ae_render::library::{LibraryEnv, LibraryResource, resolve};
use ae_render::terrain::app::{TerrainOutput, native_graph};
use ash::vk;
use std::path::PathBuf;

fn dot_of<R>(graph: &FrameGraph<R>) -> String {
    let baked = bake(graph, BakeOptions::default()).expect("the graph bakes");
    dot::to_dot(graph, &baked)
}

fn scene(units: bool, output: TerrainOutput) -> String {
    let env = LibraryEnv { color_format: vk::Format::B8G8R8A8_SRGB, depth_format: vk::Format::D32_SFLOAT, extent: vk::Extent2D { width: 1280, height: 720 }, units };
    let mut graph = native_graph(&env, output);
    // Fixed library sizes (the renderers' real ones depend on the map), with
    // every usage a library buffer may need.
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    resolve(&mut graph, &env, |r: LibraryResource| Some(BufferDesc { size: 1024 * (1 + r as u64), usage })).expect("library resources resolve");
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
    let cases: [(&str, fn() -> String); 5] = [
        ("scene_units_present", || scene(true, TerrainOutput::Present)),
        ("scene_units_readback", || scene(true, TerrainOutput::Readback)),
        ("scene_present", || scene(false, TerrainOutput::Present)),
        ("demo_present", || dot_of(&build_demo_graph(vk::Extent2D { width: 1280, height: 720 }, DemoOutput::Present { format: vk::Format::B8G8R8A8_SRGB }).0)),
        ("demo_readback", || dot_of(&build_demo_graph(vk::Extent2D { width: 256, height: 256 }, DemoOutput::Readback).0)),
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
