//! M15 step D's guard: outside the files that still own descriptor sets, no
//! renderer source creates, allocates or writes a set of its own, and no
//! shader declares a numbered set. Everything else reads the global set
//! (`src/bindless.rs`, `shaders/bindless.glsl`).
//!
//! Still on their own sets, until the steps that move them:
//! - `fullscreen.rs`: the module `FullscreenProgram` (the render-module
//!   contract, step F) and the engine `BindlessProgram`'s uniform block at
//!   set 0 (tonemap, bloom, encode; also step F), with their shaders;
//! - `demo.rs` and its shader (step E);
//! - `upload.rs`'s mip generator, which stays as it is.

use std::fs;
use std::path::Path;

/// Sources allowed to own sets: the registry itself, the pipeline helpers,
/// and the files listed above.
const SOURCE_EXCEPTIONS: &[&str] = &["bindless.rs", "pipeline.rs", "fullscreen.rs", "demo.rs", "upload.rs"];

/// Shaders still declaring numbered sets (see the module docs).
const SHADER_EXCEPTIONS: &[&str] = &[
    "bloom_downsample.frag",
    "composite.frag",
    "demo_filter.comp",
    "encode.frag",
    "mipgen.comp",
    "test_post.frag",
    "test_post_storage.frag",
    "tonemap.frag",
];

fn files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn name(path: &Path) -> &str {
    path.file_name().unwrap().to_str().unwrap()
}

#[test]
fn only_the_global_set_outside_the_exceptions() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = Vec::new();
    files(&root.join("src"), &mut sources);
    let mut found = Vec::new();
    for path in sources.iter().filter(|p| p.extension().is_some_and(|e| e == "rs")) {
        if SOURCE_EXCEPTIONS.contains(&name(path)) {
            continue;
        }
        let text = fs::read_to_string(path).unwrap();
        for needle in [
            "create_descriptor_set_layout",
            "pipeline::set_layout(",
            "allocate_descriptor_sets",
            "update_descriptor_sets",
            "create_descriptor_pool",
        ] {
            if text.contains(needle) {
                found.push(format!("{}: {needle}", path.display()));
            }
        }
    }
    let mut shaders = Vec::new();
    files(&root.join("shaders"), &mut shaders);
    let numbered = |line: &str| {
        let line = line.split("//").next().unwrap_or("");
        line.contains("layout(")
            && line
                .split("set = ")
                .nth(1)
                .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
    };
    for path in &shaders {
        if SHADER_EXCEPTIONS.contains(&name(path)) {
            continue;
        }
        let text = fs::read_to_string(path).unwrap();
        for (n, line) in text.lines().enumerate() {
            if numbered(line) {
                found.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
            }
        }
    }
    assert!(
        found.is_empty(),
        "per-pass descriptor sets outside the exceptions:\n{}",
        found.join("\n")
    );
}
