//! M15 step E's guard: outside the files that still own descriptor sets, no
//! renderer source creates, allocates or writes a set of its own, no shader
//! declares a numbered set, and no shader moves the global set off 0.
//! No source outside them makes a pipeline layout either: every pass uses
//! `Bindless::pipeline_layout`.
//! Everything else reads the global set (`src/bindless.rs`,
//! `shaders/bindless.glsl`) at set 0.
//!
//! Samplers are the registry's too: `Bindless::sampler` creates one per
//! distinct create info and shares it, so no other source creates or
//! destroys a sampler (the mip generator's own, in `upload.rs`, stays).
//!
//! Uniform buffers (binding 4, the former `uniform` blocks) are the
//! registry's too: `Bindless::uniform_buffer` and its slotted and per-slot
//! forms are the only makers of a `UNIFORM_BUFFER` descriptor outside the
//! per-pass form in `fullscreen.rs`.
//!
//! Still on their own sets, by design:
//! - `fullscreen.rs`: the per-pass form a contract-2 module's program may
//!   still use (M15 F keeps it for one version), and its reflection
//!   fixtures (`test_post.frag`, `test_post_storage.frag`);
//! - `upload.rs`'s mip generator, which stays as it is.

use std::fs;
use std::path::Path;

/// Sources allowed to own sets: the registry itself, the pipeline helpers,
/// and the files listed above.
const SOURCE_EXCEPTIONS: &[&str] = &["bindless.rs", "pipeline.rs", "fullscreen.rs", "upload.rs"];

/// Shaders still declaring numbered sets or pinning the global set off 0
/// (see the module docs).
const SHADER_EXCEPTIONS: &[&str] = &[
    "mipgen.comp",
    "test_mixed.frag",
    "test_post.frag",
    "test_post_storage.frag",
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
            // One shared pipeline layout (`Bindless::pipeline_layout`).
            "pipeline::pipeline_layout(",
            "create_pipeline_layout",
            // One sampler object per distinct create info.
            "create_sampler",
            "destroy_sampler",
            // One uniform-buffer slab (binding 4).
            "DescriptorType::UNIFORM_BUFFER",
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
            // The global set is 0 everywhere else.
            if let Some(rest) = line.trim().strip_prefix("#define AE_BINDLESS_SET ")
                && rest.trim() != "0"
            {
                found.push(format!(
                    "{}:{}: {} (the global set is 0 outside the post chain)",
                    path.display(),
                    n + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        found.is_empty(),
        "per-pass descriptor sets outside the exceptions:\n{}",
        found.join("\n")
    );
}
