//! Compiles `shaders/*.{vert,frag,comp}` to SPIR-V with `glslc` (Vulkan 1.3
//! target; `GLSLC` overrides the program). Lifted from ecs_sim, where it ran
//! only with the `gpu` feature; here the whole crate is the GPU side.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shaders");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| "glslc".into());
    let mut entries: Vec<_> = std::fs::read_dir("shaders")
        .expect("shaders/ directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("vert" | "frag" | "comp")
            )
        })
        .collect();
    entries.sort();
    for path in entries {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path.file_name().unwrap().to_string_lossy();
        let target = Path::new(&out).join(format!("{name}.spv"));
        let status = Command::new(&glslc)
            .args(["--target-env=vulkan1.3", "-O", "-Werror", "-I", "shaders"])
            .arg(&path)
            .arg("-o")
            .arg(&target)
            .status()
            .unwrap_or_else(|err| panic!("running {glslc} (install shaderc/glslc or set GLSLC): {err}"));
        assert!(status.success(), "glslc failed for {}", path.display());
    }
}
