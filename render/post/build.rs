//! Compiles `shaders/*.frag` to SPIR-V with `glslc` (`GLSLC` overrides); the
//! module embeds them in its program table.
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=../../crates/ae_render/shaders/bindless.glsl");
    println!("cargo:rerun-if-changed=../../crates/ae_render/shaders/program.glsl");
    let out = std::env::var("OUT_DIR").unwrap();
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| "glslc".into());
    for entry in std::fs::read_dir("shaders").unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("frag") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        // The program interface headers are the engine's (the global set's
        // shader side), shared by every fullscreen program.
        let status = Command::new(&glslc)
            .args([
                "--target-env=vulkan1.3",
                "-O",
                "-Werror",
                "-I",
                "../../crates/ae_render/shaders",
            ])
            .arg(&path)
            .arg("-o")
            .arg(format!("{out}/{name}.spv"))
            .status()
            .unwrap_or_else(|e| panic!("running {glslc}: {e}"));
        assert!(status.success(), "glslc failed for {}", path.display());
    }
}
