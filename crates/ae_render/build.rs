//! Compiles `shaders/*.{vert,frag,comp}` to SPIR-V with `glslc` (Vulkan 1.3
//! target; `GLSLC` overrides the program). Lifted from ecs_sim, where it ran
//! only with the `gpu` feature; here the whole crate is the GPU side.
//!
//! The binaries' hashes and the compiling toolchain's identity are written
//! to `$OUT_DIR/spirv_hashes.txt` for the M16.1 SPIR-V gate
//! (`tests/spirv.rs`): an edit to a shader that changes its binary shows up
//! as a pinned-hash change on the pinned compiler, while a different
//! supported compiler makes the hashes advisory unless the gate is strict.

use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shaders");
    // The compiler is an input too: a different `GLSLC` must recompile and
    // restamp the list, not reuse a warm build's.
    println!("cargo:rerun-if-env-changed=GLSLC");
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
    // The pinned list's header: the gate's rules, the compiler that made
    // these binaries, and its identity (all of `glslc --version`, trimmed,
    // so a shaderc/glslang/SPIRV-Tools change is a different identity).
    let mut hashes = String::new();
    hashes.push_str(
        "# M16.1's shader-binary gate (tests/spirv.rs). On the pinned\n\
         # toolchain a hash difference is shader drift and fails. On a\n\
         # different compiler it is advisory unless AE_STRICT_SPIRV=1, which\n\
         # requires the pinned toolchain. AE_UPDATE_SPIRV=1 regenerates this\n\
         # file with the compiler that ran.\n",
    );
    let version = Command::new(&glslc).arg("--version").output();
    let text = version
        .as_ref()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout.clone()).ok())
        .map(|text| {
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.is_empty());
    let compiler = text
        .as_deref()
        .and_then(|text| text.lines().next())
        .unwrap_or(&glslc)
        .to_string();
    let identity = text.as_deref().unwrap_or(&glslc);
    let shown = identity.replace('\n', " | ");
    writeln!(hashes, "# compiler: {compiler}").unwrap();
    writeln!(hashes, "# compiler-id: {:016x}", fnv(identity.as_bytes())).unwrap();
    writeln!(hashes, "# compiler-version: {shown}").unwrap();
    let mut shaders: Vec<_> = entries
        .into_iter()
        .map(|path| (path.file_name().unwrap().to_string_lossy().into_owned(), path, None))
        .collect();
    for name in [
        "cbt_prepare.comp",
        "cbt_update.comp",
        "terrain.vert",
        "terrain.frag",
        "units_cull.comp",
        "units_models_cull.comp",
        "units.vert",
        "units.frag",
        "units_models.vert",
        "units_models_opaque.frag",
        "units_models_masked.frag",
    ] {
        let (stem, stage) = name.rsplit_once('.').unwrap();
        shaders.push((
            format!("{stem}_mirror.{stage}"),
            Path::new("shaders").join(name),
            Some("MIRROR"),
        ));
    }
    shaders.push((
        "water_reflect.frag".into(),
        Path::new("shaders/water.frag").into(),
        Some("REFLECT"),
    ));
    for (source, name, defines) in [
        ("terrain.frag", "terrain_refract_owner.frag", "WATER_REFRACT_OWNER"),
        ("water.frag", "water_refract.frag", "REFRACT"),
        ("water.frag", "water_full.frag", "REFRACT,REFLECT"),
        ("water.vert", "water_mesh.vert", "WATER_MESH"),
        ("water_underwater.frag", "water_underwater_waves.frag", "WATER_MESH"),
        ("particles_sim.comp", "particles_sim_water.comp", "WATER_SPLIT"),
        ("particles_spawn.comp", "particles_spawn_water.comp", "WATER_SPLIT"),
        ("particles.vert", "particles_underwater.vert", "WATER_SPLIT"),
        ("water.frag", "water_mesh.frag", "WATER_MESH"),
        ("water.frag", "water_mesh_reflect.frag", "WATER_MESH,REFLECT"),
        ("water.frag", "water_mesh_refract.frag", "WATER_MESH,REFRACT"),
        ("water.frag", "water_mesh_full.frag", "WATER_MESH,REFRACT,REFLECT"),
        ("water_volume.frag", "water_volume_waves.frag", "WATER_MESH"),
        ("water.frag", "water_rich_refract.frag", "REFRACT,RICH"),
        ("water.frag", "water_rich_full.frag", "REFRACT,REFLECT,RICH"),
        ("water.frag", "water_mesh_rich_refract.frag", "WATER_MESH,REFRACT,RICH"),
        (
            "water.frag",
            "water_mesh_rich_full.frag",
            "WATER_MESH,REFRACT,REFLECT,RICH",
        ),
        ("cbt_prepare.comp", "cbt_prepare_water.comp", "WATER_MESH"),
        ("cbt_reduce.comp", "cbt_reduce_water.comp", "WATER_MESH"),
        ("cbt_update.comp", "water_lod_error.comp", "WATER_MESH"),
        ("units_cull.comp", "units_cull_sub.comp", "SUB"),
        ("units_models_cull.comp", "units_models_cull_sub.comp", "SUB"),
        ("units_lights.comp", "units_lights_sub.comp", "SUB"),
        ("units.vert", "units_sub.vert", "SUB"),
        ("units.vert", "units_wave.vert", "RIDE_WAVES"),
        ("units.vert", "units_wave_sub.vert", "RIDE_WAVES,SUB"),
        ("units_models.vert", "units_models_wave.vert", "RIDE_WAVES"),
        ("units_models.vert", "units_models_wave_sub.vert", "RIDE_WAVES,SUB"),
        ("units.frag", "units_sub.frag", "SUB"),
        ("units_models.vert", "units_models_sub.vert", "SUB"),
        ("units_models_opaque.frag", "units_models_opaque_sub.frag", "SUB"),
        ("units_models_masked.frag", "units_models_masked_sub.frag", "SUB"),
        ("units_shadow.vert", "units_shadow_sub.vert", "SUB"),
        ("units_models_shadow.vert", "units_models_shadow_sub.vert", "SUB"),
        (
            "units_models_shadow_masked.vert",
            "units_models_shadow_masked_sub.vert",
            "SUB",
        ),
        // Floating hulls cast and reflect from where they ride the waves.
        ("units_shadow.vert", "units_shadow_wave.vert", "RIDE_WAVES"),
        ("units_shadow.vert", "units_shadow_wave_sub.vert", "RIDE_WAVES,SUB"),
        (
            "units_models_shadow.vert",
            "units_models_shadow_wave.vert",
            "RIDE_WAVES",
        ),
        (
            "units_models_shadow.vert",
            "units_models_shadow_wave_sub.vert",
            "RIDE_WAVES,SUB",
        ),
        (
            "units_models_shadow_masked.vert",
            "units_models_shadow_masked_wave.vert",
            "RIDE_WAVES",
        ),
        (
            "units_models_shadow_masked.vert",
            "units_models_shadow_masked_wave_sub.vert",
            "RIDE_WAVES,SUB",
        ),
        ("units.vert", "units_mirror_wave.vert", "MIRROR,RIDE_WAVES"),
        (
            "units_models.vert",
            "units_models_mirror_wave.vert",
            "MIRROR,RIDE_WAVES",
        ),
    ] {
        shaders.push((name.into(), Path::new("shaders").join(source), Some(defines)));
    }
    shaders.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, path, define) in shaders {
        println!("cargo:rerun-if-changed={}", path.display());
        let target = Path::new(&out).join(format!("{name}.spv"));
        let mut command = Command::new(&glslc);
        command.args(["--target-env=vulkan1.3", "-O", "-Werror", "-I", "shaders"]);
        if let Some(define) = define {
            command.args(define.split(',').map(|name| format!("-D{name}")));
        }
        let status = command
            .arg(&path)
            .arg("-o")
            .arg(&target)
            .status()
            .unwrap_or_else(|err| panic!("running {glslc} (install glslc or set GLSLC): {err}"));
        assert!(status.success(), "glslc failed for {}", path.display());
        let bytes = std::fs::read(&target).expect("the compiled SPIR-V");
        writeln!(hashes, "{name} {:016x}", fnv(&bytes)).unwrap();
    }
    std::fs::write(Path::new(&out).join("spirv_hashes.txt"), hashes).expect("writing the SPIR-V hash list");
}

/// FNV-1a, as the tests' frame hash uses it.
fn fnv(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}
