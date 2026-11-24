//! The M16.1 SPIR-V gate (advisory): every shader `build.rs` compiles is
//! pinned by its binary's hash in `tests/data/m16_spirv.txt`.
//!
//! This is the fine-grained "what did the water work touch" check that sits
//! under the binding frame-identity gate (`tests/gpu.rs`, `m16_baseline_*`):
//! a shader edit that changes its compiled binary is a hash change, and a
//! hash change is allowed only in a commit that names the file and the
//! reason. Comment-only edits may not change the binary (the compiler drops
//! them); a functional edit always does.
//!
//! `build.rs` writes the list it compiled to `$OUT_DIR/spirv_hashes.txt`:
//! a hash per shader, the `compiler:` line and the compiler's *identity*
//! (all of `glslc --version`, trimmed, so a shaderc, glslang or SPIRV-Tools
//! change is a different identity).
//!
//! The rules:
//! - **Pinned toolchain** (the identity matches the file's): hash
//!   differences are shader drift and fail in every mode.
//! - **Different toolchain**: hash differences are advisory by default (a
//!   supported compiler may emit different but valid SPIR-V), with the
//!   differing files printed. `AE_STRICT_SPIRV=1` requires the pinned
//!   toolchain and fails instead.
//!
//! Either way an intended change is recorded with `AE_UPDATE_SPIRV=1
//! cargo test -p ae_render --test spirv`, committed with the reason; that
//! stores the compiling toolchain, so a later editor's compiler stays
//! advisory until the pin moves.

use std::fmt::Write as _;
use std::path::PathBuf;

/// What `build.rs` compiled in this build.
const BUILT: &str = include_str!(concat!(env!("OUT_DIR"), "/spirv_hashes.txt"));

fn data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/m16_spirv.txt")
}

/// A pinned list: the compiler's name, identity and version text, and the
/// `(file, hash)` entries in name order.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Pinned {
    compiler: String,
    identity: String,
    version: String,
    entries: Vec<(String, u64)>,
}

fn parse(text: &str) -> Pinned {
    let mut pinned = Pinned {
        compiler: String::new(),
        identity: String::new(),
        version: String::new(),
        entries: Vec::new(),
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# compiler-id:") {
            pinned.identity = rest.trim().to_string();
            continue;
        }
        if let Some(rest) = line.strip_prefix("# compiler-version:") {
            pinned.version = rest.trim().to_string();
            continue;
        }
        if let Some(rest) = line.strip_prefix("# compiler:") {
            pinned.compiler = rest.trim().to_string();
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let (name, hash) = line
            .split_once(' ')
            .unwrap_or_else(|| panic!("SPIR-V hash line without a hash: {line:?}"));
        pinned.entries.push((
            name.to_string(),
            u64::from_str_radix(hash, 16).unwrap_or_else(|_| panic!("bad hash {hash:?} for {name}")),
        ));
    }
    pinned.entries.sort();
    pinned
}

fn render(pinned: &Pinned) -> String {
    let mut out = String::new();
    out.push_str(
        "# M16.1's shader-binary gate (tests/spirv.rs). On the pinned\n\
         # toolchain a hash difference is shader drift and fails. On a\n\
         # different compiler it is advisory unless AE_STRICT_SPIRV=1, which\n\
         # requires the pinned toolchain. AE_UPDATE_SPIRV=1 regenerates this\n\
         # file with the compiler that ran.\n",
    );
    writeln!(out, "# compiler: {}", pinned.compiler).unwrap();
    writeln!(out, "# compiler-id: {}", pinned.identity).unwrap();
    writeln!(out, "# compiler-version: {}", pinned.version).unwrap();
    for (name, hash) in &pinned.entries {
        writeln!(out, "{name} {hash:016x}").unwrap();
    }
    out
}

/// The hash differences between the build's list and the pinned one, one
/// line each; empty when every file's binary agrees.
fn hash_differences(built: &Pinned, pinned: &Pinned) -> Vec<String> {
    let mut out = Vec::new();
    for (name, hash) in &built.entries {
        match pinned.entries.iter().find(|(n, _)| n == name) {
            Some((_, pinned_hash)) if pinned_hash == hash => {}
            Some((_, pinned_hash)) => out.push(format!("{name}: {pinned_hash:016x} -> {hash:016x}")),
            None => out.push(format!("{name}: new (hash {hash:016x})")),
        }
    }
    for (name, _) in &pinned.entries {
        if !built.entries.iter().any(|(n, _)| n == name) {
            out.push(format!("{name}: no longer built"));
        }
    }
    out
}

/// Whether the build used the toolchain the file was pinned with. An empty
/// identity on either side (an older file) counts as unknown, not pinned.
fn same_toolchain(built: &Pinned, pinned: &Pinned) -> bool {
    !built.identity.is_empty() && built.identity == pinned.identity
}

/// What the gate should do with this build.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Verdict {
    /// The pinned toolchain, every binary unchanged.
    Clean,
    /// A different (or unknown) toolchain: differences are advisory in the
    /// default mode. The list is the hash differences (possibly empty).
    Advisory(Vec<String>),
    /// A failure and its message.
    Failed(String),
}

fn verdict(built: &Pinned, pinned: &Pinned, strict: bool) -> Verdict {
    let diffs = hash_differences(built, pinned);
    if same_toolchain(built, pinned) {
        return if diffs.is_empty() {
            Verdict::Clean
        } else {
            Verdict::Failed(format!(
                "the compiled shaders differ from tests/data/m16_spirv.txt on the pinned compiler:\n  {}\n\
                 An intended shader edit updates this file in the same commit and says why \
                 (AE_UPDATE_SPIRV=1 cargo test -p ae_render --test spirv).",
                diffs.join("\n  ")
            ))
        };
    }
    if strict {
        return Verdict::Failed(format!(
            "AE_STRICT_SPIRV=1 requires the pinned toolchain: pinned {} ({:?}), built {} ({:?}). \
             Compile with the pinned glslc or unset AE_STRICT_SPIRV.",
            pinned.identity, pinned.version, built.identity, built.version
        ));
    }
    Verdict::Advisory(diffs)
}

/// `AE_STRICT_SPIRV` set to anything but an empty string or `0`.
fn strict() -> bool {
    matches!(std::env::var("AE_STRICT_SPIRV"), Ok(v) if !v.is_empty() && v != "0")
}

#[test]
fn shader_binaries_match_the_pinned_list() {
    let built = parse(BUILT);
    assert!(
        built.entries.len() > 20,
        "build.rs recorded only {} shaders",
        built.entries.len()
    );
    if std::env::var_os("AE_UPDATE_SPIRV").is_some() {
        std::fs::write(data_path(), render(&built)).expect("writing the pinned SPIR-V list");
        return;
    }
    let text = std::fs::read_to_string(data_path()).unwrap_or_else(|err| {
        panic!(
            "reading {} ({err}); create it with AE_UPDATE_SPIRV=1",
            data_path().display()
        )
    });
    let pinned = parse(&text);
    match verdict(&built, &pinned, strict()) {
        Verdict::Clean => {}
        Verdict::Advisory(diff) => {
            eprintln!(
                "note: the SPIR-V gate is advisory here: this build used a different toolchain \
                 (pinned {} [{:?}], built {} [{:?}])",
                pinned.identity, pinned.version, built.identity, built.version
            );
            for line in diff {
                eprintln!("  {line}");
            }
            eprintln!(
                "      A supported compiler may emit different but valid SPIR-V. Set \
                 AE_STRICT_SPIRV=1 to require the pinned toolchain; regenerate with \
                 AE_UPDATE_SPIRV=1 when a new toolchain becomes the pin."
            );
        }
        Verdict::Failed(why) => panic!("{why}"),
    }
}

/// The gate's rules: hash drift on the pinned toolchain fails, a different
/// toolchain is advisory by default, strict mode demands the pinned one, and
/// the file format round trips.
#[test]
fn the_gate_keys_strictness_to_the_toolchain() {
    let pinned = Pinned {
        compiler: "glslc test".into(),
        identity: "00000000000000aa".into(),
        version: "test | 1.0".into(),
        entries: vec![("a.frag".into(), 1), ("b.comp".into(), 2)],
    };
    assert_eq!(parse(&render(&pinned)), pinned, "the format round trips");
    let same = Pinned {
        identity: pinned.identity.clone(),
        ..pinned.clone()
    };
    assert_eq!(verdict(&same, &pinned, true), Verdict::Clean);

    // The pinned toolchain with a changed binary is shader drift, strict or not.
    let drifted = Pinned {
        entries: vec![("a.frag".into(), 9), ("b.comp".into(), 2)],
        ..same.clone()
    };
    for strict in [false, true] {
        assert!(matches!(verdict(&drifted, &pinned, strict), Verdict::Failed(_)));
    }
    assert_eq!(
        verdict(&drifted, &pinned, false),
        Verdict::Failed(
            "the compiled shaders differ from tests/data/m16_spirv.txt on the pinned compiler:\n  \
             a.frag: 0000000000000001 -> 0000000000000009\n\
             An intended shader edit updates this file in the same commit and says why \
             (AE_UPDATE_SPIRV=1 cargo test -p ae_render --test spirv)."
                .into()
        )
    );

    // A different toolchain: advisory by default, with the differences, and
    // a failure under strict.
    let other = Pinned {
        identity: "00000000000000bb".into(),
        version: "other | 2.0".into(),
        ..drifted.clone()
    };
    assert_eq!(
        verdict(&other, &pinned, false),
        Verdict::Advisory(vec!["a.frag: 0000000000000001 -> 0000000000000009".into()])
    );
    assert!(matches!(verdict(&other, &pinned, true), Verdict::Failed(_)));
    // Even with every binary equal, strict rejects the toolchain.
    let other_same = Pinned {
        entries: pinned.entries.clone(),
        ..other.clone()
    };
    assert_eq!(verdict(&other_same, &pinned, false), Verdict::Advisory(Vec::new()));
    assert!(matches!(verdict(&other_same, &pinned, true), Verdict::Failed(_)));

    // A file from before the identity was recorded is not the pinned
    // toolchain: advisory by default, a failure under strict.
    let old = Pinned {
        identity: String::new(),
        ..pinned.clone()
    };
    assert_eq!(verdict(&same, &old, false), Verdict::Advisory(Vec::new()));
    assert!(matches!(verdict(&same, &old, true), Verdict::Failed(_)));

    // A new file appears, or one stops being built: a hash difference, so
    // it fails on the pinned toolchain and is advisory on another.
    let added = Pinned {
        entries: vec![("a.frag".into(), 1), ("b.comp".into(), 2), ("c.vert".into(), 3)],
        ..same.clone()
    };
    assert_eq!(
        hash_differences(&added, &pinned),
        vec!["c.vert: new (hash 0000000000000003)"]
    );
    assert!(matches!(verdict(&added, &pinned, false), Verdict::Failed(_)));
}
