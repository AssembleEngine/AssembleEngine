//! Builds the guest module for wasm32-wasip1-threads and exposes its path as
//! the `GUEST_WASM` compile-time environment variable.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=guest/src/lib.rs");
    println!("cargo:rerun-if-changed=guest/build.rs");
    println!("cargo:rerun-if-changed=guest/Cargo.toml");
    println!("cargo:rerun-if-changed=guest2/src/lib.rs");
    println!("cargo:rerun-if-changed=guest2/build.rs");
    println!("cargo:rerun-if-changed=guest2/Cargo.toml");

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let wasm = build_guest(&out, "guest", "parallel_guest_module");
    println!("cargo:rustc-env=GUEST_WASM={}", wasm.display());
    let wasm2 = build_guest(&out, "guest2", "parallel_guest_module2");
    println!("cargo:rustc-env=GUEST2_WASM={}", wasm2.display());
}

fn build_guest(out: &std::path::Path, dir: &str, name: &str) -> PathBuf {
    let target_dir = out.join(format!("{dir}-target"));
    let mut cargo = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cargo
        .args(["build", "--release", "--target", "wasm32-wasip1-threads", "--manifest-path", &format!("{dir}/Cargo.toml")])
        .arg("--target-dir")
        .arg(&target_dir);
    // The outer build's flags are for the host target.
    for var in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"] {
        cargo.env_remove(var);
    }
    let status = cargo.status().expect("run cargo for a guest module");
    assert!(status.success(), "guest build failed (is the wasm32-wasip1-threads target installed?)");
    let wasm = target_dir.join(format!("wasm32-wasip1-threads/release/{name}.wasm"));
    assert!(wasm.exists(), "guest module missing at {}", wasm.display());
    wasm
}
