//! Builds the guest module for wasm32-wasip1-threads and exposes its path as
//! the `GUEST_WASM` compile-time environment variable.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=guest/src/lib.rs");
    println!("cargo:rerun-if-changed=guest/build.rs");
    println!("cargo:rerun-if-changed=guest/Cargo.toml");

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_dir = out.join("guest-target");
    let mut cargo = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cargo
        .args(["build", "--release", "--target", "wasm32-wasip1-threads", "--manifest-path", "guest/Cargo.toml"])
        .arg("--target-dir")
        .arg(&target_dir);
    // The outer build's flags are for the host target.
    for var in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"] {
        cargo.env_remove(var);
    }
    let status = cargo.status().expect("run cargo for the guest module");
    assert!(status.success(), "guest build failed (is the wasm32-wasip1-threads target installed?)");

    let wasm = target_dir.join("wasm32-wasip1-threads/release/parallel_guest_module.wasm");
    assert!(wasm.exists(), "guest module missing at {}", wasm.display());
    println!("cargo:rustc-env=GUEST_WASM={}", wasm.display());
}
