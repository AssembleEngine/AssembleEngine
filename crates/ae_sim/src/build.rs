//! Building in-repo sim modules (`games/*`) for tools and tests.

use anyhow::{Context, Result, ensure};
use std::path::{Path, PathBuf};
use std::process::Command;

pub const GUEST_TARGET: &str = "wasm32-wasip1-threads";

/// Repository root (from this crate's location at compile time).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `cargo build --release` a guest crate for the sim target and return its
/// wasm bytes. Guest crates have their own workspace; their build output
/// goes to `target/game-guests`.
pub fn build_guest(manifest: &Path, crate_name: &str) -> Result<Vec<u8>> {
    let target_dir = repo_root().join("target/game-guests");
    let status = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args([
            "build",
            "--release",
            "--quiet",
            "--target",
            GUEST_TARGET,
            "--manifest-path",
        ])
        .arg(manifest)
        .arg("--target-dir")
        .arg(&target_dir)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .status()
        .context("run cargo for the guest")?;
    ensure!(status.success(), "guest build failed: {}", manifest.display());
    let wasm = target_dir
        .join(GUEST_TARGET)
        .join("release")
        .join(format!("{crate_name}.wasm"));
    std::fs::read(&wasm).with_context(|| format!("read {}", wasm.display()))
}

/// Build `games/<name>` (crate `ae_game_<name>`), or read `name` as a path
/// when it ends in `.wasm`.
pub fn game(name: &str) -> Result<Vec<u8>> {
    if name.ends_with(".wasm") {
        return std::fs::read(name).with_context(|| format!("read {name}"));
    }
    let manifest = repo_root().join("games").join(name).join("Cargo.toml");
    ensure!(
        manifest.exists(),
        "no game `{name}` (looked for {})",
        manifest.display()
    );
    build_guest(&manifest, &format!("ae_game_{name}"))
}
