//! A game defined only by archives: the skirmish game's package (its
//! content plus the built sim module as `module.wasm`) and a map package
//! (`games/skirmish/maps/crossing`, the land crossing map exported with
//! `ae-tool map-export`), both zipped into a package library.
//! * Played from them alone (no `--game`, no `--nav`), a bot match ends at
//!   the same digest as the same match from the game's source, its content
//!   directory and the generated land map.
//! * The recording names its packages; a replay finds them in the library
//!   by name and verifies, and refuses without one.
//! * The committed map package is what `map-export` makes of the generated
//!   map today (same terrain), so it cannot drift from it unnoticed.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

fn headless(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_ae-headless")).args(args).output().expect("run ae-headless");
    let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

fn final_digest(out: &str) -> String {
    out.split("final digest ").nth(1).and_then(|r| r.split_whitespace().next()).unwrap_or_else(|| panic!("no final digest in {out}")).to_string()
}

#[test]
fn a_skirmish_from_archives_alone() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../games/skirmish");
    let lib = std::env::temp_dir().join(format!("ae-packages-test-{}", std::process::id()));
    std::fs::create_dir_all(&lib).unwrap();

    // The committed map is the generated one.
    let mut vfs = ae_vfs::Vfs::new();
    vfs.mount(root.join("maps/crossing")).unwrap();
    let map = ae_map::GameMap::load(&vfs).unwrap();
    assert_eq!(map.to_terrain().digest(), ae_nav::scenario::land_crossing_terrain(256).digest(), "maps/crossing is the land crossing map at 256 cells");

    let mut game: BTreeMap<String, Vec<u8>> = ae_vfs::dir_files(&root.join("content")).unwrap();
    game.insert("module.wasm".into(), ae_sim::build::game("skirmish").unwrap());
    std::fs::write(lib.join("skirmish.aegame"), ae_vfs::zip::write(&game).unwrap()).unwrap();
    std::fs::write(lib.join("crossing.aemap"), ae_vfs::zip::write(&ae_vfs::dir_files(&root.join("maps/crossing")).unwrap()).unwrap()).unwrap();
    let lib_s = lib.to_str().unwrap();
    let rec = lib.join("match.aerep");
    let rec = rec.to_str().unwrap();

    let param_hi = (2048u64 >> 8 | 6 << 8).to_string();
    let common = ["--units", "0", "--param-hi", &param_hi, "--ticks", "1500", "--script", "bot", "--workers", "2"];
    let (ok, from_source) = headless(&[&common[..], &["--game", "skirmish", "--nav", "land", "--map-cells", "256", "--content", root.join("content").to_str().unwrap(), "--record", &format!("{rec}.src")]].concat());
    assert!(ok, "{from_source}");
    let (ok, from_archives) = headless(&[&common[..], &["--package", "skirmish", "--package", "skirmish-crossing", "--packages", lib_s, "--record", rec]].concat());
    assert!(ok, "{from_archives}");
    assert_eq!(final_digest(&from_archives), final_digest(&from_source), "archives:\n{from_archives}\nsource:\n{from_source}");

    let (ok, out) = headless(&["--replay", rec, "--packages", lib_s, "--verify", "--workers", "1"]);
    assert!(ok && out.contains("50 checkpoints match"), "{out}");
    let (ok, out) = headless(&["--replay", rec, "--verify"]);
    assert!(!ok && out.contains("name a --packages DIR"), "{out}");
    std::fs::remove_dir_all(&lib).ok();
}
