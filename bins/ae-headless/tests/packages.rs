//! A game defined only by an archive: the skirmish game's content, its
//! built sim module (`module.wasm`) and its map (`games/skirmish/maps/
//! crossing`, the land crossing map exported with `ae-tool map-export`) in
//! one zip, as `ae-tool pack --module skirmish --with maps/crossing` makes.
//! * Played from that file alone (no `--game`, `--nav`, library or game
//!   parameter: the bases are the map's start points), a whole bot match
//!   (6,500 ticks; it is decided at 6,457) ends in the same state as the
//!   same match from the game's source, its content directory and the
//!   generated land map: every digest part is equal but the event store's
//!   (which also holds the map's start points). Its total digest is pinned:
//!   the bot-vs-bot golden of the archive-defined game.
//! * As two packages (game, map) in a library: a recording names them, and
//!   a replay finds them by name and verifies, and refuses without one.
//! * The committed map package is what `map-export` makes of the generated
//!   map today (same terrain), so it cannot drift from it unnoticed.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

fn headless(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_ae-headless"))
        .args(args)
        .output()
        .expect("run ae-headless");
    let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

/// The `--digest` part lines (name, rows, hash), without the event store's.
fn parts(out: &str) -> Vec<String> {
    let p: Vec<String> = out
        .lines()
        .filter(|l| l.starts_with("    ") && l.contains(" rows ") && !l.contains("(events)"))
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    assert!(p.len() > 10, "no digest parts in {out}");
    p
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
    assert_eq!(
        map.to_terrain().digest(),
        ae_nav::scenario::land_crossing_terrain(256).digest(),
        "maps/crossing is the land crossing map at 256 cells"
    );

    let mut game: BTreeMap<String, Vec<u8>> = ae_vfs::dir_files(&root.join("content")).unwrap();
    game.insert("module.wasm".into(), ae_sim::build::game("skirmish").unwrap());
    let map_files = ae_vfs::dir_files(&root.join("maps/crossing")).unwrap();
    std::fs::write(lib.join("skirmish.aegame"), ae_vfs::zip::write(&game).unwrap()).unwrap();
    std::fs::write(lib.join("crossing.aemap"), ae_vfs::zip::write(&map_files).unwrap()).unwrap();
    let mut whole = game.clone();
    whole.extend(map_files.into_iter().filter(|(p, _)| p != ae_vfs::MANIFEST));
    let one = std::env::temp_dir().join(format!("ae-packages-test-{}.aegame", std::process::id()));
    std::fs::write(&one, ae_vfs::zip::write(&whole).unwrap()).unwrap();
    let lib_s = lib.to_str().unwrap();
    let rec = lib.join("match.aerep");
    let rec = rec.to_str().unwrap();

    // From source the map side and starting tanks come in the parameter;
    // from the archive, the map's start points and the game's defaults.
    let param_hi = (2048u64 >> 8 | 6 << 8).to_string();
    let common = [
        "--units",
        "0",
        "--ticks",
        "6500",
        "--script",
        "bot",
        "--workers",
        "2",
        "--digest",
    ];
    let (ok, from_source) = headless(
        &[
            &common[..],
            &[
                "--param-hi",
                &param_hi,
                "--game",
                "skirmish",
                "--nav",
                "land",
                "--map-cells",
                "256",
                "--content",
                root.join("content").to_str().unwrap(),
            ],
        ]
        .concat(),
    );
    assert!(ok, "{from_source}");
    let (ok, from_archive) = headless(&[&common[..], &["--package", one.to_str().unwrap()]].concat());
    assert!(ok, "{from_archive}");
    assert_eq!(
        parts(&from_archive),
        parts(&from_source),
        "archive:\n{from_archive}\nsource:\n{from_source}"
    );
    let total = from_archive
        .split(" digest ")
        .nth(1)
        .and_then(|r| r.split_whitespace().next())
        .unwrap();
    assert_eq!(
        total, "5bdc31cf5379b7760c4b88f1e82fe5e4",
        "the archive's golden bot match:\n{from_archive}"
    );

    // Two packages in a library: recorded, replayed by name.
    let packages = [
        "--package",
        "skirmish",
        "--package",
        "skirmish-crossing",
        "--packages",
        lib_s,
    ];
    let (ok, out) = headless(
        &[
            &[
                "--units",
                "0",
                "--ticks",
                "1500",
                "--script",
                "bot",
                "--workers",
                "2",
                "--record",
                rec,
            ],
            &packages[..],
        ]
        .concat(),
    );
    assert!(ok, "{out}");
    let (ok, out) = headless(&["--replay", rec, "--packages", lib_s, "--verify", "--workers", "1"]);
    assert!(ok && out.contains("50 checkpoints match"), "{out}");
    let (ok, out) = headless(&["--replay", rec, "--verify"]);
    assert!(!ok && out.contains("name a --packages DIR"), "{out}");
    std::fs::remove_dir_all(&lib).ok();
    std::fs::remove_file(&one).ok();
}
