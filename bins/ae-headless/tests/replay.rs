//! Record, replay and bisect through the `ae-headless` binary.
//! * A recording of the crossing game with scripted player orders replays
//!   with every checkpoint matching, at 1 and at 2 workers.
//! * A replay refuses another game module, naming it; a game with content
//!   (skirmish) records its content hash and refuses to replay without it.
//!   Its recording is of two bots (`--script bot`): their commands are in
//!   the file, and it verifies at another worker count.
//! * A bit flipped in one entity's field in one run is localised: the
//!   first divergence is after the next tick, at that entity (and at most a
//!   couple of neighbours the nav service moved by it).

use std::process::Command;

fn headless(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_ae-headless")).args(args).output().expect("run ae-headless");
    let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

#[test]
fn record_replay_and_bisect() {
    let dir = std::env::temp_dir().join(format!("ae-replay-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("crossing.aerep");
    let file = file.to_str().unwrap();
    let (ok, out) = headless(&["--game", "crossing", "--nav", "water", "--map-cells", "256", "--units", "600", "--ticks", "120", "--workers", "2", "--param-hi", "255", "--checkpoint-every", "20", "--record", file, "--script", "orders"]);
    assert!(ok, "{out}");
    assert!(out.contains("120 ticks") && out.contains("6 checkpoints") && !out.contains(" 0 commands"), "{out}");

    for workers in ["1", "2"] {
        let (ok, out) = headless(&["--game", "crossing", "--replay", file, "--verify", "--workers", workers]);
        assert!(ok && out.contains("6 checkpoints match"), "{workers} worker(s): {out}");
    }

    let (ok, out) = headless(&["--game", "sandbox", "--replay", file, "--verify"]);
    assert!(!ok && out.contains("the game module differs"), "{out}");

    let (ok, out) = headless(&["--game", "crossing", "--replay", file, "--bisect", "--inject", "60:Body.y:17", "--workers", "2"]);
    assert!(!ok, "a desync exits 1: {out}");
    assert!(out.contains("first divergence after tick 61 (the fault was applied after tick 60)"), "{out}");
    let rows: Vec<&str> = out.lines().filter(|l| l.starts_with("  ") && l.contains(" of entity ")).collect();
    let hit = out.lines().find_map(|l| l.strip_prefix("injected: Body.y of entity ")).and_then(|r| r.split_whitespace().next()).expect("the injected entity");
    // Digests compare states after ticks, so the fault (applied between
    // ticks 60 and 61) shows after tick 61, when the nav service has already
    // moved that unit by it and nudged its neighbours: the injected entity
    // differs, and only a few entities around it.
    assert!(rows.iter().any(|r| r.starts_with(&format!("  Body.y of entity {hit}:"))), "{out}");
    let entities: std::collections::BTreeSet<&str> = rows.iter().filter_map(|r| r.split(" of entity ").nth(1)?.split(':').next()).collect();
    assert!(entities.len() <= 3, "the damage after one tick: {entities:?}\n{out}");

    // A game with content: the content hash is part of the header.
    let content = concat!(env!("CARGO_MANIFEST_DIR"), "/../../games/skirmish/content");
    let sk = dir.join("skirmish.aerep");
    let sk = sk.to_str().unwrap();
    let param_hi = (2048u64 >> 8 | 6 << 8).to_string();
    let world = ["--game", "skirmish", "--nav", "land", "--map-cells", "256", "--units", "0", "--param-hi", &param_hi, "--workers", "2"];
    let (ok, out) = headless(&[&world[..], &["--content", content, "--ticks", "1200", "--script", "bot", "--record", sk]].concat());
    // The bots' attack orders are in the recording (both sides' first wave).
    let commands: usize = out.split(" commands,").next().and_then(|l| l.rsplit(' ').next()).and_then(|n| n.parse().ok()).unwrap_or(0);
    assert!(ok && commands >= 8 + 16, "{out}");
    let (ok, out) = headless(&["--game", "skirmish", "--replay", sk, "--verify", "--content", content, "--workers", "1"]);
    assert!(ok && out.contains("40 checkpoints match"), "{out}");
    let (ok, out) = headless(&["--game", "skirmish", "--replay", sk, "--verify"]);
    assert!(!ok && out.contains("in the replay, unset here"), "{out}");
    std::fs::remove_dir_all(&dir).ok();
}
