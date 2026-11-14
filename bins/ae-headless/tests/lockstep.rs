//! Lockstep over loopback UDP: `ae-relay` and three `ae-headless
//! --connect` clients, each sending its player's scripted orders, every
//! sender with 30 ms latency, 20 ms jitter and 5% loss.
//! * In sync: the relay compared every checkpoint and the final digests
//!   agree; the game it recorded replays with every checkpoint matching.
//! * A fault injected in one client is caught: the relay reports a desync
//!   at the first checkpoint after it, and exits 1.

use std::path::PathBuf;
use std::process::{Child, Command, Output, Stdio};

fn relay_exe() -> PathBuf {
    let headless = PathBuf::from(env!("CARGO_BIN_EXE_ae-headless"));
    let relay = headless.with_file_name(format!("ae-relay{}", std::env::consts::EXE_SUFFIX));
    // Same profile as this test's binaries.
    let release = headless.components().any(|c| c.as_os_str() == "release");
    let mut cargo = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cargo.args(["build", "-q", "-p", "ae-relay"]);
    if release {
        cargo.arg("--release");
    }
    assert!(cargo.status().expect("cargo").success(), "building ae-relay");
    relay
}

const NET: [&str; 6] = ["--latency", "30", "--jitter", "20", "--loss", "5"];

fn game(port: u16, ticks: u64, record: Option<&str>, fault: Option<(u32, &str)>) -> (Output, Vec<Output>) {
    let addr = format!("127.0.0.1:{port}");
    let mut relay = Command::new(relay_exe());
    relay.args(["--players", "3", "--ticks", &ticks.to_string(), "--bind", &addr]).args(NET);
    if let Some(path) = record {
        relay.args(["--record", path]);
    }
    let relay: Child = relay.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().expect("start the relay");
    let clients: Vec<Child> = (0..3u32)
        .map(|p| {
            let mut c = Command::new(env!("CARGO_BIN_EXE_ae-headless"));
            c.args(["--game", "crossing", "--nav", "water", "--map-cells", "256", "--units", "600", "--param-hi", "255", "--workers", "1", "--checkpoint-every", "30"]);
            c.args(["--connect", &addr, "--player", &p.to_string(), "--script", "orders", "--seed", &p.to_string()]).args(NET);
            if let Some((who, inject)) = fault
                && who == p
            {
                c.args(["--inject", inject]);
            }
            c.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().expect("start a client")
        })
        .collect();
    let clients: Vec<Output> = clients.into_iter().map(|c| c.wait_with_output().unwrap()).collect();
    (relay.wait_with_output().unwrap(), clients)
}

fn text(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr)
}

#[test]
fn clients_stay_in_sync_and_a_desync_is_caught() {
    let dir = std::env::temp_dir().join(format!("ae-lockstep-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("game.aerep");
    let file = file.to_str().unwrap();

    let (relay, clients) = game(7651, 240, Some(file), None);
    let r = text(&relay);
    assert!(relay.status.success() && r.contains("in sync") && r.contains("final digests agree") && r.contains("8 of 8 digest rounds compared"), "{r}");
    for c in &clients {
        assert!(c.status.success() && text(c).contains("240 ticks"), "{}", text(c));
    }
    let digest = r.split("agree (").nth(1).unwrap().split(')').next().unwrap().to_string();
    assert!(clients.iter().all(|c| text(c).contains(&digest)), "every client ends at {digest}");
    let replay = Command::new(env!("CARGO_BIN_EXE_ae-headless")).args(["--game", "crossing", "--replay", file, "--verify", "--workers", "2"]).output().unwrap();
    assert!(replay.status.success() && text(&replay).contains("8 checkpoints match"), "{}", text(&replay));

    // Player 1 flips a bit after tick 100: caught at the checkpoint after it.
    let (relay, _) = game(7652, 180, None, Some((1, "100:Body.x:5")));
    let r = text(&relay);
    assert!(!relay.status.success() && r.contains("DESYNC at tick 120"), "{r}");
    std::fs::remove_dir_all(&dir).ok();
}
