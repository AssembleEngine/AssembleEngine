//! Developer tool.
//!
//! ```text
//! ae-tool bench list
//! ae-tool bench <suite>|all [--small] [--quick] [--threads N] [--label NAME] [--out DIR] [--no-save]
//! ae-tool bench-diff <base> <head> [--time-tolerance PCT] [--all]
//! ae-tool navbench [--scaling ...]   # runs ae_nav's navbench binary
//! ae-tool catalog-fmt FILE...          # rewrite catalogue files in canonical form
//! ae-tool pack DIR OUT [--module GAME]  # a package directory as a deterministic zip
//! ae-tool map-export land|water CELLS DIR [NAME]  # a generated map as a map package
//! ae-tool unpack-list FILE             # an archive's files and content hash
//! ```
//! `<base>` and `<head>` are report files or directories of reports
//! (`bench/results/<label>`). `bench-diff` exits with status 1 on a regression.

use ae_bench::Report;
use ae_bench::diff::{Thresholds, diff};
use ae_bench::measure::Budget;
use ae_bench::suites::{self, Options};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage:
  ae-tool bench list
  ae-tool bench <suite>|all [--small] [--quick] [--threads N] [--label NAME] [--out DIR] [--no-save]
  ae-tool bench-diff <base> <head> [--time-tolerance PCT] [--all]
  ae-tool navbench [navbench arguments, e.g. --scaling]
  ae-tool catalog-fmt FILE...
  ae-tool pack DIR OUT [--module GAME]
  ae-tool map-export land|water CELLS DIR [NAME]
  ae-tool unpack-list FILE";

/// Runs `ae_nav`'s `navbench` binary from the same target directory
/// (`cargo build --release -p ae_nav --bin navbench`), passing the arguments
/// through.
fn navbench(args: &[String]) -> Result<ExitCode> {
    let exe = std::env::current_exe().context("locating ae-tool")?;
    let path = exe.with_file_name(format!("navbench{}", std::env::consts::EXE_SUFFIX));
    if !path.exists() {
        bail!("{} not found; build it with `cargo build --release -p ae_nav --bin navbench`", path.display());
    }
    let status = std::process::Command::new(&path).args(args).status().with_context(|| format!("running {}", path.display()))?;
    Ok(ExitCode::from(status.code().unwrap_or(1).clamp(0, 255) as u8))
}

/// Rewrite catalogue files in canonical form (`ae_catalog::CatalogFile::write`).
fn catalog_fmt(files: &[String]) -> Result<()> {
    if files.is_empty() {
        bail!("catalog-fmt needs at least one file");
    }
    for f in files {
        let text = std::fs::read_to_string(f).with_context(|| format!("read {f}"))?;
        let canonical = ae_catalog::CatalogFile::parse(&text).with_context(|| f.clone())?.write();
        if canonical != text {
            std::fs::write(f, &canonical).with_context(|| format!("write {f}"))?;
            println!("{f}: rewritten");
        } else {
            println!("{f}: already canonical");
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("bench") => bench(&args[1..]).map(|()| ExitCode::SUCCESS),
        Some("bench-diff") => bench_diff(&args[1..]),
        Some("navbench") => navbench(&args[1..]),
        Some("catalog-fmt") => catalog_fmt(&args[1..]).map(|()| ExitCode::SUCCESS),
        Some("pack") => pack(&args[1..]).map(|()| ExitCode::SUCCESS),
        Some("map-export") => map_export(&args[1..]).map(|()| ExitCode::SUCCESS),
        Some("unpack-list") => unpack_list(&args[1..]).map(|()| ExitCode::SUCCESS),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    result.unwrap_or_else(|error| {
        eprintln!("error: {error:#}");
        ExitCode::from(2)
    })
}

/// Arguments split into positionals, value flags (`--name value`) and switches.
struct Args {
    positional: Vec<String>,
    values: Vec<(String, String)>,
    switches: Vec<String>,
}

impl Args {
    fn parse(args: &[String], value_flags: &[&str], switch_flags: &[&str]) -> Result<Self> {
        let mut parsed = Args { positional: Vec::new(), values: Vec::new(), switches: Vec::new() };
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            if value_flags.contains(&arg.as_str()) {
                let value = iter.next().with_context(|| format!("{arg} needs a value"))?;
                parsed.values.push((arg.clone(), value.clone()));
            } else if switch_flags.contains(&arg.as_str()) {
                parsed.switches.push(arg.clone());
            } else if arg.starts_with("--") {
                bail!("unknown option {arg}");
            } else {
                parsed.positional.push(arg.clone());
            }
        }
        Ok(parsed)
    }

    fn value(&self, flag: &str) -> Option<&str> {
        self.values.iter().find(|(name, _)| name == flag).map(|(_, value)| value.as_str())
    }

    fn switch(&self, flag: &str) -> bool {
        self.switches.iter().any(|s| s == flag)
    }
}

fn bench(args: &[String]) -> Result<()> {
    let args = Args::parse(args, &["--threads", "--label", "--out"], &["--small", "--quick", "--no-save"])?;
    let [target] = args.positional.as_slice() else { bail!("expected one suite name, `all` or `list`") };
    if target == "list" {
        for suite in suites::SUITES {
            println!("{:<18} {}", suite.name, suite.about);
        }
        return Ok(());
    }
    let selected: Vec<&suites::Suite> = if target == "all" {
        suites::SUITES.iter().collect()
    } else {
        vec![suites::find(target).with_context(|| format!("unknown suite `{target}` (try `ae-tool bench list`)"))?]
    };

    let default_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8);
    let options = Options {
        budget: if args.switch("--quick") { Budget::QUICK } else { Budget::FULL },
        small: args.switch("--small"),
        threads: match args.value("--threads") {
            Some(text) => text.parse().context("--threads must be a number")?,
            None => default_threads,
        },
    };
    let out = PathBuf::from(args.value("--out").unwrap_or("bench/results"));

    ae_profile::start();
    for suite in selected {
        let report = {
            let _zone = ae_profile::zone_dyn(suite.name);
            (suite.run)(&options)?
        };
        report.print();
        if !args.switch("--no-save") {
            let label = args.value("--label").unwrap_or(&report.commit);
            println!("saved {}", report.save(&out, label)?.display());
        }
    }
    Ok(())
}

fn bench_diff(args: &[String]) -> Result<ExitCode> {
    let args = Args::parse(args, &["--time-tolerance"], &["--all"])?;
    let [base, head] = args.positional.as_slice() else { bail!("expected <base> <head>") };
    let mut thresholds = Thresholds::default();
    if let Some(text) = args.value("--time-tolerance") {
        thresholds.time = text.parse::<f64>().context("--time-tolerance must be a percentage")? / 100.0;
    }
    let result = diff(&Report::load_all(Path::new(base))?, &Report::load_all(Path::new(head))?, thresholds);
    result.print(args.switch("--all"));
    Ok(if result.regressions() > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS })
}

/// `ae-tool pack DIR OUT [--module GAME]`: DIR (a package: it needs a
/// valid `manifest.json`) as a deterministic zip archive at OUT; with
/// `--module`, game GAME (`games/GAME`, or a `.wasm` file) built and added
/// as `module.wasm`, so the archive alone is the game.
fn pack(args: &[String]) -> Result<()> {
    let (dir, out, module) = match args {
        [dir, out] => (dir, out, None),
        [dir, out, flag, game] if flag == "--module" => (dir, out, Some(game)),
        _ => anyhow::bail!("usage: ae-tool pack DIR OUT [--module GAME]"),
    };
    let p = ae_vfs::Package::open(dir)?;
    let mut files = ae_vfs::dir_files(std::path::Path::new(dir))?;
    if let Some(game) = module {
        anyhow::ensure!(!files.contains_key("module.wasm"), "{dir} already has a module.wasm");
        files.insert("module.wasm".into(), ae_sim::build::game(game)?);
    }
    let zip = ae_vfs::zip::write(&files)?;
    std::fs::write(out, &zip)?;
    let mut v = ae_vfs::Vfs::new();
    v.mount_archive(out)?;
    println!("{out}: {}@{} ({}), {} files, {} bytes, content hash {:032x}", p.manifest.name, p.manifest.version, p.manifest.kind, files.len(), zip.len(), v.content_hash()?);
    Ok(())
}

/// `ae-tool unpack-list FILE`: the files of an archive, with sizes, and its
/// content hash (every entry is read and checked).
fn unpack_list(args: &[String]) -> Result<()> {
    let [file] = args else { anyhow::bail!("usage: ae-tool unpack-list FILE") };
    let a = ae_vfs::Archive::open(std::path::Path::new(file))?;
    for p in a.paths() {
        println!("{:>10} {p}", a.read(p).unwrap()?.len());
    }
    let mut v = ae_vfs::Vfs::new();
    v.mount_archive(file)?;
    println!("content hash {:032x}", v.content_hash()?);
    Ok(())
}

/// `ae-tool map-export land|water CELLS DIR [NAME]`: a generated crossing
/// map as a map package directory (`ae_map`): manifest, `map.json`,
/// `height.png`, `types.png`. Start points are the skirmish bases (15% in
/// from opposite corners).
fn map_export(args: &[String]) -> Result<()> {
    let (kind, cells, dir) = match args {
        [k, c, d] | [k, c, d, _] => (k.as_str(), c.parse::<i32>().context("CELLS")?, std::path::Path::new(d)),
        _ => anyhow::bail!("usage: ae-tool map-export land|water CELLS DIR [NAME]"),
    };
    let terrain = match kind {
        "land" => ae_nav::scenario::land_crossing_terrain(cells),
        "water" => ae_nav::scenario::water_crossing_terrain(cells),
        k => anyhow::bail!("unknown map {k} (land or water)"),
    };
    let name = args.get(3).cloned().unwrap_or_else(|| format!("{kind}-crossing-{cells}"));
    let e = cells as f32 * ae_nav::service::NavConfig::default().cell_size;
    let map = ae_map::GameMap::from_terrain(&terrain, vec![[e * 0.15, e * 0.15], [e * 0.85, e * 0.85]], &format!("The {kind} crossing map at {cells} cells"))?;
    std::fs::create_dir_all(dir)?;
    let mut files = map.files()?;
    let manifest = ae_vfs::Manifest { name: name.clone(), version: "1".into(), kind: "map".into(), depends: Vec::new() };
    files.insert(ae_vfs::MANIFEST.into(), format!("{}\n", serde_json::to_string_pretty(&manifest)?).into_bytes());
    for (p, bytes) in &files {
        std::fs::write(dir.join(p), bytes)?;
    }
    println!("{}: map {name}, {cells}x{cells} cells, {} terrain types{}", dir.display(), map.info.types.len(), if map.info.water_level.is_some() { ", water" } else { "" });
    Ok(())
}
