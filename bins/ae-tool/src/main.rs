//! Developer tool.
//!
//! ```text
//! ae-tool bench list
//! ae-tool bench <suite>|all [--small] [--quick] [--threads N] [--label NAME] [--out DIR] [--no-save]
//! ae-tool bench-diff <base> <head> [--time-tolerance PCT] [--all]
//! ae-tool navbench [--scaling ...]   # runs ae_nav's navbench binary
//! ae-tool catalog-fmt FILE...          # rewrite catalogue files in canonical form
//! ae-tool pack DIR OUT [--module GAME] [--with DIR]...  # a package directory as a deterministic zip
//! ae-tool map-export land|water CELLS DIR [NAME]  # a generated map as a map package
//! ae-tool unpack-list FILE             # an archive's files and content hash
//! ae-tool model convert IN.gltf|IN.glb OUT.aemodel  # glTF 2.0 to the engine model (M11)
//! ae-tool model info FILE.aemodel      # its parts, sizes and load time
//! ae-tool capture-diff DIR-A DIR-B      # compare AE_TEST_CAPTURE_DIR PPMs byte for byte
//! ae-tool hooks install                # point git at the tracked .githooks/
//! ```
//! `<base>` and `<head>` are report files or directories of reports
//! (`bench/results/<label>`). `bench-diff` exits with status 1 on a regression.

use ae_bench::Report;
use ae_bench::diff::{Thresholds, diff};
use ae_bench::measure::Budget;
use ae_bench::suites::{self, Options};
use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "usage:
  ae-tool bench list
  ae-tool bench <suite>|all [--small] [--quick] [--threads N] [--label NAME] [--out DIR] [--no-save]
  ae-tool bench-diff <base> <head> [--time-tolerance PCT] [--all]
  ae-tool navbench [navbench arguments, e.g. --scaling]
  ae-tool catalog-fmt FILE...
  ae-tool pack DIR OUT [--module GAME] [--with DIR]...
  ae-tool map-export land|water CELLS DIR [NAME]
  ae-tool unpack-list FILE
  ae-tool model convert IN.gltf|IN.glb OUT.aemodel
  ae-tool model info FILE.aemodel
  ae-tool capture-diff DIR-A DIR-B
  ae-tool hooks install";

/// Runs `ae_nav`'s `navbench` binary from the same target directory
/// (`cargo build --release -p ae_nav --bin navbench`), passing the arguments
/// through.
fn navbench(args: &[String]) -> Result<ExitCode> {
    let exe = std::env::current_exe().context("locating ae-tool")?;
    let path = exe.with_file_name(format!("navbench{}", std::env::consts::EXE_SUFFIX));
    if !path.exists() {
        bail!(
            "{} not found; build it with `cargo build --release -p ae_nav --bin navbench`",
            path.display()
        );
    }
    let status = std::process::Command::new(&path)
        .args(args)
        .status()
        .with_context(|| format!("running {}", path.display()))?;
    Ok(ExitCode::from(status.code().unwrap_or(1).clamp(0, 255) as u8))
}

/// Rewrite catalogue files in canonical form (`ae_catalog::CatalogFile::write`).
fn catalog_fmt(files: &[String]) -> Result<()> {
    if files.is_empty() {
        bail!("catalog-fmt needs at least one file");
    }
    for f in files {
        let text = std::fs::read_to_string(f).with_context(|| format!("read {f}"))?;
        let canonical = ae_catalog::CatalogFile::parse(&text)
            .with_context(|| f.clone())?
            .write();
        if canonical != text {
            std::fs::write(f, &canonical).with_context(|| format!("write {f}"))?;
            println!("{f}: rewritten");
        } else {
            println!("{f}: already canonical");
        }
    }
    Ok(())
}

/// `model convert IN OUT` / `model info FILE`.
fn model(args: &[String]) -> Result<()> {
    match args {
        [cmd, input, output] if cmd == "convert" => {
            let start = std::time::Instant::now();
            let converted = ae_gltf::convert_file(Path::new(input)).with_context(|| input.clone())?;
            for w in &converted.warnings {
                eprintln!("warning: {w}");
            }
            let bytes = converted.model.to_bytes();
            std::fs::write(output, &bytes).with_context(|| format!("write {output}"))?;
            println!(
                "{output}: {} bytes, converted in {:.1} ms",
                bytes.len(),
                start.elapsed().as_secs_f64() * 1e3
            );
            model_info(output)
        }
        [cmd, file] if cmd == "info" => model_info(file),
        _ => bail!("usage: ae-tool model convert IN OUT | ae-tool model info FILE"),
    }
}

fn model_info(file: &str) -> Result<()> {
    let bytes = std::fs::read(file).with_context(|| format!("read {file}"))?;
    // Load time: parse and validate, best of five.
    let mut best = f64::INFINITY;
    let mut model = None;
    for _ in 0..5 {
        let start = std::time::Instant::now();
        let m = ae_model::Model::from_bytes(&bytes)?;
        best = best.min(start.elapsed().as_secs_f64());
        model = Some(m);
    }
    let m = model.unwrap();
    let texture_bytes: usize = m.textures.iter().map(Vec::len).sum();
    println!("{file}: {} bytes, loaded in {:.3} ms", bytes.len(), best * 1e3);
    println!(
        "  {} bones, {} joints, {} primitives, {} materials, {} vertices ({} bytes), {} triangles, {} textures ({} bytes), {} clips, {} lights",
        m.bones.len(),
        m.joints.len(),
        m.primitives.len(),
        m.materials.len(),
        m.vertices.len(),
        m.vertices.len() * ae_model::VERTEX_BYTES,
        m.indices.len() / 3,
        m.textures.len(),
        texture_bytes,
        m.clips.len(),
        m.lights.len()
    );
    for (i, l) in m.lights.iter().enumerate() {
        println!(
            "  light {i} {:?} ({}): joint {}, {:.0} cd, range {} m, colour {:?}, angles {:.3}..{:.3} rad, source {:.2} m",
            l.kind, l.name, l.joint, l.intensity, l.range, l.color, l.inner_angle, l.outer_angle, l.size
        );
    }
    println!("  bounds {:?} .. {:?}", m.bounds.min, m.bounds.max);
    let gpu_bytes: usize = m
        .primitives
        .iter()
        .map(|p| p.vertex_count as usize * p.layout.words() * 4)
        .sum();
    println!("  GPU vertices: {gpu_bytes} bytes");
    for (i, p) in m.primitives.iter().enumerate() {
        println!(
            "  primitive {i}: {:?}, material {} ({}), {} vertices, {} triangles",
            p.layout,
            p.material,
            m.materials[p.material as usize].name,
            p.vertex_count,
            p.index_count / 3
        );
    }
    for (i, t) in m.textures.iter().enumerate() {
        let img = ae_model::ktx::read(t)?;
        println!(
            "  texture {i} {}: VkFormat {} {}x{}, {} levels, {} bytes",
            m.texture_names[i],
            img.format.0,
            img.width,
            img.height,
            img.levels.len(),
            t.len()
        );
    }
    for c in &m.clips {
        println!("  clip {}: {:.2} s, {} channels", c.name, c.duration, c.channels.len());
    }
    Ok(())
}

/// `ae-tool hooks install`: point git at the tracked hooks directory.
///
/// Git does not clone hooks, so the repository carries them in `.githooks/` and
/// every clone runs this once: `git config core.hooksPath .githooks`. Safe to
/// run twice; the hook itself refuses a commit the formatter or clippy rejects.
fn hooks_install() -> Result<()> {
    let root = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("run git rev-parse --show-toplevel")?;
    ensure!(root.status.success(), "not a git repository");
    let root = PathBuf::from(String::from_utf8_lossy(&root.stdout).trim());
    let hook = root.join(".githooks").join("pre-commit");
    ensure!(
        hook.is_file(),
        "{} is missing; run this from the AssembleEngine checkout",
        hook.display()
    );

    let current = Command::new("git")
        .args(["config", "--get", "core.hooksPath"])
        .output()
        .context("run git config")?;
    let current = String::from_utf8_lossy(&current.stdout).trim().to_string();
    if current == ".githooks" {
        println!("core.hooksPath is already .githooks; the pre-commit hook is installed");
        return Ok(());
    }
    let status = Command::new("git")
        .args(["config", "core.hooksPath", ".githooks"])
        .status()
        .context("run git config core.hooksPath")?;
    ensure!(status.success(), "git config core.hooksPath .githooks failed");
    if current.is_empty() {
        println!("core.hooksPath = .githooks: the pre-commit hook is installed");
    } else {
        println!("core.hooksPath = .githooks (was {current}): the pre-commit hook is installed");
    }
    println!("it checks `tools/fmt.sh --check` and `cargo clippy --workspace --all-targets -- -D warnings`");
    println!("on the staged files' kinds; `git commit --no-verify` skips it");
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
        Some("model") => model(&args[1..]).map(|()| ExitCode::SUCCESS),
        Some("capture-diff") => capture_diff(&args[1..]),
        Some("hooks") => match args[1..] {
            [ref cmd] if cmd == "install" => hooks_install().map(|()| ExitCode::SUCCESS),
            _ => {
                eprintln!("usage: ae-tool hooks install");
                Ok(ExitCode::from(2))
            }
        },
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

/// `ae-tool capture-diff DIR-A DIR-B`: the binary PPMs written by two
/// `AE_TEST_CAPTURE_DIR` runs, compared byte for byte. Exits 1 when a file
/// is missing, extra or different.
fn capture_diff(args: &[String]) -> Result<ExitCode> {
    let [a, b] = args else {
        bail!("expected <dir-a> <dir-b>")
    };
    fn captures(root: &Path) -> Result<std::collections::BTreeMap<String, PathBuf>> {
        fn walk(root: &Path, dir: &Path, out: &mut std::collections::BTreeMap<String, PathBuf>) -> Result<()> {
            for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    walk(root, &path, out)?;
                } else if path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("ppm")) {
                    let relative = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().into_owned();
                    out.insert(relative, path);
                }
            }
            Ok(())
        }
        let mut out = std::collections::BTreeMap::new();
        walk(root, root, &mut out)?;
        Ok(out)
    }
    let left = captures(Path::new(a))?;
    let right = captures(Path::new(b))?;
    let mut problems = 0u32;
    for (name, path) in &left {
        match right.get(name) {
            None => {
                println!("missing in {}: {name}", b);
                problems += 1;
            }
            Some(other) => {
                let mine = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
                let theirs = std::fs::read(other).with_context(|| format!("read {}", other.display()))?;
                if mine != theirs {
                    println!("differs: {name} ({} bytes vs {} bytes)", mine.len(), theirs.len());
                    problems += 1;
                }
            }
        }
    }
    for name in right.keys() {
        if !left.contains_key(name) {
            println!("missing in {a}: {name}");
            problems += 1;
        }
    }
    if problems == 0 {
        println!("{} PPM capture(s) identical", left.len());
        Ok(ExitCode::SUCCESS)
    } else {
        println!("{problems} capture difference(s)");
        Ok(ExitCode::from(1))
    }
}

/// Arguments split into positionals, value flags (`--name value`) and switches.
struct Args {
    positional: Vec<String>,
    values: Vec<(String, String)>,
    switches: Vec<String>,
}

impl Args {
    fn parse(args: &[String], value_flags: &[&str], switch_flags: &[&str]) -> Result<Self> {
        let mut parsed = Args {
            positional: Vec::new(),
            values: Vec::new(),
            switches: Vec::new(),
        };
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
        self.values
            .iter()
            .find(|(name, _)| name == flag)
            .map(|(_, value)| value.as_str())
    }

    fn switch(&self, flag: &str) -> bool {
        self.switches.iter().any(|s| s == flag)
    }
}

fn bench(args: &[String]) -> Result<()> {
    let args = Args::parse(
        args,
        &["--threads", "--label", "--out"],
        &["--small", "--quick", "--no-save"],
    )?;
    let [target] = args.positional.as_slice() else {
        bail!("expected one suite name, `all` or `list`")
    };
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

    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8);
    let options = Options {
        budget: if args.switch("--quick") {
            Budget::QUICK
        } else {
            Budget::FULL
        },
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
    let [base, head] = args.positional.as_slice() else {
        bail!("expected <base> <head>")
    };
    let mut thresholds = Thresholds::default();
    if let Some(text) = args.value("--time-tolerance") {
        thresholds.time = text.parse::<f64>().context("--time-tolerance must be a percentage")? / 100.0;
    }
    let result = diff(
        &Report::load_all(Path::new(base))?,
        &Report::load_all(Path::new(head))?,
        thresholds,
    );
    result.print(args.switch("--all"));
    Ok(if result.regressions() > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    })
}

/// `ae-tool pack DIR OUT [--module GAME] [--with DIR]...`: DIR (a
/// package: it needs a valid `manifest.json`) as a deterministic zip
/// archive at OUT; with `--module`, game GAME (`games/GAME`, or a `.wasm`
/// file) built and added as `module.wasm`; with `--with`, another
/// package's files (not its manifest) bundled in, e.g. a map. So one
/// archive can be the whole game.
fn pack(args: &[String]) -> Result<()> {
    let [dir, out, rest @ ..] = args else {
        anyhow::bail!("usage: ae-tool pack DIR OUT [--module GAME] [--with DIR]...")
    };
    let (mut module, mut with) = (None, Vec::new());
    let mut it = rest.iter();
    while let Some(flag) = it.next() {
        let v = it.next().with_context(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--module" => module = Some(v),
            "--with" => with.push(v),
            f => anyhow::bail!("unknown pack option {f} (--module GAME, --with DIR)"),
        }
    }
    let p = ae_vfs::Package::open(dir)?;
    let mut files = ae_vfs::dir_files(std::path::Path::new(dir))?;
    // Other packages' files bundled in (their manifests left out): one
    // archive that is the whole game, e.g. with its map.
    for w in &with {
        for (path, bytes) in ae_vfs::dir_files(std::path::Path::new(w))? {
            if path == ae_vfs::MANIFEST {
                continue;
            }
            anyhow::ensure!(!files.contains_key(&path), "{path} is in {dir} and in {w}");
            files.insert(path, bytes);
        }
    }
    if let Some(game) = module {
        anyhow::ensure!(!files.contains_key("module.wasm"), "{dir} already has a module.wasm");
        files.insert("module.wasm".into(), ae_sim::build::game(game)?);
    }
    let zip = ae_vfs::zip::write(&files)?;
    std::fs::write(out, &zip)?;
    let mut v = ae_vfs::Vfs::new();
    v.mount_archive(out)?;
    println!(
        "{out}: {}@{} ({}), {} files, {} bytes, content hash {:032x}",
        p.manifest.name,
        p.manifest.version,
        p.manifest.kind,
        files.len(),
        zip.len(),
        v.content_hash()?
    );
    Ok(())
}

/// `ae-tool unpack-list FILE`: the files of an archive, with sizes, and its
/// content hash (every entry is read and checked).
fn unpack_list(args: &[String]) -> Result<()> {
    let [file] = args else {
        anyhow::bail!("usage: ae-tool unpack-list FILE")
    };
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
    let name = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("{kind}-crossing-{cells}"));
    let e = cells as f32 * ae_nav::service::NavConfig::default().cell_size;
    let map = ae_map::GameMap::from_terrain(
        &terrain,
        vec![[e * 0.15, e * 0.15], [e * 0.85, e * 0.85]],
        &format!("The {kind} crossing map at {cells} cells"),
    )?;
    std::fs::create_dir_all(dir)?;
    let mut files = map.files()?;
    let manifest = ae_vfs::Manifest {
        name: name.clone(),
        version: "1".into(),
        kind: "map".into(),
        depends: Vec::new(),
    };
    files.insert(
        ae_vfs::MANIFEST.into(),
        format!("{}\n", serde_json::to_string_pretty(&manifest)?).into_bytes(),
    );
    for (p, bytes) in &files {
        std::fs::write(dir.join(p), bytes)?;
    }
    println!(
        "{}: map {name}, {cells}x{cells} cells, {} terrain types{}",
        dir.display(),
        map.info.types.len(),
        if map.info.water_level.is_some() { ", water" } else { "" }
    );
    Ok(())
}
