//! Developer tool.
//!
//! ```text
//! ae-tool bench list
//! ae-tool bench <suite>|all [--small] [--quick] [--threads N] [--label NAME] [--out DIR] [--no-save]
//! ae-tool bench-diff <base> <head> [--time-tolerance PCT] [--all]
//! ae-tool navbench [--scaling ...]   # runs ae_nav's navbench binary
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
  ae-tool navbench [navbench arguments, e.g. --scaling]";

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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("bench") => bench(&args[1..]).map(|()| ExitCode::SUCCESS),
        Some("bench-diff") => bench_diff(&args[1..]),
        Some("navbench") => navbench(&args[1..]),
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
