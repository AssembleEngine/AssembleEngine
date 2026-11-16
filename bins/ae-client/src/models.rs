//! `--model [CLASS=]FILE` and `--clip MOVING[,STILL]`: unit classes drawn
//! with a model (M11, `ae_render::models`).
//!
//! FILE is an engine model (`.aemodel`), or a glTF (`.glb`, `.gltf`)
//! converted by `ae-tool model convert` into `target/models/<name>.aemodel`
//! (again when the glTF is newer). The client does not link the converter
//! (its image decoders and the C++ Basis transcoder); `ae-tool` is built
//! into `target/tools` on first use.

use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

pub use ae_model::Model;

/// A class's model (`None`: every class but the observer's blips).
#[derive(Clone, Debug)]
pub struct ClassModel {
    pub class: Option<u32>,
    pub model: Arc<Model>,
}

/// Which clip units play: the model's clip numbers from 1 (0: the rest
/// pose), for units moving and units standing still.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClipChoice {
    pub moving: u32,
    pub still: u32,
}

impl ClipChoice {
    pub fn parse(text: &str) -> Result<Self> {
        let mut parts = text.split(',').map(|p| p.trim().parse::<u32>());
        let moving = parts.next().context("--clip MOVING[,STILL]")?.context("--clip MOVING[,STILL]")?;
        let still = parts.next().transpose().context("--clip MOVING[,STILL]")?.unwrap_or(0);
        ensure!(parts.next().is_none(), "--clip takes one or two clip numbers");
        Ok(Self { moving, still })
    }

    /// The clip of a unit (stream attribute `clip`).
    pub fn of(self, moving: bool) -> u32 {
        if moving { self.moving } else { self.still }
    }
}

/// `--model [CLASS=]FILE`.
pub fn parse_arg(text: &str) -> Result<ClassModel> {
    let (class, file) = match text.split_once('=') {
        Some((c, f)) if !c.is_empty() && c.bytes().all(|b| b.is_ascii_digit()) => (Some(c.parse().context("--model CLASS=FILE")?), f),
        _ => (None, text),
    };
    let model = load(Path::new(file))?;
    eprintln!("model {file}: {} vertices, {} joints, {} clip(s){}", model.vertices.len(), model.joints.len(), model.clips.len(), clip_names(&model));
    Ok(ClassModel { class, model: Arc::new(model) })
}

fn clip_names(model: &Model) -> String {
    if model.clips.is_empty() {
        return String::new();
    }
    let names: Vec<String> = model.clips.iter().enumerate().map(|(k, c)| format!("{} {}", k + 1, c.name)).collect();
    format!(" ({})", names.join(", "))
}

/// Read an engine model, converting a glTF first.
pub fn load(path: &Path) -> Result<Model> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let file = match ext.as_str() {
        "aemodel" => path.to_path_buf(),
        "glb" | "gltf" => converted(path)?,
        _ => bail!("{}: expected a .aemodel, .glb or .gltf file", path.display()),
    };
    let bytes = std::fs::read(&file).with_context(|| format!("read {}", file.display()))?;
    Model::from_bytes(&bytes).with_context(|| format!("{}", file.display()))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The Khronos Fox (the bench's model).
pub fn fox() -> Result<Arc<Model>> {
    Ok(Arc::new(load(&repo_root().join("assets/models/khronos/Fox/Fox.glb"))?))
}

/// `target/models/<stem>.aemodel`, converted when missing or older than
/// the glTF.
fn converted(gltf: &Path) -> Result<PathBuf> {
    let stem = gltf.file_stem().and_then(|s| s.to_str()).context("model file name")?;
    let out = repo_root().join("target/models").join(format!("{stem}.aemodel"));
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let source = modified(gltf).with_context(|| format!("no model {}", gltf.display()))?;
    if modified(&out).is_some_and(|m| m >= source) {
        return Ok(out);
    }
    std::fs::create_dir_all(out.parent().unwrap())?;
    eprintln!("converting {} (ae-tool model convert)", gltf.display());
    let root = repo_root();
    let status = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args(["run", "--release", "--quiet", "-p", "ae-tool", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(root.join("target/tools"))
        .args(["--", "model", "convert"])
        .arg(gltf)
        .arg(&out)
        .status()
        .context("run cargo for ae-tool")?;
    ensure!(status.success(), "converting {} failed", gltf.display());
    Ok(out)
}

/// `UnitsSetup::models` for `classes` classes: each class's model, the
/// last `--model` naming it winning; a model without a class goes to
/// every class but `blip`.
pub fn per_class(models: &[ClassModel], classes: usize, blip: Option<usize>) -> Vec<Option<Arc<Model>>> {
    let mut out = vec![None; if models.is_empty() { 0 } else { classes }];
    for m in models {
        match m.class {
            Some(c) => {
                if let Some(slot) = out.get_mut(c as usize) {
                    *slot = Some(m.model.clone());
                }
            }
            None => {
                for (c, slot) in out.iter_mut().enumerate() {
                    if Some(c) != blip {
                        *slot = Some(m.model.clone());
                    }
                }
            }
        }
    }
    out
}

/// A unit's clip time at `tick` (seconds): the sim's time, offset per unit
/// so a crowd does not move in step.
pub fn clip_time(key: u64, tick: u64, tick_seconds: f32) -> f32 {
    let h = key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 40;
    let phase = h as f64 / (1u64 << 24) as f64 * 8.0;
    (tick as f64 * tick_seconds as f64 + phase) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clips_and_classes() {
        assert_eq!(ClipChoice::parse("2").unwrap(), ClipChoice { moving: 2, still: 0 });
        assert_eq!(ClipChoice::parse("2,1").unwrap(), ClipChoice { moving: 2, still: 1 });
        assert!(ClipChoice::parse("2,1,3").is_err());
        assert!(ClipChoice::parse("x").is_err());
        let a = Arc::new(Model::default());
        let b = Arc::new(Model::default());
        let given = [ClassModel { class: None, model: a.clone() }, ClassModel { class: Some(1), model: b.clone() }];
        let per = per_class(&given, 4, Some(3));
        assert!(Arc::ptr_eq(per[0].as_ref().unwrap(), &a) && Arc::ptr_eq(per[1].as_ref().unwrap(), &b) && per[2].is_some() && per[3].is_none());
        assert!(per_class(&[], 4, None).is_empty());
        let t = clip_time(7, 300, 1.0 / 30.0);
        assert!((10.0..18.0).contains(&t), "{t}");
        assert_ne!(clip_time(7, 0, 0.1), clip_time(8, 0, 0.1));
    }
}
