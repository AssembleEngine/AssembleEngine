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
pub use ae_model::pose::PlayMode;

/// A class's model (`None`: every class but the observer's blips).
#[derive(Clone, Debug)]
pub struct ClassModel {
    pub class: Option<u32>,
    pub model: Arc<Model>,
}

/// One clip choice: the model's clip number from 1 (0: the rest pose) and
/// how it plays past its end.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClipPlay {
    pub clip: u32,
    pub mode: PlayMode,
}

impl ClipPlay {
    pub const fn looping(clip: u32) -> Self {
        Self { clip, mode: PlayMode::Loop }
    }

    /// `N[:loop|pingpong|once]`.
    pub fn parse(text: &str) -> Result<Self> {
        let (n, mode) = text.split_once(':').unwrap_or((text, "loop"));
        let clip = n.trim().parse().with_context(|| format!("--clip: {n:?} is not a clip number"))?;
        let mode = match mode.trim() {
            "loop" => PlayMode::Loop,
            "pingpong" | "ping-pong" => PlayMode::PingPong,
            "once" => PlayMode::Once,
            other => bail!("--clip: play mode {other:?}: expected loop, pingpong or once"),
        };
        Ok(Self { clip, mode })
    }

    /// The stream's `clip` word.
    pub fn word(self) -> u32 {
        ae_render::units::clip_word(self.clip, self.mode)
    }
}

/// Which clip units play, for units moving and units standing still.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClipChoice {
    pub moving: ClipPlay,
    pub still: ClipPlay,
}

impl ClipChoice {
    /// `MOVING[,STILL]`, each `N[:loop|pingpong|once]`.
    pub fn parse(text: &str) -> Result<Self> {
        let mut parts = text.split(',');
        let moving = ClipPlay::parse(parts.next().context("--clip MOVING[,STILL]")?)?;
        let still = parts.next().map(ClipPlay::parse).transpose()?.unwrap_or_default();
        ensure!(parts.next().is_none(), "--clip takes one or two clips");
        Ok(Self { moving, still })
    }

    /// The clip of a unit.
    pub fn of(self, moving: bool) -> ClipPlay {
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
/// every class but `blip`. Classes past `classes` are left out (the caller
/// refuses them).
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

/// A unit's clip time `ticks` after its clip started (seconds of sim
/// time): looping clips start at a phase of their own per unit, so a crowd
/// does not move in step; a clip played once starts at its beginning.
pub fn clip_time(key: u64, play: ClipPlay, ticks: u64, tick_seconds: f32) -> f32 {
    let phase = match play.mode {
        PlayMode::Once => 0.0,
        _ => (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 40) as f64 / (1u64 << 24) as f64 * 8.0,
    };
    (ticks as f64 * tick_seconds as f64 + phase) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clips_and_classes() {
        let (l, p, o) = (PlayMode::Loop, PlayMode::PingPong, PlayMode::Once);
        assert_eq!(ClipChoice::parse("2").unwrap(), ClipChoice { moving: ClipPlay { clip: 2, mode: l }, still: ClipPlay { clip: 0, mode: l } });
        assert_eq!(ClipChoice::parse("2,1:pingpong").unwrap(), ClipChoice { moving: ClipPlay { clip: 2, mode: l }, still: ClipPlay { clip: 1, mode: p } });
        assert_eq!(ClipChoice::parse("3:once").unwrap().moving, ClipPlay { clip: 3, mode: o });
        assert!(ClipChoice::parse("2,1,3").is_err());
        assert!(ClipChoice::parse("x").is_err());
        assert!(ClipChoice::parse("2:bounce").is_err());
        assert_eq!(ClipPlay { clip: 1, mode: p }.word(), 1 | 1 << 30);
        let a = Arc::new(Model::default());
        let b = Arc::new(Model::default());
        let given = [ClassModel { class: None, model: a.clone() }, ClassModel { class: Some(1), model: b.clone() }];
        let per = per_class(&given, 4, Some(3));
        assert!(Arc::ptr_eq(per[0].as_ref().unwrap(), &a) && Arc::ptr_eq(per[1].as_ref().unwrap(), &b) && per[2].is_some() && per[3].is_none());
        assert!(per_class(&[], 4, None).is_empty());
        let walk = ClipPlay::looping(2);
        let t = clip_time(7, walk, 300, 1.0 / 30.0);
        assert!((10.0..18.0).contains(&t), "{t}");
        assert_ne!(clip_time(7, walk, 0, 0.1), clip_time(8, walk, 0, 0.1));
        assert_eq!(clip_time(7, ClipPlay { clip: 2, mode: o }, 0, 0.1), 0.0, "once: from the start");
    }
}
