//! `--model [CLASS=]FILE`: unit classes drawn with a model (M11,
//! `ae_render::models`). A model is also the sim's animation model of its
//! file's lower-case stem (`Fox.glb` is model `fox`): units whose `Anim`
//! names it are posed from their channels (`poser.rs`).
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

/// A class's model (`None`: every class but the observer's blips), and
/// its name as the sim's animation model.
#[derive(Clone, Debug)]
pub struct ClassModel {
    pub class: Option<u32>,
    pub name: String,
    pub model: Arc<Model>,
}

/// `--model [CLASS=]FILE`.
pub fn parse_arg(text: &str) -> Result<ClassModel> {
    let (class, file) = match text.split_once('=') {
        Some((c, f)) if !c.is_empty() && c.bytes().all(|b| b.is_ascii_digit()) => (Some(c.parse().context("--model CLASS=FILE")?), f),
        _ => (None, text),
    };
    let model = load(Path::new(file))?;
    let name = Path::new(file).file_stem().and_then(|s| s.to_str()).context("model file name")?.to_ascii_lowercase();
    eprintln!("model {name} ({file}): {} vertices, {} joints, {} clip(s){}", model.vertices.len(), model.joints.len(), model.clips.len(), clip_names(&model));
    Ok(ClassModel { class, name, model: Arc::new(model) })
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

/// The Khronos Fox (the bench's model), as model `fox` for every class.
pub fn fox() -> Result<ClassModel> {
    Ok(ClassModel { class: None, name: "fox".into(), model: Arc::new(load(&repo_root().join("assets/models/khronos/Fox/Fox.glb"))?) })
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
    assign(models, classes, blip).into_iter().map(|m| m.map(|m| m.model.clone())).collect()
}

/// As [`per_class`], each class's model name.
pub fn names_per_class(models: &[ClassModel], classes: usize, blip: Option<usize>) -> Vec<Option<String>> {
    assign(models, classes, blip).into_iter().map(|m| m.map(|m| m.name.clone())).collect()
}

fn assign(models: &[ClassModel], classes: usize, blip: Option<usize>) -> Vec<Option<&ClassModel>> {
    let mut out = vec![None; if models.is_empty() { 0 } else { classes }];
    for m in models {
        match m.class {
            Some(c) => {
                if let Some(slot) = out.get_mut(c as usize) {
                    *slot = Some(m);
                }
            }
            None => {
                for (c, slot) in out.iter_mut().enumerate() {
                    if Some(c) != blip {
                        *slot = Some(m);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        let a = Arc::new(Model::default());
        let b = Arc::new(Model::default());
        let given = [ClassModel { class: None, name: "a".into(), model: a.clone() }, ClassModel { class: Some(1), name: "b".into(), model: b.clone() }];
        let per = per_class(&given, 4, Some(3));
        assert!(Arc::ptr_eq(per[0].as_ref().unwrap(), &a) && Arc::ptr_eq(per[1].as_ref().unwrap(), &b) && per[2].is_some() && per[3].is_none());
        assert_eq!(names_per_class(&given, 4, Some(3)), [Some("a".to_string()), Some("b".into()), Some("a".into()), None]);
        assert!(per_class(&[], 4, None).is_empty());
    }
}
