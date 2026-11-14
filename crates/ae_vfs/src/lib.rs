//! Content file system.
//!
//! Content paths are relative, `/`-separated (`units/tank.json`). A [`Vfs`]
//! is a stack of mounts; a later mount overrides an earlier one file by
//! file, so a mod can replace single files of a game. Today a mount is a
//! directory; archives will be another kind of mount behind the same
//! interface.
//!
//! [`Vfs::content_hash`] hashes every visible file (path and bytes, in path
//! order): two machines with the same hash build the same game, and replay
//! and lockstep headers carry it so different content is refused by name
//! instead of desyncing.

use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
enum Mount {
    Dir(PathBuf),
}

#[derive(Clone, Debug, Default)]
pub struct Vfs {
    mounts: Vec<Mount>,
}

/// A content path: relative, `/`-separated, no `..`, no empty parts.
fn check_path(path: &str) -> Result<()> {
    ensure!(!path.is_empty() && !path.starts_with('/') && !path.contains('\\'), "content path {path:?} must be relative with `/` separators");
    ensure!(path.split('/').all(|p| !p.is_empty() && p != "." && p != ".."), "content path {path:?} has an empty, `.` or `..` part");
    Ok(())
}

impl Vfs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mount a directory on top of the existing mounts.
    pub fn mount_dir(&mut self, dir: impl Into<PathBuf>) -> Result<&mut Self> {
        let dir = dir.into();
        ensure!(dir.is_dir(), "{} is not a directory", dir.display());
        self.mounts.push(Mount::Dir(dir));
        Ok(self)
    }

    /// The bytes of `path` from the topmost mount that has it.
    pub fn read(&self, path: &str) -> Result<Vec<u8>> {
        check_path(path)?;
        for m in self.mounts.iter().rev() {
            match m {
                Mount::Dir(dir) => {
                    let file = dir.join(path);
                    if file.is_file() {
                        return std::fs::read(&file).with_context(|| format!("read {}", file.display()));
                    }
                }
            }
        }
        bail!("no content file {path:?} in {} mount(s)", self.mounts.len())
    }

    pub fn read_string(&self, path: &str) -> Result<String> {
        String::from_utf8(self.read(path)?).with_context(|| format!("{path} is not UTF-8"))
    }

    pub fn exists(&self, path: &str) -> bool {
        check_path(path).is_ok() && self.mounts.iter().any(|m| match m {
            Mount::Dir(dir) => dir.join(path).is_file(),
        })
    }

    /// Every visible file path, sorted.
    pub fn files(&self) -> Result<Vec<String>> {
        let mut all = BTreeMap::new();
        for (i, m) in self.mounts.iter().enumerate() {
            match m {
                Mount::Dir(dir) => walk(dir, dir, &mut |p| {
                    all.insert(p, i);
                })?,
            }
        }
        Ok(all.into_keys().collect())
    }

    /// Files under `prefix` (a directory path, `""` for all) with
    /// extension `ext`, sorted.
    pub fn list(&self, prefix: &str, ext: &str) -> Result<Vec<String>> {
        Ok(self.files()?.into_iter().filter(|p| (prefix.is_empty() || p.starts_with(&format!("{prefix}/"))) && p.ends_with(&format!(".{ext}"))).collect())
    }

    /// xxh3-128 over every visible file's path and bytes, in path order.
    pub fn content_hash(&self) -> Result<u128> {
        let mut h = xxhash_rust::xxh3::Xxh3::new();
        for p in self.files()? {
            let bytes = self.read(&p)?;
            h.update(&(p.len() as u64).to_le_bytes());
            h.update(p.as_bytes());
            h.update(&(bytes.len() as u64).to_le_bytes());
            h.update(&bytes);
        }
        Ok(h.digest128())
    }
}

fn walk(root: &Path, dir: &Path, f: &mut dyn FnMut(String)) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?.collect::<Result<_, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let path = e.path();
        if path.is_dir() {
            walk(root, &path, f)?;
        } else if path.is_file() {
            let rel = path.strip_prefix(root).unwrap();
            let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            f(parts.join("/"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str, files: &[(&str, &str)]) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ae-vfs-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        for (p, text) in files {
            let f = d.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, text).unwrap();
        }
        d
    }

    #[test]
    fn later_mounts_override_and_the_hash_follows_content() {
        let base = dir("base", &[("units/tank.json", "base tank"), ("units/scout.json", "scout"), ("maps/a.txt", "map")]);
        let moded = dir("mod", &[("units/tank.json", "mod tank")]);
        let mut v = Vfs::new();
        v.mount_dir(&base).unwrap();
        let h0 = v.content_hash().unwrap();
        assert_eq!(v.files().unwrap(), ["maps/a.txt", "units/scout.json", "units/tank.json"]);
        v.mount_dir(&moded).unwrap();
        assert_eq!(v.read_string("units/tank.json").unwrap(), "mod tank");
        assert_eq!(v.read_string("units/scout.json").unwrap(), "scout");
        assert_eq!(v.list("units", "json").unwrap(), ["units/scout.json", "units/tank.json"]);
        assert_ne!(v.content_hash().unwrap(), h0, "an override changes the hash");
        let mut again = Vfs::new();
        again.mount_dir(&base).unwrap();
        assert_eq!(again.content_hash().unwrap(), h0, "same files, same hash");
        for bad in ["/etc/passwd", "../x", "a//b", "a\\b", ""] {
            assert!(v.read(bad).is_err(), "{bad}");
        }
        assert!(v.read("units/none.json").is_err());
        for d in [base, moded] {
            std::fs::remove_dir_all(d).ok();
        }
    }
}
