//! Content file system.
//!
//! Content paths are relative, `/`-separated (`units/tank.json`). A [`Vfs`]
//! is a stack of mounts; a later mount overrides an earlier one file by
//! file, so a mod can replace single files of a game. A mount is a
//! directory or a zip archive ([`zip`]); both look the same from above.
//!
//! A package is a directory or an archive with a `manifest.json` at its
//! root ([`Manifest`]: name, version, kind, the packages it depends on).
//! [`Library`] finds packages in directories and resolves one with its
//! dependencies into a mount order (dependencies first, so a package
//! overrides what it builds on). The manifest itself is package metadata:
//! it is not a content file and not part of the content hash.
//!
//! [`Vfs::content_hash`] hashes every visible file (path and bytes, in path
//! order): two machines with the same hash build the same game, and replay
//! and lockstep headers carry it so different content is refused by name
//! instead of desyncing.

use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod zip;

pub use zip::Archive;

/// A package's metadata file, at its root.
pub const MANIFEST: &str = "manifest.json";

#[derive(Clone, Debug)]
enum Mount {
    Dir(PathBuf),
    Archive(Arc<Archive>),
}

impl Mount {
    fn read(&self, path: &str) -> Option<Result<Vec<u8>>> {
        match self {
            Mount::Dir(dir) => {
                let file = dir.join(path);
                file.is_file().then(|| std::fs::read(&file).with_context(|| format!("read {}", file.display())))
            }
            Mount::Archive(a) => a.read(path),
        }
    }

    fn has(&self, path: &str) -> bool {
        match self {
            Mount::Dir(dir) => dir.join(path).is_file(),
            Mount::Archive(a) => a.contains(path),
        }
    }

    fn walk(&self, f: &mut dyn FnMut(String)) -> Result<()> {
        match self {
            Mount::Dir(dir) => walk(dir, dir, f),
            Mount::Archive(a) => {
                a.paths().for_each(|p| f(p.to_owned()));
                Ok(())
            }
        }
    }
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

    /// Mount a zip archive on top of the existing mounts.
    pub fn mount_archive(&mut self, file: impl AsRef<Path>) -> Result<&mut Self> {
        self.mounts.push(Mount::Archive(Arc::new(Archive::open(file.as_ref())?)));
        Ok(self)
    }

    /// Mount a directory or an archive file.
    pub fn mount(&mut self, path: impl AsRef<Path>) -> Result<&mut Self> {
        let path = path.as_ref();
        if path.is_dir() { self.mount_dir(path) } else { self.mount_archive(path) }
    }

    /// Mount `packages` in order (as [`Library::resolve`] gives them).
    pub fn mount_packages(&mut self, packages: &[Package]) -> Result<&mut Self> {
        for p in packages {
            self.mount(&p.source)?;
        }
        Ok(self)
    }

    /// The bytes of `path` from the topmost mount that has it.
    pub fn read(&self, path: &str) -> Result<Vec<u8>> {
        check_path(path)?;
        ensure!(path != MANIFEST, "{MANIFEST} is package metadata, not content");
        for m in self.mounts.iter().rev() {
            if let Some(r) = m.read(path) {
                return r;
            }
        }
        bail!("no content file {path:?} in {} mount(s)", self.mounts.len())
    }

    pub fn read_string(&self, path: &str) -> Result<String> {
        String::from_utf8(self.read(path)?).with_context(|| format!("{path} is not UTF-8"))
    }

    pub fn exists(&self, path: &str) -> bool {
        check_path(path).is_ok() && path != MANIFEST && self.mounts.iter().any(|m| m.has(path))
    }

    /// Every visible file path, sorted.
    pub fn files(&self) -> Result<Vec<String>> {
        let mut all = BTreeMap::new();
        for (i, m) in self.mounts.iter().enumerate() {
            m.walk(&mut |p| {
                if p != MANIFEST {
                    all.insert(p, i);
                }
            })?;
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

/// Every file of a directory, by content path (for packing it).
pub fn dir_files(dir: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut paths = Vec::new();
    walk(dir, dir, &mut |p| paths.push(p))?;
    paths.into_iter().map(|p| Ok((p.clone(), std::fs::read(dir.join(&p)).with_context(|| format!("read {p}"))?))).collect()
}

/// A package's `manifest.json`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Letters, digits, `-` and `_`.
    pub name: String,
    pub version: String,
    /// `game`, `map`, `mod` or `library`.
    pub kind: String,
    /// Package names, each optionally `name@version` (that exact version).
    #[serde(default)]
    pub depends: Vec<String>,
}

pub const KINDS: [&str; 4] = ["game", "map", "mod", "library"];

impl Manifest {
    pub fn parse(text: &str) -> Result<Self> {
        let m: Manifest = serde_json::from_str(text).context("manifest.json")?;
        let ok = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.');
        ensure!(ok(&m.name) && !m.name.contains('.'), "package name {:?}: letters, digits, `-` and `_`", m.name);
        ensure!(ok(&m.version), "package {} version {:?}", m.name, m.version);
        ensure!(KINDS.contains(&m.kind.as_str()), "package {} kind {:?} (one of {KINDS:?})", m.name, m.kind);
        for d in &m.depends {
            let (n, v) = d.split_once('@').unwrap_or((d, "x"));
            ensure!(ok(n) && ok(v), "package {} depends on {d:?}", m.name);
        }
        Ok(m)
    }
}

/// A package found by [`Library::scan`] or opened with [`Package::open`].
#[derive(Clone, Debug)]
pub struct Package {
    pub manifest: Manifest,
    /// Its directory or archive file.
    pub source: PathBuf,
}

impl Package {
    /// The package at `source` (a directory or an archive with a manifest).
    pub fn open(source: impl Into<PathBuf>) -> Result<Self> {
        let source = source.into();
        let text = if source.is_dir() {
            std::fs::read_to_string(source.join(MANIFEST)).with_context(|| format!("{} has no {MANIFEST}", source.display()))?
        } else {
            let a = Archive::open(&source)?;
            String::from_utf8(a.read(MANIFEST).with_context(|| format!("{} has no {MANIFEST}", source.display()))??)?
        };
        Ok(Self { manifest: Manifest::parse(&text).with_context(|| format!("package {}", source.display()))?, source })
    }
}

/// Packages by name (and version), from directories of packages.
#[derive(Clone, Debug, Default)]
pub struct Library {
    packages: BTreeMap<(String, String), Package>,
}

impl Library {
    /// Every package directly inside `dirs`: subdirectories with a
    /// manifest and zip files (any extension) with one. Other entries are
    /// skipped; a package name and version found twice is an error.
    pub fn scan(dirs: &[impl AsRef<Path>]) -> Result<Self> {
        let mut lib = Self::default();
        for d in dirs {
            let d = d.as_ref();
            let mut entries: Vec<_> = std::fs::read_dir(d).with_context(|| format!("read {}", d.display()))?.collect::<Result<_, _>>()?;
            entries.sort_by_key(|e| e.file_name());
            for e in entries {
                let path = e.path();
                let is_package = if path.is_dir() { path.join(MANIFEST).is_file() } else { std::fs::File::open(&path).ok().and_then(|mut f| { let mut sig = [0u8; 4]; std::io::Read::read_exact(&mut f, &mut sig).ok().map(|()| sig == *b"PK\x03\x04") }).unwrap_or(false) };
                if is_package {
                    lib.add(Package::open(path)?)?;
                }
            }
        }
        Ok(lib)
    }

    pub fn add(&mut self, p: Package) -> Result<()> {
        let key = (p.manifest.name.clone(), p.manifest.version.clone());
        if let Some(old) = self.packages.get(&key) {
            bail!("package {}@{} twice: {} and {}", key.0, key.1, old.source.display(), p.source.display());
        }
        self.packages.insert(key, p);
        Ok(())
    }

    /// `name` or `name@version`: that version, or the only version there is.
    pub fn get(&self, spec: &str) -> Result<&Package> {
        let (name, version) = match spec.split_once('@') {
            Some((n, v)) => (n, Some(v)),
            None => (spec, None),
        };
        let found: Vec<&Package> = self.packages.values().filter(|p| p.manifest.name == name && version.is_none_or(|v| p.manifest.version == v)).collect();
        match found[..] {
            [p] => Ok(p),
            [] => bail!("no package {spec} (have: {})", self.packages.keys().map(|(n, v)| format!("{n}@{v}")).collect::<Vec<_>>().join(", ")),
            _ => bail!("package {name} in several versions ({}): name one", found.iter().map(|p| p.manifest.version.as_str()).collect::<Vec<_>>().join(", ")),
        }
    }

    /// `spec` and everything it depends on, dependencies first (each once,
    /// in the order the manifests list them); a cycle is an error.
    pub fn resolve(&self, spec: &str) -> Result<Vec<Package>> {
        fn visit(lib: &Library, spec: &str, path: &mut Vec<String>, out: &mut Vec<Package>) -> Result<()> {
            let p = lib.get(spec).with_context(|| if let Some(by) = path.last() { format!("needed by {by}") } else { "the package to run".into() })?;
            let id = format!("{}@{}", p.manifest.name, p.manifest.version);
            ensure!(!path.contains(&id), "dependency cycle: {} -> {id}", path.join(" -> "));
            if out.iter().any(|o| o.manifest.name == p.manifest.name) {
                ensure!(out.iter().any(|o| o.manifest == p.manifest), "package {} needed in two versions", p.manifest.name);
                return Ok(());
            }
            path.push(id);
            for d in &p.manifest.depends {
                visit(lib, d, path, out)?;
            }
            path.pop();
            out.push(p.clone());
            Ok(())
        }
        let mut out = Vec::new();
        visit(self, spec, &mut Vec::new(), &mut out)?;
        Ok(out)
    }
}

/// The packages `specs` name (package paths, or `name[@version]` found in
/// the libraries `dirs`), each after its dependencies, each once; one
/// name in two versions is an error.
pub fn resolve_specs(dirs: &[impl AsRef<Path>], specs: &[String]) -> Result<Vec<Package>> {
    let mut lib = if dirs.is_empty() { Library::default() } else { Library::scan(dirs)? };
    let mut out: Vec<Package> = Vec::new();
    for spec in specs {
        let name = if Path::new(spec).exists() {
            let p = Package::open(spec)?;
            let name = format!("{}@{}", p.manifest.name, p.manifest.version);
            if lib.get(&name).is_err() {
                lib.add(p)?;
            }
            name
        } else {
            spec.clone()
        };
        for p in lib.resolve(&name)? {
            match out.iter().find(|o| o.manifest.name == p.manifest.name) {
                Some(o) => ensure!(o.manifest.version == p.manifest.version, "package {} in versions {} and {}", p.manifest.name, o.manifest.version, p.manifest.version),
                None => out.push(p),
            }
        }
    }
    Ok(out)
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

    fn manifest(name: &str, kind: &str, depends: &[&str]) -> String {
        format!("{{\"name\": \"{name}\", \"version\": \"1\", \"kind\": \"{kind}\", \"depends\": {depends:?}}}")
    }

    #[test]
    fn archives_mount_like_directories() {
        let base = dir("arc-base", &[("units/tank.json", "base tank"), ("units/scout.json", "scout")]);
        let mut files = BTreeMap::new();
        files.insert("units/tank.json".to_string(), b"zip tank".to_vec());
        files.insert(MANIFEST.to_string(), manifest("tanks", "mod", &[]).into_bytes());
        let zip = base.with_extension("aepkg");
        std::fs::write(&zip, zip::write(&files).unwrap()).unwrap();
        let mut v = Vfs::new();
        v.mount(&base).unwrap().mount(&zip).unwrap();
        assert_eq!(v.read_string("units/tank.json").unwrap(), "zip tank");
        assert_eq!(v.files().unwrap(), ["units/scout.json", "units/tank.json"], "the manifest is not content");
        assert!(v.read(MANIFEST).is_err() && !v.exists(MANIFEST));
        // An archive of a directory hashes like the directory.
        let mut packed = Vfs::new();
        let z2 = base.with_extension("zip");
        std::fs::write(&z2, zip::write(&dir_files(&base).unwrap()).unwrap()).unwrap();
        packed.mount(&z2).unwrap();
        let mut plain = Vfs::new();
        plain.mount(&base).unwrap();
        assert_eq!(packed.content_hash().unwrap(), plain.content_hash().unwrap());
        for p in [zip, z2] {
            std::fs::remove_file(p).ok();
        }
        std::fs::remove_dir_all(base).ok();
    }

    #[test]
    fn libraries_resolve_dependencies_first() {
        let root = std::env::temp_dir().join(format!("ae-vfs-{}-lib", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let pkg = |name: &str, kind: &str, deps: &[&str], file: &str| {
            let d = root.join(name);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join(MANIFEST), manifest(name, kind, deps)).unwrap();
            std::fs::write(d.join("who.txt"), file).unwrap();
        };
        pkg("core", "library", &[], "core");
        pkg("game", "game", &["core"], "game");
        pkg("map", "map", &["core"], "map");
        pkg("match", "mod", &["game", "map@1"], "match");
        pkg("loop_a", "mod", &["loop_b"], "a");
        pkg("loop_b", "mod", &["loop_a"], "b");
        pkg("broken", "mod", &["nowhere"], "x");
        // Not a package: no manifest.
        std::fs::create_dir_all(root.join("stray")).unwrap();
        let lib = Library::scan(&[&root]).unwrap();
        let order: Vec<String> = lib.resolve("match").unwrap().into_iter().map(|p| p.manifest.name).collect();
        assert_eq!(order, ["core", "game", "map", "match"]);
        let mut v = Vfs::new();
        v.mount_packages(&lib.resolve("match").unwrap()).unwrap();
        assert_eq!(v.read_string("who.txt").unwrap(), "match", "the package overrides what it builds on");
        let e = format!("{:#}", lib.resolve("loop_a").unwrap_err());
        assert!(e.contains("cycle"), "{e}");
        let e = format!("{:#}", lib.resolve("broken").unwrap_err());
        assert!(e.contains("no package nowhere") && e.contains("needed by broken@1"), "{e}");
        assert!(lib.get("game@2").is_err());
        assert!(Manifest::parse(&manifest("bad name", "game", &[])).is_err());
        assert!(Manifest::parse(&manifest("x", "weapon", &[])).is_err());
        assert!(Manifest::parse(r#"{"name": "x", "version": "1", "kind": "game", "extra": 1}"#).is_err());
        std::fs::remove_dir_all(root).ok();
    }
}
