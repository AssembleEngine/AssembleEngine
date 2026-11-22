//! Render module hot reload: a thread that watches a module's sources (an
//! in-repo module's `render/<name>` directory: `src/`, `shaders/`,
//! `build.rs`, `Cargo.toml`), its `.wasm` file, or a graph file (`.json`)
//! and the directory it is in, rebuilds it when they
//! change and hands the new bytes to the render thread, which swaps
//! modules at its next frame (or keeps the running one if the new one is
//! rejected).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime};

pub struct ModuleWatch {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

/// Newest modification time under `paths` (files, and directories walked).
fn newest(paths: &[PathBuf]) -> Option<SystemTime> {
    fn walk(path: &Path, newest: &mut Option<SystemTime>) {
        let Ok(meta) = std::fs::metadata(path) else { return };
        if meta.is_dir() {
            for entry in std::fs::read_dir(path).into_iter().flatten().flatten() {
                walk(&entry.path(), newest);
            }
        } else if let Ok(modified) = meta.modified() {
            *newest = (*newest).max(Some(modified));
        }
    }
    let mut out = None;
    for p in paths {
        walk(p, &mut out);
    }
    out
}

impl ModuleWatch {
    /// Watch module `name` (as `ae_render::module::build` takes it); the
    /// running module came from `current`.
    pub fn start(name: String, current: Vec<u8>, reload: Arc<Mutex<Option<Vec<u8>>>>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let paths: Vec<PathBuf> = if name.ends_with(".wasm") {
            vec![PathBuf::from(&name)]
        } else if name.ends_with(".json") {
            // A graph file and the shaders beside it (paths are relative to it).
            let file = PathBuf::from(&name);
            let dir = file.parent().map(Path::to_path_buf).unwrap_or_default();
            vec![file, dir]
        } else {
            let dir = ae_render::module::module_dir(&name);
            ["src", "shaders", "build.rs", "Cargo.toml"]
                .iter()
                .map(|p| dir.join(p))
                .collect()
        };
        let handle = std::thread::Builder::new()
            .name("module watch".into())
            .spawn(move || {
                let mut seen = newest(&paths);
                let mut last = current;
                while !stop2.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(400));
                    let now = newest(&paths);
                    if now == seen {
                        continue;
                    }
                    seen = now;
                    eprintln!("render module `{name}` changed; rebuilding");
                    match ae_render::module::build(&name) {
                        Ok(bytes) if bytes == last => {
                            eprintln!("render module `{name}`: same bytes, nothing to reload")
                        }
                        Ok(bytes) => {
                            eprintln!("render module `{name}`: {} bytes, reloading", bytes.len());
                            last = bytes.clone();
                            *reload.lock().unwrap_or_else(|p| p.into_inner()) = Some(bytes);
                        }
                        Err(error) => eprintln!("render module `{name}`: {error:#}"),
                    }
                }
            })
            .expect("spawn the module watcher");
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for ModuleWatch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
