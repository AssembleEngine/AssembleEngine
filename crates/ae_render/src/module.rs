//! Render modules on the host: loading one with wasmtime, calling its
//! `declare` and `frame` exports (contract in [`ae_render_abi::module`]),
//! and building the in-repo modules (`render/<name>`).
//!
//! A render module is trusted (like the sim module) but gets its own store
//! and memory and no imports: everything it hands back is copied out of its
//! memory with bounds checks, and the graph and program table are
//! validated before use (`ae_framegraph::descriptor::decode`, the pass
//! library's checks). A trap is an error the caller can survive: the scene
//! keeps its last good graph.

use ae_render_abi::module::{self as abi, FrameInput};
use anyhow::{Context as _, Result, bail, ensure};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use wasmtime::{Engine, Instance, Memory, Module, Store, TypedFunc};

/// What `declare` returned: the graph declaration and the program table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Declared {
    pub graph: Vec<u8>,
    pub programs: Vec<u8>,
}

/// What `frame` returned.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModuleFrame {
    /// Bitset of disabled programs (payloads).
    pub disabled: Vec<u32>,
    pub uniforms: Vec<u8>,
}

impl ModuleFrame {
    pub fn is_disabled(&self, program: u32) -> bool {
        self.disabled.get(program as usize / 32).is_some_and(|w| w & (1 << (program % 32)) != 0)
    }
}

pub struct RenderModule {
    store: Store<()>,
    memory: Memory,
    declare: TypedFunc<(u32, u32, u32, u32), u32>,
    frame: TypedFunc<(), u32>,
    input: u32,
    pub loaded: Instant,
}

impl RenderModule {
    pub fn load(engine: &Engine, bytes: &[u8]) -> Result<Self> {
        let module = Module::new(engine, bytes).map_err(anyhow::Error::from).context("compiling the render module")?;
        if let Some(import) = module.imports().next() {
            bail!("render modules import nothing; this one imports `{}::{}`", import.module(), import.name());
        }
        let mut store = Store::new(engine, ());
        let instance = Instance::new(&mut store, &module, &[]).map_err(anyhow::Error::from).context("instantiating the render module")?;
        let memory = instance.get_memory(&mut store, "memory").context("the render module exports no `memory`")?;
        let version = instance.get_typed_func::<(), u32>(&mut store, abi::EXPORT_VERSION).map_err(anyhow::Error::from).with_context(|| format!("export `{}`", abi::EXPORT_VERSION))?;
        let v = version.call(&mut store, ()).map_err(anyhow::Error::from).context("render module version")?;
        ensure!(v == abi::MODULE_VERSION, "render module contract version {v}; this host runs {}", abi::MODULE_VERSION);
        let declare = instance.get_typed_func(&mut store, abi::EXPORT_DECLARE).map_err(anyhow::Error::from).with_context(|| format!("export `{}`", abi::EXPORT_DECLARE))?;
        let frame = instance.get_typed_func(&mut store, abi::EXPORT_FRAME).map_err(anyhow::Error::from).with_context(|| format!("export `{}`", abi::EXPORT_FRAME))?;
        let input_fn = instance.get_typed_func::<(), u32>(&mut store, abi::EXPORT_INPUT).map_err(anyhow::Error::from).with_context(|| format!("export `{}`", abi::EXPORT_INPUT))?;
        let input = input_fn.call(&mut store, ()).map_err(anyhow::Error::from).context("render module input buffer")?;
        let mut m = Self { store, memory, declare, frame, input, loaded: Instant::now() };
        m.bytes(input, abi::FRAME_INPUT_BYTES as u32).context("the input buffer is outside the module's memory")?;
        Ok(m)
    }

    fn bytes(&mut self, ptr: u32, len: u32) -> Result<Vec<u8>> {
        let data = self.memory.data(&self.store);
        data.get(ptr as usize..ptr as usize + len as usize).map(<[u8]>::to_vec).with_context(|| format!("{len} bytes at {ptr:#x} are outside the module's memory ({} bytes)", data.len()))
    }

    fn words<const N: usize>(&mut self, ptr: u32) -> Result<[u32; N]> {
        let bytes = self.bytes(ptr, 4 * N as u32)?;
        Ok(std::array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())))
    }

    /// The module's graph for a target, `None` if it declines.
    pub fn declare(&mut self, width: u32, height: u32, color_format: u32, flags: u32) -> Result<Option<Declared>> {
        let ptr = self.declare.call(&mut self.store, (width, height, color_format, flags)).map_err(anyhow::Error::from).context("render module `declare`")?;
        if ptr == 0 {
            return Ok(None);
        }
        let [gp, gl, pp, pl] = self.words::<4>(ptr)?;
        Ok(Some(Declared { graph: self.bytes(gp, gl).context("graph declaration")?, programs: self.bytes(pp, pl).context("program table")? }))
    }

    pub fn frame(&mut self, input: &FrameInput) -> Result<ModuleFrame> {
        let at = self.input as usize;
        self.memory.data_mut(&mut self.store)[at..at + abi::FRAME_INPUT_BYTES].copy_from_slice(&input.to_bytes());
        let ptr = self.frame.call(&mut self.store, ()).map_err(anyhow::Error::from).context("render module `frame`")?;
        let [dp, dn, up, ul] = self.words::<4>(ptr)?;
        ensure!(dn <= 1 << 16, "{dn} words of disabled programs");
        let disabled = self.bytes(dp, dn * 4)?.chunks(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
        Ok(ModuleFrame { disabled, uniforms: self.bytes(up, ul)? })
    }
}

/// The wasmtime engine for render modules (default config: Cranelift,
/// private memories).
pub fn engine() -> Engine {
    Engine::default()
}

pub const MODULE_TARGET: &str = "wasm32-unknown-unknown";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The source directory of an in-repo module (`render/<name>`).
pub fn module_dir(name: &str) -> PathBuf {
    repo_root().join("render").join(name)
}

/// `cargo build --release` the in-repo module `render/<name>` (crate
/// `ae_render_<name>`) for [`MODULE_TARGET`] and return its bytes, or read
/// `name` as a file when it ends in `.wasm`. Build output goes to
/// `target/render-modules`.
pub fn build(name: &str) -> Result<Vec<u8>> {
    if name.ends_with(".wasm") {
        return std::fs::read(name).with_context(|| format!("read {name}"));
    }
    let manifest = module_dir(name).join("Cargo.toml");
    ensure!(manifest.exists(), "no render module `{name}` (looked for {})", manifest.display());
    let target_dir = repo_root().join("target/render-modules");
    let output = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args(["build", "--release", "--quiet", "--target", MODULE_TARGET, "--manifest-path"])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&target_dir)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .output()
        .map_err(anyhow::Error::from).context("run cargo for the render module")?;
    ensure!(output.status.success(), "render module build failed ({}):\n{}", manifest.display(), String::from_utf8_lossy(&output.stderr));
    let wasm = target_dir.join(MODULE_TARGET).join("release").join(format!("ae_render_{name}.wasm"));
    std::fs::read(&wasm).with_context(|| format!("read {}", wasm.display()))
}
