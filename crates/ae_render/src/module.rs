//! Render modules on the host: loading one with wasmtime, calling its
//! `declare` and `frame` exports (contract in [`ae_render_abi::module`]),
//! and building the in-repo modules (`render/<name>`). A graph file
//! ([`crate::graph_file`]) loads through the same door.
//!
//! A render module is trusted (like the sim module) but gets its own store
//! and memory and no imports: everything it hands back is copied out of its
//! memory with bounds checks, and the graph and program table are
//! validated before use (`ae_framegraph::descriptor::decode`, the pass
//! library's checks). A trap is an error the caller can survive: the scene
//! keeps its last good graph.

use crate::colorspace::{Gamut, WorkingSpace};
use crate::encode::{DisplayLevels, OutputEncoding};
use ae_render_abi::module::{self as abi, DeclareInput, FrameInput};
use anyhow::{Context as _, Result, bail, ensure};
use ash::vk;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use wasmtime::{Engine, Instance, Memory, Module, Store, TypedFunc};

/// What a module is asked to declare for: the target, the scene's flags and
/// the output's display spec (contract version 2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeclareTarget {
    pub width: u32,
    pub height: u32,
    pub color_format: vk::Format,
    /// `DECLARE_*` flags: units, readback, water, shadows, particles.
    pub flags: u32,
    /// How the output encodes the display's light: HDR keeps the display
    /// intermediate and ends in `post.encode`, SDR has neither.
    pub encoding: OutputEncoding,
    /// The display's paper white and peak, in signal nits.
    pub levels: DisplayLevels,
    /// The space the scene is lit in and the display intermediate stores.
    pub working: WorkingSpace,
    /// The gamut `post.encode` maps the display to.
    pub gamut: Gamut,
}

impl DeclareTarget {
    /// The [`DeclareInput`] the host writes into the module's memory.
    pub fn to_abi(self) -> DeclareInput {
        DeclareInput {
            width: self.width,
            height: self.height,
            color_format: self.color_format.as_raw() as u32,
            flags: self.flags,
            encoding: self.encoding.to_wire(),
            working_space: self.working.to_wire(),
            output_gamut: self.gamut.to_wire(),
            paper_white: self.levels.paper_white,
            peak: self.levels.peak,
            reserved: [0; 6],
        }
    }
}

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
        self.disabled
            .get(program as usize / 32)
            .is_some_and(|w| w & (1 << (program % 32)) != 0)
    }
}

/// A loaded render module: compiled WASM, or a graph file
/// ([`crate::graph_file`]) that declares a fixed graph.
pub enum RenderModule {
    Wasm(WasmModule),
    File(crate::graph_file::GraphFile),
}

impl RenderModule {
    /// A self-contained graph file (JSON, as [`crate::graph_file::inline`]
    /// writes it: it starts with `{`), or else WASM (binary or text).
    pub fn load(engine: &Engine, bytes: &[u8]) -> Result<Self> {
        if bytes.trim_ascii_start().starts_with(b"{") {
            crate::graph_file::GraphFile::parse(bytes).map(Self::File)
        } else {
            WasmModule::load(engine, bytes).map(Self::Wasm)
        }
    }

    /// The module's graph for a target, `None` if it declines.
    pub fn declare(&mut self, target: &DeclareTarget) -> Result<Option<Declared>> {
        match self {
            Self::Wasm(m) => {
                m.declare_input(target)?;
                m.declare()
            }
            Self::File(f) => f.declare(target),
        }
    }

    pub fn frame(&mut self, input: &FrameInput) -> Result<ModuleFrame> {
        match self {
            Self::Wasm(m) => m.frame(input),
            Self::File(f) => Ok(f.frame(input)),
        }
    }

    /// The contract version the module was built for (a graph file is the
    /// current one).
    pub fn version(&self) -> u32 {
        match self {
            Self::Wasm(m) => m.version,
            Self::File(_) => abi::MODULE_VERSION,
        }
    }

    /// When it was loaded.
    pub fn loaded(&self) -> Instant {
        match self {
            Self::Wasm(m) => m.loaded,
            Self::File(f) => f.loaded,
        }
    }
}

pub struct WasmModule {
    store: Store<()>,
    memory: Memory,
    /// The contract version the module exports.
    version: u32,
    declare: TypedFunc<(), u32>,
    frame: TypedFunc<(), u32>,
    /// Where [`DeclareInput`] goes before each `declare`.
    declare_input: u32,
    input: u32,
    pub loaded: Instant,
}

impl WasmModule {
    pub fn load(engine: &Engine, bytes: &[u8]) -> Result<Self> {
        let module = Module::new(engine, bytes)
            .map_err(anyhow::Error::from)
            .context("compiling the render module")?;
        if let Some(import) = module.imports().next() {
            bail!(
                "render modules import nothing; this one imports `{}::{}`",
                import.module(),
                import.name()
            );
        }
        let mut store = Store::new(engine, ());
        let instance = Instance::new(&mut store, &module, &[])
            .map_err(anyhow::Error::from)
            .context("instantiating the render module")?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .context("the render module exports no `memory`")?;
        let version = instance
            .get_typed_func::<(), u32>(&mut store, abi::EXPORT_VERSION)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("export `{}`", abi::EXPORT_VERSION))?;
        let v = version
            .call(&mut store, ())
            .map_err(anyhow::Error::from)
            .context("render module version")?;
        // The version comes first: a module built for another ABI is refused
        // by number, never called with the wrong signature. Version 2 is
        // still accepted while its fullscreen programs keep the per-pass
        // form; the next contract version drops it.
        ensure!(
            (2..=abi::MODULE_VERSION).contains(&v),
            "render module ABI version {v}; this host runs {} (rebuild the module)",
            abi::MODULE_VERSION
        );
        let declare_input_fn = instance
            .get_typed_func::<(), u32>(&mut store, abi::EXPORT_DECLARE_INPUT)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("export `{}`", abi::EXPORT_DECLARE_INPUT))?;
        let declare_input = declare_input_fn
            .call(&mut store, ())
            .map_err(anyhow::Error::from)
            .context("render module declaration input buffer")?;
        let declare = instance
            .get_typed_func(&mut store, abi::EXPORT_DECLARE)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("export `{}`", abi::EXPORT_DECLARE))?;
        let frame = instance
            .get_typed_func(&mut store, abi::EXPORT_FRAME)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("export `{}`", abi::EXPORT_FRAME))?;
        let input_fn = instance
            .get_typed_func::<(), u32>(&mut store, abi::EXPORT_INPUT)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("export `{}`", abi::EXPORT_INPUT))?;
        let input = input_fn
            .call(&mut store, ())
            .map_err(anyhow::Error::from)
            .context("render module input buffer")?;
        let mut m = Self {
            store,
            memory,
            version: v,
            declare,
            frame,
            declare_input,
            input,
            loaded: Instant::now(),
        };
        m.bytes(input, abi::FRAME_INPUT_BYTES as u32)
            .context("the frame input buffer is outside the module's memory")?;
        m.bytes(declare_input, abi::DECLARE_INPUT_BYTES as u32)
            .context("the declaration input buffer is outside the module's memory")?;
        Ok(m)
    }

    fn bytes(&mut self, ptr: u32, len: u32) -> Result<Vec<u8>> {
        let data = self.memory.data(&self.store);
        data.get(ptr as usize..ptr as usize + len as usize)
            .map(<[u8]>::to_vec)
            .with_context(|| {
                format!(
                    "{len} bytes at {ptr:#x} are outside the module's memory ({} bytes)",
                    data.len()
                )
            })
    }

    fn words<const N: usize>(&mut self, ptr: u32) -> Result<[u32; N]> {
        let bytes = self.bytes(ptr, 4 * N as u32)?;
        Ok(std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
        }))
    }

    /// Write the target and display spec the next `declare` reads.
    fn declare_input(&mut self, target: &DeclareTarget) -> Result<()> {
        let at = self.declare_input as usize;
        let bytes = target.to_abi().to_bytes();
        let memory = self.memory.data_mut(&mut self.store);
        memory
            .get_mut(at..at + bytes.len())
            .context("the declaration input buffer is outside the module's memory")?
            .copy_from_slice(&bytes);
        Ok(())
    }

    /// The module's graph for the target last written, `None` if it declines.
    pub fn declare(&mut self) -> Result<Option<Declared>> {
        let ptr = self
            .declare
            .call(&mut self.store, ())
            .map_err(anyhow::Error::from)
            .context("render module `declare`")?;
        if ptr == 0 {
            return Ok(None);
        }
        let [gp, gl, pp, pl] = self.words::<4>(ptr)?;
        Ok(Some(Declared {
            graph: self.bytes(gp, gl).context("graph declaration")?,
            programs: self.bytes(pp, pl).context("program table")?,
        }))
    }

    pub fn frame(&mut self, input: &FrameInput) -> Result<ModuleFrame> {
        let at = self.input as usize;
        self.memory.data_mut(&mut self.store)[at..at + abi::FRAME_INPUT_BYTES].copy_from_slice(&input.to_bytes());
        let ptr = self
            .frame
            .call(&mut self.store, ())
            .map_err(anyhow::Error::from)
            .context("render module `frame`")?;
        let [dp, dn, up, ul] = self.words::<4>(ptr)?;
        ensure!(dn <= 1 << 16, "{dn} words of disabled programs");
        let disabled = self
            .bytes(dp, dn * 4)?
            .chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        Ok(ModuleFrame {
            disabled,
            uniforms: self.bytes(up, ul)?,
        })
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
/// `name` as a file when it ends in `.wasm`, or as a graph file when it
/// ends in `.json` (made self-contained, [`crate::graph_file::inline`]).
/// Build output goes to `target/render-modules`.
pub fn build(name: &str) -> Result<Vec<u8>> {
    if name.ends_with(".wasm") {
        return std::fs::read(name).with_context(|| format!("read {name}"));
    }
    if name.ends_with(".json") {
        return crate::graph_file::inline(Path::new(name));
    }
    let manifest = module_dir(name).join("Cargo.toml");
    ensure!(
        manifest.exists(),
        "no render module `{name}` (looked for {})",
        manifest.display()
    );
    let target_dir = repo_root().join("target/render-modules");
    let output = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args([
            "build",
            "--release",
            "--quiet",
            "--target",
            MODULE_TARGET,
            "--manifest-path",
        ])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&target_dir)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .output()
        .map_err(anyhow::Error::from)
        .context("run cargo for the render module")?;
    ensure!(
        output.status.success(),
        "render module build failed ({}):\n{}",
        manifest.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let wasm = target_dir
        .join(MODULE_TARGET)
        .join("release")
        .join(format!("ae_render_{name}.wasm"));
    std::fs::read(&wasm).with_context(|| format!("read {}", wasm.display()))
}
