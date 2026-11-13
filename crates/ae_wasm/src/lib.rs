//! Runtime for sim modules.
//!
//! A sim module is a `wasm32-wasip1-threads` module that imports a shared
//! `env.memory`. The engine instantiates it once per worker thread, all over
//! the same [`SharedMemory`], so systems can run on several cores at once
//! and work in place on component columns.
//!
//! Rules every instance set follows (established by the M0 spikes):
//! * Instance 0 owns the guest heap: it runs `__wasm_call_ctors`, and the
//!   other instances' stacks and TLS blocks are allocated with its `malloc`.
//! * Every other instance gets a private shadow stack (`__stack_pointer`)
//!   and TLS block (`__wasm_init_tls`) before its first call. Without that it
//!   shares instance 0's stack and has no TLS block at all.
//! * Host-owned ranges (the column arena) are claimed by growing the memory
//!   from the host; the guest allocator then grows around them.

use anyhow::{Context, Result, bail, ensure};
use wasmtime::{Config, Engine, ExternType, Instance as WasmInstance, Linker, Module, SharedMemory, Store, TypedFunc, Val, WasmParams, WasmResults};

pub const PAGE_BYTES: u64 = 65536;
pub const DEFAULT_STACK_BYTES: u32 = 256 << 10;

/// What a sim module run is measured with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metering {
    /// Release: no per-instruction cost. Overruns are caught by a host watchdog.
    None,
    /// Dev and bench: wasmtime fuel as a deterministic per-call cost counter.
    Fuel,
}

/// wasmtime settings every sim module runs with. NaN canonicalization and
/// disabled relaxed SIMD keep float results identical on every platform.
pub fn sim_config(metering: Metering) -> Config {
    let mut config = Config::new();
    config.wasm_threads(true);
    config.shared_memory(true);
    config.memory_may_move(false);
    config.wasm_relaxed_simd(false);
    config.cranelift_nan_canonicalization(true);
    config.consume_fuel(metering == Metering::Fuel);
    config
}

pub struct SimModule {
    pub engine: Engine,
    pub module: Module,
    pub metering: Metering,
}

impl SimModule {
    pub fn new(bytes: &[u8], metering: Metering) -> Result<Self> {
        let engine = Engine::new(&sim_config(metering))?;
        let module = Module::new(&engine, bytes).map_err(anyhow::Error::from).context("compile sim module")?;
        Ok(Self { engine, module, metering })
    }

    /// A new shared memory of the type the module imports as `env.memory`.
    pub fn new_memory(&self) -> Result<SharedMemory> {
        let ty = self
            .module
            .imports()
            .find_map(|import| match import.ty() {
                ExternType::Memory(ty) if import.module() == "env" && import.name() == "memory" => Some(ty),
                _ => None,
            })
            .context("sim module does not import env.memory")?;
        ensure!(ty.is_shared(), "sim module's env.memory is not shared (build for wasm32-wasip1-threads)");
        Ok(SharedMemory::new(&self.engine, ty)?)
    }
}

/// Host view of a shared memory: guest offsets <-> host pointers.
#[derive(Clone, Copy, Debug)]
pub struct GuestMemory {
    base: *mut u8,
}

// The base of a shared memory never moves (`memory_may_move(false)`), and
// shared memory is meant to be accessed from many threads.
unsafe impl Send for GuestMemory {}
unsafe impl Sync for GuestMemory {}

impl GuestMemory {
    pub fn of(memory: &SharedMemory) -> Self {
        Self { base: memory.data().as_ptr() as *mut u8 }
    }

    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// Host pointer for a guest offset. The caller must have checked bounds.
    pub fn ptr(&self, offset: u32) -> *mut u8 {
        unsafe { self.base.add(offset as usize) }
    }

    /// Guest offset of a host pointer into this memory.
    pub fn offset(&self, ptr: *const u8) -> u32 {
        let delta = ptr as usize - self.base as usize;
        debug_assert!(delta <= u32::MAX as usize);
        delta as u32
    }
}

/// Claim `pages` fresh pages for the host. Returns the guest offset of the
/// first byte. No guest allocator will ever hand out memory in this range.
pub fn claim_pages(memory: &SharedMemory, pages: u64) -> Result<u32> {
    let first = memory.grow(pages).map_err(anyhow::Error::from).with_context(|| format!("grow shared memory by {pages} pages"))?;
    let offset = first * PAGE_BYTES;
    ensure!(offset + pages * PAGE_BYTES <= u32::MAX as u64 + 1, "claim exceeds the 32-bit address space");
    Ok(offset as u32)
}

pub struct Instance<T: 'static> {
    pub store: Store<T>,
    pub instance: WasmInstance,
}

impl<T: 'static> Instance<T> {
    pub fn func<P: WasmParams, R: WasmResults>(&mut self, name: &str) -> Result<TypedFunc<P, R>> {
        self.instance.get_typed_func(&mut self.store, name).map_err(anyhow::Error::from).with_context(|| format!("sim module export `{name}`"))
    }

    pub fn global_i32(&mut self, name: &str) -> Result<i32> {
        let global = self.instance.get_global(&mut self.store, name).with_context(|| format!("sim module does not export `{name}`"))?;
        global.get(&mut self.store).i32().with_context(|| format!("`{name}` is not an i32"))
    }

    fn set_global_i32(&mut self, name: &str, value: i32) -> Result<()> {
        let global = self.instance.get_global(&mut self.store, name).with_context(|| format!("sim module does not export `{name}`"))?;
        global.set(&mut self.store, Val::I32(value))?;
        Ok(())
    }
}

/// N instances of one sim module over one shared memory.
pub struct InstanceSet<T: 'static> {
    pub memory: SharedMemory,
    pub instances: Vec<Instance<T>>,
    malloc: TypedFunc<u32, u32>,
}

/// How to build an [`InstanceSet`].
pub struct InstanceSetBuilder<'a, T: 'static> {
    pub module: &'a SimModule,
    pub count: usize,
    pub stack_bytes: u32,
    /// Defines the module's host imports (except `env.memory`).
    pub link: &'a dyn Fn(&mut Linker<T>) -> Result<()>,
    /// Per-instance store data.
    pub data: &'a dyn Fn(usize) -> T,
}

impl<T: 'static> InstanceSetBuilder<'_, T> {
    /// Instantiate. `after_heap_owner` runs once instance 0 has run its
    /// constructors and before any other allocation: the place to claim
    /// host ranges.
    pub fn build(self, memory: SharedMemory, after_heap_owner: impl FnOnce(&SharedMemory, &mut Instance<T>) -> Result<()>) -> Result<InstanceSet<T>> {
        ensure!(self.count >= 1, "an instance set needs at least one instance");
        let mut owner = self.instantiate(&memory, 0)?;
        owner.func::<(), ()>("__wasm_call_ctors")?.call(&mut owner.store, ())?;
        let malloc: TypedFunc<u32, u32> = owner.func("malloc")?;
        after_heap_owner(&memory, &mut owner)?;

        let mut instances = vec![owner];
        for index in 1..self.count {
            let mut instance = self.instantiate(&memory, index)?;
            let stack = guest_malloc(&malloc, &mut instances[0], self.stack_bytes)?;
            instance.set_global_i32("__stack_pointer", ((stack + self.stack_bytes) & !15) as i32)?;
            let size = instance.global_i32("__tls_size")? as u32;
            let align = (instance.global_i32("__tls_align")? as u32).max(1);
            if size > 0 {
                let raw = guest_malloc(&malloc, &mut instances[0], size + align)?;
                instance.func::<u32, ()>("__wasm_init_tls")?.call(&mut instance.store, raw.next_multiple_of(align))?;
            }
            instances.push(instance);
        }
        Ok(InstanceSet { memory, instances, malloc })
    }

    fn instantiate(&self, memory: &SharedMemory, index: usize) -> Result<Instance<T>> {
        let engine = &self.module.engine;
        let mut store = Store::new(engine, (self.data)(index));
        if self.module.metering == Metering::Fuel {
            store.set_fuel(u64::MAX)?;
        }
        let mut linker = Linker::new(engine);
        linker.define(&store, "env", "memory", memory.clone())?;
        (self.link)(&mut linker)?;
        // Sim modules get no WASI: clocks, randomness and files would break
        // determinism. Anything still imported traps if called.
        linker.define_unknown_imports_as_traps(&self.module.module)?;
        let instance = linker.instantiate(&mut store, &self.module.module).map_err(anyhow::Error::from).context("instantiate sim module")?;
        Ok(Instance { store, instance })
    }
}

fn guest_malloc<T: 'static>(malloc: &TypedFunc<u32, u32>, owner: &mut Instance<T>, bytes: u32) -> Result<u32> {
    let offset = malloc.call(&mut owner.store, bytes)?;
    if offset == 0 {
        bail!("guest malloc({bytes}) failed");
    }
    Ok(offset)
}

impl<T: 'static> InstanceSet<T> {
    pub fn guest_memory(&self) -> GuestMemory {
        GuestMemory::of(&self.memory)
    }

    /// Allocate from the guest heap (through instance 0).
    pub fn malloc(&mut self, bytes: u32) -> Result<u32> {
        guest_malloc(&self.malloc, &mut self.instances[0], bytes)
    }
}
