//! Shared helpers for the M0 spikes: a guest module instantiated N times
//! over one shared linear memory, each instance with its own shadow stack
//! and TLS block.

use anyhow::{Context, Result, ensure};
use wasmtime::{Config, Engine, ExternType, Instance, Linker, Module, SharedMemory, Store, TypedFunc, Val};

/// Path of the guest module built by `build.rs`.
pub const GUEST_WASM: &str = env!("GUEST_WASM");
pub const STACK_BYTES: u32 = 256 << 10;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Metering {
    None,
    Fuel,
    Epoch,
    /// No metering and no NaN canonicalization: shows what canonical NaNs cost.
    RawNans,
}

pub fn engine(metering: Metering) -> Result<Engine> {
    let mut config = Config::new();
    config.wasm_threads(true);
    config.shared_memory(true);
    config.memory_may_move(false);
    // Same determinism settings the sim will use.
    config.wasm_relaxed_simd(false);
    config.cranelift_nan_canonicalization(metering != Metering::RawNans);
    config.consume_fuel(metering == Metering::Fuel);
    config.epoch_interruption(metering == Metering::Epoch);
    Ok(Engine::new(&config)?)
}

/// One guest instance with its own store. Moved to the worker thread that uses it.
pub struct Guest {
    pub store: Store<()>,
    pub instance: Instance,
    pub noop: TypedFunc<(), ()>,
    pub integrate: TypedFunc<(u32, u32, u32, f32), ()>,
    pub tls_bump: TypedFunc<(), u32>,
    pub stack_probe: TypedFunc<(u32, u32), u32>,
    pub stack_address: TypedFunc<(), u32>,
    pub malloc: TypedFunc<u32, u32>,
}

impl Guest {
    pub fn instantiate(engine: &Engine, module: &Module, memory: &SharedMemory, metering: Metering) -> Result<Self> {
        let mut store = Store::new(engine, ());
        match metering {
            Metering::Fuel => store.set_fuel(u64::MAX)?,
            Metering::Epoch => store.set_epoch_deadline(u64::MAX / 2),
            Metering::None | Metering::RawNans => {}
        }
        let mut linker = Linker::new(engine);
        linker.define(&store, "env", "memory", memory.clone())?;
        // The spike guest never reaches WASI; anything it would import traps.
        linker.define_unknown_imports_as_traps(module)?;
        // The start function initialises memory once (guarded by a flag in
        // the shared memory), so later instances leave the data alone.
        let instance = linker.instantiate(&mut store, module)?;
        Ok(Self {
            noop: instance.get_typed_func(&mut store, "noop")?,
            integrate: instance.get_typed_func(&mut store, "integrate")?,
            tls_bump: instance.get_typed_func(&mut store, "tls_bump")?,
            stack_probe: instance.get_typed_func(&mut store, "stack_probe")?,
            stack_address: instance.get_typed_func(&mut store, "stack_address")?,
            malloc: instance.get_typed_func(&mut store, "malloc")?,
            store,
            instance,
        })
    }

    pub fn global_i32(&mut self, name: &str) -> Result<i32> {
        let global = self.instance.get_global(&mut self.store, name).with_context(|| format!("guest does not export {name}"))?;
        global.get(&mut self.store).i32().with_context(|| format!("{name} is not an i32"))
    }

    pub fn alloc(&mut self, bytes: u32) -> Result<u32> {
        let offset = self.malloc.call(&mut self.store, bytes)?;
        ensure!(offset != 0, "guest malloc({bytes}) failed");
        Ok(offset)
    }

    /// Give this instance a private shadow stack and TLS block, allocated by
    /// `allocator` (the first instance, which owns the guest heap).
    pub fn adopt_private_stack_and_tls(&mut self, allocator: &mut Guest) -> Result<()> {
        let stack = allocator.alloc(STACK_BYTES)?;
        let top = (stack + STACK_BYTES) & !15;
        let stack_pointer = self.instance.get_global(&mut self.store, "__stack_pointer").context("no __stack_pointer export")?;
        stack_pointer.set(&mut self.store, Val::I32(top as i32))?;

        let size = self.global_i32("__tls_size")? as u32;
        let align = (self.global_i32("__tls_align")? as u32).max(1);
        if size > 0 {
            let raw = allocator.alloc(size + align)?;
            let block = raw.next_multiple_of(align);
            let init_tls: TypedFunc<u32, ()> = self.instance.get_typed_func(&mut self.store, "__wasm_init_tls")?;
            init_tls.call(&mut self.store, block)?;
        }
        Ok(())
    }
}

/// `count` instances over one new shared memory. Instance 0 runs the
/// constructors and owns the heap; the rest get private stacks and TLS when
/// `isolate` is set.
pub fn spawn_guests(engine: &Engine, module: &Module, count: usize, metering: Metering, isolate: bool) -> Result<(SharedMemory, Vec<Guest>)> {
    spawn_guests_with(engine, module, count, metering, isolate, |_, _| Ok(()))
}

/// [`spawn_guests`], calling `after_first` once the first instance has run
/// its constructors and before any other instance allocates its stack.
pub fn spawn_guests_with(
    engine: &Engine,
    module: &Module,
    count: usize,
    metering: Metering,
    isolate: bool,
    after_first: impl FnOnce(&SharedMemory, &mut Guest) -> Result<()>,
) -> Result<(SharedMemory, Vec<Guest>)> {
    let memory_type = module
        .imports()
        .find_map(|import| match import.ty() {
            ExternType::Memory(ty) if import.name() == "memory" => Some(ty),
            _ => None,
        })
        .context("guest does not import a memory")?;
    ensure!(memory_type.is_shared(), "guest memory import is not shared");
    let memory = SharedMemory::new(engine, memory_type)?;

    let mut first = Guest::instantiate(engine, module, &memory, metering)?;
    let ctors: TypedFunc<(), ()> = first.instance.get_typed_func(&mut first.store, "__wasm_call_ctors")?;
    ctors.call(&mut first.store, ())?;
    after_first(&memory, &mut first)?;

    let mut guests = Vec::with_capacity(count);
    for _ in 1..count {
        let mut guest = Guest::instantiate(engine, module, &memory, metering)?;
        if isolate {
            guest.adopt_private_stack_and_tls(&mut first)?;
        }
        guests.push(guest);
    }
    guests.insert(0, first);
    Ok((memory, guests))
}

pub fn memory_base(memory: &SharedMemory) -> *mut u8 {
    memory.data().as_ptr() as *mut u8
}

