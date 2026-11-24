//! Water: the map's look ([`WaterParams`]) and the data the rich tiers bake
//! from it — the shore travel-time field (`shore`) and the caustic texture
//! (`caustics`). M16 stage 2 builds the inputs; the M7 surface keeps its own
//! constants and ignores them (the port's stage 3 reads them).
//!
//! `ae_render` takes no dependency on `ae_map`: the client converts a map's
//! `WaterLook` into [`WaterParams`] field by field.
//!
//! A [`WaterRenderer`] holds only the inputs a graph asked for: the full-map
//! shore field (three corner-grid images) exists only when the graph declares
//! `water.shore`, and the caustic web only when it declares `water.caustics`.
//! Every allocation gets a fresh bindless identity, so a graph that replaces
//! the data (a reload that removes and restores it) never inherits the old
//! image's tracked layout.

pub mod caustics;
pub(crate) mod detail;
mod feed;
pub(crate) mod mesh;
pub mod params;
pub(crate) mod quality;
pub(crate) mod reflect;
pub mod shore;
pub(crate) mod underwater;
pub(crate) mod volume;
pub(crate) mod waves;
pub use waves::swell_height;

pub use feed::{WaterFeed, WaterImpact};
pub use params::WaterParams;
pub use shore::ShoreField;

use crate::bindless::{Bindless, GroupId};
use crate::device::GpuDevice;
use crate::executor::{ImageBinding, PassContext};
use anyhow::{Context, Result};
use std::sync::atomic::{AtomicU64, Ordering};

/// The water stack's data for one scene, for the inputs a graph declared.
pub struct WaterRenderer {
    /// `None` when the graph does not sample `water.shore`: a full-map field
    /// it would never read.
    field: Option<ShoreField>,
    /// `None` when the graph does not sample `water.caustics`.
    caustics: Option<caustics::Caustics>,
    /// The shore field's global sampled-image index: a graph that samples it
    /// (the rich surface does) needs the bindless handle, not the resource.
    shore_sampled: Option<u32>,
    /// A fresh number per allocation: the executor keys external access
    /// states by identity, so a replacement image must not reuse one.
    generation: u64,
    group: Option<GroupId>,
}

/// The next renderer's generation (unique across allocations in a process).
static GENERATION: AtomicU64 = AtomicU64::new(1);

impl WaterRenderer {
    /// Make the inputs `shore` and/or `caustics` ask for (at least one).
    /// `norm_level` is the water level in the height texture's normalised
    /// units; `cell_size` metres per texel.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &GpuDevice,
        bindless: &mut Bindless,
        width: u32,
        height: u32,
        norm_level: f32,
        cell_size: f32,
        shore: bool,
        caustics: bool,
    ) -> Result<Self> {
        anyhow::ensure!(shore || caustics, "a water renderer needs an input");
        let generation = GENERATION.fetch_add(1, Ordering::Relaxed);
        let identity = |slot: u64| (0xAE7 << 40) | (4 << 32) | (generation << 8) | slot;
        // A failure destroys what was already made: the group rollback
        // releases registered handles only, not images or buffers.
        let built = bindless.group(|bindless| -> Result<Self> {
            let caustics = if caustics {
                Some(caustics::Caustics::new(gpu, bindless, identity(2)).context("caustic web")?)
            } else {
                None
            };
            let mut field = if shore {
                match ShoreField::new(gpu, bindless, width, height, norm_level, cell_size, identity(1)) {
                    Ok(field) => Some(field),
                    Err(error) => {
                        if let Some(mut caustics) = caustics {
                            caustics.destroy(gpu);
                        }
                        return Err(error.context("shore field"));
                    }
                }
            } else {
                None
            };
            let shore_sampled = if shore {
                let view = field.as_ref().expect("the shore field was made").view();
                match bindless.sampled_2d(view, ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) {
                    Ok(handle) => Some(handle.0.index),
                    Err(error) => {
                        if let Some(mut field) = field.take() {
                            field.retire(bindless);
                            field.destroy(gpu);
                        }
                        if let Some(mut caustics) = caustics {
                            caustics.destroy(gpu);
                        }
                        return Err(error.context("shore field's sampled handle"));
                    }
                }
            } else {
                None
            };
            Ok(Self {
                field,
                caustics,
                shore_sampled,
                generation,
                group: None,
            })
        });
        match built {
            Ok((mut this, group)) => {
                this.group = Some(group);
                Ok(this)
            }
            Err(error) => Err(error),
        }
    }

    pub fn has_shore(&self) -> bool {
        self.field.is_some()
    }

    pub fn has_caustics(&self) -> bool {
        self.caustics.is_some()
    }

    pub fn shore(&self) -> Option<&ShoreField> {
        self.field.as_ref()
    }

    pub fn caustics(&self) -> Option<&caustics::Caustics> {
        self.caustics.as_ref()
    }

    /// The allocation's generation (tests: identities are fresh per rebuild).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The shore field as a graph declares and binds it (`None` when the
    /// graph does not declare it).
    pub fn shore_binding(&self) -> Option<ImageBinding> {
        let mut binding = self.field.as_ref()?.binding();
        binding.sampled = self.shore_sampled;
        Some(binding)
    }

    /// The caustic web as a graph declares and binds it.
    pub fn caustics_binding(&self) -> Option<ImageBinding> {
        self.caustics.as_ref().map(|c| c.binding())
    }

    /// The ground changed since the last bake (false without a shore field).
    pub fn shore_dirty(&self) -> bool {
        self.field.as_ref().is_some_and(|s| s.dirty())
    }

    /// Bakes recorded (one on the first frame, one after each ground edit).
    pub fn bakes(&self) -> u64 {
        self.field.as_ref().map_or(0, |s| s.bakes())
    }

    /// The app's edit hook: the ground changed (the feed's cells, when it
    /// gave a rectangle).
    pub fn mark_dirty(&self, rect: Option<(u32, u32, u32, u32)>) {
        if let Some(field) = &self.field {
            field.mark_dirty(rect);
        }
    }

    /// `water.shore`: rebuild the field from the ground height texture.
    pub fn record_shore(&self, ctx: &PassContext<'_>, ground: u32, sampler: u32) -> Result<()> {
        self.field
            .as_ref()
            .context("the graph samples `water.shore` without allocating it")?
            .record(ctx, ground, sampler)
    }

    /// Give the bindless handles back (before [`Self::destroy`]).
    pub fn retire(&mut self, bindless: &mut Bindless) {
        if let Some(field) = self.field.as_mut() {
            field.retire(bindless);
        }
        if let Some(group) = self.group.take() {
            bindless.retire_group(group);
        }
    }

    pub fn destroy(&mut self, gpu: &GpuDevice) {
        if let Some(field) = self.field.as_mut() {
            field.destroy(gpu);
        }
        if let Some(caustics) = self.caustics.as_mut() {
            caustics.destroy(gpu);
        }
    }
}
