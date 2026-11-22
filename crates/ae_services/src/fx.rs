//! Effects at joints (M14, feature `fx`): the host system `fx` drains the
//! [`EFFECT`](ae_abi::engine::EFFECT) stream, in key order, and publishes
//! the events to an optional [`FxFeed`]. The client turns them into
//! particles where its observer sees them; headless runs have no feed and
//! drop them. The system is a one-way sink, like the ground feed: nothing
//! reaches the sim through it.
//!
//! Its reader cursor is part of the world digest, and it advances the same
//! way with or without a feed, so a game that schedules `host(fx)` digests
//! alike in the client and in `ae-headless` (both register the plugin).
//! Schedule it last.

use ae_sdk::fx::Effect;
use ae_sim::host::{HostCtx, HostDecl, HostInit, HostPlugin, HostSystem, expect_component};
use ae_sim::schema::Schema;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// The effects a tick emitted, from the sim thread to whoever draws them.
#[derive(Debug, Default)]
pub struct FxFeed {
    pending: Mutex<Vec<Effect>>,
}

impl FxFeed {
    /// Append a tick's effects (in the order the sim emitted them).
    pub fn publish(&self, effects: impl IntoIterator<Item = Effect>) {
        self.pending.lock().unwrap_or_else(|p| p.into_inner()).extend(effects);
    }

    /// Every effect published since the last call, oldest first.
    pub fn take(&self) -> Vec<Effect> {
        std::mem::take(&mut *self.pending.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

/// Provides the host system `fx`. `feed` is `None` where nothing draws
/// them (ae-headless): the events are then dropped, after the same
/// bookkeeping.
#[derive(Debug, Default)]
pub struct FxPlugin {
    feed: Option<Arc<FxFeed>>,
}

impl FxPlugin {
    pub fn new(feed: Option<Arc<FxFeed>>) -> Self {
        Self { feed }
    }
}

impl HostPlugin for FxPlugin {
    fn systems(&self) -> Vec<HostDecl> {
        let mut fx = HostDecl::new("fx");
        fx.reads_events = vec![ae_abi::engine::EFFECT];
        vec![fx]
    }

    fn check(&self, name: &str, schema: &Schema) -> Result<()> {
        anyhow::ensure!(name == "fx", "no host system `{name}`");
        expect_component(schema, <Effect as ae_sdk::Event>::SCHEMA)?;
        Ok(())
    }

    fn instantiate(&self, _init: &HostInit, _schema: &Schema, names: &[&str]) -> Result<Vec<Box<dyn HostSystem>>> {
        Ok(names
            .iter()
            .map(|&name| {
                assert_eq!(name, "fx", "no host system `{name}`");
                Box::new(FxSystem {
                    feed: self.feed.clone(),
                }) as Box<dyn HostSystem>
            })
            .collect())
    }
}

struct FxSystem {
    feed: Option<Arc<FxFeed>>,
}

impl HostSystem for FxSystem {
    fn run(&mut self, ctx: &mut HostCtx<'_>) -> Result<()> {
        // Seen either way: the cursor must not depend on a reader.
        let effects = ctx.events::<Effect>(ae_abi::engine::EFFECT)?;
        if let Some(feed) = &self.feed {
            feed.publish(effects);
        }
        Ok(())
    }
}
