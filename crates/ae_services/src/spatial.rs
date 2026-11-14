//! The spatial index as a host system: `spatial` rebuilds a uniform grid
//! over every `Body` each time it runs and publishes it to module systems,
//! which query it with `ae_sdk::spatial_radius`. A module system that wants
//! this tick's positions runs `after(spatial)`; one in an earlier layer sees
//! the previous build (deterministically: the index is replaced only in the
//! spatial system's own layer).
//!
//! Items are the bodies in query order (table order, then row order), which
//! is a function of world state, so query hits are too.

use ae_abi::engine::BODY;
use ae_ecs::Access;
use ae_sdk::Component;
use ae_sdk::nav::Body;
use ae_sim::host::{HostCtx, HostDecl, HostInit, HostPlugin, HostSystem, SpatialIndex, expect_component};
use ae_sim::schema::Schema;
use ae_spatial::{Bounds, Grid};
use anyhow::{Result, bail};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct SpatialPlugin {
    pub bounds: Bounds,
    /// Grid cell side, world units. The rebuild clears and prefix-sums one
    /// counter per cell, so a fine grid on a big map costs time even when
    /// empty: 64 on a 20,480-unit map (102k cells) rebuilds 10k bodies in
    /// about 0.1 ms, 32 (410k cells) in about 0.4 ms.
    pub cell: f32,
}

impl SpatialPlugin {
    /// A square map from the origin.
    pub fn square(extent: f32, cell: f32) -> Self {
        Self { bounds: Bounds { min_x: 0.0, min_y: 0.0, max_x: extent, max_y: extent }, cell }
    }
}

impl HostPlugin for SpatialPlugin {
    fn systems(&self) -> Vec<HostDecl> {
        let mut d = HostDecl::new("spatial");
        d.query = vec![(BODY, Access::Read)];
        vec![d]
    }

    fn check(&self, _: &str, schema: &Schema) -> Result<()> {
        expect_component(schema, Body::SCHEMA).map(drop)
    }

    fn instantiate(&self, _: &HostInit, _: &Schema, names: &[&str]) -> Result<Vec<Box<dyn HostSystem>>> {
        names
            .iter()
            .map(|&n| -> Result<Box<dyn HostSystem>> {
                if n != "spatial" {
                    bail!("no host system `{n}` in the spatial plugin");
                }
                Ok(Box::new(Spatial { plugin: *self, x: Vec::new(), y: Vec::new(), r: Vec::new() }))
            })
            .collect()
    }
}

struct Spatial {
    plugin: SpatialPlugin,
    x: Vec<f32>,
    y: Vec<f32>,
    r: Vec<f32>,
}

impl HostSystem for Spatial {
    fn run(&mut self, ctx: &mut HostCtx<'_>) -> Result<()> {
        // Reuse the last index's allocations when no batch still holds it.
        let mut index = match ctx.take_spatial().map(Arc::try_unwrap) {
            Some(Ok(index)) => index,
            _ => SpatialIndex { grid: Grid::new(self.plugin.bounds, self.plugin.cell), entities: Vec::new() },
        };
        index.entities.clear();
        self.x.clear();
        self.y.clear();
        self.r.clear();
        for chunk in ctx.chunks() {
            let bodies = unsafe { chunk.column::<Body>(0) };
            index.entities.extend_from_slice(chunk.entities);
            for b in bodies.iter() {
                self.x.push(b.x);
                self.y.push(b.y);
                self.r.push(b.radius);
            }
        }
        index.grid.rebuild(&self.x, &self.y, &self.r);
        ctx.publish_spatial(Arc::new(index));
        Ok(())
    }
}
