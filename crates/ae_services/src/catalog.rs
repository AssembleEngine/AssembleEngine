//! The catalogue service: `catalog_spawn` (every tick) spawns one entity
//! per `SpawnRequest` event (`ae_sdk::catalog`), in event order: the def's
//! components (`ae_catalog`), with `Body.x/y` set to the request's
//! position, a `Team` of the request's allyteam and a `Def` naming the def.
//! A request for a def the catalogue does not have is skipped (the same on
//! every peer: the catalogue is part of the content hash) and counted in
//! the service's digest part.

use ae_abi::engine::{BODY, DEF, SPAWN_REQUEST, TEAM};
use ae_catalog::Catalog;
use ae_sdk::Component;
use ae_sdk::catalog::{Def, SpawnRequest};
use ae_sdk::nav::{Body, Team};
use ae_sim::host::{HostCtx, HostDecl, HostInit, HostPlugin, HostSystem, bytes_of, expect_component};
use ae_sim::schema::Schema;
use anyhow::{Result, bail, ensure};
use std::collections::BTreeSet;
use std::sync::Arc;

#[derive(Clone)]
pub struct CatalogPlugin {
    pub catalog: Arc<Catalog>,
}

impl CatalogPlugin {
    pub fn new(catalog: Arc<Catalog>) -> Self {
        Self { catalog }
    }

    /// Every component any def has, plus `Team` and `Def`.
    fn creates(&self) -> Vec<u64> {
        let mut ids: BTreeSet<u64> = self
            .catalog
            .defs
            .values()
            .flat_map(|d| d.components.iter().map(|c| c.component))
            .collect();
        ids.extend([TEAM, DEF]);
        ids.into_iter().collect()
    }
}

impl HostPlugin for CatalogPlugin {
    fn systems(&self) -> Vec<HostDecl> {
        let mut spawn = HostDecl::new("catalog_spawn");
        spawn.reads_events = vec![SPAWN_REQUEST];
        spawn.creates = self.creates();
        vec![spawn]
    }

    fn check(&self, _name: &str, schema: &Schema) -> Result<()> {
        expect_component(schema, <SpawnRequest as ae_sdk::Event>::SCHEMA)?;
        expect_component(schema, Def::SCHEMA)?;
        expect_component(schema, Team::SCHEMA)?;
        expect_component(schema, Body::SCHEMA)?;
        for def in self.catalog.defs.values() {
            for c in &def.components {
                let decl = schema.component(c.component);
                ensure!(
                    decl.is_some_and(|d| d.size as usize == c.bytes.len()),
                    "def `{}`: component `{}` does not match the module's schema (resolve the catalogue against this module)",
                    def.name,
                    c.name
                );
            }
        }
        Ok(())
    }

    fn instantiate(&self, _init: &HostInit, _schema: &Schema, names: &[&str]) -> Result<Vec<Box<dyn HostSystem>>> {
        names
            .iter()
            .map(|&name| -> Result<Box<dyn HostSystem>> {
                match name {
                    "catalog_spawn" => Ok(Box::new(Spawner {
                        catalog: self.catalog.clone(),
                        spawned: 0,
                        unknown: 0,
                    })),
                    other => bail!("no host system `{other}` in the catalogue plugin"),
                }
            })
            .collect()
    }
}

struct Spawner {
    catalog: Arc<Catalog>,
    spawned: u64,
    unknown: u64,
}

impl HostSystem for Spawner {
    fn run(&mut self, ctx: &mut HostCtx<'_>) -> Result<()> {
        let requests: Vec<SpawnRequest> = ctx.events(SPAWN_REQUEST)?;
        for r in requests {
            let Some(def) = self.catalog.defs.get(&r.def) else {
                self.unknown += 1;
                continue;
            };
            let team = Team { ally: r.ally };
            let tag = Def { id: def.id };
            let mut parts: Vec<(u64, Vec<u8>)> =
                def.components.iter().map(|c| (c.component, c.bytes.clone())).collect();
            match parts.iter_mut().find(|(id, _)| *id == BODY) {
                Some((_, bytes)) => {
                    let mut body: Body = unsafe { bytes.as_ptr().cast::<Body>().read_unaligned() };
                    body.x = r.x;
                    body.y = r.y;
                    *bytes = bytes_of(&body).to_vec();
                }
                None => parts.push((
                    BODY,
                    bytes_of(&Body {
                        x: r.x,
                        y: r.y,
                        vx: 0.0,
                        vy: 0.0,
                        radius: 0.0,
                    })
                    .to_vec(),
                )),
            }
            parts.retain(|(id, _)| *id != TEAM && *id != DEF);
            parts.push((TEAM, bytes_of(&team).to_vec()));
            parts.push((DEF, bytes_of(&tag).to_vec()));
            let refs: Vec<(u64, &[u8])> = parts.iter().map(|(id, b)| (*id, b.as_slice())).collect();
            ctx.spawn(&refs)?;
            self.spawned += 1;
        }
        Ok(())
    }

    fn digest(&self) -> Option<u128> {
        Some((self.spawned as u128) << 64 | self.unknown as u128)
    }
}
