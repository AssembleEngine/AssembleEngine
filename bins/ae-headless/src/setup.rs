//! What a run is built from: the sim module, its content and its map.
//!
//! Content is a stack of mounts ([`ae_vfs`]): the packages named with
//! `--package SPEC|PATH` (repeatable; each with its dependencies, found in
//! the `--packages DIR` libraries), then `--content DIR` directories on top.
//! The stack decides the rest:
//! * a `module.wasm` in it is the sim module (else `--game NAME` is built);
//! * `defs/*.json` are the catalogue, loaded against the module's schema;
//! * a `map.json` (a map package, `ae_map`) is the map (else `--nav`).
//!
//! The header records the content hash (every visible file, so module,
//! defs and map), the packages by `name@version`, and the map. A replay
//! made from packages finds them again by those names in `--packages`, so
//! it needs no `--package` of its own.

use super::Opts;
use ae_replay::Header;
use ae_services::nav::{Map, NavPlugin};
use ae_services::spatial::SpatialPlugin;
use ae_sim::host::HostPlugin;
use ae_sim::runtime::{Runtime, RuntimeConfig};
use ae_vfs::{Library, Package, Vfs};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// A package's sim module.
pub const MODULE: &str = "module.wasm";

pub fn map_name(m: Map) -> &'static str {
    match m {
        Map::WaterCrossing => "water",
        Map::LandCrossing => "land",
        Map::Loaded => "package",
    }
}

pub fn map_of(name: &str) -> Result<Map> {
    Ok(match name {
        "water" => Map::WaterCrossing,
        "land" => Map::LandCrossing,
        "package" => Map::Loaded,
        other => bail!("unknown map {other}"),
    })
}

/// Mounted content: its hash, packages and catalogue.
pub struct Content {
    pub hash: u128,
    /// `name@version` of each package, in mount order.
    pub packages: Vec<String>,
    pub catalog: Option<Arc<ae_catalog::Catalog>>,
}

pub struct Setup {
    pub wasm: Vec<u8>,
    pub nav: Option<NavPlugin>,
    pub content: Option<Content>,
}

/// The packages `specs` name (paths or `name[@version]` in the
/// libraries), each after its dependencies, each once.
fn resolve(o: &Opts, specs: &[String]) -> Result<Vec<Package>> {
    let mut lib = if o.package_dirs.is_empty() { Library::default() } else { Library::scan(&o.package_dirs)? };
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

impl Setup {
    /// From the options, or for a replay from its header (`recorded`): its
    /// packages (unless `--package` names others), map and teams.
    pub fn new(o: &Opts, recorded: Option<&Header>) -> Result<Self> {
        let mut specs = o.packages.clone();
        if specs.is_empty()
            && let Some(p) = recorded.and_then(|h| h.config.get("packages"))
        {
            ensure!(!o.package_dirs.is_empty(), "the replay was made from packages {p}: name a --packages DIR to find them");
            specs = p.split(',').map(String::from).collect();
        }
        let packages = resolve(o, &specs)?;
        let mut vfs = Vfs::new();
        vfs.mount_packages(&packages)?;
        for dir in &o.content {
            vfs.mount_dir(dir)?;
        }
        let mounted = !packages.is_empty() || !o.content.is_empty();
        let wasm = if vfs.exists(MODULE) {
            vfs.read(MODULE)?
        } else {
            ensure!(!o.game.is_empty(), "no --game, and no package has a {MODULE}");
            ae_sim::build::game(&o.game)?
        };
        let catalog = if vfs.list("defs", "json")?.is_empty() {
            None
        } else {
            let schema = ae_sim::schema::Schema::from_module(&wasm)?;
            Some(Arc::new(ae_catalog::Catalog::load(&vfs, "defs", &schema).context("the content's catalogue")?))
        };
        let teams = match recorded.and_then(|h| h.config.get("teams")) {
            Some(t) => t.parse()?,
            None => o.teams,
        };
        let nav = if vfs.exists(ae_map::MAP_JSON) {
            let map = ae_map::GameMap::load(&vfs)?;
            Some(NavPlugin { teams, ..NavPlugin::loaded(Arc::new(map.to_terrain())) })
        } else if let Some(h) = recorded {
            match h.config.get("map") {
                Some(m) => {
                    let map = map_of(m)?;
                    ensure!(map != Map::Loaded, "the replay's map came from a package; mount the same packages");
                    Some(NavPlugin { teams, ..NavPlugin::new(map, h.config["map_cells"].parse()?) })
                }
                None => None,
            }
        } else {
            o.nav.map(|m| NavPlugin { teams, ..NavPlugin::new(m, o.map_cells) })
        };
        let content = if mounted { Some(Content { hash: vfs.content_hash()?, packages: packages.iter().map(|p| format!("{}@{}", p.manifest.name, p.manifest.version)).collect(), catalog }) } else { None };
        Ok(Self { wasm, nav, content })
    }

    /// The world's header.
    pub fn header(&self, param: u64, checkpoint_every: u64) -> Result<Header> {
        let schema = ae_sim::schema::custom_sections(&self.wasm, ae_abi::SCHEMA_SECTION)?;
        let mut config = BTreeMap::new();
        if let Some(c) = &self.content {
            config.insert("content".into(), format!("{:032x}", c.hash));
            if !c.packages.is_empty() {
                config.insert("packages".into(), c.packages.join(","));
            }
        }
        if let Some(n) = &self.nav {
            config.insert("map".into(), map_name(n.map).into());
            config.insert("map_cells".into(), n.map_cells.to_string());
            config.insert("teams".into(), n.teams.to_string());
        }
        Ok(Header { engine: env!("CARGO_PKG_VERSION").into(), module_hash: ae_replay::hash(&self.wasm), schema_hash: ae_replay::hash(&schema), param, config, checkpoint_every })
    }

    /// The host plugins: nav and spatial with a map, the catalogue with
    /// content defs.
    pub fn plugins(&self) -> Vec<Arc<dyn HostPlugin>> {
        let mut plugins: Vec<Arc<dyn HostPlugin>> = Vec::new();
        if let Some(n) = &self.nav {
            plugins.push(Arc::new(n.clone()));
            plugins.push(Arc::new(SpatialPlugin::square(n.extent(), 64.0)));
        }
        if let Some(c) = self.content.as_ref().and_then(|c| c.catalog.clone()) {
            plugins.push(Arc::new(ae_services::catalog::CatalogPlugin::new(c)));
        }
        plugins
    }

    pub fn runtime(&self, workers: usize, range_rows: usize) -> Result<Runtime> {
        let cfg = RuntimeConfig { max_workers: workers, range_rows, ..RuntimeConfig::default() };
        Runtime::with_hosts(&self.wasm, cfg, self.plugins())
    }
}
