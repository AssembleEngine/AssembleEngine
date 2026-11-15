//! Map packages (`.aemap`: a package of kind `map`, a directory or a zip).
//!
//! A map is three content files at the package root:
//! * `map.json` ([`MapInfo`]): the side in cells, the terrain seed, the
//!   water level (decimetres; absent on a dry map), the terrain-type table,
//!   start points (world units) and a description.
//! * `height.png`: 16-bit greyscale, one pixel per cell (row 0 = cell row
//!   0), ground height in decimetres (0..=2400), before water.
//! * `types.png`: 8-bit greyscale, one pixel per cell: an index into the
//!   type table. A type is what gameplay needs of a cell (traversal cost in
//!   tenths, whether it is solid) and the material it shows.
//!
//! Colour and normal detail are not in the map: render takes them from the
//! shared material tilesets by material and slope.
//!
//! [`GameMap::to_terrain`] gives the nav terrain (water put over the ground
//! last, as the generators do); [`GameMap::from_terrain`] goes the other way
//! (one type per distinct cost / solid / material triple), so a generated
//! map can be exported and loads back bit-identical. Files are written
//! deterministically ([`GameMap::files`]): `map.json` with sorted keys,
//! PNGs with fixed settings.

use ae_nav::service::NavConfig;
use ae_nav::terrain::{NO_WATER, TERRAIN_VERSION_INITIAL, Terrain};
use ae_vfs::Vfs;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MAP_JSON: &str = "map.json";
pub const HEIGHT_PNG: &str = "height.png";
pub const TYPES_PNG: &str = "types.png";
/// Highest ground height, decimetres.
pub const MAX_HEIGHT: u16 = 2400;
/// Material names by id (`ae_nav::terrain::Material`).
pub const MATERIALS: [&str; 7] = ["grass", "forest", "sand", "rock", "water", "road", "mud"];

/// One terrain type: what a cell of it is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerrainType {
    pub name: String,
    /// Traversal cost in tenths (10 = ordinary ground, 1..=255).
    pub cost: u8,
    #[serde(default)]
    pub solid: bool,
    /// One of [`MATERIALS`].
    pub material: String,
}

/// `map.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapInfo {
    pub version: u32,
    /// Side in cells (the map is square; 8 world units a cell).
    pub cells: i32,
    #[serde(default)]
    pub seed: u64,
    /// Water surface in decimetres; `None`: a dry map.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub water_level: Option<i16>,
    pub types: Vec<TerrainType>,
    /// Start points, world units.
    #[serde(default)]
    pub starts: Vec<[f32; 2]>,
    #[serde(default)]
    pub description: String,
}

/// A loaded map: its info and per-cell ground heights and type indices.
#[derive(Clone, Debug, PartialEq)]
pub struct GameMap {
    pub info: MapInfo,
    pub height: Vec<u16>,
    pub types: Vec<u8>,
}

fn decode_png(bytes: &[u8], name: &str, cells: i32, depth: png::BitDepth) -> Result<Vec<u8>> {
    let mut d = png::Decoder::new(bytes);
    d.set_transformations(png::Transformations::IDENTITY);
    let mut r = d.read_info().with_context(|| format!("{name}: not a PNG"))?;
    let info = r.info();
    ensure!(info.width == cells as u32 && info.height == cells as u32, "{name} is {}x{}; the map is {cells}x{cells} cells", info.width, info.height);
    ensure!(info.color_type == png::ColorType::Grayscale && info.bit_depth == depth, "{name} must be {}-bit greyscale (is {:?} {:?})", depth as u8, info.color_type, info.bit_depth);
    let mut buf = vec![0; r.output_buffer_size()];
    let f = r.next_frame(&mut buf).with_context(|| format!("{name}: bad PNG data"))?;
    buf.truncate(f.buffer_size());
    Ok(buf)
}

fn encode_png(data: &[u8], cells: i32, depth: png::BitDepth) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut e = png::Encoder::new(&mut out, cells as u32, cells as u32);
    e.set_color(png::ColorType::Grayscale);
    e.set_depth(depth);
    e.set_compression(png::Compression::Default);
    e.set_filter(png::FilterType::Paeth);
    e.set_adaptive_filter(png::AdaptiveFilterType::NonAdaptive);
    let mut w = e.write_header()?;
    w.write_image_data(data)?;
    w.finish()?;
    Ok(out)
}

impl GameMap {
    /// The map at the root of `vfs` (the map package mounted, possibly
    /// under others).
    pub fn load(vfs: &Vfs) -> Result<Self> {
        let info: MapInfo = serde_json::from_slice(&vfs.read(MAP_JSON)?).context(MAP_JSON)?;
        ensure!(info.version == 1, "{MAP_JSON} version {} (this engine reads 1)", info.version);
        ensure!((16..=8192).contains(&info.cells), "{MAP_JSON}: {} cells a side (16..=8192)", info.cells);
        ensure!(!info.types.is_empty() && info.types.len() <= 256, "{MAP_JSON}: {} terrain types (1..=256)", info.types.len());
        for t in &info.types {
            ensure!(t.cost >= 1, "terrain type {}: cost 0 (1..=255)", t.name);
            ensure!(MATERIALS.contains(&t.material.as_str()), "terrain type {}: material {:?} (one of {MATERIALS:?})", t.name, t.material);
        }
        if let Some(w) = info.water_level {
            ensure!((0..=MAX_HEIGHT as i16).contains(&w), "{MAP_JSON}: water level {w} (0..={MAX_HEIGHT})");
        }
        let extent = info.cells as f32 * NavConfig::default().cell_size;
        for s in &info.starts {
            ensure!(s.iter().all(|v| (0.0..extent).contains(v)), "start {s:?} outside the map (0..{extent})");
        }
        let raw = decode_png(&vfs.read(HEIGHT_PNG)?, HEIGHT_PNG, info.cells, png::BitDepth::Sixteen)?;
        let height: Vec<u16> = raw.as_chunks::<2>().0.iter().map(|b| u16::from_be_bytes(*b)).collect();
        if let Some(i) = height.iter().position(|&h| h > MAX_HEIGHT) {
            bail!("{HEIGHT_PNG}: cell ({}, {}) is {} dm high (at most {MAX_HEIGHT})", i as i32 % info.cells, i as i32 / info.cells, height[i]);
        }
        let types = decode_png(&vfs.read(TYPES_PNG)?, TYPES_PNG, info.cells, png::BitDepth::Eight)?;
        if let Some(i) = types.iter().position(|&t| t as usize >= info.types.len()) {
            bail!("{TYPES_PNG}: cell ({}, {}) has type {}; the table has {}", i as i32 % info.cells, i as i32 / info.cells, types[i], info.types.len());
        }
        Ok(Self { info, height, types })
    }

    /// Side in world units.
    pub fn extent(&self) -> f32 {
        self.info.cells as f32 * NavConfig::default().cell_size
    }

    /// The nav terrain (the engine's cell size and sectors).
    pub fn to_terrain(&self) -> Terrain {
        let cfg = NavConfig { map_cells: self.info.cells, seed: self.info.seed, ..NavConfig::default() };
        let g = cfg.terrain_config().geom;
        let n = self.height.len();
        let material: Vec<u8> = self.info.types.iter().map(|t| MATERIALS.iter().position(|m| *m == t.material).unwrap() as u8).collect();
        let mut t = Terrain {
            geom: g,
            seed: self.info.seed,
            version: TERRAIN_VERSION_INITIAL,
            blocked: self.types.iter().map(|&k| self.info.types[k as usize].solid as u8).collect(),
            height: self.height.iter().map(|&h| h as i16).collect(),
            move_cost: self.types.iter().map(|&k| self.info.types[k as usize].cost).collect(),
            material: self.types.iter().map(|&k| material[k as usize]).collect(),
            water_level: NO_WATER,
            water_depth: Vec::new(),
            overlay: Vec::new(),
            base_cost: Vec::new(),
        };
        debug_assert_eq!(t.blocked.len(), n);
        if let Some(level) = self.info.water_level {
            t.flood(level);
        }
        t
    }

    /// The map of a terrain (square, without overlays): one type per
    /// distinct (cost, solid, material), in that order.
    pub fn from_terrain(t: &Terrain, starts: Vec<[f32; 2]>, description: &str) -> Result<Self> {
        ensure!(t.geom.cells_x == t.geom.cells_y, "maps are square");
        ensure!(t.overlay.is_empty(), "a terrain with cost overlays");
        let n = t.len();
        let mut keys: Vec<(u8, u8, u8)> = (0..n).map(|i| (t.move_cost[i], t.blocked[i], t.material[i])).collect();
        keys.sort_unstable();
        keys.dedup();
        ensure!(keys.len() <= 256, "{} distinct terrain types (at most 256)", keys.len());
        let mut height = Vec::with_capacity(n);
        for i in 0..n {
            let g = t.ground_height(i);
            ensure!((0..=MAX_HEIGHT as i16).contains(&g), "cell {i}: ground {g} dm (0..={MAX_HEIGHT})");
            height.push(g as u16);
        }
        let types = (0..n).map(|i| keys.binary_search(&(t.move_cost[i], t.blocked[i], t.material[i])).unwrap() as u8).collect();
        let types_table = keys
            .iter()
            .map(|&(cost, solid, m)| {
                let material = MATERIALS.get(m as usize).with_context(|| format!("material id {m}"))?;
                Ok(TerrainType { name: format!("{material}-{cost}{}", if solid != 0 { "-solid" } else { "" }), cost, solid: solid != 0, material: material.to_string() })
            })
            .collect::<Result<_>>()?;
        let water_level = (t.water_level != NO_WATER).then_some(t.water_level);
        Ok(Self { info: MapInfo { version: 1, cells: t.geom.cells_x, seed: t.seed, water_level, types: types_table, starts, description: description.into() }, height, types })
    }

    /// `map.json`, `height.png` and `types.png`, deterministic.
    pub fn files(&self) -> Result<BTreeMap<String, Vec<u8>>> {
        let mut f = BTreeMap::new();
        // serde_json::Value objects keep keys sorted.
        let mut json = serde_json::to_string_pretty(&serde_json::to_value(&self.info)?)?;
        json.push('\n');
        f.insert(MAP_JSON.into(), json.into_bytes());
        let be: Vec<u8> = self.height.iter().flat_map(|h| h.to_be_bytes()).collect();
        f.insert(HEIGHT_PNG.into(), encode_png(&be, self.info.cells, png::BitDepth::Sixteen)?);
        f.insert(TYPES_PNG.into(), encode_png(&self.types, self.info.cells, png::BitDepth::Eight)?);
        Ok(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(a: &Terrain, b: &Terrain) {
        assert_eq!((a.geom.cells_x, a.geom.cell_size, a.geom.sector_cells, a.seed, a.version), (b.geom.cells_x, b.geom.cell_size, b.geom.sector_cells, b.seed, b.version));
        assert!(a.height == b.height && a.blocked == b.blocked && a.move_cost == b.move_cost && a.material == b.material);
        assert!(a.water_level == b.water_level && a.water_depth == b.water_depth);
        assert_eq!(a.digest(), b.digest());
    }

    fn mount(files: &BTreeMap<String, Vec<u8>>, name: &str) -> (Vfs, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!("ae-map-{}-{name}.aemap", std::process::id()));
        let mut all = files.clone();
        all.insert(ae_vfs::MANIFEST.into(), format!("{{\"name\": \"{name}\", \"version\": \"1\", \"kind\": \"map\"}}").into_bytes());
        std::fs::write(&path, ae_vfs::zip::write(&all).unwrap()).unwrap();
        let mut v = Vfs::new();
        v.mount(&path).unwrap();
        (v, path)
    }

    /// Both crossing maps survive export, an archive and loading, bit for
    /// bit (nav digests included), and the files are deterministic.
    #[test]
    fn generated_maps_round_trip_through_a_package() {
        for (name, t) in [("land", ae_nav::scenario::land_crossing_terrain(256)), ("water", ae_nav::scenario::water_crossing_terrain(256))] {
            let m = GameMap::from_terrain(&t, vec![[300.0, 1800.0], [1800.0, 300.0]], name).unwrap();
            let files = m.files().unwrap();
            assert_eq!(files, m.files().unwrap(), "deterministic");
            let (vfs, path) = mount(&files, name);
            let back = GameMap::load(&vfs).unwrap();
            assert_eq!(back, m);
            same(&back.to_terrain(), &t);
            assert_eq!(back.info.water_level.is_some(), name == "water");
            std::fs::remove_file(path).ok();
        }
    }

    type Edit<'a> = &'a dyn Fn(&mut BTreeMap<String, Vec<u8>>);

    #[test]
    fn bad_maps_are_named() {
        let m = GameMap::from_terrain(&ae_nav::scenario::land_crossing_terrain(64), vec![], "").unwrap();
        let check = |edit: Edit, want: &str| {
            let mut f = m.files().unwrap();
            edit(&mut f);
            let (vfs, path) = mount(&f, "bad");
            let e = format!("{:#}", GameMap::load(&vfs).unwrap_err());
            std::fs::remove_file(path).ok();
            assert!(e.contains(want), "{want:?} in {e}");
        };
        let mut wrong = m.clone();
        wrong.types[5] = 250;
        let types = wrong.files().unwrap().remove(TYPES_PNG).unwrap();
        check(&|f| drop(f.insert(TYPES_PNG.into(), types.clone())), "has type 250");
        let mut small = m.clone();
        small.info.cells = 32;
        small.height.truncate(32 * 32);
        small.types.truncate(32 * 32);
        let h = small.files().unwrap().remove(HEIGHT_PNG).unwrap();
        check(&|f| drop(f.insert(HEIGHT_PNG.into(), h.clone())), "height.png is 32x32; the map is 64x64");
        let t8 = m.files().unwrap().remove(TYPES_PNG).unwrap();
        check(&|f| drop(f.insert(HEIGHT_PNG.into(), t8.clone())), "16-bit greyscale");
        check(&|f| drop(f.remove(TYPES_PNG)), "types.png");
        check(&|f| drop(f.insert(MAP_JSON.into(), String::from_utf8(f[MAP_JSON].clone()).unwrap().replace("\"road\"", "\"lava\"").into_bytes())), "material \"lava\"");
        check(&|f| drop(f.insert(MAP_JSON.into(), String::from_utf8(f[MAP_JSON].clone()).unwrap().replace("\"version\": 1", "\"version\": 1, \"colour\": 3").into_bytes())), "unknown field");
    }
}
