//! The copy layer on a running sim: an end-of-tick capture of every field
//! the schema gives a view class (`#[field(view = ...)]`), sensor coverage
//! stamped from it, and per-observer view tables ([`ae_view`]).
//!
//! A [`ViewPipeline`] covers the entities with a `Body`, a `Team` and a
//! `Sensor` (and any further components named): the map's units when the
//! nav plugin has teams. Coverage comes from `Body.x/y`, `Team.ally` and
//! `Sensor.los/radar`; capturing reads the world and changes nothing in it.

use ae_sensors::{Coverage, SensorGrid, Sensors};
use ae_sim::capture::{Capture, CaptureSpec};
use ae_sim::runtime::{Runtime, World};
use ae_sim::schema::Schema;
use ae_view::{Frame, Observer, ViewClass, ViewSpec, ViewTable};
use anyhow::{Context, Result, ensure};
use std::collections::HashMap;

/// Sensor cell side, world units.
pub const SENSOR_CELL: f32 = 32.0;

pub struct ViewPipeline {
    pub spec: ViewSpec,
    capture_spec: CaptureSpec,
    capture: Capture,
    pub coverage: Coverage,
    dive: Option<WaterColumn>,
    sonar: Option<WaterColumn>,
    /// `(component, field)` of each spec field.
    pub names: Vec<(String, String)>,
    ally: usize,
    los: usize,
    radar: usize,
    /// Scratch columns as floats for the sensors.
    xs: Vec<f32>,
    ys: Vec<f32>,
    ls: Vec<f32>,
    rs: Vec<f32>,
    allies: Vec<u32>,
}

/// Capture an optional component separately so entities that lack it stay in the view.
pub struct WaterColumn {
    spec: CaptureSpec,
    capture: Capture,
    words: HashMap<u64, u32>,
}

impl WaterColumn {
    pub fn new(schema: &Schema, component: &str, field: &str) -> Result<Option<Self>> {
        if !schema.components.iter().any(|c| c.name == component) {
            return Ok(None);
        }
        Ok(Some(Self {
            spec: CaptureSpec::new(schema, &[(component, field)])?,
            capture: Capture::default(),
            words: HashMap::new(),
        }))
    }

    pub fn capture(&mut self, rt: &Runtime, world: &World) -> Result<()> {
        rt.capture(world, &self.spec, &mut self.capture)?;
        self.words.clear();
        self.words.extend(
            self.capture
                .entities
                .iter()
                .copied()
                .zip(self.capture.columns[0].iter().copied()),
        );
        Ok(())
    }

    pub fn get(&self, entity: u64) -> Option<u32> {
        self.words.get(&entity).copied()
    }
}

impl ViewPipeline {
    /// Fields of `Body`, `Team`, `Sensor` and `extra` components with a view
    /// class other than never (4-byte fields only); `allies` allyteams on a
    /// square map of `extent` world units.
    pub fn new(schema: &Schema, extra: &[&str], allies: usize, extent: f32) -> Result<Self> {
        let mut components = vec!["Body", "Team", "Sensor"];
        components.extend(extra.iter().copied().filter(|c| !matches!(*c, "Dive" | "Sonar")));
        let mut names = Vec::new();
        let mut fields = Vec::new();
        for &c in &components {
            let decl = schema
                .components
                .iter()
                .find(|d| d.name == c)
                .with_context(|| format!("the module declares no `{c}`"))?;
            for f in &decl.fields {
                let class = ViewClass::from_wire(f.view())
                    .with_context(|| format!("`{c}.{}` has an unknown view class {}", f.name, f.view()))?;
                if class == ViewClass::Never || f.size != 4 {
                    continue;
                }
                names.push((c.to_string(), f.name.clone()));
                fields.push((format!("{c}.{}", f.name), class));
            }
        }
        // Sensor inputs must be captured even if a game hid them.
        for (c, f) in [
            ("Body", "x"),
            ("Body", "y"),
            ("Team", "ally"),
            ("Sensor", "los"),
            ("Sensor", "radar"),
        ] {
            if !names.iter().any(|(nc, nf)| nc == c && nf == f) {
                names.push((c.to_string(), f.to_string()));
                fields.push((format!("{c}.{f}"), ViewClass::Never));
            }
        }
        let at = |c: &str, f: &str| names.iter().position(|(nc, nf)| nc == c && nf == f).unwrap();
        let (x, y, ally, los, radar) = (
            at("Body", "x"),
            at("Body", "y"),
            at("Team", "ally"),
            at("Sensor", "los"),
            at("Sensor", "radar"),
        );
        let refs: Vec<(&str, &str)> = names.iter().map(|(c, f)| (c.as_str(), f.as_str())).collect();
        let capture_spec = CaptureSpec::new(schema, &refs)?;
        let dive = WaterColumn::new(schema, "Dive", "depth")?;
        let sonar = WaterColumn::new(schema, "Sonar", "radius")?;
        let submerged = dive.as_ref().map(|_| {
            names.push(("Dive".into(), "depth".into()));
            fields.push(("Dive.depth".into(), ViewClass::Los));
            fields.len() - 1
        });
        ensure!(
            fields.len() <= 32,
            "{} viewed fields; a view table takes 32",
            fields.len()
        );
        let spec = ViewSpec {
            fields,
            x,
            y,
            submerged,
            wobble: 3.0 * SENSOR_CELL,
            wobble_ticks: 30,
        };
        let grid = SensorGrid::new(extent, extent, SENSOR_CELL);
        Ok(Self {
            spec,
            capture_spec,
            dive,
            sonar,
            capture: Capture::default(),
            coverage: Coverage::new(grid, allies),
            names,
            ally,
            los,
            radar,
            xs: vec![],
            ys: vec![],
            ls: vec![],
            rs: vec![],
            allies: vec![],
        })
    }

    /// Capture the world's current tick and stamp coverage.
    pub fn capture(&mut self, rt: &Runtime, world: &World) -> Result<()> {
        rt.capture(world, &self.capture_spec, &mut self.capture)?;
        if let Some(dive) = &mut self.dive {
            dive.capture(rt, world)?;
            self.capture.columns.push(
                self.capture
                    .entities
                    .iter()
                    .map(|&e| dive.get(e).unwrap_or_default())
                    .collect(),
            );
        }
        let c = &self.capture;
        let sonar = if let Some(sonar) = &mut self.sonar {
            sonar.capture(rt, world)?;
            Some(
                c.entities
                    .iter()
                    .map(|&e| f32::from_bits(sonar.get(e).unwrap_or_default()))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let f = |col: usize| c.columns[col].iter().map(|w| f32::from_bits(*w)).collect::<Vec<f32>>();
        self.xs = f(self.spec.x);
        self.ys = f(self.spec.y);
        self.ls = f(self.los);
        self.rs = f(self.radar);
        self.allies.clone_from(&c.columns[self.ally]);
        self.coverage.stamp(
            &Sensors {
                x: &self.xs,
                y: &self.ys,
                ally: &self.allies,
                los: &self.ls,
                radar: &self.rs,
                sonar: sonar.as_deref(),
            },
            true,
        );
        Ok(())
    }

    /// Rows captured.
    pub fn rows(&self) -> usize {
        self.capture.len()
    }

    /// The frame of the last capture, for observers to build from.
    pub fn frame(&self) -> (u64, Vec<&[u32]>) {
        (
            self.capture.tick,
            self.capture.columns.iter().map(Vec::as_slice).collect(),
        )
    }

    /// Build `observers`' tables from the last capture.
    pub fn observe(&self, observers: &mut [Observer]) -> Vec<ViewTable> {
        let (tick, columns) = self.frame();
        let frame = Frame::new(tick, &self.capture.entities, &columns, &self.allies, &self.coverage);
        observers.iter_mut().map(|o| o.build(&self.spec, &frame)).collect()
    }

    /// The captured columns (tests compare views against them).
    pub fn captured(&self) -> &Capture {
        &self.capture
    }

    /// Spec field index of `component.field`.
    pub fn field(&self, component: &str, field: &str) -> Option<usize> {
        self.names.iter().position(|(c, f)| c == component && f == field)
    }
}
