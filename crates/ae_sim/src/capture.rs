//! End-of-tick column capture: a copy of chosen 4-byte fields of every
//! entity that has all their components, taken between ticks for consumers
//! outside the sim (the renderer's instance streams today, M4's copy layer
//! later). Fields are named by component and field, resolved through the
//! schema, so the sim core names no components itself.
//!
//! A capture reads the world through an uncached query, which creates no
//! flecs entity: capturing every tick leaves entity ids, and so the digest,
//! exactly as without it (`tests/capture.rs`).

use crate::schema::Schema;
use anyhow::{Context, Result, ensure};

/// Which fields to capture, resolved against one schema.
#[derive(Clone, Debug)]
pub struct CaptureSpec {
    /// Schema component index of each query term, in first-use order.
    pub(crate) components: Vec<usize>,
    /// Per captured field: `(term, byte offset in the component)`.
    pub(crate) fields: Vec<(usize, usize)>,
    pub(crate) sizes: Vec<usize>,
}

impl CaptureSpec {
    /// `fields` as `(component, field)` names; each field must be 4 bytes
    /// (an `f32`, `u32` or `i32`). Rows are the entities with every one of
    /// the named components.
    pub fn new(schema: &Schema, fields: &[(&str, &str)]) -> Result<Self> {
        ensure!(!fields.is_empty(), "a capture needs at least one field");
        let mut components = Vec::new();
        let mut out = Vec::new();
        for &(component, field) in fields {
            let ci = schema.components.iter().position(|c| c.name == component).with_context(|| format!("no component `{component}`"))?;
            let decl = &schema.components[ci];
            ensure!(!decl.event, "`{component}` is an event stream");
            let f = decl.fields.iter().find(|f| f.name == field).with_context(|| format!("`{component}` has no field `{field}`"))?;
            ensure!(f.size == 4, "`{component}.{field}` is {} bytes; captures take 4-byte fields", f.size);
            let term = match components.iter().position(|&c| c == ci) {
                Some(t) => t,
                None => {
                    components.push(ci);
                    components.len() - 1
                }
            };
            out.push((term, f.offset as usize));
        }
        ensure!(components.len() <= ae_ecs::MAX_QUERY_TERMS, "a capture spans at most {} components", ae_ecs::MAX_QUERY_TERMS);
        let sizes = components.iter().map(|&c| schema.components[c].size as usize).collect();
        Ok(Self { components, fields: out, sizes })
    }

    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

/// One capture: entity ids and one column of 4-byte words per field, row
/// `r` of every column belonging to `entities[r]`. Rows come in table
/// order, which changes when entities join or leave tables; match rows
/// across captures by entity. Reused between captures to keep allocations.
#[derive(Clone, Debug, Default)]
pub struct Capture {
    pub tick: u64,
    pub entities: Vec<u64>,
    /// Field-major: `columns[field][row]`, raw bits (`f32::from_bits` for
    /// float fields).
    pub columns: Vec<Vec<u32>>,
}

impl Capture {
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Field `field` of row `row` as an `f32`.
    pub fn f32(&self, field: usize, row: usize) -> f32 {
        f32::from_bits(self.columns[field][row])
    }
}
