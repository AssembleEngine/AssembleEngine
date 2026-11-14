//! The catalogue: unit and building definitions ("defs") as data.
//!
//! A def names components and field values; `extends` takes another def's
//! components and values and changes some (field by field); an `abstract`
//! def only exists to be extended. Files are JSON:
//!
//! ```json
//! {
//!   "version": 1,
//!   "defs": {
//!     "unit": { "abstract": true, "components": { "Body": { "radius": 4.0 } } },
//!     "tank": { "extends": "unit", "components": { "NavAgent": { "max_speed": 5.0, "profile": 1 } } }
//!   }
//! }
//! ```
//!
//! [`Catalog::resolve`] checks every def against the game's schema (the
//! component exists and is not an event; the field exists; the value fits
//! its kind: numbers for floats, integers in range for integer kinds,
//! `true`/`false` for booleans; entity, byte and array fields cannot be set
//! from data) and turns it into the bytes of each component, unset fields
//! zero; a 64-bit unsigned field may be given a def's name for its id. A
//! def's id is a hash of its name ([`ae_abi::name_id`], like
//! component ids), so adding a def never renumbers the others.
//!
//! [`CatalogFile::write`] writes the canonical text (defs, components and
//! fields sorted by name, a def's keys in a fixed order, two-space indent;
//! `ae-tool catalog-fmt` applies it): parsing it and writing again gives the same bytes, so files
//! can round-trip through tools.

use ae_sim::schema::{ComponentDecl, FieldDecl, Schema};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FILE_VERSION: u32 = 1;

/// One file as written: defs by name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogFile {
    pub version: u32,
    pub defs: BTreeMap<String, DefSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DefSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extends: Option<String>,
    #[serde(default, rename = "abstract", skip_serializing_if = "std::ops::Not::not")]
    pub is_abstract: bool,
    /// Component name -> field name -> value.
    #[serde(default)]
    pub components: BTreeMap<String, BTreeMap<String, serde_json::Value>>,
}

impl CatalogFile {
    pub fn parse(text: &str) -> Result<Self> {
        let f: CatalogFile = serde_json::from_str(text).context("parsing the catalogue")?;
        ensure!(f.version == FILE_VERSION, "catalogue version {}; this engine reads {FILE_VERSION}", f.version);
        Ok(f)
    }

    /// The canonical text: names sorted, a def's keys in a fixed order,
    /// two-space indent, trailing newline.
    pub fn write(&self) -> String {
        serde_json::to_string_pretty(self).expect("a catalogue serialises") + "\n"
    }
}

/// A def's id: the FNV-1a hash of its name.
pub fn def_id(name: &str) -> u64 {
    ae_abi::name_id(name)
}

/// One component of a resolved def: its schema id and value bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefComponent {
    pub component: u64,
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Def {
    pub id: u64,
    pub name: String,
    /// By component id.
    pub components: Vec<DefComponent>,
}

impl Def {
    pub fn component(&self, name: &str) -> Option<&DefComponent> {
        self.components.iter().find(|c| c.name == name)
    }
}

/// Every concrete def, resolved, by id.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Catalog {
    pub defs: BTreeMap<u64, Def>,
}

impl Catalog {
    pub fn get(&self, name: &str) -> Option<&Def> {
        self.defs.get(&def_id(name))
    }

    /// Every `*.json` under `dir` in `vfs`, as one catalogue (a def name
    /// defined twice is an error).
    pub fn load(vfs: &ae_vfs::Vfs, dir: &str, schema: &Schema) -> Result<Self> {
        let mut files = Vec::new();
        for path in vfs.list(dir, "json")? {
            files.push((path.clone(), CatalogFile::parse(&vfs.read_string(&path)?).with_context(|| path.clone())?));
        }
        ensure!(!files.is_empty(), "no catalogue files under {dir}/");
        Self::resolve(&files, schema)
    }

    /// Merge files, follow `extends`, check against `schema`, build bytes.
    pub fn resolve(files: &[(String, CatalogFile)], schema: &Schema) -> Result<Self> {
        let mut specs: BTreeMap<&str, (&str, &DefSpec)> = BTreeMap::new();
        for (path, f) in files {
            for (name, spec) in &f.defs {
                if let Some((other, _)) = specs.insert(name, (path, spec)) {
                    bail!("def `{name}` is defined in {other} and in {path}");
                }
            }
        }
        let mut ids: BTreeMap<u64, &str> = BTreeMap::new();
        for name in specs.keys() {
            if let Some(other) = ids.insert(def_id(name), name) {
                bail!("defs `{other}` and `{name}` hash to the same id");
            }
        }
        let mut out = Catalog::default();
        for (&name, (path, spec)) in &specs {
            let at = || format!("def `{name}` ({path})");
            // The extends chain, root first.
            let mut chain = vec![(name, *spec)];
            let mut seen = BTreeSet::from([name]);
            let mut cur = *spec;
            while let Some(parent) = &cur.extends {
                let (_, p) = specs.get(parent.as_str()).with_context(|| format!("{}: extends unknown def `{parent}`", at()))?;
                ensure!(seen.insert(parent.as_str()), "{}: `extends` cycle through `{parent}`", at());
                chain.push((parent.as_str(), *p));
                cur = p;
            }
            if spec.is_abstract {
                // Still checked: an abstract def's values must fit.
                for (n, s) in &chain {
                    check_values(schema, n, s).with_context(at)?;
                }
                continue;
            }
            let mut values: BTreeMap<&str, BTreeMap<&str, &serde_json::Value>> = BTreeMap::new();
            for (n, s) in chain.iter().rev() {
                check_values(schema, n, s).with_context(at)?;
                for (c, fields) in &s.components {
                    let entry = values.entry(c.as_str()).or_default();
                    for (f, v) in fields {
                        entry.insert(f.as_str(), v);
                    }
                }
            }
            let mut components = Vec::new();
            for (c, fields) in values {
                let decl = component(schema, c)?;
                let mut bytes = vec![0u8; decl.size as usize];
                for (f, v) in fields {
                    let field = decl.fields.iter().find(|d| d.name == f).unwrap();
                    let encoded = encode(field, v).unwrap();
                    bytes[field.offset as usize..field.offset as usize + encoded.len()].copy_from_slice(&encoded);
                }
                components.push(DefComponent { component: decl.id, name: c.to_string(), bytes });
            }
            components.sort_by_key(|c| c.component);
            out.defs.insert(def_id(name), Def { id: def_id(name), name: name.to_string(), components });
        }
        Ok(out)
    }
}

fn component<'a>(schema: &'a Schema, name: &str) -> Result<&'a ComponentDecl> {
    let c = schema.components.iter().find(|c| c.name == name).with_context(|| format!("no component `{name}` in the game's schema"))?;
    ensure!(!c.event, "`{name}` is an event stream, not a component");
    Ok(c)
}

fn check_values(schema: &Schema, def: &str, spec: &DefSpec) -> Result<()> {
    for (c, fields) in &spec.components {
        let decl = component(schema, c).with_context(|| format!("in `{def}`"))?;
        for (f, v) in fields {
            let field = decl.fields.iter().find(|d| d.name == *f).with_context(|| format!("in `{def}`: `{c}` has no field `{f}`"))?;
            encode(field, v).with_context(|| format!("in `{def}`: `{c}.{f}`"))?;
        }
    }
    Ok(())
}

/// A JSON value as the field's bytes.
fn encode(field: &FieldDecl, v: &serde_json::Value) -> Result<Vec<u8>> {
    use ae_abi::*;
    let int = |lo: i128, hi: i128| -> Result<i128> {
        let n = v.as_i64().map(i128::from).or_else(|| v.as_u64().map(i128::from)).with_context(|| format!("expected an integer, got {v}"))?;
        ensure!((lo..=hi).contains(&n), "{n} is out of range {lo}..={hi}");
        Ok(n)
    };
    let float = || v.as_f64().with_context(|| format!("expected a number, got {v}"));
    Ok(match field.kind {
        FIELD_F32 => (float()? as f32).to_le_bytes().to_vec(),
        FIELD_F64 => float()?.to_le_bytes().to_vec(),
        FIELD_U8 => vec![int(0, u8::MAX as i128)? as u8],
        FIELD_I8 => vec![int(i8::MIN as i128, i8::MAX as i128)? as i8 as u8],
        FIELD_U16 => (int(0, u16::MAX as i128)? as u16).to_le_bytes().to_vec(),
        FIELD_I16 => (int(i16::MIN as i128, i16::MAX as i128)? as i16).to_le_bytes().to_vec(),
        FIELD_U32 => (int(0, u32::MAX as i128)? as u32).to_le_bytes().to_vec(),
        FIELD_I32 => (int(i32::MIN as i128, i32::MAX as i128)? as i32).to_le_bytes().to_vec(),
        // A def name stands for its id (`"builds": "tank"`).
        FIELD_U64 if v.is_string() => def_id(v.as_str().unwrap()).to_le_bytes().to_vec(),
        FIELD_U64 => (int(0, u64::MAX as i128)? as u64).to_le_bytes().to_vec(),
        FIELD_I64 => (int(i64::MIN as i128, i64::MAX as i128)? as i64).to_le_bytes().to_vec(),
        FIELD_BOOL => {
            let b = v.as_bool().with_context(|| format!("expected true or false, got {v}"))?;
            ensure!(field.size == 1 || field.size == 4, "a bool of {} bytes", field.size);
            let mut out = vec![0u8; field.size as usize];
            out[0] = b as u8;
            out
        }
        k => bail!("fields of kind {k} (entity, bytes, arrays) cannot be set from data"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_sim::schema::{ComponentDecl, FieldDecl};

    fn field(name: &str, offset: u32, size: u32, kind: u32) -> FieldDecl {
        FieldDecl { id: ae_abi::name_id(name), name: name.into(), offset, size, kind, flags: 0 }
    }

    fn schema() -> Schema {
        let c = |name: &str, size, fields| ComponentDecl { id: ae_abi::name_id(name), name: name.into(), size, align: 4, event: false, command: false, fields };
        let mut components = vec![
            c("Body", 20, vec![field("x", 0, 4, ae_abi::FIELD_F32), field("y", 4, 4, ae_abi::FIELD_F32), field("radius", 16, 4, ae_abi::FIELD_F32)]),
            c("Health", 8, vec![field("hp", 0, 4, ae_abi::FIELD_U32), field("armor", 4, 2, ae_abi::FIELD_I16)]),
            c("Target", 8, vec![field("entity", 0, 8, ae_abi::FIELD_ENTITY)]),
            ComponentDecl { event: true, ..c("Hit", 4, vec![field("dmg", 0, 4, ae_abi::FIELD_U32)]) },
        ];
        components.sort_by_key(|c| c.id);
        Schema { components, ..Schema::default() }
    }

    const TEXT: &str = r#"{
  "version": 1,
  "defs": {
    "heavy": {
      "extends": "unit",
      "components": {
        "Health": {
          "armor": -3,
          "hp": 900
        }
      }
    },
    "unit": {
      "abstract": true,
      "components": {
        "Body": {
          "radius": 4.5
        },
        "Health": {
          "hp": 100
        }
      }
    }
  }
}
"#;

    #[test]
    fn canonical_text_round_trips_byte_for_byte() {
        let f = CatalogFile::parse(TEXT).unwrap();
        assert_eq!(f.write(), TEXT);
        assert_eq!(CatalogFile::parse(&f.write()).unwrap(), f);
        // Non-canonical input (other key order, one line) writes canonically.
        let messy = r#"{"defs":{"unit":{"components":{"Health":{"hp":100},"Body":{"radius":4.5}},"abstract":true},"heavy":{"components":{"Health":{"hp":900,"armor":-3}},"extends":"unit"}},"version":1}"#;
        assert_eq!(CatalogFile::parse(messy).unwrap().write(), TEXT);
    }

    #[test]
    fn defs_resolve_through_extends_to_component_bytes() {
        let c = Catalog::resolve(&[("a.json".into(), CatalogFile::parse(TEXT).unwrap())], &schema()).unwrap();
        assert_eq!(c.defs.len(), 1, "abstract defs are not spawnable");
        let heavy = c.get("heavy").unwrap();
        assert_eq!(heavy.id, def_id("heavy"));
        let health = heavy.component("Health").unwrap();
        assert_eq!(health.bytes, [900u32.to_le_bytes().as_slice(), &(-3i16).to_le_bytes(), &[0, 0]].concat());
        let body = heavy.component("Body").unwrap();
        assert_eq!(&body.bytes[16..20], &4.5f32.to_le_bytes(), "inherited");
        assert_eq!(&body.bytes[0..16], &[0; 16], "unset fields are zero");
    }

    #[test]
    fn errors_name_the_def_and_the_problem() {
        let err = |text: &str| {
            let f = CatalogFile::parse(text).map_err(|e| format!("{e:#}"))?;
            Catalog::resolve(&[("x.json".into(), f)], &schema()).map(drop).map_err(|e| format!("{e:#}"))
        };
        let one = |def: &str| format!(r#"{{"version": 1, "defs": {{"a": {def}}}}}"#);
        let cases = [
            (one(r#"{"components": {"Tank": {"hp": 1}}}"#), "no component `Tank`"),
            (one(r#"{"components": {"Health": {"hpp": 1}}}"#), "`Health` has no field `hpp`"),
            (one(r#"{"components": {"Health": {"hp": 1.5}}}"#), "expected an integer"),
            (one(r#"{"components": {"Health": {"hp": -1}}}"#), "out of range"),
            (one(r#"{"components": {"Health": {"armor": 40000}}}"#), "out of range"),
            (one(r#"{"components": {"Body": {"x": "far"}}}"#), "expected a number"),
            (one(r#"{"components": {"Target": {"entity": 5}}}"#), "cannot be set from data"),
            (one(r#"{"components": {"Hit": {"dmg": 5}}}"#), "is an event stream"),
            (one(r#"{"extends": "b"}"#), "extends unknown def `b`"),
            (r#"{"version": 1, "defs": {"a": {"extends": "b"}, "b": {"extends": "a"}}}"#.to_string(), "`extends` cycle"),
            (r#"{"version": 2, "defs": {}}"#.to_string(), "version 2"),
            (one(r#"{"colour": "red"}"#), "unknown field"),
        ];
        for (text, want) in cases {
            let e = err(&text).unwrap_err();
            assert!(e.contains(want), "{want:?} not in {e:?}");
            assert!(e.contains("parsing") || e.contains("`a`") || e.contains("`b`") || e.contains("version"), "names the def: {e}");
        }
        // The same def in two files.
        let f = CatalogFile::parse(&one("{}")).unwrap();
        let e = Catalog::resolve(&[("x.json".into(), f.clone()), ("y.json".into(), f)], &schema()).unwrap_err().to_string();
        assert!(e.contains("defined in x.json and in y.json"), "{e}");
    }
}
