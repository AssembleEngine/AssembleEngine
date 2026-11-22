//! The schema a sim module declares in its `ae.schema` custom section
//! (record layout in `ae_abi`). Parsed and validated before anything is
//! instantiated; the host derives component registration, digest layout and
//! the system schedule from it, with no hard-coded component names.

use crate::schedule::{Cadence, Mode};
use ae_abi as abi;
use ae_ecs::Access;
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldDecl {
    pub id: u64,
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub kind: u32,
    pub flags: u32,
}

impl FieldDecl {
    /// The view class bits (`abi::VIEW_*`; 0, never, when unclassified).
    pub fn view(&self) -> u32 {
        (self.flags & abi::FIELD_VIEW_MASK) >> abi::FIELD_VIEW_SHIFT
    }

    pub fn hashed(&self) -> bool {
        self.flags & abi::FIELD_HASHED != 0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentDecl {
    pub id: u64,
    pub name: String,
    pub size: u32,
    pub align: u32,
    /// An event stream rather than a component.
    pub event: bool,
    /// An event stream players may send commands on.
    pub command: bool,
    pub fields: Vec<FieldDecl>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemRecord {
    pub name: String,
    pub mode: Mode,
    pub cadence: Cadence,
    /// `(component id, access)` in declaration order.
    pub terms: Vec<(u64, Access)>,
    pub emits: Vec<u64>,
    pub reads_events: Vec<u64>,
    /// Components read of other entities during the run (`get(T)`).
    pub gets: Vec<u64>,
    pub after: Vec<String>,
}

impl SystemRecord {
    pub fn export(&self) -> String {
        format!("{}{}", abi::SYSTEM_EXPORT_PREFIX, self.name)
    }
}

/// One `schedule!` entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Slot {
    /// `systems[i]`.
    Module(usize),
    /// A host system the engine must provide (`host(name)`).
    Host(String),
}

#[derive(Clone, Debug, Default)]
pub struct Schema {
    /// Components and event streams, sorted by id.
    pub components: Vec<ComponentDecl>,
    /// The module's own systems, in registration order.
    pub systems: Vec<SystemRecord>,
    /// The whole `schedule!`: module and host systems in registration order.
    pub order: Vec<Slot>,
    /// Named data for engine services (`KIND_DATA` records), by name.
    pub data: BTreeMap<String, Vec<u8>>,
}

/// Parse one component record (as `ae_sdk::Component::SCHEMA` holds it).
pub fn component_record(rec: &[u8]) -> Result<ComponentDecl> {
    ensure!(
        rec.len() >= 8 && abi::get_u32(rec, 0) == abi::KIND_COMPONENT && abi::get_u32(rec, 4) as usize == rec.len(),
        "not a component record"
    );
    component(rec)
}

/// Payloads of every custom section named `name`, concatenated in file order.
pub fn custom_sections(wasm: &[u8], name: &str) -> Result<Vec<u8>> {
    ensure!(wasm.len() >= 8 && &wasm[..4] == b"\0asm", "not a wasm module");
    let mut at = 8;
    let mut out = Vec::new();
    let leb = |at: &mut usize| -> Result<usize> {
        let (mut v, mut shift) = (0usize, 0);
        loop {
            let b = *wasm.get(*at).context("truncated wasm")?;
            *at += 1;
            v |= ((b & 0x7f) as usize) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            ensure!(shift < 35, "bad LEB128 in wasm");
        }
    };
    while at < wasm.len() {
        let id = wasm[at];
        at += 1;
        let size = leb(&mut at)?;
        let end = at
            .checked_add(size)
            .filter(|&e| e <= wasm.len())
            .context("wasm section runs past the end")?;
        if id == 0 {
            let mut p = at;
            let len = leb(&mut p)?;
            let section_name = wasm.get(p..p + len).context("truncated custom section name")?;
            if section_name == name.as_bytes() {
                out.extend_from_slice(&wasm[p + len..end]);
            }
        }
        at = end;
    }
    Ok(out)
}

struct Reader<'a> {
    rec: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn u32(&mut self) -> Result<u32> {
        ensure!(self.at + 4 <= self.rec.len(), "schema record truncated");
        let v = abi::get_u32(self.rec, self.at);
        self.at += 4;
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64> {
        ensure!(self.at + 8 <= self.rec.len(), "schema record truncated");
        let v = abi::get_u64(self.rec, self.at);
        self.at += 8;
        Ok(v)
    }
    fn name(&mut self, len: usize) -> Result<String> {
        let end = self.at + len;
        ensure!(end <= self.rec.len(), "schema record truncated");
        let s = std::str::from_utf8(&self.rec[self.at..end])
            .context("schema name is not UTF-8")?
            .to_string();
        self.at += abi::pad4(len);
        Ok(s)
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        self.name(len)
    }
    fn done(&self) -> Result<()> {
        ensure!(
            self.at == self.rec.len(),
            "schema record has {} unread bytes",
            self.rec.len().saturating_sub(self.at)
        );
        Ok(())
    }
}

fn component(rec: &[u8]) -> Result<ComponentDecl> {
    let mut r = Reader { rec, at: 8 };
    let id = r.u64()?;
    let (size, align, flags, field_count, name_len, _) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?);
    let name = r.name(name_len as usize)?;
    let mut fields = Vec::new();
    for _ in 0..field_count {
        let fid = r.u64()?;
        let (offset, fsize, kind, fflags, fname_len, _) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?);
        let fname = r.name(fname_len as usize)?;
        ensure!(
            offset as u64 + fsize as u64 <= size as u64,
            "{name}.{fname} lies outside the component"
        );
        fields.push(FieldDecl {
            id: fid,
            name: fname,
            offset,
            size: fsize,
            kind,
            flags: fflags,
        });
    }
    r.done()?;
    ensure!(
        align.is_power_of_two() && size > 0 && size % align == 0,
        "{name}: bad size {size} / align {align}"
    );
    let mut ids = BTreeSet::new();
    ensure!(
        fields.iter().all(|f| ids.insert(f.id)),
        "{name}: two fields share an id"
    );
    let mut spans: Vec<(u32, u32)> = fields.iter().map(|f| (f.offset, f.offset + f.size)).collect();
    spans.sort_unstable();
    ensure!(spans.windows(2).all(|w| w[0].1 <= w[1].0), "{name}: fields overlap");
    Ok(ComponentDecl {
        id,
        name,
        size,
        align,
        event: flags & abi::COMPONENT_EVENT != 0,
        command: flags & abi::COMPONENT_COMMAND != 0,
        fields,
    })
}

fn system(rec: &[u8]) -> Result<SystemRecord> {
    let mut r = Reader { rec, at: 8 };
    let version = r.u32()?;
    ensure!(
        version == abi::SYSTEM_RECORD_VERSION,
        "system record version {version} (this host reads {})",
        abi::SYSTEM_RECORD_VERSION
    );
    let (mode, cadence, n, phase) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?);
    let (terms, emits, reads, gets, after, name_len, _) =
        (r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?);
    let name = r.name(name_len as usize)?;
    let mode = match mode {
        abi::MODE_PARALLEL => Mode::Parallel,
        abi::MODE_SERIAL => Mode::Serial,
        m => bail!("system `{name}`: unknown mode {m}"),
    };
    let cadence = match cadence {
        abi::CADENCE_EVERY_TICK => Cadence::EveryTick,
        abi::CADENCE_EVERY => Cadence::Every { n, phase },
        abi::CADENCE_STAGGERED => Cadence::Staggered { n },
        abi::CADENCE_ONCE => Cadence::Once,
        c => bail!("system `{name}`: unknown cadence {c}"),
    };
    ensure!(terms as usize <= abi::MAX_TERMS, "system `{name}`: {terms} terms");
    ensure!(
        reads as usize <= abi::MAX_EVENT_READS,
        "system `{name}`: reads {reads} streams"
    );
    ensure!(gets as usize <= abi::MAX_GETS, "system `{name}`: {gets} gets");
    let terms = (0..terms)
        .map(|_| -> Result<(u64, Access)> {
            let id = r.u64()?;
            let access = match r.u32()? {
                abi::ACCESS_READ => Access::Read,
                abi::ACCESS_WRITE => Access::Write,
                a => bail!("system `{name}`: unknown access {a}"),
            };
            r.u32()?;
            Ok((id, access))
        })
        .collect::<Result<_>>()?;
    let emits = (0..emits).map(|_| r.u64()).collect::<Result<_>>()?;
    let reads_events = (0..reads).map(|_| r.u64()).collect::<Result<_>>()?;
    let gets = (0..gets).map(|_| r.u64()).collect::<Result<_>>()?;
    let after = (0..after).map(|_| r.string()).collect::<Result<_>>()?;
    r.done()?;
    Ok(SystemRecord {
        name,
        mode,
        cadence,
        terms,
        emits,
        reads_events,
        gets,
        after,
    })
}

impl Schema {
    pub fn from_module(wasm: &[u8]) -> Result<Self> {
        let section = custom_sections(wasm, abi::SCHEMA_SECTION)?;
        ensure!(
            !section.is_empty(),
            "module has no `{}` section: was it built with ae-sdk?",
            abi::SCHEMA_SECTION
        );
        Self::parse(&section)
    }

    /// Parse concatenated records. Record order does not matter.
    pub fn parse(section: &[u8]) -> Result<Self> {
        let mut components: BTreeMap<u64, ComponentDecl> = BTreeMap::new();
        let mut systems: BTreeMap<String, SystemRecord> = BTreeMap::new();
        let mut order: Option<Vec<String>> = None;
        let mut data: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut at = 0;
        while at < section.len() {
            ensure!(at + 8 <= section.len(), "schema section truncated");
            let (kind, len) = (abi::get_u32(section, at), abi::get_u32(section, at + 4) as usize);
            ensure!(
                len >= 8 && len % 4 == 0 && at + len <= section.len(),
                "bad schema record length {len}"
            );
            let rec = &section[at..at + len];
            match kind {
                abi::KIND_COMPONENT => {
                    let c = component(rec)?;
                    if let Some(old) = components.get(&c.id) {
                        ensure!(
                            *old == c,
                            "components `{}` and `{}` share id {:#x}",
                            old.name,
                            c.name,
                            c.id
                        );
                    }
                    components.insert(c.id, c);
                }
                abi::KIND_SYSTEM => {
                    let s = system(rec)?;
                    ensure!(!systems.contains_key(&s.name), "system `{}` declared twice", s.name);
                    systems.insert(s.name.clone(), s);
                }
                abi::KIND_ORDER => {
                    ensure!(order.is_none(), "more than one schedule! in the module");
                    let mut r = Reader { rec, at: 8 };
                    let count = r.u32()?;
                    r.u32()?;
                    order = Some((0..count).map(|_| r.string()).collect::<Result<_>>()?);
                    r.done()?;
                }
                abi::KIND_DATA => {
                    let mut r = Reader { rec, at: 8 };
                    let name_len = r.u32()? as usize;
                    r.u32()?;
                    let start = abi::pad4(name_len) + 16;
                    ensure!(start <= rec.len(), "data record name overruns the record");
                    let name = std::str::from_utf8(&rec[16..16 + name_len])
                        .context("data record name is not UTF-8")?
                        .to_string();
                    ensure!(!data.contains_key(&name), "data `{name}` declared twice");
                    data.insert(name, rec[start..].to_vec());
                }
                k => bail!("unknown schema record kind {k}"),
            }
            at += len;
        }
        let mut names = BTreeSet::new();
        for c in components.values() {
            ensure!(names.insert(c.name.as_str()), "two components are named `{}`", c.name);
        }
        let order = order.context("module has no schedule! listing its systems")?;
        let mut ordered = Vec::new();
        let mut slots = Vec::new();
        let mut hosts = BTreeSet::new();
        for name in &order {
            if let Some(host) = name.strip_prefix(abi::HOST_SYSTEM_PREFIX) {
                ensure!(
                    !systems.contains_key(host) && !ordered.iter().any(|s: &SystemRecord| s.name == host),
                    "`{host}` is both a module system and a host system"
                );
                ensure!(
                    hosts.insert(host.to_string()),
                    "schedule! lists host system `{host}` twice"
                );
                slots.push(Slot::Host(host.to_string()));
                continue;
            }
            ensure!(
                !hosts.contains(name),
                "`{name}` is both a module system and a host system"
            );
            slots.push(Slot::Module(ordered.len()));
            ordered.push(
                systems
                    .remove(name)
                    .with_context(|| format!("schedule! lists `{name}`, which is not a #[system]"))?,
            );
        }
        if let Some(name) = systems.keys().next() {
            bail!("system `{name}` is missing from schedule!");
        }
        let schema = Schema {
            components: components.into_values().collect(),
            systems: ordered,
            order: slots,
            data,
        };
        for s in &schema.systems {
            for (id, _) in &s.terms {
                let c = schema
                    .component(*id)
                    .with_context(|| format!("system `{}` queries unknown component {id:#x}", s.name))?;
                ensure!(!c.event, "system `{}` queries event type `{}`", s.name, c.name);
            }
            for id in s.emits.iter().chain(&s.reads_events) {
                let c = schema
                    .component(*id)
                    .with_context(|| format!("system `{}` uses unknown event stream {id:#x}", s.name))?;
                ensure!(
                    c.event,
                    "system `{}` uses component `{}` as an event stream",
                    s.name,
                    c.name
                );
            }
            let mut written = BTreeSet::new();
            for (id, access) in &s.terms {
                if *access == Access::Write {
                    written.insert(*id);
                }
            }
            let mut seen = BTreeSet::new();
            for id in &s.gets {
                let c = schema
                    .component(*id)
                    .with_context(|| format!("system `{}` gets unknown component {id:#x}", s.name))?;
                ensure!(!c.event, "system `{}` gets event type `{}`", s.name, c.name);
                ensure!(
                    !written.contains(id),
                    "system `{}` gets `{}`, which it writes",
                    s.name,
                    c.name
                );
                ensure!(seen.insert(*id), "system `{}` gets `{}` twice", s.name, c.name);
            }
        }
        Ok(schema)
    }

    pub fn component(&self, id: u64) -> Option<&ComponentDecl> {
        self.components
            .binary_search_by_key(&id, |c| c.id)
            .ok()
            .map(|i| &self.components[i])
    }
}
