//! Record and replay (`--record FILE`, `--replay FILE [--verify]`).
//!
//! Recording runs one worker count and writes a replay ([`ae_replay`]):
//! the header (module and schema hashes, parameter, map configuration),
//! every tick's command bundle (from `--script orders` or `--script bot`,
//! or none), and a
//! checkpoint digest every `--checkpoint-every` ticks and at the end.
//!
//! Replaying builds the world the header names (parameter, map, size and
//! teams from the file; `--game` must be the same module), feeds it the
//! file's bundles only, and with `--verify` compares the digest at every
//! checkpoint: the first mismatch is reported with the digest parts that
//! differ, and the exit status is 1.

use super::setup::Setup;
use super::{Opts, Script};
use ae_abi::engine::MOVE_ORDER;
use ae_replay::{Checkpoint, Recorder, Replay};
use ae_sdk::Entity;
use ae_sdk::nav::{Body, MoveOrder};
use ae_sim::input::{Command, TickBundle};
use ae_sim::runtime::{Runtime, World};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// `--script orders`: `players` players (at most 16); every 40 ticks (from
/// tick 5) each orders about one unit in 16 (by a hash of entity and tick)
/// to a point up to 800 m away, as one group. Deterministic from the world
/// alone.
pub fn orders(rt: &Runtime, world: &World, players: u32) -> Result<TickBundle> {
    let tick = world.tick;
    let mut b = TickBundle::new(tick);
    if tick % 40 != 5 {
        return Ok(b);
    }
    for (e, bytes) in rt.rows(world, "Body")? {
        let h = mix(e ^ tick.rotate_left(17));
        let player = (h % 16) as u32;
        if player >= players {
            continue;
        }
        let body: Body = unsafe { bytes.as_ptr().cast::<Body>().read_unaligned() };
        let (dx, dy) = (((h >> 8) % 1600) as f32 - 800.0, ((h >> 24) % 1600) as f32 - 800.0);
        let group = 1 << 40 | tick << 8 | player as u64;
        let order = MoveOrder::move_to(Entity(e), body.x + dx, body.y + dy, group);
        let bytes = unsafe {
            std::slice::from_raw_parts((&order as *const MoveOrder).cast::<u8>(), size_of::<MoveOrder>()).to_vec()
        };
        b.commands.push(Command {
            player,
            stream: MOVE_ORDER,
            bytes,
        });
    }
    // Canonical order: by player, then as gathered (entity order).
    b.commands.sort_by_key(|c| c.player);
    Ok(b)
}

pub fn record(o: &Opts, path: &str) -> Result<bool> {
    ensure!(o.workers.len() == 1, "--record runs one worker count");
    let setup = Setup::new(o, None)?;
    let param = o.units | (o.param_hi as u64) << 32;
    let every = o.checkpoint_every.max(1);
    let header = setup.header(param, every)?;
    let mut rt = setup.runtime(o.workers[0], o.range_rows)?;
    let mut world = rt.world_for(param, o.workers[0])?;
    let file = std::fs::File::create(path).with_context(|| format!("create {path}"))?;
    let mut rec = Recorder::new(std::io::BufWriter::new(file), &header)?;
    let (mut commands, mut checkpoints) = (0usize, 0usize);
    let bots = if o.script == Script::Bot {
        ae_bot::SkirmishBot::pair(&rt)?.to_vec()
    } else {
        Vec::new()
    };
    let started = Instant::now();
    for _ in 0..o.ticks {
        let mut bundle = if o.script == Script::Orders {
            orders(&rt, &world, 2)?
        } else {
            TickBundle::new(world.tick)
        };
        for b in &bots {
            b.commands(&rt, &world, &mut bundle.commands)?;
        }
        commands += bundle.commands.len();
        rec.bundle(&bundle)?;
        rt.tick_with(&mut world, o.workers[0], &bundle)?;
        if world.tick % every == 0 || world.tick == o.ticks {
            rec.checkpoint(&Checkpoint::of(world.tick, &rt.digest(&world)?))?;
            checkpoints += 1;
        }
    }
    rec.finish(world.tick)?;
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!(
        "recorded {path}: {} ticks, {commands} commands, {checkpoints} checkpoints, {size} bytes, {:.2} s; final digest {:016x}",
        world.tick,
        started.elapsed().as_secs_f64(),
        rt.digest(&world)?.short()
    );
    Ok(true)
}

pub fn replay(o: &Opts, path: &str, verify: bool) -> Result<bool> {
    let replay = Replay::load(Path::new(path))?;
    let h = replay.header()?.clone();
    let setup = Setup::new(o, Some(&h))?;
    h.check(&setup.header(h.param, h.checkpoint_every)?)
        .context("this replay does not match the world (a game with content needs the same --content or packages)")?;
    let workers = o.workers[0];
    let mut rt = setup.runtime(workers, o.range_rows)?;
    let mut world = rt.world_for(h.param, workers)?;
    let mut checkpoints = replay.checkpoints.iter().peekable();
    let started = Instant::now();
    let (mut checked, mut commands) = (0, 0u64);
    while world.tick < replay.end {
        let bundle = replay.bundle(world.tick);
        let s = rt.tick_with(&mut world, workers, &bundle)?;
        commands += s.commands_in;
        while let Some(c) = checkpoints.next_if(|c| c.tick <= world.tick) {
            if c.tick != world.tick || !verify {
                continue;
            }
            let here = Checkpoint::of(world.tick, &rt.digest(&world)?);
            if here.total != c.total {
                println!(
                    "DESYNC at checkpoint tick {}: digest {:016x}, the replay has {:016x}; parts that differ: {}",
                    c.tick,
                    here.total as u64,
                    c.total as u64,
                    here.differing(c).join(", ")
                );
                return Ok(false);
            }
            checked += 1;
        }
    }
    println!(
        "replayed {path}: {} ticks, {commands} commands, {:.2} s at {workers} worker(s); {}; final digest {:016x}",
        world.tick,
        started.elapsed().as_secs_f64(),
        if verify {
            format!("{checked} checkpoints match")
        } else {
            "not verified".into()
        },
        rt.digest(&world)?.short()
    );
    Ok(true)
}

/// A fault to inject: flip the lowest bit of `component.field` of the
/// `row`th entity (by id) holding the component, when the world has run
/// `tick` ticks (before the next one).
#[derive(Clone, Debug)]
pub struct Inject {
    pub tick: u64,
    pub component: String,
    pub field: String,
    pub row: usize,
}

impl Inject {
    /// `TICK:Component.field:ROW`.
    pub fn parse(text: &str) -> Result<Self> {
        let parts: Vec<&str> = text.split(':').collect();
        let [tick, cf, row] = parts.as_slice() else {
            bail!("--inject TICK:Component.field:ROW")
        };
        let (component, field) = cf.split_once('.').context("--inject TICK:Component.field:ROW")?;
        Ok(Self {
            tick: tick.parse()?,
            component: component.into(),
            field: field.into(),
            row: row.parse()?,
        })
    }

    /// Apply it if the world is at its tick; the entity it hit.
    pub fn apply(&self, rt: &Runtime, w: &mut World) -> Result<Option<u64>> {
        if w.tick != self.tick {
            return Ok(None);
        }
        let rows = rt.rows(w, &self.component)?;
        let (entity, bytes) = rows
            .get(self.row)
            .with_context(|| format!("no row {} of `{}`", self.row, self.component))?;
        let c = rt.schema.components.iter().find(|c| c.name == self.component).unwrap();
        let f = c
            .fields
            .iter()
            .find(|f| f.name == self.field)
            .with_context(|| format!("no field `{}`", self.field))?;
        let old = u32::from_le_bytes(bytes[f.offset as usize..f.offset as usize + 4].try_into().unwrap());
        rt.poke(w, *entity, &self.component, &self.field, old ^ 1)?;
        Ok(Some(*entity))
    }
}

/// Every component's rows at one tick, by component name.
type Rows = BTreeMap<String, Vec<(u64, Vec<u8>)>>;

fn dump(rt: &Runtime, w: &World) -> Result<Rows> {
    let mut out = BTreeMap::new();
    for c in rt.schema.components.iter().filter(|c| !c.event) {
        out.insert(c.name.clone(), rt.rows(w, &c.name)?);
    }
    Ok(out)
}

fn show(kind: u32, bits: u32) -> String {
    if kind == ae_abi::FIELD_F32 {
        format!("{} ({bits:#010x})", f32::from_bits(bits))
    } else {
        format!("{bits} ({bits:#010x})")
    }
}

/// `--replay FILE --bisect --inject ...`: find where a run with the fault
/// first differs from a clean run of the same replay, down to the tick,
/// the digest parts, and the entities and fields. Three runs from tick 0
/// (there are no snapshots, and one sim per process): the faulty run
/// recording every tick's digest; the clean run up to the first tick whose
/// digest differs, dumping every row there; the faulty run again to that
/// tick, dumping and comparing. A divergence that shows only in a host
/// service's part (its internal state) is reported as such.
pub fn bisect(o: &Opts, path: &str, inject: &Inject) -> Result<bool> {
    let replay = Replay::load(Path::new(path))?;
    let h = replay.header()?.clone();
    let setup = Setup::new(o, Some(&h))?;
    h.check(&setup.header(h.param, h.checkpoint_every)?)
        .context("this replay does not match the world (a game with content needs the same --content or packages)")?;
    let workers = o.workers[0];
    // Run to `until`, calling `each` after every tick (stop when it says).
    let play =
        |fault: bool, until: u64, each: &mut dyn FnMut(&Runtime, &World) -> Result<bool>| -> Result<Option<u64>> {
            let mut rt = setup.runtime(workers, o.range_rows)?;
            let mut world = rt.world_for(h.param, workers)?;
            let mut hit = None;
            while world.tick < until {
                if fault && let Some(e) = inject.apply(&rt, &mut world)? {
                    hit = Some(e);
                }
                let bundle = replay.bundle(world.tick);
                rt.tick_with(&mut world, workers, &bundle)?;
                if each(&rt, &world)? {
                    break;
                }
            }
            Ok(hit)
        };
    let started = Instant::now();
    // 1. The faulty run: every tick's digest.
    let mut faulty: Vec<u128> = vec![0];
    let hit = play(true, replay.end, &mut |rt, w| {
        faulty.push(rt.digest(w)?.total);
        Ok(false)
    })?;
    let hit = hit.context("the fault's tick was never reached")?;
    // 2. The clean run, to the first tick whose digest differs.
    let mut first = None;
    let mut clean_rows = Rows::new();
    let mut clean_parts = None;
    play(false, replay.end, &mut |rt, w| {
        let d = rt.digest(w)?;
        if d.total != faulty[w.tick as usize] {
            first = Some(w.tick);
            clean_rows = dump(rt, w)?;
            clean_parts = Some(Checkpoint::of(w.tick, &d));
            return Ok(true);
        }
        Ok(false)
    })?;
    let Some(t) = first else {
        println!(
            "no divergence: the fault at tick {} (entity {hit:#x}) left every tick's digest unchanged",
            inject.tick
        );
        return Ok(true);
    };
    // 3. The faulty run again, to that tick.
    let mut faulty_rows = Rows::new();
    let mut faulty_parts = None;
    play(true, t, &mut |rt, w| {
        if w.tick == t {
            faulty_rows = dump(rt, w)?;
            faulty_parts = Some(Checkpoint::of(w.tick, &rt.digest(w)?));
        }
        Ok(false)
    })?;
    let parts = clean_parts.unwrap().differing(&faulty_parts.unwrap());
    println!(
        "first divergence after tick {t} (the fault was applied after tick {}); digest parts that differ: {}",
        inject.tick,
        parts.join(", ")
    );
    // Rows that differ, field by field.
    let schema = {
        let rt = setup.runtime(workers, o.range_rows)?;
        rt.schema.clone()
    };
    let mut lines = 0;
    for (name, clean) in &clean_rows {
        let other: BTreeMap<u64, &Vec<u8>> = faulty_rows
            .get(name)
            .map(|v| v.iter().map(|(e, b)| (*e, b)).collect())
            .unwrap_or_default();
        let c = schema.components.iter().find(|c| &c.name == name).unwrap();
        for (e, a) in clean {
            let Some(b) = other.get(e) else {
                println!("  {name}: entity {e:#x} missing in the faulty run");
                lines += 1;
                continue;
            };
            for f in &c.fields {
                let (o0, o1) = (f.offset as usize, (f.offset + f.size) as usize);
                if a[o0..o1] != b[o0..o1] {
                    if f.size == 4 {
                        let (x, y) = (
                            u32::from_le_bytes(a[o0..o1].try_into().unwrap()),
                            u32::from_le_bytes(b[o0..o1].try_into().unwrap()),
                        );
                        println!(
                            "  {name}.{} of entity {e:#x}: clean {}, faulty {}",
                            f.name,
                            show(f.kind, x),
                            show(f.kind, y)
                        );
                    } else {
                        println!(
                            "  {name}.{} of entity {e:#x}: {:?} vs {:?}",
                            f.name,
                            &a[o0..o1],
                            &b[o0..o1]
                        );
                    }
                    lines += 1;
                }
            }
        }
    }
    if lines == 0 {
        println!(
            "  no component row differs: the divergence is inside host service state ({})",
            parts.join(", ")
        );
    }
    println!(
        "injected: {}.{} of entity {hit:#x} after tick {}; bisect took {:.2} s",
        inject.component,
        inject.field,
        inject.tick,
        started.elapsed().as_secs_f64()
    );
    Ok(false)
}
