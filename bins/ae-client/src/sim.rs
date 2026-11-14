//! The sim beside the renderer: the crossing game (`games/crossing`) with
//! the nav and spatial host systems on the same map the terrain shows,
//! ticked at the nav service's rate on its own thread. After every tick the
//! units' `Body` and `NavAgent` fields are captured
//! ([`ae_sim::runtime::Runtime::capture`]) and turned into the renderer's
//! [`InstanceStream`]: rows matched by entity against the previous capture
//! for the interpolation start and the heading. Stand-in for M4's copy
//! layer, which will publish filtered view tables instead.

use ae_nav::math::TICK_SECONDS;
use ae_services::nav::{GroundFeed, Map, NavPlugin};
use ae_services::spatial::SpatialPlugin;
use ae_render::units::{ATTRIBUTE_COUNT, InstanceStream, UnitClass, UnitFeed};
use ae_sim::capture::{Capture, CaptureSpec};
use ae_sim::runtime::{Runtime, RuntimeConfig, World};
use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// What the renderer needs of a unit, by field.
const FIELDS: [(&str, &str); 6] = [("Body", "x"), ("Body", "y"), ("Body", "vx"), ("Body", "vy"), ("Body", "radius"), ("NavAgent", "profile")];
const F_X: usize = 0;
const F_Y: usize = 1;
const F_VX: usize = 2;
const F_VY: usize = 3;
const F_RADIUS: usize = 4;
const F_PROFILE: usize = 5;

#[derive(Clone, Copy, Debug)]
pub struct SimOptions {
    pub map: Map,
    pub map_cells: i32,
    pub units: u32,
    pub workers: usize,
    /// The crossing game's `quake` tick (1..=255; 0: none): rock squares,
    /// then raised squares a tick later, landing with the nav repairs.
    pub quake: u8,
}

impl SimOptions {
    fn plugin(&self) -> NavPlugin {
        NavPlugin::new(self.map, self.map_cells)
    }

    /// The game parameter: unit count, quake tick (orders at tick 0).
    fn param(&self) -> u64 {
        self.units as u64 | (self.quake as u64) << 24
    }

    /// One look per movement class of the map (the crossing game declares
    /// the same classes): ships and hovercraft float.
    pub fn classes(&self) -> Vec<UnitClass> {
        const COLORS: [[f32; 3]; 5] = [[0.95, 0.75, 0.10], [0.80, 0.12, 0.08], [0.92, 0.92, 0.96], [0.10, 0.80, 0.90], [0.75, 0.20, 0.85]];
        self.plugin().map_profiles().iter().enumerate().map(|(i, p)| UnitClass { color: COLORS[i % COLORS.len()], floats: p.medium.on_surface() }).collect()
    }
}

/// The sim and its world, on the thread that made them.
pub struct Sim {
    /// Before `rt`: fields drop in order, and the world's columns live in
    /// the runtime's memory.
    world: World,
    rt: Runtime,
    workers: usize,
    spec: CaptureSpec,
    capture: Capture,
    streamer: Streamer,
}

impl Sim {
    /// The world; landed ground changes go to `ground`, if given.
    pub fn new(o: &SimOptions, ground: Option<Arc<GroundFeed>>) -> Result<Self> {
        let wasm = ae_sim::build::game("crossing").context("building games/crossing")?;
        let nav = NavPlugin { ground, ..o.plugin() };
        let spatial = SpatialPlugin::square(nav.extent(), 64.0);
        let cfg = RuntimeConfig { max_workers: o.workers, ..RuntimeConfig::default() };
        let rt = Runtime::with_hosts(&wasm, cfg, vec![Arc::new(nav), Arc::new(spatial)])?;
        let world = rt.world_for(o.param(), o.workers)?;
        let spec = CaptureSpec::new(&rt.schema, &FIELDS)?;
        Ok(Self { world, rt, workers: o.workers, spec, capture: Capture::default(), streamer: Streamer::default() })
    }

    pub fn tick(&mut self) -> Result<()> {
        self.rt.tick(&mut self.world, self.workers)?;
        Ok(())
    }

    /// Units of movement class `class` with a non-zero velocity in the last
    /// capture.
    pub fn moving(&self, class: u32) -> usize {
        let c = &self.capture;
        (0..c.len()).filter(|&r| c.columns[F_PROFILE][r] == class && (c.f32(F_VX, r) != 0.0 || c.f32(F_VY, r) != 0.0)).count()
    }

    /// Capture the world and build the stream (`tick_seconds` 0: no
    /// interpolation).
    pub fn stream(&mut self, tick_seconds: f32) -> Result<InstanceStream> {
        self.rt.capture(&self.world, &self.spec, &mut self.capture)?;
        Ok(self.streamer.stream(&self.capture, Instant::now(), tick_seconds))
    }
}

/// Turns captures into streams, remembering the last one by entity.
#[derive(Default)]
pub struct Streamer {
    /// Previous capture, sorted by entity: `(entity, x, y, heading x, heading y)`.
    previous: Vec<(u64, f32, f32, f32, f32)>,
    order: Vec<u32>,
}

impl Streamer {
    /// The stream of `capture` (rows in entity order). A unit's previous
    /// position is its position in the previous capture (its current one
    /// if it is new); its heading is its velocity, or its last heading
    /// while it stands still.
    pub fn stream(&mut self, capture: &Capture, published: Instant, tick_seconds: f32) -> InstanceStream {
        let n = capture.len();
        self.order.clear();
        self.order.extend(0..n as u32);
        self.order.sort_unstable_by_key(|&r| capture.entities[r as usize]);
        let mut columns: [Vec<u32>; ATTRIBUTE_COUNT] = std::array::from_fn(|_| Vec::with_capacity(n));
        let mut next = Vec::with_capacity(n);
        let mut p = 0;
        for &r in &self.order {
            let r = r as usize;
            let entity = capture.entities[r];
            while p < self.previous.len() && self.previous[p].0 < entity {
                p += 1;
            }
            let before = self.previous.get(p).filter(|prev| prev.0 == entity);
            let (x, y) = (capture.f32(F_X, r), capture.f32(F_Y, r));
            let (vx, vy) = (capture.f32(F_VX, r), capture.f32(F_VY, r));
            let (px, py) = before.map_or((x, y), |b| (b.1, b.2));
            let (hx, hy) = if vx * vx + vy * vy > 1e-6 { (vx, vy) } else { before.map_or((1.0, 0.0), |b| (b.3, b.4)) };
            for (column, word) in columns.iter_mut().zip([x.to_bits(), y.to_bits(), px.to_bits(), py.to_bits(), hx.to_bits(), hy.to_bits(), capture.columns[F_RADIUS][r], capture.columns[F_PROFILE][r]]) {
                column.push(word);
            }
            next.push((entity, x, y, hx, hy));
        }
        self.previous = next;
        InstanceStream { tick: capture.tick, published, tick_seconds, columns }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimStats {
    pub tick: u64,
    pub units: usize,
    /// Mean over the last report interval.
    pub tick_ms: f64,
    pub capture_ms: f64,
    /// Ticks run late (the sim could not keep the tick rate).
    pub late: u64,
    pub ready: bool,
    /// The sim stopped with an error (printed when it happened).
    pub failed: bool,
}

/// The sim thread.
pub struct SimThread {
    stop: Arc<AtomicBool>,
    stats: Arc<Mutex<SimStats>>,
    handle: Option<JoinHandle<Result<()>>>,
}

impl SimThread {
    /// Build the world and tick it at the nav rate, publishing units to
    /// `feed` and landed ground changes to `ground`.
    pub fn start(o: SimOptions, feed: Arc<UnitFeed>, ground: Arc<GroundFeed>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(SimStats::default()));
        let (stop2, stats2) = (stop.clone(), stats.clone());
        let handle = std::thread::Builder::new()
            .name("sim".into())
            .spawn(move || {
                let result = run(o, feed, ground, &stop2, &stats2);
                if let Err(error) = &result {
                    eprintln!("sim failed: {error:#} (--units 0 runs without the sim)");
                    stats2.lock().unwrap_or_else(|p| p.into_inner()).failed = true;
                }
                result
            })
            .expect("spawn the sim thread");
        Self { stop, stats, handle: Some(handle) }
    }

    pub fn stats(&self) -> SimStats {
        *self.stats.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn stop(mut self) -> Result<SimStats> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().map_err(|_| anyhow::anyhow!("the sim thread panicked"))??;
        }
        Ok(self.stats())
    }
}

fn run(o: SimOptions, feed: Arc<UnitFeed>, ground: Arc<GroundFeed>, stop: &AtomicBool, stats: &Mutex<SimStats>) -> Result<()> {
    let started = Instant::now();
    let mut sim = Sim::new(&o, Some(ground))?;
    eprintln!("sim: crossing game, {} units, {} workers, world built in {:.2} s", o.units, o.workers, started.elapsed().as_secs_f32());
    let period = Duration::from_secs_f32(TICK_SECONDS);
    let mut next = Instant::now();
    let (mut tick_sum, mut capture_sum, mut count) = (0.0, 0.0, 0u32);
    let mut late = 0;
    while !stop.load(Ordering::Relaxed) {
        let t0 = Instant::now();
        sim.tick()?;
        let t1 = Instant::now();
        let stream = sim.stream(TICK_SECONDS)?;
        let (tick, units) = (stream.tick, stream.len());
        feed.publish(Arc::new(stream));
        tick_sum += (t1 - t0).as_secs_f64() * 1e3;
        capture_sum += t1.elapsed().as_secs_f64() * 1e3;
        count += 1;
        if count == 30 {
            let mut s = stats.lock().unwrap_or_else(|p| p.into_inner());
            *s = SimStats { tick, units, tick_ms: tick_sum / count as f64, capture_ms: capture_sum / count as f64, late, ready: true, failed: false };
            (tick_sum, capture_sum, count) = (0.0, 0.0, 0);
        }
        next += period;
        let now = Instant::now();
        if now < next {
            std::thread::sleep(next - now);
        } else {
            late += 1;
            if now - next > period * 8 {
                // Too far behind to catch up: drop the backlog.
                next = now;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_render::units::ATTRIBUTES;

    fn capture(tick: u64, rows: &[(u64, f32, f32, f32, f32)]) -> Capture {
        let mut c = Capture { tick, entities: Vec::new(), columns: vec![Vec::new(); FIELDS.len()] };
        for &(e, x, y, vx, vy) in rows {
            c.entities.push(e);
            for (f, word) in [x.to_bits(), y.to_bits(), vx.to_bits(), vy.to_bits(), 2.5f32.to_bits(), (e % 3) as u32].into_iter().enumerate() {
                c.columns[f].push(word);
            }
        }
        c
    }

    fn column(s: &InstanceStream, name: &str) -> Vec<f32> {
        let a = ATTRIBUTES.iter().position(|&n| n == name).unwrap();
        s.columns[a].iter().map(|&w| f32::from_bits(w)).collect()
    }

    /// Rows come out in entity order; previous positions and headings
    /// follow entities across reordered captures; new units start where
    /// they are; standing units keep their heading.
    #[test]
    fn streams_match_rows_by_entity() {
        let mut s = Streamer::default();
        let now = Instant::now();
        let first = s.stream(&capture(1, &[(9, 90.0, 0.0, 0.0, 1.0), (4, 40.0, 0.0, 1.0, 0.0)]), now, 0.1);
        assert_eq!(column(&first, "x"), [40.0, 90.0]);
        assert_eq!(column(&first, "prev_x"), [40.0, 90.0], "first sight: no motion");
        // Table order changed, 4 stopped, 7 joined, 9 moved.
        let second = s.stream(&capture(2, &[(7, 70.0, 5.0, 0.0, 0.0), (9, 91.0, 0.5, 0.0, 1.0), (4, 40.0, 0.0, 0.0, 0.0)]), now, 0.1);
        assert_eq!(column(&second, "x"), [40.0, 70.0, 91.0]);
        assert_eq!(column(&second, "prev_x"), [40.0, 70.0, 90.0]);
        assert_eq!(column(&second, "prev_y"), [0.0, 5.0, 0.0]);
        assert_eq!(column(&second, "heading_x"), [1.0, 1.0, 0.0], "4 keeps its heading, 7 gets the default");
        assert_eq!(column(&second, "heading_y"), [0.0, 0.0, 1.0]);
        assert_eq!(second.columns[7], [1, 1, 0], "class = profile");
        assert_eq!(second.tick, 2);
        // 9 left.
        let third = s.stream(&capture(3, &[(4, 41.0, 0.0, 1.0, 0.0), (7, 70.0, 5.0, 0.0, 0.0)]), now, 0.1);
        assert_eq!(column(&third, "prev_x"), [40.0, 70.0]);
    }
}
