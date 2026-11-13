//! The engine's navigation service: pathsim's simulation with unit bodies
//! moved out to caller-owned columns.
//!
//! Started as a copy of [`crate::reference::sim`]; the differences are:
//! * units are added and removed by engine entity id (`add_unit`,
//!   `remove_unit`); iteration order is insertion order and the local unit
//!   id is a stable handle, no longer `index + 1`;
//! * [`NavService::step`] reads every unit's position, velocity and radius
//!   from [`BodyColumns`] at the start of the tick and writes position and
//!   velocity back at the end, so the ECS is authoritative for bodies
//!   between ticks and the service keeps only navigation bookkeeping;
//! * movement profiles come from the caller;
//! * no render snapshots by default (the copy layer reads the ECS).
//!
//! Driven with the same scenario, it gives the same state hash as the
//! reference (`tests/service_vs_reference.rs`).
//!
//! Tick order (all deterministic):
//! 1. queued terrain edits become visible (and start a local nav repair);
//! 2. worker results due at this tick are applied in job-ID order;
//! 3. queued commands are interpreted (only against an up-to-date hierarchy);
//! 4. steering from shared fields, arrival slots and stage progress;
//! 5. fixed 1/30 s integration with collision substeps;
//! 6. progress / blocked-unit policy and command status;
//! 7. new navigation jobs are submitted under a fixed per-tick quota, due at
//!    the next tick boundary.
use crate::avoidance::{compute_velocities, Agent, AvoidanceConfig};
use crate::par::Par;
use crate::collision::{Body, Collider, CollisionConfig, CollisionStats, ContactView};
pub use crate::reference::commands::{ArrivalSlot, CommandKind, CommandPayload, CommandRecord, CommandStatus};
use crate::math::{CellPos, Rect, SectorPos, WorldPos, TICK_HZ, TICK_SECONDS};
use crate::nav::crowd::{CrowdMap, Traffic, CROWD_AWARE_MIN_UNITS, UPDATE_TICKS as CROWD_UPDATE_TICKS};

/// Times a stalled unit's give-up timer restarts before it gives up.
const STUCK_RETRIES: u8 = 2;

/// Ticks an arrived unit stands before it counts as standing crowd.
const CROWD_ARRIVED_TICKS: u64 = 150;
/// At most this many cohorts adopt a new crowd map per crowd update, those
/// that adopted one longest ago first: a map changes every second or so
/// while traffic moves, and re-planning every crowd-aware cohort on each
/// change (hundreds for a large formation) kept the simulation thread busy
/// with context builds and re-keys. A command with few cohorts still
/// adopts each map at once.
const CROWD_REKEYS_PER_UPDATE: usize = 32;
/// A unit on its way that has made no progress for this long counts as
/// standing in the crowd maps: a queue at a gap.
const CROWD_QUEUED_TICKS: u32 = 45;
use crate::nav::fields::{FieldCache, FieldCacheStats, FieldKey, FlowField};
use crate::nav::hierarchy::{NavHierarchy, NavStats, Portal, ProfileHierarchy};
use crate::nav::planner::{build_context, context_key, field_recipe, regions_ahead, route_polyline, ContextKey, DestContext};
use crate::nav::{profiles_for, MovementProfile, RADIUS_JITTER_MAX};
use crate::rng::{Rng, StateHasher};
use crate::terrain::{CellEdit, SharedTerrain, Terrain, TerrainConfig, TerrainPatch};
use crate::workers::{ComputePool, WorkerJob, WorkerResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

pub const SNAPSHOT_CONTACT_LIMIT: usize = 512;
pub const HISTORY_LIMIT: usize = 64;
/// Deterministic per-tick work quotas (counts, never wall-clock time).
pub const FIELD_JOBS_PER_TICK: usize = 96;
/// Destination contexts built per tick. They are built synchronously, in
/// parallel on the simulation's thread pool (about 0.6 ms each on the
/// default map), so units have their coarse route on the command's tick.
pub const CONTEXT_BUILDS_PER_TICK: usize = 24;
/// Field-job pacing (see `submit_requests`): assumed field-building
/// throughput, in cells per millisecond of sim time. Deliberately about
/// half of one worker (a 64² field takes about 0.28 ms), so a burst is
/// spread over several ticks instead of all being due on the next one, and
/// even a single worker rarely makes a tick wait. (15,000, one worker's
/// full rate, still left a 10k-unit command tick over budget with one
/// worker.)
pub const FIELD_PACE_CELLS_PER_MS: u64 = 8000;
/// Latest a requested field may land after its request. Until then the
/// unit steers along the coarse (portal-level) route.
pub const FIELD_MAX_DELAY_MS: u64 = 200;
/// Prefetch requests wait while the pacing backlog is longer than this.
pub const PREFETCH_BACKLOG_MS: u64 = 33;
/// Sim time from a terrain edit to its installation (with the navigation
/// repair it needs): two ticks at 30 Hz, so the copy and the repair (about
/// 10 ms together on a 2560² map) do not hold up a tick.
pub const REPAIR_DELAY_MS: u64 = 66;
/// Sectors of fields requested ahead of each unit along its abstract route.
pub const PREFETCH_SECTORS: usize = 3;
/// Avoidance look-ahead (seconds) for a unit in its final approach to its slot.
pub const ARRIVAL_AVOIDANCE_HORIZON: f32 = 0.33;
/// Field steps walked ahead of a unit when choosing its steering waypoint.
pub const LOOKAHEAD_STEPS: usize = 48;
/// Cost slack (a tenth of a plain straight step) within which the look-ahead
/// walk treats field steps as equally good and prefers the direct heading.
pub const LOOKAHEAD_TIE_BAND: u64 = 10;
/// Farthest (octile cells) a unit looks along its abstract route for a
/// visible portal point or the stage target.
pub const ROUTE_LOOKAHEAD_CELLS: i32 = 192;
/// Arc length (cells) of curve covered by one shared cohort of a
/// `CurveFormation` command.
pub const FORMATION_SEGMENT_CELLS: i32 = 32;
/// Ticks between waypoint refinements of one unit (staggered by unit id).
pub const WAYPOINT_REFRESH_TICKS: u64 = 10;
pub const CONTEXT_CACHE_MAX: usize = 256;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct NavConfig {
    pub seed: u64,
    pub map_cells: i32,
    pub cell_size: f32,
    pub sector_cells: i32,
    pub unit_count: u32,
    /// Threads of the per-tick pool (steering, avoidance, physics) and of
    /// hierarchy builds and repairs.
    pub worker_count: usize,
    /// Navigation job threads (flow fields, terrain repair jobs). Field jobs
    /// are paced for about half of one thread, so they were under 1.3% busy
    /// with 8; the second covers bursts. Results do not depend on it.
    pub nav_worker_count: usize,
    pub radius_scale: f32,
    pub speed_scale: f32,
    pub field_cache_max: usize,
    pub spawn_center_cell: (i32, i32),
    pub spawn_radius_cells: i32,
    pub collision_restitution: f32,
    pub collision_iterations: u32,
    /// Distance (cells) at which a unit with line of sight to its slot leaves
    /// the shared field and steers to its individual slot.
    pub arrival_switch_cells: i32,
    /// Minimum spacing (cells) between accepted curve checkpoints.
    pub guide_stride_cells: i32,
    pub max_skipped_guides: i32,
    pub corridor_tolerance_world: f32,
    /// Blocked-unit policy: ticks without progress before a unit inside its
    /// arrival region settles, and before any unit gives up.
    pub stuck_settle_ticks: u32,
    pub stuck_give_up_ticks: u32,
    /// Explicit traffic rule (separate from collision response): a unit
    /// without progress for `yield_after_ticks` alternates deterministically
    /// between pushing and briefly backing off, phase-shifted by a hash of its
    /// ID, which breaks symmetric jams at chokepoints. 0 disables it. Off by
    /// default: with `avoidance` on it is no longer needed and slows arrivals
    /// (a unit slowed by avoidance near its slot started backing off).
    pub yield_after_ticks: u32,
    /// Reciprocal velocity-obstacle avoidance (ORCA) between units.
    pub avoidance: bool,
    /// ORCA look-ahead in seconds.
    pub avoidance_time_horizon: f32,
    /// Nearest neighbours each unit avoids.
    pub avoidance_neighbors: usize,
}

impl Default for NavConfig {
    fn default() -> Self {
        Self {
            seed: 0x5EED_CAFE_F00D_1234,
            map_cells: 2560,
            cell_size: 8.0,
            sector_cells: 64,
            unit_count: 1000,
            worker_count: 8,
            nav_worker_count: 2,
            radius_scale: 1.0,
            speed_scale: 1.0,
            field_cache_max: 2048,
            spawn_center_cell: (260, 2290),
            spawn_radius_cells: 96,
            collision_restitution: 0.02,
            collision_iterations: 1,
            arrival_switch_cells: 18,
            guide_stride_cells: 24,
            max_skipped_guides: 4,
            corridor_tolerance_world: 48.0,
            stuck_settle_ticks: 90,
            stuck_give_up_ticks: 1800,
            avoidance: true, avoidance_time_horizon: 1.0, avoidance_neighbors: 10,
            yield_after_ticks: 0,
        }
    }
}

impl NavConfig {
    pub fn terrain_config(&self) -> TerrainConfig {
        TerrainConfig::new(self.seed, self.map_cells, self.cell_size, self.sector_cells)
    }
}

/// A unit to add: where it starts (at rest), its body radius, top speed and
/// movement profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitSpec {
    pub pos: WorldPos,
    pub radius: f32,
    pub max_speed: f32,
    pub profile_id: u8,
}

/// A terrain version that became authoritative: the tick it did, and the
/// inclusive cell rectangle `(x0, y0, x1, y1)` covering every edited cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerrainChange {
    pub tick: u64,
    pub version: u32,
    pub cells: (i32, i32, i32, i32),
}

/// Unit bodies as columns, one row per unit in [`NavService::entities`] order.
pub struct BodyColumns<'a> {
    pub pos_x: &'a mut [f32],
    pub pos_y: &'a mut [f32],
    pub vel_x: &'a mut [f32],
    pub vel_y: &'a mut [f32],
    pub radius: &'a [f32],
}

impl BodyColumns<'_> {
    fn check(&self, n: usize) {
        assert!(
            self.pos_x.len() == n && self.pos_y.len() == n && self.vel_x.len() == n && self.vel_y.len() == n && self.radius.len() == n,
            "body columns must have one row per nav unit ({n})"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum UnitMode {
    Idle,
    WaitingForNav,
    Moving,
    Arrived,
    Blocked,
    Unreachable,
}

impl UnitMode {
    pub fn label(self) -> &'static str {
        match self {
            UnitMode::Idle => "idle",
            UnitMode::WaitingForNav => "waiting",
            UnitMode::Moving => "moving",
            UnitMode::Arrived => "arrived",
            UnitMode::Blocked => "blocked",
            UnitMode::Unreachable => "unreachable",
        }
    }
}

/// Authoritative unit state. Per-unit navigation state is deliberately
/// small: a command/cohort handle, a stage index and the arrival slot.
#[derive(Clone, Debug)]
pub struct Unit {
    /// Local handle: stable for the unit's lifetime, never reused.
    pub id: u32,
    /// Engine entity id the unit belongs to.
    pub entity: u64,
    pub body: Body,
    pub max_speed: f32,
    pub accel: f32,
    pub profile_id: u8,
    pub command_id: Option<u64>,
    pub cohort_id: Option<u32>,
    pub slot: Option<WorldPos>,
    pub slot_cell: Option<CellPos>,
    pub stage_idx: u32,
    pub arrival_phase: bool,
    pub arrived: bool,
    /// Arrived by the settle policy (inside the arrival region, not at the
    /// exact slot). Always implies `arrived`.
    pub settled: bool,
    pub unreachable: bool,
    pub mode: UnitMode,
    pub desired_speed: f32,
    /// Ticks since the progress metric last improved.
    pub stuck_ticks: u32,
    best_progress: (u32, u64),
    /// Closest approach to the slot (world units x 100) in the arrival
    /// phase; tracked apart from `best_progress`, whose field costs are on a
    /// different scale.
    best_slot_dist: u64,
    last_region: Option<(u32, u16)>,
    /// Steering waypoint: (cell, stage it belongs to).
    waypoint: Option<(CellPos, u32)>,
    /// Where a stationary unit without a slot (idle, unreachable) stands:
    /// where it last came to rest.
    home: Option<WorldPos>,
    pub last_pos: WorldPos,
    pub arrived_tick: Option<u64>,
    /// The progress metric's scale changed (crowd costs re-keyed the
    /// context): the next measurement only re-bases `best_progress` and
    /// does not count as progress, so a stuck unit still gives up.
    rebase_progress: bool,
    /// Times the give-up timer expired while the goal was still reachable
    /// (see `update_unit_progress`).
    stuck_retries: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnitSnapshot {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub radius: f32,
    pub profile_id: u8,
    pub command_id: Option<u64>,
    pub cohort_id: Option<u32>,
    pub stage_idx: u32,
    pub arrival_phase: bool,
    pub slot_x: Option<f32>,
    pub slot_y: Option<f32>,
    pub stuck_ticks: u32,
    pub field_sector: Option<SectorPos>,
    pub field_cost: Option<u64>,
    pub mode: UnitMode,
    pub arrived: bool,
    pub unreachable: bool,
    pub desired_speed: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlotsSnapshot {
    pub unit_id: u32,
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommandSnapshot {
    pub id: u64,
    /// "point", "route" (follow-curve) or "formation" (curve formation).
    pub kind: String,
    pub commit_tick: u64,
    pub status: CommandStatus,
    pub requested_x: f32,
    pub requested_y: f32,
    pub curve_points: Vec<(f32, f32)>,
    /// Slots of units still under this command.
    pub slots: Vec<SlotsSnapshot>,
    /// Units still under this command (0 once all were re-commanded).
    pub assigned_units: u32,
    pub total_units: u32,
    /// Includes `settled_units`.
    pub arrived_units: u32,
    pub settled_units: u32,
    pub unreachable_units: u32,
    pub active_units: u32,
    pub replaced_units: u32,
    pub first_move_latency_ticks: Option<u64>,
    pub first_move_latency_ms: Option<f64>,
    pub route_polylines: Vec<Vec<WorldPos>>,
    pub truncated: bool,
    pub skipped_guides: u32,
    pub corridor_tolerance_world: f32,
    pub repair_rerouted_units: u32,
    pub repair_invalidated_units: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CohortSnapshot {
    pub id: u32,
    pub command_id: u64,
    pub profile_id: u8,
    pub status: &'static str,
    pub active_units: u32,
    pub stage_count: u32,
    pub field_slots: u32,
    pub fields_ready: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingJobSnapshot {
    pub id: u64,
    pub kind: &'static str,
    pub due_tick: u64,
    pub stale: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LatencySummary {
    pub samples: u32,
    pub median_ticks: f64,
    pub p95_ticks: f64,
    pub worst_ticks: u64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub worst_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NavRuntimeStats {
    pub worker_count: usize,
    pub jobs_submitted: u64,
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub jobs_stale: u64,
    pub plans_completed: u64,
    pub fields_built: u64,
    pub field_expansions: u64,
    pub abstract_expansions: u64,
    /// Summed worker CPU time (wall clock, instrumentation only).
    pub nav_busy_ms_total: f64,
    /// busy / (elapsed wall time x workers) since the simulation started.
    pub worker_utilization: f64,
    pub last_plan_ms: f64,
    pub last_field_ms: f64,
    pub pending_jobs: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_fields: usize,
    pub cache_bytes: usize,
    pub cache_evictions: u64,
    pub context_hits: u64,
    pub context_misses: u64,
    pub contexts_cached: usize,
    pub hierarchy_bytes: usize,
    pub command_latency: LatencySummary,
    pub active_cohorts: usize,
    pub unique_destination_groups: usize,
    pub active_profiles: usize,
    pub terrain_rebuilds: u64,
    pub terrain_rerouted_units: u64,
    pub terrain_invalidated_units: u64,
    pub terrain_relocated_units: u64,
    pub last_repair_latency_ticks: u64,
    pub last_repair_ms: f64,
    pub last_repair_sectors: usize,
    pub local_recenters: u64,
    pub skipped_stages: u64,
    /// Units whose current region has no route to the goal (reported unreachable).
    pub stranded_units: u64,
    pub stuck_settled: u64,
    pub stuck_gave_up: u64,
    /// Give-up timers that expired and restarted (`STUCK_RETRIES`).
    #[serde(default)]
    pub stuck_retries: u64,
    /// Unit-ticks spent backing off under the yielding rule.
    pub yield_ticks: u64,
    pub field_requests_deferred: u64,
    /// Standing-crowd map version (`nav::crowd`) and how many times
    /// crowd-aware stages were re-keyed to a new one.
    #[serde(default)]
    pub crowd_version: u32,
    /// Arrival slots handed out by the last-resort pass without line of
    /// sight to the goal (not in pathsim; see `assign_command_slots`).
    #[serde(default)]
    pub fallback_slots: u64,
    #[serde(default)]
    pub crowd_rekeys: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SimStats {
    pub tick: u64,
    pub units: usize,
    pub moving: usize,
    pub waiting: usize,
    /// Includes `settled`.
    pub arrived: usize,
    pub settled: usize,
    pub blocked: usize,
    pub unreachable: usize,
    pub idle: usize,
    pub collision: CollisionStats,
    pub nav: NavRuntimeStats,
    pub tick_ms: f64,
    pub terrain_version: u32,
    pub last_terrain_edit_tick: Option<u64>,
    pub awaiting_hierarchy: bool,
    pub nav_refresh_ms: f64,
    /// Wall time of the last tick's pipeline phases (see `advance_tick`).
    #[serde(default)]
    pub phases: TickPhases,
}

/// Wall time per `advance_tick` phase, in milliseconds. Measurement only;
/// never part of the authoritative state.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TickPhases {
    /// Queued terrain edits and applying due worker results, including the
    /// wait at the deterministic job barrier.
    pub jobs_ms: f64,
    /// Interpreting queued commands.
    pub commands_ms: f64,
    /// Requesting destination contexts for stages.
    pub contexts_ms: f64,
    /// Steering (field lookups, look-ahead, slots), excluding avoidance.
    pub steer_ms: f64,
    pub avoidance_ms: f64,
    /// Integration and collision response.
    pub physics_ms: f64,
    /// Progress, settle and give-up policy.
    pub progress_ms: f64,
    /// Command status updates.
    pub statuses_ms: f64,
    /// Submitting worker jobs.
    pub submit_ms: f64,
    /// Building the published snapshot.
    pub snapshot_ms: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateHashParts {
    pub combined: u64,
    /// Unit positions and velocities.
    pub bodies: u64,
    /// Per-unit navigation state (command, stage, flags, slot).
    pub unit_nav: u64,
    pub commands: u64,
    pub terrain: u64,
    pub nav_version: u32,
}

#[derive(Clone)]
pub struct SimSnapshot {
    pub tick: u64,
    pub units: Vec<UnitSnapshot>,
    pub commands: Vec<CommandSnapshot>,
    pub cohorts: Vec<CohortSnapshot>,
    pub stats: SimStats,
    pub hierarchy: Arc<NavHierarchy>,
    pub terrain: SharedTerrain,
    pub debug_fields: Vec<Arc<FlowField>>,
    pub contacts: Vec<ContactView>,
    pub pending_jobs: Vec<PendingJobSnapshot>,
    /// Terrain edited but not yet reflected in navigation (world rect).
    pub pending_dirty: Option<Rect>,
    pub nav_stats: NavStats,
    pub state_hash: u64,
    pub published_at: Instant,
}

/// Field resolution for one (sector, region) of one stage.
#[derive(Clone)]
struct FieldSlot {
    key: FieldKey,
    offset: u64,
    next_node: Option<u32>,
    field: Option<Arc<FlowField>>,
}

#[derive(Clone)]
struct Stage {
    target: CellPos,
    tolerance_cells: i32,
    /// Context wanted at the current navigation version.
    ctx_key: ContextKey,
    /// Context in use. It may belong to an older navigation version during a
    /// terrain repair; then only fields of unchanged sectors are used.
    ctx: Option<Arc<DestContext>>,
    /// `None` = no route from that region to this stage's target.
    slots: BTreeMap<(u32, u16), Option<FieldSlot>>,
}

impl Stage {
    fn fresh(&self) -> bool { self.ctx.as_ref().map(|c| c.key == self.ctx_key).unwrap_or(false) }
}

struct Cohort {
    id: u32,
    command_id: u64,
    profile_id: u8,
    stages: Vec<Stage>,
    active_units: u32,
    /// Its command has at least `CROWD_AWARE_MIN_UNITS` units: its contexts
    /// and look-ahead avoid dense standing crowds.
    crowd_aware: bool,
    /// Tick its stages last adopted a crowd map (`CROWD_REKEYS_PER_UPDATE`).
    crowd_keyed_tick: u64,
}

/// What a batch of `attach_context` calls changed, for `contexts_attached`.
#[derive(Default)]
struct Attached {
    /// Commands whose stages switched to a new navigation version or goal.
    touched: BTreeSet<u64>,
    /// Commands with stages re-keyed to a new crowd map only.
    crowd_touched: BTreeSet<u64>,
    /// Cohorts with such stages.
    rekeyed: BTreeSet<u32>,
    /// (cohort, stage) of those stages: their field slots are re-checked.
    prune: Vec<(u32, usize)>,
}

#[derive(Clone)]
enum PendingKind {
    Field(FieldKey),
    Repair,
}

struct PendingJob {
    due_tick: u64,
    kind: PendingKind,
    stale: bool,
    label: &'static str,
}

struct QueuedCommand {
    payload: CommandPayload,
    submitted_at: Option<Instant>,
}

/// Engine core. It is single-threaded; independent navigation jobs run on the
/// compute pool and are only published at fixed tick boundaries.
pub struct NavService {
    pub cfg: NavConfig,
    pub profiles: Vec<MovementProfile>,
    pub terrain: SharedTerrain,
    pub hierarchy: Arc<NavHierarchy>,
    pub units: Vec<Unit>,
    /// Count behind `NavRuntimeStats::fallback_slots` (slot assignment
    /// only borrows the service).
    fallback_slots: std::cell::Cell<u64>,
    /// Terrain versions installed since the last `take_terrain_changes`.
    terrain_changes: Vec<TerrainChange>,
    /// Local id -> index in `units` (`u32::MAX` once removed).
    index_of: Vec<u32>,
    next_unit_id: u32,
    entity_index: BTreeMap<u64, u32>,
    cohorts: BTreeMap<u32, Cohort>,
    commands: BTreeMap<u64, CommandRecord>,
    command_order: VecDeque<u64>,
    next_cohort_id: u32,
    pool: ComputePool,
    /// Replaced hierarchies (with their terrain), freed on the dropper
    /// thread once nothing else holds them: freeing one on the simulation
    /// thread took 4–6 ms (tens of MB in many allocations).
    retired_hierarchies: Vec<Arc<NavHierarchy>>,
    dropper: std::sync::mpsc::Sender<Arc<NavHierarchy>>,
    pending: BTreeMap<u64, PendingJob>,
    pending_fields: BTreeMap<FieldKey, u64>,
    field_cache: FieldCache,
    contexts: BTreeMap<ContextKey, (Arc<DestContext>, u64)>,
    /// (priority, key): 0 = needed now, 1 = prefetch.
    /// (priority, key): see `field_priority`.
    field_needs: BTreeSet<(u64, FieldKey)>,
    /// Sim time (µs) at which the field jobs submitted so far are assumed
    /// done (`FIELD_PACE_CELLS_PER_MS`); new jobs are due after it.
    field_clock_us: u64,
    contacts: Vec<ContactView>,
    tick: u64,
    queued_commands: VecDeque<QueuedCommand>,
    queued_terrain: VecDeque<TerrainPatch>,
    command_received_at: BTreeMap<u64, Instant>,
    collision_cfg: CollisionConfig,
    collider: Collider,
    /// Thread pool for per-tick data-parallel work (avoidance, collision).
    par: Par,
    collision_stats: CollisionStats,
    last_avoidance_ms: f64,
    nav: NavRuntimeStats,
    latency_ticks: Vec<u64>,
    latency_ms: Vec<f64>,
    last_terrain_edit_tick: Option<u64>,
    /// Union of edited cells not yet reflected in `hierarchy`.
    pending_rect: Option<(i32, i32, i32, i32)>,
    last_snapshot: Option<Arc<SimSnapshot>>,
    /// Whether `advance_tick` publishes a snapshot (see `set_publishing`).
    publishing: bool,
    /// Standing-crowd density and route costs (`nav::crowd`).
    crowd: Arc<CrowdMap>,
    /// For crowd-aware commands still arriving: the crowd without their own
    /// arrived units (see `update_crowd`).
    own_crowd: BTreeMap<u64, Arc<CrowdMap>>,
    /// How fast moving units advance, per crowd block (`nav::crowd`).
    traffic: Traffic,
    /// Last version number given to a crowd map.
    crowd_versions: u32,
    /// Bodies the last physics step left exactly as it found them, with the
    /// terrain version and that step's stats: stepping them again on the
    /// same terrain gives the same result, so a crowd at rest skips it.
    physics_rest: Option<(Vec<Body>, u32, CollisionStats)>,
    rng: Rng,
    started: Instant,
    last_tick_ms: f64,
    phases: TickPhases,
    /// (terrain version, full terrain digest), recomputed once per edit.
    terrain_digest: (u32, u64),
    /// Digest of each terrain band (see `Terrain::digest`).
    terrain_bands: Vec<u64>,
    /// Committed edits not yet in a repair job.
    unrepaired_patches: Vec<TerrainPatch>,
    /// Cells covered by the repair job in flight.
    repair_rect: Option<(i32, i32, i32, i32)>,
}

impl NavService {
    /// A service on the generated terrain with pathsim's default profiles.
    pub fn new(cfg: NavConfig) -> Self {
        let terrain = Terrain::generate(&cfg.terrain_config());
        let profiles = profiles_for(cfg.cell_size, cfg.radius_scale);
        Self::with_terrain(cfg, terrain, profiles)
    }

    /// Build from an explicit terrain (tests and hand-built benchmark maps).
    /// A service on an explicit terrain with explicit movement profiles.
    /// Profile ids must equal their index. No units: add them with
    /// [`Self::add_unit`] (or [`Self::spawn_plan`] for a pathsim-style lattice).
    pub fn with_terrain(cfg: NavConfig, terrain: Terrain, profiles: Vec<MovementProfile>) -> Self {
        assert!(profiles.iter().enumerate().all(|(i, p)| p.id as usize == i), "profile ids must equal their index");
        let terrain = Arc::new(terrain);
        // Parallel build; the result is identical for any thread count.
        let hierarchy = Arc::new(NavHierarchy::build_parallel(Arc::clone(&terrain), &profiles, cfg.worker_count));
        let pool = ComputePool::new(cfg.nav_worker_count);
        let (dropper, retired) = std::sync::mpsc::channel::<Arc<NavHierarchy>>();
        std::thread::Builder::new().name("nav-dropper".into())
            .spawn(move || { for h in retired { drop(h); } })
            .expect("spawn dropper thread");
        let mut sim = Self {
            cfg: cfg.clone(), profiles, terrain, hierarchy, units: Vec::new(), fallback_slots: std::cell::Cell::new(0), terrain_changes: Vec::new(), index_of: vec![u32::MAX], next_unit_id: 1, entity_index: BTreeMap::new(),
            cohorts: BTreeMap::new(), commands: BTreeMap::new(), command_order: VecDeque::new(),
            next_cohort_id: 1, pool, retired_hierarchies: Vec::new(), dropper, pending: BTreeMap::new(), pending_fields: BTreeMap::new(),
            field_cache: FieldCache::new(cfg.field_cache_max), contexts: BTreeMap::new(),
            field_needs: BTreeSet::new(), field_clock_us: 0,
            contacts: Vec::new(), tick: 0,
            queued_commands: VecDeque::new(), queued_terrain: VecDeque::new(),
            command_received_at: BTreeMap::new(),
            collider: Collider::new(),
            par: Par::new(cfg.worker_count),
            collision_cfg: CollisionConfig {
                solver_iterations: cfg.collision_iterations,
                restitution: cfg.collision_restitution,
                correction: 0.18,
            },
            collision_stats: CollisionStats::default(),
            last_avoidance_ms: 0.0,
            nav: NavRuntimeStats { worker_count: cfg.nav_worker_count, ..Default::default() },
            latency_ticks: Vec::new(), latency_ms: Vec::new(),
            last_terrain_edit_tick: None, pending_rect: None,
            last_snapshot: None, publishing: false, crowd: Arc::new(CrowdMap::default()), own_crowd: BTreeMap::new(), traffic: Traffic::default(), crowd_versions: 0, physics_rest: None, rng: Rng::new(cfg.seed ^ 0xABC), started: Instant::now(),
            last_tick_ms: 0.0, phases: TickPhases::default(), terrain_digest: (0, 0), terrain_bands: Vec::new(), unrepaired_patches: Vec::new(), repair_rect: None,
        };
        sim.crowd = Arc::new(CrowdMap::empty(&sim.terrain.geom));
        sim.traffic = Traffic::new(&sim.crowd);
        sim
    }

    pub fn tick(&self) -> u64 { self.tick }
    /// Standing-crowd density used by crowd-aware routing.
    pub fn crowd_map(&self) -> &CrowdMap { &self.crowd }

    /// The crowd a command's units route around: without its own arrived
    /// units while it is still arriving.
    fn crowd_for(&self, command_id: u64) -> &Arc<CrowdMap> { self.own_crowd.get(&command_id).unwrap_or(&self.crowd) }

    fn crowd_by_version(&self, version: u32) -> Option<&Arc<CrowdMap>> {
        std::iter::once(&self.crowd).chain(self.own_crowd.values()).find(|m| m.version == version)
    }

    /// Wall time of the last `advance_tick`, excluding the snapshot.
    pub fn last_tick_ms(&self) -> f64 { self.last_tick_ms }

    /// Whether `advance_tick` publishes a snapshot (on by default). The app
    /// turns it off while the renderer has not taken the previous one:
    /// snapshots never feed back into the simulation, only cost time.
    pub fn set_publishing(&mut self, on: bool) { self.publishing = on; }

    /// pathsim's deterministic spawn lattice around `cfg.spawn_center_cell`
    /// for `cfg.unit_count` units, as specs for [`Self::add_unit`] (in order).
    /// Uses the service's seeded RNG, so call it once, before adding units.
    pub fn spawn_plan(&mut self) -> Vec<UnitSpec> {
        let geom = self.hierarchy.terrain.geom.clone();
        let center = self.cfg.spawn_center_cell;
        let spawn_center = geom.cell_center(CellPos::new(center.0, center.1));
        let max_radius_world = self.profiles.iter().map(|p| p.unit_radius).fold(0.0f32, f32::max) * RADIUS_JITTER_MAX;
        // A deterministic lattice guarantees physically disjoint initial
        // bodies.
        let spacing = (max_radius_world * 2.0 * 1.15 + self.cfg.cell_size * 0.3).max(self.cfg.cell_size * 1.1);
        let max_spawn_radius = (self.cfg.spawn_radius_cells.max(1) as f32) * self.cfg.cell_size;
        let max_ring = ((max_spawn_radius / spacing).ceil() as i32) + ((self.cfg.unit_count as f32).sqrt() * 0.9).ceil() as i32 + 4;
        let mut units = Vec::with_capacity(self.cfg.unit_count as usize);
        'rings: for ring in 0..=max_ring {
            let mut ring_candidates: Vec<(i32, i32)> = Vec::new();
            if ring == 0 {
                ring_candidates.push((0, 0));
            } else {
                for d in -ring..=ring { ring_candidates.push((d, -ring)); ring_candidates.push((d, ring)); }
                for d in -ring + 1..ring { ring_candidates.push((-ring, d)); ring_candidates.push((ring, d)); }
            }
            ring_candidates.sort_by_key(|(x, y)| (x.abs() + y.abs(), *y, *x));
            for (gx, gy) in ring_candidates {
                if units.len() >= self.cfg.unit_count as usize { break 'rings; }
                let wx = spawn_center.x + gx as f32 * spacing;
                let wy = spawn_center.y + gy as f32 * spacing;
                let cell = geom.world_to_cell(WorldPos::new(wx, wy));
                if !geom.in_bounds(cell) { continue; }
                let profile_id: u8 = if self.rng.chance(0.82) { 0 } else { 1 };
                if !self.hierarchy.grid(profile_id).is_passable(&self.terrain, cell) { continue; }
                let profile = &self.profiles[profile_id as usize];
                // Profile radii already include `radius_scale`.
                let radius = profile.unit_radius * self.rng.range(2.0 - RADIUS_JITTER_MAX, RADIUS_JITTER_MAX);
                let speed = profile.base_speed * self.rng.range(0.9, 1.1) * self.cfg.speed_scale;
                let jx = self.rng.range(-0.1, 0.1) * self.cfg.cell_size;
                let jy = self.rng.range(-0.1, 0.1) * self.cfg.cell_size;
                let pos = geom.cell_center(cell).add(WorldPos::new(jx, jy));
                units.push(UnitSpec { pos, radius, max_speed: speed, profile_id });
            }
        }
        // Fallback for tiny or mostly blocked maps: deterministic open cells
        // nearest the spawn centre; collision response separates overlaps.
        if units.len() < self.cfg.unit_count as usize {
            let ci = geom.clamp_cell(CellPos::new(center.0, center.1));
            let mut used: BTreeSet<(i32, i32)> = units.iter()
                .map(|u| { let c = geom.world_to_cell(u.pos); (c.x, c.y) })
                .collect();
            let mut open = Vec::new();
            for y in 0..geom.cells_y {
                for x in 0..geom.cells_x {
                    let c = CellPos::new(x, y);
                    if self.hierarchy.grid(0).is_passable(&self.terrain, c) { open.push(c); }
                }
            }
            open.sort_by_key(|c| { let dx = (c.x - ci.x) as i64; let dy = (c.y - ci.y) as i64; (dx * dx + dy * dy, c.y, c.x) });
            for c in open {
                if units.len() >= self.cfg.unit_count as usize { break; }
                // Never stack fallback units on a lattice cell: two bodies on
                // one cell cause a large tick-one collision impulse.
                if !used.insert((c.x, c.y)) { continue; }
                let profile = &self.profiles[0];
                units.push(UnitSpec { pos: geom.cell_center(c), radius: profile.unit_radius, max_speed: profile.base_speed * self.cfg.speed_scale, profile_id: 0 });
            }
        }
        units
    }

    /// Add a unit for `entity`, at rest at `spec.pos`, after every existing
    /// unit. Returns its local id. Call between ticks only.
    pub fn add_unit(&mut self, entity: u64, spec: UnitSpec) -> u32 {
        assert!((spec.profile_id as usize) < self.profiles.len(), "unknown movement profile {}", spec.profile_id);
        assert!(!self.entity_index.contains_key(&entity), "entity {entity} already has a nav unit");
        let id = self.next_unit_id;
        self.next_unit_id += 1;
        let mut u = Self::new_unit(id, spec.pos, spec.radius, spec.max_speed, spec.profile_id);
        u.entity = entity;
        self.index_of.push(self.units.len() as u32);
        self.entity_index.insert(entity, id);
        self.units.push(u);
        self.physics_rest = None;
        id
    }

    /// Remove `entity`'s unit; the others keep their order. Call between
    /// ticks only. Returns false when it has none.
    pub fn remove_unit(&mut self, entity: u64) -> bool {
        self.remove_units(&[entity]) > 0
    }

    /// Remove several units at once (one re-index). Returns how many existed.
    pub fn remove_units(&mut self, entities: &[u64]) -> usize {
        let mut gone = BTreeSet::new();
        for e in entities {
            if let Some(id) = self.entity_index.remove(e) { gone.insert(id); }
        }
        if gone.is_empty() { return 0; }
        self.units.retain(|u| !gone.contains(&u.id));
        for &id in &gone { self.index_of[id as usize] = u32::MAX; }
        for (i, u) in self.units.iter().enumerate() { self.index_of[u.id as usize] = i as u32; }
        self.physics_rest = None;
        gone.len()
    }

    /// Local id of `entity`'s unit (what [`CommandPayload::unit_ids`] holds).
    pub fn local_id(&self, entity: u64) -> Option<u32> { self.entity_index.get(&entity).copied() }

    /// Engine entity ids in service order: row `i` of every [`BodyColumns`]
    /// passed to [`Self::step`] belongs to `entities()[i]`.
    pub fn entities(&self) -> impl ExactSizeIterator<Item = u64> + '_ { self.units.iter().map(|u| u.entity) }

    pub fn len(&self) -> usize { self.units.len() }
    pub fn is_empty(&self) -> bool { self.units.is_empty() }

    /// One tick against caller-owned bodies (rows in [`Self::entities`]
    /// order): positions, velocities and radii are read first, positions and
    /// velocities written back last. A caller may move, stop or resize a
    /// unit between ticks; the service plans from whatever it finds.
    pub fn step(&mut self, bodies: &mut BodyColumns<'_>) {
        let n = self.units.len();
        bodies.check(n);
        for (i, u) in self.units.iter_mut().enumerate() {
            let pos = WorldPos::new(bodies.pos_x[i], bodies.pos_y[i]);
            if pos != u.body.pos {
                // Moved by the caller since the last tick (or new): a unit
                // without a slot now stands here instead of walking back.
                u.last_pos = pos;
                if u.home.is_some() { u.home = Some(pos); }
            }
            u.body = Body { pos, vel: WorldPos::new(bodies.vel_x[i], bodies.vel_y[i]), radius: bodies.radius[i] };
        }
        self.advance_tick();
        for (i, u) in self.units.iter().enumerate() {
            bodies.pos_x[i] = u.body.pos.x; bodies.pos_y[i] = u.body.pos.y;
            bodies.vel_x[i] = u.body.vel.x; bodies.vel_y[i] = u.body.vel.y;
        }
    }

    /// Terrain edits that became authoritative (visible to collision and
    /// navigation) since the last call, oldest first: what the renderer
    /// re-uploads and the copy layer republishes. Edits queued with
    /// [`Self::queue_terrain_patch`] land a fixed number of ticks later
    /// (`REPAIR_DELAY_MS`), together with their nav repair.
    pub fn take_terrain_changes(&mut self) -> Vec<TerrainChange> { std::mem::take(&mut self.terrain_changes) }

    /// Write every unit's [`UnitMode`] (as `u8`, rows in service order).
    pub fn write_modes(&self, out: &mut [u8]) {
        assert_eq!(out.len(), self.units.len());
        for (o, u) in out.iter_mut().zip(&self.units) { *o = u.mode as u8; }
    }

    fn new_unit(id: u32, pos: WorldPos, radius: f32, speed: f32, profile_id: u8) -> Unit {
        Unit {
            id, entity: 0, body: Body { pos, vel: WorldPos::ZERO, radius },
            max_speed: speed, accel: speed * 4.0, profile_id,
            command_id: None, cohort_id: None, slot: None, slot_cell: None,
            stage_idx: 0, arrival_phase: false, arrived: false, settled: false, unreachable: false,
            mode: UnitMode::Idle, desired_speed: 0.0, stuck_ticks: 0,
            best_progress: (0, u64::MAX), best_slot_dist: u64::MAX, last_region: None, waypoint: None, home: Some(pos), last_pos: pos, arrived_tick: None, rebase_progress: false, stuck_retries: 0,
        }
    }

    /// Unit IDs are 1-based and dense; storage index is `id - 1`.
    fn unit_index(&self, id: u32) -> Option<usize> {
        match self.index_of.get(id as usize) {
            Some(&i) if i != u32::MAX => Some(i as usize),
            _ => None,
        }
    }

    /// Queue a live command (wall-clock receipt time kept for latency stats).
    /// Inputs become visible at the next tick boundary and are recorded with
    /// that tick, so `ReplayLog` reproduces the run exactly.
    pub fn queue_command(&mut self, payload: CommandPayload) {
        self.queued_commands.push_back(QueuedCommand { payload, submitted_at: Some(Instant::now()) });
    }
    /// Queue a scripted/replayed command (no wall-clock latency sample).
    pub fn queue_command_at(&mut self, payload: CommandPayload) {
        self.queued_commands.push_back(QueuedCommand { payload, submitted_at: None });
    }
    pub fn queue_terrain_patch(&mut self, patch: TerrainPatch) {
        self.queued_terrain.push_back(patch);
    }
    fn command_goal(&self, profile_id: u8, requested: WorldPos) -> Option<CellPos> {
        let geom = &self.hierarchy.terrain.geom;
        let grid = self.hierarchy.grid(profile_id);
        let cc = geom.clamp_cell(geom.world_to_cell(requested));
        if grid.is_passable(&self.hierarchy.terrain, cc) { return Some(cc); }
        self.hierarchy.terrain.nearest_open_cell(cc, 48, |p| grid.is_passable(&self.hierarchy.terrain, p))
    }

    /// The clearance-passable cell a unit navigates from: its own cell, or
    /// the nearest passable cell within 3 rings (collision can push a body
    /// slightly off the clearance grid).
    fn nav_cell_of(h: &NavHierarchy, profile_id: u8, pos: WorldPos) -> Option<CellPos> {
        let geom = &h.terrain.geom;
        let grid = h.grid(profile_id);
        let c = geom.world_to_cell(pos);
        if grid.is_passable(&h.terrain, c) { return Some(c); }
        let mut best: Option<(f32, CellPos)> = None;
        for r in 1..=3i32 {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs() != r && dy.abs() != r { continue; }
                    let p = CellPos::new(c.x + dx, c.y + dy);
                    if !grid.is_passable(&h.terrain, p) { continue; }
                    let d = geom.cell_center(p).dist(pos);
                    let better = match best {
                        None => true,
                        Some((bd, bp)) => d.total_cmp(&bd).then((p.y, p.x).cmp(&(bp.y, bp.x))).is_lt(),
                    };
                    if better { best = Some((d, p)); }
                }
            }
            if best.is_some() { break; }
        }
        best.map(|b| b.1)
    }

    /// Deterministic ring search for physical arrival slots. All slots of one
    /// command share a single lattice anchored at the requested point, and
    /// `taken` is shared across profiles and cohorts, so no two units of a
    /// command are ever assigned the same physical slot. Each slot must be
    /// clearance-passable for the unit's profile, in the goal's component and
    /// have clearance-aware line of sight to that profile's snapped goal.
    /// With `avoid_crowd`, cells in blocks with a standing crowd are skipped.
    #[allow(clippy::too_many_arguments)]
    /// `need_sight`: a slot must see the goal in a straight passable line
    /// (pathsim's rule; off only for the last-resort pass, see
    /// `assign_command_slots`).
    #[allow(clippy::too_many_arguments)]
    fn generate_slots(&self, profile_id: u8, anchor: CellPos, goal: CellPos, unit_ids: &[u32], spacing: i32, taken: &mut BTreeSet<(i32, i32)>, reserved: &BTreeSet<CellPos>, avoid_crowd: bool, need_sight: bool) -> Vec<ArrivalSlot> {
        let mut out = Vec::with_capacity(unit_ids.len());
        if unit_ids.is_empty() { return out; }
        let terrain = &self.hierarchy.terrain;
        let geom = &terrain.geom;
        let grid = self.hierarchy.grid(profile_id);
        let spacing = spacing.max(1);
        let goal_comp = self.hierarchy.component_at(profile_id, goal);
        let need = (unit_ids.len() + taken.len()) as f32;
        let max_ring = ((need.sqrt() * 1.35) as i32 + 32).min(420);
        for ring in 0..=max_ring {
            let mut ring_candidates: Vec<(i32, i32)> = Vec::new();
            for gy in -ring..=ring {
                for gx in -ring..=ring {
                    if gx.abs() != ring && gy.abs() != ring { continue; }
                    ring_candidates.push((gx, gy));
                }
            }
            ring_candidates.sort_by_key(|&(gx, gy)| (gx * gx + gy * gy, gy, gx));
            for (gx, gy) in ring_candidates {
                if out.len() >= unit_ids.len() { break; }
                if taken.contains(&(gx, gy)) { continue; }
                let c = CellPos::new(anchor.x + gx * spacing, anchor.y + gy * spacing);
                if reserved.contains(&c) { continue; }
                if !geom.in_bounds(c) || !grid.is_passable(terrain, c) { continue; }
                if avoid_crowd && self.crowd.raw_level_at(c) > 0 { continue; }
                if self.hierarchy.component_at(profile_id, c) != goal_comp { continue; }
                if need_sight && !line_clear(&self.hierarchy, profile_id, goal, c) { continue; }
                taken.insert((gx, gy));
                let idx = out.len();
                out.push(ArrivalSlot { unit_id: unit_ids[idx], cell: c, position: geom.cell_center(c) });
            }
            if out.len() >= unit_ids.len() { break; }
        }
        out
    }

    /// Assign slots to `(profile, goal, unit ids)` groups of one command.
    /// Larger-clearance profiles pick first (fewer valid cells). Callers sort
    /// a group's units nearest-first, so when cells run out the farthest go
    /// without; the cells found are then matched to the units by position
    /// (`match_slots_in_place`).
    /// `reserved` holds slot cells that must stay untouched (arrived units
    /// during a terrain repair).
    ///
    /// A command of at least `CROWD_AWARE_MIN_UNITS` sent next to a standing
    /// crowd (its point is not inside one) gathers next to it: slots skip
    /// the crowd's blocks, and only units left without a slot then take the
    /// skipped ones. Sent into a crowd, it gathers in it. (Slots inside the
    /// crowd had units plow into it to reach them.)
    fn assign_command_slots(&self, requested: WorldPos, groups: &[(u8, CellPos, Vec<u32>)], max_radius_world: f32, reserved: &BTreeSet<CellPos>) -> Vec<ArrivalSlot> {
        let geom = &self.hierarchy.terrain.geom;
        // Anchor the lattice at the requested cell when some group can stand
        // there, else at the first (largest-clearance) group's snapped goal,
        // so slots stay clustered around where units actually gather.
        let requested_cell = geom.clamp_cell(geom.world_to_cell(requested));
        let anchor = if groups.iter().any(|g| self.hierarchy.grid(g.0).is_passable(&self.hierarchy.terrain, requested_cell)) {
            requested_cell
        } else {
            groups.iter().max_by_key(|g| (self.hierarchy.grid(g.0).profile.clearance_cells, std::cmp::Reverse(g.0))).map(|g| g.1).unwrap_or(requested_cell)
        };
        let spacing = ((2.35 * max_radius_world / geom.cell_size).ceil() as i32).max(1);
        let mut order: Vec<usize> = (0..groups.len()).collect();
        order.sort_by_key(|&i| (std::cmp::Reverse(self.hierarchy.grid(groups[i].0).profile.clearance_cells), groups[i].0, i));
        let units: usize = groups.iter().map(|g| g.2.len()).sum();
        let avoid_crowd = units as u32 >= CROWD_AWARE_MIN_UNITS && self.crowd.raw_level_at(anchor) == 0;
        let mut taken = BTreeSet::new();
        let mut out = Vec::new();
        for i in order {
            let (pid, goal, ids) = &groups[i];
            let mut slots = self.generate_slots(*pid, anchor, *goal, ids, spacing, &mut taken, reserved, avoid_crowd, true);
            if slots.len() < ids.len() && avoid_crowd {
                slots.extend(self.generate_slots(*pid, anchor, *goal, &ids[slots.len()..], spacing, &mut taken, reserved, false, true));
            }
            // Last resort (not in pathsim): in a narrow winding channel (a
            // river for ships) few cells see the goal in a straight line, and
            // units without a slot gave up as unreachable. Any cell of the
            // goal's component will do; units still route there by field.
            if slots.len() < ids.len() {
                let extra = self.generate_slots(*pid, anchor, *goal, &ids[slots.len()..], spacing, &mut taken, reserved, false, false);
                self.fallback_slots.set(self.fallback_slots.get() + extra.len() as u64);
                slots.extend(extra);
            }
            self.match_slots_in_place(&mut slots);
            out.extend(slots);
        }
        out
    }

    /// Trade a group's slots by position: units and slots are halved alike,
    /// by the coordinate along which the slots spread most, down to pairs.
    /// The front of an army takes the far side of its gathering and the rear
    /// the near side, so the crowd arrives as a block and no two paths
    /// cross. Nearest-first (which picks the slot cells, inner rings first)
    /// ignored the approach direction within a ring: half of a 20,000-unit
    /// army got slots on the far side of the goal and walked through the
    /// part already standing; a dense blob formed there, half the army
    /// settled in it, and it took minutes to dissolve.
    fn match_slots_in_place(&self, slots: &mut [ArrivalSlot]) {
        if slots.len() < 2 { return; }
        let pos = |id: u32| self.units[self.unit_index(id).expect("slot of a live unit")].body.pos;
        let mut units: Vec<(WorldPos, u32)> = slots.iter().map(|s| (pos(s.unit_id), s.unit_id)).collect();
        let mut places: Vec<(WorldPos, CellPos)> = slots.iter().map(|s| (s.position, s.cell)).collect();
        // Only which side of the split each entry lands on matters, and the
        // orders are total, so selecting the median gives the same halves
        // as sorting (O(n) per level instead of O(n log n)). Halves are
        // independent: large ones run in parallel.
        fn halve(par: &Par, units: &mut [(WorldPos, u32)], places: &mut [(WorldPos, CellPos)]) {
            if units.len() < 2 { return; }
            let (lo, hi) = places.iter().fold((WorldPos::new(f32::MAX, f32::MAX), WorldPos::new(f32::MIN, f32::MIN)),
                |(lo, hi), p| (WorldPos::new(lo.x.min(p.0.x), lo.y.min(p.0.y)), WorldPos::new(hi.x.max(p.0.x), hi.y.max(p.0.y))));
            let key = |p: WorldPos| if hi.x - lo.x >= hi.y - lo.y { (p.x, p.y) } else { (p.y, p.x) };
            let h = units.len() / 2;
            units.select_nth_unstable_by(h, |a, b| { let (ka, kb) = (key(a.0), key(b.0)); ka.0.total_cmp(&kb.0).then(ka.1.total_cmp(&kb.1)).then(a.1.cmp(&b.1)) });
            places.select_nth_unstable_by(h, |a, b| { let (ka, kb) = (key(a.0), key(b.0)); ka.0.total_cmp(&kb.0).then(ka.1.total_cmp(&kb.1)).then((a.1.x, a.1.y).cmp(&(b.1.x, b.1.y))) });
            let (ul, ur) = units.split_at_mut(h);
            let (pl, pr) = places.split_at_mut(h);
            if h >= 1024 {
                par.join(|| halve(par, ul, pl), || halve(par, ur, pr));
            } else {
                halve(par, ul, pl);
                halve(par, ur, pr);
            }
        }
        halve(&self.par, &mut units, &mut places);
        for (s, (u, p)) in slots.iter_mut().zip(units.iter().zip(&places)) {
            *s = ArrivalSlot { unit_id: u.1, cell: p.1, position: p.0 };
        }
    }

    /// Unit IDs sorted nearest-first to a point (ties by ID).
    fn ids_by_distance(&self, idxs: &[usize], to: WorldPos) -> Vec<u32> {
        let mut v: Vec<(f32, u32)> = idxs.iter().map(|&i| (self.units[i].body.pos.dist(to), self.units[i].id)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        v.into_iter().map(|x| x.1).collect()
    }

    fn reset_unit_nav(u: &mut Unit) {
        u.cohort_id = None;
        u.slot = None;
        u.slot_cell = None;
        u.stage_idx = 0;
        u.arrival_phase = false;
        u.arrived = false;
        u.settled = false;
        u.unreachable = false;
        u.mode = UnitMode::WaitingForNav;
        u.stuck_ticks = 0;
        u.best_progress = (0, u64::MAX); u.best_slot_dist = u64::MAX;
        u.last_region = None;
        u.waypoint = None;
        u.home = None;
        u.arrived_tick = None;
        u.stuck_retries = 0;
    }

    /// Stopped (S) and not yet at rest: it has no home until it is.
    fn braking(u: &Unit) -> bool { u.command_id.is_none() && u.home.is_none() && !u.unreachable }

    fn mark_unreachable(u: &mut Unit) {
        u.unreachable = true;
        u.home = Some(u.body.pos);
        u.mode = UnitMode::Unreachable;
        u.desired_speed = 0.0;
    }

    /// Snap curve guides for one profile: nearest passable cell within 24
    /// cells, same component as the goal, at least `stride` from the previous
    /// accepted checkpoint and not right next to the start or goal. Rejected
    /// samples are counted as skipped. Mouse samples never become hard
    /// waypoints: at most 64 ordered checkpoints survive.
    fn plan_guides(&self, pid: u8, goal: CellPos, start: CellPos, raw: &[CellPos], tolerance: i32) -> (Vec<CellPos>, u32) {
        let terrain = &self.hierarchy.terrain;
        let grid = self.hierarchy.grid(pid);
        let goal_comp = self.hierarchy.component_at(pid, goal);
        let stride = self.cfg.guide_stride_cells.max(tolerance * 2).max(4);
        let far = |a: CellPos, b: CellPos, d: i32| (a.x - b.x).pow(2) + (a.y - b.y).pow(2) > d * d;
        let mut out: Vec<CellPos> = Vec::new();
        let mut skipped = 0u32;
        for &g in raw {
            let c = terrain.geom.clamp_cell(g);
            let snapped = if grid.is_passable(terrain, c) { Some(c) } else { terrain.nearest_open_cell(c, 24, |p| grid.is_passable(terrain, p)) };
            let Some(s) = snapped.filter(|s| self.hierarchy.component_at(pid, *s) == goal_comp) else { skipped += 1; continue; };
            let prev = out.last().copied().unwrap_or(start);
            if !far(s, prev, if out.is_empty() { stride / 2 } else { stride }) { continue; }
            if !far(s, goal, stride / 2) { continue; }
            out.push(s);
        }
        if out.len() > 64 {
            let step = out.len().div_ceil(64);
            out = out.into_iter().step_by(step).collect();
        }
        (out, skipped)
    }

    /// A crowd-aware cohort's stages want contexts with the current crowd
    /// costs (none while the map is empty).
    fn key_stages_to_crowd(&self, stages: &mut [Stage], crowd_aware: bool) {
        if crowd_aware { for st in stages { st.ctx_key.crowd = self.crowd.version; } }
    }

    fn make_stage(&self, pid: u8, target: CellPos, tolerance_cells: i32) -> Option<Stage> {
        let ctx_key = context_key(&self.hierarchy, pid, target)?;
        Some(Stage { target, tolerance_cells, ctx_key, ctx: None, slots: BTreeMap::new() })
    }

    /// Slots and cohorts of a `CurveFormation` command. The curve is
    /// resampled by arc length: one row of evenly spread slots when the curve
    /// is long enough for the group at slot spacing, otherwise columns at
    /// slot spacing with extra rows offset along the normal (alternating
    /// sides). Units are ordered by their projection on the curve's chord and
    /// split into columns in that order, so paths do not cross; within a
    /// column the unit nearest the curve takes the row on it. A slot must be
    /// passable for the unit's profile, in the goal's component and at least
    /// ~0.75 spacing from every other slot of the command; a blocked row
    /// position moves to the column's next row, then to the nearest
    /// acceptable cell.
    /// Units without a slot on the curve get one at the endpoint (counted in
    /// `skipped_guides`, so the command reports a truncated curve).
    ///
    /// Navigation is shared per curve segment: slots are grouped into
    /// `FORMATION_SEGMENT_CELLS` pieces of arc, one cohort per (profile,
    /// segment) with its stage target on the curve in that segment. A slot
    /// without line of sight to its segment target gets its own cohort.
    fn plan_formation(&mut self, command_id: u64, curve: &[WorldPos], specs: Vec<(u8, Vec<usize>, CellPos)>, max_radius: f32, tolerance: i32, rec: &mut CommandRecord) -> Vec<ArrivalSlot> {
        let h = Arc::clone(&self.hierarchy);
        let terrain = &h.terrain;
        let geom = &terrain.geom;
        let spacing_cells = ((2.35 * max_radius / geom.cell_size).ceil() as i32).max(1);
        let spacing = spacing_cells as f32 * geom.cell_size;
        let min_sep2 = {
            let m = (spacing_cells as f32 * 0.75).max(1.0);
            (m * m) as i32
        };
        let pts: Vec<WorldPos> = if curve.is_empty() { vec![rec.payload.requested_point] } else { curve.to_vec() };
        let mut arc = vec![0.0f32; pts.len()];
        for k in 1..pts.len() { arc[k] = arc[k - 1] + pts[k].dist(pts[k - 1]); }
        let total = *arc.last().unwrap();
        // Point and unit tangent at arc length `t`.
        let at = |t: f32| -> (WorldPos, WorldPos) {
            let k = arc.partition_point(|&a| a <= t).clamp(1, pts.len().max(2) - 1).min(pts.len() - 1);
            if pts.len() < 2 { return (pts[0], WorldPos::new(1.0, 0.0)); }
            let (a, b) = (pts[k - 1], pts[k]);
            let seg = arc[k] - arc[k - 1];
            let f = if seg > 0.0 { ((t - arc[k - 1]) / seg).clamp(0.0, 1.0) } else { 0.0 };
            (a.add(b.sub(a).scaled(f)), b.sub(a).normalized_or(WorldPos::new(1.0, 0.0)))
        };
        let goal_comp: BTreeMap<u8, Option<u32>> = specs.iter().map(|(pid, _, goal)| (*pid, h.component_at(*pid, *goal))).collect();
        let mut units: Vec<(usize, u8)> = specs.iter().flat_map(|(pid, idxs, _)| idxs.iter().map(move |&i| (i, *pid))).collect();
        let n = units.len();
        let mut out = Vec::new();
        if n == 0 { return out; }
        let first = pts[0];
        let chord = pts[pts.len() - 1].sub(first);
        let dir = chord.normalized_or(at(0.0).1);
        let side = WorldPos::new(-dir.y, dir.x);
        let key = |p: WorldPos| { let d = p.sub(first); (d.x * dir.x + d.y * dir.y, d.x * side.x + d.y * side.y) };
        units.sort_by(|a, b| {
            let (ka, kb) = (key(self.units[a.0].body.pos), key(self.units[b.0].body.pos));
            ka.0.total_cmp(&kb.0).then(ka.1.total_cmp(&kb.1)).then(self.units[a.0].id.cmp(&self.units[b.0].id))
        });
        let per_row = if n as f32 * spacing <= total { n } else { ((total / spacing) as usize + 1).clamp(1, n) };
        let rows = n.div_ceil(per_row);
        let mut taken: BTreeSet<(i32, i32)> = BTreeSet::new();
        let ok = |pid: u8, c: CellPos, taken: &BTreeSet<(i32, i32)>| -> bool {
            if !geom.in_bounds(c) || !h.grid(pid).is_passable(terrain, c) || h.component_at(pid, c) != goal_comp[&pid] { return false; }
            let r = spacing_cells;
            !near_taken(taken, c, r, min_sep2)
        };
        // (unit index, profile, arc position, cell)
        let mut placed: Vec<(usize, u8, f32, CellPos)> = Vec::with_capacity(n);
        let mut fallback: Vec<(usize, u8)> = Vec::new();
        for col in 0..per_row {
            let (lo, hi) = (col * n / per_row, (col + 1) * n / per_row);
            let t = (col as f32 + 0.5) * total / per_row as f32;
            let (base, tangent) = at(t);
            let normal = WorldPos::new(-tangent.y, tangent.x);
            let mut members: Vec<(usize, u8)> = units[lo..hi].to_vec();
            members.sort_by(|a, b| self.units[a.0].body.pos.dist(base).total_cmp(&self.units[b.0].body.pos.dist(base))
                .then(self.units[a.0].id.cmp(&self.units[b.0].id)));
            let mut next_row = 0usize;
            for (i, pid) in members {
                // Row offsets 0, +1, -1, +2, -2, ... spacings off the curve.
                let row_cell = |k: usize| {
                    let off = if k == 0 { 0.0 } else if k % 2 == 1 { k.div_ceil(2) as f32 } else { -((k / 2) as f32) };
                    geom.world_to_cell(base.add(normal.scaled(off * spacing)))
                };
                let mut found = None;
                for k in next_row..rows.max(next_row + 1) {
                    if ok(pid, row_cell(k), &taken) { found = Some(row_cell(k)); next_row = k + 1; break; }
                }
                // Blocked (an obstacle on the curve): the nearest acceptable
                // cell to the intended row position, then anywhere nearby.
                if found.is_none() {
                    found = terrain.nearest_open_cell(row_cell(next_row), spacing_cells * 2, |c| ok(pid, c, &taken))
                        .or_else(|| terrain.nearest_open_cell(geom.world_to_cell(base), spacing_cells * 4 + 8, |c| ok(pid, c, &taken)));
                    if found.is_some() { next_row += 1; }
                }
                match found {
                    Some(c) => { taken.insert((c.x, c.y)); placed.push((i, pid, t, c)); }
                    None => fallback.push((i, pid)),
                }
            }
        }
        // Group into (profile, segment) cohorts with a shared target.
        let seg_len = FORMATION_SEGMENT_CELLS as f32 * geom.cell_size;
        let mut groups: BTreeMap<(u8, u32, Option<(i32, i32)>), (CellPos, Vec<usize>)> = BTreeMap::new();
        let mut seg_targets: BTreeMap<(u8, u32), Option<CellPos>> = BTreeMap::new();
        for &(i, pid, t, c) in &placed {
            let seg = (t / seg_len) as u32;
            let target = *seg_targets.entry((pid, seg)).or_insert_with(|| {
                let mid = ((seg as f32 + 0.5) * seg_len).min(total);
                terrain.nearest_open_cell(geom.world_to_cell(at(mid).0), 24, |p| h.grid(pid).is_passable(terrain, p) && h.component_at(pid, p) == goal_comp[&pid])
            });
            let (gkey, target) = match target {
                Some(tc) if line_clear(&h, pid, tc, c) => ((pid, seg, None), tc),
                _ => ((pid, seg, Some((c.x, c.y))), c),
            };
            groups.entry(gkey).or_insert((target, Vec::new())).1.push(i);
            let u = &mut self.units[i];
            u.slot = Some(geom.cell_center(c));
            u.slot_cell = Some(c);
            out.push(ArrivalSlot { unit_id: u.id, cell: c, position: geom.cell_center(c) });
        }
        // Units that found no slot on the curve gather at its endpoint.
        if !fallback.is_empty() {
            let reserved: BTreeSet<CellPos> = placed.iter().map(|p| p.3).collect();
            let mut fb_groups: Vec<(u8, CellPos, Vec<u32>)> = Vec::new();
            for (pid, _, goal) in &specs {
                let idxs: Vec<usize> = fallback.iter().filter(|f| f.1 == *pid).map(|f| f.0).collect();
                if !idxs.is_empty() { fb_groups.push((*pid, *goal, self.ids_by_distance(&idxs, geom.cell_center(*goal)))); }
            }
            let slots = self.assign_command_slots(rec.payload.requested_point, &fb_groups, max_radius, &reserved);
            let by_id: BTreeMap<u32, &ArrivalSlot> = slots.iter().map(|s| (s.unit_id, s)).collect();
            for (pid, goal, ids) in &fb_groups {
                for id in ids {
                    let i = self.unit_index(*id).unwrap();
                    match by_id.get(id) {
                        Some(s) => {
                            self.units[i].slot = Some(s.position);
                            self.units[i].slot_cell = Some(s.cell);
                            out.push((*s).clone());
                            groups.entry((*pid, u32::MAX, None)).or_insert((*goal, Vec::new())).1.push(i);
                        }
                        None => Self::mark_unreachable(&mut self.units[i]),
                    }
                }
            }
            rec.skipped_guides = fallback.len() as u32;
        }
        for ((pid, _, _), (target, members)) in groups {
            let Some(stage) = self.make_stage(pid, target, tolerance) else {
                for i in members { Self::mark_unreachable(&mut self.units[i]); }
                continue;
            };
            let cid = self.next_cohort_id;
            self.next_cohort_id += 1;
            for &i in &members { self.units[i].cohort_id = Some(cid); }
            let crowd_aware = rec.total_units >= CROWD_AWARE_MIN_UNITS;
            let mut stages = vec![stage];
            self.key_stages_to_crowd(&mut stages, crowd_aware);
            self.cohorts.insert(cid, Cohort { id: cid, command_id, profile_id: pid, stages, active_units: members.len() as u32, crowd_aware, crowd_keyed_tick: self.tick });
            rec.cohort_ids.push(cid);
        }
        out
    }

    /// Terrain repair for a `CurveFormation` command: only slots that became
    /// invalid move, to the nearest acceptable cell around the old slot (or
    /// around the cohort target), so the formation keeps its shape.
    fn repair_formation_slots(&mut self, command_id: u64) {
        let h = Arc::clone(&self.hierarchy);
        let terrain = &h.terrain;
        let geom = &terrain.geom;
        let max_radius = self.units.iter().filter(|u| u.command_id == Some(command_id)).map(|u| u.body.radius).fold(0.0f32, f32::max);
        let spacing_cells = ((2.35 * max_radius / geom.cell_size).ceil() as i32).max(1);
        let min_sep2 = { let m = (spacing_cells as f32 * 0.75).max(1.0); (m * m) as i32 };
        let mut taken: BTreeSet<(i32, i32)> = self.units.iter()
            .filter(|u| u.command_id == Some(command_id)).filter_map(|u| u.slot_cell).map(|c| (c.x, c.y)).collect();
        for i in 0..self.units.len() {
            let u = &self.units[i];
            if u.command_id != Some(command_id) || u.arrived || u.unreachable { continue; }
            let (Some(cid), Some(old)) = (u.cohort_id, u.slot_cell) else { continue; };
            let Some(target) = self.cohorts.get(&cid).and_then(|c| c.stages.last()).map(|s| s.target) else { continue; };
            let pid = u.profile_id;
            let comp = h.component_at(pid, target);
            let valid = |c: CellPos| geom.in_bounds(c) && h.grid(pid).is_passable(terrain, c) && h.component_at(pid, c) == comp && line_clear(&h, pid, target, c);
            if valid(old) { continue; }
            taken.remove(&(old.x, old.y));
            let r = spacing_cells;
            let free = |c: CellPos, taken: &BTreeSet<(i32, i32)>| !near_taken(taken, c, r, min_sep2);
            let found = terrain.nearest_open_cell(old, spacing_cells * 4 + 8, |c| valid(c) && free(c, &taken))
                .or_else(|| terrain.nearest_open_cell(target, 32, |c| valid(c) && free(c, &taken)));
            let u = &mut self.units[i];
            match found {
                Some(c) => {
                    taken.insert((c.x, c.y));
                    u.slot = Some(geom.cell_center(c));
                    u.slot_cell = Some(c);
                    u.arrival_phase = false;
                    u.best_slot_dist = u64::MAX;
                    u.stuck_ticks = 0;
                }
                None => { u.slot = None; u.slot_cell = None; Self::mark_unreachable(u); }
            }
        }
        let mut all_slots: Vec<ArrivalSlot> = self.units.iter().filter_map(|u| {
            if u.command_id != Some(command_id) { return None; }
            Some(ArrivalSlot { unit_id: u.id, cell: u.slot_cell?, position: u.slot? })
        }).collect();
        all_slots.sort_by_key(|s| s.unit_id);
        if let Some(rec) = self.commands.get_mut(&command_id) { rec.slots = all_slots; }
    }

    fn handle_command(&mut self, q: QueuedCommand, tick: u64) {
        let now = Instant::now();
        let payload = q.payload;
        if self.commands.contains_key(&payload.id) { return; }
        let mut selected: Vec<u32> = payload.unit_ids.clone();
        selected.sort_unstable();
        selected.dedup();
        selected.retain(|id| self.unit_index(*id).is_some());
        // Replacement is O(selected): units drop their handles; old cohorts
        // and shared structures are reclaimed lazily (cohorts when their last
        // unit leaves, fields/contexts by bounded LRU caches).
        let mut replaced: BTreeMap<u64, u32> = BTreeMap::new();
        for uid in &selected {
            let i = self.unit_index(*uid).unwrap();
            let u = &mut self.units[i];
            if let Some(old) = u.command_id { if old != payload.id { *replaced.entry(old).or_insert(0) += 1; } }
            u.command_id = Some(payload.id);
            Self::reset_unit_nav(u);
        }
        for (old, n) in replaced {
            if let Some(rec) = self.commands.get_mut(&old) { rec.replaced_units = rec.replaced_units.saturating_add(n); }
        }
        if matches!(payload.kind, CommandKind::Stop) {
            // No record: the units go idle. Home is set once each comes to
            // rest (`update_unit_progress`), so a fast unit does not brake
            // past it and walk back.
            for uid in &selected {
                let i = self.unit_index(*uid).unwrap();
                let u = &mut self.units[i];
                u.command_id = None;
                u.mode = UnitMode::Idle;
            }
            return;
        }
        let requested = payload.requested_point;
        let geom = self.hierarchy.terrain.geom.clone();
        let raw_guides: Vec<CellPos> = match &payload.kind {
            CommandKind::FollowCurve { guide_points } => guide_points.iter().map(|p| geom.world_to_cell(*p)).collect(),
            CommandKind::MoveTo | CommandKind::CurveFormation { .. } | CommandKind::Stop => Vec::new(),
        };
        let tolerance = ((payload.corridor_tolerance_world / geom.cell_size).ceil() as i32).max(2);
        let mut rec = CommandRecord {
            payload: payload.clone(), commit_tick: tick, requested_tick: tick,
            slots: Vec::new(), status: CommandStatus::Planning, unreachable_units: 0,
            arrived_units: 0, total_units: selected.len() as u32, active_units: selected.len() as u32,
            replaced_units: 0, first_move_tick: None, first_move_wall_ms: None,
            committed_wall_ms: q.submitted_at.map(|t| now.duration_since(t).as_secs_f64() * 1000.0).unwrap_or(0.0),
            skipped_guides: 0, truncated: false, cohort_ids: Vec::new(), route_polylines: Vec::new(),
            repair_rerouted_units: 0, repair_invalidated_units: 0,
        };
        self.command_received_at.insert(payload.id, q.submitted_at.unwrap_or(now));
        // Units by profile; validate connectivity to the snapped goal.
        let mut by_profile: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
        for uid in &selected { let i = self.unit_index(*uid).unwrap(); by_profile.entry(self.units[i].profile_id).or_default().push(i); }
        let mut slot_groups: Vec<(u8, CellPos, Vec<u32>)> = Vec::new();
        let mut cohort_specs: Vec<(u8, Vec<usize>, CellPos)> = Vec::new();
        let mut max_radius = 0.0f32;
        for (pid, idxs) in by_profile {
            let goal = self.command_goal(pid, requested);
            let goal_comp = goal.and_then(|g| self.hierarchy.component_at(pid, g));
            let mut ok = Vec::new();
            for i in idxs {
                let comp = Self::nav_cell_of(&self.hierarchy, pid, self.units[i].body.pos).and_then(|c| self.hierarchy.component_at(pid, c));
                if goal_comp.is_some() && comp == goal_comp { ok.push(i); } else { Self::mark_unreachable(&mut self.units[i]); }
            }
            let Some(goal) = goal else { continue; };
            if ok.is_empty() { continue; }
            max_radius = ok.iter().map(|i| self.units[*i].body.radius).fold(max_radius, f32::max);
            slot_groups.push((pid, goal, self.ids_by_distance(&ok, geom.cell_center(goal))));
            cohort_specs.push((pid, ok, goal));
        }
        let mut slot_by_id: BTreeMap<u32, ArrivalSlot> = BTreeMap::new();
        if let CommandKind::CurveFormation { guide_points } = &payload.kind {
            for s in self.plan_formation(payload.id, guide_points, cohort_specs, max_radius, tolerance, &mut rec) { slot_by_id.insert(s.unit_id, s); }
        } else {
            let reserved = BTreeSet::new();
            for s in self.assign_command_slots(requested, &slot_groups, max_radius, &reserved) { slot_by_id.insert(s.unit_id, s); }
            for (pid, idxs, goal) in cohort_specs {
                let mut members = Vec::new();
                for i in idxs {
                    match slot_by_id.get(&self.units[i].id) {
                        Some(s) => { self.units[i].slot = Some(s.position); self.units[i].slot_cell = Some(s.cell); members.push(i); }
                        None => Self::mark_unreachable(&mut self.units[i]),
                    }
                }
                if members.is_empty() { continue; }
                let start = Self::nav_cell_of(&self.hierarchy, pid, self.units[members[0]].body.pos).unwrap_or(goal);
                let (guides, skipped) = self.plan_guides(pid, goal, start, &raw_guides, tolerance);
                // Guides are shared by all profiles of the command: report the
                // worst profile, not the sum.
                rec.skipped_guides = rec.skipped_guides.max(skipped);
                let mut stages: Vec<Stage> = guides.iter().filter_map(|g| self.make_stage(pid, *g, tolerance)).collect();
                match self.make_stage(pid, goal, tolerance) {
                    Some(s) => stages.push(s),
                    None => { for i in members { Self::mark_unreachable(&mut self.units[i]); } continue; }
                }
                let cid = self.next_cohort_id;
                self.next_cohort_id += 1;
                for &i in &members { self.units[i].cohort_id = Some(cid); }
                let crowd_aware = rec.total_units >= CROWD_AWARE_MIN_UNITS;
                self.key_stages_to_crowd(&mut stages, crowd_aware);
                self.cohorts.insert(cid, Cohort { id: cid, command_id: payload.id, profile_id: pid, stages, active_units: members.len() as u32, crowd_aware, crowd_keyed_tick: self.tick });
                rec.cohort_ids.push(cid);
            }
        }
        rec.truncated = rec.skipped_guides > 0;
        rec.slots = slot_by_id.into_values().collect();
        rec.slots.sort_by_key(|s| s.unit_id);
        self.commands.insert(payload.id, rec);
        self.command_order.push_back(payload.id);
        while self.command_order.len() > HISTORY_LIMIT {
            let old = *self.command_order.front().unwrap();
            if self.units.iter().any(|u| u.command_id == Some(old)) { break; }
            self.command_order.pop_front();
            self.commands.remove(&old);
            self.command_received_at.remove(&old);
        }
    }

    // ----- job scheduling ---------------------------------------------------

    fn submit(&mut self, job: WorkerJob, kind: PendingKind, label: &'static str, due_tick: u64) -> u64 {
        let jid = self.pool.submit(job);
        self.pending.insert(jid, PendingJob { due_tick, kind, stale: false, label });
        self.nav.jobs_submitted += 1;
        jid
    }

    /// Submit queued field requests under the per-tick quota, in priority
    /// order (`field_priority`), and give each a due time from a
    /// deterministic cost clock in sim time: a job is due when the assumed
    /// throughput (`FIELD_PACE_CELLS_PER_MS`) would have built it and every
    /// job before it, but never later than `FIELD_MAX_DELAY_MS` after the
    /// request and never before the next tick. A burst of requests is thus
    /// spread over several ticks; the tick that applies them waits only if
    /// the workers are slower than assumed. Nothing here reads wall time.
    fn submit_requests(&mut self) {
        let now = sim_time_us(self.tick);
        let cap = now + FIELD_MAX_DELAY_MS * 1000;
        let needs = std::mem::take(&mut self.field_needs);
        let mut submitted = 0;
        for (prio, key) in needs {
            if self.field_cache.contains(&key) || self.pending_fields.contains_key(&key) { continue; }
            if key.sector_version != self.hierarchy.sector_version(key.profile_id, key.sector) { continue; }
            let backlog = self.field_clock_us.saturating_sub(now);
            let prefetch = prio >> PRIORITY_CLASS_SHIFT >= PRIORITY_PREFETCH;
            if submitted >= FIELD_JOBS_PER_TICK || (prefetch && backlog > PREFETCH_BACKLOG_MS * 1000) {
                self.field_needs.insert((prio, key));
                self.nav.field_requests_deferred += 1;
                continue;
            }
            let (x0, y0, x1, y1) = self.hierarchy.terrain.geom.sector_bounds(key.sector);
            let cells = ((x1 - x0) * (y1 - y0)).max(1) as u64;
            self.field_clock_us = (self.field_clock_us.max(now) + cells * 1000 / FIELD_PACE_CELLS_PER_MS).min(cap);
            let due = tick_at_or_after_us(self.field_clock_us).max(self.tick + 1);
            let job = WorkerJob::BuildField { hierarchy: Arc::clone(&self.hierarchy), key: key.clone() };
            let jid = self.submit(job, PendingKind::Field(key.clone()), "field", due);
            self.pending_fields.insert(key, jid);
            submitted += 1;
        }
    }

    fn apply_due_jobs(&mut self) {
        let tick = self.tick;
        let due: Vec<u64> = self.pending.iter().filter(|(_, p)| p.due_tick <= tick).map(|(id, _)| *id).collect();
        let wait_ids: Vec<u64> = due.iter().copied().filter(|id| !self.pending[id].stale).collect();
        // Deterministic barrier: a due result is applied at this boundary even
        // if the machine is slow (the tick waits instead of diverging).
        self.pool.wait_for(&wait_ids);
        for id in due {
            let meta = self.pending.remove(&id).unwrap();
            match &meta.kind {
                PendingKind::Field(k) => { self.pending_fields.remove(k); }
                PendingKind::Repair => {}
            }
            if meta.stale {
                self.pool.discard(id);
                self.nav.jobs_stale += 1;
                continue;
            }
            let Some(result) = self.pool.take(id) else { continue; };
            self.nav.jobs_completed += 1;
            match result {
                WorkerResult::Field { field, .. } => match field {
                    Some(f) => self.insert_field(f),
                    None => self.nav.jobs_failed += 1,
                },
                WorkerResult::Hierarchy { hierarchy } => {
                    self.install_hierarchy(hierarchy);
                    self.start_repair();
                }
            }
        }
    }

    fn insert_field(&mut self, f: FlowField) {
        if f.key.sector_version != self.hierarchy.sector_version(f.key.profile_id, f.key.sector) { return; }
        self.nav.fields_built += 1;
        self.nav.field_expansions += f.expansions as u64;
        self.nav.last_field_ms = f.build_ms;
        let f = self.field_cache.insert(f, self.tick);
        for c in self.cohorts.values_mut() {
            if c.profile_id != f.key.profile_id { continue; }
            for st in &mut c.stages {
                for slot in st.slots.values_mut().flatten() {
                    if slot.field.is_none() && slot.key == f.key { slot.field = Some(Arc::clone(&f)); }
                }
            }
        }
    }

    fn evict_contexts(&mut self) {
        while self.contexts.len() > CONTEXT_CACHE_MAX {
            let victim = self.contexts.iter().min_by_key(|(k, (_, used))| (*used, (*k).clone())).map(|(k, _)| k.clone()).unwrap();
            self.contexts.remove(&victim);
        }
    }

    /// Give every stage waiting for `key` the new context. A stage switching
    /// from an older-version context drops its slots; they are re-derived
    /// lazily and identical recipes hit the field cache. Returns whether any
    /// stage changed; callers then run `contexts_attached` once per batch.
    ///
    /// A stage whose context changed only in its crowd costs (same
    /// navigation version and goal) instead keeps the slots whose recipe is
    /// unchanged, and only its own cohort's units re-enter their regions
    /// and re-base their progress metric (the new labels include the crowd
    /// costs, so the old best would make them look stuck, and a plain reset
    /// would make a stuck unit look like it progressed). Their commands go
    /// to `crowd_touched`, for route redrawing only.
    ///
    /// Checking the kept slots' recipes is deferred to `contexts_attached`,
    /// which checks all of a batch's stages on the pool. A stage is attached
    /// at most once per batch (its cohort's stages share one crowd version),
    /// so checking against its final context is the same as checking here.
    fn attach_context(&mut self, key: &ContextKey, out: &mut Attached) {
        let Some((ctx, _)) = self.contexts.get(key) else { return; };
        let ctx = Arc::clone(ctx);
        for c in self.cohorts.values_mut() {
            for (si, st) in c.stages.iter_mut().enumerate() {
                if st.ctx_key != *key || st.fresh() { continue; }
                let crowd_only = st.ctx.as_ref().is_some_and(|old| ContextKey { crowd: key.crowd, ..old.key.clone() } == *key);
                if crowd_only {
                    out.prune.push((c.id, si));
                    out.rekeyed.insert(c.id);
                    out.crowd_touched.insert(c.command_id);
                } else {
                    st.slots.clear();
                    out.touched.insert(c.command_id);
                }
                st.ctx = Some(Arc::clone(&ctx));
            }
        }
    }

    /// Rebuild the standing-crowd maps from units standing still: idle once
    /// at rest, unreachable, arrived `CROWD_ARRIVED_TICKS` ago, or queued
    /// (`CROWD_QUEUED_TICKS` without progress on the way), with the traffic
    /// levels of units on their way (`Traffic`: speed toward their
    /// waypoints). Queued and slow units make a jammed gap cost more, so the
    /// tail of a queue takes another way while its front still goes
    /// through. Every
    /// command sees all of them, except that a crowd-aware command still
    /// arriving gets its own map without its own arrived units: its late
    /// units must not route around their own gathering (counting their
    /// comrades made 2,700 of 10,000 units never arrive), while other
    /// commands must see it at once (waiting for a command to finish, or to
    /// stop receiving arrivals, hid whole armies for minutes behind a trickle
    /// of stragglers). When a command's map changed, its crowd-aware stages
    /// want contexts with the new costs (built by `request_stage_contexts`).
    fn update_crowd(&mut self) {
        let tick = self.tick;
        {
            let geom = &self.hierarchy.terrain.geom;
            let movers = self.units.iter().filter_map(|u| {
                if u.cohort_id.is_none() || u.arrived || u.unreachable || u.arrival_phase { return None; }
                if !matches!(u.mode, UnitMode::Moving | UnitMode::Blocked) { return None; }
                let (wc, _) = u.waypoint?;
                let to = geom.cell_center(wc).sub(u.body.pos);
                let d = to.length();
                if d < 1e-3 { return None; }
                let along = (u.body.vel.x * to.x + u.body.vel.y * to.y) / d;
                Some((u.body.pos, u.body.radius, (along / u.max_speed).clamp(0.0, 1.0)))
            });
            self.traffic.update(&self.crowd, geom, movers);
        }
        // (position, radius, the command it arrived under).
        let standing: Vec<(WorldPos, f32, Option<u64>)> = self.units.iter()
            .filter(|u| u.unreachable || (u.command_id.is_none() && u.home.is_some())
                || (u.arrived && u.arrived_tick.is_some_and(|t| tick >= t + CROWD_ARRIVED_TICKS))
                || (u.cohort_id.is_some() && !u.arrived && !u.arrival_phase && u.stuck_ticks > CROWD_QUEUED_TICKS))
            .map(|u| (u.body.pos, u.body.radius, if u.arrived { u.command_id } else { None }))
            .collect();
        if let Some(next) = self.crowd.update(&self.hierarchy, standing.iter().map(|b| (b.0, b.1)), &self.traffic.slow, self.crowd_versions + 1) {
            self.crowd_versions += 1;
            self.crowd = Arc::new(next);
            self.nav.crowd_version = self.crowd_versions;
        }
        let with_arrivals: BTreeSet<u64> = standing.iter().filter_map(|b| b.2).collect();
        let arriving: BTreeSet<u64> = self.cohorts.values()
            .filter(|c| c.crowd_aware && c.active_units > 0 && with_arrivals.contains(&c.command_id))
            .map(|c| c.command_id).collect();
        self.own_crowd.retain(|id, _| arriving.contains(id));
        for id in arriving {
            let base = Arc::clone(self.crowd_for(id));
            let mine = standing.iter().filter(|b| b.2 != Some(id)).map(|b| (b.0, b.1));
            if let Some(next) = base.update(&self.hierarchy, mine, &self.traffic.slow, self.crowd_versions + 1) {
                self.crowd_versions += 1;
                self.own_crowd.insert(id, Arc::new(next));
            }
        }
        let versions: BTreeMap<u64, u32> = self.cohorts.values().filter(|c| c.crowd_aware)
            .map(|c| (c.command_id, self.crowd_for(c.command_id).version)).collect();
        let mut due: Vec<(u64, u32)> = self.cohorts.values()
            .filter(|c| c.crowd_aware && c.stages.iter().any(|st| st.ctx_key.crowd != versions[&c.command_id]))
            .map(|c| (c.crowd_keyed_tick, c.id)).collect();
        due.sort_unstable();
        for (_, id) in due.into_iter().take(CROWD_REKEYS_PER_UPDATE) {
            let c = self.cohorts.get_mut(&id).expect("cohort");
            let v = versions[&c.command_id];
            for st in &mut c.stages { st.ctx_key.crowd = v; }
            c.crowd_keyed_tick = tick;
        }
    }

    /// After a batch of `attach_context` calls: crowd-only stages keep the
    /// slots whose recipe is unchanged (checked in parallel, one stage per
    /// job), units re-enter their regions (prefetch again) and the touched
    /// commands' route lines are redrawn. Units of `rekeyed` cohorts
    /// (crowd-only changes) re-base their progress, in one pass.
    fn contexts_attached(&mut self, attached: Attached) {
        let Attached { touched, crowd_touched, rekeyed, prune } = attached;
        if !prune.is_empty() {
            let (h, cohorts) = (&*self.hierarchy, &self.cohorts);
            let keeps: Vec<Vec<bool>> = self.par.map(&prune, |&(cid, si)| {
                let st = &cohorts[&cid].stages[si];
                let ctx = st.ctx.as_ref().expect("attached stage");
                st.slots.iter().map(|(&(sector, region), slot)| match (slot, field_recipe(h, ctx, sector, region, st.target)) {
                    (Some(f), Some(rc)) => rc.key == f.key && rc.offset == f.offset && rc.best_exit_far_node == f.next_node,
                    (None, None) => true,
                    _ => false,
                }).collect()
            });
            for (&(cid, si), keep) in prune.iter().zip(keeps) {
                let mut keep = keep.into_iter();
                self.cohorts.get_mut(&cid).expect("cohort").stages[si].slots.retain(|_, _| keep.next().expect("one flag per slot"));
            }
        }
        if !rekeyed.is_empty() {
            self.nav.crowd_rekeys += rekeyed.len() as u64;
            for u in self.units.iter_mut().filter(|u| u.cohort_id.is_some_and(|c| rekeyed.contains(&c))) {
                u.last_region = None;
                u.rebase_progress = true;
            }
        }
        if !crowd_touched.is_empty() { self.refresh_route_polylines(&crowd_touched); }
        if touched.is_empty() { return; }
        for u in &mut self.units { u.last_region = None; }
        self.refresh_route_polylines(&touched);
    }

    /// A context failed to build (its target is no longer valid).
    fn fail_context(&mut self, key: &ContextKey) {
        let mut doomed: Vec<(u32, usize)> = Vec::new();
        for c in self.cohorts.values() {
            for (si, st) in c.stages.iter().enumerate() { if st.ctx_key == *key { doomed.push((c.id, si)); } }
        }
        for (cid, si) in doomed {
            let last = self.cohorts[&cid].stages.len() - 1;
            if si < last {
                // An impossible intermediate checkpoint is skipped.
                for u in self.units.iter_mut().filter(|u| u.cohort_id == Some(cid) && u.stage_idx as usize == si) { u.stage_idx += 1; }
                self.nav.skipped_stages += 1;
            } else {
                for u in self.units.iter_mut().filter(|u| u.cohort_id == Some(cid) && !u.arrived) { Self::mark_unreachable(u); }
            }
        }
    }

    fn refresh_route_polylines(&mut self, commands: &BTreeSet<u64>) {
        // Each cohort's line starts at its first unit (one pass over units).
        let mut first: BTreeMap<u32, WorldPos> = BTreeMap::new();
        for u in &self.units { if let Some(c) = u.cohort_id { first.entry(c).or_insert(u.body.pos); } }
        // (command, context, from, target) for every line, in drawing order;
        // the lines are traced on the pool.
        let mut legs: Vec<(u64, &Arc<DestContext>, CellPos, CellPos)> = Vec::new();
        for &command_id in commands {
            let Some(rec) = self.commands.get(&command_id) else { continue; };
            for cid in &rec.cohort_ids {
                let (Some(c), Some(pos)) = (self.cohorts.get(cid), first.get(cid)) else { continue; };
                let mut from = self.hierarchy.terrain.geom.world_to_cell(*pos);
                for st in &c.stages {
                    if let Some(ctx) = &st.ctx {
                        if ctx.key.nav_version == self.hierarchy.version { legs.push((command_id, ctx, from, st.target)); }
                    }
                    from = st.target;
                }
            }
        }
        let h = &*self.hierarchy;
        let traced = self.par.map(&legs, |&(_, ctx, from, target)| route_polyline(h, ctx, from, target));
        let mut lines: BTreeMap<u64, Vec<Vec<WorldPos>>> = BTreeMap::new();
        for (leg, line) in legs.iter().zip(traced) { lines.entry(leg.0).or_default().push(line); }
        for &command_id in commands {
            if let Some(rec) = self.commands.get_mut(&command_id) { rec.route_polylines = lines.remove(&command_id).unwrap_or_default(); }
        }
    }

    // ----- terrain ----------------------------------------------------------

    /// A terrain edit takes effect together with its navigation repair:
    /// both are built on a worker and installed when the job is due
    /// (`REPAIR_DELAY_MS`). One repair runs at a time; edits made meanwhile
    /// queue for the next, which starts as soon as the current one lands.
    fn handle_terrain_patch(&mut self, patch: TerrainPatch, tick: u64) {
        if patch.cells.is_empty() { return; }
        let Some(rect) = patch.dirty_cells() else { return; };
        self.last_terrain_edit_tick = Some(tick);
        self.pending_rect = Some(union_rect(self.pending_rect, rect));
        self.unrepaired_patches.push(patch);
        self.start_repair();
    }

    fn start_repair(&mut self) {
        if self.unrepaired_patches.is_empty() || self.repair_rect.is_some() { return; }
        let patches = std::mem::take(&mut self.unrepaired_patches);
        let Some(rect) = patches.iter().filter_map(|p| p.dirty_cells()).reduce(|a, b| union_rect(Some(a), b)) else { return; };
        self.repair_rect = Some(rect);
        let job = WorkerJob::RepairHierarchy { previous: Arc::clone(&self.hierarchy), patches, rect, threads: self.cfg.worker_count };
        let due = tick_at_or_after_us(sim_time_us(self.tick) + REPAIR_DELAY_MS * 1000).max(self.tick + 1);
        self.submit(job, PendingKind::Repair, "terrain edit + nav repair", due);
    }

    /// Update the cached terrain digest after the rows `y0..=y1` changed
    /// (`None`: anything may have changed). Bands are hashed in parallel.
    fn refresh_terrain_digest(&mut self, rows: Option<(i32, i32)>) {
        let t = &self.terrain;
        let bands: Vec<usize> = match rows {
            Some((y0, y1)) if self.terrain_bands.len() == t.digest_bands() => (t.digest_band_of(y0)..=t.digest_band_of(y1)).collect(),
            _ => { self.terrain_bands = vec![0; t.digest_bands()]; (0..t.digest_bands()).collect() }
        };
        let digests = self.par.map(&bands, |&b| t.band_digest(b));
        for (b, d) in bands.into_iter().zip(digests) { self.terrain_bands[b] = d; }
        self.terrain_digest = (t.version, Terrain::digest_of_bands_with_water(t.version, t.water_level, &self.terrain_bands));
    }

    fn ensure_terrain_digest(&mut self) {
        if self.terrain_digest.0 != self.terrain.version { self.refresh_terrain_digest(None); }
    }

    /// Sectors whose navigation may be invalid until the pending repair
    /// lands: the edit rect plus the clearance influence margin.
    fn pending_dirty_sectors(&self) -> BTreeSet<u32> {
        let mut out = BTreeSet::new();
        let Some((x0, y0, x1, y1)) = self.pending_rect else { return out; };
        let g = &self.hierarchy.terrain.geom;
        let m = self.hierarchy.grids.iter().map(|g| g.influence_margin()).max().unwrap_or(1) + 1;
        let (x0, y0, x1, y1) = ((x0 - m).max(0), (y0 - m).max(0), (x1 + m).min(g.cells_x - 1), (y1 + m).min(g.cells_y - 1));
        for sy in y0 / g.sector_cells..=y1 / g.sector_cells {
            for sx in x0 / g.sector_cells..=x1 / g.sector_cells { out.insert(g.sector_index(SectorPos::new(sx, sy)) as u32); }
        }
        out
    }

    /// Install a repaired hierarchy. Policy (safety first):
    /// * fields of changed sectors are invalid immediately (their version is
    ///   part of the key); fields of unchanged sectors stay valid and cached;
    /// * every stage asks for a context at the new version; until it arrives
    ///   (normally the next tick) units keep using old-version fields only in
    ///   unchanged sectors: bounded, explicit temporary suboptimality;
    /// * units now standing in rock are displaced to the nearest open cell;
    /// * goals/slots that became invalid are re-snapped; units whose region
    ///   lost connectivity to the goal are reported unreachable.
    fn install_hierarchy(&mut self, hierarchy: Arc<NavHierarchy>) {
        let tick = self.tick;
        let old = std::mem::replace(&mut self.hierarchy, hierarchy);
        self.retired_hierarchies.push(old);
        self.terrain = Arc::clone(&self.hierarchy.terrain);
        if let Some(r) = self.repair_rect.take() {
            self.refresh_terrain_digest(Some((r.1, r.3)));
            self.terrain_changes.push(TerrainChange { tick, version: self.terrain.version, cells: r });
        }
        self.pending_rect = self.unrepaired_patches.iter().filter_map(|p| p.dirty_cells()).reduce(|a, b| union_rect(Some(a), b));
        self.nav.terrain_rebuilds += 1;
        self.nav.last_repair_ms = self.hierarchy.build_ms;
        self.nav.last_repair_sectors = self.hierarchy.sectors_recomputed;
        self.nav.last_repair_latency_ticks = self.last_terrain_edit_tick.map(|t| tick - t).unwrap_or(0);
        let h = Arc::clone(&self.hierarchy);
        self.field_cache.purge_sector_versions(|pid, s| h.sector_version(pid, s));
        self.contexts.retain(|k, _| k.nav_version == h.version);
        self.crowd = Arc::new(self.crowd.for_hierarchy(&h));
        for m in self.own_crowd.values_mut() { *m = Arc::new(m.for_hierarchy(&h)); }
        let crowd_versions: BTreeMap<u64, u32> = self.cohorts.values().map(|c| (c.command_id, self.crowd_for(c.command_id).version)).collect();
        let geom = h.terrain.geom.clone();
        // Displace units buried by the edit (scripted terrain event).
        for u in &mut self.units {
            let c = geom.world_to_cell(u.body.pos);
            u.waypoint = None;
            // Buried means in a cell the unit's medium may not occupy (an
            // obstacle on a dry map, also the wrong depth on a wet one).
            if !geom.in_bounds(c) || h.terrain.open_for(geom.cell_index(c), &self.profiles[u.profile_id as usize].medium) { continue; }
            let grid = h.grid(u.profile_id);
            if let Some(open) = h.terrain.nearest_open_cell(c, 32, |p| grid.is_passable(&h.terrain, p)) {
                u.body.pos = geom.cell_center(open);
                u.body.vel = WorldPos::ZERO;
                self.nav.terrain_relocated_units += 1;
            }
        }
        // Re-key every stage; drop slots in changed sectors.
        let dirty: BTreeSet<u32> = h.dirty_sectors.iter().copied().collect();
        let cids: Vec<u32> = self.cohorts.keys().copied().collect();
        let mut reslot_commands = BTreeSet::new();
        let mut failed: Vec<ContextKey> = Vec::new();
        for &cid in &cids {
            let pid = self.cohorts[&cid].profile_id;
            let grid = h.grid(pid);
            let c = self.cohorts.get_mut(&cid).unwrap();
            for st in &mut c.stages {
                st.slots.retain(|(s, _), _| !dirty.contains(s));
                let target = if grid.is_passable(&h.terrain, st.target) { Some(st.target) }
                    else { h.terrain.nearest_open_cell(st.target, 24, |p| grid.is_passable(&h.terrain, p)) };
                match target.and_then(|t| context_key(&h, pid, t).map(|k| (t, k))) {
                    Some((t, k)) => { st.target = t; st.ctx_key = ContextKey { crowd: if c.crowd_aware { crowd_versions[&c.command_id] } else { 0 }, ..k }; }
                    None => failed.push(st.ctx_key.clone()),
                }
            }
        }
        for k in failed { self.fail_context(&k); }
        for &cid in &cids {
            let c = &self.cohorts[&cid];
            let goal_comp = c.stages.last().and_then(|st| h.component_at(c.profile_id, st.target));
            for u in self.units.iter().filter(|u| u.cohort_id == Some(cid) && !u.arrived && !u.unreachable) {
                let ok = u.slot_cell.map(|s| h.grid(c.profile_id).is_passable(&h.terrain, s) && h.component_at(c.profile_id, s) == goal_comp).unwrap_or(false);
                if !ok { reslot_commands.insert(c.command_id); }
            }
        }
        // Distinguish invalid routes (lost connectivity) from merely
        // re-costed ones.
        let mut invalid_by_cmd: BTreeMap<u64, u32> = BTreeMap::new();
        let mut rerouted_by_cmd: BTreeMap<u64, u32> = BTreeMap::new();
        for i in 0..self.units.len() {
            let u = &self.units[i];
            let (Some(cid), false, false) = (u.cohort_id, u.arrived, u.unreachable) else { continue; };
            let Some(c) = self.cohorts.get(&cid) else { continue; };
            let goal_comp = c.stages.last().and_then(|st| h.component_at(c.profile_id, st.target));
            let here = Self::nav_cell_of(&h, u.profile_id, u.body.pos).and_then(|p| h.component_at(u.profile_id, p));
            let cmd = c.command_id;
            if goal_comp.is_none() || here != goal_comp {
                Self::mark_unreachable(&mut self.units[i]);
                *invalid_by_cmd.entry(cmd).or_insert(0) += 1;
            } else {
                *rerouted_by_cmd.entry(cmd).or_insert(0) += 1;
                let u = &mut self.units[i];
                u.best_progress = (u.stage_idx, u64::MAX); u.best_slot_dist = u64::MAX;
                u.last_region = None;
            }
        }
        for (cmd, n) in invalid_by_cmd {
            self.nav.terrain_invalidated_units += n as u64;
            if let Some(r) = self.commands.get_mut(&cmd) { r.repair_invalidated_units += n; }
        }
        for (cmd, n) in rerouted_by_cmd {
            self.nav.terrain_rerouted_units += n as u64;
            if let Some(r) = self.commands.get_mut(&cmd) { r.repair_rerouted_units = n; }
        }
        for cmd in reslot_commands { self.reassign_slots(cmd); }
    }

    /// Recompute all slots of a command's unfinished units (deterministic).
    /// Arrived units are intentionally excluded: a terrain edit must never
    /// resurrect a unit that already reached its slot.
    fn reassign_slots(&mut self, command_id: u64) {
        let Some(rec) = self.commands.get(&command_id) else { return; };
        if matches!(rec.payload.kind, CommandKind::CurveFormation { .. }) { return self.repair_formation_slots(command_id); }
        let requested = rec.payload.requested_point;
        let mut groups: Vec<(u8, CellPos, Vec<u32>)> = Vec::new();
        // Spacing must account for every body of the command, including units
        // that already arrived and therefore keep their slots.
        let max_radius = self.units.iter()
            .filter(|u| u.command_id == Some(command_id))
            .map(|u| u.body.radius)
            .fold(0.0f32, f32::max);
        for cid in rec.cohort_ids.clone() {
            let Some(c) = self.cohorts.get(&cid) else { continue; };
            let Some(goal) = c.stages.last().map(|s| s.target) else { continue; };
            let idxs: Vec<usize> = (0..self.units.len()).filter(|&i| {
                let u = &self.units[i];
                u.cohort_id == Some(cid) && !u.arrived && !u.unreachable
            }).collect();
            if idxs.is_empty() { continue; }
            groups.push((c.profile_id, goal, self.ids_by_distance(&idxs, self.hierarchy.terrain.geom.cell_center(goal))));
        }
        let reserved: BTreeSet<CellPos> = self.units.iter()
            .filter(|u| u.command_id == Some(command_id) && (u.arrived || u.unreachable))
            .filter_map(|u| u.slot_cell)
            .collect();
        let slots = self.assign_command_slots(requested, &groups, max_radius, &reserved);
        let by_id: BTreeMap<u32, ArrivalSlot> = slots.iter().map(|s| (s.unit_id, s.clone())).collect();
        for (_, _, ids) in &groups {
            for id in ids {
                let i = self.unit_index(*id).unwrap();
                match by_id.get(id) {
                    Some(s) => {
                        let u = &mut self.units[i];
                        u.slot = Some(s.position);
                        u.slot_cell = Some(s.cell);
                        u.arrival_phase = false;
                        u.best_progress = (u.stage_idx, u64::MAX); u.best_slot_dist = u64::MAX;
                        u.last_region = None;
                        u.stuck_ticks = 0;
                        u.mode = UnitMode::WaitingForNav;
                    }
                    None => {
                        let u = &mut self.units[i];
                        u.slot = None;
                        u.slot_cell = None;
                        Self::mark_unreachable(u);
                    }
                }
            }
        }
        // Keep the display record in sync with every unit, including units
        // that had already arrived before the edit.
        let mut all_slots: Vec<ArrivalSlot> = self.units.iter().filter_map(|u| {
            if u.command_id != Some(command_id) { return None; }
            Some(ArrivalSlot { unit_id: u.id, cell: u.slot_cell?, position: u.slot? })
        }).collect();
        all_slots.sort_by_key(|s| s.unit_id);
        if let Some(rec) = self.commands.get_mut(&command_id) { rec.slots = all_slots; }
    }

    /// Make sure each cohort's current stage (and the next one, as prefetch)
    /// has its shared context, from the cache or via a job.
    /// Give every stage that units are on (or about to reach) its
    /// destination context. Missing contexts are built here, synchronously
    /// and in parallel on the thread pool (up to `CONTEXT_BUILDS_PER_TICK`
    /// per tick, in key order), so a command's units have their coarse route
    /// on the tick it is interpreted; fields follow as paced worker jobs.
    fn request_stage_contexts(&mut self) {
        // Cohort stages units are on, in parallel over ranges of units.
        let all = &self.units;
        let used: BTreeSet<(u32, u32)> = self.par.ranges(all.len(), |a, b| {
            let mut used: BTreeSet<(u32, u32)> = BTreeSet::new();
            for u in &all[a..b] {
                if let (Some(cid), false, false) = (u.cohort_id, u.arrived, u.unreachable) { used.insert((cid, u.stage_idx)); }
            }
            used
        }).into_iter().flatten().collect();
        let mut wanted: BTreeSet<(u32, usize)> = BTreeSet::new();
        for (cid, stage_idx) in used {
            let Some(c) = self.cohorts.get(&cid) else { continue; };
            let si = (stage_idx as usize).min(c.stages.len().saturating_sub(1));
            if c.stages[si].fresh() && c.stages.get(si + 1).map(|s| s.fresh()).unwrap_or(true) { continue; }
            wanted.insert((cid, si));
            wanted.insert((cid, si + 1));
        }
        let mut missing: BTreeSet<ContextKey> = BTreeSet::new();
        let mut attached = Attached::default();
        for (cid, si) in wanted {
            let Some(st) = self.cohorts.get(&cid).and_then(|c| c.stages.get(si)) else { continue; };
            if st.fresh() { continue; }
            let mut key = st.ctx_key.clone();
            if key.crowd != 0 && self.crowd_by_version(key.crowd).is_none() {
                // The map the cohort was keyed to was replaced while it
                // waited for its turn (`CROWD_REKEYS_PER_UPDATE`): it needs
                // a context anyway, so it adopts the current map now.
                let cmd = self.cohorts[&cid].command_id;
                let v = self.crowd_for(cmd).version;
                let tick = self.tick;
                let c = self.cohorts.get_mut(&cid).expect("cohort");
                for st in &mut c.stages { st.ctx_key.crowd = v; }
                c.crowd_keyed_tick = tick;
                key.crowd = v;
                if c.stages[si].fresh() { continue; }
            }
            if let Some((_, used)) = self.contexts.get_mut(&key) {
                *used = self.tick;
                self.nav.context_hits += 1;
                self.attach_context(&key, &mut attached);
                continue;
            }
            if key.nav_version != self.hierarchy.version { continue; }
            if key.crowd != 0 && self.crowd_by_version(key.crowd).is_none() { continue; }
            missing.insert(key);
        }
        let keys: Vec<ContextKey> = missing.into_iter().take(CONTEXT_BUILDS_PER_TICK).collect();
        if keys.is_empty() { self.contexts_attached(attached); return; }
        self.nav.context_misses += keys.len() as u64;
        let h = Arc::clone(&self.hierarchy);
        let crowds: Vec<(ContextKey, Option<Arc<CrowdMap>>)> = keys.iter().map(|k| (k.clone(), self.crowd_by_version(k.crowd).cloned())).collect();
        let built = self.par.map(&crowds, |(k, m)| build_context(&h, k, m.as_ref()));
        let tick = self.tick;
        for (key, ctx) in keys.into_iter().zip(built) {
            match ctx {
                Some(ctx) => {
                    self.nav.plans_completed += 1;
                    self.nav.abstract_expansions += ctx.expansions as u64;
                    self.nav.last_plan_ms = ctx.build_ms;
                    self.contexts.insert(key.clone(), (Arc::new(ctx), tick));
                    self.evict_contexts();
                    self.attach_context(&key, &mut attached);
                }
                None => self.fail_context(&key),
            }
        }
        self.contexts_attached(attached);
    }

    // ----- tick -------------------------------------------------------------

    /// Run `f` (typically a loop of ticks) on a thread of the simulation's
    /// own pool, so each parallel section in a tick starts inside the pool.
    /// The result is the same as calling `f(self)` directly.
    pub fn in_pool<R: Send>(&mut self, f: impl FnOnce(&mut Self) -> R + Send) -> R {
        let par = self.par.clone();
        par.install(|| f(self))
    }

    fn advance_tick(&mut self) {
        let start = Instant::now();
        let mut lap = start;
        let mut split = || { let now = Instant::now(); let ms = (now - lap).as_secs_f64() * 1000.0; lap = now; ms };
        let mut ph = TickPhases::default();
        self.tick += 1;
        let tick = self.tick;
        while let Some(p) = self.queued_terrain.pop_front() { self.handle_terrain_patch(p, tick); }
        self.apply_due_jobs();
        ph.jobs_ms = split();
        // Commands are interpreted against an up-to-date hierarchy only; while
        // a terrain repair is outstanding they stay queued (one tick).
        if self.pending_rect.is_none() {
            while let Some(c) = self.queued_commands.pop_front() { self.handle_command(c, tick); }
        }
        if tick % CROWD_UPDATE_TICKS == 0 { self.update_crowd(); }
        ph.commands_ms = split();
        if self.pending_rect.is_none() { self.request_stage_contexts(); }
        ph.contexts_ms = split();
        let nav_driven = self.steer_units();
        ph.avoidance_ms = self.last_avoidance_ms;
        ph.steer_ms = (split() - ph.avoidance_ms).max(0.0);
        self.resolve_physics();
        ph.physics_ms = split();
        self.update_unit_progress(&nav_driven);
        ph.progress_ms = split();
        self.update_command_statuses();
        ph.statuses_ms = split();
        self.submit_requests();
        ph.submit_ms = split();
        // Part of the state hash: kept current whether or not a snapshot is
        // published.
        self.ensure_terrain_digest();
        self.last_tick_ms = start.elapsed().as_secs_f64() * 1000.0;
        if self.publishing { self.publish_snapshot(); }
        ph.snapshot_ms = split();
        self.phases = ph;
    }

    /// Compute desired velocities. Every unit decides in parallel from the
    /// tick's shared navigation state and changes only itself; the few units
    /// that need new cohort slots (on entering a region) get them created
    /// serially in unit order and then decide again. Slots created this tick
    /// are therefore seen by other units from the next tick on, whatever the
    /// thread count.
    fn steer_units(&mut self) -> Vec<bool> {
        let dt = TICK_SECONDS;
        let h = Arc::clone(&self.hierarchy);
        let geom = h.terrain.geom.clone();
        let dirty_pending = self.pending_dirty_sectors();
        let switch = self.cfg.arrival_switch_cells.max(2) as f32 * geom.cell_size;
        let n = self.units.len();
        let mut desired = vec![WorldPos::ZERO; n];
        let mut nav_driven = vec![false; n];
        let mut needy: Vec<usize> = Vec::new();
        let mut outs: Vec<SteerOut> = Vec::new();
        {
            let env = SteerEnv { h: &h, tick: self.tick, dirty_pending: &dirty_pending, switch, cohorts: &self.cohorts, pending_fields: &self.pending_fields, crowd: &self.crowd, own_crowd: &self.own_crowd };
            let parts = self.par.chunks_mut_at(&mut self.units, |start, chunk| {
                let mut out = SteerOut::default();
                let mut chain = Vec::with_capacity(LOOKAHEAD_STEPS);
                let mut steps = Vec::with_capacity(chunk.len());
                let mut needy = Vec::new();
                for (k, u) in chunk.iter_mut().enumerate() {
                    match steer_one(&env, u, &mut out, &mut chain) {
                        Steer::Done(v) => steps.push(v),
                        Steer::NeedSlots => { steps.push(None); needy.push(start + k); }
                    }
                }
                (start, steps, needy, out)
            });
            for (start, steps, part_needy, out) in parts {
                for (k, v) in steps.into_iter().enumerate() {
                    if let Some(v) = v { desired[start + k] = v; nav_driven[start + k] = true; }
                }
                needy.extend(part_needy);
                outs.push(out);
            }
        }
        for &i in &needy { self.create_unit_slots(i, &h); }
        if !needy.is_empty() {
            // Again in parallel: on a command's tick nearly every unit is here.
            let mut flagged = vec![false; n];
            for &i in &needy { flagged[i] = true; }
            let env = SteerEnv { h: &h, tick: self.tick, dirty_pending: &dirty_pending, switch, cohorts: &self.cohorts, pending_fields: &self.pending_fields, crowd: &self.crowd, own_crowd: &self.own_crowd };
            let parts = self.par.chunks_mut_at(&mut self.units, |start, chunk| {
                let mut out = SteerOut::default();
                let mut chain = Vec::with_capacity(LOOKAHEAD_STEPS);
                let mut steps = Vec::new();
                for (k, u) in chunk.iter_mut().enumerate() {
                    if !flagged[start + k] { continue; }
                    match steer_one(&env, u, &mut out, &mut chain) {
                        Steer::Done(v) => steps.push((start + k, v)),
                        Steer::NeedSlots => { debug_assert!(false, "slots were just created"); u.mode = UnitMode::WaitingForNav; }
                    }
                }
                (steps, out)
            });
            for (steps, out) in parts {
                for (i, v) in steps {
                    if let Some(v) = v { desired[i] = v; nav_driven[i] = true; }
                }
                outs.push(out);
            }
        }
        for out in outs {
            self.field_needs.extend(out.needs);
            self.nav.skipped_stages += out.skipped_stages;
            self.nav.stranded_units += out.stranded_units;
            self.nav.local_recenters += out.local_recenters;
        }
        // Yielding rule (see `NavConfig::yield_after_ticks`): 20-tick phases,
        // one in three spent backing off at 30% speed.
        let ya = self.cfg.yield_after_ticks;
        if ya > 0 {
            for i in 0..self.units.len() {
                let u = &mut self.units[i];
                if !nav_driven[i] || u.stuck_ticks < ya { continue; }
                let phase = (u.stuck_ticks / 20 + crate::rng::hash2(0x71E1D, u.id as i32, 0)) % 3;
                if phase == 0 {
                    desired[i] = desired[i].scaled(-0.3);
                    u.mode = UnitMode::Blocked;
                    self.nav.yield_ticks += 1;
                }
            }
        }
        // Avoidance: navigation's desired velocity is the preferred velocity;
        // ORCA returns the nearest velocity that stays clear of neighbours.
        // Every unit takes part, so stationary units make way; an arrived unit
        // then drifts back to its slot, an idle or unreachable one stands
        // where it comes to rest (a unit waiting for navigation stands
        // still). Without this, slots (2.35 radii apart) would be walls that
        // a late unit cannot pass to reach an inner slot.
        let avoiding = self.cfg.avoidance;
        if avoiding {
            let t0 = Instant::now();
            let agents: Vec<Agent> = self.units.iter().enumerate().map(|(i, u)| {
                let home = if u.arrived { u.slot } else { u.home };
                let pref = if nav_driven[i] { desired[i] } else {
                    let d = home.map(|s| s.sub(u.body.pos)).unwrap_or(WorldPos::ZERO);
                    let dl = d.length();
                    if dl > u.body.radius * 0.25 { d.scaled((u.max_speed * 0.5).min(dl * 2.0) / dl) } else { WorldPos::ZERO }
                };
                Agent { pos: u.body.pos, vel: u.body.vel, radius: u.body.radius, max_speed: u.max_speed, pref, avoids: !Self::braking(u), moving: nav_driven[i],
                    horizon: if u.arrival_phase { ARRIVAL_AVOIDANCE_HORIZON } else { 0.0 } }
            }).collect();
            let cfg = AvoidanceConfig { time_horizon: self.cfg.avoidance_time_horizon, max_neighbors: self.cfg.avoidance_neighbors };
            let v = compute_velocities(&agents, &cfg, dt, &self.par);
            desired.copy_from_slice(&v);
            self.last_avoidance_ms = t0.elapsed().as_secs_f64() * 1000.0;
        }
        // Velocity update with explicit acceleration limits.
        for i in 0..self.units.len() {
            let u = &mut self.units[i];
            let max_dv = u.accel * dt;
            if Self::braking(u) {
                // A stopped unit brakes at full rate, outside avoidance:
                // sharing it with units still streaming behind would carry
                // it along for seconds. Neighbours avoid it instead.
                let vlen = u.body.vel.length();
                u.body.vel = if vlen > max_dv { u.body.vel.scaled((vlen - max_dv) / vlen) } else { WorldPos::ZERO };
                continue;
            }
            if avoiding && !nav_driven[i] {
                // Making way (or returning to the slot) at a gentle rate.
                let delta = desired[i].sub(u.body.vel);
                let lim = max_dv * 0.5;
                u.body.vel = u.body.vel.add(if delta.length() > lim { delta.normalized_or(WorldPos::ZERO).scaled(lim) } else { delta });
                // A velocity that cannot move the body (a whole tick of it is
                // lost to f32 rounding of the position) is zero. Avoidance
                // separates still units that overlap by a hair at such a
                // crawl; at map coordinates around 17,000 the overlap never
                // closed, and three such units of 20,000 kept the crowd from
                // ever being at rest, so the avoidance and physics rest
                // shortcuts never ran. Slower crawls that do move a unit
                // are kept: zeroing them stranded units at a filled chokepoint.
                let p = u.body.pos;
                if p.x + u.body.vel.x * dt == p.x && p.y + u.body.vel.y * dt == p.y { u.body.vel = WorldPos::ZERO; }
                continue;
            }
            if u.arrived || u.unreachable { u.body.vel = WorldPos::ZERO; continue; }
            if !nav_driven[i] {
                u.desired_speed = 0.0;
                let vlen = u.body.vel.length();
                if vlen > 1e-4 { u.body.vel = u.body.vel.scaled(((vlen - max_dv) / vlen).max(0.0)); }
                continue;
            }
            let delta = desired[i].sub(u.body.vel);
            let delta = if delta.length() > max_dv { delta.normalized_or(WorldPos::ZERO).scaled(max_dv) } else { delta };
            u.body.vel = u.body.vel.add(delta);
        }
        nav_driven
    }

    /// Create the cohort slots unit `i` was missing in `steer_one`: its
    /// region's and, on entering the region, those it prefetches along the
    /// abstract route. A slot another unit created first is reused.
    fn create_unit_slots(&mut self, i: usize, h: &NavHierarchy) {
        let tick = self.tick;
        let geom = &h.terrain.geom;
        let u = &self.units[i];
        let (Some(cid), last_region, pos, stage_idx) = (u.cohort_id, u.last_region, u.body.pos, u.stage_idx) else { return; };
        let Some(c) = self.cohorts.get_mut(&cid) else { return; };
        let pid = c.profile_id;
        let Some(nav_cell) = Self::nav_cell_of(h, pid, pos) else { return; };
        let si = (stage_idx as usize).min(c.stages.len() - 1);
        let st = &mut c.stages[si];
        let Some(ctx) = st.ctx.clone() else { return; };
        if ctx.key != st.ctx_key { return; }
        let Some(region) = h.region_at(pid, nav_cell) else { return; };
        let dist = pos.dist(geom.cell_center(st.target)) / geom.cell_size;
        let (cache, needs) = (&mut self.field_cache, &mut self.field_needs);
        let target = st.target;
        let mut make = |r: (u32, u16), prio: u64| field_recipe(h, &ctx, r.0, r.1, target).map(|rc| {
            let field = cache.get(&rc.key, tick);
            if field.is_none() { needs.insert((prio, rc.key.clone())); }
            FieldSlot { key: rc.key, offset: rc.offset, next_node: rc.best_exit_far_node, field }
        });
        if !st.slots.contains_key(&region) {
            let slot = make(region, field_priority(PRIORITY_NEEDED, dist));
            st.slots.insert(region, slot);
        }
        let Some(Some(far)) = st.slots.get(&region).map(|s| s.as_ref().and_then(|s| s.next_node)) else { return; };
        if last_region == Some(region) { return; }
        for r in regions_ahead(h.hierarchy(pid), &ctx, far, PREFETCH_SECTORS) {
            if st.slots.contains_key(&r) { continue; }
            let slot = make(r, field_priority(PRIORITY_PREFETCH, dist));
            st.slots.insert(r, slot);
        }
    }

    fn resolve_physics(&mut self) {
        if self.units.is_empty() { return; }
        let geom = self.hierarchy.terrain.geom.clone();
        for u in &mut self.units { u.last_pos = u.body.pos; }
        let max_vel = self.units.iter().map(|u| u.body.vel.length()).fold(0.0f32, f32::max);
        // Tunnelling safeguard: no body moves more than 0.45 cells per
        // substep. Speeds beyond 12 substeps' worth are clamped.
        let safe_step = geom.cell_size * 0.45;
        let max_substeps = 12u32;
        let substeps = ((max_vel * TICK_SECONDS / safe_step).ceil() as u32).clamp(1, max_substeps);
        let supported = safe_step * max_substeps as f32 / TICK_SECONDS;
        if max_vel > supported {
            let factor = supported / max_vel;
            for u in &mut self.units { u.body.vel = u.body.vel.scaled(factor); }
        }
        self.contacts.clear();
        if let Some((rest, version, stats)) = &self.physics_rest {
            if *version == self.terrain.version && rest.len() == self.units.len()
                && self.units.iter().zip(rest).all(|(u, b)| same_body(&u.body, b)) {
                self.collision_stats = CollisionStats { solver_ms: 0.0, integrate_ms: 0.0, broadphase_ms: 0.0, pair_solve_ms: 0.0,
                    avoidance_ms: self.last_avoidance_ms, ..stats.clone() };
                return;
            }
        }
        let ids: Vec<u32> = self.units.iter().map(|u| u.id).collect();
        let mut bodies: Vec<Body> = self.units.iter().map(|u| u.body).collect();
        let dt = TICK_SECONDS / substeps as f32;
        // On a map with water each body also collides with the cells its
        // profile may not enter; a dry map takes pathsim's path unchanged.
        let media: Vec<crate::terrain::Medium> = self.profiles.iter().map(|p| p.medium).collect();
        let class_of: Vec<u8> = if self.terrain.has_water() { self.units.iter().map(|u| u.profile_id).collect() } else { Vec::new() };
        let body_media = self.terrain.has_water().then(|| crate::collision::BodyMedia { media: &media, class_of: &class_of });
        let mut total = self.collider.step_in_media(&ids, &mut bodies, &self.terrain, body_media, &self.collision_cfg, dt, substeps,
            &self.par, &mut self.contacts, SNAPSHOT_CONTACT_LIMIT);
        total.avoidance_ms = self.last_avoidance_ms;
        let at_rest = self.units.iter().zip(&bodies).all(|(u, b)| same_body(&u.body, b));
        self.physics_rest = at_rest.then(|| (bodies.clone(), self.terrain.version, total.clone()));
        self.collision_stats = total;
        for (u, b) in self.units.iter_mut().zip(bodies) { u.body = b; }
    }

    /// Movement latency, progress tracking and the blocked-unit policy:
    /// * no progress for `stuck_settle_ticks` while inside the arrival region
    ///   (a few radii from the slot), or anywhere inside its gathering (no
    ///   farther from the goal than its slot): settle there, report arrived.
    ///   A unit shoved off its slot by others streaming past otherwise
    ///   fought them for a minute, gave up and stood wherever it ended up;
    /// * no progress for `stuck_give_up_ticks`: the timer restarts up to
    ///   `STUCK_RETRIES` times, then the unit gives up and reports
    ///   unreachable (exposed in stats as `stuck_gave_up`). An unreachable
    ///   goal is found by the planner, not here; what stalls a unit for a
    ///   minute is mostly a jam (a 50,000-unit column at a chokepoint), and
    ///   units that gave up stood in it and made it worse.
    ///
    /// Collision response alone does not resolve congestion; this policy makes
    /// the outcome explicit rather than leaving units spinning forever.
    fn update_unit_progress(&mut self, nav_driven: &[bool]) {
        let tick = self.tick;
        let geom = self.hierarchy.terrain.geom.clone();
        for i in 0..self.units.len() {
            let goal = self.units[i].cohort_id.and_then(|c| self.cohorts.get(&c)).and_then(|c| c.stages.last()).map(|s| geom.cell_center(s.target));
            let moved = self.units[i].body.pos.dist(self.units[i].last_pos);
            if nav_driven[i] && moved > self.cfg.cell_size * 0.004 {
                if let Some(cid) = self.units[i].command_id {
                    if let Some(rec) = self.commands.get_mut(&cid) {
                        if rec.first_move_tick.is_none() {
                            rec.first_move_tick = Some(tick);
                            let ms = self.command_received_at.get(&cid).map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
                            rec.first_move_wall_ms = Some(ms);
                            self.latency_ticks.push(tick.saturating_sub(rec.commit_tick));
                            self.latency_ms.push(ms);
                        }
                    }
                }
            }
            let u = &mut self.units[i];
            // A unit without a slot (stopped or unreachable) stands where it
            // comes to rest, and again wherever a neighbour nudges it to:
            // homes fixed in a packed crowd can overlap, and units walking
            // back to them would jostle the crowd indefinitely.
            let at_rest = u.body.vel.length() < u.max_speed * 0.02;
            if u.unreachable {
                u.mode = UnitMode::Unreachable;
                if at_rest { u.home = Some(u.body.pos); }
                continue;
            }
            if u.arrived { u.mode = UnitMode::Arrived; continue; }
            if u.command_id.is_none() {
                u.mode = UnitMode::Idle;
                if at_rest { u.home = Some(u.body.pos); }
                continue;
            }
            if u.cohort_id.is_none() { continue; }
            if u.arrival_phase {
                if let Some(s) = u.slot {
                    let d = (s.dist(u.body.pos) * 100.0) as u64;
                    if d < u.best_slot_dist { u.best_slot_dist = d; u.stuck_ticks = 0; }
                }
            }
            // Waiting for navigation data is not being stuck.
            if !nav_driven[i] { continue; }
            u.stuck_ticks = u.stuck_ticks.saturating_add(1);
            if u.stuck_ticks > 45 && u.mode == UnitMode::Moving { u.mode = UnitMode::Blocked; }
            let settle = u.body.radius * 4.0 + self.cfg.cell_size * 2.0;
            let in_arrival_region = u.arrival_phase && u.slot.map(|s| s.dist(u.body.pos) <= settle).unwrap_or(false);
            let in_gathering = u.arrival_phase && match (u.slot, goal) {
                (Some(s), Some(g)) => u.body.pos.dist(g) <= s.dist(g) + settle,
                _ => false,
            };
            if (in_arrival_region || in_gathering) && u.stuck_ticks >= self.cfg.stuck_settle_ticks {
                u.arrived = true; u.settled = true; u.arrived_tick = Some(tick); u.mode = UnitMode::Arrived; u.body.vel = WorldPos::ZERO;
                self.nav.stuck_settled += 1;
            } else if u.stuck_ticks >= self.cfg.stuck_give_up_ticks {
                if u.stuck_retries < STUCK_RETRIES {
                    u.stuck_retries += 1;
                    u.stuck_ticks = 0;
                    self.nav.stuck_retries += 1;
                } else {
                    Self::mark_unreachable(u);
                    self.nav.stuck_gave_up += 1;
                }
            }
        }
    }

    fn update_command_statuses(&mut self) {
        // Counted in parallel over ranges of units and merged (sums: the
        // same in any order).
        let all = &self.units;
        let parts = self.par.ranges(all.len(), |a, b| {
            let mut tallies: BTreeMap<u64, (u32, u32, u32)> = BTreeMap::new();
            let mut cohort_active: BTreeMap<u32, u32> = BTreeMap::new();
            for u in &all[a..b] {
                if let Some(cid) = u.command_id {
                    let e = tallies.entry(cid).or_insert((0, 0, 0));
                    e.0 += 1;
                    e.1 += u.arrived as u32;
                    e.2 += u.unreachable as u32;
                }
                if let (Some(c), false, false) = (u.cohort_id, u.arrived, u.unreachable) { *cohort_active.entry(c).or_insert(0) += 1; }
            }
            (tallies, cohort_active)
        });
        let mut tallies: BTreeMap<u64, (u32, u32, u32)> = BTreeMap::new();
        let mut cohort_active: BTreeMap<u32, u32> = BTreeMap::new();
        for (t, c) in parts {
            for (k, v) in t { let e = tallies.entry(k).or_insert((0, 0, 0)); e.0 += v.0; e.1 += v.1; e.2 += v.2; }
            for (k, v) in c { *cohort_active.entry(k).or_insert(0) += v; }
        }
        // Cohorts with no active unit are reclaimed (their shared fields and
        // contexts live on in the bounded caches).
        for c in self.cohorts.values_mut() { c.active_units = cohort_active.get(&c.id).copied().unwrap_or(0); }
        self.cohorts.retain(|_, c| c.active_units > 0);
        let ids: Vec<u64> = self.command_order.iter().copied().collect();
        for id in ids {
            let Some(rec) = self.commands.get_mut(&id) else { continue; };
            let (total, arrived, unreachable) = tallies.get(&id).copied().unwrap_or((0, rec.arrived_units, rec.unreachable_units));
            let terminal = matches!(rec.status, CommandStatus::Completed | CommandStatus::PartlyUnreachable | CommandStatus::Unreachable);
            if total == 0 {
                // Every unit was re-commanded before finishing.
                if !terminal { rec.status = CommandStatus::Interrupted; rec.active_units = 0; }
                continue;
            }
            rec.arrived_units = arrived;
            rec.unreachable_units = unreachable;
            rec.active_units = total.saturating_sub(arrived + unreachable);
            let planning = rec.cohort_ids.iter().any(|cid| self.cohorts.get(cid).map(|c| c.stages.first().map(|s| s.ctx.is_none()).unwrap_or(false)).unwrap_or(false));
            rec.status = if arrived + unreachable >= total {
                if unreachable == 0 { CommandStatus::Completed }
                else if arrived == 0 { CommandStatus::Unreachable }
                else { CommandStatus::PartlyUnreachable }
            } else if self.pending_rect.is_some() {
                CommandStatus::Repairing
            } else if planning && arrived == 0 {
                CommandStatus::Planning
            } else if rec.truncated {
                CommandStatus::TruncatedCurve
            } else {
                CommandStatus::Executing
            };
        }
        // Latency summary and gauges (instrumentation only).
        let mut lt = self.latency_ticks.clone();
        lt.sort_unstable();
        let mut lm = self.latency_ms.clone();
        lm.sort_by(|a, b| a.total_cmp(b));
        self.nav.command_latency = if lt.is_empty() { LatencySummary::default() } else {
            let at = |n: usize, p: f64| (((n - 1) as f64) * p).round() as usize;
            LatencySummary {
                samples: lt.len() as u32,
                median_ticks: lt[at(lt.len(), 0.5)] as f64, p95_ticks: lt[at(lt.len(), 0.95)] as f64, worst_ticks: *lt.last().unwrap(),
                median_ms: lm[at(lm.len(), 0.5)], p95_ms: lm[at(lm.len(), 0.95)], worst_ms: *lm.last().unwrap(),
            }
        };
        // Cohort stages in use, in parallel over ranges of units.
        let (all, cohorts) = (&self.units, &self.cohorts);
        let parts = self.par.ranges(all.len(), |a, b| {
            let mut used: BTreeSet<(u32, u32)> = BTreeSet::new();
            for u in &all[a..b] {
                if let (Some(cid), false, false) = (u.cohort_id, u.arrived, u.unreachable) { used.insert((cid, u.stage_idx)); }
            }
            used
        });
        let mut groups = BTreeSet::new();
        let mut profiles = BTreeSet::new();
        for (cid, si) in parts.into_iter().flatten().collect::<BTreeSet<_>>() {
            if let Some(c) = cohorts.get(&cid) {
                if let Some(st) = c.stages.get(si as usize) { groups.insert(st.ctx_key.clone()); }
                profiles.insert(c.profile_id);
            }
        }
        self.nav.unique_destination_groups = groups.len();
        self.nav.active_profiles = profiles.len();
        self.nav.active_cohorts = self.cohorts.len();
        self.nav.pending_jobs = self.pending.len();
        self.nav.cache_hits = self.field_cache.stats.hits;
        self.nav.cache_misses = self.field_cache.stats.misses;
        self.nav.cache_fields = self.field_cache.len();
        self.nav.cache_bytes = self.field_cache.bytes() + self.contexts.values().map(|(c, _)| c.labels.len() * 12).sum::<usize>();
        self.nav.cache_evictions = self.field_cache.stats.evictions;
        self.nav.contexts_cached = self.contexts.len();
        self.nav.hierarchy_bytes = self.hierarchy.bytes();
        let busy = self.pool.busy_ns() as f64 / 1_000_000.0;
        self.nav.nav_busy_ms_total = busy;
        let wall = self.started.elapsed().as_secs_f64() * 1000.0;
        self.nav.worker_utilization = if wall > 0.0 { busy / (wall * self.cfg.nav_worker_count.max(1) as f64) } else { 0.0 };
    }

    /// Digest of authoritative state. Positions/velocities are hashed by their
    /// exact f32 bit patterns; navigation integers directly.
    pub fn state_hash(&self) -> u64 { self.state_hash_parts().combined }

    /// Per-part digests so a replay divergence report can say what split.
    pub fn state_hash_parts(&self) -> StateHashParts {
        let mut uh = StateHasher::new();
        for u in &self.units {
            uh.u32(u.id);
            uh.f32_bits(u.body.pos.x); uh.f32_bits(u.body.pos.y);
            uh.f32_bits(u.body.vel.x); uh.f32_bits(u.body.vel.y);
        }
        let mut nh = StateHasher::new();
        for u in &self.units {
            nh.u64(u.command_id.unwrap_or(0)); nh.u32(u.cohort_id.unwrap_or(0)); nh.u32(u.stage_idx);
            nh.u32(u.arrived as u32 | (u.unreachable as u32) << 1 | (u.arrival_phase as u32) << 2 | (u.settled as u32) << 3);
            nh.u32(u.stuck_ticks);
            if let Some(s) = u.slot_cell { nh.i32(s.x); nh.i32(s.y); }
        }
        let mut ch = StateHasher::new();
        for (id, c) in &self.commands {
            ch.u64(*id); ch.u64(c.commit_tick); ch.u32(c.status as u32);
            ch.u32(c.total_units); ch.u32(c.arrived_units); ch.u32(c.unreachable_units);
        }
        ch.u64(self.cohorts.len() as u64);
        let mut h = StateHasher::new();
        h.u64(self.tick);
        h.u32(self.terrain.version);
        h.u64(self.terrain_digest.1);
        h.u32(self.hierarchy.version);
        h.u64(uh.clone().finish()); h.u64(nh.clone().finish()); h.u64(ch.clone().finish());
        StateHashParts {
            combined: h.finish(), bodies: uh.finish(), unit_nav: nh.finish(), commands: ch.finish(),
            terrain: self.terrain_digest.1, nav_version: self.hierarchy.version,
        }
    }

    pub fn stats(&self) -> SimStats {
        let mut out = SimStats {
            tick: self.tick, units: self.units.len(), terrain_version: self.terrain.version,
            last_terrain_edit_tick: self.last_terrain_edit_tick, awaiting_hierarchy: self.pending_rect.is_some(),
            nav_refresh_ms: self.nav.last_repair_ms,
            tick_ms: self.last_tick_ms, collision: self.collision_stats.clone(), nav: self.nav.clone(),
            phases: self.phases.clone(),
            ..Default::default()
        };
        for u in &self.units {
            out.settled += u.settled as usize;
            match u.mode {
                UnitMode::Idle => out.idle += 1,
                UnitMode::WaitingForNav => out.waiting += 1,
                UnitMode::Moving => out.moving += 1,
                UnitMode::Arrived => out.arrived += 1,
                UnitMode::Blocked => out.blocked += 1,
                UnitMode::Unreachable => out.unreachable += 1,
            }
        }
        out.nav.worker_count = self.cfg.nav_worker_count;
        out.nav.fallback_slots = self.fallback_slots.get();
        out
    }

    fn unit_field<'a>(cohorts: &'a BTreeMap<u32, Cohort>, h: &NavHierarchy, u: &Unit) -> Option<(&'a Arc<FlowField>, u64)> {
        let c = cohorts.get(&u.cohort_id?)?;
        let st = c.stages.get(u.stage_idx as usize)?;
        let cell = Self::nav_cell_of(h, u.profile_id, u.body.pos)?;
        let region = h.region_at(u.profile_id, cell)?;
        let slot = st.slots.get(&region)?.as_ref()?;
        Some((slot.field.as_ref()?, slot.offset))
    }

    pub fn publish_snapshot(&mut self) {
        self.ensure_terrain_digest();
        let stats = self.stats();
        let geom = self.hierarchy.terrain.geom.clone();
        // In parallel over ranges of units; each range also lists the first
        // 32 distinct fields it uses, merged in order below, so the debug
        // fields are the first 32 distinct ones in unit order as before.
        let (all, cohorts, h) = (&self.units, &self.cohorts, &*self.hierarchy);
        let parts = self.par.ranges(all.len(), |a, b| {
            let mut units = Vec::with_capacity(b - a);
            let mut fields: Vec<Arc<FlowField>> = Vec::new();
            for u in &all[a..b] {
                let field = if u.arrived || u.unreachable { None } else { Self::unit_field(cohorts, h, u) };
                if let Some((f, _)) = field {
                    if fields.len() < 32 && !fields.iter().any(|g| g.key == f.key) { fields.push(Arc::clone(f)); }
                }
                let cell = geom.world_to_cell(u.body.pos);
                units.push(UnitSnapshot {
                    id: u.id, x: u.body.pos.x, y: u.body.pos.y, vx: u.body.vel.x, vy: u.body.vel.y,
                    radius: u.body.radius, profile_id: u.profile_id, command_id: u.command_id,
                    cohort_id: u.cohort_id, stage_idx: u.stage_idx, arrival_phase: u.arrival_phase,
                    slot_x: u.slot.map(|p| p.x), slot_y: u.slot.map(|p| p.y), stuck_ticks: u.stuck_ticks,
                    field_sector: field.map(|(f, _)| f.sector),
                    field_cost: field.map(|(f, off)| f.cost_at(cell).saturating_add(off)),
                    mode: u.mode, arrived: u.arrived, unreachable: u.unreachable, desired_speed: u.desired_speed,
                });
            }
            (units, fields)
        });
        let mut units = Vec::with_capacity(self.units.len());
        let mut debug_fields: Vec<Arc<FlowField>> = Vec::new();
        let mut seen: BTreeSet<FieldKey> = BTreeSet::new();
        for (part, fields) in parts {
            units.extend(part);
            for f in fields {
                if debug_fields.len() < 32 && !seen.contains(&f.key) { seen.insert(f.key.clone()); debug_fields.push(f); }
            }
        }
        // Units currently under each command: re-commanded units no longer
        // show their old slots, and a command with none left draws nothing.
        let mut assigned: BTreeMap<u64, (u32, u32)> = BTreeMap::new();
        let mut run: Option<(u64, u32, u32)> = None;
        for u in &self.units {
            // Units of a command mostly sit next to each other: count runs.
            match (&mut run, u.command_id) {
                (Some((c, n, settled)), Some(id)) if *c == id => { *n += 1; *settled += u.settled as u32; }
                (r, id) => {
                    if let Some((c, n, settled)) = r.take() { let e = assigned.entry(c).or_insert((0, 0)); e.0 += n; e.1 += settled; }
                    *r = id.map(|id| (id, 1, u.settled as u32));
                }
            }
        }
        if let Some((c, n, settled)) = run { let e = assigned.entry(c).or_insert((0, 0)); e.0 += n; e.1 += settled; }
        let still_under = |cmd: u64, uid: u32| self.unit_index(uid).is_some_and(|i| self.units[i].command_id == Some(cmd));
        let commands = self.command_order.iter().filter_map(|id| self.commands.get(id)).map(|rec| CommandSnapshot {
            id: rec.payload.id, commit_tick: rec.commit_tick, status: rec.status,
            kind: match &rec.payload.kind {
                CommandKind::MoveTo => "point",
                CommandKind::FollowCurve { .. } => "route",
                CommandKind::CurveFormation { .. } => "formation",
                CommandKind::Stop => "stop",
            }.to_string(),
            requested_x: rec.payload.requested_point.x, requested_y: rec.payload.requested_point.y,
            curve_points: match &rec.payload.kind {
                CommandKind::MoveTo | CommandKind::Stop => Vec::new(),
                CommandKind::FollowCurve { guide_points } | CommandKind::CurveFormation { guide_points } => guide_points.iter().map(|p| (p.x, p.y)).collect(),
            },
            slots: rec.slots.iter().filter(|s| still_under(rec.payload.id, s.unit_id))
                .map(|s| SlotsSnapshot { unit_id: s.unit_id, x: s.position.x, y: s.position.y }).collect(),
            assigned_units: assigned.get(&rec.payload.id).map_or(0, |a| a.0),
            total_units: rec.total_units, arrived_units: rec.arrived_units, unreachable_units: rec.unreachable_units,
            settled_units: assigned.get(&rec.payload.id).map_or(0, |a| a.1),
            active_units: rec.active_units, replaced_units: rec.replaced_units,
            first_move_latency_ticks: rec.first_move_tick.map(|t| t.saturating_sub(rec.commit_tick)),
            first_move_latency_ms: rec.first_move_wall_ms,
            route_polylines: rec.route_polylines.clone(), truncated: rec.truncated, skipped_guides: rec.skipped_guides,
            corridor_tolerance_world: rec.payload.corridor_tolerance_world,
            repair_rerouted_units: rec.repair_rerouted_units, repair_invalidated_units: rec.repair_invalidated_units,
        }).collect();
        let cohorts = self.cohorts.values().map(|c| CohortSnapshot {
            id: c.id, command_id: c.command_id, profile_id: c.profile_id,
            status: if c.stages.iter().any(|s| s.ctx.is_some() && !s.fresh()) { "repairing" }
                else if c.stages.first().map(|s| s.ctx.is_none()).unwrap_or(true) { "planning" } else { "active" },
            active_units: c.active_units, stage_count: c.stages.len() as u32,
            field_slots: c.stages.iter().map(|s| s.slots.len() as u32).sum(),
            fields_ready: c.stages.iter().map(|s| s.slots.values().flatten().filter(|f| f.field.is_some()).count() as u32).sum(),
        }).collect();
        let pending_jobs = self.pending.iter().take(32).map(|(id, p)| PendingJobSnapshot { id: *id, kind: p.label, due_tick: p.due_tick, stale: p.stale }).collect();
        let pending_dirty = self.pending_rect.map(|(x0, y0, x1, y1)| Rect {
            min_x: x0 as f32 * geom.cell_size, min_y: y0 as f32 * geom.cell_size,
            max_x: (x1 + 1) as f32 * geom.cell_size, max_y: (y1 + 1) as f32 * geom.cell_size,
        });
        let snap = SimSnapshot {
            tick: self.tick, units, commands, cohorts, stats,
            hierarchy: Arc::clone(&self.hierarchy), terrain: Arc::clone(&self.terrain),
            debug_fields, contacts: self.contacts.clone(), pending_jobs, pending_dirty,
            nav_stats: self.hierarchy.stats(), state_hash: self.state_hash(), published_at: Instant::now(),
        };
        self.last_snapshot = Some(Arc::new(snap));
        // The previous snapshot may have held the last other reference.
        let (free, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retired_hierarchies).into_iter().partition(|h| Arc::strong_count(h) == 1);
        self.retired_hierarchies = keep;
        for h in free { let _ = self.dropper.send(h); }
    }

    pub fn snapshot(&self) -> Option<&SimSnapshot> { self.last_snapshot.as_deref() }
    pub fn snapshot_arc(&self) -> Option<Arc<SimSnapshot>> { self.last_snapshot.clone() }
    pub fn field_cache_stats(&self) -> FieldCacheStats { self.field_cache.stats.clone() }
    /// Due ticks of all outstanding worker jobs, in job order.
    pub fn pending_job_due_ticks(&self) -> Vec<(&'static str, u64)> {
        self.pending.values().filter(|p| !p.stale).map(|p| (p.label, p.due_tick)).collect()
    }
}

/// Read-only inputs to every unit's steering decision.
struct SteerEnv<'a> {
    h: &'a NavHierarchy,
    tick: u64,
    dirty_pending: &'a BTreeSet<u32>,
    switch: f32,
    cohorts: &'a BTreeMap<u32, Cohort>,
    pending_fields: &'a BTreeMap<FieldKey, u64>,
    crowd: &'a CrowdMap,
    own_crowd: &'a BTreeMap<u64, Arc<CrowdMap>>,
}

/// Shared-state effects of steering some units, applied after the pass.
#[derive(Default)]
struct SteerOut {
    needs: Vec<(u64, FieldKey)>,
    skipped_stages: u64,
    stranded_units: u64,
    local_recenters: u64,
}

enum Steer {
    /// The desired velocity when navigation drives the unit this tick.
    Done(Option<WorldPos>),
    /// The unit's stage has no slot for its region, or for a region it
    /// prefetches on entering it. Nothing was changed past the stage gates.
    NeedSlots,
}

/// One unit's steering decision; changes only `u`.
fn steer_one(env: &SteerEnv, u: &mut Unit, out: &mut SteerOut, chain: &mut Vec<CellPos>) -> Steer {
    let (h, tick, switch) = (env.h, env.tick, env.switch);
    let geom = &h.terrain.geom;
    if u.arrived || u.unreachable || u.command_id.is_none() { u.desired_speed = 0.0; return Steer::Done(None); }
    let Some(cid) = u.cohort_id else { return Steer::Done(None); };
    let Some(c) = env.cohorts.get(&cid) else { return Steer::Done(None); };
    let pid = c.profile_id;
    // Look-ahead lines of a crowd-aware cohort avoid dense standing crowds.
    let crowd = Some(env.own_crowd.get(&c.command_id).map_or(env.crowd, |m| &**m)).filter(|m| c.crowd_aware && m.version != 0);
    let Some(nav_cell) = NavService::nav_cell_of(h, pid, u.body.pos) else { u.mode = UnitMode::Blocked; return Steer::Done(None); };
    let cur_cell = geom.world_to_cell(u.body.pos);
    // Ordered progress gates for curve stages: advance when inside
    // the corridor tolerance of the checkpoint, or when near it and
    // already closer to the next checkpoint than it is. Never back.
    // (Idempotent, so a unit steered twice in one tick advances once.)
    while (u.stage_idx as usize) + 1 < c.stages.len() {
        let st = &c.stages[u.stage_idx as usize];
        let nx = &c.stages[u.stage_idx as usize + 1];
        let t = geom.cell_center(st.target);
        let tol = st.tolerance_cells as f32 * geom.cell_size;
        let d = u.body.pos.dist(t);
        let past = d <= tol * 3.0 && u.body.pos.dist(geom.cell_center(nx.target)) <= t.dist(geom.cell_center(nx.target));
        if d <= tol || past { u.stage_idx += 1; u.best_progress = (u.stage_idx, u64::MAX); u.best_slot_dist = u64::MAX; u.stuck_ticks = 0; } else { break; }
    }
    let si = (u.stage_idx as usize).min(c.stages.len() - 1);
    let is_final = si + 1 == c.stages.len();
    // Exact final arrival: individual slot steering once in range
    // with clearance-aware line of sight.
    if is_final {
        if let (Some(slot), Some(slot_cell)) = (u.slot, u.slot_cell) {
            let dist_slot = u.body.pos.dist(slot);
            // In range of the slot, or already inside the ring of the
            // gathering that the slot belongs to.
            let goal = geom.cell_center(c.stages[si].target);
            let in_ring = u.body.pos.dist(goal) <= slot.dist(goal) + switch * 0.5;
            // A crowd-aware unit heads straight for its slot only when the
            // line does not cross a standing crowd denser than its ends;
            // with slots spread around a crowd the ring is wide, and units
            // cut through the crowd to reach them. Once in final approach
            // only terrain blocks the line: a unit that had walked the field
            // to the goal cell, with its slot behind the gathered crowd,
            // otherwise stood there for good (the field ends at the goal).
            let approach = if u.arrival_phase { slot_line_clear(h, pid, nav_cell, slot_cell, None) }
                else { (dist_slot <= switch || in_ring) && slot_line_clear(h, pid, nav_cell, slot_cell, crowd) };
            if approach {
                u.arrival_phase = true;
                let stop = (geom.cell_size * 0.32).max(u.body.radius * 0.55);
                // Within a cell of the slot and already moving away from it
                // (shoved past it by units streaming behind): arrived; as a
                // standing unit it makes way and drifts back to its slot.
                let to_slot = slot.sub(u.body.pos);
                let passed = dist_slot <= geom.cell_size && u.body.vel.x * to_slot.x + u.body.vel.y * to_slot.y < 0.0;
                if dist_slot <= stop || passed {
                    u.arrived = true; u.arrived_tick = Some(tick); u.mode = UnitMode::Arrived;
                    u.body.vel = WorldPos::ZERO; u.desired_speed = 0.0;
                    return Steer::Done(None);
                }
                let dir = slot.sub(u.body.pos).normalized_or(WorldPos::new(1.0, 0.0));
                let speed = u.max_speed.min(dist_slot * 2.5 + geom.cell_size);
                u.desired_speed = speed;
                u.mode = UnitMode::Moving;
                return Steer::Done(Some(dir.scaled(speed)));
            }
        }
    }
    let st = &c.stages[si];
    let Some(ctx) = st.ctx.as_ref() else { u.mode = UnitMode::WaitingForNav; return Steer::Done(None); };
    let fresh = ctx.key == st.ctx_key;
    let Some(region) = h.region_at(pid, nav_cell) else { u.mode = UnitMode::WaitingForNav; return Steer::Done(None); };
    if env.dirty_pending.contains(&region.0) { u.mode = UnitMode::WaitingForNav; return Steer::Done(None); }
    if !st.slots.contains_key(&region) {
        if !fresh { u.mode = UnitMode::WaitingForNav; return Steer::Done(None); }
        return Steer::NeedSlots;
    }
    let Some(slot) = st.slots.get(&region).and_then(|s| s.as_ref()) else {
        // No route from this region to the stage target (e.g. a unit
        // shoved by collisions into a pocket cut off by slopes).
        if !is_final { u.stage_idx += 1; out.skipped_stages += 1; } else { NavService::mark_unreachable(u); out.stranded_units += 1; }
        return Steer::Done(None);
    };
    // Prefetch fields along the abstract route on entering a region.
    if u.last_region != Some(region) && fresh {
        if let Some(far) = slot.next_node {
            if regions_ahead(h.hierarchy(pid), ctx, far, PREFETCH_SECTORS).iter().any(|r| !st.slots.contains_key(r)) { return Steer::NeedSlots; }
        }
        u.last_region = Some(region);
    }
    let Some(field) = slot.field.as_ref() else {
        // Coarse phase: the region's field is not built yet (paced,
        // at most `FIELD_MAX_DELAY_MS`). On the unit's staggered
        // refresh tick it looks for the farthest visible portal
        // crossing of the abstract route; a unit with none in view
        // gets its field first. Until its first refresh (at most
        // `WAYPOINT_REFRESH_TICKS`) it heads for the next route node
        // without a visibility check: computing every unit's
        // waypoint on the command's tick cost 16 ms at 10k units.
        let refresh = (tick + u.id as u64) % WAYPOINT_REFRESH_TICKS == 0;
        let mut blocked = false;
        if fresh && refresh {
            let aim = if is_final { u.slot_cell.unwrap_or(st.target) } else { st.target };
            u.waypoint = route_waypoint(h, pid, ctx, slot.next_node, nav_cell, aim, crowd, chain).map(|wc| (wc, u.stage_idx));
            blocked = u.waypoint.is_none();
        }
        if !env.pending_fields.contains_key(&slot.key) {
            let class = if blocked { PRIORITY_BLOCKED } else { PRIORITY_NEEDED };
            out.needs.push((field_priority(class, u.body.pos.dist(geom.cell_center(st.target)) / geom.cell_size), slot.key.clone()));
        }
        if u.waypoint.is_none() && fresh && !refresh {
            let ahead = slot.next_node.map(|n| h.hierarchy(pid).nodes[n as usize].cell).unwrap_or(st.target);
            if ahead != nav_cell {
                u.desired_speed = u.max_speed;
                u.mode = UnitMode::Moving;
                return Steer::Done(Some(geom.cell_center(ahead).sub(u.body.pos).normalized_or(WorldPos::new(1.0, 0.0)).scaled(u.max_speed)));
            }
        }
        if let Some((wc, ws)) = u.waypoint {
            let wp = geom.cell_center(wc);
            if ws == u.stage_idx && wp.dist(u.body.pos) > geom.cell_size * 1.5 && line_clear(h, pid, nav_cell, wc) {
                u.desired_speed = u.max_speed;
                u.mode = UnitMode::Moving;
                return Steer::Done(Some(wp.sub(u.body.pos).normalized_or(WorldPos::new(1.0, 0.0)).scaled(u.max_speed)));
            }
        }
        u.mode = UnitMode::WaitingForNav;
        return Steer::Done(None);
    };
    // Progress metric: (stage, absolute cost-to-go).
    let cost = field.cost_at(nav_cell).saturating_add(slot.offset);
    if u.rebase_progress {
        u.rebase_progress = false;
        u.best_progress = (u.stage_idx, cost);
    } else if (u.stage_idx, cost) < u.best_progress { u.best_progress = (u.stage_idx, cost); u.stuck_ticks = 0; }
    let target_pos = if nav_cell != cur_cell {
        // Pushed off the clearance grid: steer back (never teleport).
        out.local_recenters += 1;
        Some(geom.cell_center(nav_cell))
    } else {
        field.best_step_spread(&h.terrain, h.grid(pid), nav_cell, u.id as u64).map(|(next, _, _)| {
            // Steer at the farthest visible cell of the field chain
            // ahead rather than the next grid neighbour, so units cut
            // straight across open ground instead of following the
            // octile staircase. Refined every few ticks.
            // Refreshes are staggered by unit id so a group never
            // recomputes on the same tick; in between, a unit keeps
            // its waypoint (refining at once when it reaches it) or
            // takes the plain field step.
            let far = |w: Option<(CellPos, u32)>| match w {
                Some((wc, ws)) => ws == u.stage_idx && wc != nav_cell
                    && geom.cell_center(wc).dist(u.body.pos) > geom.cell_size * 1.5,
                None => false,
            };
            if (tick + u.id as u64) % WAYPOINT_REFRESH_TICKS == 0 || (u.waypoint.is_some() && !far(u.waypoint)) {
                let route = if fresh && ctx.key.nav_version == h.version { Some((&**ctx, slot.next_node)) } else { None };
                // On the final stage aim at the unit's own slot, so a
                // group fans out to its slots instead of converging
                // on the goal cell.
                let aim = if is_final { u.slot_cell.unwrap_or(st.target) } else { st.target };
                let wc = lookahead_waypoint(h, pid, &st.slots, field, route, nav_cell, aim, crowd, chain).unwrap_or(next);
                // Only far waypoints are kept, so a unit whose view is
                // one cell recomputes on its staggered tick only.
                u.waypoint = Some((wc, u.stage_idx)).filter(|_| far(Some((wc, u.stage_idx))));
            }
            match u.waypoint {
                Some((wc, _)) if far(u.waypoint) => geom.cell_center(wc),
                _ => geom.cell_center(next),
            }
        })
    };
    match target_pos {
        Some(tp) => {
            let dir = tp.sub(u.body.pos).normalized_or(WorldPos::new(1.0, 0.0));
            let speed = if nav_cell != cur_cell { u.max_speed.min(u.body.pos.dist(tp) * 4.0 + geom.cell_size) } else { u.max_speed };
            u.desired_speed = speed;
            u.mode = UnitMode::Moving;
            Steer::Done(Some(dir.scaled(speed)))
        }
        None => {
            // At the field minimum (the stage target cell). An
            // intermediate stage is complete; at the final target
            // every slot is in line of sight by construction, so the
            // unit switches to individual slot steering.
            if !is_final { u.stage_idx += 1; }
            else if u.slot_cell.map(|sc| slot_line_clear(h, pid, nav_cell, sc, None)).unwrap_or(false) { u.arrival_phase = true; }
            u.desired_speed = 0.0;
            u.mode = UnitMode::WaitingForNav;
            Steer::Done(None)
        }
    }
}

/// Steering waypoint for a unit at `start` heading for `target` (the stage
/// target, or the unit's own slot on the final stage): the farthest point
/// in straight-line view among (1) the portal points of the abstract route
/// within `ROUTE_LOOKAHEAD_CELLS`, ending at `target`, or, when none of
/// those is in view, (2) up to `LOOKAHEAD_STEPS` cells of the field chain
/// (ties broken toward `target`, continuing into loaded fields of the next
/// regions). Pulling the string across portal points removes the
/// sector-scale bends that the portal-run seeding puts into fields. The line
/// must not cross terrain dearer than the points it replaces, so shortcuts
/// never trade a road for mud, nor (with `crowd`) a standing crowd denser
/// than they are, so it does not cut back across a crowd the route avoids.
#[allow(clippy::too_many_arguments)]
fn lookahead_waypoint(
    h: &NavHierarchy, pid: u8, slots: &BTreeMap<(u32, u16), Option<FieldSlot>>,
    field: &FlowField, route: Option<(&DestContext, Option<u32>)>,
    start: CellPos, target: CellPos, crowd: Option<&CrowdMap>, chain: &mut Vec<CellPos>,
) -> Option<CellPos> {
    let terrain = &h.terrain;
    let grid = h.grid(pid);
    // (1) Route points: usually one of these is in view, and then the field
    // chain is not needed at all.
    // A route point that does not see the next one is a dead end: the string
    // is pulled through portal runs only, so a crossing can be pinned against
    // a wall that runs on inside the sector, with the route continuing round
    // its far end. Units walked to it and then along the wall, a right-angled
    // detour; the field chain leads round the wall's end directly.
    if let Some((ctx, next_node)) = route {
        route_points(h, pid, ctx, next_node, start, target, chain);
        if let Some(k) = farthest_visible(h, pid, start, chain, crowd) {
            if chain.get(k + 1).is_none_or(|&n| line_clear(h, pid, chain[k], n)) { return Some(chain[k]); }
        }
    }
    // (2) Otherwise up to `LOOKAHEAD_STEPS` cells of the field chain.
    chain.clear();
    let mut f = field;
    let mut at = start;
    while chain.len() < LOOKAHEAD_STEPS {
        let Some(next) = f.best_step_toward(terrain, grid, at, start, target, LOOKAHEAD_TIE_BAND) else { break; };
        chain.push(next);
        at = next;
        if f.idx(next.x, next.y).is_none() {
            let Some(r) = h.region_at(pid, next) else { break; };
            match slots.get(&r) {
                Some(Some(FieldSlot { field: Some(nf), .. })) => f = nf,
                _ => break,
            }
        }
    }
    farthest_visible(h, pid, start, chain, crowd).map(|k| chain[k])
}

/// The farthest point in view among the abstract route's portal crossings
/// (within `ROUTE_LOOKAHEAD_CELLS`) and the target once the route reaches
/// the goal region. Needs only the context, not fields, so it also serves
/// as the coarse route while a unit's field is being built.
#[allow(clippy::too_many_arguments)]
fn route_waypoint(h: &NavHierarchy, pid: u8, ctx: &DestContext, next_node: Option<u32>, start: CellPos, target: CellPos, crowd: Option<&CrowdMap>, chain: &mut Vec<CellPos>) -> Option<CellPos> {
    route_points(h, pid, ctx, next_node, start, target, chain);
    farthest_visible(h, pid, start, chain, crowd).map(|k| chain[k])
}

/// The abstract route's portal crossings within `ROUTE_LOOKAHEAD_CELLS` of
/// `start`, string-pulled through their runs, then the target once the route
/// reaches the goal region.
fn route_points(h: &NavHierarchy, pid: u8, ctx: &DestContext, next_node: Option<u32>, start: CellPos, target: CellPos, chain: &mut Vec<CellPos>) {
    let octile = |c: CellPos| {
        let (dx, dy) = ((c.x - start.x).abs(), (c.y - start.y).abs());
        dx.max(dy) + dx.min(dy) / 2
    };
    chain.clear();
    {
        // Portal nodes ahead on the abstract route (within range), then the
        // target once the route reaches the goal region.
        let ph = h.hierarchy(pid);
        let mut reached_goal = next_node.is_none();
        let mut n = next_node.unwrap_or(u32::MAX);
        let mut nodes = [0u32; 32];
        let mut count = 0;
        while n != u32::MAX && count < nodes.len() {
            if octile(ph.nodes[n as usize].cell) > ROUTE_LOOKAHEAD_CELLS { break; }
            nodes[count] = n;
            count += 1;
            n = ctx.next_hop[n as usize];
            reached_goal = n == u32::MAX;
        }
        let end = if reached_goal && octile(target) <= ROUTE_LOOKAHEAD_CELLS { Some(target) } else { None };
        // Place each portal point backwards from the end: where the line from
        // `start` to the following point crosses that portal's run. The last
        // point aims at the target even when that is out of range (it is not
        // a candidate then), so it is not pinned to its run's middle.
        let mut anchor = Some(target);
        chain.resize(count, start);
        for (k, &n) in nodes[..count].iter().enumerate().rev() {
            let node = &ph.nodes[n as usize];
            let (lo, hi) = downhill_stretch(ph, ctx, nodes[k]);
            let c = anchor.map_or(node.cell, |a| ph.portals[node.portal as usize].crossing_cell(node.a_side, start, a, lo, hi));
            chain[k] = c;
            anchor = Some(c);
        }
        // Tighten: re-place each point between its neighbours (two forward
        // passes), approaching the taut string through the portal runs.
        for _ in 0..2 {
            for k in 0..count {
                let prev = if k == 0 { start } else { chain[k - 1] };
                let nxt = chain.get(k + 1).copied().unwrap_or(target);
                let node = &ph.nodes[nodes[k] as usize];
                let (lo, hi) = downhill_stretch(ph, ctx, nodes[k]);
                chain[k] = ph.portals[node.portal as usize].crossing_cell(node.a_side, prev, nxt, lo, hi);
            }
        }
        if let Some(t) = end { chain.push(t); }
        chain.dedup();
    }
}

/// Field request priority classes (lower is served first).
const PRIORITY_BLOCKED: u64 = 0;
const PRIORITY_NEEDED: u64 = 1;
const PRIORITY_PREFETCH: u64 = 2;
const PRIORITY_CLASS_SHIFT: u32 = 40;

/// Field request priority: the class, then the requesting unit's distance
/// to its stage target in cells, so units closer to their target are
/// served first within a class.
fn field_priority(class: u64, dist_cells: f32) -> u64 {
    (class << PRIORITY_CLASS_SHIFT) | (dist_cells.max(0.0) as u64).min((1 << PRIORITY_CLASS_SHIFT) - 1)
}

/// Sim time at the start of `tick`, in microseconds (fixed tick rate).
/// Smallest inclusive cell rect covering `a` (if any) and `b`.
/// Whether a taken formation cell lies within `r` cells of `c` on both axes
/// and closer than `sqrt(min_sep2)`. Each column is queried over its own
/// `y` window: a range over whole columns scanned every slot along a
/// vertical stretch of the formation.
fn near_taken(taken: &BTreeSet<(i32, i32)>, c: CellPos, r: i32, min_sep2: i32) -> bool {
    (c.x - r..=c.x + r).any(|x| taken.range((x, c.y - r)..=(x, c.y + r))
        .any(|&(x, y)| (x - c.x).pow(2) + (y - c.y).pow(2) < min_sep2))
}

fn union_rect(a: Option<(i32, i32, i32, i32)>, b: (i32, i32, i32, i32)) -> (i32, i32, i32, i32) {
    match a {
        Some(a) => (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)),
        None => b,
    }
}

fn sim_time_us(tick: u64) -> u64 { tick * 1_000_000 / TICK_HZ as u64 }

/// First tick whose sim time is at or after `us`.
fn tick_at_or_after_us(us: u64) -> u64 { (us * TICK_HZ as u64).div_ceil(1_000_000) }

/// Stretch of the boundary line through node `n`'s portal on which a route
/// point may be placed: the node's run extended over touching runs (of
/// neighbouring sectors or regions) whose node on the same side is no
/// farther from the goal. In open terrain the sector-level route is an L
/// (east, then north), and pinning crossings to its own runs made units walk
/// it in axis-aligned streams; the label condition keeps every slide
/// downhill so it cannot send a unit somewhere whose route leads back.
fn downhill_stretch(ph: &ProfileHierarchy, ctx: &DestContext, n: u32) -> (i32, i32) {
    let node = &ph.nodes[n as usize];
    let pi = node.portal as usize;
    let p = &ph.portals[pi];
    let limit = ctx.labels[n as usize];
    let ok = |q: &Portal| {
        let m = if node.a_side { q.node_a } else { q.node_b };
        q.axis == p.axis && q.boundary == p.boundary && ctx.labels[m as usize] <= limit
    };
    let (mut lo, mut hi) = (p.lo, p.hi);
    let mut j = pi;
    while j > 0 && ph.portals[j - 1].hi + 1 == lo && ok(&ph.portals[j - 1]) { j -= 1; lo = ph.portals[j].lo; }
    let mut j = pi;
    while j + 1 < ph.portals.len() && ph.portals[j + 1].lo == hi + 1 && ok(&ph.portals[j + 1]) { j += 1; hi = ph.portals[j].hi; }
    (lo, hi)
}

/// Index of the farthest point of `pts` in straight-line view of `start`,
/// over terrain no dearer than the points up to it and, with `crowd`,
/// through standing crowds no denser than where the unit stands (a unit
/// inside a crowd can get out; one outside does not see through it). Route
/// points do not raise the crowd limit: a portal crossing clamped into a
/// crowd's edge otherwise opened lines straight across it. Visibility
/// along a route is close to monotone, so this is a binary search.
fn farthest_visible(h: &NavHierarchy, pid: u8, start: CellPos, pts: &[CellPos], crowd: Option<&CrowdMap>) -> Option<usize> {
    let terrain = &h.terrain;
    let cost = |c: CellPos| terrain.move_cost[terrain.geom.cell_index(c)];
    let crowd = crowd.map(|m| (m, m.raw_level_at(start)));
    let mut max_cost = [0u8; LOOKAHEAD_STEPS + 1];
    let mut m = cost(start);
    for (k, &c) in pts.iter().enumerate().take(max_cost.len()) { m = m.max(cost(c)); max_cost[k] = m; }
    let pts = &pts[..pts.len().min(max_cost.len())];
    let visible = |k: usize| line_clear_within(h, pid, start, pts[k], max_cost[k], crowd);
    let last = pts.len().checked_sub(1)?;
    if visible(last) { return Some(last); }
    if !visible(0) { return None; }
    let (mut lo, mut hi) = (0usize, last);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if visible(mid) { lo = mid; } else { hi = mid; }
    }
    Some(lo)
}

/// Clearance-aware straight-line check between two cell centres (sampled
/// line with conservative diagonal corner checks).
pub fn line_clear(h: &NavHierarchy, profile_id: u8, a: CellPos, b: CellPos) -> bool {
    line_clear_within(h, profile_id, a, b, u8::MAX, None)
}

/// `line_clear`, and with `crowd` no crossing of a standing crowd denser
/// (raw level) than either end.
fn slot_line_clear(h: &NavHierarchy, pid: u8, a: CellPos, b: CellPos, crowd: Option<&CrowdMap>) -> bool {
    let level = crowd.map(|m| (m, m.raw_level_at(a).max(m.raw_level_at(b))));
    line_clear_within(h, pid, a, b, u8::MAX, level)
}

/// `line_clear` that also rejects cells whose move cost exceeds `max_cost`
/// and, with `crowd = (map, level)`, cells whose raw crowd level (no
/// margin) exceeds `level`.
fn line_clear_within(h: &NavHierarchy, profile_id: u8, a: CellPos, b: CellPos, max_cost: u8, crowd: Option<(&CrowdMap, u8)>) -> bool {
    let grid = h.grid(profile_id);
    let terrain = &h.terrain;
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let steps = dx.abs().max(dy.abs()).max(1);
    let mut prev = a;
    for i in 0..=steps {
        let c = CellPos::new(a.x + dx * i / steps, a.y + dy * i / steps);
        if !grid.is_passable(terrain, c) || terrain.move_cost[terrain.geom.cell_index(c)] > max_cost { return false; }
        if crowd.is_some_and(|(m, level)| m.raw_level_at(c) > level) { return false; }
        if prev.x != c.x && prev.y != c.y
            && (!grid.is_passable(terrain, CellPos::new(c.x, prev.y)) || !grid.is_passable(terrain, CellPos::new(prev.x, c.y))) {
            return false;
        }
        prev = c;
    }
    true
}

/// Helper used by benchmarks/UI to build a circular terrain edit.
pub fn brush_patch(cell: CellPos, radius: i32, blocked: Option<bool>, height: Option<i16>, cost: Option<u8>) -> TerrainPatch {
    let mut p = TerrainPatch::new();
    for y in cell.y - radius..=cell.y + radius {
        for x in cell.x - radius..=cell.x + radius {
            if (x - cell.x).pow(2) + (y - cell.y).pow(2) <= radius * radius {
                p.push(CellEdit {
                    x, y, blocked,
                    height: height.map(|h| h + (((x * 31 + y * 17) & 7) as i16 * 4)),
                    cost,
                    material: if blocked == Some(true) { Some(crate::terrain::Material::Rock as u8) }
                             else if blocked == Some(false) { Some(crate::terrain::Material::Grass as u8) } else { None },
                });
            }
        }
    }
    p.finish();
    p
}

/// Bit-identical bodies (so -0.0 and 0.0 differ, as they do in the state
/// hash).
fn same_body(a: &Body, b: &Body) -> bool {
    [a.pos.x, a.pos.y, a.vel.x, a.vel.y, a.radius].map(f32::to_bits) == [b.pos.x, b.pos.y, b.vel.x, b.vel.y, b.radius].map(f32::to_bits)
}
