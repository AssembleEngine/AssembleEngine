//! Reciprocal velocity-obstacle avoidance (ORCA, van den Berg et al. 2011),
//! ported from the RVO2 reference algorithm for agent-agent constraints.
//!
//! Each tick, every avoiding agent turns its preferred velocity (from
//! navigation) into the closest velocity that keeps it collision-free with
//! its nearest neighbours for `time_horizon` seconds, assuming they do the
//! same. Static obstacles are not modelled here: navigation keeps routes
//! clear of rock and the collision pass resolves the rest.
//!
//! Determinism: an agent's result depends only on the tick's input arrays,
//! and neighbours are chosen and ordered by (distance², index). Agents can
//! therefore be solved in any order or on any number of threads with
//! bit-identical results.
use crate::math::WorldPos;
use crate::par::Par;

#[derive(Clone, Copy, Debug)]
pub struct AvoidanceConfig {
    /// Seconds ahead that avoidance guarantees freedom from collisions.
    pub time_horizon: f32,
    /// Neighbours considered per agent (nearest first).
    pub max_neighbors: usize,
}

impl Default for AvoidanceConfig {
    fn default() -> Self { Self { time_horizon: 1.0, max_neighbors: 10 } }
}

/// One agent's state for this tick.
#[derive(Clone, Copy, Debug)]
pub struct Agent {
    pub pos: WorldPos,
    pub vel: WorldPos,
    pub radius: f32,
    pub max_speed: f32,
    /// Preferred velocity (navigation output).
    pub pref: WorldPos,
    /// Solves avoidance this tick. Non-avoiding agents are obstacles that
    /// keep their current velocity; avoiders take full responsibility for
    /// them.
    pub avoids: bool,
    /// Driven by navigation this tick. A stationary agent (not moving)
    /// steps sideways out of the corridor a nearby mover wants to use (see
    /// `make_way`). Plain ORCA deadlocks there: a mover facing a gap
    /// narrower than itself slows to a crawl, so the units forming the gap
    /// never see a threat within the horizon.
    pub moving: bool,
    /// Look-ahead override in seconds (0 = `AvoidanceConfig::time_horizon`).
    /// A unit settling into its slot uses a short one, or the neighbours
    /// already standing in their slots would keep it from closing in.
    pub horizon: f32,
}

#[derive(Clone, Copy)]
struct Line { point: WorldPos, dir: WorldPos }

const EPS: f32 = 1e-5;

#[inline] fn det(a: WorldPos, b: WorldPos) -> f32 { a.x * b.y - a.y * b.x }
#[inline] fn dot(a: WorldPos, b: WorldPos) -> f32 { a.x * b.x + a.y * b.y }
#[inline] fn len_sq(a: WorldPos) -> f32 { dot(a, a) }

/// Agents sorted by (bucket row, bucket column, index), with a dense index
/// of where each bucket row starts. A row's buckets are a contiguous range
/// found by a short binary search within that row, so memory and build time
/// scale with the number of agents, not the size of the map.
struct Grid {
    cell: f32,
    keys: Vec<(i32, i32, u32)>,
    min_y: i32,
    row_start: Vec<u32>,
}

impl Grid {
    fn build(agents: &[Agent], cell: f32, par: &Par) -> Self {
        let mut keys: Vec<(i32, i32, u32)> = agents.iter().enumerate()
            .map(|(i, a)| ((a.pos.y / cell).floor() as i32, (a.pos.x / cell).floor() as i32, i as u32)).collect();
        par.sort_unstable(&mut keys);
        let min_y = keys.first().map(|k| k.0).unwrap_or(0);
        let max_y = keys.last().map(|k| k.0).unwrap_or(0);
        let mut row_start = vec![0u32; (max_y - min_y + 2) as usize];
        for k in &keys { row_start[(k.0 - min_y + 1) as usize] += 1; }
        for r in 1..row_start.len() { row_start[r] += row_start[r - 1]; }
        Self { cell, keys, min_y, row_start }
    }

    /// Whether any agent other than `i` lies within `range` of `p`.
    fn any_within(&self, agents: &[Agent], i: usize, p: WorldPos, range: f32) -> bool {
        let r = (range / self.cell).ceil() as i32;
        let bx = (p.x / self.cell).floor() as i32;
        let by = (p.y / self.cell).floor() as i32;
        let range_sq = range * range;
        (by - r..=by + r).any(|y| self.row(y, bx - r, bx + r).iter()
            .any(|&(_, _, j)| j as usize != i && len_sq(agents[j as usize].pos.sub(p)) <= range_sq))
    }

    /// Agents in bucket row `y`, columns `x0..=x1`.
    fn row(&self, y: i32, x0: i32, x1: i32) -> &[(i32, i32, u32)] {
        let r = y - self.min_y;
        if r < 0 || r as usize + 1 >= self.row_start.len() { return &[]; }
        let row = &self.keys[self.row_start[r as usize] as usize..self.row_start[r as usize + 1] as usize];
        let lo = row.partition_point(|k| k.1 < x0);
        let hi = row.partition_point(|k| k.1 <= x1);
        &row[lo..hi.max(lo)]
    }

    /// The `k` nearest other agents within `range`, as (distance², index),
    /// sorted by that pair. Rings of buckets are scanned outwards; the scan
    /// stops once the next ring is farther than the current k-th neighbour.
    fn nearest(&self, agents: &[Agent], i: usize, k: usize, range: f32, out: &mut Vec<(f32, u32)>) {
        out.clear();
        let p = agents[i].pos;
        let bx = (p.x / self.cell).floor() as i32;
        let by = (p.y / self.cell).floor() as i32;
        // Distance from p to the inner edge of ring 1 (the nearest side of
        // its own bucket); ring r starts (r - 1) buckets beyond that.
        let fx = p.x / self.cell - bx as f32;
        let fy = p.y / self.cell - by as f32;
        let edge = fx.min(1.0 - fx).min(fy).min(1.0 - fy) * self.cell;
        let range_sq = range * range;
        let max_ring = (range / self.cell) as i32 + 1;
        let less = |a: (f32, u32), b: (f32, u32)| a.0 < b.0 || (a.0 == b.0 && a.1 < b.1);
        let take = |items: &[(i32, i32, u32)], out: &mut Vec<(f32, u32)>| {
            for &(_, _, j) in items {
                if j as usize == i { continue; }
                let d = len_sq(agents[j as usize].pos.sub(p));
                if d > range_sq { continue; }
                let c = (d, j);
                // Bounded insertion: `out` stays sorted and at most k long.
                if out.len() == k {
                    if !less(c, out[k - 1]) { continue; }
                    out.pop();
                }
                let pos = out.iter().position(|&o| less(c, o)).unwrap_or(out.len());
                out.insert(pos, c);
            }
        };
        for ring in 0..=max_ring {
            if ring >= 1 && out.len() == k {
                let near = edge + (ring - 1) as f32 * self.cell;
                if near * near > out[k - 1].0 { break; }
            }
            if ring == 0 {
                take(self.row(by, bx, bx), out);
            } else {
                take(self.row(by - ring, bx - ring, bx + ring), out);
                take(self.row(by + ring, bx - ring, bx + ring), out);
                for y in by - ring + 1..by + ring {
                    take(self.row(y, bx - ring, bx - ring), out);
                    take(self.row(y, bx + ring, bx + ring), out);
                }
            }
        }
    }
}

/// New velocities for all agents (non-avoiders keep their velocity).
/// `threads` > 1 splits agents into contiguous chunks on scoped threads; the
/// result is identical for any value.
pub fn compute_velocities(agents: &[Agent], cfg: &AvoidanceConfig, dt: f32, par: &Par) -> Vec<WorldPos> {
    if agents.is_empty() { return Vec::new(); }
    // A crowd entirely at rest: every agent gets what the per-agent idle
    // shortcut below would give it, without building the grids.
    let still = |a: &Agent| !a.moving && len_sq(a.vel) < EPS && len_sq(a.pref) < EPS;
    if agents.iter().all(|a| !a.avoids || still(a)) {
        return agents.iter().map(|a| if a.avoids { WorldPos::ZERO } else { a.vel }).collect();
    }
    let max_r = agents.iter().map(|a| a.radius).fold(0.0f32, f32::max);
    let max_v = agents.iter().map(|a| a.max_speed).fold(0.0f32, f32::max);
    // Movers only (index into `movers`), for the idle shortcut below.
    let movers: Vec<Agent> = agents.iter().filter(|a| a.moving).copied().collect();
    let (grid, mover_grid) = par.join(
        || Grid::build(agents, (max_r * 4.0).max(1.0), par),
        || Grid::build(&movers, (max_v * cfg.time_horizon).max(max_r * 4.0).max(1.0), par));
    let solve_range = |lo: usize, hi: usize| -> Vec<WorldPos> {
        let mut nbrs = Vec::with_capacity(cfg.max_neighbors * 4);
        let mut lines = Vec::with_capacity(cfg.max_neighbors);
        (lo..hi).map(|i| {
            let a = &agents[i];
            if !a.avoids { return a.vel; }
            let range = a.max_speed.max(max_v) * cfg.time_horizon + a.radius + max_r;
            // A still agent that wants to stay put keeps standing unless a
            // mover is within reach (overlaps between still agents are left
            // to the collision pass).
            if still(a) && !mover_grid.any_within(&movers, usize::MAX, a.pos, range) {
                return WorldPos::ZERO;
            }
            grid.nearest(agents, i, cfg.max_neighbors, range, &mut nbrs);
            agent_velocity(agents, i, &nbrs, cfg, dt, &mut lines)
        }).collect()
    };
    par.ranges(agents.len(), solve_range).into_iter().flatten().collect()
}

fn agent_velocity(agents: &[Agent], i: usize, nbrs: &[(f32, u32)], cfg: &AvoidanceConfig, dt: f32, lines: &mut Vec<Line>) -> WorldPos {
    let a = &agents[i];
    let inv_tau = 1.0 / if a.horizon > 0.0 { a.horizon } else { cfg.time_horizon };
    lines.clear();
    for &(dist_sq, j) in nbrs {
        let b = &agents[j as usize];
        let rel_pos = b.pos.sub(a.pos);
        let rel_vel = a.vel.sub(b.vel);
        let r = a.radius + b.radius;
        let r_sq = r * r;
        let (dir, u);
        if dist_sq > r_sq {
            // No collision yet: project onto the truncated velocity obstacle.
            let w = rel_vel.sub(rel_pos.scaled(inv_tau));
            let w_len_sq = len_sq(w);
            let dot1 = dot(w, rel_pos);
            if dot1 < 0.0 && dot1 * dot1 > r_sq * w_len_sq {
                // Cut-off circle.
                let w_len = w_len_sq.sqrt();
                let unit_w = w.scaled(1.0 / w_len);
                dir = WorldPos::new(unit_w.y, -unit_w.x);
                u = unit_w.scaled(r * inv_tau - w_len);
            } else {
                // Legs.
                let leg = (dist_sq - r_sq).sqrt();
                dir = if det(rel_pos, w) > 0.0 {
                    WorldPos::new(rel_pos.x * leg - rel_pos.y * r, rel_pos.x * r + rel_pos.y * leg).scaled(1.0 / dist_sq)
                } else {
                    WorldPos::new(rel_pos.x * leg + rel_pos.y * r, -rel_pos.x * r + rel_pos.y * leg).scaled(-1.0 / dist_sq)
                };
                u = dir.scaled(dot(rel_vel, dir)).sub(rel_vel);
            }
        } else {
            // Already overlapping: separate within one step.
            let inv_dt = 1.0 / dt;
            let w = rel_vel.sub(rel_pos.scaled(inv_dt));
            let w_len = len_sq(w).sqrt().max(EPS);
            let unit_w = w.scaled(1.0 / w_len);
            dir = WorldPos::new(unit_w.y, -unit_w.x);
            u = unit_w.scaled(r * inv_dt - w_len);
        }
        // Reciprocal: half the correction when the other agent also avoids.
        let share = if b.avoids { 0.5 } else { 1.0 };
        lines.push(Line { point: a.vel.add(u.scaled(share)), dir });
    }
    let pref = if a.moving { a.pref } else { a.pref.add(make_way(agents, i, nbrs)) };
    let mut result = WorldPos::ZERO;
    let fail = linear_program2(lines, a.max_speed, pref, false, &mut result);
    if fail < lines.len() { linear_program3(lines, fail, a.max_speed, &mut result); }
    result
}

/// Sideways velocity for a stationary agent standing in the swept corridor
/// of nearby movers: perpendicular to each mover's preferred direction, away
/// from its path, never along it (pushing a stationary unit ahead of a crowd
/// would bulldoze it into the next chokepoint). Capped at 60% of max speed.
fn make_way(agents: &[Agent], i: usize, nbrs: &[(f32, u32)]) -> WorldPos {
    let a = &agents[i];
    let mut push = WorldPos::ZERO;
    for &(_, j) in nbrs {
        let b = &agents[j as usize];
        if !b.moving { continue; }
        let speed = b.pref.length();
        if speed <= EPS { continue; }
        let fwd = b.pref.scaled(1.0 / speed);
        let off = a.pos.sub(b.pos);
        let ahead = dot(off, fwd);
        let clear = (a.radius + b.radius) * 1.1;
        // A mover already touching us (beside or ahead) can be wedged between
        // standing units, which ORCA alone cannot resolve: step aside fully.
        let touching = len_sq(off) < clear * clear;
        // Otherwise only movers about to reach us: ahead within half a
        // second of travel.
        if !touching && (ahead <= 0.0 || ahead > clear + speed * 0.5) { continue; }
        let lateral = det(fwd, off);
        if !touching && lateral.abs() >= clear { continue; }
        let side = if lateral >= 0.0 { 1.0 } else { -1.0 };
        let urgency = if touching { 1.0 } else { 1.0 - lateral.abs() / clear };
        push = push.add(WorldPos::new(-fwd.y, fwd.x).scaled(side * urgency));
    }
    let len = push.length();
    if len <= EPS { return WorldPos::ZERO; }
    push.scaled(a.max_speed * 0.6 * len.min(1.0) / len)
}

fn linear_program1(lines: &[Line], line_no: usize, radius: f32, opt: WorldPos, direction_opt: bool, result: &mut WorldPos) -> bool {
    let l = lines[line_no];
    let dot_p = dot(l.point, l.dir);
    let disc = dot_p * dot_p + radius * radius - len_sq(l.point);
    if disc < 0.0 { return false; }
    let sqrt_disc = disc.sqrt();
    let mut t_left = -dot_p - sqrt_disc;
    let mut t_right = -dot_p + sqrt_disc;
    for other in &lines[..line_no] {
        let denom = det(l.dir, other.dir);
        let numer = det(other.dir, l.point.sub(other.point));
        if denom.abs() <= EPS {
            if numer < 0.0 { return false; }
            continue;
        }
        let t = numer / denom;
        if denom >= 0.0 { t_right = t_right.min(t); } else { t_left = t_left.max(t); }
        if t_left > t_right { return false; }
    }
    *result = if direction_opt {
        if dot(opt, l.dir) > 0.0 { l.point.add(l.dir.scaled(t_right)) } else { l.point.add(l.dir.scaled(t_left)) }
    } else {
        let t = dot(l.dir, opt.sub(l.point));
        l.point.add(l.dir.scaled(t.clamp(t_left, t_right)))
    };
    true
}

fn linear_program2(lines: &[Line], radius: f32, opt: WorldPos, direction_opt: bool, result: &mut WorldPos) -> usize {
    *result = if direction_opt { opt.scaled(radius) }
        else if len_sq(opt) > radius * radius { opt.normalized_or(WorldPos::ZERO).scaled(radius) }
        else { opt };
    for i in 0..lines.len() {
        if det(lines[i].dir, lines[i].point.sub(*result)) > 0.0 {
            let saved = *result;
            if !linear_program1(lines, i, radius, opt, direction_opt, result) {
                *result = saved;
                return i;
            }
        }
    }
    lines.len()
}

/// Infeasible: minimise the largest constraint violation.
fn linear_program3(lines: &[Line], begin: usize, radius: f32, result: &mut WorldPos) {
    let mut distance = 0.0f32;
    let mut proj: Vec<Line> = Vec::with_capacity(lines.len());
    for i in begin..lines.len() {
        if det(lines[i].dir, lines[i].point.sub(*result)) > distance {
            proj.clear();
            for j in 0..i {
                let d = det(lines[i].dir, lines[j].dir);
                let point = if d.abs() <= EPS {
                    if dot(lines[i].dir, lines[j].dir) > 0.0 { continue; }
                    lines[i].point.add(lines[j].point).scaled(0.5)
                } else {
                    lines[i].point.add(lines[i].dir.scaled(det(lines[j].dir, lines[i].point.sub(lines[j].point)) / d))
                };
                let dir = lines[j].dir.sub(lines[i].dir).normalized_or(WorldPos::ZERO);
                proj.push(Line { point, dir });
            }
            let saved = *result;
            if linear_program2(&proj, radius, WorldPos::new(-lines[i].dir.y, lines[i].dir.x), true, result) < proj.len() {
                *result = saved;
            }
            distance = det(lines[i].dir, lines[i].point.sub(*result));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(x: f32, y: f32, vx: f32, vy: f32) -> Agent {
        Agent { pos: WorldPos::new(x, y), vel: WorldPos::new(vx, vy), radius: 1.0, max_speed: 2.0, pref: WorldPos::new(vx, vy), avoids: true, moving: true, horizon: 0.0 }
    }

    #[test]
    fn head_on_agents_sidestep_and_never_overlap() {
        let mut a = vec![agent(0.0, 0.0, 1.0, 0.0), agent(10.0, 0.05, -1.0, 0.0)];
        let cfg = AvoidanceConfig { time_horizon: 2.0, max_neighbors: 10 };
        let dt = 0.1;
        let mut min_gap = f32::MAX;
        for _ in 0..200 {
            let v = compute_velocities(&a, &cfg, dt, &Par::new(1));
            for (ag, nv) in a.iter_mut().zip(v) { ag.vel = nv; ag.pos = ag.pos.add(nv.scaled(dt)); }
            min_gap = min_gap.min(a[0].pos.dist(a[1].pos) - 2.0);
        }
        assert!(min_gap > -0.05, "agents overlapped by {}", -min_gap);
        assert!(a[0].pos.x > 10.0 && a[1].pos.x < 0.0, "agents should pass each other");
    }

    #[test]
    fn result_is_independent_of_thread_count() {
        let mut agents = Vec::new();
        for k in 0..3000 {
            let (x, y) = ((k % 60) as f32 * 2.3, (k / 60) as f32 * 2.1);
            let mut a = agent(x, y, ((k * 7) % 5) as f32 * 0.3 - 0.6, ((k * 3) % 7) as f32 * 0.2 - 0.6);
            a.avoids = k % 9 != 0;
            a.moving = k % 4 != 0;
            agents.push(a);
        }
        let cfg = AvoidanceConfig::default();
        let one = compute_velocities(&agents, &cfg, 1.0 / 30.0, &Par::new(1));
        for t in [2, 3, 8] {
            let many = compute_velocities(&agents, &cfg, 1.0 / 30.0, &Par::new(t));
            assert!(one.iter().zip(&many).all(|(a, b)| a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits()), "threads={t}");
        }
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    #[test]
    #[ignore]
    fn time_dense_crowd() {
        let mut agents = Vec::new();
        for k in 0..10000u32 {
            let (x, y) = ((k % 100) as f32 * 11.0 + 100.0, (k / 100) as f32 * 11.0 + 100.0);
            agents.push(Agent { pos: WorldPos::new(x, y), vel: WorldPos::new(80.0, 10.0), radius: 5.0, max_speed: 112.0,
                pref: WorldPos::new(100.0, 20.0), avoids: true, moving: k % 5 != 0, horizon: 0.0 });
        }
        let cfg = AvoidanceConfig::default();
        let max_r = 5.0; let max_v = 112.0;
        let t = std::time::Instant::now();
        let grid = Grid::build(&agents, max_r * 4.0, &Par::new(1));
        let build = t.elapsed();
        let mut nb = Vec::new();
        let t = std::time::Instant::now();
        let mut total = 0usize;
        for i in 0..agents.len() { grid.nearest(&agents, i, 10, max_v * cfg.time_horizon + 2.0 * max_r, &mut nb); total += nb.len(); }
        let search = t.elapsed();
        let t = std::time::Instant::now();
        let v = compute_velocities(&agents, &cfg, 1.0 / 30.0, &Par::new(1));
        let all = t.elapsed();
        eprintln!("build {:?} search {:?} ({} nbrs) full serial {:?} ({})", build, search, total, all, v.len());
    }
}
