//! The simulation world and the fixed-timestep loop.
//!
//! ```text
//! World::step(dt)
//!   1. integrate_velocities   v ← (v + (a_user + g·[m⁻¹>0])·dt)·damp           [SIMD, SoA]
//!   2. broad_phase.update     positions → AABBs → hash grid → candidate pairs   [SIMD AABB]
//!   3. narrow_phase::detect   pairs → exact tests → Contact{n, depth, point}
//!   4. solver.prepare + N × velocity_sweep   (sequential impulses)
//!   5. integrate_positions    x ← x + v·dt                                      [SIMD, SoA]
//!   6. solver.solve_position  M × Baumgarte shift
//! ```
//! Every buffer on this path is a `Vec` that is `clear()`ed, never dropped, so after
//! the first few steps (warm-up) `step()` performs **zero heap allocations**
//! (verified by the counting allocator in `bin/bench.rs`).

use crate::body::{BodyId, BodyStore, RigidBody};
use crate::broad_phase::BroadPhase;
use crate::narrow_phase::{self, Contact};
use crate::solver::{Solver, SolverParams};
use crate::vec2::Vec2;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct WorldConfig {
    pub gravity: Vec2,
    /// Fixed physics timestep (seconds).
    pub fixed_dt: f32,
    /// Max catch-up steps per `advance()` call (prevents the "spiral of death").
    pub max_steps_per_frame: u32,
    pub solver_iterations: u32,
    pub position_iterations: u32,
    pub restitution_threshold: f32,
    /// Speed (m/s) under which a dynamic body counts as "at rest" in the statistics.
    /// NOTE: sleeping itself is not implemented yet (see roadmap); this only feeds
    /// `World::resting_bodies()`.
    pub sleep_threshold: f32,
    pub baumgarte_beta: f32,
    pub slop: f32,
    /// Seed the solver with the previous step's contact impulses (greatly improves stacking).
    pub warm_starting: bool,
    /// Linear damping coefficient c (1/s) in v' = −c·v.
    pub linear_damping: f32,
    /// Broad-phase cell edge length (m). Use `World::fit_cell_size()` to set it automatically.
    pub cell_size: f32,
    /// Counting-sort candidate pairs by body id (cache locality). Off by default: measured ±2 % (noise).
    pub sort_pairs: bool,
    /// Use explicit SIMD kernels (results are bit-identical either way).
    pub use_simd: bool,
    /// Measure per-phase wall-clock time (costs ~5 `Instant::now()` ≈ 100 ns / step).
    pub collect_timings: bool,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
            gravity: Vec2::new(0.0, -9.81),
            fixed_dt: 1.0 / 60.0,
            max_steps_per_frame: 5,
            solver_iterations: 8,
            position_iterations: 3,
            restitution_threshold: 1.0,
            sleep_threshold: 0.05,
            baumgarte_beta: 0.4,
            slop: 0.01,
            warm_starting: true,
            linear_damping: 0.0,
            cell_size: 1.0,
            sort_pairs: false, // measured: no gain (±2 %), see README optimisation log
            use_simd: true,
            collect_timings: true,
        }
    }
}

/// Wall-clock nanoseconds spent in each phase of one step.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepTimings {
    pub integrate_ns: u64,
    pub broad_ns: u64,
    pub narrow_ns: u64,
    pub solve_ns: u64,
}
impl StepTimings {
    pub fn total_ns(&self) -> u64 {
        self.integrate_ns + self.broad_ns + self.narrow_ns + self.solve_ns
    }
    pub fn add(&mut self, o: &StepTimings) {
        self.integrate_ns += o.integrate_ns;
        self.broad_ns += o.broad_ns;
        self.narrow_ns += o.narrow_ns;
        self.solve_ns += o.solve_ns;
    }
}

pub struct World {
    pub config: WorldConfig,
    pub bodies: BodyStore,
    broad: BroadPhase,
    contacts: Vec<Contact>,
    solver: Solver,
    accumulator: f32,
    last_timings: StepTimings,
    last_pairs: usize,
}

impl World {
    pub fn new(config: WorldConfig) -> World {
        World {
            config,
            bodies: BodyStore::new(),
            broad: BroadPhase::new(),
            contacts: Vec::new(),
            solver: Solver::new(),
            accumulator: 0.0,
            last_timings: StepTimings::default(),
            last_pairs: 0,
        }
    }

    pub fn add_body(&mut self, b: RigidBody) -> BodyId {
        self.bodies.add(b)
    }

    /// Pre-size all pooled buffers so even the first step does not allocate.
    pub fn reserve(&mut self, bodies: usize, pairs: usize, contacts: usize) {
        self.bodies.reserve(bodies);
        self.broad.reserve(bodies, pairs);
        self.contacts.reserve(contacts);
        self.solver.reserve(contacts);
    }

    /// Set `config.cell_size` from the current bodies: 1.05 × the largest *dynamic* AABB extent.
    /// That guarantees every dynamic body covers ≤ 2×2 cells (so none falls into the slow
    /// large-body path) while keeping cells as small as possible (fewest false candidates).
    /// Call after adding bodies; large static bodies (floors/walls) are fine.
    pub fn fit_cell_size(&mut self) {
        let b = &self.bodies;
        let mut m = 0.0f32;
        for i in 0..b.len() {
            if b.inv_mass[i] > 0.0 {
                m = m.max(2.0 * b.half_x[i]).max(2.0 * b.half_y[i]);
            }
        }
        if m > 0.0 {
            self.config.cell_size = 1.05 * m;
        }
    }

    pub fn contacts(&self) -> &[Contact] {
        &self.contacts
    }
    pub fn last_timings(&self) -> StepTimings {
        self.last_timings
    }
    pub fn broad_timings(&self) -> crate::broad_phase::BroadTimings {
        self.broad.timings
    }
    pub fn last_pair_count(&self) -> usize {
        self.last_pairs
    }
    /// Dynamic bodies slower than `sleep_threshold` (statistic only).
    pub fn resting_bodies(&self) -> usize {
        let t2 = self.config.sleep_threshold * self.config.sleep_threshold;
        (0..self.bodies.len())
            .filter(|&i| self.bodies.inv_mass[i] > 0.0 && self.bodies.vel_x[i] * self.bodies.vel_x[i] + self.bodies.vel_y[i] * self.bodies.vel_y[i] < t2)
            .count()
    }

    /// Advance by `frame_time` using the fixed-timestep accumulator.
    /// Returns the number of physics steps taken. `alpha()` is the render interpolation factor.
    pub fn advance(&mut self, frame_time: f32) -> u32 {
        let dt = self.config.fixed_dt;
        self.accumulator += frame_time;
        let mut steps = 0;
        while self.accumulator >= dt && steps < self.config.max_steps_per_frame {
            self.step(dt);
            self.accumulator -= dt;
            steps += 1;
        }
        if steps == self.config.max_steps_per_frame {
            self.accumulator = 0.0; // drop the backlog instead of spiralling
        }
        steps
    }
    /// Fraction of a step left in the accumulator, in [0, 1).
    pub fn alpha(&self) -> f32 {
        self.accumulator / self.config.fixed_dt
    }

    /// One simulation step of exactly `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        let cfg = self.config;
        let timing = cfg.collect_timings;
        let mut t = StepTimings::default();
        let mut mark = if timing { Some(Instant::now()) } else { None };
        let mut lap = |slot: &mut u64| {
            if let Some(m) = mark {
                let now = Instant::now();
                *slot += now.duration_since(m).as_nanos() as u64;
                mark = Some(now);
            }
        };

        // 1. gravity + forces → velocities  (damp = 1/(1 + c·dt), implicit-Euler damping)
        let damp = 1.0 / (1.0 + cfg.linear_damping * dt);
        self.bodies.integrate_velocities(dt, cfg.gravity, damp, cfg.use_simd);
        lap(&mut t.integrate_ns);

        // 2. broad phase
        self.broad.profile = timing;
        self.broad.sort_pairs = cfg.sort_pairs;
        self.broad.update(&self.bodies, cfg.cell_size, cfg.use_simd);
        self.last_pairs = self.broad.pairs.len();
        lap(&mut t.broad_ns);

        // 3. narrow phase
        narrow_phase::detect(&self.bodies, &self.broad.pairs, &mut self.contacts);
        lap(&mut t.narrow_ns);

        // 4. velocity constraints
        let params = SolverParams {
            velocity_iterations: cfg.solver_iterations,
            position_iterations: cfg.position_iterations,
            restitution_threshold: cfg.restitution_threshold,
            baumgarte_beta: cfg.baumgarte_beta,
            slop: cfg.slop,
            warm_start: cfg.warm_starting,
            ..SolverParams::default()
        };
        self.solver.prepare(&self.contacts, &mut self.bodies, &params);
        self.solver.solve_velocity(&mut self.bodies, params.velocity_iterations);
        lap(&mut t.solve_ns);

        // 5. velocities → positions
        self.bodies.integrate_positions(dt, cfg.use_simd);
        lap(&mut t.integrate_ns);

        // 6. positional (Baumgarte) correction
        self.solver.solve_position(&mut self.bodies, &params);
        lap(&mut t.solve_ns);

        self.last_timings = t;
    }

    /// FNV-1a hash of every position & velocity bit — for reproducibility checks.
    pub fn state_hash(&self) -> u64 {
        let mut h = crate::util::BitHasher::new();
        for i in 0..self.bodies.len() {
            h.push_f32(self.bodies.pos_x[i]);
            h.push_f32(self.bodies.pos_y[i]);
            h.push_f32(self.bodies.vel_x[i]);
            h.push_f32(self.bodies.vel_y[i]);
        }
        h.finish()
    }
}
