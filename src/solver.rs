//! Sequential-impulse contact solver (velocity pass + Baumgarte position pass).
//!
//! # Derivation of the normal impulse
//! Let n be the unit contact normal (A → B), vA, vB the velocities and iA = 1/mA,
//! iB = 1/mB. The relative normal velocity is `vn = (vB − vA)·n`; `vn < 0` ⇒ approaching.
//!
//! Newton's restitution law says the *post*-impact relative normal velocity is
//! `vn' = −e·vn⁻` (vn⁻ = pre-impact value). An impulse `J = j·n` changes velocities by
//! `vA' = vA − iA·j·n`, `vB' = vB + iB·j·n`, hence
//!
//! ```text
//!   vn' = vn + (iA + iB)·j
//!   set vn' = −e·vn      ⇒   j = −(1 + e)·vn / (iA + iB)          (the formula in the spec)
//! ```
//! We solve in the general "velocity target" form `vn' = target`:
//! `j = (target − vn)·m_eff`, with `m_eff = 1/(iA + iB)` and `target = −e·vn⁻` (or 0).
//!
//! # Why accumulated impulses
//! A contact can only *push* (`j_total ≥ 0`), never pull. With several contacts
//! solved one after another, clamping each *increment* at 0 is wrong: a later contact
//! may need to *take back* part of an earlier over-push. So we keep
//! `λ_acc` per contact and clamp the **sum**: `λ_new = max(λ_acc + j, 0)`, and apply
//! only `Δ = λ_new − λ_acc`. (Erin Catto, "Iterative Dynamics", GDC 2006.)
//!
//! # Warm starting
//! A resting pile needs, every step, an impulse of ≈ m·g·dt per body to cancel the
//! velocity gravity just added; for the bottom of a deep stack that is the weight of
//! everything above. Gauss–Seidel propagates that information only ~1 contact per
//! sweep, so from a cold start an 8-iteration solver cannot converge in a 15-layer pile
//! (measured: median residual speed 0.55 m/s). Warm starting seeds each contact with
//! the accumulated impulse of the *same body pair* from the previous step and applies
//! it up-front, so the iterations only have to correct the (small) change.
//! Pairs are matched by key `(a, b)` through an allocation-free open-addressing table
//! with integer hashing (deterministic; `std::HashMap` is randomly seeded).
//! Contacts that are *impacts* (closing speed above the restitution threshold) are not
//! warm-started: their impulse is a bounce, not a support force.
//!
//! # Static bodies without branches
//! `inv_mass = 0` ⇒ Δv = 0 automatically; if both are 0, `m_eff` is set to 0 in the
//! prepare step and the contact becomes a no-op. The inner loop has **no** `is_static`
//! test and no conditional other than the `max(·,0)` clamp (compiled to `maxss`).
//!
//! # Restitution threshold
//! Gravity adds `g·dt ≈ 0.16 m/s` every step. If `e` were applied to such tiny
//! closing speeds a resting body would micro-bounce forever. If `|vn⁻| <
//! restitution_threshold` we use `target = 0` (perfectly inelastic).
//!
//! # Baumgarte positional correction
//! Velocity solving removes approach speed but not existing overlap. Feeding the
//! overlap back into the *velocity* (classic Baumgarte `bias = β/dt·pen`) injects
//! energy; we instead apply it as a **direct position shift** ("pseudo-velocity"),
//! which never touches the real velocities:
//!
//! ```text
//!   pen_now   = pen₀ − ((xB − xA) − rel₀)·n          (exact for translating bodies)
//!   correction = β · max(pen_now − slop, 0) / (iA + iB)
//!   xA −= correction·iA·n ;  xB += correction·iB·n
//! ```
//! `slop` (≈ 1 cm) is deliberate leftover penetration: it keeps the contact alive
//! frame-to-frame so resting bodies don't alternate between touching and separated
//! (flicker). β ∈ [0.2, 0.8]: larger is stiffer but can overshoot.

use crate::body::BodyStore;
use crate::narrow_phase::Contact;

/// Velocity-pass constraint, 36 bytes, `repr(C)` so the hot loop streams linearly.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct VelocityConstraint {
    pub a: u32,
    pub b: u32,
    pub nx: f32,
    pub ny: f32,
    /// m_eff = 1/(iA + iB)   (0 if both bodies are immovable)
    pub eff_mass: f32,
    /// Desired post-solve relative normal velocity (0 or the bounce speed).
    pub target_vn: f32,
    /// Accumulated impulse magnitude λ (≥ 0).
    pub acc: f32,
    pub inv_a: f32,
    pub inv_b: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct PositionConstraint {
    pub a: u32,
    pub b: u32,
    pub nx: f32,
    pub ny: f32,
    pub penetration: f32,
    /// (xB − xA) when the contact was generated.
    pub rel0_x: f32,
    pub rel0_y: f32,
    /// β/(iA+iB)  (pre-multiplied)
    pub k: f32,
    pub inv_a: f32,
    pub inv_b: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SolverParams {
    pub velocity_iterations: u32,
    pub position_iterations: u32,
    pub restitution_threshold: f32,
    pub baumgarte_beta: f32,
    pub slop: f32,
    /// Hard cap on the per-iteration shift to avoid explosive de-penetration.
    pub max_correction: f32,
    /// Seed contacts with last step's impulses (see module docs).
    pub warm_start: bool,
}

impl Default for SolverParams {
    fn default() -> Self {
        SolverParams {
            velocity_iterations: 8,
            position_iterations: 3,
            restitution_threshold: 1.0,
            baumgarte_beta: 0.4,
            slop: 0.01,
            max_correction: 0.2,
            warm_start: true,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct CacheEntry {
    /// `((a << 32) | b) + 1`; 0 marks an empty slot.
    key: u64,
    acc: f32,
}

/// Previous-step impulses keyed by body pair. Two open-addressing tables (linear probing,
/// load ≤ 0.5) swapped every step; both are cleared with `resize`, never freed.
#[derive(Default)]
struct ImpulseCache {
    prev: Vec<CacheEntry>,
    cur: Vec<CacheEntry>,
    prev_mask: usize,
    cur_mask: usize,
}

#[inline(always)]
fn pair_key(a: u32, b: u32) -> u64 {
    (((a as u64) << 32) | b as u64) + 1
}
#[inline(always)]
fn slot(key: u64, mask: usize) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask
}

impl ImpulseCache {
    fn reserve(&mut self, contacts: usize) {
        let n = (2 * contacts).next_power_of_two().max(16);
        self.prev.reserve(n);
        self.cur.reserve(n);
    }
    /// Make last step's `cur` the new `prev` and start an empty `cur` sized for `n` contacts.
    fn begin_step(&mut self, n: usize) {
        std::mem::swap(&mut self.prev, &mut self.cur);
        self.prev_mask = self.cur_mask;
        let size = (2 * n).next_power_of_two().max(16);
        self.cur.clear();
        self.cur.resize(size, CacheEntry::default());
        self.cur_mask = size - 1;
    }
    #[inline]
    fn lookup(&self, key: u64) -> f32 {
        if self.prev.is_empty() {
            return 0.0;
        }
        let mut i = slot(key, self.prev_mask);
        loop {
            let e = self.prev[i];
            if e.key == key {
                return e.acc;
            }
            if e.key == 0 {
                return 0.0;
            }
            i = (i + 1) & self.prev_mask;
        }
    }
    #[inline]
    fn store(&mut self, key: u64, acc: f32) {
        let mut i = slot(key, self.cur_mask);
        while self.cur[i].key != 0 {
            i = (i + 1) & self.cur_mask;
        }
        self.cur[i] = CacheEntry { key, acc };
    }
}

/// Pooled solver buffers (cleared, never freed).
#[derive(Default)]
pub struct Solver {
    pub vel: Vec<VelocityConstraint>,
    pub pos: Vec<PositionConstraint>,
    cache: ImpulseCache,
}

#[inline(always)]
fn clamp_min0(x: f32) -> f32 {
    // explicit compare ⇒ identical semantics on every platform (f32::max NaN handling is unspecified)
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

impl Solver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reserve(&mut self, contacts: usize) {
        self.vel.reserve(contacts);
        self.pos.reserve(contacts);
        self.cache.reserve(contacts);
    }

    /// Build constraints from contacts using the *current* velocities/positions.
    pub fn prepare(&mut self, contacts: &[Contact], bodies: &mut BodyStore, p: &SolverParams) {
        self.vel.clear();
        self.pos.clear();
        self.cache.begin_step(contacts.len());
        for c in contacts {
            let (a, b) = (c.body_a.index(), c.body_b.index());
            let (ia, ib) = (bodies.inv_mass[a], bodies.inv_mass[b]);
            let sum = ia + ib;
            let eff = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            // e = min(eA, eB)
            let e = if bodies.restitution[a] < bodies.restitution[b] { bodies.restitution[a] } else { bodies.restitution[b] };
            let vn = (bodies.vel_x[b] - bodies.vel_x[a]) * c.normal.x + (bodies.vel_y[b] - bodies.vel_y[a]) * c.normal.y;
            let impact = vn < -p.restitution_threshold;
            let target = if impact { -e * vn } else { 0.0 };
            // Cached impulse for resting contacts only (applied in a *second* pass below, so
            // that every contact's closing speed above is measured on untouched velocities).
            let acc0 = if p.warm_start && !impact { self.cache.lookup(pair_key(a as u32, b as u32)) } else { 0.0 };
            self.vel.push(VelocityConstraint {
                a: a as u32,
                b: b as u32,
                nx: c.normal.x,
                ny: c.normal.y,
                eff_mass: eff,
                target_vn: target,
                acc: acc0,
                inv_a: ia,
                inv_b: ib,
            });
            self.pos.push(PositionConstraint {
                a: a as u32,
                b: b as u32,
                nx: c.normal.x,
                ny: c.normal.y,
                penetration: c.penetration,
                rel0_x: bodies.pos_x[b] - bodies.pos_x[a],
                rel0_y: bodies.pos_y[b] - bodies.pos_y[a],
                k: p.baumgarte_beta * eff,
                inv_a: ia,
                inv_b: ib,
            });
        }
        // Pass 2: warm start — apply the cached impulses.
        for c in &self.vel {
            let (a, b) = (c.a as usize, c.b as usize);
            let (px, py) = (c.acc * c.nx, c.acc * c.ny);
            bodies.vel_x[a] -= px * c.inv_a;
            bodies.vel_y[a] -= py * c.inv_a;
            bodies.vel_x[b] += px * c.inv_b;
            bodies.vel_y[b] += py * c.inv_b;
        }
    }

    /// One Gauss–Seidel sweep over all contacts.
    #[inline]
    pub fn velocity_sweep(&mut self, vx: &mut [f32], vy: &mut [f32]) {
        for c in self.vel.iter_mut() {
            let (a, b) = (c.a as usize, c.b as usize);
            let (vax, vay, vbx, vby) = (vx[a], vy[a], vx[b], vy[b]);
            let vn = (vbx - vax) * c.nx + (vby - vay) * c.ny; // relative normal velocity
            let j = (c.target_vn - vn) * c.eff_mass; // impulse that would reach the target
            let new_acc = clamp_min0(c.acc + j); // clamp the *accumulated* impulse
            let d = new_acc - c.acc; // apply only the increment
            c.acc = new_acc;
            let (px, py) = (d * c.nx, d * c.ny);
            vx[a] = vax - px * c.inv_a;
            vy[a] = vay - py * c.inv_a;
            vx[b] = vbx + px * c.inv_b;
            vy[b] = vby + py * c.inv_b;
        }
    }

    pub fn solve_velocity(&mut self, bodies: &mut BodyStore, iterations: u32) {
        for _ in 0..iterations {
            self.velocity_sweep(&mut bodies.vel_x, &mut bodies.vel_y);
        }
        // remember the converged impulses for next step's warm start
        for c in &self.vel {
            self.cache.store(pair_key(c.a, c.b), c.acc);
        }
    }

    /// Baumgarte position pass (after positions were integrated).
    pub fn solve_position(&mut self, bodies: &mut BodyStore, p: &SolverParams) {
        for _ in 0..p.position_iterations {
            for c in &self.pos {
                let (a, b) = (c.a as usize, c.b as usize);
                let (rx, ry) = (bodies.pos_x[b] - bodies.pos_x[a], bodies.pos_y[b] - bodies.pos_y[a]);
                let pen = c.penetration - ((rx - c.rel0_x) * c.nx + (ry - c.rel0_y) * c.ny);
                let mut corr = clamp_min0(pen - p.slop) * c.k;
                if corr > p.max_correction {
                    corr = p.max_correction;
                }
                let (sx, sy) = (corr * c.nx, corr * c.ny);
                bodies.pos_x[a] -= sx * c.inv_a;
                bodies.pos_y[a] -= sy * c.inv_a;
                bodies.pos_x[b] += sx * c.inv_b;
                bodies.pos_y[b] += sy * c.inv_b;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::{BodyId, RigidBody};
    use crate::shape::Shape;
    use crate::vec2::Vec2;

    fn head_on(ma: f32, mb: f32, va: f32, vb: f32, e: f32) -> (f32, f32) {
        let mut s = BodyStore::new();
        let a = s.add(RigidBody::dynamic(Shape::circle(1.0), ma / core::f32::consts::PI, Vec2::new(0.0, 0.0)).with_velocity(Vec2::new(va, 0.0)).with_restitution(e));
        let b = s.add(RigidBody::dynamic(Shape::circle(1.0), mb / core::f32::consts::PI, Vec2::new(1.9, 0.0)).with_velocity(Vec2::new(vb, 0.0)).with_restitution(e));
        let c = Contact { normal: Vec2::UNIT_X, penetration: 0.1, contact_point: Vec2::new(0.95, 0.0), body_a: a, body_b: b };
        let mut sol = Solver::new();
        let p = SolverParams::default();
        sol.prepare(&[c], &mut s, &p);
        sol.solve_velocity(&mut s, 4);
        (s.velocity(BodyId(0)).x, s.velocity(BodyId(1)).x)
    }

    #[test]
    fn elastic_equal_masses_swap_velocities() {
        let (va, vb) = head_on(1.0, 1.0, 5.0, -3.0, 1.0);
        assert!((va + 3.0).abs() < 1e-4 && (vb - 5.0).abs() < 1e-4, "{va} {vb}");
    }

    #[test]
    fn momentum_conserved_and_restitution_ratio() {
        let (ma, mb, ua, ub, e) = (2.0f32, 5.0f32, 4.0f32, -1.0f32, 0.6f32);
        let (va, vb) = head_on(ma, mb, ua, ub, e);
        assert!((ma * ua + mb * ub - (ma * va + mb * vb)).abs() < 1e-3, "momentum");
        // separation speed = e × approach speed
        assert!(((vb - va) - e * (ua - ub)).abs() < 1e-3, "{va} {vb}");
    }

    #[test]
    fn matches_closed_form_impulse() {
        // j = −(1+e)·vn / (1/mA + 1/mB)
        let (ma, mb, e) = (3.0f32, 7.0f32, 0.5f32);
        let (ua, ub) = (2.0f32, -4.0f32);
        let vn = ub - ua;
        let j = -(1.0 + e) * vn / (1.0 / ma + 1.0 / mb);
        let (va, vb) = head_on(ma, mb, ua, ub, e);
        assert!((va - (ua - j / ma)).abs() < 1e-4 && (vb - (ub + j / mb)).abs() < 1e-4);
    }

    #[test]
    fn below_threshold_is_inelastic_and_static_does_not_move() {
        let (va, vb) = head_on(1.0, 1.0, 0.3, 0.0, 1.0); // closing speed 0.3 < 1.0
        assert!((va - vb).abs() < 1e-5, "should stick: {va} {vb}");
        // static wall: inv_mass = 0, dynamic body bounces with e
        let mut s = BodyStore::new();
        let wall = s.add(RigidBody::fixed(Shape::aligned_box(Vec2::new(1.0, 1.0)), Vec2::ZERO).with_restitution(0.5));
        let ball = s.add(RigidBody::dynamic(Shape::circle(0.5), 1.0, Vec2::new(1.4, 0.0)).with_velocity(Vec2::new(-6.0, 0.0)).with_restitution(0.5));
        let c = Contact { normal: Vec2::UNIT_X, penetration: 0.1, contact_point: Vec2::ZERO, body_a: wall, body_b: ball };
        let mut sol = Solver::new();
        sol.prepare(&[c], &mut s, &SolverParams::default());
        sol.solve_velocity(&mut s, 1);
        assert_eq!(s.velocity(wall), Vec2::ZERO);
        assert!((s.velocity(ball).x - 3.0).abs() < 1e-5); // e = min(0.5, 0.5) ⇒ 0.5·6 = 3
        // …and the pair uses the *smaller* restitution
        let mut s2 = s.clone();
        s2.restitution[0] = 0.1;
        s2.set_velocity(ball, Vec2::new(-6.0, 0.0));
        sol.prepare(&[c], &mut s2, &SolverParams::default());
        sol.solve_velocity(&mut s2, 1);
        assert!((s2.velocity(ball).x - 0.6).abs() < 1e-5);
    }

    #[test]
    fn position_correction_removes_overlap_down_to_slop() {
        let mut s = BodyStore::new();
        let a = s.add(RigidBody::fixed(Shape::aligned_box(Vec2::new(5.0, 0.5)), Vec2::ZERO));
        let b = s.add(RigidBody::dynamic(Shape::circle(0.5), 1.0, Vec2::new(0.0, 0.9)));
        let c = Contact { normal: Vec2::UNIT_Y, penetration: 0.1, contact_point: Vec2::ZERO, body_a: a, body_b: b };
        let p = SolverParams { position_iterations: 20, ..Default::default() };
        let mut sol = Solver::new();
        sol.prepare(&[c], &mut s, &p);
        sol.solve_position(&mut s, &p);
        let gap = s.position(b).y - 1.0; // 0 = touching
        assert!(gap < 0.0 && gap > -p.slop - 1e-4, "left penetration {}", -gap);
        assert_eq!(s.position(a), Vec2::ZERO);
    }
}
