//! Rigid bodies and their struct-of-arrays store.
//!
//! * [`RigidBody`] is the *descriptor / scalar reference* type (AoS): convenient to
//!   construct and the ground truth the SIMD kernels are tested against.
//! * [`BodyStore`] is the *runtime* layout: one `Vec<f32>` per scalar component.
//!   With separate x/y arrays one 128-bit load fetches **4 bodies' x components**
//!   (8 with AVX2), so the integrator and AABB builder process 4 bodies per
//!   instruction with no shuffles. (`Vec<Vec2>` would give only 2 bodies/op.)
//!
//! **Static bodies** are encoded as `inv_mass == 0.0`; every kernel is branch-free
//! with respect to `is_static` (the flag exists purely for user queries).

use crate::shape::Shape;
use crate::simd::{as4, as4_mut, F32x4, LANES};
use crate::vec2::Vec2;

/// Stable handle to a body (index into the SoA arrays).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BodyId(pub u32);
impl BodyId {
    #[inline(always)]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Descriptor of one body, plus scalar reference dynamics.
#[derive(Clone, Copy, Debug)]
pub struct RigidBody {
    pub position: Vec2,
    pub velocity: Vec2,
    /// Accumulated acceleration for the next step (forces/m). Cleared by `integrate`.
    pub acceleration: Vec2,
    pub mass: f32,
    pub inv_mass: f32,
    pub restitution: f32,
    pub is_static: bool,
    pub shape: Shape,
}

impl RigidBody {
    /// Dynamic body whose mass is `density × area`.
    pub fn dynamic(shape: Shape, density: f32, position: Vec2) -> RigidBody {
        let mass = shape.mass(density);
        assert!(mass > 0.0 && mass.is_finite(), "dynamic body needs positive finite mass");
        RigidBody {
            position,
            velocity: Vec2::ZERO,
            acceleration: Vec2::ZERO,
            mass,
            inv_mass: 1.0 / mass,
            restitution: 0.3,
            is_static: false,
            shape,
        }
    }

    /// Immovable body: `inv_mass = 0` encodes "infinite mass" for the solver.
    pub fn fixed(shape: Shape, position: Vec2) -> RigidBody {
        RigidBody {
            position,
            velocity: Vec2::ZERO,
            acceleration: Vec2::ZERO,
            mass: 0.0,
            inv_mass: 0.0,
            restitution: 0.3,
            is_static: true,
            shape,
        }
    }

    pub fn with_velocity(mut self, v: Vec2) -> Self {
        if !self.is_static {
            self.velocity = v;
        }
        self
    }
    pub fn with_restitution(mut self, e: f32) -> Self {
        self.restitution = e;
        self
    }

    /// F = m·a  ⇒  a += F·(1/m). A static body (inv_mass = 0) ignores forces for free.
    #[inline]
    pub fn apply_force(&mut self, force: Vec2) {
        self.acceleration += force * self.inv_mass;
    }

    /// Instantaneous change of momentum J = m·Δv  ⇒  Δv = J·(1/m).
    #[inline]
    pub fn apply_impulse(&mut self, impulse: Vec2) {
        self.velocity += impulse * self.inv_mass;
    }

    /// Semi-implicit (symplectic) Euler:  v ← v + a·dt ;  x ← x + v_new·dt.
    /// Using the *updated* velocity for the position step is what makes it
    /// symplectic (bounded energy error for oscillators) at the same cost as explicit Euler.
    #[inline]
    pub fn integrate(&mut self, dt: f32) {
        self.velocity += self.acceleration * dt;
        self.position += self.velocity * dt;
        self.acceleration = Vec2::ZERO;
    }
}

/// Struct-of-arrays body storage.
#[derive(Default, Clone)]
pub struct BodyStore {
    pub pos_x: Vec<f32>,
    pub pos_y: Vec<f32>,
    pub vel_x: Vec<f32>,
    pub vel_y: Vec<f32>,
    /// Accumulated external acceleration (from `apply_force`), cleared every step.
    pub acc_x: Vec<f32>,
    pub acc_y: Vec<f32>,
    pub inv_mass: Vec<f32>,
    pub mass: Vec<f32>,
    pub restitution: Vec<f32>,
    /// World-AABB half extents (exact, because bodies never rotate).
    pub half_x: Vec<f32>,
    pub half_y: Vec<f32>,
    pub shapes: Vec<Shape>,
    /// Metadata only. **Never read by the solver**.
    pub is_static: Vec<bool>,
}

impl BodyStore {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pos_x.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos_x.is_empty()
    }

    pub fn reserve(&mut self, n: usize) {
        self.pos_x.reserve(n);
        self.pos_y.reserve(n);
        self.vel_x.reserve(n);
        self.vel_y.reserve(n);
        self.acc_x.reserve(n);
        self.acc_y.reserve(n);
        self.inv_mass.reserve(n);
        self.mass.reserve(n);
        self.restitution.reserve(n);
        self.half_x.reserve(n);
        self.half_y.reserve(n);
        self.shapes.reserve(n);
        self.is_static.reserve(n);
    }

    pub fn add(&mut self, b: RigidBody) -> BodyId {
        let id = BodyId(self.len() as u32);
        let h = b.shape.aabb_half_extents();
        self.pos_x.push(b.position.x);
        self.pos_y.push(b.position.y);
        self.vel_x.push(b.velocity.x);
        self.vel_y.push(b.velocity.y);
        self.acc_x.push(b.acceleration.x);
        self.acc_y.push(b.acceleration.y);
        self.inv_mass.push(b.inv_mass);
        self.mass.push(b.mass);
        self.restitution.push(b.restitution);
        self.half_x.push(h.x);
        self.half_y.push(h.y);
        self.shapes.push(b.shape);
        self.is_static.push(b.is_static);
        id
    }

    #[inline(always)]
    pub fn position(&self, id: BodyId) -> Vec2 {
        let i = id.index();
        Vec2::new(self.pos_x[i], self.pos_y[i])
    }
    #[inline(always)]
    pub fn velocity(&self, id: BodyId) -> Vec2 {
        let i = id.index();
        Vec2::new(self.vel_x[i], self.vel_y[i])
    }
    pub fn set_position(&mut self, id: BodyId, p: Vec2) {
        let i = id.index();
        self.pos_x[i] = p.x;
        self.pos_y[i] = p.y;
    }
    pub fn set_velocity(&mut self, id: BodyId, v: Vec2) {
        let i = id.index();
        // Multiply by (inv_mass > 0) would be branch-free; plain assignment is fine here
        // because this is a user-facing API, not the solver. Static bodies stay still:
        if self.inv_mass[i] > 0.0 {
            self.vel_x[i] = v.x;
            self.vel_y[i] = v.y;
        }
    }

    pub fn apply_force(&mut self, id: BodyId, f: Vec2) {
        let i = id.index();
        self.acc_x[i] += f.x * self.inv_mass[i];
        self.acc_y[i] += f.y * self.inv_mass[i];
    }
    pub fn apply_impulse(&mut self, id: BodyId, j: Vec2) {
        let i = id.index();
        self.vel_x[i] += j.x * self.inv_mass[i];
        self.vel_y[i] += j.y * self.inv_mass[i];
    }

    /// Snapshot as an AoS descriptor (mostly for tests/debug).
    pub fn get(&self, id: BodyId) -> RigidBody {
        let i = id.index();
        RigidBody {
            position: Vec2::new(self.pos_x[i], self.pos_y[i]),
            velocity: Vec2::new(self.vel_x[i], self.vel_y[i]),
            acceleration: Vec2::new(self.acc_x[i], self.acc_y[i]),
            mass: self.mass[i],
            inv_mass: self.inv_mass[i],
            restitution: self.restitution[i],
            is_static: self.is_static[i],
            shape: self.shapes[i],
        }
    }

    // ------------------------------------------------------------------
    // Integration kernels
    // ------------------------------------------------------------------

    /// Velocity update (semi-implicit Euler, first half):
    ///
    /// ```text
    /// a_total = acc_user + g·[inv_mass > 0]          (gravity doesn't depend on mass)
    /// v      ← (v + a_total·dt) · damp               damp = 1/(1 + c·dt)
    /// ```
    /// `damp` is the *implicit-Euler* solution of  v' = −c·v , unconditionally stable
    /// (unlike v·(1 − c·dt), which flips sign when c·dt > 1).
    ///
    /// The `[inv_mass > 0]` gate is computed as a **lane mask** (cmpgt + and), not a branch:
    /// static bodies get a = 0 and v stays exactly 0 (0·damp = 0).
    ///
    /// Accumulated user accelerations are consumed and zeroed.
    pub fn integrate_velocities(&mut self, dt: f32, gravity: Vec2, damp: f32, simd: bool) {
        if simd {
            self.integrate_velocities_simd(dt, gravity, damp)
        } else {
            self.integrate_velocities_scalar(dt, gravity, damp)
        }
    }

    #[inline(never)] // keep it a distinct symbol so profilers/benchmarks can see it
    fn integrate_velocities_simd(&mut self, dt: f32, gravity: Vec2, damp: f32) {
        let n = self.len();
        let split = n - n % LANES;
        let (vdt, vdamp, vgx, vgy, vzero) =
            (F32x4::splat(dt), F32x4::splat(damp), F32x4::splat(gravity.x), F32x4::splat(gravity.y), F32x4::splat(0.0));
        // 4 bodies per iteration: 6 loads, 2 stores.
        let it = self.vel_x[..split]
            .chunks_exact_mut(LANES)
            .zip(self.vel_y[..split].chunks_exact_mut(LANES))
            .zip(self.acc_x[..split].chunks_exact_mut(LANES))
            .zip(self.acc_y[..split].chunks_exact_mut(LANES))
            .zip(self.inv_mass[..split].chunks_exact(LANES));
        for ((((vx, vy), ax), ay), im) in it {
            let dynamic = F32x4::load(as4(im)).gt(vzero);
            let tx = F32x4::load(as4(ax)) + vgx.and_mask(dynamic);
            let ty = F32x4::load(as4(ay)) + vgy.and_mask(dynamic);
            (((F32x4::load(as4(vx)) + tx * vdt) * vdamp)).store(as4_mut(vx));
            (((F32x4::load(as4(vy)) + ty * vdt) * vdamp)).store(as4_mut(vy));
            F32x4::splat(0.0).store(as4_mut(ax));
            F32x4::splat(0.0).store(as4_mut(ay));
        }
        // Scalar tail (n % 4 bodies) using the *same* operation order ⇒ same bits.
        for i in split..n {
            self.integrate_velocity_one(i, dt, gravity, damp);
        }
    }

    #[inline(never)]
    fn integrate_velocities_scalar(&mut self, dt: f32, gravity: Vec2, damp: f32) {
        for i in 0..self.len() {
            self.integrate_velocity_one(i, dt, gravity, damp);
        }
    }

    #[inline(always)]
    fn integrate_velocity_one(&mut self, i: usize, dt: f32, g: Vec2, damp: f32) {
        let dynamic = self.inv_mass[i] > 0.0;
        let gx = if dynamic { g.x } else { 0.0 }; // compiles to a select, not a branch
        let gy = if dynamic { g.y } else { 0.0 };
        let tx = self.acc_x[i] + gx;
        let ty = self.acc_y[i] + gy;
        self.vel_x[i] = (self.vel_x[i] + tx * dt) * damp;
        self.vel_y[i] = (self.vel_y[i] + ty * dt) * damp;
        self.acc_x[i] = 0.0;
        self.acc_y[i] = 0.0;
    }

    /// Position update (semi-implicit Euler, second half): `x ← x + v·dt`.
    pub fn integrate_positions(&mut self, dt: f32, simd: bool) {
        if simd {
            self.integrate_positions_simd(dt)
        } else {
            self.integrate_positions_scalar(dt)
        }
    }

    #[inline(never)]
    fn integrate_positions_simd(&mut self, dt: f32) {
        let n = self.len();
        let split = n - n % LANES;
        let vdt = F32x4::splat(dt);
        let it = self.pos_x[..split]
            .chunks_exact_mut(LANES)
            .zip(self.pos_y[..split].chunks_exact_mut(LANES))
            .zip(self.vel_x[..split].chunks_exact(LANES))
            .zip(self.vel_y[..split].chunks_exact(LANES));
        for (((px, py), vx), vy) in it {
            (F32x4::load(as4(px)) + F32x4::load(as4(vx)) * vdt).store(as4_mut(px));
            (F32x4::load(as4(py)) + F32x4::load(as4(vy)) * vdt).store(as4_mut(py));
        }
        for i in split..n {
            self.pos_x[i] += self.vel_x[i] * dt;
            self.pos_y[i] += self.vel_y[i] * dt;
        }
    }

    #[inline(never)]
    fn integrate_positions_scalar(&mut self, dt: f32) {
        for i in 0..self.len() {
            self.pos_x[i] += self.vel_x[i] * dt;
            self.pos_y[i] += self.vel_y[i] * dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::Rng;

    fn random_store(n: usize, seed: u32) -> BodyStore {
        let mut rng = Rng::new(seed);
        let mut s = BodyStore::new();
        for i in 0..n {
            let shape = Shape::circle(rng.range(0.1, 0.5));
            let p = Vec2::new(rng.range(-50.0, 50.0), rng.range(-50.0, 50.0));
            let b = if i % 7 == 0 { RigidBody::fixed(shape, p) } else { RigidBody::dynamic(shape, 1.0, p) };
            let id = s.add(b.with_velocity(Vec2::new(rng.range(-5.0, 5.0), rng.range(-5.0, 5.0))));
            s.apply_force(id, Vec2::new(rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)));
        }
        s
    }

    fn bits(s: &BodyStore) -> Vec<u32> {
        let mut v = Vec::new();
        for i in 0..s.len() {
            v.extend([s.pos_x[i], s.pos_y[i], s.vel_x[i], s.vel_y[i], s.acc_x[i], s.acc_y[i]].map(f32::to_bits));
        }
        v
    }

    #[test]
    fn simd_and_scalar_integration_are_bit_identical() {
        // n = 1003 exercises the 3-element scalar tail.
        for n in [1usize, 3, 4, 5, 64, 1003] {
            let mut a = random_store(n, 99);
            let mut b = a.clone();
            for _ in 0..50 {
                a.integrate_velocities(1.0 / 60.0, Vec2::new(0.0, -9.81), 0.99, true);
                a.integrate_positions(1.0 / 60.0, true);
                b.integrate_velocities(1.0 / 60.0, Vec2::new(0.0, -9.81), 0.99, false);
                b.integrate_positions(1.0 / 60.0, false);
            }
            assert_eq!(bits(&a), bits(&b), "n={n}");
        }
    }

    #[test]
    fn static_bodies_never_move_and_never_branch_on_flag() {
        let mut s = random_store(64, 5);
        let before: Vec<_> = (0..64).filter(|i| i % 7 == 0).map(|i| (s.pos_x[i], s.pos_y[i])).collect();
        // Force gravity + try to push them: inv_mass=0 must absorb everything.
        for i in 0..64 {
            s.apply_impulse(BodyId(i as u32), Vec2::new(100.0, 100.0));
        }
        for _ in 0..100 {
            s.integrate_velocities(1.0 / 60.0, Vec2::new(0.0, -9.81), 1.0, true);
            s.integrate_positions(1.0 / 60.0, true);
        }
        let after: Vec<_> = (0..64).filter(|i| i % 7 == 0).map(|i| (s.pos_x[i], s.pos_y[i])).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn matches_rigid_body_reference_integrator() {
        // Free fall from rest: v = g·t,  x = x0 + Σ v_k·dt  (symplectic Euler closed form)
        let mut rb = RigidBody::dynamic(Shape::circle(0.5), 1.0, Vec2::new(0.0, 100.0));
        rb.acceleration = Vec2::new(0.0, -9.81);
        let mut s = BodyStore::new();
        let id = s.add(RigidBody::dynamic(Shape::circle(0.5), 1.0, Vec2::new(0.0, 100.0)));
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            let a = rb.acceleration;
            rb.integrate(dt);
            rb.acceleration = a; // constant "gravity" for the reference
            s.integrate_velocities(dt, Vec2::new(0.0, -9.81), 1.0, true);
            s.integrate_positions(dt, true);
        }
        let p = s.position(id);
        assert_eq!(p.y.to_bits(), rb.position.y.to_bits());
        // Symplectic Euler closed form: y_N = y0 − g·dt²·N(N+1)/2  (differs from ½gt² by ½·g·dt·t ≈ 0.16 m)
        let n = 120.0f32;
        let exact = 100.0 - 9.81 * dt * dt * n * (n + 1.0) / 2.0;
        assert!((p.y - exact).abs() < 0.01, "{} vs {}", p.y, exact);
    }

    #[test]
    fn force_and_impulse_scale_by_inverse_mass() {
        let mut b = RigidBody::dynamic(Shape::aligned_box(Vec2::new(1.0, 1.0)), 1.0, Vec2::ZERO); // m = 4
        b.apply_impulse(Vec2::new(8.0, 0.0));
        assert_eq!(b.velocity.x, 2.0);
        b.apply_force(Vec2::new(0.0, 8.0));
        assert_eq!(b.acceleration.y, 2.0);
        let mut f = RigidBody::fixed(Shape::circle(1.0), Vec2::ZERO);
        f.apply_impulse(Vec2::new(1e6, 1e6));
        f.apply_force(Vec2::new(1e6, 1e6));
        assert_eq!((f.velocity, f.acceleration), (Vec2::ZERO, Vec2::ZERO));
    }
}
