//! 2D vector of `f32`.
//!
//! ## Why `f32` and not `f64`?
//! * **Cache density**: 8 bytes per vector vs 16. The solver gathers velocities at
//!   random indices; twice the bytes means twice the cache lines touched.
//! * **SIMD width**: an SSE register holds 4 lanes of `f32` but only 2 of `f64`;
//!   AVX2 holds 8 vs 4. The integration/broad-phase kernels are bandwidth-bound,
//!   so lanes-per-op translates directly to throughput.
//! * **Precision is sufficient**: `f32` has 24 mantissa bits (~1.2e-7 relative).
//!   At 100 m from the origin the absolute resolution is ~7.6 µm, three orders of
//!   magnitude below the 1 cm contact slop. Beyond ~10 km from the origin you would
//!   want origin-rebasing (see roadmap) or `f64`.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 2D vector. `#[repr(C)]` so layout is exactly `[x, y]`, 8 bytes, align 4.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };
    pub const ONE: Vec2 = Vec2 { x: 1.0, y: 1.0 };
    pub const UNIT_X: Vec2 = Vec2 { x: 1.0, y: 0.0 };
    pub const UNIT_Y: Vec2 = Vec2 { x: 0.0, y: 1.0 };

    #[inline(always)]
    pub const fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }
    #[inline(always)]
    pub const fn splat(v: f32) -> Vec2 {
        Vec2 { x: v, y: v }
    }

    /// a · b = |a||b|cosθ
    #[inline(always)]
    pub fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }

    /// 2D "cross product": the z-component of the 3D cross product of (a,0) × (b,0).
    /// a × b = ax·by − ay·bx = |a||b|sinθ  (positive when b is CCW from a).
    #[inline(always)]
    pub fn cross(self, o: Vec2) -> f32 {
        self.x * o.y - self.y * o.x
    }

    #[inline(always)]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// `sqrt` is correctly rounded by IEEE-754 (sqrtss), hence deterministic.
    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Unit vector, or `ZERO` for (near-)zero input. Never produces NaN/inf.
    #[inline(always)]
    pub fn normalize(self) -> Vec2 {
        let l2 = self.length_squared();
        if l2 > 1.0e-24 {
            self * (1.0 / l2.sqrt())
        } else {
            Vec2::ZERO
        }
    }

    #[inline(always)]
    pub fn distance_squared(self, o: Vec2) -> f32 {
        (self - o).length_squared()
    }
    #[inline(always)]
    pub fn distance(self, o: Vec2) -> f32 {
        (self - o).length()
    }

    /// Linear interpolation: `self + (o − self)·t`.
    #[inline(always)]
    pub fn lerp(self, o: Vec2, t: f32) -> Vec2 {
        self + (o - self) * t
    }

    /// Rotate +90° (CCW): (x, y) → (−y, x).  `a.dot(a.perpendicular()) == 0`.
    #[inline(always)]
    pub fn perpendicular(self) -> Vec2 {
        Vec2 { x: -self.y, y: self.x }
    }

    #[inline(always)]
    pub fn abs(self) -> Vec2 {
        Vec2 { x: self.x.abs(), y: self.y.abs() }
    }
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
    /// Component-wise product.
    #[inline(always)]
    pub fn mul_elem(self, o: Vec2) -> Vec2 {
        Vec2 { x: self.x * o.x, y: self.y * o.y }
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    #[inline(always)]
    fn add(self, o: Vec2) -> Vec2 {
        Vec2 { x: self.x + o.x, y: self.y + o.y }
    }
}
impl Sub for Vec2 {
    type Output = Vec2;
    #[inline(always)]
    fn sub(self, o: Vec2) -> Vec2 {
        Vec2 { x: self.x - o.x, y: self.y - o.y }
    }
}
impl Mul<f32> for Vec2 {
    type Output = Vec2;
    #[inline(always)]
    fn mul(self, s: f32) -> Vec2 {
        Vec2 { x: self.x * s, y: self.y * s }
    }
}
impl Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline(always)]
    fn mul(self, v: Vec2) -> Vec2 {
        v * self
    }
}
impl Div<f32> for Vec2 {
    type Output = Vec2;
    #[inline(always)]
    fn div(self, s: f32) -> Vec2 {
        Vec2 { x: self.x / s, y: self.y / s }
    }
}
impl Neg for Vec2 {
    type Output = Vec2;
    #[inline(always)]
    fn neg(self) -> Vec2 {
        Vec2 { x: -self.x, y: -self.y }
    }
}
impl AddAssign for Vec2 {
    #[inline(always)]
    fn add_assign(&mut self, o: Vec2) {
        self.x += o.x;
        self.y += o.y;
    }
}
impl SubAssign for Vec2 {
    #[inline(always)]
    fn sub_assign(&mut self, o: Vec2) {
        self.x -= o.x;
        self.y -= o.y;
    }
}
impl MulAssign<f32> for Vec2 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
    }
}
impl DivAssign<f32> for Vec2 {
    #[inline(always)]
    fn div_assign(&mut self, s: f32) {
        self.x /= s;
        self.y /= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_packed_pair() {
        assert_eq!(core::mem::size_of::<Vec2>(), 8);
        assert_eq!(core::mem::align_of::<Vec2>(), 4);
    }

    #[test]
    fn basic_algebra() {
        let a = Vec2::new(3.0, 4.0);
        let b = Vec2::new(-1.0, 2.0);
        assert_eq!(a + b, Vec2::new(2.0, 6.0));
        assert_eq!(a - b, Vec2::new(4.0, 2.0));
        assert_eq!(a * 2.0, Vec2::new(6.0, 8.0));
        assert_eq!(2.0 * a, a * 2.0);
        assert_eq!(a / 2.0, Vec2::new(1.5, 2.0));
        assert_eq!(-a, Vec2::new(-3.0, -4.0));
        assert_eq!(a.length(), 5.0);
        assert_eq!(a.length_squared(), 25.0);
        assert_eq!(a.dot(b), 5.0);
        assert_eq!(a.cross(b), 3.0 * 2.0 - 4.0 * -1.0);
        assert_eq!(a.distance(Vec2::ZERO), 5.0);
        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 1.0), b);
        assert_eq!(a.dot(a.perpendicular()), 0.0);
        assert_eq!(Vec2::UNIT_X.cross(Vec2::UNIT_Y), 1.0);
        let mut c = a;
        c += b; c -= b; c *= 2.0; c /= 2.0;
        assert_eq!(c, a);
    }

    #[test]
    fn normalize_is_safe() {
        assert_eq!(Vec2::ZERO.normalize(), Vec2::ZERO);
        assert_eq!(Vec2::splat(1e-30).normalize(), Vec2::ZERO);
        let n = Vec2::new(3.0, 4.0).normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
        assert!(Vec2::new(1e18, 1e18).normalize().is_finite());
    }
}
