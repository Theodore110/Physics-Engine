//! Collision shapes: circle and oriented box (OBB).
//!
//! Bodies in this engine **translate but do not rotate** (angular dynamics are on
//! the roadmap). A box therefore carries a *fixed* orientation `rotation` chosen at
//! construction. The consequence is that its world AABB half-extents never change,
//! so they are computed once here and stored in the body SoA (`half_x/half_y`).

use crate::util::unit_from_angle;
use crate::vec2::Vec2;

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec2,
    pub max: Vec2,
}

impl Aabb {
    #[inline(always)]
    pub fn from_center_half(c: Vec2, h: Vec2) -> Aabb {
        Aabb { min: c - h, max: c + h }
    }
    /// Closed-interval overlap test on both axes (the SAT of two AABBs: 2 axes).
    #[inline(always)]
    pub fn overlaps(&self, o: &Aabb) -> bool {
        self.min.x <= o.max.x && self.max.x >= o.min.x && self.min.y <= o.max.y && self.max.y >= o.min.y
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Circle {
        radius: f32,
    },
    /// Oriented box. `axis` = (cos rotation, sin rotation) is the box's local +X axis
    /// in world space (cached so the hot path never evaluates trig).
    Box {
        half_extents: Vec2,
        rotation: f32,
        axis: Vec2,
    },
}

impl Shape {
    pub fn circle(radius: f32) -> Shape {
        assert!(radius > 0.0 && radius.is_finite(), "circle radius must be positive & finite");
        Shape::Circle { radius }
    }

    /// Box with half-extents `(hx, hy)` rotated CCW by `rotation` radians.
    pub fn oriented_box(half_extents: Vec2, rotation: f32) -> Shape {
        assert!(
            half_extents.x > 0.0 && half_extents.y > 0.0 && half_extents.is_finite() && rotation.is_finite(),
            "box half extents must be positive & finite"
        );
        Shape::Box { half_extents, rotation, axis: unit_from_angle(rotation) }
    }

    /// Axis-aligned box (rotation 0).
    pub fn aligned_box(half_extents: Vec2) -> Shape {
        Self::oriented_box(half_extents, 0.0)
    }

    /// Half-extents of the world-space AABB around the shape's centre.
    ///
    /// For an OBB with half-extents (hx, hy) and axes u=(c,s), v=(−s,c):
    /// the support along world-X is hx·|u.x| + hy·|v.x| = hx·|c| + hy·|s|,
    /// and along world-Y it is hx·|s| + hy·|c|.
    #[inline]
    pub fn aabb_half_extents(&self) -> Vec2 {
        match *self {
            Shape::Circle { radius } => Vec2::splat(radius),
            Shape::Box { half_extents: h, axis, .. } => {
                let (c, s) = (axis.x.abs(), axis.y.abs());
                Vec2::new(h.x * c + h.y * s, h.x * s + h.y * c)
            }
        }
    }

    /// The AABB of this shape when its centre is at `position`.
    #[inline]
    pub fn aabb(&self, position: Vec2) -> Aabb {
        Aabb::from_center_half(position, self.aabb_half_extents())
    }

    /// Area in m².
    pub fn area(&self) -> f32 {
        match *self {
            Shape::Circle { radius } => core::f32::consts::PI * radius * radius,
            Shape::Box { half_extents: h, .. } => 4.0 * h.x * h.y,
        }
    }

    /// Mass from uniform areal density (kg/m²): m = ρ·A.
    pub fn mass(&self, density: f32) -> f32 {
        density * self.area()
    }

    /// Axis-aligned (in local frame) box axes (u, v) for SAT; circle has none.
    #[inline(always)]
    pub fn box_axes(axis: Vec2) -> (Vec2, Vec2) {
        (axis, axis.perpendicular())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_from_density() {
        let c = Shape::circle(1.0);
        assert!((c.mass(2.0) - 2.0 * core::f32::consts::PI).abs() < 1e-5);
        let b = Shape::aligned_box(Vec2::new(1.0, 0.5));
        assert_eq!(b.mass(3.0), 3.0 * 2.0);
    }

    #[test]
    fn rotated_box_aabb_is_tight() {
        // 45°-rotated unit square (half 1,1) has AABB half-extent sqrt(2).
        let s = Shape::oriented_box(Vec2::new(1.0, 1.0), core::f32::consts::FRAC_PI_4);
        let h = s.aabb_half_extents();
        assert!((h.x - 2f32.sqrt()).abs() < 1e-5 && (h.y - 2f32.sqrt()).abs() < 1e-5);
        let a = s.aabb(Vec2::new(10.0, -3.0));
        assert!(a.min.x < 10.0 && a.max.x > 10.0 && a.min.y < -3.0 && a.max.y > -3.0);
        assert!(a.overlaps(&a));
        assert!(!a.overlaps(&Aabb { min: Vec2::new(20.0, 20.0), max: Vec2::new(21.0, 21.0) }));
    }
}
