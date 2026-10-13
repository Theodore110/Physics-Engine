//! Narrow phase: exact collision tests and contact generation.
//!
//! **Convention**: for a pair (A, B) the contact `normal` is a unit vector pointing
//! from A towards B, and `penetration > 0` is the overlap depth along it, i.e.
//! translating B by `+normal·penetration` (or A by `−normal·penetration`) separates
//! the shapes. (Minimum-translation-vector property — verified by the fuzz tests.)

use crate::body::{BodyId, BodyStore};
use crate::broad_phase::Pair;
use crate::shape::Shape;
use crate::vec2::Vec2;

/// One contact point between two bodies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Unit normal, A → B.
    pub normal: Vec2,
    /// Overlap depth along `normal` (> 0).
    pub penetration: f32,
    /// A representative world-space point inside the overlap region.
    pub contact_point: Vec2,
    pub body_a: BodyId,
    pub body_b: BodyId,
}

/// Geometry-only result of a shape test (no body ids).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Manifold {
    pub normal: Vec2,
    pub penetration: f32,
    pub point: Vec2,
}

/// Deterministic fallback normal for exactly coincident centres.
const FALLBACK_NORMAL: Vec2 = Vec2::UNIT_Y;

// ---------------------------------------------------------------------------
// Circle – Circle
// ---------------------------------------------------------------------------

/// Overlap iff |pb − pa| < ra + rb.
///
/// We compare **squared** distances first (`d² < (ra+rb)²`) so the common
/// "not touching" case costs 3 mul + 2 add and no `sqrt`.
#[inline]
pub fn circle_circle(pa: Vec2, ra: f32, pb: Vec2, rb: f32) -> Option<Manifold> {
    let d = pb - pa;
    let r = ra + rb;
    let d2 = d.length_squared();
    if d2 >= r * r {
        return None;
    }
    let dist = d2.sqrt();
    let (normal, dist) = if dist > 1.0e-6 { (d * (1.0 / dist), dist) } else { (FALLBACK_NORMAL, 0.0) };
    let penetration = r - dist;
    // Representative point = midpoint of the overlap *interval along the normal*.
    // Along the centre line A spans [−ra, +ra] and B spans [dist−rb, dist+rb], so the
    // shared interval is [lo, hi] = [max(−ra, dist−rb), min(ra, dist+rb)].
    //  • partial overlap (a lens): lo = dist−rb, hi = ra  ⇒ midpoint (ra + dist − rb)/2
    //  • B inside A: interval = B's diameter ⇒ B's centre;  A inside B ⇒ A's centre.
    // (The naive lens formula alone would land *outside* the smaller circle when one
    //  circle contains the other — found by the fuzz test.)
    let lo = if dist - rb > -ra { dist - rb } else { -ra };
    let hi = if dist + rb < ra { dist + rb } else { ra };
    let point = pa + normal * (0.5 * (lo + hi));
    Some(Manifold { normal, penetration, point })
}

// ---------------------------------------------------------------------------
// Circle – Box
// ---------------------------------------------------------------------------

/// Circle (centre `c`, radius `r`) vs OBB (centre `pb`, half extents `h`, local X axis `ax`).
/// Returns the manifold with the normal pointing **box → circle**.
///
/// Method (closest point on the OBB):
///  1. Express the circle centre in the box frame: `l = (rel·u, rel·v)`, u=ax, v=⊥ax.
///  2. `q = clamp(l, −h, +h)` is the closest point of the box to the centre.
///  3. If `l ≠ q` the centre is outside: `delta = l − q`, overlap iff |delta| < r,
///     normal = delta/|delta|, depth = r − |delta|.
///  4. If `l == q` the centre is inside: push out through the nearest face,
///     depth = r + (distance from centre to that face); the contact point is the
///     circle centre itself (it lies inside both shapes by definition).
fn circle_vs_box(c: Vec2, r: f32, pb: Vec2, h: Vec2, ax: Vec2) -> Option<Manifold> {
    let ay = ax.perpendicular();
    let rel = c - pb;
    let l = Vec2::new(rel.dot(ax), rel.dot(ay));
    let q = Vec2::new(l.x.clamp(-h.x, h.x), l.y.clamp(-h.y, h.y));
    let delta = l - q;
    let d2 = delta.length_squared();

    let (n_local, penetration, surface_local) = if d2 > 0.0 {
        if d2 >= r * r {
            return None;
        }
        let d = d2.sqrt();
        (delta * (1.0 / d), r - d, q)
    } else {
        // inside: distances to the two nearest faces on each axis
        let (dx, dy) = (h.x - l.x.abs(), h.y - l.y.abs());
        if dx < dy {
            let s = if l.x >= 0.0 { 1.0 } else { -1.0 };
            (Vec2::new(s, 0.0), r + dx, l)
        } else {
            let s = if l.y >= 0.0 { 1.0 } else { -1.0 };
            (Vec2::new(0.0, s), r + dy, l)
        }
    };
    // back to world space
    let normal = ax * n_local.x + ay * n_local.y;
    let point = pb + ax * surface_local.x + ay * surface_local.y;
    Some(Manifold { normal, penetration, point })
}

// ---------------------------------------------------------------------------
// Box – Box  (Separating Axis Theorem)
// ---------------------------------------------------------------------------

/// SAT for two oriented boxes.
///
/// **Theorem**: two convex shapes are disjoint iff there is a *separating axis*; for
/// two 2D convex polygons it suffices to test the edge normals of both. A box has
/// only 2 distinct edge normals, so we test 4 axes in total: `ua, va, ub, vb`.
///
/// For a unit axis L, the projection of a box of half extents (hx, hy) and local axes
/// (u, v) is an interval centred on its centre with radius
/// `r = hx·|L·u| + hy·|L·v|`.  The intervals overlap by
/// `overlap(L) = rA + rB − |(pb − pa)·L|`.
/// * `overlap ≤ 0` on any axis ⇒ **separated, return immediately** (early-out).
/// * otherwise the axis with the **smallest** overlap is the minimum translation
///   vector: normal = ±L (signed so it points A → B), depth = that overlap.
///
/// Ties (e.g. two aligned boxes where `ua` and `ub` give identical overlap) go to A's
/// axes, and a B axis must win by more than `AXIS_TOL`. This hysteresis stops the
/// reported normal flipping between near-equal axes from frame to frame — a classic
/// source of jitter in box stacks.
fn box_box(pa: Vec2, ha: Vec2, ua: Vec2, pb: Vec2, hb: Vec2, ub: Vec2) -> Option<Manifold> {
    const AXIS_TOL: f32 = 1.0e-4;
    let va = ua.perpendicular();
    let vb = ub.perpendicular();
    let d = pb - pa;

    // Rotation of B's frame expressed in A's frame (absolute values for the radius formula).
    let (c00, c01) = (ua.dot(ub).abs(), ua.dot(vb).abs());
    let (c10, c11) = (va.dot(ub).abs(), va.dot(vb).abs());

    // Axis ua :  rA = ha.x ;  rB = hb.x|ua·ub| + hb.y|ua·vb|
    let proj = d.dot(ua);
    let ov = ha.x + (hb.x * c00 + hb.y * c01) - proj.abs();
    if ov <= 0.0 {
        return None;
    }
    let mut best = ov;
    let mut best_axis = ua;
    let mut best_sign = if proj >= 0.0 { 1.0f32 } else { -1.0 };

    // Axis va
    let proj = d.dot(va);
    let ov = ha.y + (hb.x * c10 + hb.y * c11) - proj.abs();
    if ov <= 0.0 {
        return None;
    }
    if ov < best - AXIS_TOL {
        best = ov;
        best_axis = va;
        best_sign = if proj >= 0.0 { 1.0 } else { -1.0 };
    }

    // Axis ub :  rB = hb.x ;  rA = ha.x|ub·ua| + ha.y|ub·va|
    let proj = d.dot(ub);
    let ov = hb.x + (ha.x * c00 + ha.y * c10) - proj.abs();
    if ov <= 0.0 {
        return None;
    }
    if ov < best - AXIS_TOL {
        best = ov;
        best_axis = ub;
        best_sign = if proj >= 0.0 { 1.0 } else { -1.0 };
    }

    // Axis vb
    let proj = d.dot(vb);
    let ov = hb.y + (ha.x * c01 + ha.y * c11) - proj.abs();
    if ov <= 0.0 {
        return None;
    }
    if ov < best - AXIS_TOL {
        best = ov;
        best_axis = vb;
        best_sign = if proj >= 0.0 { 1.0 } else { -1.0 };
    }

    let normal = best_axis * best_sign;
    let point = overlap_centroid(pa, ha, ua, pb, hb, ub);
    Some(Manifold { normal, penetration: best, point })
}

/// Contact point = centroid of the polygon `A ∩ B`.
///
/// B's four corners are clipped against A's four half-planes (Sutherland–Hodgman).
/// Since both boxes are convex, the intersection is convex with ≤ 8 vertices; fixed
/// stack arrays are used so there is no allocation. If clipping degenerates (only
/// possible through rounding when the overlap is ~0) fall back to the midpoint of the centres.
fn overlap_centroid(pa: Vec2, ha: Vec2, ua: Vec2, pb: Vec2, hb: Vec2, ub: Vec2) -> Vec2 {
    let vb = ub.perpendicular();
    let va = ua.perpendicular();
    let (ex, ey) = (ub * hb.x, vb * hb.y);
    let mut poly = [Vec2::ZERO; 8];
    poly[0] = pb - ex - ey;
    poly[1] = pb + ex - ey;
    poly[2] = pb + ex + ey;
    poly[3] = pb - ex + ey;
    let mut n = 4usize;

    // keep points with  sign·((p − pa)·axis) ≤ half
    let planes = [(ua, 1.0f32, ha.x), (ua, -1.0, ha.x), (va, 1.0, ha.y), (va, -1.0, ha.y)];
    for (axis, sign, half) in planes {
        let mut out = [Vec2::ZERO; 8];
        let mut m = 0usize;
        for i in 0..n {
            let (p, q) = (poly[i], poly[(i + 1) % n]);
            let dp = sign * (p - pa).dot(axis) - half; // ≤ 0 inside
            let dq = sign * (q - pa).dot(axis) - half;
            if dp <= 0.0 && m < 8 {
                out[m] = p;
                m += 1;
            }
            if (dp < 0.0 && dq > 0.0) || (dp > 0.0 && dq < 0.0) {
                // edge crosses the plane: p + t(q − p) with t = dp/(dp − dq)
                let t = dp / (dp - dq);
                if m < 8 {
                    out[m] = p + (q - p) * t;
                    m += 1;
                }
            }
        }
        poly = out;
        n = m;
        if n == 0 {
            return (pa + pb) * 0.5;
        }
    }
    let mut c = Vec2::ZERO;
    for p in &poly[..n] {
        c += *p;
    }
    c * (1.0 / n as f32)
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Test any two shapes. Normal points from `a` to `b`.
#[inline]
pub fn collide(sa: &Shape, pa: Vec2, sb: &Shape, pb: Vec2) -> Option<Manifold> {
    match (*sa, *sb) {
        (Shape::Circle { radius: ra }, Shape::Circle { radius: rb }) => circle_circle(pa, ra, pb, rb),
        // A = circle, B = box:  box→circle normal points B→A, so negate for A→B
        (Shape::Circle { radius }, Shape::Box { half_extents, axis, .. }) => {
            circle_vs_box(pa, radius, pb, half_extents, axis).map(|m| Manifold { normal: -m.normal, ..m })
        }
        // A = box, B = circle:  box→circle is already A→B
        (Shape::Box { half_extents, axis, .. }, Shape::Circle { radius }) => circle_vs_box(pb, radius, pa, half_extents, axis),
        (Shape::Box { half_extents: ha, axis: ua, .. }, Shape::Box { half_extents: hb, axis: ub, .. }) => {
            box_box(pa, ha, ua, pb, hb, ub)
        }
    }
}

/// Run the narrow phase over all broad-phase pairs, appending to `out`.
/// `out` is cleared first but its capacity is kept ⇒ no per-frame allocation after warm-up.
pub fn detect(bodies: &BodyStore, pairs: &[Pair], out: &mut Vec<Contact>) {
    out.clear();
    for p in pairs {
        let (i, j) = (p.a as usize, p.b as usize);
        let (pa, pb) = (Vec2::new(bodies.pos_x[i], bodies.pos_y[i]), Vec2::new(bodies.pos_x[j], bodies.pos_y[j]));
        if let Some(m) = collide(&bodies.shapes[i], pa, &bodies.shapes[j], pb) {
            out.push(Contact {
                normal: m.normal,
                penetration: m.penetration,
                contact_point: m.point,
                body_a: BodyId(p.a),
                body_b: BodyId(p.b),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_circle_basic() {
        let m = circle_circle(Vec2::ZERO, 1.0, Vec2::new(1.5, 0.0), 1.0).unwrap();
        assert_eq!(m.normal, Vec2::UNIT_X);
        assert!((m.penetration - 0.5).abs() < 1e-6);
        assert!((m.point.x - 0.75).abs() < 1e-6);
        assert!(circle_circle(Vec2::ZERO, 1.0, Vec2::new(2.0, 0.0), 1.0).is_none()); // exactly touching
        // coincident centres: finite, unit normal
        let m = circle_circle(Vec2::ZERO, 1.0, Vec2::ZERO, 1.0).unwrap();
        assert_eq!(m.normal, Vec2::UNIT_Y);
        assert_eq!(m.penetration, 2.0);
    }

    #[test]
    fn circle_box_cases() {
        let bx = Shape::aligned_box(Vec2::new(1.0, 1.0));
        let c = Shape::circle(0.5);
        // circle above the top face (A = box, B = circle): normal +Y, depth 0.25
        let m = collide(&bx, Vec2::ZERO, &c, Vec2::new(0.0, 1.25)).unwrap();
        assert!((m.normal - Vec2::UNIT_Y).length() < 1e-6 && (m.penetration - 0.25).abs() < 1e-6);
        // same pair swapped: normal flips
        let m2 = collide(&c, Vec2::new(0.0, 1.25), &bx, Vec2::ZERO).unwrap();
        assert!((m2.normal + Vec2::UNIT_Y).length() < 1e-6 && (m2.penetration - 0.25).abs() < 1e-6);
        // corner region: nearest feature is the corner (1,1)
        let m = collide(&bx, Vec2::ZERO, &c, Vec2::new(1.3, 1.3)).unwrap();
        assert!(m.normal.x > 0.0 && (m.normal.x - m.normal.y).abs() < 1e-6);
        // centre inside the box
        let m = collide(&bx, Vec2::ZERO, &c, Vec2::new(0.9, 0.1)).unwrap();
        assert!((m.normal - Vec2::UNIT_X).length() < 1e-6 && (m.penetration - 0.6).abs() < 1e-6);
        // far away
        assert!(collide(&bx, Vec2::ZERO, &c, Vec2::new(3.0, 0.0)).is_none());
    }

    #[test]
    fn box_box_aligned_and_rotated() {
        let a = Shape::aligned_box(Vec2::new(1.0, 1.0));
        let m = collide(&a, Vec2::ZERO, &a, Vec2::new(1.5, 0.2)).unwrap();
        assert_eq!(m.normal, Vec2::UNIT_X);
        assert!((m.penetration - 0.5).abs() < 1e-6);
        // overlap region is x∈[0.5,1], y∈[−0.8,1] ⇒ centroid (0.75, 0.1)
        assert!((m.point - Vec2::new(0.75, 0.1)).length() < 1e-5, "{:?}", m.point);
        assert!(collide(&a, Vec2::ZERO, &a, Vec2::new(2.0, 0.0)).is_none());
        assert!(collide(&a, Vec2::ZERO, &a, Vec2::new(0.0, -2.5)).is_none());
        // 45° diamond whose tip pokes 0.1 into the top face of the unit square
        let d = Shape::oriented_box(Vec2::new(1.0, 1.0), core::f32::consts::FRAC_PI_4);
        let tip = 2f32.sqrt();
        let m = collide(&a, Vec2::ZERO, &d, Vec2::new(0.0, 1.0 + tip - 0.1)).unwrap();
        assert!(m.normal.y > 0.99 && (m.penetration - 0.1).abs() < 1e-4, "{:?}", m);
    }
}
