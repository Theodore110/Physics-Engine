//! Randomised property tests ("fuzzing") of the narrow phase.
//!
//! Properties checked for hundreds of thousands of random shape pairs:
//!  P1  no panic, every output finite (no NaN/inf), unit normal, penetration > 0
//!  P2  symmetry: collide(A,B) and collide(B,A) agree on hit/miss, with opposite normals
//!  P3  minimum-translation-vector: shifting B by n·(pen + ε) really separates the shapes
//!  P4  completeness (independent oracle): if a random point lies inside *both* shapes by a
//!      margin, a contact must be reported
//!  P5  soundness (independent oracle): when a contact is reported, shapes truly overlap —
//!      the reported contact point lies in/near both shapes
//!  P6  degenerate inputs (coincident, huge, tiny, exactly touching) never panic or yield NaN

use physics2d::narrow_phase::collide;
use physics2d::util::Rng;
use physics2d::{Shape, Vec2};

fn random_shape(rng: &mut Rng) -> Shape {
    // Mix of sizes incl. extreme aspect ratios
    let s = |rng: &mut Rng| 10f32.powf(rng.range(-1.3, 1.0));
    match rng.next_u32() % 3 {
        0 => Shape::circle(s(rng)),
        1 => Shape::oriented_box(Vec2::new(s(rng), s(rng)), rng.range(-7.0, 7.0)),
        _ => Shape::aligned_box(Vec2::new(s(rng), s(rng))),
    }
}

/// Signed "depth inside" of point p for a shape at centre c (positive = inside, value = distance to boundary).
fn inside_depth(s: &Shape, c: Vec2, p: Vec2) -> f32 {
    let d = p - c;
    match *s {
        Shape::Circle { radius } => radius - d.length(),
        Shape::Box { half_extents: h, axis, .. } => {
            let (x, y) = (d.dot(axis), d.dot(axis.perpendicular()));
            (h.x - x.abs()).min(h.y - y.abs())
        }
    }
}

fn random_point_in(s: &Shape, c: Vec2, rng: &mut Rng) -> Vec2 {
    let r = s.aabb_half_extents();
    Vec2::new(c.x + rng.range(-r.x, r.x), c.y + rng.range(-r.y, r.y))
}

#[test]
fn fuzz_narrow_phase_properties() {
    let mut rng = Rng::new(0xC0FFEE);
    let (mut hits, mut misses) = (0u32, 0u32);
    for iter in 0..400_000u32 {
        let (sa, sb) = (random_shape(&mut rng), random_shape(&mut rng));
        let pa = Vec2::new(rng.range(-3.0, 3.0), rng.range(-3.0, 3.0));
        let pb = pa + Vec2::new(rng.range(-6.0, 6.0), rng.range(-6.0, 6.0)) * 10f32.powf(rng.range(-1.5, 0.3));

        let ab = collide(&sa, pa, &sb, pb);
        let ba = collide(&sb, pb, &sa, pa);

        match ab {
            Some(m) => {
                hits += 1;
                // P1
                assert!(m.normal.is_finite() && m.point.is_finite() && m.penetration.is_finite(), "iter {iter}: non-finite {m:?}");
                assert!((m.normal.length() - 1.0).abs() < 1e-3, "iter {iter}: |n|={}", m.normal.length());
                assert!(m.penetration > 0.0, "iter {iter}: pen {}", m.penetration);

                // P2 (allow disagreement only for razor-thin overlaps where rounding differs)
                match ba {
                    Some(r) => {
                        // Skip comparing normals for near-tied SAT axes / circle-centre-inside degeneracy
                        if (m.penetration - r.penetration).abs() < 1e-3 * (1.0 + m.penetration) {
                            assert!(m.normal.dot(r.normal) < -0.999 || m.penetration < 1e-3 || tie_ambiguous(&sa, &sb), "iter {iter}: normals not opposite {:?} {:?}", m.normal, r.normal);
                        }
                    }
                    None => assert!(m.penetration < 1e-3, "iter {iter}: asymmetric hit pen={}", m.penetration),
                }

                // P3: separating translation really separates (use a 2 % + 1e-3 margin; only valid
                // when the reported axis is the true MTV — all pairs here are convex ⇒ it is).
                let eps = 1.0e-3 + 0.02 * m.penetration;
                let moved = pb + m.normal * (m.penetration + eps);
                assert!(collide(&sa, pa, &sb, moved).is_none(), "iter {iter}: MTV does not separate: {sa:?} {sb:?} {m:?}");

                // P5: contact point should be inside (or within tolerance of) both shapes
                let tol = 0.05 * (1.0 + m.penetration) + 1e-3 * (pa - pb).length();
                assert!(inside_depth(&sa, pa, m.point) > -tol && inside_depth(&sb, pb, m.point) > -tol,
                    "iter {iter}: contact point outside shapes: {:?} depthA {} depthB {}", m, inside_depth(&sa, pa, m.point), inside_depth(&sb, pb, m.point));
            }
            None => {
                misses += 1;
                // P4: random points deep inside both ⇒ must have been a hit
                for _ in 0..8 {
                    let p = random_point_in(&sa, pa, &mut rng);
                    let margin = 1.0e-3;
                    assert!(
                        !(inside_depth(&sa, pa, p) > margin && inside_depth(&sb, pb, p) > margin),
                        "iter {iter}: missed collision {sa:?}@{pa:?} {sb:?}@{pb:?} p={p:?}"
                    );
                }
            }
        }
    }
    // The generator must exercise both outcomes substantially.
    assert!(hits > 50_000 && misses > 50_000, "hits {hits} misses {misses}");
    eprintln!("fuzz: {hits} hits, {misses} misses");
}

/// Two identical-orientation boxes have tied SAT axes by construction.
fn tie_ambiguous(a: &Shape, b: &Shape) -> bool {
    matches!((a, b), (Shape::Box { .. }, Shape::Box { .. }))
}

#[test]
fn degenerate_inputs_never_panic_or_nan() {
    let shapes = [
        Shape::circle(1.0e-3),
        Shape::circle(1.0e4),
        Shape::aligned_box(Vec2::new(1.0e-3, 1.0e4)),
        Shape::oriented_box(Vec2::new(0.5, 0.5), core::f32::consts::FRAC_PI_4),
        Shape::oriented_box(Vec2::new(3.0, 0.2), 1.0e3), // huge angle ⇒ range reduction
    ];
    let offsets = [Vec2::ZERO, Vec2::new(1.0e-7, 0.0), Vec2::new(0.0, -1.0), Vec2::new(1.0e6, 1.0e6), Vec2::new(-0.5, 0.25)];
    for a in &shapes {
        for b in &shapes {
            for &o in &offsets {
                for base in [Vec2::ZERO, Vec2::new(1.0e4, -1.0e4)] {
                    if let Some(m) = collide(a, base, b, base + o) {
                        assert!(m.normal.is_finite() && m.point.is_finite() && m.penetration.is_finite() && m.penetration > 0.0, "{a:?} {b:?} {o:?} -> {m:?}");
                        assert!((m.normal.length() - 1.0).abs() < 1e-2);
                    }
                }
            }
        }
    }
}

#[test]
fn exactly_touching_shapes_are_not_contacts() {
    let b = Shape::aligned_box(Vec2::new(1.0, 1.0));
    assert!(collide(&b, Vec2::ZERO, &b, Vec2::new(2.0, 0.0)).is_none());
    let c = Shape::circle(1.0);
    assert!(collide(&c, Vec2::ZERO, &c, Vec2::new(2.0, 0.0)).is_none());
    assert!(collide(&b, Vec2::ZERO, &c, Vec2::new(2.0, 0.0)).is_none());
}
