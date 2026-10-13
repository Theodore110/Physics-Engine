//! End-to-end simulation tests: stability, conservation, determinism, no-NaN.

use physics2d::body::RigidBody;
use physics2d::util::Rng;
use physics2d::{scene, Shape, Vec2, World, WorldConfig};

#[test]
fn three_box_tower_is_stable_at_60hz() {
    let mut w = World::new(WorldConfig { solver_iterations: 10, ..WorldConfig::default() });
    let ids = scene::box_tower(&mut w, 3);
    let steps_total = 60 * 12;
    let mut top_y = Vec::new();
    let mut max_speed_late = 0f32;
    let mut max_x_drift = 0f32;
    for s in 0..steps_total {
        w.step(1.0 / 60.0);
        for &id in &ids {
            let p = w.bodies.position(id);
            assert!(p.is_finite() && w.bodies.velocity(id).is_finite(), "NaN at step {s}");
            max_x_drift = max_x_drift.max(p.x.abs());
        }
        if s >= 60 * 6 {
            // the last 6 seconds: look for jitter
            top_y.push(w.bodies.position(ids[2]).y);
            for &id in &ids {
                max_speed_late = max_speed_late.max(w.bodies.velocity(id).length());
            }
        }
    }
    let (lo, hi) = top_y.iter().fold((f32::MAX, f32::MIN), |(l, h), &y| (l.min(y), h.max(y)));
    let top = *top_y.last().unwrap();
    eprintln!("tower: top y = {top:.4} (ideal 2.5), jitter range {:.2e} m, max |v| late {:.2e} m/s, x-drift {:.1e}", hi - lo, max_speed_late, max_x_drift);
    assert!(max_x_drift < 1e-4, "tower drifted sideways");
    assert!(top > 2.5 - 0.06 && top <= 2.5 + 1e-3, "tower sagged/exploded: top y = {top}");
    assert!(hi - lo < 2.0e-3, "top box jitters by {} m", hi - lo);
    assert!(max_speed_late < 0.05, "tower still moving at {max_speed_late} m/s");
    // box order preserved (nothing tunnelled through anything)
    let ys: Vec<f32> = ids.iter().map(|&i| w.bodies.position(i).y).collect();
    assert!(ys[0] < ys[1] && ys[1] < ys[2]);
}

#[test]
fn tall_tower_of_eight_survives() {
    // Not required by the spec, but a useful stress: without warm-starting this will sag
    // more than the 3-stack, yet must not explode, tip (no rotation) or invert.
    let mut w = World::new(WorldConfig { solver_iterations: 20, position_iterations: 4, ..WorldConfig::default() });
    let ids = scene::box_tower(&mut w, 8);
    for _ in 0..60 * 10 {
        w.step(1.0 / 60.0);
    }
    let ys: Vec<f32> = ids.iter().map(|&i| w.bodies.position(i).y).collect();
    eprintln!("8-tower ys: {:?}", ys.iter().map(|y| (y * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    for k in 1..8 {
        assert!(ys[k] > ys[k - 1] + 0.8, "boxes interpenetrated at {k}: {ys:?}");
    }
    assert!(ys[7] > 7.5 - 0.5 && ys[7] < 7.5 + 0.01, "{ys:?}");
}

#[test]
fn circle_pile_comes_to_rest_without_nans_or_escapes() {
    let mut w = World::new(WorldConfig::default());
    let width = scene::circle_pile(&mut w, 600, 99);
    for _ in 0..60 * 15 {
        w.step(1.0 / 60.0);
    }
    let b = &w.bodies;
    for i in 3..b.len() {
        assert!(b.pos_x[i].is_finite() && b.pos_y[i].is_finite() && b.vel_x[i].is_finite() && b.vel_y[i].is_finite());
        assert!(b.pos_x[i].abs() < width * 0.5 + 0.1, "body {i} escaped through a wall: x={}", b.pos_x[i]);
        assert!(b.pos_y[i] > -0.1, "body {i} fell through the floor: y={}", b.pos_y[i]);
    }
    let dyn_count = b.len() - 3;
    let resting = w.resting_bodies();
    eprintln!("pile: {resting}/{dyn_count} below sleep threshold after 15 s");
    assert!(resting as f32 > 0.9 * dyn_count as f32, "pile is not settling: {resting}/{dyn_count}");
}

#[test]
fn bounce_height_follows_restitution() {
    // Drop from h with restitution e: rebound height ≈ e²·h.
    let e = 0.6f32;
    let mut w = World::new(WorldConfig::default());
    w.add_body(RigidBody::fixed(Shape::aligned_box(Vec2::new(10.0, 0.5)), Vec2::new(0.0, -0.5)).with_restitution(e));
    let ball = w.add_body(RigidBody::dynamic(Shape::circle(0.25), 1.0, Vec2::new(0.0, 5.25)).with_restitution(e));
    let mut peak_after_bounce = 0f32;
    let mut bounced = false;
    for _ in 0..60 * 4 {
        w.step(1.0 / 60.0);
        let (p, v) = (w.bodies.position(ball), w.bodies.velocity(ball));
        if v.y > 0.0 {
            bounced = true;
        }
        if bounced {
            peak_after_bounce = peak_after_bounce.max(p.y - 0.25);
        }
        if bounced && v.y < 0.0 {
            break;
        }
    }
    let expected = e * e * 5.0;
    eprintln!("bounce: peak {peak_after_bounce:.3} m, e²h = {expected:.3} m");
    assert!((peak_after_bounce - expected).abs() < 0.15, "peak {peak_after_bounce} vs {expected}");
}

#[test]
fn momentum_conserved_in_free_collisions() {
    // No gravity, no walls: total momentum must be conserved by the impulse solver (position
    // correction moves bodies but never changes velocities).
    let cfg = WorldConfig { gravity: Vec2::ZERO, restitution_threshold: 0.0, ..WorldConfig::default() };
    let mut w = World::new(cfg);
    let mut rng = Rng::new(3);
    for _ in 0..200 {
        let r = rng.range(0.2, 0.4);
        let p = Vec2::new(rng.range(-6.0, 6.0), rng.range(-6.0, 6.0));
        let v = Vec2::new(rng.range(-3.0, 3.0), rng.range(-3.0, 3.0));
        w.add_body(RigidBody::dynamic(Shape::circle(r), 1.0, p).with_velocity(v).with_restitution(0.9));
    }
    let momentum = |w: &World| {
        let mut m = Vec2::ZERO;
        for i in 0..w.bodies.len() {
            m += Vec2::new(w.bodies.vel_x[i], w.bodies.vel_y[i]) * w.bodies.mass[i];
        }
        m
    };
    let m0 = momentum(&w);
    let mut contacts_seen = 0;
    for _ in 0..300 {
        w.step(1.0 / 60.0);
        contacts_seen += w.contacts().len();
    }
    let m1 = momentum(&w);
    eprintln!("momentum drift {:.3e} over {contacts_seen} contacts", (m1 - m0).length());
    assert!(contacts_seen > 100, "scene produced too few collisions");
    assert!((m1 - m0).length() < 1e-2 * (1.0 + m0.length()), "{m0:?} -> {m1:?}");
}

#[test]
fn bit_exact_determinism_across_runs_and_simd_paths() {
    let run = |simd: bool| {
        let mut w = World::new(WorldConfig { use_simd: simd, collect_timings: false, ..WorldConfig::default() });
        scene::circle_pile(&mut w, 400, 5);
        // add boxes too so SAT/clipping is part of the hash
        for i in 0..20 {
            w.add_body(RigidBody::dynamic(Shape::oriented_box(Vec2::new(0.5, 0.3), i as f32 * 0.37), 2.0, Vec2::new(-3.0 + i as f32 * 0.3, 14.0 + i as f32)));
        }
        for _ in 0..600 {
            w.step(1.0 / 60.0);
        }
        w.state_hash()
    };
    let (a, b, scalar) = (run(true), run(true), run(false));
    eprintln!("hash {a:016x}");
    assert_eq!(a, b, "two identical runs diverged");
    assert_eq!(a, scalar, "SIMD and scalar paths diverged");
}

#[test]
fn fixed_timestep_accumulator() {
    let mut w = World::new(WorldConfig::default());
    let ball = w.add_body(RigidBody::dynamic(Shape::circle(0.5), 1.0, Vec2::new(0.0, 100.0)));
    // 1 second delivered as irregular frames must produce exactly 60 steps
    let mut steps = 0;
    let mut rng = Rng::new(1);
    let mut left = 1.0f32;
    while left > 0.0 {
        let f = rng.range(0.001, 0.03).min(left);
        steps += w.advance(f);
        left -= f;
    }
    assert!((59..=60).contains(&steps), "steps = {steps}");
    // a stall of 10 s must be clamped (no spiral of death)
    assert_eq!(w.advance(10.0), w.config.max_steps_per_frame);
    assert!(w.bodies.position(ball).y < 100.0);
}

#[test]
fn zero_gravity_free_flight_is_exact() {
    let mut w = World::new(WorldConfig { gravity: Vec2::ZERO, ..WorldConfig::default() });
    let b = w.add_body(RigidBody::dynamic(Shape::circle(0.5), 1.0, Vec2::ZERO).with_velocity(Vec2::new(3.0, -2.0)));
    for _ in 0..60 {
        w.step(1.0 / 60.0);
    }
    let p = w.bodies.position(b);
    assert!((p.x - 3.0).abs() < 1e-3 && (p.y + 2.0).abs() < 1e-3, "{p:?}");
}
