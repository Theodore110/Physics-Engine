//! Scene builders and an ASCII renderer shared by the demo, benchmark and tests.

use crate::body::{BodyId, RigidBody};
use crate::shape::Shape;
use crate::util::Rng;
use crate::vec2::Vec2;
use crate::world::World;

/// Interior width of a container wide enough for `n` circles on a jittered grid.
pub fn container_width_for(n: usize) -> f32 {
    let cols = ((n as f32).sqrt() * 1.5).ceil().max(6.0);
    cols * SPACING + 0.6
}
const SPACING: f32 = 0.58;

/// Adds floor + two tall walls. Interior is x ∈ [−w/2, w/2], y ≥ 0.
pub fn add_container(world: &mut World, w: f32) {
    let wall_h = 60.0;
    world.add_body(RigidBody::fixed(Shape::aligned_box(Vec2::new(w * 0.5 + 1.5, 0.5)), Vec2::new(0.0, -0.5)));
    world.add_body(RigidBody::fixed(Shape::aligned_box(Vec2::new(0.5, wall_h)), Vec2::new(-w * 0.5 - 0.5, wall_h)));
    world.add_body(RigidBody::fixed(Shape::aligned_box(Vec2::new(0.5, wall_h)), Vec2::new(w * 0.5 + 0.5, wall_h)));
}

/// `n` circles (r ∈ [0.18, 0.28] m) dropped from a jittered grid into a container.
/// Returns the container interior width.
pub fn circle_pile(world: &mut World, n: usize, seed: u32) -> f32 {
    let w = container_width_for(n);
    add_container(world, w);
    let cols = ((w - 0.6) / SPACING).floor() as usize;
    let mut rng = Rng::new(seed);
    world.reserve(n + 3, n * 4, n * 4);
    for i in 0..n {
        let (col, row) = (i % cols, i / cols);
        let x = -w * 0.5 + 0.3 + (col as f32 + 0.5) * SPACING + rng.range(-0.04, 0.04);
        let y = 0.6 + row as f32 * SPACING + rng.range(-0.04, 0.04);
        let r = rng.range(0.18, 0.28);
        world.add_body(RigidBody::dynamic(Shape::circle(r), 1.0, Vec2::new(x, y)).with_restitution(0.3));
    }
    world.fit_cell_size();
    w
}

/// A tower of `n` unit boxes (1×1 m, density 1) resting on a floor.
pub fn box_tower(world: &mut World, n: usize) -> Vec<BodyId> {
    world.add_body(RigidBody::fixed(Shape::aligned_box(Vec2::new(20.0, 0.5)), Vec2::new(0.0, -0.5)));
    (0..n)
        .map(|i| {
            world.add_body(
                RigidBody::dynamic(Shape::aligned_box(Vec2::new(0.5, 0.5)), 1.0, Vec2::new(0.0, 0.5 + i as f32))
                    .with_restitution(0.0),
            )
        })
        .collect()
}

/// Rasterise the world to ASCII (`#` static, `o` circle, `+` box). Centre-sampled.
pub fn ascii(world: &World, x0: f32, x1: f32, y0: f32, y1: f32, cols: usize, rows: usize) -> String {
    let b = &world.bodies;
    let mut out = String::with_capacity((cols + 1) * rows);
    for r in 0..rows {
        let y = y1 - (r as f32 + 0.5) * (y1 - y0) / rows as f32;
        for c in 0..cols {
            let x = x0 + (c as f32 + 0.5) * (x1 - x0) / cols as f32;
            let mut ch = ' ';
            for i in 0..b.len() {
                let d = Vec2::new(x - b.pos_x[i], y - b.pos_y[i]);
                let hit = match b.shapes[i] {
                    Shape::Circle { radius } => d.length_squared() <= radius * radius,
                    Shape::Box { half_extents: h, axis, .. } => {
                        d.dot(axis).abs() <= h.x && d.dot(axis.perpendicular()).abs() <= h.y
                    }
                };
                if hit {
                    ch = if b.inv_mass[i] == 0.0 {
                        '#'
                    } else if matches!(b.shapes[i], Shape::Circle { .. }) {
                        'o'
                    } else {
                        '+'
                    };
                    break;
                }
            }
            out.push(ch);
        }
        out.push('\n');
    }
    out
}
