//! ASCII demo: circles and boxes poured into a container, rendered at a few instants.
//! Run:  cargo run --release --bin demo

use physics2d::body::RigidBody;
use physics2d::scene;
use physics2d::util::Rng;
use physics2d::{Shape, Vec2, World, WorldConfig};

fn main() {
    let mut world = World::new(WorldConfig::default());
    let w = scene::circle_pile(&mut world, 160, 2024);

    // a few heavy boxes dropped on top
    let mut rng = Rng::new(7);
    for i in 0..6 {
        let x = rng.range(-w * 0.3, w * 0.3);
        let shape = Shape::oriented_box(Vec2::new(0.6, 0.4), rng.range(0.0, 3.0));
        world.add_body(RigidBody::dynamic(shape, 3.0, Vec2::new(x, 9.0 + i as f32 * 1.4)).with_restitution(0.1));
    }
    world.fit_cell_size(); // boxes are larger than the circles
    println!("physics2d demo — {} bodies, SIMD backend: {}\n", world.bodies.len(), physics2d::simd::BACKEND);

    let (x0, x1, y0, y1) = (-w * 0.5 - 1.0, w * 0.5 + 1.0, -1.0, 13.0);
    let snapshots = [0u32, 60, 150, 600];
    let mut step = 0u32;
    for &snap in &snapshots {
        while step < snap {
            world.advance(world.config.fixed_dt);
            step += 1;
        }
        let t = world.last_timings();
        println!(
            "t = {:5.2} s   pairs {:4}  contacts {:4}  at-rest {:3}/{}   step {:6.1} µs",
            step as f32 / 60.0,
            world.last_pair_count(),
            world.contacts().len(),
            world.resting_bodies(),
            world.bodies.inv_mass.iter().filter(|&&m| m > 0.0).count(),
            t.total_ns() as f64 / 1e3
        );
        println!("{}", scene::ascii(&world, x0, x1, y0, y1, 96, 28));
    }
}
