use physics2d::{scene, World, WorldConfig};
fn main() {
    let mut w = World::new(WorldConfig { collect_timings: false, ..WorldConfig::default() });
    scene::circle_pile(&mut w, 1000, 1234);
    for _ in 0..1200 { w.step(1.0 / 60.0); }
    println!("{:28} state hash {:016x}", physics2d::simd::BACKEND, w.state_hash());
}
