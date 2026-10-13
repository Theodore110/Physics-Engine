use physics2d::{scene, World, WorldConfig};
fn run(label: &str, cfg: WorldConfig, fit: bool) {
    let mut best = f64::MAX; let mut parts = (0.0, 0.0, 0.0, 0.0);
    for _ in 0..7 {
        let mut w = World::new(cfg); scene::circle_pile(&mut w, 1000, 1234); if fit { w.fit_cell_size(); }
        for _ in 0..900 { w.step(1.0 / 60.0); }
        let (mut i, mut b, mut n, mut s) = (0u64, 0u64, 0u64, 0u64); let steps = 300u64;
        for _ in 0..steps { w.step(1.0 / 60.0); let t = w.last_timings(); i += t.integrate_ns; b += t.broad_ns; n += t.narrow_ns; s += t.solve_ns; }
        let tot = (i + b + n + s) as f64 / steps as f64 / 1e3;
        if tot < best { best = tot; parts = (i as f64 / steps as f64 / 1e3, b as f64 / steps as f64 / 1e3, n as f64 / steps as f64 / 1e3, s as f64 / steps as f64 / 1e3); }
    }
    println!("{label:38} total {best:6.1} µs  (integrate {:.1}, broad {:.1}, narrow {:.1}, solve {:.1})", parts.0, parts.1, parts.2, parts.3);
}
fn main() {
    let base = WorldConfig::default();
    run("cell 1.0, unsorted pairs", WorldConfig { cell_size: 1.0, sort_pairs: false, ..base }, false);
    run("cell 1.0, sorted pairs", WorldConfig { cell_size: 1.0, sort_pairs: true, ..base }, false);
    run("auto-fit cell, unsorted pairs", WorldConfig { sort_pairs: false, ..base }, true);
    run("auto-fit cell, sorted pairs", WorldConfig { sort_pairs: true, ..base }, true);
}
