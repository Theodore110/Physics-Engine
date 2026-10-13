//! Benchmark harness (std-only; `criterion` would be an external dependency).
//!
//! Methodology
//! * `cargo run --release --bin bench`  (opt-level 3, fat LTO, 1 codegen unit)
//! * Single thread. Fixed dt = 1/60 s, 8 velocity iterations, 3 position iterations.
//! * Scene: N circles (r ∈ [0.18,0.28] m) poured into a container (see `scene::circle_pile`).
//! * Two measurement windows per N:
//!     - "impact"  : steps  20..320   (bodies falling and slamming into the floor/each other)
//!     - "settled" : steps 900..1200  (a resting pile: the maximum persistent contact load)
//!   Everything before a window is warm-up and is not timed.
//! * Each window is repeated R times on a freshly built world; we report the **median**
//!   of the per-step mean (and the best repeat). FPS = 1 / mean step time (physics only,
//!   no rendering).
//! * Wall time is measured with `collect_timings = false` (no per-phase probes in the loop);
//!   the per-phase breakdown comes from one extra run with probes on.
//! * Heap allocations are counted by a wrapping global allocator (instrumentation only —
//!   the `unsafe` below lives in this binary, not in the engine).

use physics2d::{scene, World, WorldConfig};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        System.alloc_zeroed(l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        System.realloc(p, l, n)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}
#[global_allocator]
static A: Counting = Counting;

fn build(n: usize, cfg: WorldConfig) -> World {
    let mut w = World::new(cfg);
    scene::circle_pile(&mut w, n, 1234);
    w
}

struct Window {
    mean_ns: f64,
    allocs: usize,
    avg_pairs: f64,
    avg_contacts: f64,
}

/// Run `warm` untimed steps, then time `steps` steps.
fn run_window(n: usize, cfg: WorldConfig, warm: usize, steps: usize) -> Window {
    let mut w = build(n, cfg);
    for _ in 0..warm {
        w.step(cfg.fixed_dt);
    }
    let (mut pairs, mut contacts) = (0usize, 0usize);
    let a0 = ALLOCS.load(Relaxed);
    let t0 = Instant::now();
    for _ in 0..steps {
        w.step(cfg.fixed_dt);
        // reading the two counters is two loads; negligible next to a ~ms step
        pairs += w.last_pair_count();
        contacts += w.contacts().len();
    }
    let dt = t0.elapsed();
    let allocs = ALLOCS.load(Relaxed) - a0;
    black_box(w.state_hash());
    Window {
        mean_ns: dt.as_nanos() as f64 / steps as f64,
        allocs,
        avg_pairs: pairs as f64 / steps as f64,
        avg_contacts: contacts as f64 / steps as f64,
    }
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Mean per-phase times (ns/step) over a window, probes on.
fn phase_breakdown(n: usize, cfg: WorldConfig, warm: usize, steps: usize) -> [f64; 4] {
    let cfg = WorldConfig { collect_timings: true, ..cfg };
    let mut w = build(n, cfg);
    for _ in 0..warm {
        w.step(cfg.fixed_dt);
    }
    let mut acc = physics2d::StepTimings::default();
    for _ in 0..steps {
        w.step(cfg.fixed_dt);
        acc.add(&w.last_timings());
    }
    let s = steps as f64;
    [acc.integrate_ns as f64 / s, acc.broad_ns as f64 / s, acc.narrow_ns as f64 / s, acc.solve_ns as f64 / s]
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let reps = if quick { 3 } else { 7 };
    let base = WorldConfig { collect_timings: false, ..WorldConfig::default() };
    let sizes = [250usize, 500, 1000, 2000, 4000];

    println!("# physics2d benchmark");
    println!("SIMD backend : {}", physics2d::simd::BACKEND);
    println!("profile      : release (opt 3, fat LTO), single thread, dt = 1/60, {} vel iters, {} pos iters, {} repeats/window\n",
        base.solver_iterations, base.position_iterations, reps);

    // ---- main table ----------------------------------------------------------------
    let windows = [("impact (steps 20-320)", 20usize, 300usize), ("settled (steps 900-1200)", 900, 300)];
    for (label, warm, steps) in windows {
        println!("## {label}");
        println!("| bodies | pairs/step | contacts/step | median step | best step | FPS (median) | ns/body/step | allocs in window |");
        println!("|---:|---:|---:|---:|---:|---:|---:|---:|");
        for &n in &sizes {
            let mut means = Vec::new();
            let (mut allocs, mut pairs, mut contacts) = (0, 0.0, 0.0);
            for _ in 0..reps {
                let r = run_window(n, base, warm, steps);
                means.push(r.mean_ns);
                allocs += r.allocs;
                pairs = r.avg_pairs;
                contacts = r.avg_contacts;
            }
            let best = means.iter().cloned().fold(f64::INFINITY, f64::min);
            let med = median(&mut means);
            println!(
                "| {n} | {pairs:.0} | {contacts:.0} | {:.1} µs | {:.1} µs | {:.0} | {:.0} | {allocs} |",
                med / 1e3,
                best / 1e3,
                1e9 / med,
                med / n as f64
            );
        }
        println!();
    }

    // ---- per-phase breakdown ----------------------------------------------------------
    println!("## per-phase breakdown, settled window, µs/step (probes on)");
    println!("| bodies | integrate | broad | narrow | solve (vel+pos) | total |");
    println!("|---:|---:|---:|---:|---:|---:|");
    for &n in &sizes {
        let p = phase_breakdown(n, base, 900, 300);
        println!("| {n} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |", p[0] / 1e3, p[1] / 1e3, p[2] / 1e3, p[3] / 1e3, p.iter().sum::<f64>() / 1e3);
    }
    println!();
    println!("## per-phase breakdown, impact window, µs/step (probes on)");
    println!("| bodies | integrate | broad | narrow | solve (vel+pos) | total |");
    println!("|---:|---:|---:|---:|---:|---:|");
    for &n in &sizes {
        let p = phase_breakdown(n, base, 20, 300);
        println!("| {n} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |", p[0] / 1e3, p[1] / 1e3, p[2] / 1e3, p[3] / 1e3, p.iter().sum::<f64>() / 1e3);
    }
    println!();

    // ---- SIMD vs scalar ---------------------------------------------------------------
    println!("## explicit SIMD vs scalar reference (N = 1000 and 4000, settled window, µs/step, best of {reps})");
    println!("| bodies | kernel | scalar | SIMD | speed-up |");
    println!("|---:|---|---:|---:|---:|");
    for n in [1000usize, 4000] {
        let mut best_s = [f64::INFINITY; 2];
        let mut best_v = [f64::INFINITY; 2];
        for _ in 0..reps {
            let s = phase_breakdown(n, WorldConfig { use_simd: false, ..base }, 900, 100);
            let v = phase_breakdown(n, WorldConfig { use_simd: true, ..base }, 900, 100);
            best_s[0] = best_s[0].min(s[0]);
            best_s[1] = best_s[1].min(s[1]);
            best_v[0] = best_v[0].min(v[0]);
            best_v[1] = best_v[1].min(v[1]);
        }
        for (k, name) in ["integrate (vel + pos)", "broad phase (AABB+grid+pairs)"].iter().enumerate() {
            println!("| {n} | {name} | {:.2} | {:.2} | {:.2}x |", best_s[k] / 1e3, best_v[k] / 1e3, best_s[k] / best_v[k]);
        }
    }
    println!();

    // ---- determinism --------------------------------------------------------------------
    println!("## determinism (N = 1000, 1200 steps)");
    let hash = |cfg: WorldConfig| {
        let mut w = build(1000, cfg);
        for _ in 0..1200 {
            w.step(cfg.fixed_dt);
        }
        w.state_hash()
    };
    let (h1, h2) = (hash(base), hash(base));
    let hs = hash(WorldConfig { use_simd: false, ..base });
    println!("run 1 (SIMD)   : {h1:016x}");
    println!("run 2 (SIMD)   : {h2:016x}   identical to run 1: {}", h1 == h2);
    println!("run 3 (scalar) : {hs:016x}   identical to SIMD  : {}", h1 == hs);
    println!("(compare this hash across builds/platforms, e.g. `--features portable_simd`)");
}
