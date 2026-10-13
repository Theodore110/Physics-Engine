# physics2d — a from-scratch, deterministic 2D rigid-body engine (Rust, zero dependencies)

```
cargo test  --release                      # 36 tests: unit, fuzz, simulation (also with --features portable_simd)
cargo run   --release --bin demo           # ASCII demo
cargo run   --release --bin bench          # full benchmark (~4 min); output of the run quoted below: BENCH_RESULTS.md
sh lang_bench/run.sh                       # the cross-language kernel benchmark behind the language choice
```

Everything — vector math, SIMD layer, broad phase, SAT, impulse solver — is hand-written. `[dependencies]` is empty.
Toolchain used: rustc/cargo 1.75, Ubuntu 24.04, 1 shared vCPU (Xeon @ 2.1 GHz). **No desktop CPU was measured.**

## 1. Language decision: Rust

Kernel under test: the contact-solver hot loop (gather/scatter on bodies, sqrt, clamped sequential impulses), 40,000 bodies,
60 steps, one shared C timing harness, bit-identical checksum in all variants (`lang_bench/`).

| variant | best of 7 |
|---|---:|
| Rust, safe indexing | 335 ms |
| Zig 0.16 ReleaseFast | 373 ms |
| Rust, `get_unchecked` | 383 ms |
| C++ Clang 18 | 420 ms |
| C++ GCC 13 | 483 ms |

The VM is noisy (10–25 % round-to-round spread), so the honest reading is: **performance does not discriminate** between
these; safe Rust paid no visible bounds-check penalty in this kernel. A later, longer interleaved run was done but its
figures are not reproduced here.

| criterion | Rust | C++ | Zig |
|---|---|---|---|
| speed ceiling | same | same | same |
| float determinism | no FMA contraction / fast-math by default | GCC fuses by default → needs `-ffp-contract=off` | none unless `@mulAdd` |
| SIMD | stable `core::arch` (`unsafe`); `std::simd` nightly | intrinsics / xsimd | `@Vector` — best ergonomics |
| memory safety | safe by default | RAII, UB possible | manual |
| test / fuzz / bench tooling | built in | external | built in |
| maturity | stable | stable | pre-1.0 (0.16 API churn) |

**Pick: Rust** — safe indexing in an index-heavy solver, determinism by default, built-in tooling. C++ or Zig are defensible.

## 2. Architecture

```
World::step(dt)
 1. integrate_velocities  v += (a_user + g·[inv_mass>0])·dt, then damping      [SIMD, SoA]
 2. broad_phase.update    AABBs → hash grid (counting sort, fat slots) → pairs   [SIMD AABB]
 3. narrow_phase::detect  pairs → circle/circle, circle/OBB, OBB SAT → Contact{n, depth, point, a, b}
 4. solver.prepare        e = min(eA,eB); restitution target; warm start from last step
    solver sweeps ×N      sequential impulses, clamp on the ACCUMULATED impulse
 5. integrate_positions   x += v·dt                                             [SIMD, SoA]
 6. solver.solve_position Baumgarte shift (β, slop, cap)
```

Modules (dependency order): `simd, vec2, util, shape, body, broad_phase, narrow_phase, solver, world`, plus `scene`
(builders + ASCII renderer) and the `demo` / `bench` binaries. Math derivations are comments at the point of use
(impulse formula and Baumgarte in `solver.rs`, SAT axes in `narrow_phase.rs`).

Key design decisions
* **f32**, `#[repr(C)] Vec2`; true component SoA (`pos_x`, `pos_y`, …) so one SSE load = 4 bodies.
* **Static = `inv_mass == 0`**; the solver never reads `is_static`. Gravity uses a lane *mask*, not a branch.
* **Boxes have a fixed orientation** (rotation given at construction; bodies don't rotate yet) ⇒ AABB half-extents are exact and constant.
* **Broad phase: hash grid, not SAP** — O(n) build, no clustering pathologies on a floor pile, parallelisable. Bodies spanning >2×2 cells go to a "large body" list tested with a 4-wide SIMD AABB scan.
* **Determinism**: only `+ − × / sqrt` in the loop, no FMA, no libm (own polynomial `det_sin_cos`), no `HashMap`, integer hashing, single thread.
* **Zero allocation**: every buffer is cleared, never freed; verified by a counting global allocator in the bench.
* **Only `unsafe`**: the SSE2 wrapper in `simd.rs` (sound: SSE2 is baseline on x86-64; references guarantee valid pointers). `--features portable_simd` swaps in a safe scalar backend and the library is then compiled with `#![forbid(unsafe_code)]`.

## 3. Results (final code, `BENCH_RESULTS.md`)

Method: release, fat LTO, 1 thread, dt=1/60, 8 velocity + 3 position iterations; N circles r∈[0.18,0.28] m poured into a
container. "Impact" = steps 20–320, "settled" = steps 900–1200 (warm-up before each window is untimed); 7 repeats on freshly
built worlds, **median** reported. FPS = 1 / mean physics-step time (no rendering). Allocations counted by a wrapping global allocator.

| bodies | impact step | impact FPS | settled step | settled FPS | ns/body/step (settled) | allocs in window |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 106 µs | 9,427 | 95 µs | 10,484 | 382 | 0 |
| 500 | 223 µs | 4,488 | 230 µs | 4,358 | 459 | 0 |
| **1,000** | **460 µs** | **2,172** | **531 µs** | **1,882** | 531 | **0** |
| 2,000 | 980 µs | 1,020 | 1,082 µs | 924 | 541 | 0 |
| 4,000 | 2,164 µs | 462 | 2,839 µs | 352 | 710 | 0 |

**Target met**: 1,000 circles ≥ 1,000 FPS single-threaded (1.9–2.2× margin on this VM). 2,000 is borderline; 4,000 is below.
Noise warning: the 4,000-body settled row has median 2.84 ms vs best 2.30 ms, and the single-run per-phase tables are noisier than the medians.

Per-phase, 1,000 bodies, µs/step (settled / impact): integrate 1.2 / 1.2 · broad 161 / 156 · narrow 28 / 26 · solve 302 / 264.
**The solver is ~55–60 % of a step, the broad phase ~30 %.**

Explicit SIMD vs the scalar reference (best of 7): integrate 0.98× (N=1000), 1.03× (N=4000); broad phase 1.02×, 1.05×.
**No meaningful speed-up**: LLVM already auto-vectorises the SoA loops and integration is <0.3 % of a step. The SIMD paths are kept
because the spec requires them; they are bit-identical to the scalar paths. The SoA layout, not the intrinsics, delivers the performance.

Determinism (N=1000, 1200 steps): two SIMD runs, a scalar run, and a `--features portable_simd` build all give state hash
`10439adf8664a907`. (Two backends on one x86-64 machine — not yet a cross-OS/CPU/ARM proof.)

## 4. Quality evidence (all pass on both backends)

* 25 unit tests: Vec2, SIMD-vs-scalar bit equality, sin/cos accuracy, broad phase == brute force with no duplicates, closed-form impulse, restitution, momentum, Baumgarte → slop.
* Fuzz (400,000 random shape pairs; ~258 k hits / 142 k misses): finite outputs, unit normals, pen>0, symmetry, **MTV really separates**, independent point-in-both-shapes oracle for completeness, soundness of the contact point, degenerate inputs.
* Simulation: 3-box tower at 60 Hz (jitter 0 m, late max speed ~3e-8 m/s, sag 16 mm ≈ 3 contacts at the 1 cm slop); 8-box tower stable; 600-circle pile fully at rest after 15 s; bounce peak 1.816 m vs e²h = 1.800 m; momentum drift 2.4e-6 over 1,543 contacts; fixed-timestep accumulator; bit-exact determinism.

## 5. Optimisation / debugging log (what actually happened)

| finding | action | measured effect |
|---|---|---|
| Fuzz: circle fully inside another gave a contact point outside the smaller one | midpoint of the overlap *interval*; circle-in-box uses the circle centre | fuzz passes |
| 600-circle pile never settled (median 0.55 m/s residual, 8 iters, cold start) | added warm starting keyed by body pair (open-addressing table, deterministic) | residual 0.55 → 0.002 m/s |
| First warm-start made it worse (3–7 m/s) | closing speeds were read after earlier contacts were warm-started; split into two passes | fixed |
| Pair scan ~21 ns/comparison: dependent loads index→entry→body arrays | counting sort scatters fat 40-byte slots; scan walks bucket runs | broad phase ~221 → ~161 µs |
| Default cell size 1.0 too large | `World::fit_cell_size()` = 1.05 × largest dynamic AABB extent | step 543 → 495 µs (−9 %) |
| Counting-sort pairs by body id for solver locality | kept as opt-in `sort_pairs` | ±2 % — **no gain**, default off |
| Explicit SIMD | kept (spec) | ~1.0× — **no gain** (see §3) |
| Remaining pair-scan cost | hypothesis: branch misprediction (data-dependent tests). **Not verified with perf counters.** | — |

## 6. Limitations and roadmap

* **Not implemented**: body rotation / angular dynamics, friction (piles spread sideways), joints, sleeping. `sleep_threshold` only feeds the `resting_bodies()` statistic.
* One contact point per pair (centroid of the overlap polygon for boxes); fine without rotation, insufficient once torque exists → needs a 2-point manifold with feature ids.
* Contact matching for warm start is by body pair — will need feature ids once manifolds have 2 points.
* Positional correction is Baumgarte-style (shifts positions); split impulses would be the next step for stiffer stacks.
* SSE2 only (4 lanes). AVX2 (8 lanes, runtime-detected) and NEON are the obvious next kernels — but see §3: measure before adding more `unsafe`.
* Solver is the dominant cost: next candidates are contact ordering / graph colouring, and SIMD across independent contacts.
* Multithreading would slot into: broad-phase slot insertion (per-body), narrow phase (per-pair, read-only), solver (islands or graph-coloured batches). It must keep a deterministic order to preserve bit-exactness.
* f32 precision degrades far from the origin (~7.6 µm at 100 m); origin re-basing or f64 beyond ~10 km.
* Cross-platform bit-exactness is verified only between two backends on one machine.
* GPU offload: intentionally out of scope (CPU-only).
