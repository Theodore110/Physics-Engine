//! # physics2d — a from-scratch, deterministic 2D rigid-body engine
//!
//! Module dependency order (each only depends on the ones above it):
//!
//! ```text
//!   simd ─┐
//!   vec2 ─┼─► shape ─► body ─► broad_phase ─► narrow_phase ─► solver ─► world
//!   util ─┘                                                              │
//!                                                              scene ◄───┘ (demo helpers)
//! ```
//!
//! ## Determinism contract
//! Identical inputs give bit-identical outputs because:
//!  * only IEEE-754 `+ - * / sqrt` on `f32` are used in the simulation loop
//!    (all correctly rounded, so identical on every conforming CPU);
//!  * no FMA contraction, no fast-math, no `libm` transcendental in the loop
//!    (box orientation uses our own polynomial [`util::det_sin_cos`]);
//!  * no `HashMap` (randomised seeds) — the broad-phase uses integer hashing
//!    and counting sort, so iteration order is a pure function of the input;
//!  * single-threaded; no data-dependent scheduling.

// With `--features portable_simd` the whole library is compiled with `unsafe` forbidden;
// otherwise the only `unsafe` is the SSE2 wrapper in `simd.rs` (see its module docs).
#![cfg_attr(feature = "portable_simd", forbid(unsafe_code))]

pub mod simd;
pub mod vec2;
pub mod util;
pub mod shape;
pub mod body;
pub mod broad_phase;
pub mod narrow_phase;
pub mod solver;
pub mod world;
pub mod scene;

pub use body::{BodyId, BodyStore, RigidBody};
pub use shape::{Aabb, Shape};
pub use vec2::Vec2;
pub use world::{StepTimings, World, WorldConfig};
