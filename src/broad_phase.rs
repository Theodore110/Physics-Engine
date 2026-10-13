//! Broad phase: uniform spatial hash grid, built by counting sort.
//!
//! ## Why a hash grid and not sweep-and-prune (SAP)?
//! | | hash grid | SAP |
//! |---|---|---|
//! | build | **O(n)**, no comparisons | O(n) only if almost sorted (insertion sort), else O(n log n) |
//! | bodies piled on a floor | cell occupancy bounded by density | all bodies share a y-range & spread on x → many false overlaps on the sorted axis |
//! | teleports / explosions | no cost change | temporal coherence lost → full re-sort |
//! | parallelisable | per-body cell insertion, per-bucket pair emission | inherently sequential sweep |
//! | weakness | needs a sensible cell size; huge bodies span many cells | none of those, but degenerates on clustered axes |
//!
//! With ~10³–10⁴ similarly-sized, highly dynamic bodies (the target workload) the
//! grid wins. Its one weakness — a body much larger than a cell (e.g. a 50 m floor)
//! would be inserted into thousands of cells — is handled by the **large-body list**:
//! any body whose AABB spans more than 2×2 cells is *not* inserted in the grid and
//! instead tested against every body with a 4-wide SIMD AABB test (there are only a
//! handful of such bodies: floors/walls).
//!
//! ## Algorithm (all buffers reused ⇒ zero allocation after warm-up)
//! 1. **SIMD** – for all bodies compute AABB `[pos ± half]` and the integer cell
//!    range `floor(aabb · 1/cell)`, 4 bodies per instruction.
//! 2. Classify: span ≤ 2×2 cells ⇒ *small* (grid), else *large*.
//! 3. Count entries (Σ cells covered by each small body, ≤ 4 each) to size the table.
//! 4. Counting sort by `hash(cx,cy) & mask`: histogram → prefix sum → **scatter fat
//!    `Slot`s** (body id, cell, cell-min, AABB, static flag) straight into bucket order.
//!    Because every datum the pair test needs travels *inside* the slot, step 5 walks
//!    contiguous memory: no index → entry → body-array dependent-load chains
//!    (my first version had those and the pair scan was latency-bound, ~21 ns per
//!    comparison; see README "Optimisation log").
//! 5. For each bucket test all slot pairs: same cell? static-static? AABB overlap?
//! 6. **Duplicate suppression without a set**: two bodies sharing several cells would
//!    be found several times. A pair is emitted only in the cell
//!    `(max(cx0_i, cx0_j), max(cy0_i, cy0_j))` — the lowest-left cell of the
//!    intersection of their cell ranges — which exactly one bucket visit satisfies.
//! 7. Large bodies × all bodies, 4 lanes at a time, `movemask` to extract hits.

use crate::body::BodyStore;
use crate::simd::{as4, F32x4, LANES};
use std::time::Instant;

/// Optional sub-phase timings (nanoseconds) of the last `update`, filled when `profile` is set.
#[derive(Clone, Copy, Debug, Default)]
pub struct BroadTimings {
    pub aabb_ns: u64,
    pub build_ns: u64,
    pub pairs_ns: u64,
    pub large_ns: u64,
}

/// A candidate pair, always `a < b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pair {
    pub a: u32,
    pub b: u32,
}

/// One (body × cell) membership, with everything the pair test needs stored inline (40 B).
#[derive(Clone, Copy, Default)]
struct Slot {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    body: u32,
    /// the cell this slot belongs to
    cx: i32,
    cy: i32,
    /// lowest-left cell of the owning body (for duplicate suppression)
    cx0: i32,
    cy0: i32,
    /// 1 if inv_mass == 0 (immovable), else 0
    fixed: u32,
    /// hash bucket this slot was sorted into (lets the pair scan find bucket runs in O(entries))
    bucket: u32,
}

#[derive(Default)]
pub struct BroadPhase {
    inv_cell: f32,
    // AABBs (SoA) — computed with SIMD.
    min_x: Vec<f32>,
    min_y: Vec<f32>,
    max_x: Vec<f32>,
    max_y: Vec<f32>,
    // Cell ranges (inclusive).
    cx0: Vec<i32>,
    cy0: Vec<i32>,
    cx1: Vec<i32>,
    cy1: Vec<i32>,
    slots: Vec<Slot>,
    counts: Vec<u32>,
    large: Vec<u32>,
    /// Output of the last `update`.
    pub pairs: Vec<Pair>,
    /// Record sub-phase timings into `timings` (costs 4 `Instant::now()` per update).
    pub profile: bool,
    pub timings: BroadTimings,
    /// Counting-sort the final pair list by first body id (O(n+pairs)) so the solver's
    /// body accesses become mostly sequential. Deterministic (stable sort).
    pub sort_pairs: bool,
    pair_counts: Vec<u32>,
    pair_tmp: Vec<Pair>,
}

#[inline(always)]
fn hash_cell(cx: i32, cy: i32) -> u32 {
    // Integer-only mixing (Teschner et al. primes + a final xor-shift) ⇒ identical on all platforms.
    let h = (cx as u32).wrapping_mul(0x9E37_79B1) ^ (cy as u32).wrapping_mul(0x85EB_CA6B);
    h ^ (h >> 15)
}

impl BroadPhase {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-size internal buffers (optional; they grow automatically during warm-up).
    pub fn reserve(&mut self, bodies: usize, pairs: usize) {
        self.pairs.reserve(pairs);
        self.pair_tmp.reserve(pairs);
        self.pair_counts.reserve(bodies + 1);
        self.slots.reserve(bodies * 4);
        self.counts.reserve((bodies * 8).next_power_of_two() + 1);
    }

    pub fn aabb_arrays(&self) -> (&[f32], &[f32], &[f32], &[f32]) {
        (&self.min_x, &self.min_y, &self.max_x, &self.max_y)
    }

    /// Rebuild the grid and the candidate pair list for the current positions.
    pub fn update(&mut self, bodies: &BodyStore, cell_size: f32, simd: bool) -> &[Pair] {
        assert!(cell_size > 0.0 && cell_size.is_finite());
        let n = bodies.len();
        self.inv_cell = 1.0 / cell_size;
        for v in [&mut self.min_x, &mut self.min_y, &mut self.max_x, &mut self.max_y] {
            v.resize(n, 0.0);
        }
        for v in [&mut self.cx0, &mut self.cy0, &mut self.cx1, &mut self.cy1] {
            v.resize(n, 0);
        }
        self.pairs.clear();
        self.large.clear();

        let t0 = self.profile.then(Instant::now);
        if simd {
            self.compute_aabbs_simd(bodies);
        } else {
            self.compute_aabbs_scalar(bodies);
        }
        let t1 = self.profile.then(Instant::now);
        self.insert_and_sort(bodies);
        let t2 = self.profile.then(Instant::now);
        self.emit_grid_pairs();
        let t3 = self.profile.then(Instant::now);
        self.emit_large_pairs(bodies, simd);
        if self.sort_pairs {
            self.counting_sort_pairs(n);
        }
        if let (Some(t0), Some(t1), Some(t2), Some(t3)) = (t0, t1, t2, t3) {
            let t4 = Instant::now();
            self.timings = BroadTimings {
                aabb_ns: (t1 - t0).as_nanos() as u64,
                build_ns: (t2 - t1).as_nanos() as u64,
                pairs_ns: (t3 - t2).as_nanos() as u64,
                large_ns: (t4 - t3).as_nanos() as u64,
            };
        }
        &self.pairs
    }

    // ---------------- step 1 ----------------

    #[inline(never)]
    fn compute_aabbs_simd(&mut self, b: &BodyStore) {
        let n = b.len();
        let split = n - n % LANES;
        let inv = F32x4::splat(self.inv_cell);
        for i in (0..split).step_by(LANES) {
            let r = i..i + LANES;
            let (px, py) = (F32x4::load(as4(&b.pos_x[r.clone()])), F32x4::load(as4(&b.pos_y[r.clone()])));
            let (hx, hy) = (F32x4::load(as4(&b.half_x[r.clone()])), F32x4::load(as4(&b.half_y[r.clone()])));
            let (x0, y0, x1, y1) = (px - hx, py - hy, px + hx, py + hy);
            let st = |v: F32x4, dst: &mut [f32]| v.store(<&mut [f32; 4]>::try_from(dst).expect("4"));
            st(x0, &mut self.min_x[r.clone()]);
            st(y0, &mut self.min_y[r.clone()]);
            st(x1, &mut self.max_x[r.clone()]);
            st(y1, &mut self.max_y[r.clone()]);
            // cell = floor(coord · 1/cell)
            self.cx0[r.clone()].copy_from_slice(&(x0 * inv).floor_i32());
            self.cy0[r.clone()].copy_from_slice(&(y0 * inv).floor_i32());
            self.cx1[r.clone()].copy_from_slice(&(x1 * inv).floor_i32());
            self.cy1[r.clone()].copy_from_slice(&(y1 * inv).floor_i32());
        }
        self.aabb_tail(b, split);
    }

    #[inline(never)]
    fn compute_aabbs_scalar(&mut self, b: &BodyStore) {
        self.aabb_tail(b, 0);
    }

    /// Scalar AABB + cell computation for `from..n`; same op order as the SIMD lanes.
    fn aabb_tail(&mut self, b: &BodyStore, from: usize) {
        let inv = self.inv_cell;
        for i in from..b.len() {
            let (x0, y0) = (b.pos_x[i] - b.half_x[i], b.pos_y[i] - b.half_y[i]);
            let (x1, y1) = (b.pos_x[i] + b.half_x[i], b.pos_y[i] + b.half_y[i]);
            self.min_x[i] = x0;
            self.min_y[i] = y0;
            self.max_x[i] = x1;
            self.max_y[i] = y1;
            let f = |v: f32| F32x4::splat(v).floor_i32()[0];
            self.cx0[i] = f(x0 * inv);
            self.cy0[i] = f(y0 * inv);
            self.cx1[i] = f(x1 * inv);
            self.cy1[i] = f(y1 * inv);
        }
    }

    // ---------------- steps 2-4 ----------------

    fn insert_and_sort(&mut self, bodies: &BodyStore) {
        let n = bodies.len();
        // pass 0: classify + count entries (to size the hash table)
        let mut e = 0usize;
        for i in 0..n {
            let (span_x, span_y) = (self.cx1[i] - self.cx0[i], self.cy1[i] - self.cy0[i]);
            if span_x > 1 || span_y > 1 {
                self.large.push(i as u32);
            } else {
                e += ((span_x + 1) * (span_y + 1)) as usize;
            }
        }
        // Table ≈ entries ⇒ load ≤ 1. Collisions are cheap (rejected by the same-cell test) and
        // the pair scan visits slots, not buckets, so a smaller table only saves cache.
        let table = e.next_power_of_two().max(16);
        let mask = (table - 1) as u32;
        self.counts.clear();
        self.counts.resize(table + 1, 0);
        // pass A: histogram
        for i in 0..n {
            if self.cx1[i] - self.cx0[i] > 1 || self.cy1[i] - self.cy0[i] > 1 {
                continue;
            }
            for cy in self.cy0[i]..=self.cy1[i] {
                for cx in self.cx0[i]..=self.cx1[i] {
                    self.counts[(hash_cell(cx, cy) & mask) as usize] += 1;
                }
            }
        }
        // inclusive prefix sum: counts[h] = end of bucket h
        let mut acc = 0u32;
        for c in self.counts[..table].iter_mut() {
            acc += *c;
            *c = acc;
        }
        self.counts[table] = e as u32;
        // pass B: scatter fat slots directly into bucket order (reverse walk ⇒ counts[h] ends as start of h)
        // every slot [0, e) is overwritten below, so only the length needs adjusting (no full re-zero)
        self.slots.resize(e, Slot::default());
        for i in (0..n).rev() {
            if self.cx1[i] - self.cx0[i] > 1 || self.cy1[i] - self.cy0[i] > 1 {
                continue;
            }
            let proto = Slot {
                min_x: self.min_x[i],
                min_y: self.min_y[i],
                max_x: self.max_x[i],
                max_y: self.max_y[i],
                body: i as u32,
                cx: 0,
                cy: 0,
                cx0: self.cx0[i],
                cy0: self.cy0[i],
                fixed: (bodies.inv_mass[i] == 0.0) as u32,
                bucket: 0,
            };
            for cy in (self.cy0[i]..=self.cy1[i]).rev() {
                for cx in (self.cx0[i]..=self.cx1[i]).rev() {
                    let h = (hash_cell(cx, cy) & mask) as usize;
                    self.counts[h] -= 1;
                    self.slots[self.counts[h] as usize] = Slot { cx, cy, bucket: h as u32, ..proto };
                }
            }
        }
    }

    // ---------------- steps 5-6 ----------------

    fn emit_grid_pairs(&mut self) {
        let slots = &self.slots[..];
        let mut start = 0usize;
        while start < slots.len() {
            // run of slots sharing a bucket (they are contiguous after the counting sort)
            let bucket = slots[start].bucket;
            let mut end = start + 1;
            while end < slots.len() && slots[end].bucket == bucket {
                end += 1;
            }
            let run = &slots[start..end];
            for (k, p) in run.iter().enumerate() {
                for q in &run[k + 1..] {
                    if p.cx != q.cx || p.cy != q.cy {
                        continue; // hash collision between different cells
                    }
                    // canonical-cell rule (see module docs, step 6)
                    if p.cx != p.cx0.max(q.cx0) || p.cy != p.cy0.max(q.cy0) {
                        continue;
                    }
                    // two infinite-mass bodies can never produce impulses
                    if p.fixed & q.fixed != 0 {
                        continue;
                    }
                    if p.min_x <= q.max_x && p.max_x >= q.min_x && p.min_y <= q.max_y && p.max_y >= q.min_y {
                        let (a, b) = if p.body < q.body { (p.body, q.body) } else { (q.body, p.body) };
                        self.pairs.push(Pair { a, b });
                    }
                }
            }
            start = end;
        }
    }

    /// Stable counting sort of `pairs` by `a`. Scratch buffers are pooled.
    fn counting_sort_pairs(&mut self, n: usize) {
        self.pair_counts.clear();
        self.pair_counts.resize(n + 1, 0);
        for p in &self.pairs {
            self.pair_counts[p.a as usize + 1] += 1;
        }
        for i in 0..n {
            self.pair_counts[i + 1] += self.pair_counts[i];
        }
        self.pair_tmp.clear();
        self.pair_tmp.resize(self.pairs.len(), Pair { a: 0, b: 0 });
        for p in &self.pairs {
            let c = &mut self.pair_counts[p.a as usize];
            self.pair_tmp[*c as usize] = *p;
            *c += 1;
        }
        std::mem::swap(&mut self.pairs, &mut self.pair_tmp);
    }

    // ---------------- step 7 ----------------

    fn emit_large_pairs(&mut self, bodies: &BodyStore, simd: bool) {
        let n = bodies.len();
        let split = n - n % LANES;
        for li in 0..self.large.len() {
            let l = self.large[li] as usize;
            let (lx0, ly0, lx1, ly1) = (self.min_x[l], self.min_y[l], self.max_x[l], self.max_y[l]);
            let mut j = 0usize;
            if simd {
                let (vlx0, vly0, vlx1, vly1) =
                    (F32x4::splat(lx0), F32x4::splat(ly0), F32x4::splat(lx1), F32x4::splat(ly1));
                while j < split {
                    let r = j..j + LANES;
                    // overlap  <=>  min_j ≤ max_L  &  max_j ≥ min_L  (per axis)
                    let m = F32x4::load(as4(&self.min_x[r.clone()]))
                        .le(vlx1)
                        .and(F32x4::load(as4(&self.max_x[r.clone()])).ge(vlx0))
                        .and(F32x4::load(as4(&self.min_y[r.clone()])).le(vly1))
                        .and(F32x4::load(as4(&self.max_y[r.clone()])).ge(vly0));
                    let mut bits = m.bits();
                    while bits != 0 {
                        let lane = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        Self::push_large(&mut self.pairs, bodies, &self.cx0, &self.cx1, &self.cy0, &self.cy1, l, j + lane);
                    }
                    j += LANES;
                }
            }
            while j < n {
                if lx0 <= self.max_x[j] && lx1 >= self.min_x[j] && ly0 <= self.max_y[j] && ly1 >= self.min_y[j] {
                    Self::push_large(&mut self.pairs, bodies, &self.cx0, &self.cx1, &self.cy0, &self.cy1, l, j);
                }
                j += 1;
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn push_large(pairs: &mut Vec<Pair>, bodies: &BodyStore, cx0: &[i32], cx1: &[i32], cy0: &[i32], cy1: &[i32], l: usize, j: usize) {
        if j == l || (bodies.inv_mass[l] == 0.0 && bodies.inv_mass[j] == 0.0) {
            return;
        }
        // large×large: only the lower-id large body emits, so each pair appears once
        let j_large = cx1[j] - cx0[j] > 1 || cy1[j] - cy0[j] > 1;
        if j_large && j < l {
            return;
        }
        let (a, b) = if l < j { (l, j) } else { (j, l) };
        pairs.push(Pair { a: a as u32, b: b as u32 });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use crate::shape::Shape;
    use crate::util::Rng;
    use crate::vec2::Vec2;
    use std::collections::BTreeSet;

    fn brute_force(b: &BodyStore) -> BTreeSet<(u32, u32)> {
        let mut out = BTreeSet::new();
        for i in 0..b.len() {
            for j in i + 1..b.len() {
                if b.inv_mass[i] == 0.0 && b.inv_mass[j] == 0.0 {
                    continue;
                }
                let (ai, aj) = (b.shapes[i].aabb(b.get(crate::BodyId(i as u32)).position), b.shapes[j].aabb(b.get(crate::BodyId(j as u32)).position));
                if ai.overlaps(&aj) {
                    out.insert((i as u32, j as u32));
                }
            }
        }
        out
    }

    #[test]
    fn matches_brute_force_exactly_with_no_duplicates() {
        for (seed, n, cell, simd) in [(1u32, 300usize, 1.0f32, true), (2, 301, 0.5, true), (3, 303, 2.0, false), (4, 150, 0.3, true)] {
            let mut rng = Rng::new(seed);
            let mut s = BodyStore::new();
            // some huge static boxes (large list) + mixed small bodies, incl. negative coords
            s.add(RigidBody::fixed(Shape::aligned_box(Vec2::new(40.0, 0.5)), Vec2::new(0.0, -12.0)));
            s.add(RigidBody::fixed(Shape::oriented_box(Vec2::new(30.0, 1.0), 0.6), Vec2::new(5.0, 3.0)));
            for _ in 0..n {
                let p = Vec2::new(rng.range(-20.0, 20.0), rng.range(-12.0, 12.0));
                let shape = if rng.next_f32() < 0.5 {
                    Shape::circle(rng.range(0.1, 0.6))
                } else {
                    Shape::oriented_box(Vec2::new(rng.range(0.1, 0.7), rng.range(0.1, 0.7)), rng.range(0.0, 3.0))
                };
                s.add(RigidBody::dynamic(shape, 1.0, p));
            }
            let mut bp = BroadPhase::new();
            let got: Vec<_> = bp.update(&s, cell, simd).to_vec();
            let set: BTreeSet<_> = got.iter().map(|p| (p.a, p.b)).collect();
            assert_eq!(set.len(), got.len(), "duplicate pairs emitted (seed {seed})");
            assert_eq!(set, brute_force(&s), "pair set differs (seed {seed})");
        }
    }

    #[test]
    fn simd_and_scalar_agree_exactly() {
        let mut rng = Rng::new(11);
        let mut s = BodyStore::new();
        for _ in 0..257 {
            s.add(RigidBody::dynamic(Shape::circle(rng.range(0.1, 0.4)), 1.0, Vec2::new(rng.range(-9.0, 9.0), rng.range(-9.0, 9.0))));
        }
        let (mut a, mut b) = (BroadPhase::new(), BroadPhase::new());
        let pa = a.update(&s, 0.8, true).to_vec();
        let pb = b.update(&s, 0.8, false).to_vec();
        assert_eq!(pa, pb);
        assert_eq!(a.cx0, b.cx0);
        assert_eq!(a.min_x.iter().map(|f| f.to_bits()).collect::<Vec<_>>(), b.min_x.iter().map(|f| f.to_bits()).collect::<Vec<_>>());
    }
}
