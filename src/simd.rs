//! Minimal explicit-SIMD layer: 4 x f32 lanes.
//!
//! ## Why a hand-written wrapper instead of `std::simd`?
//! `std::simd` is still nightly-only. `core::arch` is stable but every intrinsic is
//! an `unsafe fn`. This module is the **only** place in the engine that contains
//! `unsafe`; the rest of the crate sees a 100 % safe API.
//!
//! ## Why the `unsafe` is sound
//! * `x86_64` guarantees SSE2 in the baseline ISA, so calling SSE2 intrinsics can
//!   never execute an illegal instruction (the only precondition of these fns).
//! * Loads/stores take `&[f32; 4]` / `&mut [f32; 4]`, so the pointers handed to
//!   `_mm_loadu_ps` / `_mm_storeu_ps` are always valid, in-bounds and (for the
//!   unaligned variants) alignment-free.
//!
//! ## Determinism
//! Only `add, sub, mul, min, max, compare, and, convert` are exposed — no FMA, no
//! reciprocal estimates (`rcpps`/`rsqrtps` differ between Intel and AMD!).
//! The portable fallback implements the *same* semantics (including `min`/`max`
//! NaN behaviour), and `tests` + the benchmark verify bit-equality of both paths.

#[cfg(all(target_arch = "x86_64", not(feature = "portable_simd")))]
mod imp {
    use core::arch::x86_64::*;
    use core::ops::{Add, Mul, Sub};

    /// Four packed `f32`.
    #[derive(Clone, Copy)]
    pub struct F32x4(__m128);
    /// Four packed all-ones / all-zeros lane masks.
    #[derive(Clone, Copy)]
    pub struct Mask4(__m128);

    pub const BACKEND: &str = "x86_64 SSE2 intrinsics";

    impl F32x4 {
        #[inline(always)]
        pub fn splat(v: f32) -> Self {
            // SAFETY: SSE2 is baseline on x86_64.
            unsafe { Self(_mm_set1_ps(v)) }
        }
        #[inline(always)]
        pub fn load(a: &[f32; 4]) -> Self {
            // SAFETY: `a` is a valid reference to 16 readable bytes; loadu has no alignment need.
            unsafe { Self(_mm_loadu_ps(a.as_ptr())) }
        }
        #[inline(always)]
        pub fn store(self, a: &mut [f32; 4]) {
            // SAFETY: `a` is a valid exclusive reference to 16 writable bytes.
            unsafe { _mm_storeu_ps(a.as_mut_ptr(), self.0) }
        }
        /// `if a < b { a } else { b }` per lane (matches `minps`).
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            unsafe { Self(_mm_min_ps(self.0, o.0)) }
        }
        /// `if a > b { a } else { b }` per lane (matches `maxps`).
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            unsafe { Self(_mm_max_ps(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn gt(self, o: Self) -> Mask4 {
            unsafe { Mask4(_mm_cmpgt_ps(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn le(self, o: Self) -> Mask4 {
            unsafe { Mask4(_mm_cmple_ps(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn ge(self, o: Self) -> Mask4 {
            unsafe { Mask4(_mm_cmpge_ps(self.0, o.0)) }
        }
        /// Lane-wise `self & mask` (lanes where mask is false become +0.0).
        #[inline(always)]
        pub fn and_mask(self, m: Mask4) -> Self {
            unsafe { Self(_mm_and_ps(self.0, m.0)) }
        }
        /// Lane-wise `floor(x)` converted to `i32`, input clamped to ±1e9 first so the
        /// result is identical on every backend (SSE returns 0x8000_0000 for overflow,
        /// scalar casts saturate — clamping removes the discrepancy).
        #[inline(always)]
        pub fn floor_i32(self) -> [i32; 4] {
            unsafe {
                let x = _mm_min_ps(_mm_max_ps(self.0, _mm_set1_ps(-1.0e9)), _mm_set1_ps(1.0e9));
                let t = _mm_cvttps_epi32(x); // truncate toward zero
                let tf = _mm_cvtepi32_ps(t);
                let lt = _mm_castps_si128(_mm_cmplt_ps(x, tf)); // -1 where x < trunc(x) (negative non-integers)
                let r = _mm_add_epi32(t, lt);
                let mut out = [0i32; 4];
                _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, r);
                out
            }
        }
    }
    impl Mask4 {
        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            unsafe { Self(_mm_and_ps(self.0, o.0)) }
        }
        /// Bit i set  <=>  lane i true.
        #[inline(always)]
        pub fn bits(self) -> u32 {
            unsafe { _mm_movemask_ps(self.0) as u32 }
        }
    }
    impl Add for F32x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            unsafe { Self(_mm_add_ps(self.0, o.0)) }
        }
    }
    impl Sub for F32x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            unsafe { Self(_mm_sub_ps(self.0, o.0)) }
        }
    }
    impl Mul for F32x4 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            unsafe { Self(_mm_mul_ps(self.0, o.0)) }
        }
    }
}

#[cfg(any(not(target_arch = "x86_64"), feature = "portable_simd"))]
mod imp {
    use core::ops::{Add, Mul, Sub};

    #[derive(Clone, Copy)]
    pub struct F32x4([f32; 4]);
    #[derive(Clone, Copy)]
    pub struct Mask4([bool; 4]);

    pub const BACKEND: &str = "portable scalar fallback";

    #[inline(always)]
    fn zip(a: [f32; 4], b: [f32; 4], f: impl Fn(f32, f32) -> f32) -> F32x4 {
        F32x4([f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])])
    }
    #[inline(always)]
    fn cmp(a: [f32; 4], b: [f32; 4], f: impl Fn(f32, f32) -> bool) -> Mask4 {
        Mask4([f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])])
    }

    impl F32x4 {
        #[inline(always)]
        pub fn splat(v: f32) -> Self {
            Self([v; 4])
        }
        #[inline(always)]
        pub fn load(a: &[f32; 4]) -> Self {
            Self(*a)
        }
        #[inline(always)]
        pub fn store(self, a: &mut [f32; 4]) {
            *a = self.0;
        }
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            zip(self.0, o.0, |a, b| if a < b { a } else { b })
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            zip(self.0, o.0, |a, b| if a > b { a } else { b })
        }
        #[inline(always)]
        pub fn gt(self, o: Self) -> Mask4 {
            cmp(self.0, o.0, |a, b| a > b)
        }
        #[inline(always)]
        pub fn le(self, o: Self) -> Mask4 {
            cmp(self.0, o.0, |a, b| a <= b)
        }
        #[inline(always)]
        pub fn ge(self, o: Self) -> Mask4 {
            cmp(self.0, o.0, |a, b| a >= b)
        }
        #[inline(always)]
        pub fn and_mask(self, m: Mask4) -> Self {
            let s = self.0;
            Self([
                if m.0[0] { s[0] } else { 0.0 },
                if m.0[1] { s[1] } else { 0.0 },
                if m.0[2] { s[2] } else { 0.0 },
                if m.0[3] { s[3] } else { 0.0 },
            ])
        }
        #[inline(always)]
        pub fn floor_i32(self) -> [i32; 4] {
            let mut out = [0i32; 4];
            for i in 0..4 {
                let a = self.0[i];
                // same NaN/ordering semantics as maxps/minps
                let x = if a > -1.0e9 { a } else { -1.0e9 };
                let x = if x < 1.0e9 { x } else { 1.0e9 };
                let t = x as i32;
                out[i] = if x < t as f32 { t - 1 } else { t };
            }
            out
        }
    }
    impl Mask4 {
        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            Self([self.0[0] & o.0[0], self.0[1] & o.0[1], self.0[2] & o.0[2], self.0[3] & o.0[3]])
        }
        #[inline(always)]
        pub fn bits(self) -> u32 {
            (self.0[0] as u32) | (self.0[1] as u32) << 1 | (self.0[2] as u32) << 2 | (self.0[3] as u32) << 3
        }
    }
    impl Add for F32x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            zip(self.0, o.0, |a, b| a + b)
        }
    }
    impl Sub for F32x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            zip(self.0, o.0, |a, b| a - b)
        }
    }
    impl Mul for F32x4 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            zip(self.0, o.0, |a, b| a * b)
        }
    }
}

pub use imp::{F32x4, Mask4, BACKEND};

/// Number of lanes processed per vector operation.
pub const LANES: usize = 4;

/// View a slice chunk of exactly 4 elements as `&[f32; 4]` (safe: `chunks_exact` guarantees length).
#[inline(always)]
pub fn as4(c: &[f32]) -> &[f32; 4] {
    c.try_into().expect("chunk of 4")
}
/// Mutable version of [`as4`].
#[inline(always)]
pub fn as4_mut(c: &mut [f32]) -> &mut [f32; 4] {
    c.try_into().expect("chunk of 4")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_matches_scalar_bitwise() {
        let a = [1.5f32, -2.25, 1e-30, 3.0e7];
        let b = [0.1f32, 4.0, -7.0, 1.0 / 3.0];
        let (va, vb) = (F32x4::load(&a), F32x4::load(&b));
        let mut s = [0f32; 4];
        (va + vb).store(&mut s);
        for i in 0..4 { assert_eq!(s[i].to_bits(), (a[i] + b[i]).to_bits()); }
        (va - vb).store(&mut s);
        for i in 0..4 { assert_eq!(s[i].to_bits(), (a[i] - b[i]).to_bits()); }
        (va * vb).store(&mut s);
        for i in 0..4 { assert_eq!(s[i].to_bits(), (a[i] * b[i]).to_bits()); }
    }

    #[test]
    fn floor_matches_std_floor() {
        let xs = [-2.5f32, -0.0, 0.999, -1.0];
        assert_eq!(F32x4::load(&xs).floor_i32(), [-3, 0, 0, -1]);
        let ys = [1.0e12f32, -1.0e12, 5.0, -5.5];
        assert_eq!(F32x4::load(&ys).floor_i32(), [1_000_000_000, -1_000_000_000, 5, -6]);
        let mut rng = crate::util::Rng::new(7);
        for _ in 0..10_000 {
            let v = [rng.range(-1e4, 1e4), rng.range(-10.0, 10.0), rng.range(-1.0, 1.0), rng.range(-1e6, 1e6)];
            let got = F32x4::load(&v).floor_i32();
            for i in 0..4 { assert_eq!(got[i], v[i].floor() as i32, "x={}", v[i]); }
        }
    }

    #[test]
    fn mask_bits_and_select() {
        let a = F32x4::load(&[1.0, 5.0, 1.0, 5.0]);
        let b = F32x4::splat(3.0);
        assert_eq!(a.gt(b).bits(), 0b1010);
        assert_eq!(a.le(b).bits(), 0b0101);
        let mut o = [9f32; 4];
        a.and_mask(a.gt(b)).store(&mut o);
        assert_eq!(o, [0.0, 5.0, 0.0, 5.0]);
    }
}
