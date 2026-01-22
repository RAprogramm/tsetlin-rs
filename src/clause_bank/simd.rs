//! SIMD-optimized feedback operations for [`ClauseBank`].
//!
//! Uses `portable_simd` to vectorize automata state updates.
//! Processes 8 features (16 states) per SIMD operation.
//!
//! # Performance
//!
//! | Operation | Scalar | SIMD | Speedup |
//! |-----------|--------|------|---------|
//! | Type II | 1.4µs | ~0.4µs | ~3.5x |
//! | Type I | 4.2µs | ~1.2µs | ~3.5x |
//!
//! # Feature Requirements
//!
//! Requires `simd` feature and nightly Rust:
//! ```toml
//! [features]
//! simd = []
//! ```

#![cfg(feature = "simd")]

use std::simd::{
    Mask, Simd,
    cmp::{SimdPartialEq, SimdPartialOrd},
    i16x16
};

use rand::Rng;

use super::ClauseBank;

/// SIMD lane width for i16 operations.
const LANES: usize = 16;

/// Features processed per SIMD iteration (16 states / 2 per feature).
const FEATURES_PER_SIMD: usize = LANES / 2;

impl ClauseBank {
    /// SIMD-optimized Type II feedback.
    ///
    /// Vectorized version of [`ClauseBank::type_ii`]. Processes 8 features
    /// per SIMD operation using i16x16 vectors.
    ///
    /// # Arguments
    ///
    /// * `clause` - Clause index
    /// * `x` - Binary input that caused false positive
    ///
    /// # Algorithm
    ///
    /// For each chunk of 8 features:
    /// 1. Load 16 states (8 pos + 8 neg interleaved)
    /// 2. Build mask based on x values
    /// 3. Conditionally increment where state ≤ threshold and < max
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tsetlin_rs::ClauseBank;
    ///
    /// let mut bank = ClauseBank::new(1, 64, 100);
    /// bank.type_ii_simd(0, &input);
    /// ```
    pub fn type_ii_simd(&mut self, clause: usize, x: &[u8]) {
        let n = x.len().min(self.n_features);
        let base = clause * self.stride;
        let max = 2 * self.n_states;
        let threshold = self.n_states;

        let threshold_vec = i16x16::splat(threshold);
        let max_vec = i16x16::splat(max);
        let one = i16x16::splat(1);
        let zero = i16x16::splat(0);

        let chunks = n / FEATURES_PER_SIMD;

        for chunk in 0..chunks {
            let offset = base + chunk * LANES;
            let x_offset = chunk * FEATURES_PER_SIMD;

            // Load 16 states (8 feature pairs: pos0,neg0,pos1,neg1,...)
            let mut states = i16x16::from_slice(&self.states[offset..offset + LANES]);

            // Build update mask from x values
            // x[k]=0 -> increment pos[k] at index 2k
            // x[k]=1 -> increment neg[k] at index 2k+1
            let mask = build_type_ii_mask(&x[x_offset..x_offset + FEATURES_PER_SIMD]);

            // Only update if in exclusion zone (state <= threshold) and below max
            let in_exclusion = states.simd_le(threshold_vec);
            let below_max = states.simd_lt(max_vec);
            let can_update = in_exclusion & below_max;

            // Apply conditional increment
            let want_update = mask.simd_eq(one);
            let final_mask = want_update & can_update;
            states += final_mask.select(one, zero);

            // Store back
            states.copy_to_slice(&mut self.states[offset..offset + LANES]);
        }

        // Scalar fallback for remaining features
        for k in (chunks * FEATURES_PER_SIMD)..n {
            let pos = base + 2 * k;
            let neg = base + 2 * k + 1;

            if x[k] == 0 {
                if self.states[pos] <= threshold && self.states[pos] < max {
                    self.states[pos] += 1;
                }
            } else if self.states[neg] <= threshold && self.states[neg] < max {
                self.states[neg] += 1;
            }
        }
    }

    /// SIMD-optimized Type I feedback.
    ///
    /// Vectorized version of [`ClauseBank::type_i`]. Pre-generates random
    /// numbers and uses SIMD for threshold comparisons and updates.
    ///
    /// # Arguments
    ///
    /// * `clause` - Clause index
    /// * `x` - Binary input
    /// * `fires` - Whether clause fired
    /// * `s` - Specificity parameter
    /// * `rng` - Random number generator
    ///
    /// # Algorithm
    ///
    /// 1. Pre-generate random floats for all automata
    /// 2. Convert to threshold masks using SIMD comparison
    /// 3. Build update masks based on x values and firing status
    /// 4. Apply conditional increment/decrement
    pub fn type_i_simd<R: Rng>(
        &mut self,
        clause: usize,
        x: &[u8],
        fires: bool,
        s: f32,
        rng: &mut R
    ) {
        let prob_str = (s - 1.0) / s;
        let prob_wk = 1.0 / s;
        let n = x.len().min(self.n_features);
        let base = clause * self.stride;
        let max = 2 * self.n_states;

        let max_vec = i16x16::splat(max);
        let min_vec = i16x16::splat(1);
        let one = i16x16::splat(1);
        let zero = i16x16::splat(0);

        if !fires {
            // Clause didn't fire: weaken all automata
            let chunks = self.stride / LANES;

            for chunk in 0..chunks {
                let offset = base + chunk * LANES;
                let mut states = i16x16::from_slice(&self.states[offset..offset + LANES]);

                // Generate 16 random checks
                let weaken_mask = gen_random_mask(rng, prob_wk, LANES);
                let above_min = states.simd_gt(min_vec);
                let can_decrement = weaken_mask & above_min;

                states -= can_decrement.select(one, zero);
                states.copy_to_slice(&mut self.states[offset..offset + LANES]);
            }

            // Scalar fallback for remaining
            for i in (chunks * LANES)..self.stride {
                if rng.random::<f32>() <= prob_wk && self.states[base + i] > 1 {
                    self.states[base + i] -= 1;
                }
            }
            return;
        }

        // Clause fired: reinforce matching pattern
        let chunks = n / FEATURES_PER_SIMD;

        for chunk in 0..chunks {
            let offset = base + chunk * LANES;
            let x_offset = chunk * FEATURES_PER_SIMD;
            let x_slice = &x[x_offset..x_offset + FEATURES_PER_SIMD];

            let mut states = i16x16::from_slice(&self.states[offset..offset + LANES]);

            // Build masks for strengthen and weaken operations
            let (strengthen_mask, weaken_mask) = build_type_i_masks(x_slice);

            // Generate random threshold masks
            let str_rand = gen_random_mask(rng, prob_str, LANES);
            let wk_rand = gen_random_mask(rng, prob_wk, LANES);

            // Strengthen: increment if want_str AND random passed AND below max
            let want_str = strengthen_mask.simd_eq(one);
            let below_max = states.simd_lt(max_vec);
            let do_str = want_str & str_rand & below_max;

            // Weaken: decrement if want_wk AND random passed AND above min
            let want_wk = weaken_mask.simd_eq(one);
            let above_min = states.simd_gt(min_vec);
            let do_wk = want_wk & wk_rand & above_min;

            // Apply updates
            states += do_str.select(one, zero);
            states -= do_wk.select(one, zero);

            states.copy_to_slice(&mut self.states[offset..offset + LANES]);
        }

        // Scalar fallback for remaining features
        for k in (chunks * FEATURES_PER_SIMD)..n {
            let pos = base + 2 * k;
            let neg = base + 2 * k + 1;

            if x[k] == 1 {
                if rng.random::<f32>() <= prob_str && self.states[pos] < max {
                    self.states[pos] += 1;
                }
                if rng.random::<f32>() <= prob_wk && self.states[neg] > 1 {
                    self.states[neg] -= 1;
                }
            } else {
                if rng.random::<f32>() <= prob_str && self.states[neg] < max {
                    self.states[neg] += 1;
                }
                if rng.random::<f32>() <= prob_wk && self.states[pos] > 1 {
                    self.states[pos] -= 1;
                }
            }
        }
    }
}

/// Builds Type II update mask from input values.
///
/// For 8 features, creates a 16-element mask where:
/// - Position 2k = 1 if x[k] = 0 (increment pos literal)
/// - Position 2k+1 = 1 if x[k] = 1 (increment neg literal)
#[inline]
fn build_type_ii_mask(x: &[u8]) -> Simd<i16, LANES> {
    let mut mask = [0i16; LANES];
    for k in 0..FEATURES_PER_SIMD.min(x.len()) {
        if x[k] == 0 {
            mask[2 * k] = 1;
        } else {
            mask[2 * k + 1] = 1;
        }
    }
    i16x16::from_array(mask)
}

/// Builds Type I strengthen and weaken masks from input values.
///
/// When clause fires:
/// - x[k]=1: strengthen pos[k], weaken neg[k]
/// - x[k]=0: strengthen neg[k], weaken pos[k]
#[inline]
fn build_type_i_masks(x: &[u8]) -> (Simd<i16, LANES>, Simd<i16, LANES>) {
    let mut strengthen = [0i16; LANES];
    let mut weaken = [0i16; LANES];

    for k in 0..FEATURES_PER_SIMD.min(x.len()) {
        if x[k] == 1 {
            strengthen[2 * k] = 1; // strengthen pos
            weaken[2 * k + 1] = 1; // weaken neg
        } else {
            strengthen[2 * k + 1] = 1; // strengthen neg
            weaken[2 * k] = 1; // weaken pos
        }
    }

    (i16x16::from_array(strengthen), i16x16::from_array(weaken))
}

/// Generates a boolean mask based on random probability threshold.
#[inline]
fn gen_random_mask<R: Rng>(rng: &mut R, prob: f32, count: usize) -> Mask<i16, LANES> {
    let mut bits = [false; LANES];
    for b in bits.iter_mut().take(count) {
        *b = rng.random::<f32>() <= prob;
    }
    Mask::from_array(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::rng_from_seed;

    #[test]
    fn type_ii_simd_matches_scalar() {
        let mut bank1 = ClauseBank::new(1, 64, 100);
        let mut bank2 = ClauseBank::new(1, 64, 100);
        let x: Vec<u8> = (0..64).map(|i| (i % 2) as u8).collect();

        // Apply scalar
        bank1.type_ii(0, &x);

        // Apply SIMD
        bank2.type_ii_simd(0, &x);

        assert_eq!(bank1.states, bank2.states);
    }

    #[test]
    fn type_i_simd_fires_updates_states() {
        let mut bank = ClauseBank::new(1, 64, 100);
        let x: Vec<u8> = (0..64).map(|i| (i % 2) as u8).collect();
        let mut rng = rng_from_seed(42);

        let initial: Vec<i16> = bank.states.clone();

        for _ in 0..50 {
            bank.type_i_simd(0, &x, true, 3.9, &mut rng);
        }

        // States should have changed
        assert_ne!(bank.states, initial);
    }

    #[test]
    fn type_i_simd_no_fire_weakens() {
        let mut bank = ClauseBank::new(1, 64, 100);
        let x: Vec<u8> = (0..64).map(|i| (i % 2) as u8).collect();
        let mut rng = rng_from_seed(42);

        for _ in 0..100 {
            bank.type_i_simd(0, &x, false, 3.9, &mut rng);
        }

        // All states should have decreased
        assert!(bank.states.iter().all(|&s| s < 100));
    }

    #[test]
    fn build_type_ii_mask_correct() {
        let x = [0u8, 1, 0, 1, 1, 0, 1, 0];
        let mask = build_type_ii_mask(&x);
        let arr = mask.to_array();

        // x[0]=0 -> pos[0]=arr[0]=1
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 0);

        // x[1]=1 -> neg[1]=arr[3]=1
        assert_eq!(arr[2], 0);
        assert_eq!(arr[3], 1);

        // x[2]=0 -> pos[2]=arr[4]=1
        assert_eq!(arr[4], 1);
        assert_eq!(arr[5], 0);
    }
}
