//! Training feedback operations for [`ClauseBank`].
//!
//! This module implements the Tsetlin Machine learning algorithm's feedback
//! mechanisms. Feedback adjusts automata states to learn patterns from data.
//!
//! # Feedback Types
//!
//! | Type | When Applied | Purpose |
//! |------|--------------|---------|
//! | Type I | Target class (y=1) | Reinforce matching patterns |
//! | Type II | Wrong class (y=0) | Block false positives |
//!
//! # Algorithm Overview
//!
//! ```text
//! For each training sample (x, y):
//!   evaluate all clauses on x
//!   for positive clauses (polarity = +1):
//!     if y = 1: apply Type I feedback (reinforce)
//!     if y = 0 and fires: apply Type II feedback (block)
//!   for negative clauses (polarity = -1):
//!     if y = 0: apply Type I feedback (reinforce)
//!     if y = 1 and fires: apply Type II feedback (block)
//! ```

use rand::Rng;

use super::ClauseBank;
use crate::config::prob_to_threshold;

impl ClauseBank {
    /// Applies Type I feedback to reinforce patterns for the target class.
    ///
    /// Type I feedback is applied when the sample belongs to the clause's
    /// target class. It either reinforces matching patterns (when clause fires)
    /// or weakens all literals (when clause doesn't fire).
    ///
    /// # Arguments
    ///
    /// * `clause` - Index of the clause to update
    /// * `x` - Binary input vector
    /// * `fires` - Whether the clause fired on this input
    /// * `s` - Specificity parameter (controls learning granularity)
    /// * `rng` - Random number generator for stochastic updates
    ///
    /// # Behavior
    ///
    /// ## When clause fires (`fires = true`):
    ///
    /// For each feature `k`:
    /// - If `x[k] = 1`: strengthen `x_k` literal, weaken `¬x_k` literal
    /// - If `x[k] = 0`: strengthen `¬x_k` literal, weaken `x_k` literal
    ///
    /// ## When clause doesn't fire (`fires = false`):
    ///
    /// Weaken all automata toward exclusion (decrement states).
    ///
    /// # Probabilities
    ///
    /// Updates are stochastic with probabilities controlled by `s`:
    /// - Strengthen probability: `(s - 1) / s`
    /// - Weaken probability: `1 / s`
    ///
    /// Higher `s` values lead to more specific patterns (fewer literals).
    ///
    /// # Example
    ///
    /// ```
    /// use tsetlin_rs::{ClauseBank, utils::rng_from_seed};
    ///
    /// let mut bank = ClauseBank::new(1, 4, 100);
    /// let mut rng = rng_from_seed(42);
    ///
    /// // Reinforce pattern [1, 0, 1, 0] for clause 0
    /// bank.type_i(0, &[1, 0, 1, 0], true, 3.9, &mut rng);
    /// ```
    ///
    /// Uses integer threshold comparison for ~2x faster RNG checks.
    pub fn type_i<R: Rng>(&mut self, clause: usize, x: &[u8], fires: bool, s: f32, rng: &mut R) {
        let threshold_str = prob_to_threshold((s - 1.0) / s);
        let threshold_wk = prob_to_threshold(1.0 / s);
        let n = x.len().min(self.n_features);
        let base = clause * self.stride;
        let max = 2 * self.n_states;

        if !fires {
            // Clause didn't fire: weaken all automata toward exclusion
            for i in 0..self.stride {
                if rng.random::<u32>() < threshold_wk && self.states[base + i] > 1 {
                    self.states[base + i] -= 1;
                }
            }
            return;
        }

        // Clause fired: reinforce matching pattern
        for (k, &xk) in x.iter().enumerate().take(n) {
            let pos = base + 2 * k;
            let neg = base + 2 * k + 1;

            if xk == 1 {
                // x[k] = 1: strengthen x_k, weaken ¬x_k
                if rng.random::<u32>() < threshold_str && self.states[pos] < max {
                    self.states[pos] += 1;
                }
                if rng.random::<u32>() < threshold_wk && self.states[neg] > 1 {
                    self.states[neg] -= 1;
                }
            } else {
                // x[k] = 0: strengthen ¬x_k, weaken x_k
                if rng.random::<u32>() < threshold_str && self.states[neg] < max {
                    self.states[neg] += 1;
                }
                if rng.random::<u32>() < threshold_wk && self.states[pos] > 1 {
                    self.states[pos] -= 1;
                }
            }
        }
    }

    /// Applies Type II feedback to correct false positives.
    ///
    /// Type II feedback is applied when a clause fires for a sample that
    /// belongs to the opposite class. It activates "blocking" literals to
    /// prevent the clause from firing on similar inputs in the future.
    ///
    /// # Arguments
    ///
    /// * `clause` - Index of the clause to update
    /// * `x` - Binary input vector that caused the false positive
    ///
    /// # Behavior
    ///
    /// For each feature `k`:
    /// - If `x[k] = 0` and `x_k` is excluded: move toward including `x_k`
    /// - If `x[k] = 1` and `¬x_k` is excluded: move toward including `¬x_k`
    ///
    /// This adds a literal that contradicts the input, causing the clause
    /// to not fire on this pattern.
    ///
    /// # Note
    ///
    /// Updates are deterministic (no probability). Only automata in the
    /// exclusion zone (state ≤ threshold) are updated.
    ///
    /// # Example
    ///
    /// ```
    /// use tsetlin_rs::ClauseBank;
    ///
    /// let mut bank = ClauseBank::new(1, 4, 100);
    ///
    /// // Block false positive on input [1, 0, 1, 0]
    /// bank.type_ii(0, &[1, 0, 1, 0]);
    /// ```
    pub fn type_ii(&mut self, clause: usize, x: &[u8]) {
        let n = x.len().min(self.n_features);
        let base = clause * self.stride;
        let max = 2 * self.n_states;
        let threshold = self.n_states;

        for (k, &xk) in x.iter().enumerate().take(n) {
            let pos = base + 2 * k;
            let neg = base + 2 * k + 1;

            if xk == 0 {
                // x[k] = 0: add x_k literal (which will fail since x[k]=0)
                if self.states[pos] <= threshold && self.states[pos] < max {
                    self.states[pos] += 1;
                }
            } else if self.states[neg] <= threshold && self.states[neg] < max {
                // x[k] = 1: add ¬x_k literal (which will fail since x[k]=1)
                self.states[neg] += 1;
            }
        }
    }

    /// Trains on a single sample using bitmap-based feedback skipping.
    ///
    /// This method is more efficient than the naive approach because it:
    /// 1. Evaluates all clauses once and caches firing status in bitmap
    /// 2. Skips Type II feedback for non-firing clauses using bitmap iteration
    /// 3. Performs probability check before processing each clause
    ///
    /// # Arguments
    ///
    /// * `x` - Binary input vector
    /// * `y` - Target class (0 or 1)
    /// * `threshold` - Voting threshold T
    /// * `s` - Specificity parameter
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// The sum of weighted votes (for computing prediction).
    ///
    /// # Performance
    ///
    /// Provides ~40% speedup on converged models where most clauses
    /// don't require feedback updates.
    pub fn train_sample<R: Rng>(
        &mut self,
        x: &[u8],
        y: u8,
        threshold: f32,
        s: f32,
        rng: &mut R
    ) -> f32 {
        // Phase 1: Evaluate all clauses and populate bitmap
        let sum = self.evaluate_all(x).clamp(-threshold, threshold);

        // Compute feedback probability
        let inv_2t = 1.0 / (2.0 * threshold);
        let prob = if y == 1 {
            (threshold - sum) * inv_2t
        } else {
            (threshold + sum) * inv_2t
        };
        let prob_threshold = prob_to_threshold(prob);

        // Phase 2: Apply feedback based on y
        if y == 1 {
            // Positive class: Type I for positive clauses, Type II for negative firing
            for clause in 0..self.n_clauses {
                let p = self.polarities[clause];
                if p == 1 {
                    // Type I for positive clauses (probability check)
                    if rng.random::<u32>() < prob_threshold {
                        let fires = self.clause_fires(clause);
                        self.type_i(clause, x, fires, s, rng);
                    }
                } else {
                    // Type II for negative clauses that fire
                    if self.clause_fires(clause) && rng.random::<u32>() < prob_threshold {
                        self.type_ii(clause, x);
                    }
                }
            }
        } else {
            // Negative class: Type I for negative clauses, Type II for positive firing
            for clause in 0..self.n_clauses {
                let p = self.polarities[clause];
                if p == -1 {
                    // Type I for negative clauses (probability check)
                    if rng.random::<u32>() < prob_threshold {
                        let fires = self.clause_fires(clause);
                        self.type_i(clause, x, fires, s, rng);
                    }
                } else {
                    // Type II for positive clauses that fire
                    if self.clause_fires(clause) && rng.random::<u32>() < prob_threshold {
                        self.type_ii(clause, x);
                    }
                }
            }
        }

        sum
    }

    /// Applies Type II feedback only to firing clauses of specified polarity.
    ///
    /// This is an optimized method that uses the bitmap to iterate only
    /// over clauses that actually need Type II feedback.
    ///
    /// # Arguments
    ///
    /// * `x` - Binary input vector
    /// * `target_polarity` - Apply to clauses with this polarity
    /// * `prob_threshold` - Pre-computed probability threshold
    /// * `rng` - Random number generator
    pub fn type_ii_firing<R: Rng>(
        &mut self,
        x: &[u8],
        target_polarity: i8,
        prob_threshold: u32,
        rng: &mut R
    ) {
        // Iterate only over firing clauses using bitmap
        for word_idx in 0..self.fires_bitmap.len() {
            let mut word = self.fires_bitmap[word_idx];
            let base = word_idx * 64;

            while word != 0 {
                let tz = word.trailing_zeros() as usize;
                word &= word - 1; // Clear lowest set bit

                let clause = base + tz;
                if clause >= self.n_clauses {
                    break;
                }

                // Only apply to clauses with matching polarity
                if self.polarities[clause] == target_polarity
                    && rng.random::<u32>() < prob_threshold
                {
                    self.type_ii(clause, x);
                }
            }
        }
    }
}
