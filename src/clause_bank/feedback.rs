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
}
