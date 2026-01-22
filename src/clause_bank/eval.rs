//! Evaluation and voting methods for [`ClauseBank`].
//!
//! This module provides clause evaluation, vote computation, and weight
//! management for the Structure of Arrays clause storage.
//!
//! # Performance
//!
//! All evaluation methods use unchecked indexing with manually verified
//! safety invariants for maximum performance. The `sum_votes` method
//! is approximately 30% faster than iterating over `Vec<Clause>`.

use super::ClauseBank;

impl ClauseBank {
    /// Evaluates a single clause against binary input.
    ///
    /// A clause "fires" (returns `true`) when all included literals are
    /// satisfied by the input:
    /// - For literal `x_k`: if included, `x[k]` must be `1`
    /// - For literal `¬x_k`: if included, `x[k]` must be `0`
    ///
    /// # Arguments
    ///
    /// * `clause_idx` - Index of the clause to evaluate
    /// * `x` - Binary input vector (values should be 0 or 1)
    ///
    /// # Returns
    ///
    /// `true` if the clause fires, `false` otherwise.
    ///
    /// # Performance
    ///
    /// Uses unchecked indexing for both state and input arrays.
    /// The loop processes `min(n_features, x.len())` features.
    ///
    /// # Safety
    ///
    /// This method uses `unsafe` blocks with the following invariants:
    /// - `k < n` where `n <= n_features` and `n <= x.len()`
    /// - `base + 2*k + 1 < states.len()` because `states.len() = n_clauses *
    ///   stride` and `stride = 2 * n_features`
    ///
    /// # Example
    ///
    /// ```
    /// use tsetlin_rs::ClauseBank;
    ///
    /// let bank = ClauseBank::new(1, 4, 100);
    /// // Fresh clauses have no included literals, so they always fire
    /// assert!(bank.evaluate_clause(0, &[0, 0, 0, 0]));
    /// assert!(bank.evaluate_clause(0, &[1, 1, 1, 1]));
    /// ```
    #[inline]
    #[must_use]
    pub fn evaluate_clause(&self, clause_idx: usize, x: &[u8]) -> bool {
        let base = clause_idx * self.stride;
        let n = self.n_features.min(x.len());
        let threshold = self.n_states;
        let states = &self.states;

        for k in 0..n {
            // SAFETY: k < n <= n_features, base = clause * 2 * n_features
            // 2 * k + 1 < 2 * n_features, so base + 2*k + 1 < states.len()
            let include = unsafe { *states.get_unchecked(base + 2 * k) > threshold };
            let negated = unsafe { *states.get_unchecked(base + 2 * k + 1) > threshold };

            // SAFETY: k < n <= x.len()
            let xk = unsafe { *x.get_unchecked(k) };

            if include && xk == 0 {
                return false;
            }
            if negated && xk == 1 {
                return false;
            }
        }
        true
    }

    /// Evaluates a clause and increments its activation counter if it fires.
    ///
    /// Use this during training to track which clauses are active.
    /// The activation count is used for clause pruning.
    ///
    /// # Arguments
    ///
    /// * `clause_idx` - Index of the clause to evaluate
    /// * `x` - Binary input vector
    ///
    /// # Returns
    ///
    /// `true` if the clause fires, `false` otherwise.
    #[inline]
    pub fn evaluate_clause_tracked(&mut self, clause_idx: usize, x: &[u8]) -> bool {
        let fires = self.evaluate_clause(clause_idx, x);
        if fires {
            self.activations[clause_idx] = self.activations[clause_idx].saturating_add(1);
        }
        fires
    }

    /// Computes the weighted vote for a single clause.
    ///
    /// Returns `polarity * weight` if the clause fires, `0.0` otherwise.
    ///
    /// # Arguments
    ///
    /// * `clause_idx` - Index of the clause
    /// * `x` - Binary input vector
    ///
    /// # Returns
    ///
    /// Weighted vote contribution (positive, negative, or zero).
    #[inline(always)]
    #[must_use]
    pub fn clause_vote(&self, clause_idx: usize, x: &[u8]) -> f32 {
        if self.evaluate_clause(clause_idx, x) {
            self.polarities[clause_idx] as f32 * self.weights[clause_idx]
        } else {
            0.0
        }
    }

    /// Computes the sum of weighted votes from all clauses.
    ///
    /// This is the main inference operation. The sign of the result
    /// determines the predicted class:
    /// - Positive sum → class 1
    /// - Negative sum → class 0
    ///
    /// # Arguments
    ///
    /// * `x` - Binary input vector
    ///
    /// # Returns
    ///
    /// Sum of `polarity * weight` for all firing clauses.
    ///
    /// # Performance
    ///
    /// Optimized with unchecked indexing. Approximately 30% faster than
    /// iterating over `Vec<Clause>` due to better cache locality.
    ///
    /// # Example
    ///
    /// ```
    /// use tsetlin_rs::ClauseBank;
    ///
    /// let bank = ClauseBank::new(100, 64, 100);
    /// let input = vec![1u8; 64];
    /// let vote = bank.sum_votes(&input);
    /// // All clauses fire with initial state, votes cancel out (alternating polarity)
    /// assert!((vote - 0.0).abs() < 0.001);
    /// ```
    #[inline]
    #[must_use]
    pub fn sum_votes(&self, x: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.n_clauses {
            if self.evaluate_clause(i, x) {
                // SAFETY: i < n_clauses, arrays have length n_clauses
                sum += unsafe {
                    *self.polarities.get_unchecked(i) as f32 * *self.weights.get_unchecked(i)
                };
            }
        }
        sum
    }

    /// Computes sum of votes while tracking activations.
    ///
    /// Combines [`sum_votes`](Self::sum_votes) and activation tracking in
    /// a single pass. Use during training.
    ///
    /// # Arguments
    ///
    /// * `x` - Binary input vector
    ///
    /// # Returns
    ///
    /// Sum of weighted votes from all firing clauses.
    #[inline]
    pub fn sum_votes_tracked(&mut self, x: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.n_clauses {
            if self.evaluate_clause(i, x) {
                self.activations[i] = self.activations[i].saturating_add(1);
                sum += self.polarities[i] as f32 * self.weights[i];
            }
        }
        sum
    }

    /// Records the outcome of a prediction for weight learning.
    ///
    /// Call this when a clause fires and a prediction is made.
    /// The accumulated correct/incorrect counts are used by
    /// [`update_weights`](Self::update_weights) at epoch end.
    ///
    /// # Arguments
    ///
    /// * `clause_idx` - Index of the clause that fired
    /// * `was_correct` - Whether the prediction was correct
    #[inline]
    pub fn record_outcome(&mut self, clause_idx: usize, was_correct: bool) {
        if was_correct {
            self.correct[clause_idx] = self.correct[clause_idx].saturating_add(1);
        } else {
            self.incorrect[clause_idx] = self.incorrect[clause_idx].saturating_add(1);
        }
    }

    /// Updates all clause weights based on accumulated prediction outcomes.
    ///
    /// Should be called at the end of each training epoch. Weights are
    /// adjusted based on each clause's accuracy:
    /// - Accuracy > 50%: weight increases
    /// - Accuracy < 50%: weight decreases
    ///
    /// After updating, the correct/incorrect counters are reset.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate (how fast weights change)
    /// * `min` - Minimum allowed weight
    /// * `max` - Maximum allowed weight
    ///
    /// # Algorithm
    ///
    /// ```text
    /// accuracy = correct / (correct + incorrect)
    /// adjustment = (accuracy - 0.5) * 2.0 * lr
    /// new_weight = clamp(weight + adjustment, min, max)
    /// ```
    pub fn update_weights(&mut self, lr: f32, min: f32, max: f32) {
        for i in 0..self.n_clauses {
            let total = self.correct[i] + self.incorrect[i];
            if total == 0 {
                continue;
            }
            let acc = self.correct[i] as f32 / total as f32;
            self.weights[i] = (self.weights[i] + (acc - 0.5) * 2.0 * lr).clamp(min, max);
            self.correct[i] = 0;
            self.incorrect[i] = 0;
        }
    }

    /// Resets all activation counters to zero.
    ///
    /// Call at the start of each training epoch.
    #[inline]
    pub fn reset_activations(&mut self) {
        self.activations.fill(0);
    }

    /// Checks if a clause is considered "dead" (ineffective).
    ///
    /// A dead clause either:
    /// - Rarely activates (activation count below threshold)
    /// - Has very low weight (weight below threshold)
    ///
    /// Dead clauses can be reset via [`prune_dead`](Self::prune_dead).
    ///
    /// # Arguments
    ///
    /// * `idx` - Clause index
    /// * `min_act` - Minimum activation count to be considered alive
    /// * `min_wt` - Minimum weight to be considered alive
    ///
    /// # Returns
    ///
    /// `true` if the clause should be pruned.
    #[inline]
    #[must_use]
    pub fn is_dead(&self, idx: usize, min_act: u32, min_wt: f32) -> bool {
        self.activations[idx] < min_act || self.weights[idx] < min_wt
    }

    /// Resets a clause to its initial state.
    ///
    /// Sets all automata to threshold state, weight to 1.0, and clears
    /// all counters. Use for clause pruning or reinitialization.
    ///
    /// # Arguments
    ///
    /// * `idx` - Index of the clause to reset
    pub fn reset_clause(&mut self, idx: usize) {
        let n_states = self.n_states;
        let start = idx * self.stride;
        self.states[start..start + self.stride].fill(n_states);
        self.weights[idx] = 1.0;
        self.activations[idx] = 0;
        self.correct[idx] = 0;
        self.incorrect[idx] = 0;
    }

    /// Resets all dead clauses to their initial state.
    ///
    /// Iterates through all clauses and resets any that are considered
    /// dead according to [`is_dead`](Self::is_dead).
    ///
    /// # Arguments
    ///
    /// * `min_act` - Minimum activation count to be considered alive
    /// * `min_wt` - Minimum weight to be considered alive
    pub fn prune_dead(&mut self, min_act: u32, min_wt: f32) {
        for i in 0..self.n_clauses {
            if self.is_dead(i, min_act, min_wt) {
                self.reset_clause(i);
            }
        }
    }

    /// Evaluates all clauses and populates the firing bitmap.
    ///
    /// Returns the sum of weighted votes while recording which clauses
    /// fired in the internal bitmap. Use [`clause_fires`](Self::clause_fires)
    /// to check individual clause status.
    ///
    /// This is more efficient than calling `evaluate_clause` in a loop
    /// when you need both the vote sum and firing status for feedback.
    ///
    /// # Arguments
    ///
    /// * `x` - Binary input vector
    ///
    /// # Returns
    ///
    /// Sum of weighted votes from all firing clauses.
    #[inline]
    pub fn evaluate_all(&mut self, x: &[u8]) -> f32 {
        self.fires_bitmap.fill(0);
        let mut sum = 0.0f32;

        for i in 0..self.n_clauses {
            if self.evaluate_clause(i, x) {
                // Set bit in bitmap
                let word = i / 64;
                let bit = i % 64;
                self.fires_bitmap[word] |= 1u64 << bit;

                // Accumulate vote
                sum += unsafe {
                    *self.polarities.get_unchecked(i) as f32 * *self.weights.get_unchecked(i)
                };

                // Track activation
                self.activations[i] = self.activations[i].saturating_add(1);
            }
        }
        sum
    }

    /// Checks if a specific clause fired in the last evaluation.
    ///
    /// # Arguments
    ///
    /// * `clause` - Index of the clause to check
    ///
    /// # Returns
    ///
    /// `true` if the clause fired during the last
    /// [`evaluate_all`](Self::evaluate_all) call.
    ///
    /// # Panics
    ///
    /// Panics if `clause >= n_clauses`.
    #[inline(always)]
    #[must_use]
    pub fn clause_fires(&self, clause: usize) -> bool {
        let word = clause / 64;
        let bit = clause % 64;
        self.fires_bitmap[word] & (1u64 << bit) != 0
    }

    /// Returns an iterator over indices of clauses that fired.
    ///
    /// More efficient than checking each clause individually when
    /// processing only firing clauses.
    #[inline]
    pub fn firing_clauses(&self) -> impl Iterator<Item = usize> + '_ {
        self.fires_bitmap
            .iter()
            .enumerate()
            .flat_map(|(word_idx, &word)| FiringBits {
                word,
                base: word_idx * 64,
                max: self.n_clauses
            })
    }

    /// Returns the number of clauses that fired in the last evaluation.
    #[inline]
    #[must_use]
    pub fn firing_count(&self) -> usize {
        self.fires_bitmap
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }
}

/// Iterator over set bits in a u64 word.
struct FiringBits {
    word: u64,
    base: usize,
    max:  usize
}

impl Iterator for FiringBits {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.word != 0 {
            let tz = self.word.trailing_zeros() as usize;
            self.word &= self.word - 1; // Clear lowest set bit
            let idx = self.base + tz;
            if idx < self.max {
                return Some(idx);
            }
        }
        None
    }
}
