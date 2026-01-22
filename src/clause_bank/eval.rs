//! Evaluation methods for ClauseBank.

use super::ClauseBank;

impl ClauseBank {
    /// Evaluates a single clause on binary input.
    ///
    /// Returns `true` if all included literals are satisfied.
    #[inline]
    #[must_use]
    pub fn evaluate_clause(&self, clause_idx: usize, x: &[u8]) -> bool {
        let states = self.clause_states(clause_idx);
        let n = self.n_features.min(x.len());
        let threshold = self.n_states;

        for (k, &xk) in x.iter().enumerate().take(n) {
            // SAFETY: k < n <= n_features, states.len() == 2 * n_features
            let include = unsafe { *states.get_unchecked(2 * k) > threshold };
            let negated = unsafe { *states.get_unchecked(2 * k + 1) > threshold };

            if include && xk == 0 {
                return false;
            }
            if negated && xk == 1 {
                return false;
            }
        }
        true
    }

    /// Evaluates clause and tracks activation.
    #[inline]
    pub fn evaluate_clause_tracked(&mut self, clause_idx: usize, x: &[u8]) -> bool {
        let fires = self.evaluate_clause(clause_idx, x);
        if fires {
            self.activations[clause_idx] = self.activations[clause_idx].saturating_add(1);
        }
        fires
    }

    /// Returns weighted vote for a clause.
    #[inline(always)]
    #[must_use]
    pub fn clause_vote(&self, clause_idx: usize, x: &[u8]) -> f32 {
        if self.evaluate_clause(clause_idx, x) {
            self.polarities[clause_idx] as f32 * self.weights[clause_idx]
        } else {
            0.0
        }
    }

    /// Computes sum of all weighted clause votes.
    #[inline]
    #[must_use]
    pub fn sum_votes(&self, x: &[u8]) -> f32 {
        (0..self.n_clauses).map(|i| self.clause_vote(i, x)).sum()
    }

    /// Computes sum of votes with activation tracking.
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

    /// Records prediction outcome for weight learning.
    #[inline]
    pub fn record_outcome(&mut self, clause_idx: usize, was_correct: bool) {
        if was_correct {
            self.correct[clause_idx] = self.correct[clause_idx].saturating_add(1);
        } else {
            self.incorrect[clause_idx] = self.incorrect[clause_idx].saturating_add(1);
        }
    }

    /// Updates all clause weights. Call at end of epoch.
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

    /// Resets all activation counters.
    #[inline]
    pub fn reset_activations(&mut self) {
        self.activations.fill(0);
    }

    /// Checks if clause is dead.
    #[inline]
    #[must_use]
    pub fn is_dead(&self, idx: usize, min_act: u32, min_wt: f32) -> bool {
        self.activations[idx] < min_act || self.weights[idx] < min_wt
    }

    /// Resets a clause to initial state.
    pub fn reset_clause(&mut self, idx: usize) {
        let n_states = self.n_states;
        let start = idx * self.stride;
        self.states[start..start + self.stride].fill(n_states);
        self.weights[idx] = 1.0;
        self.activations[idx] = 0;
        self.correct[idx] = 0;
        self.incorrect[idx] = 0;
    }

    /// Prunes dead clauses.
    pub fn prune_dead(&mut self, min_act: u32, min_wt: f32) {
        for i in 0..self.n_clauses {
            if self.is_dead(i, min_act, min_wt) {
                self.reset_clause(i);
            }
        }
    }
}
