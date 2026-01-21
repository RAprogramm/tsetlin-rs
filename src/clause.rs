//! Clause - a conjunction of literals with weighted voting and activation
//! tracking.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Automaton;

/// A clause with 2*n_features automata, weight, and activation tracking.
///
/// Cache-aligned for better performance. Layout optimized to minimize padding.
///
/// - `automata[2*k]` controls literal `x_k`
/// - `automata[2*k+1]` controls literal `NOT x_k`
///
/// # Memory Layout
///
/// Fields are ordered to minimize padding on 64-bit systems:
/// - `automata: Vec` (24 bytes)
/// - `n_features: usize` (8 bytes)
/// - `weight: f32` (4 bytes)
/// - `activations: u32` (4 bytes)
/// - `correct: u32` (4 bytes)
/// - `incorrect: u32` (4 bytes)
/// - `polarity: i8` (1 byte + 7 padding to 64-byte alignment)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(align(64))]
pub struct Clause {
    automata:    Vec<Automaton>,
    n_features:  usize,
    weight:      f32,
    activations: u32,
    correct:     u32,
    incorrect:   u32,
    polarity:    i8
}

impl Clause {
    /// Creates clause with given features, states, and polarity.
    ///
    /// # Arguments
    ///
    /// * `n_features` - Number of input features
    /// * `n_states` - States per automaton (threshold for action)
    /// * `polarity` - Must be +1 or -1
    ///
    /// # Panics
    ///
    /// Debug-asserts that polarity is +1 or -1.
    #[must_use]
    pub fn new(n_features: usize, n_states: i16, polarity: i8) -> Self {
        debug_assert!(polarity == 1 || polarity == -1, "polarity must be +1 or -1");
        let automata = (0..2 * n_features)
            .map(|_| Automaton::new(n_states))
            .collect();
        Self {
            automata,
            n_features,
            weight: 1.0,
            activations: 0,
            correct: 0,
            incorrect: 0,
            polarity
        }
    }

    /// Returns the clause polarity (+1 or -1).
    #[inline(always)]
    #[must_use]
    pub const fn polarity(&self) -> i8 {
        self.polarity
    }

    /// Returns the number of input features.
    #[inline(always)]
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the current clause weight.
    #[inline(always)]
    #[must_use]
    pub const fn weight(&self) -> f32 {
        self.weight
    }

    /// Returns the activation count since last reset.
    #[inline(always)]
    #[must_use]
    pub const fn activations(&self) -> u32 {
        self.activations
    }

    /// Returns read-only access to automata.
    #[inline(always)]
    #[must_use]
    pub fn automata(&self) -> &[Automaton] {
        &self.automata
    }

    /// Returns mutable access to automata.
    #[inline(always)]
    pub fn automata_mut(&mut self) -> &mut [Automaton] {
        &mut self.automata
    }

    /// Evaluates clause on binary input with early exit on violation.
    ///
    /// Returns `true` if all included literals are satisfied:
    /// - For each feature k where `include[k]` is active, `x[k]` must be 1
    /// - For each feature k where `negated[k]` is active, `x[k]` must be 0
    ///
    /// # Performance
    ///
    /// Uses unchecked indexing for performance. The safety invariants
    /// are maintained by the loop bounds.
    #[inline]
    #[must_use]
    pub fn evaluate(&self, x: &[u8]) -> bool {
        let automata = &self.automata;
        let n = self.n_features.min(x.len());

        for k in 0..n {
            // SAFETY: `k < n <= self.n_features`, and `automata.len() == 2 * n_features`.
            // Therefore `2 * k < 2 * n_features == automata.len()` and
            // `2 * k + 1 < 2 * n_features == automata.len()`.
            let include = unsafe { automata.get_unchecked(2 * k).action() };
            let negated = unsafe { automata.get_unchecked(2 * k + 1).action() };

            // SAFETY: `k < n <= x.len()`, so `k` is always in bounds.
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

    /// Evaluates clause and tracks activation count.
    ///
    /// Use during training to track which clauses are active.
    #[inline]
    pub fn evaluate_tracked(&mut self, x: &[u8]) -> bool {
        let fires = self.evaluate(x);
        if fires {
            self.activations = self.activations.saturating_add(1);
        }
        fires
    }

    /// Returns weighted vote: `polarity * weight` if clause fires, `0.0`
    /// otherwise.
    #[inline(always)]
    #[must_use]
    pub fn vote(&self, x: &[u8]) -> f32 {
        if self.evaluate(x) {
            self.polarity as f32 * self.weight
        } else {
            0.0
        }
    }

    /// Returns unweighted vote: `polarity` if fires, `0` otherwise.
    ///
    /// Use for compatibility with original Tsetlin Machine algorithm.
    #[inline(always)]
    #[must_use]
    pub fn vote_unweighted(&self, x: &[u8]) -> i32 {
        if self.evaluate(x) {
            self.polarity as i32
        } else {
            0
        }
    }

    /// Records prediction outcome for weight learning.
    ///
    /// Call when clause fired and prediction was made.
    #[inline]
    pub fn record_outcome(&mut self, was_correct: bool) {
        if was_correct {
            self.correct = self.correct.saturating_add(1);
        } else {
            self.incorrect = self.incorrect.saturating_add(1);
        }
    }

    /// Updates weight based on accumulated outcomes.
    ///
    /// Weight increases when clause predictions are accurate,
    /// decreases when inaccurate. Call at end of each epoch.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - How fast weight changes (0.0 - 1.0)
    /// * `min_weight` - Minimum allowed weight
    /// * `max_weight` - Maximum allowed weight
    pub fn update_weight(&mut self, learning_rate: f32, min_weight: f32, max_weight: f32) {
        let total = self.correct + self.incorrect;
        if total == 0 {
            return;
        }

        let accuracy = self.correct as f32 / total as f32;
        let adjustment = (accuracy - 0.5) * 2.0 * learning_rate;
        self.weight = (self.weight + adjustment).clamp(min_weight, max_weight);

        self.correct = 0;
        self.incorrect = 0;
    }

    /// Returns `true` if clause is "dead" (rarely activates or very low
    /// weight).
    ///
    /// Dead clauses can be pruned and reset during training.
    #[inline]
    #[must_use]
    pub const fn is_dead(&self, min_activations: u32, min_weight: f32) -> bool {
        self.activations < min_activations || self.weight < min_weight
    }

    /// Resets activation counter. Call at start of each epoch.
    #[inline]
    pub fn reset_activations(&mut self) {
        self.activations = 0;
    }

    /// Resets all statistics (activations, correct, incorrect).
    #[inline]
    pub fn reset_stats(&mut self) {
        self.activations = 0;
        self.correct = 0;
        self.incorrect = 0;
    }

    /// Batch evaluation on multiple inputs.
    #[inline]
    #[must_use]
    pub fn evaluate_batch(&self, xs: &[Vec<u8>]) -> Vec<bool> {
        xs.iter().map(|x| self.evaluate(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_clause() {
        let c = Clause::new(5, 100, 1);
        assert_eq!(c.n_features(), 5);
        assert_eq!(c.automata().len(), 10);
        assert!((c.weight() - 1.0).abs() < 0.001);
    }

    #[test]
    fn empty_clause_fires() {
        let c = Clause::new(3, 100, 1);
        assert!(c.evaluate(&[0, 1, 0]));
    }

    #[test]
    fn include_violation() {
        let mut c = Clause::new(3, 100, 1);
        for _ in 0..100 {
            c.automata_mut()[0].increment();
        }
        assert!(!c.evaluate(&[0, 0, 0]));
        assert!(c.evaluate(&[1, 0, 0]));
    }

    #[test]
    fn weighted_vote() {
        let mut c = Clause::new(2, 100, 1);
        c.weight = 0.5;
        assert!((c.vote(&[0, 0]) - 0.5).abs() < 0.001);
    }

    #[test]
    fn activation_tracking() {
        let mut c = Clause::new(2, 100, 1);
        c.evaluate_tracked(&[0, 0]);
        c.evaluate_tracked(&[1, 1]);
        assert_eq!(c.activations(), 2);
    }

    #[test]
    fn weight_update() {
        let mut c = Clause::new(2, 100, 1);
        c.correct = 8;
        c.incorrect = 2;
        c.update_weight(0.1, 0.1, 2.0);
        assert!(c.weight() > 1.0);
    }

    #[test]
    fn is_dead_check() {
        let mut c = Clause::new(2, 100, 1);
        c.weight = 0.05;
        assert!(c.is_dead(10, 0.1));
    }

    #[test]
    fn batch_evaluate() {
        let c = Clause::new(2, 100, 1);
        let xs = vec![vec![0, 0], vec![1, 1], vec![0, 1]];
        let results = c.evaluate_batch(&xs);
        assert_eq!(results.len(), 3);
    }
}
