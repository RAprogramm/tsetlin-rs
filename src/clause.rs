//! Clause - a conjunction of literals with weighted voting and activation tracking.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Automaton;

/// # Overview
///
/// A clause with 2*n_features automata, weight, and activation tracking.
/// Cache-aligned for better performance.
///
/// - automata[2*k] controls literal x_k
/// - automata[2*k+1] controls literal NOT x_k
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(align(64))]
pub struct Clause {
    automata:   Vec<Automaton>,
    polarity:   i8,
    n_features: usize,
    weight:     f32,
    activations: u32,
    correct:    u32,
    incorrect:  u32
}

impl Clause {
    /// # Overview
    ///
    /// Creates clause with given features, states, and polarity.
    pub fn new(n_features: usize, n_states: i16, polarity: i8) -> Self {
        debug_assert!(polarity == 1 || polarity == -1);
        let automata = (0..2 * n_features).map(|_| Automaton::new(n_states)).collect();
        Self {
            automata,
            polarity,
            n_features,
            weight: 1.0,
            activations: 0,
            correct: 0,
            incorrect: 0
        }
    }

    #[inline(always)]
    pub fn polarity(&self) -> i8 {
        self.polarity
    }

    #[inline(always)]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    #[inline(always)]
    pub fn weight(&self) -> f32 {
        self.weight
    }

    #[inline(always)]
    pub fn activations(&self) -> u32 {
        self.activations
    }

    #[inline(always)]
    pub fn automata(&self) -> &[Automaton] {
        &self.automata
    }

    #[inline(always)]
    pub fn automata_mut(&mut self) -> &mut [Automaton] {
        &mut self.automata
    }

    /// # Overview
    ///
    /// Evaluates clause on binary input. Early exit on violation.
    #[inline]
    pub fn evaluate(&self, x: &[u8]) -> bool {
        let automata = &self.automata;
        let n = self.n_features.min(x.len());

        for k in 0..n {
            let include = unsafe { automata.get_unchecked(2 * k).action() };
            let negated = unsafe { automata.get_unchecked(2 * k + 1).action() };
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

    /// # Overview
    ///
    /// Evaluates and tracks activation. Use during training.
    #[inline]
    pub fn evaluate_tracked(&mut self, x: &[u8]) -> bool {
        let fires = self.evaluate(x);
        if fires {
            self.activations = self.activations.saturating_add(1);
        }
        fires
    }

    /// # Overview
    ///
    /// Returns weighted vote: polarity * weight if fires, 0 otherwise.
    #[inline(always)]
    pub fn vote(&self, x: &[u8]) -> f32 {
        if self.evaluate(x) {
            self.polarity as f32 * self.weight
        } else {
            0.0
        }
    }

    /// # Overview
    ///
    /// Returns unweighted vote (original behavior).
    #[inline(always)]
    pub fn vote_unweighted(&self, x: &[u8]) -> i32 {
        if self.evaluate(x) { self.polarity as i32 } else { 0 }
    }

    /// # Overview
    ///
    /// Records outcome when clause fired. Updates correct/incorrect counters.
    #[inline]
    pub fn record_outcome(&mut self, was_correct: bool) {
        if was_correct {
            self.correct = self.correct.saturating_add(1);
        } else {
            self.incorrect = self.incorrect.saturating_add(1);
        }
    }

    /// # Overview
    ///
    /// Updates weight based on accumulated outcomes.
    /// Call periodically (e.g., end of epoch).
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

    /// # Overview
    ///
    /// Returns true if clause is "dead" (never activates or very low weight).
    #[inline]
    pub fn is_dead(&self, min_activations: u32, min_weight: f32) -> bool {
        self.activations < min_activations || self.weight < min_weight
    }

    /// # Overview
    ///
    /// Resets activation counter. Call at start of epoch.
    #[inline]
    pub fn reset_activations(&mut self) {
        self.activations = 0;
    }

    /// # Overview
    ///
    /// Resets all stats (activations, correct, incorrect).
    #[inline]
    pub fn reset_stats(&mut self) {
        self.activations = 0;
        self.correct = 0;
        self.incorrect = 0;
    }

    /// # Overview
    ///
    /// Batch evaluation on multiple inputs.
    #[inline]
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
