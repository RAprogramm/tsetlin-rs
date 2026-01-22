//! Structure of Arrays (SoA) clause storage for cache-efficient operations.
//!
//! This module provides [`ClauseBank`] - a cache-optimized alternative to
//! storing clauses as Array of Structures (AoS).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod eval;
mod feedback;

#[cfg(test)]
mod tests;

/// Structure of Arrays storage for Tsetlin Machine clauses.
///
/// Instead of `Vec<Clause>` where each clause has its own automata array,
/// this stores all data in contiguous arrays for better cache locality.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClauseBank {
    /// Automata states: flat array [n_clauses * 2 * n_features].
    pub(crate) states: Vec<i16>,

    /// Weight per clause.
    pub(crate) weights: Vec<f32>,

    /// Polarity per clause (+1 or -1).
    pub(crate) polarities: Vec<i8>,

    /// Activation count per clause.
    pub(crate) activations: Vec<u32>,

    /// Correct predictions per clause.
    pub(crate) correct: Vec<u32>,

    /// Incorrect predictions per clause.
    pub(crate) incorrect: Vec<u32>,

    /// Number of clauses.
    pub(crate) n_clauses: usize,

    /// Number of input features.
    pub(crate) n_features: usize,

    /// States per automaton (threshold).
    pub(crate) n_states: i16,

    /// Stride: 2 * n_features.
    pub(crate) stride: usize
}

impl ClauseBank {
    /// Creates a new clause bank.
    ///
    /// Half clauses get polarity +1, half -1 (alternating).
    #[must_use]
    pub fn new(n_clauses: usize, n_features: usize, n_states: i16) -> Self {
        debug_assert!(n_clauses > 0 && n_features > 0);

        let stride = 2 * n_features;
        let states = vec![n_states; n_clauses * stride];
        let polarities = (0..n_clauses)
            .map(|i| if i % 2 == 0 { 1 } else { -1 })
            .collect();

        Self {
            states,
            weights: vec![1.0; n_clauses],
            polarities,
            activations: vec![0; n_clauses],
            correct: vec![0; n_clauses],
            incorrect: vec![0; n_clauses],
            n_clauses,
            n_features,
            n_states,
            stride
        }
    }

    /// Returns number of clauses.
    #[inline(always)]
    #[must_use]
    pub const fn n_clauses(&self) -> usize {
        self.n_clauses
    }

    /// Returns number of features.
    #[inline(always)]
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns states per automaton.
    #[inline(always)]
    #[must_use]
    pub const fn n_states(&self) -> i16 {
        self.n_states
    }

    /// Returns all automata states.
    #[inline(always)]
    #[must_use]
    pub fn states(&self) -> &[i16] {
        &self.states
    }

    /// Returns all weights.
    #[inline(always)]
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Returns all polarities.
    #[inline(always)]
    #[must_use]
    pub fn polarities(&self) -> &[i8] {
        &self.polarities
    }

    /// Returns all activation counts.
    #[inline(always)]
    #[must_use]
    pub fn activations(&self) -> &[u32] {
        &self.activations
    }

    /// Returns clause weight.
    #[inline(always)]
    #[must_use]
    pub fn weight(&self, idx: usize) -> f32 {
        self.weights[idx]
    }

    /// Returns clause polarity.
    #[inline(always)]
    #[must_use]
    pub fn polarity(&self, idx: usize) -> i8 {
        self.polarities[idx]
    }

    /// Returns automata states for a clause.
    #[inline]
    #[must_use]
    pub fn clause_states(&self, idx: usize) -> &[i16] {
        let start = idx * self.stride;
        &self.states[start..start + self.stride]
    }

    /// Increments automaton state.
    #[inline(always)]
    pub fn increment(&mut self, clause: usize, automaton: usize) {
        let idx = clause * self.stride + automaton;
        let max = 2 * self.n_states;
        if self.states[idx] < max {
            self.states[idx] += 1;
        }
    }

    /// Decrements automaton state.
    #[inline(always)]
    pub fn decrement(&mut self, clause: usize, automaton: usize) {
        let idx = clause * self.stride + automaton;
        if self.states[idx] > 1 {
            self.states[idx] -= 1;
        }
    }
}
