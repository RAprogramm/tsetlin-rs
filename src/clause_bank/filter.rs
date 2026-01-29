//! Bloom filter for fast clause rejection during inference.
//!
//! This module provides [`ClauseFilter`] which pre-computes signatures for each
//! clause to enable fast rejection of clauses that definitely won't fire.
//!
//! # Algorithm
//!
//! Each clause has two bitmasks:
//! - `include_mask`: bits set for features where `x_k` must be 1
//! - `negated_mask`: bits set for features where `x_k` must be 0
//!
//! For a given input `x`, we compute:
//! - `input_mask`: bits set where `x[k] == 1`
//!
//! A clause can fire only if:
//! - `(include_mask & input_mask) == include_mask` (all required 1s present)
//! - `(negated_mask & input_mask) == 0` (no forbidden 1s present)
//!
//! # Performance
//!
//! Provides 20-50% speedup on converged models where many clauses have
//! sparse literal patterns. Best for inference on trained models.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use super::ClauseBank;

/// Bloom filter for fast clause rejection.
///
/// Pre-computes per-clause bitmasks to quickly determine if a clause
/// definitely won't fire for a given input, avoiding full evaluation.
///
/// # Example
///
/// ```
/// use tsetlin_rs::{ClauseBank, ClauseFilter};
///
/// let bank = ClauseBank::new(100, 64, 100);
/// let filter = ClauseFilter::from_bank(&bank);
///
/// let input = vec![1u8; 64];
/// let candidates = filter.candidates(&input);
/// // Only evaluate clauses that might fire
/// ```
#[derive(Debug, Clone)]
pub struct ClauseFilter {
    /// Bitmask of features that must be 1 for clause to fire.
    /// One entry per clause.
    include_masks: Vec<Vec<u64>>,
    /// Bitmask of features that must be 0 for clause to fire.
    /// One entry per clause.
    negated_masks: Vec<Vec<u64>>,
    /// Number of 64-bit words needed per feature mask.
    mask_words:    usize,
    /// Number of clauses.
    n_clauses:     usize,
    /// Number of features.
    n_features:    usize
}

impl ClauseFilter {
    /// Builds a filter from a trained clause bank.
    ///
    /// Extracts included/negated literal patterns from each clause's
    /// automata states. Call after training is complete.
    #[must_use]
    pub fn from_bank(bank: &ClauseBank) -> Self {
        let n_clauses = bank.n_clauses();
        let n_features = bank.n_features();
        let mask_words = n_features.div_ceil(64);
        let threshold = bank.n_states();

        let mut include_masks = Vec::with_capacity(n_clauses);
        let mut negated_masks = Vec::with_capacity(n_clauses);

        for clause_idx in 0..n_clauses {
            let states = bank.clause_states(clause_idx);
            let mut include = vec![0u64; mask_words];
            let mut negated = vec![0u64; mask_words];

            for k in 0..n_features {
                let word = k / 64;
                let bit = k % 64;

                // State > threshold means literal is included
                if states[2 * k] > threshold {
                    include[word] |= 1u64 << bit;
                }
                if states[2 * k + 1] > threshold {
                    negated[word] |= 1u64 << bit;
                }
            }

            include_masks.push(include);
            negated_masks.push(negated);
        }

        Self {
            include_masks,
            negated_masks,
            mask_words,
            n_clauses,
            n_features
        }
    }

    /// Computes the input bitmask from binary features.
    #[inline]
    fn compute_input_mask(&self, x: &[u8]) -> Vec<u64> {
        let mut mask = vec![0u64; self.mask_words];
        let n = x.len().min(self.n_features);

        for (k, &xk) in x.iter().enumerate().take(n) {
            if xk == 1 {
                let word = k / 64;
                let bit = k % 64;
                mask[word] |= 1u64 << bit;
            }
        }
        mask
    }

    /// Tests if a clause might fire for the given input.
    ///
    /// Returns `false` if the clause definitely won't fire (no false
    /// negatives). Returns `true` if the clause might fire (may have false
    /// positives).
    #[inline]
    #[must_use]
    pub fn might_fire(&self, clause_idx: usize, input_mask: &[u64]) -> bool {
        let include = &self.include_masks[clause_idx];
        let negated = &self.negated_masks[clause_idx];

        for w in 0..self.mask_words {
            // All required 1s must be present
            if (include[w] & input_mask[w]) != include[w] {
                return false;
            }
            // No forbidden 1s may be present
            if (negated[w] & input_mask[w]) != 0 {
                return false;
            }
        }
        true
    }

    /// Returns indices of clauses that might fire for the input.
    ///
    /// Use this to filter which clauses need full evaluation.
    #[must_use]
    pub fn candidates(&self, x: &[u8]) -> Vec<usize> {
        let input_mask = self.compute_input_mask(x);
        (0..self.n_clauses)
            .filter(|&i| self.might_fire(i, &input_mask))
            .collect()
    }

    /// Computes the sum of weighted votes using filtered evaluation.
    ///
    /// Only evaluates clauses that pass the filter check, providing
    /// speedup when many clauses have sparse literal patterns.
    #[must_use]
    pub fn sum_votes_filtered(&self, bank: &ClauseBank, x: &[u8]) -> f32 {
        let input_mask = self.compute_input_mask(x);
        let mut sum = 0.0f32;

        for i in 0..self.n_clauses {
            if self.might_fire(i, &input_mask) && bank.evaluate_clause(i, x) {
                sum += bank.polarities()[i] as f32 * bank.weights()[i];
            }
        }
        sum
    }

    /// Returns the number of clauses in the filter.
    #[inline]
    #[must_use]
    pub fn n_clauses(&self) -> usize {
        self.n_clauses
    }

    /// Returns the number of features in the filter.
    #[inline]
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns filter statistics for diagnostics.
    #[must_use]
    pub fn stats(&self) -> ClauseFilterStats {
        let mut total_include = 0usize;
        let mut total_negated = 0usize;

        for i in 0..self.n_clauses {
            for w in 0..self.mask_words {
                total_include += self.include_masks[i][w].count_ones() as usize;
                total_negated += self.negated_masks[i][w].count_ones() as usize;
            }
        }

        ClauseFilterStats {
            n_clauses:            self.n_clauses,
            n_features:           self.n_features,
            avg_include_literals: total_include as f32 / self.n_clauses as f32,
            avg_negated_literals: total_negated as f32 / self.n_clauses as f32
        }
    }
}

/// Statistics about a clause filter.
#[derive(Debug, Clone, Copy)]
pub struct ClauseFilterStats {
    /// Number of clauses.
    pub n_clauses:            usize,
    /// Number of features.
    pub n_features:           usize,
    /// Average number of included positive literals per clause.
    pub avg_include_literals: f32,
    /// Average number of included negated literals per clause.
    pub avg_negated_literals: f32
}

impl ClauseFilterStats {
    /// Estimates the filter hit rate (fraction of clauses that pass filter).
    ///
    /// Lower values indicate better filtering (more clauses rejected).
    #[must_use]
    pub fn estimated_hit_rate(&self) -> f32 {
        // Rough estimate: each literal halves the probability of match
        let literals = self.avg_include_literals + self.avg_negated_literals;
        if literals <= 0.0 {
            1.0
        } else {
            0.5f32.powf(literals.min(10.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_creation() {
        let bank = ClauseBank::new(10, 64, 100);
        let filter = ClauseFilter::from_bank(&bank);

        assert_eq!(filter.n_clauses(), 10);
        assert_eq!(filter.n_features(), 64);
    }

    #[test]
    fn filter_fresh_clauses_all_pass() {
        // Fresh clauses have no included literals, all should pass
        let bank = ClauseBank::new(10, 8, 100);
        let filter = ClauseFilter::from_bank(&bank);

        let candidates = filter.candidates(&[1, 0, 1, 0, 1, 0, 1, 0]);
        assert_eq!(candidates.len(), 10);
    }

    #[test]
    fn filter_might_fire_basic() {
        let bank = ClauseBank::new(4, 4, 100);
        let filter = ClauseFilter::from_bank(&bank);

        let input_mask = filter.compute_input_mask(&[1, 0, 1, 0]);

        // All fresh clauses should pass
        for i in 0..4 {
            assert!(filter.might_fire(i, &input_mask));
        }
    }

    #[test]
    fn filter_sum_votes_matches_unfiltered() {
        let bank = ClauseBank::new(20, 8, 100);
        let filter = ClauseFilter::from_bank(&bank);

        let x = vec![1, 0, 1, 1, 0, 0, 1, 0];
        let unfiltered = bank.sum_votes(&x);
        let filtered = filter.sum_votes_filtered(&bank, &x);

        assert!((unfiltered - filtered).abs() < 0.001);
    }

    #[test]
    fn filter_stats() {
        let bank = ClauseBank::new(10, 32, 100);
        let filter = ClauseFilter::from_bank(&bank);
        let stats = filter.stats();

        assert_eq!(stats.n_clauses, 10);
        assert_eq!(stats.n_features, 32);
        // Fresh clauses have no included literals
        assert!(stats.avg_include_literals < 0.001);
        assert!(stats.avg_negated_literals < 0.001);
    }

    #[test]
    fn filter_stats_hit_rate() {
        let bank = ClauseBank::new(10, 8, 100);
        let filter = ClauseFilter::from_bank(&bank);
        let stats = filter.stats();

        // Fresh clauses: all pass, hit rate = 1.0
        assert!((stats.estimated_hit_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn filter_empty_input() {
        let bank = ClauseBank::new(4, 8, 100);
        let filter = ClauseFilter::from_bank(&bank);

        let candidates = filter.candidates(&[]);
        // Empty input still matches fresh clauses
        assert_eq!(candidates.len(), 4);
    }

    #[test]
    fn filter_large_features() {
        let bank = ClauseBank::new(10, 128, 100);
        let filter = ClauseFilter::from_bank(&bank);

        assert_eq!(filter.mask_words, 2); // 128 features = 2 words

        let x = vec![1u8; 128];
        let candidates = filter.candidates(&x);
        assert_eq!(candidates.len(), 10);
    }
}
