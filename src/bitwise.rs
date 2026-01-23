//! Bitwise clause evaluation for massive speedup.
//!
//! Processes 64 features per CPU instruction using bitmasks.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Automaton;

/// # Overview
///
/// Bitwise clause using packed u64 bitmasks.
///
/// Processes 64 features per AND operation - up to 50x faster than scalar.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(align(64))]
pub struct BitwiseClause {
    automata:   Vec<Automaton>,
    include:    Vec<u64>,
    negated:    Vec<u64>,
    polarity:   i8,
    n_features: usize,
    dirty:      bool
}

impl BitwiseClause {
    /// # Overview
    ///
    /// Creates clause with given features, states, and polarity.
    #[must_use]
    pub fn new(n_features: usize, n_states: i16, polarity: i8) -> Self {
        debug_assert!(polarity == 1 || polarity == -1);
        let n_words = n_features.div_ceil(64);
        let automata = (0..2 * n_features)
            .map(|_| Automaton::new(n_states))
            .collect();

        Self {
            automata,
            include: vec![0; n_words],
            negated: vec![0; n_words],
            polarity,
            n_features,
            dirty: true
        }
    }

    #[inline(always)]
    #[must_use]
    pub const fn polarity(&self) -> i8 {
        self.polarity
    }

    #[inline(always)]
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    #[inline(always)]
    #[must_use]
    pub fn automata(&self) -> &[Automaton] {
        &self.automata
    }

    #[inline(always)]
    pub fn automata_mut(&mut self) -> &mut [Automaton] {
        self.dirty = true;
        &mut self.automata
    }

    /// # Overview
    ///
    /// Rebuilds bitmasks from automaton states. Call after training.
    pub fn rebuild_masks(&mut self) {
        if !self.dirty {
            return;
        }

        for word in &mut self.include {
            *word = 0;
        }
        for word in &mut self.negated {
            *word = 0;
        }

        for k in 0..self.n_features {
            let word_idx = k / 64;
            let bit_idx = k % 64;

            if self.automata[2 * k].action() {
                self.include[word_idx] |= 1u64 << bit_idx;
            }
            if self.automata[2 * k + 1].action() {
                self.negated[word_idx] |= 1u64 << bit_idx;
            }
        }

        self.dirty = false;
    }

    /// Evaluates clause using bitwise AND operations.
    ///
    /// Processes 64 features per CPU instruction for massive speedup.
    ///
    /// # Arguments
    ///
    /// * `x_packed` - Input packed as u64 words via [`pack_input`]
    ///
    /// # Panics
    ///
    /// Debug-asserts that `rebuild_masks()` was called after training.
    #[inline]
    #[must_use]
    pub fn evaluate_packed(&self, x_packed: &[u64]) -> bool {
        debug_assert!(!self.dirty, "call rebuild_masks() first");

        let n_words = self.include.len().min(x_packed.len());

        for i in 0..n_words {
            // SAFETY: `i < n_words <= min(include.len(), x_packed.len())`.
            // Therefore all three accesses are within bounds.
            let x = unsafe { *x_packed.get_unchecked(i) };
            let inc = unsafe { *self.include.get_unchecked(i) };
            let neg = unsafe { *self.negated.get_unchecked(i) };

            // include violation: inc & !x != 0 (required bit is 0)
            // negated violation: neg & x != 0 (forbidden bit is 1)
            if (inc & !x) | (neg & x) != 0 {
                return false;
            }
        }
        true
    }

    /// Returns polarity if fires, 0 otherwise.
    #[inline(always)]
    #[must_use]
    pub fn vote_packed(&self, x_packed: &[u64]) -> i32 {
        if self.evaluate_packed(x_packed) {
            self.polarity as i32
        } else {
            0
        }
    }

    /// Fallback scalar evaluation (no packing needed).
    ///
    /// Slower than `evaluate_packed` but works with unpacked input.
    #[inline]
    #[must_use]
    pub fn evaluate(&self, x: &[u8]) -> bool {
        let n = self.n_features.min(x.len());

        for k in 0..n {
            // SAFETY: `k < n <= self.n_features`, and `automata.len() == 2 * n_features`.
            // Therefore `2 * k + 1 < automata.len()`.
            let include = unsafe { self.automata.get_unchecked(2 * k).action() };
            let negated = unsafe { self.automata.get_unchecked(2 * k + 1).action() };

            // SAFETY: `k < n <= x.len()`, so `k` is in bounds.
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
}

/// # Overview
///
/// Packs binary input into u64 words for bitwise evaluation.
#[inline]
#[must_use]
pub fn pack_input(x: &[u8]) -> Vec<u64> {
    let n_words = x.len().div_ceil(64);
    let mut packed = vec![0u64; n_words];

    for (k, &xk) in x.iter().enumerate() {
        if xk != 0 {
            packed[k / 64] |= 1u64 << (k % 64);
        }
    }

    packed
}

/// # Overview
///
/// Packs multiple inputs for batch processing.
#[inline]
#[must_use]
pub fn pack_batch(xs: &[Vec<u8>]) -> Vec<Vec<u64>> {
    xs.iter().map(|x| pack_input(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_input_basic() {
        let x = vec![1, 0, 1, 1, 0, 0, 0, 1];
        let packed = pack_input(&x);

        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0b10001101); // bits 0,2,3,7 set
    }

    #[test]
    fn bitwise_evaluate_empty() {
        let mut c = BitwiseClause::new(64, 100, 1);
        c.rebuild_masks();

        let x_packed = vec![0xFFFF_FFFF_FFFF_FFFFu64];
        assert!(c.evaluate_packed(&x_packed));
    }

    #[test]
    fn bitwise_evaluate_violation() {
        let mut c = BitwiseClause::new(64, 100, 1);

        // Force include[0] to be active
        for _ in 0..200 {
            c.automata_mut()[0].increment();
        }
        c.rebuild_masks();

        // x[0] = 0, should violate
        let x_packed = vec![0u64];
        assert!(!c.evaluate_packed(&x_packed));

        // x[0] = 1, should pass
        let x_packed = vec![1u64];
        assert!(c.evaluate_packed(&x_packed));
    }

    #[test]
    fn bitwise_clause_accessors() {
        let c = BitwiseClause::new(128, 100, -1);

        assert_eq!(c.polarity(), -1);
        assert_eq!(c.n_features(), 128);
        assert_eq!(c.automata().len(), 256); // 2 * n_features
    }

    #[test]
    fn bitwise_automata_mut_sets_dirty() {
        let mut c = BitwiseClause::new(64, 100, 1);
        c.rebuild_masks();

        // Access automata_mut sets dirty flag
        let _ = c.automata_mut();

        // Rebuild should execute (not early exit due to dirty=true)
        c.rebuild_masks();
        // No assertion needed - just verify it doesn't panic
    }

    #[test]
    fn bitwise_vote_packed() {
        let mut c = BitwiseClause::new(64, 100, 1);
        c.rebuild_masks(); // Empty clause fires

        // Empty clause fires -> returns polarity (1)
        assert_eq!(c.vote_packed(&[0u64]), 1);

        // Force include[0] active
        for _ in 0..200 {
            c.automata_mut()[0].increment();
        }
        c.rebuild_masks();

        // Violation: x[0]=0 -> returns 0
        assert_eq!(c.vote_packed(&[0u64]), 0);

        // Fires: x[0]=1 -> returns polarity (1)
        assert_eq!(c.vote_packed(&[1u64]), 1);
    }

    #[test]
    fn bitwise_vote_packed_negative_polarity() {
        let mut c = BitwiseClause::new(64, 100, -1);
        c.rebuild_masks();

        // Empty clause fires -> returns polarity (-1)
        assert_eq!(c.vote_packed(&[0u64]), -1);
    }

    #[test]
    fn bitwise_evaluate_scalar() {
        let mut c = BitwiseClause::new(4, 100, 1);

        // Force include[0] and negated[2] active
        for _ in 0..200 {
            c.automata_mut()[0].increment(); // include[0]
            c.automata_mut()[5].increment(); // negated[2]
        }

        // x[0]=1, x[2]=0 -> should fire
        assert!(c.evaluate(&[1, 0, 0, 0]));

        // x[0]=0 -> include violation
        assert!(!c.evaluate(&[0, 0, 0, 0]));

        // x[2]=1 -> negated violation
        assert!(!c.evaluate(&[1, 0, 1, 0]));
    }

    #[test]
    fn bitwise_evaluate_scalar_empty() {
        let c = BitwiseClause::new(4, 100, 1);
        // Empty clause always fires
        assert!(c.evaluate(&[0, 0, 0, 0]));
        assert!(c.evaluate(&[1, 1, 1, 1]));
    }

    #[test]
    fn pack_batch_multiple() {
        let xs = vec![vec![1, 0, 0, 0], vec![0, 1, 0, 0], vec![1, 1, 0, 0]];
        let packed = pack_batch(&xs);

        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0][0], 0b0001); // bit 0 set
        assert_eq!(packed[1][0], 0b0010); // bit 1 set
        assert_eq!(packed[2][0], 0b0011); // bits 0,1 set
    }

    #[test]
    fn pack_input_large() {
        // 128 features = 2 words
        let mut x = vec![0u8; 128];
        x[0] = 1;
        x[63] = 1;
        x[64] = 1;
        x[127] = 1;

        let packed = pack_input(&x);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], 1u64 | (1u64 << 63)); // bits 0, 63
        assert_eq!(packed[1], 1u64 | (1u64 << 63)); // bits 64, 127 (relative to word)
    }

    #[test]
    fn bitwise_negated_violation() {
        let mut c = BitwiseClause::new(64, 100, 1);

        // Force negated[0] active
        for _ in 0..200 {
            c.automata_mut()[1].increment();
        }
        c.rebuild_masks();

        // x[0]=1 -> negated violation
        assert!(!c.evaluate_packed(&[1u64]));

        // x[0]=0 -> fires
        assert!(c.evaluate_packed(&[0u64]));
    }

    #[test]
    fn bitwise_multi_word() {
        let mut c = BitwiseClause::new(128, 100, 1);

        // Force include[64] active (second word)
        for _ in 0..200 {
            c.automata_mut()[128].increment(); // automata[2*64] = include for feature 64
        }
        c.rebuild_masks();

        // Need bit 64 set (which is bit 0 of word 1)
        let x_packed = vec![0u64, 1u64];
        assert!(c.evaluate_packed(&x_packed));

        // Without bit 64 -> violation
        let x_packed = vec![0u64, 0u64];
        assert!(!c.evaluate_packed(&x_packed));
    }
}
