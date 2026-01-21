//! Clause - a conjunction of literals with cached actions.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Automaton;

/// # Overview
///
/// A clause with 2*n_features automata.
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
    n_features: usize
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
            n_features
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
    /// Returns polarity if fires, 0 otherwise.
    #[inline(always)]
    pub fn vote(&self, x: &[u8]) -> i32 {
        if self.evaluate(x) { self.polarity as i32 } else { 0 }
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
    fn batch_evaluate() {
        let c = Clause::new(2, 100, 1);
        let xs = vec![vec![0, 0], vec![1, 1], vec![0, 1]];
        let results = c.evaluate_batch(&xs);
        assert_eq!(results.len(), 3);
    }
}
