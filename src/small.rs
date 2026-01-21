//! Small-sized Tsetlin Machine with const generics for compile-time optimization.
//!
//! Uses stack allocation and compile-time unrolling for small feature sets.

use core::array;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Automaton;

/// # Overview
///
/// A clause with compile-time known feature count.
///
/// Stack-allocated, no heap allocations. Optimal for N <= 32.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(align(64))]
pub struct SmallClause<const N: usize> {
    include:  [Automaton; N],
    negated:  [Automaton; N],
    polarity: i8
}

impl<const N: usize> SmallClause<N> {
    /// # Overview
    ///
    /// Creates clause with given states and polarity.
    #[inline]
    #[must_use]
    pub fn new(n_states: i16, polarity: i8) -> Self {
        debug_assert!(polarity == 1 || polarity == -1);
        Self {
            include:  array::from_fn(|_| Automaton::new(n_states)),
            negated:  array::from_fn(|_| Automaton::new(n_states)),
            polarity
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
        N
    }

    #[inline(always)]
    #[must_use]
    pub fn include_automata(&self) -> &[Automaton; N] {
        &self.include
    }

    #[inline(always)]
    pub fn include_automata_mut(&mut self) -> &mut [Automaton; N] {
        &mut self.include
    }

    #[inline(always)]
    #[must_use]
    pub fn negated_automata(&self) -> &[Automaton; N] {
        &self.negated
    }

    #[inline(always)]
    pub fn negated_automata_mut(&mut self) -> &mut [Automaton; N] {
        &mut self.negated
    }

    /// # Overview
    ///
    /// Evaluates clause on input. Fully unrolled at compile time for small N.
    #[inline]
    #[must_use]
    pub fn evaluate(&self, x: &[u8; N]) -> bool {
        for k in 0..N {
            // SAFETY: k is always in bounds [0, N)
            let include_action = unsafe { self.include.get_unchecked(k).action() };
            let negated_action = unsafe { self.negated.get_unchecked(k).action() };
            let xk = unsafe { *x.get_unchecked(k) };

            if include_action && xk == 0 {
                return false;
            }
            if negated_action && xk == 1 {
                return false;
            }
        }
        true
    }

    /// # Overview
    ///
    /// Returns polarity if fires, 0 otherwise.
    #[inline(always)]
    #[must_use]
    pub fn vote(&self, x: &[u8; N]) -> i32 {
        if self.evaluate(x) {
            self.polarity as i32
        } else {
            0
        }
    }
}

/// # Overview
///
/// Type alias for common small clause sizes.
pub type Clause2 = SmallClause<2>;
pub type Clause4 = SmallClause<4>;
pub type Clause8 = SmallClause<8>;
pub type Clause16 = SmallClause<16>;
pub type Clause32 = SmallClause<32>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_clause_new() {
        let c: SmallClause<4> = SmallClause::new(100, 1);
        assert_eq!(c.n_features(), 4);
        assert_eq!(c.polarity(), 1);
    }

    #[test]
    fn small_clause_evaluate() {
        let c: SmallClause<4> = SmallClause::new(100, 1);
        let x = [0, 1, 0, 1];
        assert!(c.evaluate(&x));
    }

    #[test]
    fn small_clause_vote() {
        let c: SmallClause<4> = SmallClause::new(100, -1);
        let x = [1, 1, 1, 1];
        assert_eq!(c.vote(&x), -1);
    }
}
