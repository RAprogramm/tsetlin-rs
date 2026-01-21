//! Tsetlin Automaton - the fundamental building block.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// # Overview
///
/// A single Tsetlin Automaton with states from 1 to 2*n_states.
/// States 1..n_states produce action=false (exclude).
/// States (n_states+1)..2*n_states produce action=true (include).
///
/// Uses i16 for memory efficiency (50% less than i32).
///
/// # Examples
///
/// ```
/// use tsetlin_rs::Automaton;
///
/// let mut automaton = Automaton::new(100);
/// assert!(!automaton.action());
///
/// for _ in 0..100 {
///     automaton.increment();
/// }
/// assert!(automaton.action());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Automaton {
    state:    i16,
    n_states: i16
}

impl Automaton {
    /// # Overview
    ///
    /// Creates automaton at threshold (just below include).
    #[inline]
    pub fn new(n_states: i16) -> Self {
        Self {
            state: n_states,
            n_states
        }
    }

    /// # Overview
    ///
    /// Creates automaton with specific initial state.
    #[inline]
    pub fn with_state(state: i16, n_states: i16) -> Self {
        debug_assert!(state >= 1 && state <= 2 * n_states);
        Self {
            state,
            n_states
        }
    }

    /// # Overview
    ///
    /// Returns true if state > n_states (include literal).
    #[inline]
    pub fn action(&self) -> bool {
        self.state > self.n_states
    }

    /// # Overview
    ///
    /// Returns current state value.
    #[inline]
    pub fn state(&self) -> i16 {
        self.state
    }

    /// # Overview
    ///
    /// Increments state. Capped at 2*n_states.
    #[inline]
    pub fn increment(&mut self) {
        if self.state < 2 * self.n_states {
            self.state += 1;
        }
    }

    /// # Overview
    ///
    /// Decrements state. Floored at 1.
    #[inline]
    pub fn decrement(&mut self) {
        if self.state > 1 {
            self.state -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_at_threshold() {
        let a = Automaton::new(100);
        assert_eq!(a.state(), 100);
        assert!(!a.action());
    }

    #[test]
    fn action_changes() {
        let mut a = Automaton::new(100);
        a.increment();
        assert!(a.action());
        a.decrement();
        assert!(!a.action());
    }

    #[test]
    fn respects_bounds() {
        let mut a = Automaton::new(3);
        for _ in 0..10 {
            a.decrement();
        }
        assert_eq!(a.state(), 1);
        for _ in 0..10 {
            a.increment();
        }
        assert_eq!(a.state(), 6);
    }
}
