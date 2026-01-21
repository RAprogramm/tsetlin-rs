//! Rule extraction for interpretability.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Clause;

/// # Overview
///
/// A human-readable rule extracted from a clause.
///
/// Represents a conjunction: `(x[i1] AND x[i2] AND NOT x[j1] AND NOT x[j2])`.
///
/// # Examples
///
/// ```
/// use tsetlin_rs::{Clause, Rule};
///
/// let mut clause = Clause::new(4, 50, 1);
/// for _ in 0..100 {
///     clause.automata_mut()[0].increment();
///     clause.automata_mut()[5].increment();
/// }
///
/// let rule = Rule::from_clause(&clause);
/// assert!(rule.included.contains(&0));
/// assert!(rule.negated.contains(&2));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Rule {
    pub included: Vec<usize>,
    pub negated:  Vec<usize>,
    pub polarity: i8
}

impl Rule {
    /// # Overview
    ///
    /// Extracts rule from a trained clause.
    pub fn from_clause(clause: &Clause) -> Self {
        let mut included = Vec::new();
        let mut negated = Vec::new();
        let automata = clause.automata();

        for k in 0..clause.n_features() {
            if automata[2 * k].action() {
                included.push(k);
            }
            if automata[2 * k + 1].action() {
                negated.push(k);
            }
        }

        Self {
            included,
            negated,
            polarity: clause.polarity()
        }
    }

    /// # Overview
    ///
    /// Returns true if rule has no active literals (matches everything).
    pub fn is_empty(&self) -> bool {
        self.included.is_empty() && self.negated.is_empty()
    }

    /// # Overview
    ///
    /// Returns number of active literals.
    pub fn complexity(&self) -> usize {
        self.included.len() + self.negated.len()
    }
}

#[cfg(feature = "std")]
impl core::fmt::Display for Rule {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_empty() {
            return write!(f, "TRUE");
        }

        let mut parts = Vec::new();
        for &i in &self.included {
            parts.push(format!("x[{i}]"));
        }
        for &i in &self.negated {
            parts.push(format!("NOT x[{i}]"));
        }

        let sign = if self.polarity == 1 { "+" } else { "-" };
        write!(f, "{sign} ({})", parts.join(" AND "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_rule() {
        let clause = Clause::new(3, 100, 1);
        let rule = Rule::from_clause(&clause);

        assert!(rule.is_empty());
        assert_eq!(rule.complexity(), 0);
    }

    #[test]
    fn rule_with_literals() {
        let mut clause = Clause::new(4, 50, -1);
        for _ in 0..100 {
            clause.automata_mut()[0].increment();
            clause.automata_mut()[3].increment();
        }

        let rule = Rule::from_clause(&clause);

        assert_eq!(rule.included, vec![0]);
        assert_eq!(rule.negated, vec![1]);
        assert_eq!(rule.polarity, -1);
        assert_eq!(rule.complexity(), 2);
    }
}
