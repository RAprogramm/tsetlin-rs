//! Multi-class classification Tsetlin Machine.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    Clause, Config,
    feedback::{type_i, type_ii},
    utils::rng_from_seed
};

/// # Overview
///
/// Multi-class Tsetlin Machine using one-vs-all strategy.
///
/// # Examples
///
/// ```
/// use tsetlin_rs::{Config, MultiClass};
///
/// let config = Config::builder().clauses(100).features(4).build().unwrap();
///
/// let mut tm = MultiClass::new(config, 3, 50);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MultiClass {
    clauses:   Vec<Vec<Clause>>,
    config:    Config,
    threshold: i32
}

impl MultiClass {
    /// # Overview
    ///
    /// Creates multi-class machine with n_classes.
    pub fn new(config: Config, n_classes: usize, threshold: i32) -> Self {
        let clauses = (0..n_classes)
            .map(|_| {
                (0..config.n_clauses)
                    .map(|i| {
                        let p = if i % 2 == 0 { 1 } else { -1 };
                        Clause::new(config.n_features, config.n_states, p)
                    })
                    .collect()
            })
            .collect();

        Self {
            clauses,
            config,
            threshold
        }
    }

    #[inline]
    pub fn n_classes(&self) -> usize {
        self.clauses.len()
    }

    /// # Overview
    ///
    /// Vote sums per class.
    pub fn class_votes(&self, x: &[u8]) -> Vec<i32> {
        self.clauses
            .iter()
            .map(|cls| cls.iter().map(|c| c.vote(x)).sum())
            .collect()
    }

    /// # Overview
    ///
    /// Predicts class with highest vote.
    pub fn predict(&self, x: &[u8]) -> usize {
        self.class_votes(x)
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// # Overview
    ///
    /// Trains on single example.
    pub fn train_one<R: Rng>(&mut self, x: &[u8], y: usize, rng: &mut R) {
        let votes = self.class_votes(x);
        let t = self.threshold as f32;

        for (class_idx, class_clauses) in self.clauses.iter_mut().enumerate() {
            let is_target = class_idx == y;
            let sum = votes[class_idx].clamp(-self.threshold, self.threshold) as f32;

            for clause in class_clauses {
                let fires = clause.evaluate(x);
                let p = clause.polarity();

                if is_target {
                    let prob = (t - sum) / (2.0 * t);
                    if p == 1 && rng.random::<f32>() <= prob {
                        type_i(clause, x, fires, self.config.s, rng);
                    } else if p == -1 && fires && rng.random::<f32>() <= prob {
                        type_ii(clause, x);
                    }
                } else {
                    let prob = (t + sum) / (2.0 * t);
                    if p == -1 && rng.random::<f32>() <= prob {
                        type_i(clause, x, fires, self.config.s, rng);
                    } else if p == 1 && fires && rng.random::<f32>() <= prob {
                        type_ii(clause, x);
                    }
                }
            }
        }
    }

    /// # Overview
    ///
    /// Trains for given epochs.
    pub fn fit(&mut self, x: &[Vec<u8>], y: &[usize], epochs: usize, seed: u64) {
        let mut rng = rng_from_seed(seed);
        let mut indices: Vec<usize> = (0..x.len()).collect();

        for _ in 0..epochs {
            crate::utils::shuffle(&mut indices, &mut rng);
            for &i in &indices {
                self.train_one(&x[i], y[i], &mut rng);
            }
        }
    }

    /// # Overview
    ///
    /// Evaluates accuracy.
    pub fn evaluate(&self, x: &[Vec<u8>], y: &[usize]) -> f32 {
        let correct = x
            .iter()
            .zip(y)
            .filter(|(xi, yi)| self.predict(xi) == **yi)
            .count();
        correct as f32 / x.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_valid_class() {
        let config = Config::builder().clauses(10).features(4).build().unwrap();
        let tm = MultiClass::new(config, 3, 15);

        assert!(tm.predict(&[1, 0, 1, 0]) < 3);
    }
}
