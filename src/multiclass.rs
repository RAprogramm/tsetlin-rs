//! Multi-class classification Tsetlin Machine.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::cmp::Ordering;

use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    Clause, Config,
    feedback::{type_i, type_ii},
    utils::rng_from_seed
};

/// Multi-class Tsetlin Machine using one-vs-all strategy.
///
/// Each class has its own set of clauses. Prediction is the class
/// with the highest vote sum.
///
/// # Example
///
/// ```
/// use tsetlin_rs::{Config, MultiClass};
///
/// let config = Config::builder().clauses(100).features(4).build().unwrap();
/// let mut tm = MultiClass::new(config, 3, 50);
///
/// // Train on data where label is class index (0, 1, or 2)
/// let x = vec![vec![1, 1, 0, 0], vec![0, 0, 1, 1]];
/// let y = vec![0, 1];
/// tm.fit(&x, &y, 100, 42);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MultiClass {
    clauses:   Vec<Vec<Clause>>,
    config:    Config,
    threshold: i32
}

impl MultiClass {
    /// Creates multi-class machine with given number of classes.
    ///
    /// # Arguments
    ///
    /// * `config` - Machine configuration (clauses, features, etc.)
    /// * `n_classes` - Number of output classes
    /// * `threshold` - Vote threshold for training
    #[must_use]
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

    /// Returns the number of classes.
    #[inline]
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.clauses.len()
    }

    /// Computes vote sums for each class.
    #[must_use]
    pub fn class_votes(&self, x: &[u8]) -> Vec<f32> {
        self.clauses
            .iter()
            .map(|cls| cls.iter().map(|c| c.vote(x)).sum())
            .collect()
    }

    /// Predicts class with highest vote sum.
    ///
    /// Returns 0 if votes are empty or contain NaN.
    #[must_use]
    pub fn predict(&self, x: &[u8]) -> usize {
        self.class_votes(x)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map_or(0, |(i, _)| i)
    }

    /// Trains on a single example.
    pub fn train_one<R: Rng>(&mut self, x: &[u8], y: usize, rng: &mut R) {
        let votes = self.class_votes(x);
        let t = self.threshold as f32;

        for (class_idx, class_clauses) in self.clauses.iter_mut().enumerate() {
            let is_target = class_idx == y;
            let sum = votes[class_idx].clamp(-t, t);

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

    /// Trains for given number of epochs.
    ///
    /// # Arguments
    ///
    /// * `x` - Training inputs (binary features)
    /// * `y` - Class labels (0 to n_classes-1)
    /// * `epochs` - Number of training epochs
    /// * `seed` - Random seed for reproducibility
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

    /// Evaluates classification accuracy on test data.
    ///
    /// Returns fraction of correct predictions (0.0 to 1.0).
    #[must_use]
    pub fn evaluate(&self, x: &[Vec<u8>], y: &[usize]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        let correct = x
            .iter()
            .zip(y)
            .filter(|(xi, yi)| self.predict(xi) == **yi)
            .count();
        correct as f32 / x.len() as f32
    }

    /// Quick constructor with sensible defaults.
    ///
    /// # Panics
    ///
    /// Panics if n_clauses is odd or zero, or n_features is zero.
    #[must_use]
    pub fn quick(n_clauses: usize, n_features: usize, n_classes: usize, threshold: i32) -> Self {
        let config = Config::builder()
            .clauses(n_clauses)
            .features(n_features)
            .build()
            .expect("invalid quick config");
        Self::new(config, n_classes, threshold)
    }
}

impl crate::model::TsetlinModel<Vec<u8>, usize> for MultiClass {
    fn fit(&mut self, x: &[Vec<u8>], y: &[usize], epochs: usize, seed: u64) {
        MultiClass::fit(self, x, y, epochs, seed);
    }

    fn predict(&self, x: &Vec<u8>) -> usize {
        MultiClass::predict(self, x)
    }

    fn evaluate(&self, x: &[Vec<u8>], y: &[usize]) -> f32 {
        MultiClass::evaluate(self, x, y)
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
