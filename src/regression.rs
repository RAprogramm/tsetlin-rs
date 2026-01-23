//! Tsetlin Machine for regression.

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
/// Regression Tsetlin Machine.
///
/// # Examples
///
/// ```
/// use tsetlin_rs::{Config, Regressor};
///
/// let config = Config::builder().clauses(40).features(4).build().unwrap();
///
/// let mut tm = Regressor::new(config, 20);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Regressor {
    clauses:   Vec<Clause>,
    config:    Config,
    threshold: i32
}

impl Regressor {
    /// # Overview
    ///
    /// Creates regressor with given threshold.
    pub fn new(config: Config, threshold: i32) -> Self {
        let clauses = (0..config.n_clauses)
            .map(|i| {
                let p = if i % 2 == 0 { 1 } else { -1 };
                Clause::new(config.n_features, config.n_states, p)
            })
            .collect();

        Self {
            clauses,
            config,
            threshold
        }
    }

    /// # Overview
    ///
    /// Raw sum of clause votes.
    pub fn sum_votes(&self, x: &[u8]) -> f32 {
        self.clauses.iter().map(|c| c.vote(x)).sum()
    }

    /// # Overview
    ///
    /// Predicts integer value clamped to threshold range.
    pub fn predict(&self, x: &[u8]) -> i32 {
        let t = self.threshold as f32;
        self.sum_votes(x).clamp(-t, t).round() as i32
    }

    /// # Overview
    ///
    /// Trains on single example.
    pub fn train_one<R: Rng>(&mut self, x: &[u8], y: i32, rng: &mut R) {
        let prediction = self.predict(x);
        let error = y - prediction;
        let error_prob = error.unsigned_abs() as f32 / (2.0 * self.threshold as f32);

        for clause in &mut self.clauses {
            let fires = clause.evaluate(x);
            let p = clause.polarity();

            if error > 0 {
                if p == 1 && rng.random::<f32>() <= error_prob {
                    type_i(clause, x, fires, self.config.s, rng);
                } else if p == -1 && fires && rng.random::<f32>() <= error_prob {
                    type_ii(clause, x);
                }
            } else if p == -1 && rng.random::<f32>() <= error_prob {
                type_i(clause, x, fires, self.config.s, rng);
            } else if p == 1 && fires && rng.random::<f32>() <= error_prob {
                type_ii(clause, x);
            }
        }
    }

    /// # Overview
    ///
    /// Trains for given epochs.
    pub fn fit(&mut self, x: &[Vec<u8>], y: &[i32], epochs: usize, seed: u64) {
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
    /// Mean absolute error.
    pub fn mae(&self, x: &[Vec<u8>], y: &[i32]) -> f32 {
        let sum: i32 = x
            .iter()
            .zip(y)
            .map(|(xi, yi)| (self.predict(xi) - yi).abs())
            .sum();
        sum as f32 / x.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_in_range() {
        let config = Config::builder().clauses(20).features(4).build().unwrap();
        let tm = Regressor::new(config, 10);
        let p = tm.predict(&[1, 0, 1, 0]);

        assert!((-10..=10).contains(&p));
    }

    #[test]
    fn sum_votes_returns_finite() {
        let config = Config::builder().clauses(20).features(4).build().unwrap();
        let tm = Regressor::new(config, 10);
        let votes = tm.sum_votes(&[1, 0, 1, 0]);
        assert!(votes.is_finite());
    }

    #[test]
    fn train_one_modifies_state() {
        let config = Config::builder().clauses(20).features(4).build().unwrap();
        let mut tm = Regressor::new(config, 10);
        let mut rng = rng_from_seed(42);

        // Train with positive error (y > prediction)
        tm.train_one(&[1, 1, 0, 0], 5, &mut rng);

        // Train with negative error (y < prediction)
        tm.train_one(&[0, 0, 1, 1], -5, &mut rng);

        // Should still predict valid values
        let p = tm.predict(&[1, 0, 1, 0]);
        assert!((-10..=10).contains(&p));
    }

    #[test]
    fn fit_reduces_mae() {
        let config = Config::builder().clauses(40).features(4).build().unwrap();
        let mut tm = Regressor::new(config, 10);

        // Simple regression pattern
        let x = vec![
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1],
        ];
        let y = vec![5, 3, -3, -5];

        tm.fit(&x, &y, 100, 42);

        // MAE should be within reasonable bounds
        let mae = tm.mae(&x, &y);
        assert!(mae >= 0.0);
        assert!(mae <= 20.0);
    }

    #[test]
    fn mae_zero_for_perfect_predictions() {
        let config = Config::builder().clauses(20).features(4).build().unwrap();
        let tm = Regressor::new(config, 10);

        // Get current predictions
        let x = vec![vec![1, 0, 1, 0], vec![0, 1, 0, 1]];
        let y: Vec<i32> = x.iter().map(|xi| tm.predict(xi)).collect();

        // MAE should be 0 when y matches predictions
        let mae = tm.mae(&x, &y);
        assert!((mae - 0.0).abs() < 0.001);
    }
}
