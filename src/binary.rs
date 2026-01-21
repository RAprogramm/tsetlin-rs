//! Binary classification Tsetlin Machine.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::feedback::{type_i, type_ii};
use crate::training::{EarlyStopTracker, FitOptions, FitResult};
use crate::utils::rng_from_seed;
use crate::{Clause, Config, Rule};

/// # Overview
///
/// Binary classification Tsetlin Machine.
///
/// # Examples
///
/// ```
/// use tsetlin_rs::{Config, TsetlinMachine};
///
/// let config = Config::builder().clauses(20).features(2).build().unwrap();
///
/// let mut tm = TsetlinMachine::new(config, 15);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TsetlinMachine {
    clauses:   Vec<Clause>,
    config:    Config,
    threshold: i32,
    inv_2t:    f32
}

impl TsetlinMachine {
    /// # Overview
    ///
    /// Creates new machine. Half clauses +1 polarity, half -1.
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
            threshold,
            inv_2t: 1.0 / (2.0 * threshold as f32)
        }
    }

    /// # Overview
    ///
    /// Sum of clause votes for input x.
    #[inline]
    pub fn sum_votes(&self, x: &[u8]) -> i32 {
        self.clauses.iter().map(|c| c.vote(x)).sum()
    }

    /// # Overview
    ///
    /// Predicts class (0 or 1).
    #[inline(always)]
    pub fn predict(&self, x: &[u8]) -> u8 {
        if self.sum_votes(x) >= 0 { 1 } else { 0 }
    }

    /// # Overview
    ///
    /// Batch prediction for multiple samples.
    #[inline]
    pub fn predict_batch(&self, xs: &[Vec<u8>]) -> Vec<u8> {
        xs.iter().map(|x| self.predict(x)).collect()
    }

    /// # Overview
    ///
    /// Trains on single example.
    #[inline]
    pub fn train_one<R: Rng>(&mut self, x: &[u8], y: u8, rng: &mut R) {
        let sum = self.sum_votes(x).clamp(-self.threshold, self.threshold);
        let t = self.threshold as f32;
        let s = self.config.s;

        let prob = if y == 1 { (t - sum as f32) * self.inv_2t } else { (t + sum as f32) * self.inv_2t };

        for clause in &mut self.clauses {
            let fires = clause.evaluate(x);
            let p = clause.polarity();

            if y == 1 {
                if p == 1 && rng.random::<f32>() <= prob {
                    type_i(clause, x, fires, s, rng);
                } else if p == -1 && fires && rng.random::<f32>() <= prob {
                    type_ii(clause, x);
                }
            } else if p == -1 && rng.random::<f32>() <= prob {
                type_i(clause, x, fires, s, rng);
            } else if p == 1 && fires && rng.random::<f32>() <= prob {
                type_ii(clause, x);
            }
        }
    }

    /// # Overview
    ///
    /// Simple training for given epochs.
    pub fn fit(&mut self, x: &[Vec<u8>], y: &[u8], epochs: usize, seed: u64) {
        self.fit_with_options(x, y, FitOptions::new(epochs, seed));
    }

    /// # Overview
    ///
    /// Training with options and early stopping.
    pub fn fit_with_options(&mut self, x: &[Vec<u8>], y: &[u8], opts: FitOptions) -> FitResult {
        let mut rng = rng_from_seed(opts.seed);
        let mut indices: Vec<usize> = (0..x.len()).collect();
        let mut tracker = opts.early_stop.as_ref().map(EarlyStopTracker::new);
        let mut stopped = false;
        let mut epochs_run = 0;

        for epoch in 0..opts.epochs {
            if opts.shuffle {
                crate::utils::shuffle(&mut indices, &mut rng);
            }
            for &i in &indices {
                self.train_one(&x[i], y[i], &mut rng);
            }
            epochs_run = epoch + 1;

            if let Some(ref mut t) = tracker
                && t.update(self.evaluate(x, y))
            {
                stopped = true;
                break;
            }
        }

        FitResult::new(epochs_run, self.evaluate(x, y), stopped)
    }

    /// # Overview
    ///
    /// Evaluates accuracy on test data.
    #[inline]
    pub fn evaluate(&self, x: &[Vec<u8>], y: &[u8]) -> f32 {
        let correct = x.iter().zip(y).filter(|(xi, yi)| self.predict(xi) == **yi).count();
        correct as f32 / x.len() as f32
    }

    /// # Overview
    ///
    /// Extracts learned rules from all clauses.
    pub fn rules(&self) -> Vec<Rule> {
        self.clauses.iter().map(Rule::from_clause).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_convergence() {
        let config = Config::builder().clauses(20).features(2).build().unwrap();
        let mut tm = TsetlinMachine::new(config, 10);

        let x = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
        let y = vec![0, 1, 1, 0];

        tm.fit(&x, &y, 200, 42);
        assert!(tm.evaluate(&x, &y) >= 0.75);
    }

    #[test]
    fn batch_predict() {
        let config = Config::builder().clauses(10).features(2).build().unwrap();
        let tm = TsetlinMachine::new(config, 5);

        let xs = vec![vec![0, 0], vec![1, 1]];
        let preds = tm.predict_batch(&xs);
        assert_eq!(preds.len(), 2);
    }
}
