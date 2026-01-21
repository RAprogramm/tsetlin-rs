//! Binary classification Tsetlin Machine with weighted clauses and adaptive
//! threshold.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    Clause, Config, Rule,
    feedback::{type_i, type_ii},
    training::{EarlyStopTracker, FitOptions, FitResult},
    utils::rng_from_seed
};

/// # Overview
///
/// Configuration for advanced training features.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdvancedOptions {
    pub adaptive_t:      bool,
    pub t_min:           f32,
    pub t_max:           f32,
    pub t_lr:            f32,
    pub weight_lr:       f32,
    pub weight_min:      f32,
    pub weight_max:      f32,
    pub prune_threshold: u32,
    pub prune_weight:    f32
}

impl Default for AdvancedOptions {
    fn default() -> Self {
        Self {
            adaptive_t:      false,
            t_min:           5.0,
            t_max:           50.0,
            t_lr:            0.1,
            weight_lr:       0.05,
            weight_min:      0.1,
            weight_max:      2.0,
            prune_threshold: 0,
            prune_weight:    0.0
        }
    }
}

/// # Overview
///
/// Binary classification Tsetlin Machine with weighted clauses and adaptive
/// threshold.
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
    clauses:  Vec<Clause>,
    config:   Config,
    t:        f32,
    t_base:   f32,
    advanced: AdvancedOptions
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

        let t = threshold as f32;
        Self {
            clauses,
            config,
            t,
            t_base: t,
            advanced: AdvancedOptions::default()
        }
    }

    /// # Overview
    ///
    /// Creates machine with advanced options.
    pub fn with_advanced(config: Config, threshold: i32, advanced: AdvancedOptions) -> Self {
        let mut tm = Self::new(config, threshold);
        tm.advanced = advanced;
        tm
    }

    /// # Overview
    ///
    /// Current threshold value (may differ from base if adaptive).
    #[inline]
    pub fn threshold(&self) -> f32 {
        self.t
    }

    /// # Overview
    ///
    /// Base threshold (initial value).
    #[inline]
    pub fn threshold_base(&self) -> f32 {
        self.t_base
    }

    /// # Overview
    ///
    /// Resets threshold to base value.
    pub fn reset_threshold(&mut self) {
        self.t = self.t_base;
    }

    /// # Overview
    ///
    /// Sum of weighted clause votes for input x.
    #[inline]
    pub fn sum_votes(&self, x: &[u8]) -> f32 {
        self.clauses.iter().map(|c| c.vote(x)).sum()
    }

    /// # Overview
    ///
    /// Predicts class (0 or 1).
    #[inline(always)]
    pub fn predict(&self, x: &[u8]) -> u8 {
        if self.sum_votes(x) >= 0.0 { 1 } else { 0 }
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
    /// Trains on single example with tracking.
    #[inline]
    pub fn train_one<R: Rng>(&mut self, x: &[u8], y: u8, rng: &mut R) {
        let sum = self.sum_votes(x).clamp(-self.t, self.t);
        let inv_2t = 1.0 / (2.0 * self.t);
        let s = self.config.s;

        let prob = if y == 1 {
            (self.t - sum) * inv_2t
        } else {
            (self.t + sum) * inv_2t
        };

        let prediction = if sum >= 0.0 { 1 } else { 0 };
        let correct = prediction == y;

        for clause in &mut self.clauses {
            let fires = clause.evaluate_tracked(x);
            let p = clause.polarity();

            // Record outcome for weight learning
            if fires {
                let clause_correct = (p == 1 && y == 1) || (p == -1 && y == 0);
                clause.record_outcome(clause_correct);
            }

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

        // Adaptive threshold adjustment
        if self.advanced.adaptive_t {
            let adj = if correct {
                self.advanced.t_lr
            } else {
                -self.advanced.t_lr
            };
            self.t = (self.t + adj).clamp(self.advanced.t_min, self.advanced.t_max);
        }
    }

    /// # Overview
    ///
    /// Updates clause weights. Call at end of epoch.
    pub fn update_weights(&mut self) {
        let lr = self.advanced.weight_lr;
        let min = self.advanced.weight_min;
        let max = self.advanced.weight_max;

        for clause in &mut self.clauses {
            clause.update_weight(lr, min, max);
        }
    }

    /// # Overview
    ///
    /// Prunes dead clauses (low activation or weight).
    pub fn prune_dead_clauses(&mut self) {
        let min_act = self.advanced.prune_threshold;
        let min_wt = self.advanced.prune_weight;

        if min_act == 0 && min_wt == 0.0 {
            return;
        }

        for clause in &mut self.clauses {
            if clause.is_dead(min_act, min_wt) {
                // Reset dead clause to fresh state
                *clause = Clause::new(
                    self.config.n_features,
                    self.config.n_states,
                    clause.polarity()
                );
            }
        }
    }

    /// # Overview
    ///
    /// Resets activation counters. Call at start of epoch.
    pub fn reset_activations(&mut self) {
        for clause in &mut self.clauses {
            clause.reset_activations();
        }
    }

    /// Simple training for given epochs.
    ///
    /// # Arguments
    ///
    /// * `x` - Training inputs (binary features)
    /// * `y` - Binary labels (0 or 1)
    /// * `epochs` - Number of training epochs
    /// * `seed` - Random seed for reproducibility
    pub fn fit(&mut self, x: &[Vec<u8>], y: &[u8], epochs: usize, seed: u64) {
        let _ = self.fit_with_options(x, y, FitOptions::new(epochs, seed));
    }

    /// Training with full options including early stopping and callbacks.
    ///
    /// # Arguments
    ///
    /// * `x` - Training inputs (binary features)
    /// * `y` - Binary labels (0 or 1)
    /// * `opts` - Training options (epochs, early stopping, callback)
    ///
    /// # Returns
    ///
    /// [`FitResult`] with training statistics.
    pub fn fit_with_options(
        &mut self,
        x: &[Vec<u8>],
        y: &[u8],
        mut opts: FitOptions
    ) -> FitResult {
        if x.is_empty() || x.len() != y.len() {
            return FitResult::new(0, 0.0, false);
        }

        let mut rng = rng_from_seed(opts.seed);
        let mut indices: Vec<usize> = (0..x.len()).collect();
        let mut tracker = opts.early_stop.as_ref().map(EarlyStopTracker::new);
        let mut stopped = false;
        let mut epochs_run = 0;
        let mut history = Vec::with_capacity(opts.epochs);

        for epoch in 0..opts.epochs {
            self.reset_activations();

            if opts.shuffle {
                crate::utils::shuffle(&mut indices, &mut rng);
            }

            for &i in &indices {
                self.train_one(&x[i], y[i], &mut rng);
            }

            // End of epoch: update weights and prune
            self.update_weights();
            self.prune_dead_clauses();

            epochs_run = epoch + 1;
            let accuracy = self.evaluate(x, y);
            history.push(accuracy);

            // Callback
            if let Some(ref mut callback) = opts.callback
                && !callback(epoch + 1, accuracy)
            {
                stopped = true;
                break;
            }

            // Early stopping
            if let Some(ref mut t) = tracker
                && t.update(accuracy)
            {
                stopped = true;
                break;
            }
        }

        FitResult::with_history(epochs_run, self.evaluate(x, y), stopped, history)
    }

    /// Evaluates accuracy on test data.
    ///
    /// Returns fraction of correct predictions (0.0 to 1.0).
    #[inline]
    #[must_use]
    pub fn evaluate(&self, x: &[Vec<u8>], y: &[u8]) -> f32 {
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

    /// Extracts learned rules from all clauses.
    #[must_use]
    pub fn rules(&self) -> Vec<Rule> {
        self.clauses.iter().map(Rule::from_clause).collect()
    }

    /// # Overview
    ///
    /// Returns clause weights for inspection.
    pub fn clause_weights(&self) -> Vec<f32> {
        self.clauses.iter().map(|c| c.weight()).collect()
    }

    /// Returns clause activation counts.
    #[must_use]
    pub fn clause_activations(&self) -> Vec<u32> {
        self.clauses.iter().map(|c| c.activations()).collect()
    }

    /// Quick constructor with sensible defaults.
    ///
    /// Equivalent to
    /// `Config::builder().clauses(n_clauses).features(n_features).build()`
    /// followed by `TsetlinMachine::new(config, threshold)`.
    ///
    /// # Panics
    ///
    /// Panics if n_clauses is odd or zero, or n_features is zero.
    #[must_use]
    pub fn quick(n_clauses: usize, n_features: usize, threshold: i32) -> Self {
        let config = Config::builder()
            .clauses(n_clauses)
            .features(n_features)
            .build()
            .expect("invalid quick config");
        Self::new(config, threshold)
    }
}

impl crate::model::TsetlinModel<Vec<u8>, u8> for TsetlinMachine {
    fn fit(&mut self, x: &[Vec<u8>], y: &[u8], epochs: usize, seed: u64) {
        TsetlinMachine::fit(self, x, y, epochs, seed);
    }

    fn predict(&self, x: &Vec<u8>) -> u8 {
        TsetlinMachine::predict(self, x)
    }

    fn evaluate(&self, x: &[Vec<u8>], y: &[u8]) -> f32 {
        TsetlinMachine::evaluate(self, x, y)
    }

    fn predict_batch(&self, xs: &[Vec<u8>]) -> Vec<u8> {
        TsetlinMachine::predict_batch(self, xs)
    }
}

impl crate::model::VotingModel<Vec<u8>> for TsetlinMachine {
    fn sum_votes(&self, x: &Vec<u8>) -> f32 {
        TsetlinMachine::sum_votes(self, x)
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

    #[test]
    fn weighted_clauses() {
        let config = Config::builder().clauses(10).features(2).build().unwrap();
        let tm = TsetlinMachine::new(config, 5);

        let weights = tm.clause_weights();
        assert!(weights.iter().all(|&w| (w - 1.0).abs() < 0.001));
    }

    #[test]
    fn adaptive_threshold() {
        let config = Config::builder().clauses(10).features(2).build().unwrap();
        let opts = AdvancedOptions {
            adaptive_t: true,
            t_min: 3.0,
            t_max: 20.0,
            t_lr: 0.5,
            ..Default::default()
        };
        let tm = TsetlinMachine::with_advanced(config, 10, opts);
        assert!((tm.threshold() - 10.0).abs() < 0.001);
    }
}
