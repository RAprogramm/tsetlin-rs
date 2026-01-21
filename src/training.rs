//! Training options and results.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// # Overview
///
/// Options for training a Tsetlin Machine.
#[derive(Debug, Clone)]
pub struct FitOptions {
    pub epochs:     usize,
    pub seed:       u64,
    pub early_stop: Option<EarlyStop>,
    pub shuffle:    bool,
    pub verbose:    bool
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            epochs:     100,
            seed:       42,
            early_stop: None,
            shuffle:    true,
            verbose:    false
        }
    }
}

impl FitOptions {
    /// # Overview
    ///
    /// Creates options with given epochs and seed.
    pub fn new(epochs: usize, seed: u64) -> Self {
        Self {
            epochs,
            seed,
            ..Default::default()
        }
    }

    /// # Overview
    ///
    /// Enables early stopping with patience.
    pub fn with_early_stop(mut self, patience: usize, min_delta: f32) -> Self {
        self.early_stop = Some(EarlyStop {
            patience,
            min_delta
        });
        self
    }

    /// # Overview
    ///
    /// Disables shuffling.
    pub fn no_shuffle(mut self) -> Self {
        self.shuffle = false;
        self
    }
}

/// # Overview
///
/// Early stopping configuration.
#[derive(Debug, Clone, Copy)]
pub struct EarlyStop {
    pub patience:  usize,
    pub min_delta: f32
}

/// # Overview
///
/// Result of training.
#[derive(Debug, Clone)]
pub struct FitResult {
    pub epochs_run:     usize,
    pub final_accuracy: f32,
    pub stopped_early:  bool,
    pub history:        Vec<f32>
}

impl FitResult {
    pub fn new(epochs_run: usize, final_accuracy: f32, stopped_early: bool) -> Self {
        Self {
            epochs_run,
            final_accuracy,
            stopped_early,
            history: Vec::new()
        }
    }
}

/// # Overview
///
/// Tracks early stopping state.
#[derive(Debug)]
pub struct EarlyStopTracker {
    patience:  usize,
    min_delta: f32,
    best:      f32,
    wait:      usize
}

impl EarlyStopTracker {
    pub fn new(config: &EarlyStop) -> Self {
        Self {
            patience:  config.patience,
            min_delta: config.min_delta,
            best:      0.0,
            wait:      0
        }
    }

    /// # Overview
    ///
    /// Updates tracker. Returns true if should stop.
    pub fn update(&mut self, accuracy: f32) -> bool {
        if accuracy > self.best + self.min_delta {
            self.best = accuracy;
            self.wait = 0;
            false
        } else {
            self.wait += 1;
            self.wait >= self.patience
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options() {
        let opts = FitOptions::default();
        assert_eq!(opts.epochs, 100);
        assert!(opts.shuffle);
    }

    #[test]
    fn early_stop_tracker() {
        let config = EarlyStop {
            patience:  3,
            min_delta: 0.01
        };
        let mut tracker = EarlyStopTracker::new(&config);

        assert!(!tracker.update(0.5));
        assert!(!tracker.update(0.6));
        assert!(!tracker.update(0.6));
        assert!(!tracker.update(0.6));
        assert!(tracker.update(0.6));
    }
}
