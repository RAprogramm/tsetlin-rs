//! Training options, callbacks, and results.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};
#[cfg(feature = "std")]
use std::boxed::Box;

/// Progress callback type for training.
///
/// Called after each epoch with (epoch, accuracy).
/// Return `false` to stop training early.
pub type ProgressCallback = Box<dyn FnMut(usize, f32) -> bool + Send>;

/// Options for training a Tsetlin Machine.
pub struct FitOptions {
    pub epochs:     usize,
    pub seed:       u64,
    pub early_stop: Option<EarlyStop>,
    pub shuffle:    bool,
    pub verbose:    bool,
    pub callback:   Option<ProgressCallback>
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            epochs:     100,
            seed:       42,
            early_stop: None,
            shuffle:    true,
            verbose:    false,
            callback:   None
        }
    }
}

impl core::fmt::Debug for FitOptions {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FitOptions")
            .field("epochs", &self.epochs)
            .field("seed", &self.seed)
            .field("early_stop", &self.early_stop)
            .field("shuffle", &self.shuffle)
            .field("verbose", &self.verbose)
            .field("callback", &self.callback.as_ref().map(|_| "..."))
            .finish()
    }
}

impl FitOptions {
    /// Creates options with given epochs and seed.
    #[must_use]
    pub fn new(epochs: usize, seed: u64) -> Self {
        Self {
            epochs,
            seed,
            early_stop: None,
            shuffle: true,
            verbose: false,
            callback: None
        }
    }

    /// Enables early stopping with patience.
    ///
    /// Training stops if accuracy doesn't improve by `min_delta`
    /// for `patience` consecutive epochs.
    #[must_use]
    pub fn with_early_stop(mut self, patience: usize, min_delta: f32) -> Self {
        self.early_stop = Some(EarlyStop {
            patience,
            min_delta
        });
        self
    }

    /// Disables shuffling of training data.
    #[must_use]
    pub fn no_shuffle(mut self) -> Self {
        self.shuffle = false;
        self
    }

    /// Sets progress callback.
    ///
    /// The callback receives (epoch, accuracy) after each epoch.
    /// Return `false` to stop training early.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let opts = FitOptions::new(100, 42)
    ///     .with_callback(|epoch, acc| {
    ///         println!("Epoch {}: {:.1}%", epoch, acc * 100.0);
    ///         true  // continue training
    ///     });
    /// ```
    #[must_use]
    pub fn with_callback<F>(mut self, callback: F) -> Self
    where
        F: FnMut(usize, f32) -> bool + Send + 'static
    {
        self.callback = Some(Box::new(callback));
        self
    }
}

/// Early stopping configuration.
#[derive(Debug, Clone, Copy)]
pub struct EarlyStop {
    /// Number of epochs without improvement before stopping.
    pub patience:  usize,
    /// Minimum improvement required to reset patience counter.
    pub min_delta: f32
}

/// Result of training.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Number of epochs actually run.
    pub epochs_run:     usize,
    /// Final accuracy on training data.
    pub final_accuracy: f32,
    /// Whether training stopped early.
    pub stopped_early:  bool,
    /// Accuracy history per epoch.
    pub history:        Vec<f32>
}

impl FitResult {
    /// Creates a new FitResult.
    #[must_use]
    pub fn new(epochs_run: usize, final_accuracy: f32, stopped_early: bool) -> Self {
        Self {
            epochs_run,
            final_accuracy,
            stopped_early,
            history: Vec::new()
        }
    }

    /// Creates FitResult with accuracy history.
    #[must_use]
    pub fn with_history(
        epochs_run: usize,
        final_accuracy: f32,
        stopped_early: bool,
        history: Vec<f32>
    ) -> Self {
        Self {
            epochs_run,
            final_accuracy,
            stopped_early,
            history
        }
    }
}

/// Tracks early stopping state during training.
#[derive(Debug)]
pub struct EarlyStopTracker {
    patience:  usize,
    min_delta: f32,
    best:      f32,
    wait:      usize
}

impl EarlyStopTracker {
    /// Creates tracker from early stop config.
    #[must_use]
    pub fn new(config: &EarlyStop) -> Self {
        Self {
            patience:  config.patience,
            min_delta: config.min_delta,
            best:      0.0,
            wait:      0
        }
    }

    /// Updates tracker with new accuracy.
    ///
    /// Returns `true` if training should stop (no improvement for patience
    /// epochs).
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

    #[test]
    fn fit_options_new() {
        let opts = FitOptions::new(50, 123);
        assert_eq!(opts.epochs, 50);
        assert_eq!(opts.seed, 123);
        assert!(opts.shuffle);
        assert!(opts.early_stop.is_none());
        assert!(opts.callback.is_none());
    }

    #[test]
    fn fit_options_with_early_stop() {
        let opts = FitOptions::new(100, 42).with_early_stop(5, 0.001);
        let es = opts.early_stop.unwrap();
        assert_eq!(es.patience, 5);
        assert!((es.min_delta - 0.001).abs() < 0.0001);
    }

    #[test]
    fn fit_options_no_shuffle() {
        let opts = FitOptions::new(100, 42).no_shuffle();
        assert!(!opts.shuffle);
    }

    #[test]
    fn fit_options_with_callback() {
        let opts = FitOptions::new(100, 42).with_callback(|_epoch, _acc| true);
        assert!(opts.callback.is_some());
    }

    #[test]
    fn fit_options_debug() {
        let opts = FitOptions::new(100, 42).with_callback(|_, _| true);
        let debug_str = format!("{:?}", opts);
        assert!(debug_str.contains("FitOptions"));
        assert!(debug_str.contains("epochs"));
        assert!(debug_str.contains("100"));
    }

    #[test]
    fn fit_result_new() {
        let result = FitResult::new(50, 0.95, true);
        assert_eq!(result.epochs_run, 50);
        assert!((result.final_accuracy - 0.95).abs() < 0.001);
        assert!(result.stopped_early);
        assert!(result.history.is_empty());
    }

    #[test]
    fn fit_result_with_history() {
        let history = vec![0.5, 0.7, 0.85, 0.9];
        let result = FitResult::with_history(4, 0.9, false, history.clone());
        assert_eq!(result.epochs_run, 4);
        assert!((result.final_accuracy - 0.9).abs() < 0.001);
        assert!(!result.stopped_early);
        assert_eq!(result.history, history);
    }
}
