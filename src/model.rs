//! Unified trait for all Tsetlin Machine variants.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Unified interface for all Tsetlin Machine variants.
///
/// This trait provides a common API for binary, multi-class,
/// regression, and convolutional Tsetlin Machines.
///
/// # Type Parameters
///
/// * `X` - Input sample type (typically `Vec<u8>` for binary features)
/// * `Y` - Label type (varies by model: `u8`, `usize`, `i32`)
///
/// # Example
///
/// ```
/// use tsetlin_rs::{Config, TsetlinMachine, TsetlinModel};
///
/// let config = Config::builder().clauses(20).features(2).build().unwrap();
/// let mut tm = TsetlinMachine::new(config, 10);
///
/// let x = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
/// let y = vec![0, 1, 1, 0];
///
/// tm.fit(&x, &y, 100, 42);
/// let accuracy = tm.evaluate(&x, &y);
/// ```
pub trait TsetlinModel<X, Y> {
    /// Trains the model on labeled data.
    ///
    /// # Arguments
    ///
    /// * `x` - Training samples
    /// * `y` - Labels for each sample
    /// * `epochs` - Number of training iterations
    /// * `seed` - Random seed for reproducibility
    fn fit(&mut self, x: &[X], y: &[Y], epochs: usize, seed: u64);

    /// Predicts label for a single sample.
    fn predict(&self, x: &X) -> Y;

    /// Evaluates model accuracy/performance on test data.
    ///
    /// Returns a score between 0.0 and 1.0 (higher is better).
    fn evaluate(&self, x: &[X], y: &[Y]) -> f32;

    /// Batch prediction for multiple samples.
    fn predict_batch(&self, xs: &[X]) -> Vec<Y> {
        xs.iter().map(|x| self.predict(x)).collect()
    }
}

/// Extension trait for models with vote-based predictions.
pub trait VotingModel<X>: TsetlinModel<X, u8> {
    /// Returns raw vote sum for input.
    fn sum_votes(&self, x: &X) -> f32;
}
