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

#[cfg(test)]
mod tests {
    use super::*;

    struct MockModel;

    impl TsetlinModel<u8, u8> for MockModel {
        fn fit(&mut self, _x: &[u8], _y: &[u8], _epochs: usize, _seed: u64) {}

        fn predict(&self, x: &u8) -> u8 {
            *x % 2
        }

        fn evaluate(&self, x: &[u8], y: &[u8]) -> f32 {
            let correct = x
                .iter()
                .zip(y)
                .filter(|(xi, yi)| self.predict(xi) == **yi)
                .count();
            correct as f32 / x.len() as f32
        }
    }

    #[test]
    fn predict_batch_default_impl() {
        let model = MockModel;
        let xs = vec![0, 1, 2, 3, 4];
        let preds = model.predict_batch(&xs);
        assert_eq!(preds, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn mock_model_fit() {
        let mut model = MockModel;
        model.fit(&[1, 2, 3], &[0, 1, 0], 10, 42);
        // fit is no-op, just verify it doesn't panic
    }

    #[test]
    fn mock_model_evaluate() {
        let model = MockModel;
        // predict(x) = x % 2, so:
        // x=0 -> 0, y=0 -> correct
        // x=1 -> 1, y=1 -> correct
        // x=2 -> 0, y=0 -> correct
        let acc = model.evaluate(&[0, 1, 2], &[0, 1, 0]);
        assert!((acc - 1.0).abs() < 0.001);

        // 50% accuracy case
        let acc2 = model.evaluate(&[0, 1, 2, 3], &[1, 0, 1, 0]);
        assert!((acc2 - 0.0).abs() < 0.001);
    }

    struct MockVotingModel;

    impl TsetlinModel<u8, u8> for MockVotingModel {
        fn fit(&mut self, _x: &[u8], _y: &[u8], _epochs: usize, _seed: u64) {}

        fn predict(&self, x: &u8) -> u8 {
            if self.sum_votes(x) >= 0.0 { 1 } else { 0 }
        }

        fn evaluate(&self, x: &[u8], y: &[u8]) -> f32 {
            let correct = x
                .iter()
                .zip(y)
                .filter(|(xi, yi)| self.predict(xi) == **yi)
                .count();
            correct as f32 / x.len() as f32
        }
    }

    impl VotingModel<u8> for MockVotingModel {
        fn sum_votes(&self, x: &u8) -> f32 {
            (*x as f32) - 2.0 // returns negative for x < 2, positive for x >= 2
        }
    }

    #[test]
    fn voting_model_sum_votes() {
        let model = MockVotingModel;

        assert!((model.sum_votes(&0) - (-2.0)).abs() < 0.001);
        assert!((model.sum_votes(&2) - 0.0).abs() < 0.001);
        assert!((model.sum_votes(&5) - 3.0).abs() < 0.001);
    }

    #[test]
    fn voting_model_predict_uses_votes() {
        let model = MockVotingModel;

        // x < 2: negative votes -> predict 0
        assert_eq!(model.predict(&0), 0);
        assert_eq!(model.predict(&1), 0);

        // x >= 2: non-negative votes -> predict 1
        assert_eq!(model.predict(&2), 1);
        assert_eq!(model.predict(&5), 1);
    }

    #[test]
    fn voting_model_evaluate() {
        let model = MockVotingModel;
        let xs = vec![0, 1, 2, 3];
        let ys = vec![0, 0, 1, 1];
        let acc = model.evaluate(&xs, &ys);
        assert!((acc - 1.0).abs() < 0.001);
    }
}
