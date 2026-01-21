//! Parallel training and evaluation using rayon.

use rayon::prelude::*;

use crate::{Config, MultiClass, Regressor, TsetlinMachine};

/// # Overview
///
/// Parallel prediction for binary classification.
pub fn predict_batch(tm: &TsetlinMachine, x: &[Vec<u8>]) -> Vec<u8> {
    x.par_iter().map(|xi| tm.predict(xi)).collect()
}

/// # Overview
///
/// Parallel evaluation for binary classification.
pub fn evaluate_parallel(tm: &TsetlinMachine, x: &[Vec<u8>], y: &[u8]) -> f32 {
    let correct: usize = x
        .par_iter()
        .zip(y.par_iter())
        .filter(|(xi, yi)| tm.predict(xi) == **yi)
        .count();
    correct as f32 / x.len() as f32
}

/// # Overview
///
/// Parallel prediction for multi-class.
pub fn predict_batch_multiclass(tm: &MultiClass, x: &[Vec<u8>]) -> Vec<usize> {
    x.par_iter().map(|xi| tm.predict(xi)).collect()
}

/// # Overview
///
/// Parallel evaluation for multi-class.
pub fn evaluate_parallel_multiclass(tm: &MultiClass, x: &[Vec<u8>], y: &[usize]) -> f32 {
    let correct: usize = x
        .par_iter()
        .zip(y.par_iter())
        .filter(|(xi, yi)| tm.predict(xi) == **yi)
        .count();
    correct as f32 / x.len() as f32
}

/// # Overview
///
/// Parallel prediction for regression.
pub fn predict_batch_regressor(tm: &Regressor, x: &[Vec<u8>]) -> Vec<i32> {
    x.par_iter().map(|xi| tm.predict(xi)).collect()
}

/// # Overview
///
/// Parallel MAE for regression.
pub fn mae_parallel(tm: &Regressor, x: &[Vec<u8>], y: &[i32]) -> f32 {
    let sum: i32 = x
        .par_iter()
        .zip(y.par_iter())
        .map(|(xi, yi)| (tm.predict(xi) - *yi).abs())
        .sum();
    sum as f32 / x.len() as f32
}

/// # Overview
///
/// K-fold cross-validation with parallel folds.
pub fn cross_validate(
    config: Config,
    threshold: i32,
    x: &[Vec<u8>],
    y: &[u8],
    k_folds: usize,
    epochs: usize,
    seed: u64
) -> Vec<f32> {
    let fold_size = x.len() / k_folds;

    (0..k_folds)
        .into_par_iter()
        .map(|fold| {
            let test_start = fold * fold_size;
            let test_end = if fold == k_folds - 1 {
                x.len()
            } else {
                test_start + fold_size
            };

            let mut x_train = Vec::new();
            let mut y_train = Vec::new();
            let mut x_test = Vec::new();
            let mut y_test = Vec::new();

            for i in 0..x.len() {
                if i >= test_start && i < test_end {
                    x_test.push(x[i].clone());
                    y_test.push(y[i]);
                } else {
                    x_train.push(x[i].clone());
                    y_train.push(y[i]);
                }
            }

            let mut tm = TsetlinMachine::new(config, threshold);
            tm.fit(&x_train, &y_train, epochs, seed + fold as u64);
            tm.evaluate(&x_test, &y_test)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_batch_works() {
        let config = Config::builder().clauses(10).features(2).build().unwrap();
        let tm = TsetlinMachine::new(config, 5);

        let x = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
        let predictions = predict_batch(&tm, &x);

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn cross_validate_returns_k_scores() {
        let config = Config::builder()
            .clauses(10)
            .features(2)
            .states(50)
            .build()
            .unwrap();

        let x = vec![
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![1, 1],
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![1, 1],
        ];
        let y = vec![0, 1, 1, 0, 0, 1, 1, 0];

        let scores = cross_validate(config, 5, &x, &y, 2, 10, 42);

        assert_eq!(scores.len(), 2);
        for score in scores {
            assert!((0.0..=1.0).contains(&score));
        }
    }
}
