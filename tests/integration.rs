//! Integration tests for Tsetlin Machine.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use tsetlin_rs::{Config, MultiClass, Regressor, TsetlinMachine};

fn xor_data() -> (Vec<Vec<u8>>, Vec<u8>) {
    let x = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
    let y = vec![0, 1, 1, 0];
    (x, y)
}

fn noisy_xor_data(n_samples: usize, noise: f32, seed: u64) -> (Vec<Vec<u8>>, Vec<u8>) {
    use rand::Rng;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let a = (i / 2) % 2;
        let b = i % 2;
        let label = a ^ b;
        let flip_a: bool = rng.random::<f32>() < noise;
        let flip_b: bool = rng.random::<f32>() < noise;
        x.push(vec![
            if flip_a { 1 - a as u8 } else { a as u8 },
            if flip_b { 1 - b as u8 } else { b as u8 },
        ]);
        y.push(label as u8);
    }
    (x, y)
}

#[test]
fn binary_xor_convergence() {
    let config = Config::builder().clauses(20).features(2).build().unwrap();
    let mut tm = TsetlinMachine::new(config, 10);

    let (x, y) = xor_data();
    tm.fit(&x, &y, 200, 42);

    assert!(tm.evaluate(&x, &y) >= 0.75);
}

#[test]
fn binary_noisy_xor() {
    let config = Config::builder().clauses(40).features(2).build().unwrap();
    let mut tm = TsetlinMachine::new(config, 15);

    let (x_train, y_train) = noisy_xor_data(400, 0.1, 42);
    let (x_test, y_test) = noisy_xor_data(100, 0.1, 123);

    tm.fit(&x_train, &y_train, 100, 42);

    assert!(tm.evaluate(&x_test, &y_test) >= 0.6);
}

#[test]
fn multiclass_simple_patterns() {
    let config = Config::builder().clauses(60).features(4).build().unwrap();
    let mut tm = MultiClass::new(config, 3, 25);

    let x = vec![
        vec![1, 1, 0, 0],
        vec![1, 1, 0, 1],
        vec![1, 1, 1, 0],
        vec![0, 0, 1, 1],
        vec![0, 1, 1, 1],
        vec![1, 0, 1, 1],
        vec![1, 0, 1, 0],
        vec![1, 0, 0, 1],
        vec![0, 1, 0, 1],
    ];
    let y = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

    tm.fit(&x, &y, 100, 42);

    assert!(tm.evaluate(&x, &y) >= 0.5);
}

#[test]
fn regressor_basic() {
    let config = Config::builder()
        .clauses(40)
        .features(4)
        .specificity(3.0)
        .build()
        .unwrap();
    let mut tm = Regressor::new(config, 20);

    let x = vec![
        vec![1, 1, 1, 1],
        vec![1, 1, 1, 0],
        vec![1, 1, 0, 0],
        vec![1, 0, 0, 0],
        vec![0, 0, 0, 0],
    ];
    let y = vec![20, 15, 10, 5, 0];

    tm.fit(&x, &y, 100, 42);

    assert!(tm.mae(&x, &y) <= 15.0);
}

#[test]
fn model_determinism() {
    let config = Config::builder().clauses(20).features(2).build().unwrap();
    let (x, y) = xor_data();

    let mut tm1 = TsetlinMachine::new(config, 10);
    let mut tm2 = TsetlinMachine::new(config, 10);

    tm1.fit(&x, &y, 50, 42);
    tm2.fit(&x, &y, 50, 42);

    for xi in &x {
        assert_eq!(tm1.predict(xi), tm2.predict(xi));
    }
}

#[test]
fn early_stopping() {
    use tsetlin_rs::FitOptions;

    let config = Config::builder().clauses(20).features(2).build().unwrap();
    let mut tm = TsetlinMachine::new(config, 10);

    let (x, y) = xor_data();
    let opts = FitOptions::new(1000, 42).with_early_stop(10, 0.01);

    let result = tm.fit_with_options(&x, &y, opts);

    assert!(result.epochs_run < 1000 || result.final_accuracy >= 0.99);
}

#[test]
fn rule_extraction() {
    use tsetlin_rs::Rule;

    let config = Config::builder().clauses(20).features(2).build().unwrap();
    let mut tm = TsetlinMachine::new(config, 10);

    let (x, y) = xor_data();
    tm.fit(&x, &y, 200, 42);

    let rules = tm.rules();
    assert_eq!(rules.len(), 20);

    let non_empty: Vec<&Rule> = rules.iter().filter(|r| !r.is_empty()).collect();
    assert!(!non_empty.is_empty());
}
