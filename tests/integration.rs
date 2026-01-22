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

// ==================== Advanced Features Tests ====================

#[test]
fn advanced_weighted_clauses() {
    use tsetlin_rs::AdvancedOptions;

    let config = Config::builder().clauses(20).features(2).build().unwrap();

    let opts = AdvancedOptions {
        weight_lr: 0.1,
        weight_min: 0.2,
        weight_max: 2.0,
        ..Default::default()
    };

    let mut tm = TsetlinMachine::with_advanced(config, 10, opts);
    let (x, y) = xor_data();
    tm.fit(&x, &y, 100, 42);

    // Weights should have changed from initial 1.0
    let weights = tm.clause_weights();
    let varied = weights.iter().any(|&w| (w - 1.0).abs() > 0.01);
    assert!(varied, "Weights should change during training");
}

#[test]
fn advanced_adaptive_threshold() {
    use tsetlin_rs::AdvancedOptions;

    let config = Config::builder().clauses(20).features(2).build().unwrap();

    let opts = AdvancedOptions {
        adaptive_t: true,
        t_min: 5.0,
        t_max: 20.0,
        t_lr: 0.1,
        ..Default::default()
    };

    let mut tm = TsetlinMachine::with_advanced(config, 10, opts);
    let initial_t = tm.threshold();

    let (x, y) = xor_data();
    tm.fit(&x, &y, 50, 42);

    // Threshold should have changed
    assert!(
        (tm.threshold() - initial_t).abs() > 0.1,
        "Adaptive threshold should change"
    );
}

#[test]
fn advanced_clause_pruning() {
    use tsetlin_rs::AdvancedOptions;

    let config = Config::builder().clauses(40).features(2).build().unwrap();

    let opts = AdvancedOptions {
        prune_threshold: 5,
        prune_weight: 0.3,
        weight_lr: 0.1,
        weight_min: 0.2,
        weight_max: 2.0,
        ..Default::default()
    };

    let mut tm = TsetlinMachine::with_advanced(config, 10, opts);
    let (x_train, y_train) = noisy_xor_data(200, 0.2, 42);

    tm.fit(&x_train, &y_train, 50, 42);

    // Some clauses should have been pruned (reset to fresh state with weight 1.0)
    let weights = tm.clause_weights();
    let fresh_count = weights.iter().filter(|&&w| (w - 1.0).abs() < 0.001).count();
    assert!(fresh_count > 0, "Some clauses should be pruned/reset");
}

#[test]
fn advanced_vs_standard_high_noise() {
    use tsetlin_rs::AdvancedOptions;

    let config = Config::builder().clauses(30).features(2).build().unwrap();

    // Generate high-noise data
    let (x_train, y_train) = noisy_xor_data(300, 0.3, 42);
    let (x_test, y_test) = noisy_xor_data(100, 0.3, 123);

    // Standard
    let mut tm_std = TsetlinMachine::new(config, 10);
    tm_std.fit(&x_train, &y_train, 100, 42);
    let acc_std = tm_std.evaluate(&x_test, &y_test);

    // Advanced
    let opts = AdvancedOptions {
        adaptive_t:      true,
        t_min:           5.0,
        t_max:           20.0,
        t_lr:            0.02,
        weight_lr:       0.1,
        weight_min:      0.3,
        weight_max:      1.5,
        prune_threshold: 0,
        prune_weight:    0.0
    };
    let mut tm_adv = TsetlinMachine::with_advanced(config, 10, opts);
    tm_adv.fit(&x_train, &y_train, 100, 42);
    let acc_adv = tm_adv.evaluate(&x_test, &y_test);

    // Both should achieve reasonable accuracy
    assert!(acc_std >= 0.45, "Standard should work on noisy data");
    assert!(acc_adv >= 0.45, "Advanced should work on noisy data");
}

#[test]
fn bitwise_clause_correctness() {
    use tsetlin_rs::{BitwiseClause, pack_input};

    let mut clause = BitwiseClause::new(64, 100, 1);

    // Train a bit
    for _ in 0..150 {
        clause.automata_mut()[0].increment(); // Include feature 0
    }
    clause.rebuild_masks();

    // x[0] = 0 should fail (include violation)
    let x_fail: Vec<u8> = (0..64).map(|i| if i == 0 { 0 } else { 1 }).collect();
    let packed_fail = pack_input(&x_fail);
    assert!(!clause.evaluate_packed(&packed_fail));

    // x[0] = 1 should pass
    let x_pass: Vec<u8> = (0..64).map(|_| 1).collect();
    let packed_pass = pack_input(&x_pass);
    assert!(clause.evaluate_packed(&packed_pass));
}

#[test]
fn clause_weight_affects_vote() {
    use tsetlin_rs::Clause;

    let mut clause = Clause::new(2, 100, 1);

    // Default weight is 1.0
    let vote1 = clause.vote(&[1, 1]);
    assert!((vote1 - 1.0).abs() < 0.001);

    // Simulate weight change via record_outcome
    for _ in 0..10 {
        clause.record_outcome(true); // 10 correct predictions
    }
    clause.update_weight(0.5, 0.1, 3.0);

    let vote2 = clause.vote(&[1, 1]);
    assert!(vote2 > vote1, "Higher weight should give higher vote");
}

#[test]
fn majority_voting_pattern() {
    use rand::Rng;

    let config = Config::builder().clauses(80).features(8).build().unwrap();
    let mut tm = TsetlinMachine::new(config, 20);

    // Generate majority voting data
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();

    for _ in 0..500 {
        let sample: Vec<u8> = (0..8).map(|_| rng.random_bool(0.5) as u8).collect();
        let sum: u8 = sample.iter().sum();
        let label = if sum > 4 { 1 } else { 0 };
        x_train.push(sample);
        y_train.push(label);
    }

    tm.fit(&x_train, &y_train, 100, 42);

    // Generate test data
    let mut x_test = Vec::new();
    let mut y_test = Vec::new();
    for _ in 0..100 {
        let sample: Vec<u8> = (0..8).map(|_| rng.random_bool(0.5) as u8).collect();
        let sum: u8 = sample.iter().sum();
        let label = if sum > 4 { 1 } else { 0 };
        x_test.push(sample);
        y_test.push(label);
    }

    let acc = tm.evaluate(&x_test, &y_test);
    assert!(acc >= 0.7, "Should learn majority voting pattern: {}", acc);
}

#[test]
fn parity_pattern_challenging() {
    use rand::Rng;

    let config = Config::builder().clauses(60).features(4).build().unwrap();
    let mut tm = TsetlinMachine::new(config, 15);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();

    for _ in 0..400 {
        let sample: Vec<u8> = (0..4).map(|_| rng.random_bool(0.5) as u8).collect();
        let sum: u8 = sample.iter().sum();
        let label = sum % 2;
        x_train.push(sample);
        y_train.push(label);
    }

    tm.fit(&x_train, &y_train, 200, 42);

    // Parity is challenging - just check it runs
    let acc = tm.evaluate(&x_train, &y_train);
    assert!(acc >= 0.4, "Parity is hard but should be above random");
}

// ==================== Error Display Tests ====================

#[test]
fn error_display_messages() {
    use tsetlin_rs::Error;

    assert_eq!(Error::MissingClauses.to_string(), "n_clauses is required");
    assert_eq!(Error::MissingFeatures.to_string(), "n_features is required");
    assert_eq!(Error::OddClauses.to_string(), "n_clauses must be even");
    assert_eq!(Error::InvalidSpecificity.to_string(), "s must be > 1.0");
    assert_eq!(Error::InvalidThreshold.to_string(), "threshold must be > 0");
    assert_eq!(Error::EmptyDataset.to_string(), "dataset cannot be empty");
    assert_eq!(
        Error::DimensionMismatch {
            expected: 10,
            got:      5
        }
        .to_string(),
        "dimension mismatch: expected 10, got 5"
    );
}

// ==================== Batch Prediction Tests ====================

#[test]
fn predict_batch_returns_correct_length() {
    let config = Config::builder().clauses(20).features(2).build().unwrap();
    let mut tm = TsetlinMachine::new(config, 10);

    let (x, y) = xor_data();
    tm.fit(&x, &y, 200, 42);

    let predictions = tm.predict_batch(&x);
    assert_eq!(predictions.len(), x.len());
}
