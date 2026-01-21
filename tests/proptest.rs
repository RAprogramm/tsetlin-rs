//! Property-based tests for Tsetlin Machine.

use proptest::prelude::*;
use tsetlin_rs::{Automaton, Clause, Config, TsetlinMachine};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Automaton state stays within bounds after any number of operations.
    #[test]
    fn automaton_state_bounds(
        n_states in 10i16..200,
        increments in 0usize..500,
        decrements in 0usize..500
    ) {
        let mut a = Automaton::new(n_states);

        for _ in 0..increments {
            a.increment();
        }
        for _ in 0..decrements {
            a.decrement();
        }

        let state = a.state();
        prop_assert!(state >= 1);
        prop_assert!(state <= 2 * n_states);
    }

    /// Clause evaluation is deterministic.
    #[test]
    fn clause_evaluate_deterministic(
        n_features in 1usize..32,
        input in prop::collection::vec(0u8..=1, 1..32)
    ) {
        let n = n_features.min(input.len());
        let clause = Clause::new(n, 100, 1);

        let result1 = clause.evaluate(&input[..n]);
        let result2 = clause.evaluate(&input[..n]);

        prop_assert_eq!(result1, result2);
    }

    /// Empty clause (no active literals) always fires.
    #[test]
    fn empty_clause_always_fires(
        n_features in 1usize..16,
        input in prop::collection::vec(0u8..=1, 1..16)
    ) {
        let n = n_features.min(input.len());
        let clause = Clause::new(n, 100, 1);

        // New clause has no active literals
        prop_assert!(clause.evaluate(&input[..n]));
    }

    /// Clause weight stays in bounds after updates.
    #[test]
    fn clause_weight_bounds(
        correct in 0u32..100,
        incorrect in 0u32..100,
        lr in 0.01f32..0.5,
        min_w in 0.1f32..0.5,
        max_w in 1.5f32..3.0
    ) {
        let mut clause = Clause::new(4, 100, 1);

        for _ in 0..correct {
            clause.record_outcome(true);
        }
        for _ in 0..incorrect {
            clause.record_outcome(false);
        }

        clause.update_weight(lr, min_w, max_w);

        let w = clause.weight();
        prop_assert!(w >= min_w);
        prop_assert!(w <= max_w);
    }

    /// TsetlinMachine prediction is binary (0 or 1).
    #[test]
    fn prediction_is_binary(
        n_clauses in (2usize..20).prop_filter("must be even", |n| n % 2 == 0),
        n_features in 2usize..16,
        input in prop::collection::vec(0u8..=1, 2..16)
    ) {
        let n = n_features.min(input.len());
        let config = Config::builder()
            .clauses(n_clauses)
            .features(n)
            .build()
            .unwrap();
        let tm = TsetlinMachine::new(config, 10);

        let pred = tm.predict(&input[..n]);
        prop_assert!(pred == 0 || pred == 1);
    }

    /// Training doesn't panic on valid input.
    #[test]
    fn training_no_panic(
        n_samples in 4usize..20,
        n_features in 2usize..8,
        seed in 0u64..1000
    ) {
        let config = Config::builder()
            .clauses(10)
            .features(n_features)
            .build()
            .unwrap();
        let mut tm = TsetlinMachine::new(config, 5);

        let x: Vec<Vec<u8>> = (0..n_samples)
            .map(|i| (0..n_features).map(|j| ((i + j) % 2) as u8).collect())
            .collect();
        let y: Vec<u8> = (0..n_samples).map(|i| (i % 2) as u8).collect();

        tm.fit(&x, &y, 5, seed);

        // Just verify it completes without panic
        prop_assert!(true);
    }

    /// Accuracy is between 0 and 1.
    #[test]
    fn accuracy_in_range(
        n_samples in 4usize..20,
        n_features in 2usize..8
    ) {
        let config = Config::builder()
            .clauses(10)
            .features(n_features)
            .build()
            .unwrap();
        let mut tm = TsetlinMachine::new(config, 5);

        let x: Vec<Vec<u8>> = (0..n_samples)
            .map(|i| (0..n_features).map(|j| ((i + j) % 2) as u8).collect())
            .collect();
        let y: Vec<u8> = (0..n_samples).map(|i| (i % 2) as u8).collect();

        tm.fit(&x, &y, 10, 42);
        let acc = tm.evaluate(&x, &y);

        prop_assert!(acc >= 0.0);
        prop_assert!(acc <= 1.0);
    }

    /// Same seed produces same results.
    #[test]
    fn deterministic_training(
        seed in 0u64..1000
    ) {
        let config = Config::builder()
            .clauses(10)
            .features(2)
            .build()
            .unwrap();

        let x = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
        let y = vec![0, 1, 1, 0];

        let mut tm1 = TsetlinMachine::new(config, 5);
        let mut tm2 = TsetlinMachine::new(config, 5);

        tm1.fit(&x, &y, 20, seed);
        tm2.fit(&x, &y, 20, seed);

        for xi in &x {
            prop_assert_eq!(tm1.predict(xi), tm2.predict(xi));
        }
    }

    /// Config validation works correctly.
    #[test]
    fn config_validation(
        n_clauses in 1usize..100,
        n_features in 0usize..50,
        s in 0.5f32..5.0
    ) {
        let result = Config::builder()
            .clauses(n_clauses)
            .features(n_features)
            .specificity(s)
            .build();

        if n_clauses % 2 != 0 {
            prop_assert!(result.is_err());
        } else if n_features == 0 {
            prop_assert!(result.is_err());
        } else if s <= 1.0 {
            prop_assert!(result.is_err());
        } else {
            prop_assert!(result.is_ok());
        }
    }
}
