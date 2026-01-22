//! Tests for ClauseBank.

use super::ClauseBank;

#[test]
fn new_bank() {
    let bank = ClauseBank::new(10, 5, 100);
    assert_eq!(bank.n_clauses(), 10);
    assert_eq!(bank.n_features(), 5);
    assert_eq!(bank.states().len(), 10 * 2 * 5);
}

#[test]
fn alternating_polarity() {
    let bank = ClauseBank::new(4, 2, 100);
    assert_eq!(bank.polarity(0), 1);
    assert_eq!(bank.polarity(1), -1);
    assert_eq!(bank.polarity(2), 1);
    assert_eq!(bank.polarity(3), -1);
}

#[test]
fn initial_weights() {
    let bank = ClauseBank::new(5, 3, 100);
    for w in bank.weights() {
        assert!((w - 1.0).abs() < 0.001);
    }
}

#[test]
fn evaluate_empty_clause_fires() {
    let bank = ClauseBank::new(1, 3, 100);
    assert!(bank.evaluate_clause(0, &[0, 1, 0]));
}

#[test]
fn evaluate_include_violation() {
    let mut bank = ClauseBank::new(1, 3, 100);
    for _ in 0..100 {
        bank.increment(0, 0);
    }
    assert!(!bank.evaluate_clause(0, &[0, 0, 0]));
    assert!(bank.evaluate_clause(0, &[1, 0, 0]));
}

#[test]
fn sum_votes_basic() {
    let bank = ClauseBank::new(4, 2, 100);
    let sum = bank.sum_votes(&[0, 0]);
    assert!((sum - 0.0).abs() < 0.001);
}

#[test]
fn activation_tracking() {
    let mut bank = ClauseBank::new(2, 2, 100);
    bank.evaluate_clause_tracked(0, &[0, 0]);
    bank.evaluate_clause_tracked(0, &[1, 1]);
    assert_eq!(bank.activations()[0], 2);
    assert_eq!(bank.activations()[1], 0);
}

#[test]
fn weight_update() {
    let mut bank = ClauseBank::new(1, 2, 100);
    bank.correct[0] = 8;
    bank.incorrect[0] = 2;
    bank.update_weights(0.1, 0.1, 2.0);
    assert!(bank.weights[0] > 1.0);
}

#[test]
fn reset_clause() {
    let mut bank = ClauseBank::new(2, 3, 100);
    bank.increment(0, 0);
    bank.weights[0] = 0.5;
    bank.activations[0] = 10;

    bank.reset_clause(0);

    assert!(bank.clause_states(0).iter().all(|&s| s == 100));
    assert!((bank.weights[0] - 1.0).abs() < 0.001);
    assert_eq!(bank.activations[0], 0);
}
