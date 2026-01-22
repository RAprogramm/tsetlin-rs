//! Tests for ClauseBank.

use super::ClauseBank;
use crate::utils::rng_from_seed;

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

#[test]
fn type_i_fires_strengthens_matching() {
    let mut bank = ClauseBank::new(1, 2, 100);
    let mut rng = rng_from_seed(42);

    // Apply type_i when clause fires with input [1, 0]
    for _ in 0..200 {
        bank.type_i(0, &[1, 0], true, 3.9, &mut rng);
    }

    // x_0 should be strengthened (include), ¬x_1 should be strengthened
    let states = bank.clause_states(0);
    assert!(states[0] > 100, "x_0 should move toward include");
    assert!(states[3] > 100, "¬x_1 should move toward include");
}

#[test]
fn type_i_no_fire_weakens_all() {
    let mut bank = ClauseBank::new(1, 2, 100);
    let mut rng = rng_from_seed(42);

    // Set some states above threshold
    bank.states[0] = 150;
    bank.states[1] = 150;

    // Apply type_i when clause doesn't fire
    for _ in 0..200 {
        bank.type_i(0, &[1, 0], false, 3.9, &mut rng);
    }

    // States should decrease toward exclusion
    assert!(
        bank.states[0] < 150,
        "states should decrease when not firing"
    );
}

#[test]
fn type_ii_blocks_false_positive() {
    let mut bank = ClauseBank::new(1, 2, 100);

    // Apply type_ii with input [1, 0]
    for _ in 0..100 {
        bank.type_ii(0, &[1, 0]);
    }

    // ¬x_0 should be activated (to block x[0]=1)
    // x_1 should be activated (to block x[1]=0)
    let states = bank.clause_states(0);
    assert!(states[1] > 100, "¬x_0 should be included to block");
    assert!(states[2] > 100, "x_1 should be included to block");
}

#[test]
fn clause_vote_returns_weighted() {
    let mut bank = ClauseBank::new(2, 2, 100);
    bank.weights[0] = 0.5;

    let vote = bank.clause_vote(0, &[0, 0]);
    assert!((vote - 0.5).abs() < 0.001); // polarity=1 * weight=0.5
}

#[test]
fn sum_votes_tracked_updates_activations() {
    let mut bank = ClauseBank::new(2, 2, 100);

    bank.sum_votes_tracked(&[0, 0]);
    bank.sum_votes_tracked(&[1, 1]);

    assert_eq!(bank.activations[0], 2);
    assert_eq!(bank.activations[1], 2);
}

#[test]
fn record_outcome_tracks_correct_incorrect() {
    let mut bank = ClauseBank::new(1, 2, 100);

    bank.record_outcome(0, true);
    bank.record_outcome(0, true);
    bank.record_outcome(0, false);

    assert_eq!(bank.correct[0], 2);
    assert_eq!(bank.incorrect[0], 1);
}

#[test]
fn is_dead_checks_thresholds() {
    let mut bank = ClauseBank::new(2, 2, 100);

    bank.activations[0] = 5;
    bank.weights[0] = 0.5;
    bank.activations[1] = 20;
    bank.weights[1] = 0.8;

    assert!(bank.is_dead(0, 10, 0.3)); // low activations
    assert!(!bank.is_dead(1, 10, 0.3)); // both ok
}

#[test]
fn prune_dead_resets_inactive() {
    let mut bank = ClauseBank::new(2, 2, 100);

    bank.activations[0] = 1;
    bank.weights[0] = 0.1;
    bank.states[0] = 150;

    bank.activations[1] = 100;
    bank.weights[1] = 1.0;
    bank.states[bank.stride] = 150;

    bank.prune_dead(10, 0.2);

    // Clause 0 should be reset
    assert_eq!(bank.states[0], 100);
    assert!((bank.weights[0] - 1.0).abs() < 0.001);

    // Clause 1 should remain
    assert_eq!(bank.states[bank.stride], 150);
}

#[test]
fn reset_activations_clears_all() {
    let mut bank = ClauseBank::new(3, 2, 100);
    bank.activations[0] = 10;
    bank.activations[1] = 20;
    bank.activations[2] = 30;

    bank.reset_activations();

    assert!(bank.activations.iter().all(|&a| a == 0));
}

#[test]
fn decrement_respects_floor() {
    let mut bank = ClauseBank::new(1, 2, 100);
    bank.states[0] = 1;

    bank.decrement(0, 0);

    assert_eq!(bank.states[0], 1); // Should not go below 1
}

#[test]
fn increment_respects_ceiling() {
    let mut bank = ClauseBank::new(1, 2, 100);
    bank.states[0] = 200; // max = 2 * n_states = 200

    bank.increment(0, 0);

    assert_eq!(bank.states[0], 200); // Should not exceed max
}
