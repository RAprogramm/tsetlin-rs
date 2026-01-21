//! XOR classification example.

use tsetlin_rs::{Config, TsetlinMachine};

fn main() {
    let config = Config::builder()
        .clauses(20)
        .features(2)
        .build()
        .expect("valid config");

    let mut tm = TsetlinMachine::new(config, 10);

    let x = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
    let y = vec![0, 1, 1, 0];

    println!("Training on XOR dataset...");
    tm.fit(&x, &y, 200, 42);

    println!("\nPredictions:");
    for (xi, &yi) in x.iter().zip(y.iter()) {
        let pred = tm.predict(xi);
        let status = if pred == yi { "OK" } else { "WRONG" };
        println!("  {:?} -> {} (expected: {}) {}", xi, pred, yi, status);
    }

    println!("\nAccuracy: {:.1}%", tm.evaluate(&x, &y) * 100.0);

    println!("\nLearned rules:");
    for (i, rule) in tm
        .rules()
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.is_empty())
        .take(5)
    {
        println!("  Clause {}: {:?}", i, rule);
    }
}
