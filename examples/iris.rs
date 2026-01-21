//! Multi-class classification example.

use tsetlin_rs::{Config, MultiClass};

fn main() {
    let config = Config::builder()
        .clauses(100)
        .features(8)
        .build()
        .expect("valid config");

    let mut tm = MultiClass::new(config, 3, 50);

    let (x, y) = generate_data();

    println!("Training on Iris-like dataset...");
    println!("  {} samples, 8 features, 3 classes", x.len());

    tm.fit(&x, &y, 100, 42);

    println!("\nTraining accuracy: {:.1}%", tm.evaluate(&x, &y) * 100.0);

    println!("\nSample predictions:");
    for i in [0, 10, 20, 30, 40] {
        if i < x.len() {
            println!(
                "  Sample {} -> Class {} (expected: {})",
                i,
                tm.predict(&x[i]),
                y[i]
            );
        }
    }
}

fn generate_data() -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut x = Vec::new();
    let mut y = Vec::new();

    for _ in 0..20 {
        x.push(vec![1, 1, 0, 0, 0, 0, 1, 0]);
        y.push(0);
    }
    for _ in 0..20 {
        x.push(vec![0, 0, 1, 1, 0, 1, 0, 0]);
        y.push(1);
    }
    for _ in 0..20 {
        x.push(vec![0, 0, 0, 0, 1, 1, 0, 1]);
        y.push(2);
    }

    (x, y)
}
