//! Comprehensive benchmark: Standard vs Advanced TsetlinMachine
//!
//! Tests across different scenarios to find where advanced features help.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use tsetlin_rs::{AdvancedOptions, Config, TsetlinMachine};

fn generate_data(
    pattern: &str,
    n_train: usize,
    n_test: usize,
    n_features: usize,
    noise: f32,
    seed: u64
) -> (Vec<Vec<u8>>, Vec<u8>, Vec<Vec<u8>>, Vec<u8>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let gen_sample = |rng: &mut Xoshiro256PlusPlus| -> (Vec<u8>, u8) {
        let mut x: Vec<u8> = (0..n_features).map(|_| rng.random_bool(0.5) as u8).collect();
        let label = match pattern {
            "xor" => (x[0] ^ x[1]) as u8,
            "and" => (x[0] & x[1]) as u8,
            "or" => (x[0] | x[1]) as u8,
            "majority" => {
                let sum: u8 = x.iter().sum();
                if sum > (n_features / 2) as u8 { 1 } else { 0 }
            }
            "parity" => {
                let sum: u8 = x.iter().sum();
                sum % 2
            }
            "first_half" => {
                let sum: u8 = x[..n_features / 2].iter().sum();
                if sum > (n_features / 4) as u8 { 1 } else { 0 }
            }
            _ => 0
        };

        // Add noise
        for xi in &mut x {
            if rng.random::<f32>() < noise {
                *xi = 1 - *xi;
            }
        }

        (x, label)
    };

    let (x_train, y_train): (Vec<_>, Vec<_>) =
        (0..n_train).map(|_| gen_sample(&mut rng)).unzip();
    let (x_test, y_test): (Vec<_>, Vec<_>) =
        (0..n_test).map(|_| gen_sample(&mut rng)).unzip();

    (x_train, y_train, x_test, y_test)
}

fn run_experiment(
    name: &str,
    pattern: &str,
    n_features: usize,
    n_clauses: usize,
    noise: f32,
    n_runs: usize
) {
    let config = Config::builder()
        .clauses(n_clauses)
        .features(n_features)
        .build()
        .unwrap();

    let advanced_opts = AdvancedOptions {
        adaptive_t:      true,
        t_min:           5.0,
        t_max:           30.0,
        t_lr:            0.02,
        weight_lr:       0.08,
        weight_min:      0.3,
        weight_max:      1.5,
        prune_threshold: 3,
        prune_weight:    0.25
    };

    let mut std_sum = 0.0;
    let mut adv_sum = 0.0;

    for run in 0..n_runs {
        let (x_train, y_train, x_test, y_test) =
            generate_data(pattern, 500, 200, n_features, noise, run as u64 * 1000);

        // Standard
        let mut tm_std = TsetlinMachine::new(config, 15);
        tm_std.fit(&x_train, &y_train, 100, run as u64);
        std_sum += tm_std.evaluate(&x_test, &y_test);

        // Advanced
        let mut tm_adv = TsetlinMachine::with_advanced(config, 15, advanced_opts.clone());
        tm_adv.fit(&x_train, &y_train, 100, run as u64);
        adv_sum += tm_adv.evaluate(&x_test, &y_test);
    }

    let std_avg = std_sum / n_runs as f32 * 100.0;
    let adv_avg = adv_sum / n_runs as f32 * 100.0;
    let diff = adv_avg - std_avg;
    let winner = if diff > 1.0 {
        "Advanced ✓"
    } else if diff < -1.0 {
        "Standard ✓"
    } else {
        "≈ Equal"
    };

    println!(
        "| {:<20} | {:>6.1}% | {:>6.1}% | {:>+5.1}% | {} |",
        name, std_avg, adv_avg, diff, winner
    );
}

fn main() {
    println!("# Tsetlin Machine: Standard vs Advanced Features\n");
    println!("Testing across different patterns, noise levels, and feature counts.\n");

    println!("## Test 1: Noise Sensitivity (XOR, 2 features)\n");
    println!(
        "| {:<20} | {:>7} | {:>7} | {:>6} | {} |",
        "Scenario", "Std", "Adv", "Δ", "Winner"
    );
    println!("|{:-<22}|{:-<9}|{:-<9}|{:-<8}|{:-<12}|", "", "", "", "", "");

    for noise in [0.0, 0.1, 0.2, 0.3, 0.4] {
        let name = format!("XOR noise={:.0}%", noise * 100.0);
        run_experiment(&name, "xor", 2, 20, noise, 10);
    }

    println!("\n## Test 2: Pattern Complexity (10% noise)\n");
    println!(
        "| {:<20} | {:>7} | {:>7} | {:>6} | {} |",
        "Pattern", "Std", "Adv", "Δ", "Winner"
    );
    println!("|{:-<22}|{:-<9}|{:-<9}|{:-<8}|{:-<12}|", "", "", "", "", "");

    run_experiment("XOR (2 feat)", "xor", 2, 20, 0.1, 10);
    run_experiment("AND (2 feat)", "and", 2, 20, 0.1, 10);
    run_experiment("OR (2 feat)", "or", 2, 20, 0.1, 10);
    run_experiment("Majority (8 feat)", "majority", 8, 50, 0.1, 10);
    run_experiment("Parity (6 feat)", "parity", 6, 40, 0.1, 10);
    run_experiment("First-half (8 feat)", "first_half", 8, 50, 0.1, 10);

    println!("\n## Test 3: Scale (Majority voting, 10% noise)\n");
    println!(
        "| {:<20} | {:>7} | {:>7} | {:>6} | {} |",
        "Features/Clauses", "Std", "Adv", "Δ", "Winner"
    );
    println!("|{:-<22}|{:-<9}|{:-<9}|{:-<8}|{:-<12}|", "", "", "", "", "");

    run_experiment("4 feat / 20 cls", "majority", 4, 20, 0.1, 10);
    run_experiment("8 feat / 40 cls", "majority", 8, 40, 0.1, 10);
    run_experiment("16 feat / 80 cls", "majority", 16, 80, 0.1, 10);
    run_experiment("32 feat / 160 cls", "majority", 32, 160, 0.1, 5);

    println!("\n## Test 4: High Noise Stress Test\n");
    println!(
        "| {:<20} | {:>7} | {:>7} | {:>6} | {} |",
        "Pattern + Noise", "Std", "Adv", "Δ", "Winner"
    );
    println!("|{:-<22}|{:-<9}|{:-<9}|{:-<8}|{:-<12}|", "", "", "", "", "");

    run_experiment("Majority 30% noise", "majority", 8, 50, 0.3, 10);
    run_experiment("Parity 30% noise", "parity", 6, 40, 0.3, 10);
    run_experiment("XOR 40% noise", "xor", 2, 30, 0.4, 10);
}
