use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use tsetlin_rs::{AdvancedOptions, Config, TsetlinMachine};

fn noisy_xor(n: usize, noise: f32, seed: u64) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i / 2) % 2;
        let b = i % 2;
        let mut xa = a as u8;
        let mut xb = b as u8;
        if rng.random::<f32>() < noise {
            xa = 1 - xa;
        }
        if rng.random::<f32>() < noise {
            xb = 1 - xb;
        }
        x.push(vec![xa, xb]);
        y.push((a ^ b) as u8);
    }
    (x, y)
}

fn main() {
    println!("Testing weighted clauses on noisy XOR (30% noise, 10 runs):\n");

    let config = Config::builder().clauses(20).features(2).build().unwrap();

    let mut sum1 = 0.0;
    let mut sum2 = 0.0;

    for run in 0..10 {
        let (x_train, y_train) = noisy_xor(200, 0.3, 100 + run);
        let (x_test, y_test) = noisy_xor(100, 0.3, 200 + run);

        // Standard
        let mut tm1 = TsetlinMachine::new(config, 10);
        tm1.fit(&x_train, &y_train, 100, run);
        let acc1 = tm1.evaluate(&x_test, &y_test);

        // Advanced
        let opts = AdvancedOptions {
            adaptive_t:      true,
            t_min:           3.0,
            t_max:           20.0,
            t_lr:            0.02,
            weight_lr:       0.1,
            weight_min:      0.3,
            weight_max:      1.5,
            prune_threshold: 0,
            prune_weight:    0.0
        };
        let mut tm2 = TsetlinMachine::with_advanced(config, 10, opts);
        tm2.fit(&x_train, &y_train, 100, run);
        let acc2 = tm2.evaluate(&x_test, &y_test);

        sum1 += acc1;
        sum2 += acc2;
        println!(
            "Run {}: Standard={:.0}% Advanced={:.0}%",
            run,
            acc1 * 100.0,
            acc2 * 100.0
        );
    }

    println!("\nAverage:");
    println!("  Standard: {:.1}%", sum1 * 10.0);
    println!("  Advanced: {:.1}%", sum2 * 10.0);
}
