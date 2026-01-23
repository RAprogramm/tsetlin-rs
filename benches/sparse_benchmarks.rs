//! Sparse clause benchmarks with relative metrics.
//!
//! Results are machine-independent ratios (compression, speedup).
//! Uses realistic data patterns with proper sparsity levels.

use core::hint::black_box;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tsetlin_rs::{
    Clause, ClauseBank, Config, SparseClause, SparseClauseBank, TsetlinMachine,
    utils::rng_from_seed
};

/// Dataset configuration for benchmarks.
struct DatasetConfig {
    name:       &'static str,
    n_features: usize,
    n_clauses:  usize,
    n_samples:  usize,
    epochs:     usize
}

/// Realistic benchmark datasets.
const DATASETS: &[DatasetConfig] = &[
    DatasetConfig {
        name:       "xor",
        n_features: 2,
        n_clauses:  20,
        n_samples:  100,
        epochs:     50
    },
    DatasetConfig {
        name:       "parity_8",
        n_features: 8,
        n_clauses:  50,
        n_samples:  256,
        epochs:     100
    },
    DatasetConfig {
        name:       "mnist_sparse",
        n_features: 784,
        n_clauses:  200,
        n_samples:  200,
        epochs:     30
    },
    DatasetConfig {
        name:       "nlp_5k",
        n_features: 5000,
        n_clauses:  100,
        n_samples:  500,
        epochs:     20
    },
    DatasetConfig {
        name:       "nlp_10k",
        n_features: 10000,
        n_clauses:  100,
        n_samples:  500,
        epochs:     20
    }
];

/// Simple LCG for deterministic data generation.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state >> 33
}

/// Generates XOR dataset (2 features, perfect logical pattern).
fn xor_data(n_samples: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
    let x: Vec<Vec<u8>> = (0..n_samples)
        .map(|i| vec![(i % 2) as u8, ((i / 2) % 2) as u8])
        .collect();
    let y: Vec<u8> = x.iter().map(|xi| xi[0] ^ xi[1]).collect();
    (x, y)
}

/// Generates n-bit parity dataset (XOR of all bits).
fn parity_data(n_bits: usize, n_samples: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
    let actual_samples = n_samples.min(1 << n_bits);
    let x: Vec<Vec<u8>> = (0..actual_samples)
        .map(|i| (0..n_bits).map(|b| ((i >> b) & 1) as u8).collect())
        .collect();
    let y: Vec<u8> = x
        .iter()
        .map(|xi| xi.iter().fold(0u8, |acc, &b| acc ^ b))
        .collect();
    (x, y)
}

/// Generates sparse NLP-like bag-of-words data (~2% sparsity).
/// Simulates document classification with sparse word presence.
fn nlp_sparse_data(vocab_size: usize, n_samples: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut rng_state = 42u64;
    let words_per_doc = (vocab_size / 50).max(10); // ~2% density

    let x: Vec<Vec<u8>> = (0..n_samples)
        .map(|i| {
            let mut doc = vec![0u8; vocab_size];
            // Class-specific words (first 5% of vocab for class 0, next 5% for class 1)
            let class_offset = if i % 2 == 0 { 0 } else { vocab_size / 20 };
            for _ in 0..words_per_doc {
                let word_idx = (class_offset
                    + (lcg_next(&mut rng_state) as usize % (vocab_size / 20)))
                    % vocab_size;
                doc[word_idx] = 1;
            }
            // Add some random noise words
            for _ in 0..words_per_doc / 3 {
                let idx = lcg_next(&mut rng_state) as usize % vocab_size;
                doc[idx] = 1;
            }
            doc
        })
        .collect();
    let y: Vec<u8> = (0..n_samples).map(|i| (i % 2) as u8).collect();
    (x, y)
}

/// Generates MNIST-like binary image data (~15% active pixels).
/// Simulates digit classification with sparse binary features.
fn mnist_sparse_data(n_features: usize, n_samples: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut rng_state = 123u64;
    let side = (n_features as f64).sqrt() as usize;
    let active_ratio = 0.15; // ~15% pixels active

    let x: Vec<Vec<u8>> = (0..n_samples)
        .map(|i| {
            let mut img = vec![0u8; n_features];
            let class = i % 2;

            // Create class-specific patterns (strokes)
            let center_x = side / 2;
            let center_y = side / 2;

            if class == 0 {
                // Horizontal stroke pattern
                for dx in 0..side / 3 {
                    let idx = center_y * side + (center_x - side / 6 + dx);
                    if idx < n_features {
                        img[idx] = 1;
                    }
                }
            } else {
                // Vertical stroke pattern
                for dy in 0..side / 3 {
                    let idx = (center_y - side / 6 + dy) * side + center_x;
                    if idx < n_features {
                        img[idx] = 1;
                    }
                }
            }

            // Add random noise to reach ~15% density
            let target_active = (n_features as f64 * active_ratio) as usize;
            let current_active: usize = img.iter().map(|&x| x as usize).sum();
            for _ in current_active..target_active {
                let idx = lcg_next(&mut rng_state) as usize % n_features;
                img[idx] = 1;
            }
            img
        })
        .collect();
    let y: Vec<u8> = (0..n_samples).map(|i| (i % 2) as u8).collect();
    (x, y)
}

/// Generates dataset based on configuration.
fn generate_dataset(config: &DatasetConfig) -> (Vec<Vec<u8>>, Vec<u8>) {
    match config.name {
        "xor" => xor_data(config.n_samples),
        "parity_8" => parity_data(8, config.n_samples),
        "mnist_sparse" => mnist_sparse_data(config.n_features, config.n_samples),
        "nlp_5k" | "nlp_10k" => nlp_sparse_data(config.n_features, config.n_samples),
        _ => panic!("Unknown dataset: {}", config.name)
    }
}

/// Creates trained ClauseBank with realistic data.
fn create_trained_bank(config: &DatasetConfig) -> ClauseBank {
    let mut bank = ClauseBank::new(config.n_clauses, config.n_features, 100);
    let mut rng = rng_from_seed(42);
    let (x, y) = generate_dataset(config);

    for _ in 0..config.epochs {
        for (xi, &yi) in x.iter().zip(y.iter()) {
            bank.train_sample(xi, yi, 15.0, 3.9, &mut rng);
        }
    }

    bank
}

/// Creates trained TsetlinMachine with realistic data.
fn create_trained_tm(config: &DatasetConfig) -> TsetlinMachine {
    let tm_config = Config::builder()
        .clauses(config.n_clauses)
        .features(config.n_features)
        .build()
        .unwrap();
    let mut tm = TsetlinMachine::new(tm_config, 15);
    let (x, y) = generate_dataset(config);

    tm.fit(&x, &y, config.epochs, 42);
    tm
}

/// Benchmark: Memory compression ratio (dense_bytes / sparse_bytes).
fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio");
    group.sample_size(50);

    for config in DATASETS {
        let bank = create_trained_bank(config);
        let sparse = SparseClauseBank::from_clause_bank(&bank);

        let stats = sparse.memory_stats();
        let dense_bytes = config.n_clauses * 2 * config.n_features * 2; // i16 states
        let sparse_bytes = stats.total();
        let ratio = dense_bytes as f64 / sparse_bytes.max(1) as f64;

        // Print results (machine-independent)
        println!(
            "\n[{}] features={}, clauses={}\n  \
             Dense:  {} bytes\n  \
             Sparse: {} bytes\n  \
             Compression: {:.1}x\n  \
             Avg literals/clause: {:.1}\n  \
             Sparsity: {:.3}%",
            config.name,
            config.n_features,
            config.n_clauses,
            dense_bytes,
            sparse_bytes,
            ratio,
            stats.avg_literals_per_clause(),
            stats.sparsity() * 100.0
        );

        // Benchmark conversion time (for comparison only)
        group.bench_function(BenchmarkId::new("convert", config.name), |b| {
            b.iter(|| {
                let s = SparseClauseBank::from_clause_bank(black_box(&bank));
                black_box(s)
            });
        });
    }

    group.finish();
}

/// Benchmark: Inference speedup ratio (dense_time / sparse_time).
fn bench_inference_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_speedup");
    group.sample_size(100);

    for config in DATASETS {
        let bank = create_trained_bank(config);
        let sparse = SparseClauseBank::from_clause_bank(&bank);
        let (dataset, _) = generate_dataset(config);
        let x = &dataset[0];

        // Measure relative speedup (warm-up + timed runs)
        const ITERS: usize = 1000;

        // Warm-up
        for _ in 0..100 {
            black_box(bank.sum_votes(x));
            black_box(sparse.sum_votes(x));
        }

        // Dense timing
        let dense_start = Instant::now();
        for _ in 0..ITERS {
            black_box(bank.sum_votes(black_box(x)));
        }
        let dense_time = dense_start.elapsed();

        // Sparse timing
        let sparse_start = Instant::now();
        for _ in 0..ITERS {
            black_box(sparse.sum_votes(black_box(x)));
        }
        let sparse_time = sparse_start.elapsed();

        let speedup = dense_time.as_nanos() as f64 / sparse_time.as_nanos().max(1) as f64;

        println!(
            "\n[{}] Inference speedup: {:.2}x ({:?} dense vs {:?} sparse per {} iters)",
            config.name, speedup, dense_time, sparse_time, ITERS
        );

        // Criterion benchmarks for detailed stats
        group.bench_function(BenchmarkId::new("dense", config.name), |b| {
            b.iter(|| black_box(bank.sum_votes(black_box(x))));
        });

        group.bench_function(BenchmarkId::new("sparse", config.name), |b| {
            b.iter(|| black_box(sparse.sum_votes(black_box(x))));
        });
    }

    group.finish();
}

/// Benchmark: Single clause evaluation speedup.
fn bench_clause_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("clause_speedup");

    for &n_features in &[64, 784, 5000] {
        let mut clause = Clause::new(n_features, 100, 1);

        // Activate ~5% of literals
        let active_count = (n_features / 20).max(2);
        for i in 0..active_count {
            for _ in 0..200 {
                clause.automata_mut()[i * 2].increment();
            }
        }

        let sparse = SparseClause::from_clause(&clause);
        let x: Vec<u8> = (0..n_features).map(|i| (i % 2) as u8).collect();

        println!(
            "\n[clause_{}] literals: {} (dense has {} automata)",
            n_features,
            sparse.n_literals(),
            n_features * 2
        );

        group.bench_function(BenchmarkId::new("dense", n_features), |b| {
            b.iter(|| black_box(clause.evaluate(black_box(&x))));
        });

        group.bench_function(BenchmarkId::new("sparse", n_features), |b| {
            b.iter(|| black_box(sparse.evaluate(black_box(&x))));
        });
    }

    group.finish();
}

/// Benchmark: TsetlinMachine.to_sparse() conversion.
fn bench_to_sparse_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_sparse");
    group.sample_size(50);

    for config in DATASETS {
        let tm = create_trained_tm(config);

        group.bench_function(BenchmarkId::new("convert", config.name), |b| {
            b.iter(|| {
                let s = tm.to_sparse();
                black_box(s)
            });
        });
    }

    group.finish();
}

/// Benchmark: Batch prediction comparison.
fn bench_batch_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_prediction");
    group.sample_size(30);

    // Use MNIST-like config
    let config = &DATASETS[2]; // mnist_sparse
    let tm = create_trained_tm(config);
    let sparse = tm.to_sparse();

    for &batch_size in &[10, 100, 1000] {
        let batch: Vec<Vec<u8>> = (0..batch_size)
            .map(|i| {
                (0..config.n_features)
                    .map(|j| ((i + j) % 2) as u8)
                    .collect()
            })
            .collect();

        group.bench_function(BenchmarkId::new("dense", batch_size), |b| {
            b.iter(|| {
                for x in &batch {
                    black_box(tm.predict(black_box(x)));
                }
            });
        });

        group.bench_function(BenchmarkId::new("sparse", batch_size), |b| {
            b.iter(|| {
                for x in &batch {
                    black_box(sparse.predict(black_box(x)));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = sparse_benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(2));
    targets =
        bench_compression_ratio,
        bench_inference_speedup,
        bench_clause_speedup,
        bench_to_sparse_conversion,
        bench_batch_prediction
);

criterion_main!(sparse_benches);
