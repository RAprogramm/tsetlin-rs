//! Benchmarks for Tsetlin Machine operations.

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tsetlin_rs::{
    BitPlaneBank, BitwiseClause, Clause, Clause16, ClauseBank, Config, MultiClass, SmallClause,
    TsetlinMachine, feedback, pack_input, utils::rng_from_seed
};

fn bench_clause_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("clause_evaluate");

    for n_features in [16, 64, 256, 1024] {
        let clause = Clause::new(n_features, 100, 1);
        let x: Vec<u8> = (0..n_features).map(|i| (i % 2) as u8).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(n_features),
            &n_features,
            |b, _| {
                b.iter(|| black_box(clause.evaluate(black_box(&x))));
            }
        );
    }

    group.finish();
}

fn bench_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict");

    for n_clauses in [10, 50, 100, 200] {
        let config = Config::builder()
            .clauses(n_clauses)
            .features(64)
            .build()
            .unwrap();
        let tm = TsetlinMachine::new(config, 15);
        let x: Vec<u8> = (0..64).map(|i| (i % 2) as u8).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(n_clauses),
            &n_clauses,
            |b, _| {
                b.iter(|| black_box(tm.predict(black_box(&x))));
            }
        );
    }

    group.finish();
}

fn bench_train_epoch(c: &mut Criterion) {
    let config = Config::builder().clauses(50).features(64).build().unwrap();

    let x: Vec<Vec<u8>> = (0..100)
        .map(|i| (0..64).map(|j| ((i + j) % 2) as u8).collect())
        .collect();
    let y: Vec<u8> = (0..100).map(|i| (i % 2) as u8).collect();

    c.bench_function("train_epoch_100_samples", |b| {
        b.iter(|| {
            let mut tm = TsetlinMachine::new(config, 15);
            tm.fit(black_box(&x), black_box(&y), 1, 42);
        });
    });
}

fn bench_multiclass_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiclass_predict");

    for n_classes in [3, 5, 10] {
        let config = Config::builder().clauses(50).features(64).build().unwrap();
        let tm = MultiClass::new(config, n_classes, 25);
        let x: Vec<u8> = (0..64).map(|i| (i % 2) as u8).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(n_classes),
            &n_classes,
            |b, _| {
                b.iter(|| black_box(tm.predict(black_box(&x))));
            }
        );
    }

    group.finish();
}

fn bench_feedback(c: &mut Criterion) {
    let mut group = c.benchmark_group("feedback");

    for n_features in [64, 256, 1024] {
        let x: Vec<u8> = (0..n_features).map(|i| (i % 2) as u8).collect();

        group.bench_with_input(
            BenchmarkId::new("type_i", n_features),
            &n_features,
            |b, &n| {
                b.iter(|| {
                    let mut clause = Clause::new(n, 100, 1);
                    let mut rng = rng_from_seed(42);
                    feedback::type_i(&mut clause, black_box(&x), true, 3.9, &mut rng);
                });
            }
        );

        group.bench_with_input(
            BenchmarkId::new("type_ii", n_features),
            &n_features,
            |b, &n| {
                b.iter(|| {
                    let mut clause = Clause::new(n, 100, 1);
                    feedback::type_ii(&mut clause, black_box(&x));
                });
            }
        );
    }

    group.finish();
}

fn bench_rule_extraction(c: &mut Criterion) {
    let config = Config::builder().clauses(100).features(64).build().unwrap();
    let mut tm = TsetlinMachine::new(config, 15);

    let x: Vec<Vec<u8>> = (0..100)
        .map(|i| (0..64).map(|j| ((i + j) % 2) as u8).collect())
        .collect();
    let y: Vec<u8> = (0..100).map(|i| (i % 2) as u8).collect();
    tm.fit(&x, &y, 50, 42);

    c.bench_function("rule_extraction_100_clauses", |b| {
        b.iter(|| black_box(tm.rules()));
    });
}

fn bench_small_clause(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_clause_vs_dynamic");

    // Dynamic Clause with 16 features
    let dyn_clause = Clause::new(16, 100, 1);
    let dyn_x: Vec<u8> = (0..16).map(|i| (i % 2) as u8).collect();

    group.bench_function("dynamic_16", |b| {
        b.iter(|| black_box(dyn_clause.evaluate(black_box(&dyn_x))));
    });

    // SmallClause with 16 features (const generic)
    let small_clause: Clause16 = SmallClause::new(100, 1);
    let small_x: [u8; 16] = core::array::from_fn(|i| (i % 2) as u8);

    group.bench_function("const_generic_16", |b| {
        b.iter(|| black_box(small_clause.evaluate(black_box(&small_x))));
    });

    group.finish();
}

fn bench_bitwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitwise_vs_scalar");

    for n_features in [64, 256, 1024] {
        let x: Vec<u8> = (0..n_features).map(|i| (i % 2) as u8).collect();
        let x_packed = pack_input(&x);

        // Scalar clause
        let scalar_clause = Clause::new(n_features, 100, 1);

        group.bench_with_input(
            BenchmarkId::new("scalar", n_features),
            &n_features,
            |b, _| {
                b.iter(|| black_box(scalar_clause.evaluate(black_box(&x))));
            }
        );

        // Bitwise clause
        let mut bitwise_clause = BitwiseClause::new(n_features, 100, 1);
        bitwise_clause.rebuild_masks();

        group.bench_with_input(
            BenchmarkId::new("bitwise", n_features),
            &n_features,
            |b, _| {
                b.iter(|| black_box(bitwise_clause.evaluate_packed(black_box(&x_packed))));
            }
        );
    }

    group.finish();
}

fn bench_aos_vs_soa(c: &mut Criterion) {
    let mut group = c.benchmark_group("aos_vs_soa");

    for n_clauses in [50, 100, 200] {
        let n_features = 64;
        let x: Vec<u8> = (0..n_features).map(|i| (i % 2) as u8).collect();

        // AoS: Vec<Clause>
        let aos: Vec<Clause> = (0..n_clauses)
            .map(|i| Clause::new(n_features, 100, if i % 2 == 0 { 1 } else { -1 }))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("aos_sum_votes", n_clauses),
            &n_clauses,
            |b, _| {
                b.iter(|| {
                    let sum: f32 = aos.iter().map(|c| c.vote(black_box(&x))).sum();
                    black_box(sum)
                });
            }
        );

        // SoA: ClauseBank
        let soa = ClauseBank::new(n_clauses, n_features, 100);

        group.bench_with_input(
            BenchmarkId::new("soa_sum_votes", n_clauses),
            &n_clauses,
            |b, _| {
                b.iter(|| black_box(soa.sum_votes(black_box(&x))));
            }
        );
    }

    group.finish();
}

fn bench_bitplane_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitplane_vs_clausebank_eval");

    for n_features in [64, 256, 1024] {
        let n_clauses = 100;
        let x: Vec<u8> = (0..n_features).map(|i| (i % 2) as u8).collect();

        // ClauseBank (SoA)
        let clausebank = ClauseBank::new(n_clauses, n_features, 100);

        group.bench_with_input(
            BenchmarkId::new("clausebank", n_features),
            &n_features,
            |b, _| {
                b.iter(|| black_box(clausebank.sum_votes(black_box(&x))));
            }
        );

        // BitPlaneBank
        let bitplane = BitPlaneBank::new(n_clauses, n_features, 100);

        group.bench_with_input(
            BenchmarkId::new("bitplane", n_features),
            &n_features,
            |b, _| {
                b.iter(|| black_box(bitplane.sum_votes(black_box(&x))));
            }
        );
    }

    group.finish();
}

fn bench_bitplane_feedback(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitplane_vs_clausebank_feedback");

    for n_features in [64, 256, 1024] {
        let x: Vec<u8> = (0..n_features).map(|i| (i % 2) as u8).collect();

        // ClauseBank Type I
        group.bench_with_input(
            BenchmarkId::new("clausebank_type_i", n_features),
            &n_features,
            |b, &n| {
                b.iter(|| {
                    let mut bank = ClauseBank::new(1, n, 100);
                    let mut rng = rng_from_seed(42);
                    bank.type_i(0, black_box(&x), true, 3.9, &mut rng);
                });
            }
        );

        // BitPlaneBank Type I
        group.bench_with_input(
            BenchmarkId::new("bitplane_type_i", n_features),
            &n_features,
            |b, &n| {
                b.iter(|| {
                    let mut bank = BitPlaneBank::new(1, n, 100);
                    let mut rng = rng_from_seed(42);
                    bank.type_i(0, black_box(&x), true, 3.9, &mut rng);
                });
            }
        );

        // ClauseBank Type II
        group.bench_with_input(
            BenchmarkId::new("clausebank_type_ii", n_features),
            &n_features,
            |b, &n| {
                b.iter(|| {
                    let mut bank = ClauseBank::new(1, n, 50);
                    bank.type_ii(0, black_box(&x));
                });
            }
        );

        // BitPlaneBank Type II
        group.bench_with_input(
            BenchmarkId::new("bitplane_type_ii", n_features),
            &n_features,
            |b, &n| {
                b.iter(|| {
                    let mut bank = BitPlaneBank::new(1, n, 50);
                    bank.type_ii(0, black_box(&x));
                });
            }
        );
    }

    group.finish();
}

fn bench_bitplane_increment(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitplane_parallel_increment");

    for n_features in [64, 256, 1024] {
        // Single increment (baseline)
        group.bench_with_input(
            BenchmarkId::new("single_64_increments", n_features),
            &n_features,
            |b, &n| {
                b.iter(|| {
                    let mut bank = BitPlaneBank::new(1, n, 100);
                    for i in 0..64 {
                        bank.increment(0, i);
                    }
                    black_box(&bank);
                });
            }
        );

        // Masked increment (parallel)
        group.bench_with_input(
            BenchmarkId::new("masked_64_increments", n_features),
            &n_features,
            |b, &n| {
                b.iter(|| {
                    let mut bank = BitPlaneBank::new(1, n, 100);
                    bank.increment_masked(0, 0, u64::MAX);
                    black_box(&bank);
                });
            }
        );
    }

    group.finish();
}

fn bench_train_sample_bitmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("train_sample_bitmap");

    for n_clauses in [50, 100, 200] {
        let n_features = 64;
        let x: Vec<u8> = (0..n_features).map(|i| (i % 2) as u8).collect();

        // ClauseBank with bitmap-based train_sample
        group.bench_with_input(
            BenchmarkId::new("train_sample", n_clauses),
            &n_clauses,
            |b, &n| {
                b.iter(|| {
                    let mut bank = ClauseBank::new(n, n_features, 100);
                    let mut rng = rng_from_seed(42);
                    bank.train_sample(black_box(&x), 1, 10.0, 3.9, &mut rng);
                    black_box(&bank);
                });
            }
        );
    }

    group.finish();
}

fn bench_evaluate_all_bitmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_all_bitmap");

    for n_clauses in [50, 100, 200] {
        let n_features = 64;
        let x: Vec<u8> = (0..n_features).map(|i| (i % 2) as u8).collect();

        // evaluate_all with bitmap (new)
        group.bench_with_input(
            BenchmarkId::new("evaluate_all", n_clauses),
            &n_clauses,
            |b, &n| {
                b.iter(|| {
                    let mut bank = ClauseBank::new(n, n_features, 100);
                    black_box(bank.evaluate_all(black_box(&x)));
                });
            }
        );

        // sum_votes_tracked (baseline)
        group.bench_with_input(
            BenchmarkId::new("sum_votes_tracked", n_clauses),
            &n_clauses,
            |b, &n| {
                b.iter(|| {
                    let mut bank = ClauseBank::new(n, n_features, 100);
                    black_box(bank.sum_votes_tracked(black_box(&x)));
                });
            }
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_clause_evaluate,
    bench_predict,
    bench_train_epoch,
    bench_multiclass_predict,
    bench_feedback,
    bench_rule_extraction,
    bench_small_clause,
    bench_bitwise,
    bench_aos_vs_soa,
    bench_bitplane_evaluate,
    bench_bitplane_feedback,
    bench_bitplane_increment,
    bench_train_sample_bitmap,
    bench_evaluate_all_bitmap
);
criterion_main!(benches);
