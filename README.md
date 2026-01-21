# tsetlin-rs

A professional Rust implementation of the Tsetlin Machine algorithm for interpretable machine learning.

## Features

- **Binary Classification** - `TsetlinMachine`
- **Multi-class Classification** - `MultiClass`
- **Regression** - `Regressor`
- **Convolutional** - `Convolutional` for image-like data
- **Advanced Training** - Weighted clauses, adaptive threshold, clause pruning
- **BitwiseClause** - 64 features per CPU instruction (25-92x speedup)
- **SIMD Optimization** - `simd` feature (nightly)
- **Parallel Training** - `parallel` feature (rayon)
- **Serialization** - `serde` feature
- **no_std Support** - Works without standard library

## Installation

```toml
[dependencies]
tsetlin-rs = "0.1"
```

With all features:
```toml
[dependencies]
tsetlin-rs = { version = "0.1", features = ["parallel", "serde"] }
```

## Quick Start

```rust
use tsetlin_rs::{Config, TsetlinMachine};

let config = Config::builder()
    .clauses(20)
    .features(2)
    .build()
    .unwrap();

let mut tm = TsetlinMachine::new(config, 15);

let x = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
let y = vec![0, 1, 1, 0];

tm.fit(&x, &y, 200, 42);

let accuracy = tm.evaluate(&x, &y);
println!("Accuracy: {:.1}%", accuracy * 100.0);
```

## Advanced Features

Enable weighted clauses, adaptive threshold, and clause pruning:

```rust
use tsetlin_rs::{Config, TsetlinMachine, AdvancedOptions};

let config = Config::builder().clauses(40).features(8).build().unwrap();

let opts = AdvancedOptions {
    adaptive_t: true,       // Dynamic threshold
    t_min: 5.0,
    t_max: 30.0,
    t_lr: 0.02,             // Threshold learning rate
    weight_lr: 0.08,        // Clause weight learning rate
    weight_min: 0.3,
    weight_max: 1.5,
    prune_threshold: 3,     // Min activations to keep clause
    prune_weight: 0.25,     // Min weight to keep clause
};

let mut tm = TsetlinMachine::with_advanced(config, 15, opts);
tm.fit(&x_train, &y_train, 100, 42);
```

### When to Use Advanced Features

| Scenario | Standard | Advanced | Recommendation |
|----------|----------|----------|----------------|
| Clean data (0% noise) | 97.5% | **100%** | Advanced |
| High noise (30%+) | 55.9% | **58.8%** | Advanced |
| Large scale (32+ features) | 76.4% | **77.7%** | Advanced |
| Complex patterns (parity) | 49.1% | **50.5%** | Advanced |
| Simple patterns, low noise | **82.8%** | 81.1% | Standard |

Run benchmark: `cargo run --release --example benchmark_advanced`

## Benchmarks

### BitwiseClause (fastest)

| Features | Scalar | Bitwise | Speedup |
|----------|--------|---------|---------|
| 64 | 75 ns | **3 ns** | **25x** |
| 256 | 300 ns | **4.8 ns** | **62x** |
| 1024 | 1.23 µs | **13 ns** | **92x** |

### Standard Clause

| Operation | Time |
|-----------|------|
| Clause evaluate (16 features) | 23 ns |
| Clause evaluate (64 features) | 100 ns |
| Predict (10 clauses) | 1.1 µs |
| Train epoch (100 samples) | 1.1 ms |

### Optimizations

- `#[repr(align(64))]` - cache-line alignment
- `get_unchecked()` - bounds check elimination
- `#[inline(always)]` - forced inlining
- `BitwiseClause` - 64 features per CPU instruction
- `SmallClause<N>` - const generics, stack allocation
- `div_ceil()` - Rust 1.73+ intrinsic

## Examples

```bash
cargo run --example xor
cargo run --example iris
cargo run --release --example benchmark_advanced
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `std` (default) | Standard library support |
| `simd` | SIMD optimization (requires nightly) |
| `parallel` | Parallel training with rayon |
| `serde` | Serialization support |

## API Overview

| Type | Description |
|------|-------------|
| `TsetlinMachine` | Binary classification |
| `MultiClass` | Multi-class classification |
| `Regressor` | Regression |
| `Convolutional` | 2D image classification |
| `BitwiseClause` | 64 features per AND operation |
| `SmallClause<N>` | Const-generic stack-allocated clause |
| `AdvancedOptions` | Weighted clauses, adaptive T, pruning |

## License

MIT
