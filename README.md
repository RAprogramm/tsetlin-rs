# tsetlin-rs

A professional Rust implementation of the Tsetlin Machine algorithm for interpretable machine learning.

## Features

- **Binary Classification** - `TsetlinMachine`
- **Multi-class Classification** - `MultiClass`
- **Regression** - `Regressor`
- **Convolutional** - `Convolutional` for image-like data
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

## Benchmarks

Measured on AMD Ryzen / Intel Core (single thread):

### Standard Clause (optimized scalar)

| Operation | Time |
|-----------|------|
| Clause evaluate (16 features) | 18 ns |
| Clause evaluate (64 features) | 71 ns |
| Clause evaluate (256 features) | 273 ns |
| Predict (10 clauses, 64 features) | 793 ns |

### BitwiseClause (64 features per AND)

| Features | Scalar | Bitwise | Speedup |
|----------|--------|---------|---------|
| 64 | 90 ns | **2.5 ns** | **36x** |
| 256 | 378 ns | **5.2 ns** | **73x** |
| 1024 | 1.64 µs | **13 ns** | **126x** |

### Optimizations Used

- `#[repr(align(64))]` - cache-line alignment
- `get_unchecked()` - bounds check elimination
- `#[inline(always)]` - forced inlining
- Pre-computed `1/(2*T)` - no division in hot path
- `BitwiseClause` - 64 features per CPU instruction
- `SmallClause<N>` - const generics for stack allocation

Run benchmarks: `cargo bench`

## Examples

```bash
cargo run --example xor
cargo run --example iris
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `std` (default) | Standard library support |
| `simd` | SIMD optimization (requires nightly) |
| `parallel` | Parallel training with rayon |
| `serde` | Serialization support |
| `full` | All features except simd |

## License

MIT
