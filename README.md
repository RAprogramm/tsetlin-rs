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
| `SmallTsetlinMachine<N, C>` | Compile-time optimized TM |
| `AdvancedOptions` | Weighted clauses, adaptive T, pruning |

---

<details>
<summary><strong>Tsetlin Machine Algorithm</strong></summary>

### How It Works

A Tsetlin Machine is a propositional logic-based machine learning algorithm. It learns patterns as conjunctions (AND) of literals.

**Core Components:**

1. **Tsetlin Automaton** - A finite state machine with 2N states:
   - States 1..N → action = `exclude` (literal not in clause)
   - States N+1..2N → action = `include` (literal in clause)
   - `increment()` → move toward include
   - `decrement()` → move toward exclude

2. **Clause** - A conjunction of literals:
   - Each feature `x_k` has two automata: one for `x_k`, one for `NOT x_k`
   - Clause fires when ALL included literals are satisfied
   - Has polarity (+1 or -1) for voting

3. **Voting** - Classification by majority:
   - Sum all clause votes: `Σ(polarity × fires)`
   - Positive sum → class 1, negative → class 0

**Training Feedback:**

| Type | When | Effect |
|------|------|--------|
| **Type I** | Correct class, clause should fire | Reinforce matching patterns |
| **Type II** | Wrong class, clause fires | Add literals to block false positive |

**Key Parameters:**

| Parameter | Symbol | Effect |
|-----------|--------|--------|
| Clauses | C | More clauses = more patterns, slower training |
| States | N | Higher = slower learning, more stability |
| Threshold | T | Controls learning probability |
| Specificity | s | Higher = more specific patterns (fewer literals) |

</details>

<details>
<summary><strong>Clause Types Comparison</strong></summary>

### Clause Implementations

This library provides three clause types optimized for different scenarios:

#### 1. `Clause` - Dynamic, Heap-Allocated

```rust
use tsetlin_rs::Clause;

let clause = Clause::new(n_features, n_states, polarity);
clause.evaluate(&input);  // &[u8]
```

**Memory Layout (64-byte aligned):**
```
Clause {
    automata:    Vec<Automaton>  // 24 bytes (ptr + len + cap)
    n_features:  usize           // 8 bytes
    weight:      f32             // 4 bytes
    activations: u32             // 4 bytes
    correct:     u32             // 4 bytes
    incorrect:   u32             // 4 bytes
    polarity:    i8              // 1 byte + 7 padding
}
// Total: 56 bytes + heap allocation for automata
```

**When to use:**
- Feature count unknown at compile time
- Large feature sets (1000+)
- Need serialization (serde)

---

#### 2. `SmallClause<N>` - Const Generic, Stack-Allocated

```rust
use tsetlin_rs::SmallClause;

let clause: SmallClause<4> = SmallClause::new(100, 1);
clause.evaluate(&[1, 0, 1, 0]);  // &[u8; N]
```

**Memory Layout:**
```
SmallClause<N> {
    include:     [Automaton; N]  // N × 4 bytes
    negated:     [Automaton; N]  // N × 4 bytes
    weight:      f32             // 4 bytes
    activations: u32             // 4 bytes
    correct:     u32             // 4 bytes
    incorrect:   u32             // 4 bytes
    polarity:    i8              // 1 byte
}
// Total: 8N + 17 bytes (no heap)
```

**When to use:**
- Feature count known at compile time
- Small to medium features (≤64)
- Maximum performance needed
- Embedded / no_std environments

**Type Aliases:**
```rust
type Clause2 = SmallClause<2>;   // XOR
type Clause4 = SmallClause<4>;   // Small problems
type Clause8 = SmallClause<8>;
type Clause16 = SmallClause<16>;
type Clause32 = SmallClause<32>;
type Clause64 = SmallClause<64>;
```

---

#### 3. `BitwiseClause` / `SmallBitwiseClause<N, W>` - Packed Bitmasks

```rust
use tsetlin_rs::{BitwiseClause, pack_input};

let mut clause = BitwiseClause::new(64, 100, 1);
// After training:
clause.rebuild_masks();
let packed = pack_input(&input);
clause.evaluate_packed(&packed);  // 64 features per AND
```

**How it works:**
1. Automata states → bitmasks (`include`, `negated`)
2. Input → packed u64 words
3. Evaluation: `(include & !x) | (negated & x) == 0`

**Performance:**
```
64 features:  1 u64 comparison  →  25x faster
256 features: 4 u64 comparisons →  62x faster
1024 features: 16 u64 comparisons → 92x faster
```

**Const Generic Version:**
```rust
use tsetlin_rs::{SmallBitwiseClause, pack_input_small};

// N=64 features, W=1 word
let mut clause: SmallBitwiseClause<64, 1> = SmallBitwiseClause::new(100, 1);
clause.rebuild_masks();
let packed: [u64; 1] = pack_input_small(&input);
clause.evaluate_packed(&packed);
```

**Type Aliases:**
```rust
type BitwiseClause64 = SmallBitwiseClause<64, 1>;
type BitwiseClause128 = SmallBitwiseClause<128, 2>;
type BitwiseClause256 = SmallBitwiseClause<256, 4>;
```

---

### Performance Comparison

| Type | 16 features | 64 features | 256 features | Heap |
|------|-------------|-------------|--------------|------|
| `Clause` | 23 ns | 100 ns | 300 ns | Yes |
| `SmallClause<N>` | 15 ns | 60 ns | 240 ns | No |
| `BitwiseClause` | 8 ns | 3 ns | 4.8 ns | Yes |
| `SmallBitwiseClause<N,W>` | 6 ns | 2.5 ns | 4 ns | No |

</details>

<details>
<summary><strong>SmallTsetlinMachine - Compile-Time Optimization</strong></summary>

### Stack-Allocated Tsetlin Machine

`SmallTsetlinMachine<N, C>` is a fully compile-time optimized binary classifier:

```rust
use tsetlin_rs::SmallTsetlinMachine;

// N=2 features, C=20 clauses
let mut tm: SmallTsetlinMachine<2, 20> = SmallTsetlinMachine::new(100, 15);

let x = [[0, 0], [0, 1], [1, 0], [1, 1]];
let y = [0u8, 1, 1, 0];

tm.fit(&x, &y, 200, 42);
assert!(tm.evaluate(&x, &y) >= 0.75);
```

**Benefits:**

| Feature | Dynamic TM | SmallTsetlinMachine |
|---------|------------|---------------------|
| Heap allocations | O(clauses) | 0 |
| Loop unrolling | None | Full |
| Cache locality | Fragmented | Contiguous |
| Compile-time checks | None | Type-level |
| Typical speedup | 1x | 2-3x |

**Type Aliases:**
```rust
type TM2x20 = SmallTsetlinMachine<2, 20>;    // XOR
type TM4x40 = SmallTsetlinMachine<4, 40>;    // Small
type TM8x80 = SmallTsetlinMachine<8, 80>;    // Medium
type TM16x160 = SmallTsetlinMachine<16, 160>; // Large
```

**Memory Layout:**
```rust
SmallTsetlinMachine<N, C> {
    clauses: [SmallClause<N>; C],  // C × (8N + 17) bytes
    s:       f32,                   // 4 bytes
    t:       f32,                   // 4 bytes
}
// Example: TM2x20 = 20 × 33 + 8 = 668 bytes (stack)
// Compare: TsetlinMachine = ~2KB + heap
```

**When to use:**
- Known dimensions at compile time
- Performance-critical inner loops
- Embedded systems
- Batch inference with fixed input size

**Limitations:**
- No serde (const generic arrays)
- Dimensions must be literals or const
- Large N×C may overflow stack

</details>

<details>
<summary><strong>Choosing the Right Implementation</strong></summary>

### Decision Guide

```
                    ┌─────────────────────────┐
                    │ Feature count known at  │
                    │     compile time?       │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                   YES                      NO
                    │                       │
            ┌───────┴───────┐               │
            │ N ≤ 64?       │               ▼
            └───────┬───────┘       ┌───────────────┐
                    │               │    Clause     │
            ┌───────┴───────┐       │ (dynamic Vec) │
            │               │       └───────────────┘
           YES              NO
            │               │
            ▼               ▼
    ┌───────────────┐ ┌─────────────────────┐
    │ SmallClause<N>│ │SmallBitwiseClause<N>│
    │ (fastest)     │ │ (N > 64, bitwise)   │
    └───────────────┘ └─────────────────────┘
```

### Recommendations by Use Case

| Use Case | Recommended Type |
|----------|------------------|
| XOR / toy problems | `SmallTsetlinMachine<2, 20>` |
| Iris (4 features) | `SmallTsetlinMachine<4, 100>` |
| MNIST patches (64) | `SmallBitwiseClause<64, 1>` |
| Text (10K features) | `BitwiseClause` |
| Unknown dimensions | `TsetlinMachine` |
| Serialization needed | `TsetlinMachine` |
| Embedded / no_std | `SmallClause<N>` |

### Feature Comparison

| Feature | Clause | SmallClause | BitwiseClause | SmallBitwise |
|---------|--------|-------------|---------------|--------------|
| Heap-free | ❌ | ✅ | ❌ | ✅ |
| Serde | ✅ | ❌ | ✅ | ❌ |
| no_std | ✅ | ✅ | ✅ | ✅ |
| Best for N | any | ≤64 | ≥64 | 64-256 |
| Loop unroll | ❌ | ✅ | ❌ | ✅ |

</details>

<details>
<summary><strong>Memory Layout & Alignment</strong></summary>

### Cache Optimization

All clause types use `#[repr(align(64))]` for cache-line alignment:

```rust
#[repr(align(64))]  // 64-byte cache line
pub struct Clause { ... }

#[repr(align(64))]
pub struct SmallClause<const N: usize> { ... }

#[repr(align(64))]
pub struct BitwiseClause { ... }
```

**Why 64-byte alignment?**
- Modern CPUs fetch 64 bytes per cache line
- Aligned structs avoid cache line splits
- Reduces memory latency by 10-30%

### Automaton Representation

```rust
pub struct Automaton {
    state:    i16,  // Current state (1..2N)
    n_states: i16,  // Threshold for action
}
// 4 bytes per automaton
```

**State machine:**
```
exclude zone          include zone
  [1]...[N]    |    [N+1]...[2N]
     ←──────   |   ──────→
  decrement    |   increment
```

### Packed Bitmask Layout

```rust
// BitwiseClause for 128 features:
include: [u64; 2]  // 16 bytes
negated: [u64; 2]  // 16 bytes

// Bit k of word[k/64] represents feature k
// bit 0 of word[0] = feature 0
// bit 63 of word[0] = feature 63
// bit 0 of word[1] = feature 64
// ...
```

### Memory Usage Examples

| Type | N=4 | N=64 | N=256 |
|------|-----|------|-------|
| Clause | 88 + 32 heap | 88 + 512 heap | 88 + 2KB heap |
| SmallClause | 49 bytes | 529 bytes | 2KB |
| BitwiseClause | 88 + 32 heap | 88 + 512 heap | 88 + 2KB heap |
| SmallBitwiseClause | 65 bytes | 545 bytes | 2KB |

</details>

## Based On

This Rust implementation is based on the original Tsetlin Machine algorithm:

**Paper:** [The Tsetlin Machine - A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic](https://arxiv.org/abs/1804.01508)
*Ole-Christoffer Granmo, 2018*

**Original C Implementation:** [cair/TsetlinMachineC](https://github.com/cair/TsetlinMachineC)

**Related Resources:**
- [cair/TsetlinMachine](https://github.com/cair/TsetlinMachine) — Python implementation with datasets
- [cair/pyTsetlinMachine](https://github.com/cair/pyTsetlinMachine) — Extended Python library
- [cair/tmu](https://github.com/cair/tmu) — Unified TM with CUDA support

## License

MIT
