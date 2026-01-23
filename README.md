<a name="top"></a>

# tsetlin-rs

<p align="center">
  <a href="https://crates.io/crates/tsetlin-rs"><img src="https://img.shields.io/crates/v/tsetlin-rs?style=for-the-badge&logo=rust&logoColor=white&label=crates.io&color=e6522c" alt="Crates.io"/></a>
  <a href="https://docs.rs/tsetlin-rs"><img src="https://img.shields.io/docsrs/tsetlin-rs?style=for-the-badge&logo=docsdotrs&logoColor=white&label=docs.rs&color=blue" alt="docs.rs"/></a>
  <a href="https://github.com/RAprogramm/tsetlin-rs/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/RAprogramm/tsetlin-rs/ci.yml?style=for-the-badge&logo=githubactions&logoColor=white&label=CI" alt="CI"/></a>
</p>
<p align="center">
  <a href="https://codecov.io/gh/RAprogramm/tsetlin-rs"><img src="https://img.shields.io/codecov/c/github/RAprogramm/tsetlin-rs?style=for-the-badge&logo=codecov&logoColor=white&color=f01f7a" alt="codecov"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="License"/></a>
  <a href="https://api.reuse.software/info/github.com/RAprogramm/tsetlin-rs"><img src="https://img.shields.io/badge/REUSE-compliant-4cc61e?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bS0yIDE1bC01LTUgMS40MS0xLjQxTDEwIDE0LjE3bDcuNTktNy41OUwxOSA4bC05IDl6Ii8+PC9zdmc+" alt="REUSE"/></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-1.92+-93450a?style=for-the-badge&logo=rust&logoColor=white" alt="Rust"/></a>
</p>

<p align="center">
  <strong>A production-grade Rust implementation of the Tsetlin Machine algorithm for interpretable machine learning.</strong>
</p>

<p align="center">
  <em>Lock-free parallel training | 116x bitwise speedup | Zero-allocation inference | Full interpretability</em>
</p>

---

## Highlights

```
Performance:  4.4x parallel speedup  |  116x bitwise evaluation  |  O(1) inference
Memory:       Zero-allocation SmallClause  |  Cache-aligned structures  |  no_std support
Correctness:  99%+ test coverage  |  Property-based testing  |  Deterministic seeds
```

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models](#models)
- [Parallel Training](#parallel-training)
- [Clause Implementations](#clause-implementations)
- [Advanced Features](#advanced-features)
- [Benchmarks](#benchmarks)
- [Algorithm Reference](#algorithm-reference)
- [API Reference](#api-reference)
- [Coverage](#coverage)
- [In Memory of Michael Tsetlin](#in-memory-of-michael-tsetlin)
- [References](#references)
- [License](#license)

---

## Overview

The Tsetlin Machine is a machine learning algorithm based on propositional logic and game theory. Unlike neural networks, it learns human-readable rules (conjunctions of literals) that can be directly interpreted and verified.

**Key Properties:**

| Property | Tsetlin Machine | Neural Network |
|----------|-----------------|----------------|
| Interpretability | Rules in propositional logic | Black box |
| Training | Reinforcement learning | Gradient descent |
| Inference | Boolean operations | Matrix multiplication |
| Hardware | FPGA/ASIC friendly | GPU optimized |
| Memory | O(clauses × features) | O(layers × neurons²) |

<details>
<summary><strong>Terminology & Abbreviations</strong></summary>

<br/>

| Term | Definition | Category |
|:-----|:-----------|:--------:|
| **AL** | Active Literals — literals that actively contribute to predictions | `sparse` |
| **AoS** | Array of Structures — traditional object layout | `opt` |
| **Bit-plane** | Transposed bit representation for parallel operations | `opt` |
| **Clause** | Conjunction (AND) of literals; votes for/against a class | `core` |
| **CoTM** | Coalesced Tsetlin Machine | `abbr` |
| **CSR** | Compressed Sparse Row — sparse matrix format using data/indices/offsets arrays | `sparse` |
| **CTM** | Convolutional Tsetlin Machine | `abbr` |
| **Early exit** | Terminating clause evaluation on first literal violation | `opt` |
| **False sharing** | Cache line contention between CPU cores | `opt` |
| **FPGA** | Field-Programmable Gate Array | `abbr` |
| **Literal** | Boolean variable (`xₖ`) or its negation (`¬xₖ`) | `core` |
| **MSB** | Most Significant Bit — encodes automaton action | `opt` |
| **Polarity** | Clause vote direction: +1 or −1 | `core` |
| **Ripple-carry** | Bit-level addition/subtraction algorithm | `opt` |
| **RNG** | Random Number Generator | `abbr` |
| **SIMD** | Single Instruction Multiple Data | `abbr` |
| **SmallVec** | Inline vector — stack storage up to N elements, heap beyond | `opt` |
| **SoA** | Structure of Arrays — cache-friendly layout | `opt` |
| **Sparsity** | Fraction of active literals vs total possible (lower = sparser) | `sparse` |
| **Specificity (s)** | Controls pattern generality; higher = fewer literals | `train` |
| **STM** | Sparse Tsetlin Machine | `abbr` |
| **TA** | Tsetlin Automaton | `abbr` |
| **Threshold (T)** | Controls feedback probability | `train` |
| **TM** | Tsetlin Machine | `abbr` |

<sub>`core` — fundamentals · `train` — training · `opt` — optimization · `sparse` — sparse representation · `abbr` — abbreviation</sub>

</details>

<div align="right"><a href="#top">Back to top</a></div>

---

## Installation

```toml
[dependencies]
tsetlin-rs = "0.2"
```

With parallel training and serialization:

```toml
[dependencies]
tsetlin-rs = { version = "0.2", features = ["parallel", "serde"] }
```

### Feature Flags

| Feature | Default | Description |
|---------|:-------:|-------------|
| `std` | Yes | Standard library (disable for embedded) |
| `parallel` | No | Lock-free parallel training via rayon |
| `serde` | No | Serialization/deserialization |
| `simd` | No | SIMD optimization (requires nightly) |

<div align="right"><a href="#top">Back to top</a></div>

---

## Quick Start

```rust
use tsetlin_rs::{Config, TsetlinMachine};

// Configure: 20 clauses, 2 features
let config = Config::builder()
    .clauses(20)
    .features(2)
    .build()
    .unwrap();

// Create machine with threshold T=15
let mut tm = TsetlinMachine::new(config, 15);

// XOR dataset
let x = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
let y = vec![0, 1, 1, 0];

// Train for 200 epochs with seed=42
tm.fit(&x, &y, 200, 42);

// Evaluate
let accuracy = tm.evaluate(&x, &y);
println!("Accuracy: {:.1}%", accuracy * 100.0);

// Extract learned rules
for rule in tm.rules() {
    println!("{}", rule);
}
```

**Output:**
```
Accuracy: 100.0%
+: x₀ ∧ ¬x₁
+: ¬x₀ ∧ x₁
-: x₀ ∧ x₁
-: ¬x₀ ∧ ¬x₁
```

<div align="right"><a href="#top">Back to top</a></div>

---

## Models

### Binary Classification — `TsetlinMachine`

Standard two-class classification with weighted clause voting.

```rust
use tsetlin_rs::{Config, TsetlinMachine};

let config = Config::builder().clauses(100).features(64).build().unwrap();
let mut tm = TsetlinMachine::new(config, 15);

tm.fit(&x_train, &y_train, 100, 42);
let prediction = tm.predict(&x_test[0]);  // 0 or 1
```

### Multi-class Classification — `MultiClass`

One-vs-all ensemble of binary classifiers.

```rust
use tsetlin_rs::{Config, MultiClass};

let config = Config::builder().clauses(100).features(64).build().unwrap();
let mut tm = MultiClass::new(config, 10, 15);  // 10 classes

tm.fit(&x_train, &y_train, 100, 42);
let class = tm.predict(&x_test[0]);  // 0..9
```

### Regression — `Regressor`

Continuous output via clause voting with binning.

```rust
use tsetlin_rs::{Config, Regressor};

let config = Config::builder().clauses(100).features(64).build().unwrap();
let mut reg = Regressor::new(config, 15);

reg.fit(&x_train, &y_train, 100, 42);
let value = reg.predict(&x_test[0]);  // f32
```

### Convolutional — `Convolutional`

2D patch extraction for image-like data.

```rust
use tsetlin_rs::{ConvConfig, Convolutional};

let config = ConvConfig {
    clauses: 100,
    image_height: 28,
    image_width: 28,
    patch_height: 10,
    patch_width: 10,
    n_classes: 10,
};
let mut ctm = Convolutional::new(config, 15);
```

### Sparse Inference — `SparseTsetlinMachine`

Memory-efficient inference using sparse clause representation. Convert trained model for deployment with 50-125x memory reduction.

```rust
use tsetlin_rs::{Config, TsetlinMachine};

// Train as usual
let config = Config::builder().clauses(200).features(784).build().unwrap();
let mut tm = TsetlinMachine::new(config, 15);
tm.fit(&x_train, &y_train, 100, 42);

// Convert to sparse for deployment
let sparse = tm.to_sparse();

// Same predictions, much less memory
assert_eq!(tm.predict(&x_test[0]), sparse.predict(&x_test[0]));

// Check compression ratio
println!("Compression: {:.1}x", sparse.compression_ratio());
```

**Sparse Representation:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                    DENSE (ClauseBank)                                │
├─────────────────────────────────────────────────────────────────────┤
│  Clause 0: [TA₀, TA₁, TA₂, ..., TA₂ₙ₋₁]  ← stores ALL 2N automata   │
│  Clause 1: [TA₀, TA₁, TA₂, ..., TA₂ₙ₋₁]                             │
│  ...                                                                 │
│  Memory: O(clauses × 2 × features × sizeof(i16))                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    SPARSE (SparseClauseBank) — CSR Format           │
├─────────────────────────────────────────────────────────────────────┤
│  include_indices: [0, 5, 12 | 3, 7 | ...]   ← only active literals  │
│  include_offsets: [0, 3, 5, ...]            ← clause boundaries     │
│  negated_indices: [2, 8 | 1, 4, 9 | ...]                            │
│  negated_offsets: [0, 2, 5, ...]                                    │
│  weights: [1.0, 0.8, ...]                                           │
│  polarities: [+1, -1, ...]                                          │
│  Memory: O(total_active_literals × sizeof(u16))                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Benchmark Results (realistic data):**

| Dataset | Features | Compression | Speedup |
|---------|----------|-------------|---------|
| MNIST-like | 784 | **117.6x** | **65x** |
| NLP 5k vocab | 5,000 | **59.3x** | **38x** |
| NLP 10k vocab | 10,000 | **124.8x** | **75x** |

**When to use:**
- Deployment on memory-constrained devices
- High-dimensional sparse data (NLP, bag-of-words)
- Batch inference where memory bandwidth matters

<div align="right"><a href="#top">Back to top</a></div>

---

## Parallel Training

This implementation provides **lock-free parallel training** based on the [Massively Parallel and Asynchronous Tsetlin Machine Architecture (ICML 2021)](https://arxiv.org/abs/2009.04861).

### The Problem

Traditional TM training requires synchronization barriers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                     TRADITIONAL (SYNCHRONOUS)                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [Clause 0] ──┐                                                        │
│  [Clause 1] ──┼──► BARRIER ──► Sum Votes ──► BARRIER ──► Feedback      │
│  [Clause 2] ──┤        ▲                          ▲                    │
│      ...      │        │                          │                    │
│  [Clause N] ──┘    wait all                   wait all                 │
│                                                                        │
│  Problem: Threads idle during barriers = poor scaling                  │
└────────────────────────────────────────────────────────────────────────┘
```

### The Solution: Async Local Voting Tallies

Each training sample maintains its own **atomic vote accumulator**. Clauses update tallies via `fetch_add` without synchronization.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PARALLEL V1 (ASYNC TALLIES)                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Sample 0: LocalTally ─────────────────────────────────────────────┐   │
│  Sample 1: LocalTally ─────────────────────────────────────────────┤   │
│  Sample N: LocalTally ─────────────────────────────────────────────┘   │
│                 ▲                                                      │
│                 │ atomic fetch_add (no locks)                          │
│                 │                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Thread 0   │  Thread 1   │  Thread 2   │   ...   │  Thread M   │   │
│  │ [Sample 0]  │ [Sample 1]  │ [Sample 2]  │         │ [Sample N]  │   │
│  │  eval all   │  eval all   │  eval all   │         │  eval all   │   │
│  │  clauses    │  clauses    │  clauses    │         │  clauses    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Speedup: 1.5x (evaluation parallel, feedback sequential)              │
└────────────────────────────────────────────────────────────────────────┘
```

### Parallel V2: Fully Parallel Feedback

V2 parallelizes the feedback phase by processing **clauses** instead of samples:

```
┌────────────────────────────────────────────────────────────────────────┐
│                     PARALLEL V2 (FULLY PARALLEL)                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Key insight: Each clause's automata occupy DISJOINT memory regions    │
│                                                                        │
│  states: [clause_0_automata | clause_1_automata | ... | clause_N]      │
│           ▲                   ▲                         ▲              │
│           │                   │                         │              │
│        Thread 0            Thread 1                  Thread N          │
│      (no conflict)       (no conflict)             (no conflict)       │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Clause 0        │  Clause 1        │   ...   │  Clause N       │   │
│  │  ├─ sample 0     │  ├─ sample 0     │         │  ├─ sample 0    │   │
│  │  ├─ sample 1     │  ├─ sample 1     │         │  ├─ sample 1    │   │
│  │  └─ sample N     │  └─ sample N     │         │  └─ sample N    │   │
│  │  [feedback all]  │  [feedback all]  │         │  [feedback all] │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Speedup: 4.4x (both evaluation AND feedback parallel)                 │
└────────────────────────────────────────────────────────────────────────┘
```

### Usage

```rust
use tsetlin_rs::{ClauseBank, ParallelBatch};

// Create clause bank and batch
let mut bank = ClauseBank::new(100, 64, 100);  // 100 clauses, 64 features
let batch = ParallelBatch::new(&x_train, &y_train);

for epoch in 0..100 {
    // V1: Parallel evaluation, sequential feedback (1.5x speedup)
    bank.train_parallel(&batch, 15.0, 3.9, epoch as u64);

    // V2: Fully parallel (4.4x speedup) — RECOMMENDED
    bank.train_parallel_v2(&batch, 15.0, 3.9, epoch as u64);
}

// Inference
let vote_sum = bank.sum_votes(&x_test[0]);
let prediction = if vote_sum >= 0.0 { 1 } else { 0 };
```

### Implementation Details

#### LocalTally — Atomic Vote Accumulator

```rust
pub struct LocalTally {
    tally: CachePadded<AtomicI64>  // 128 bytes (cache line padded)
}
```

- **CachePadded**: Prevents false sharing between CPU cores
- **AtomicI64**: Lock-free concurrent updates
- **Scaled integers**: `f32` weights → `i64 * 10000` for atomics

```rust
const WEIGHT_SCALE: i64 = 10_000;  // 4 decimal precision

// Accumulate vote (thread-safe)
tally.fetch_add((polarity * weight * SCALE) as i64, Ordering::Relaxed);

// Read final sum
let sum = tally.load(Ordering::Acquire) as f32 / SCALE as f32;
```

#### ParallelBatch — Training Data Container

```rust
pub struct ParallelBatch<'a> {
    pub inputs: &'a [Vec<u8>],   // Feature vectors
    pub labels: &'a [u8],        // Target labels
    tallies: Vec<LocalTally>,    // One per sample (cache-aligned)
}
```

#### Memory Ordering

| Operation | Ordering | Rationale |
|-----------|----------|-----------|
| `tally.fetch_add()` | `Relaxed` | Addition is commutative |
| `tally.load()` | `Acquire` | Synchronize-with all prior stores |
| `tally.store(0)` | `Relaxed` | Reset before new epoch |

### Performance Comparison

100 clauses, 64 features, 1 epoch:

| Samples | Sequential | V1 (par eval) | V2 (fully par) | Speedup |
|---------|------------|---------------|----------------|---------|
| 100 | 2.93 ms | 2.00 ms | **0.81 ms** | **3.6x** |
| 500 | 14.96 ms | 10.11 ms | **3.39 ms** | **4.4x** |
| 1000 | 29.15 ms | 19.92 ms | **6.58 ms** | **4.4x** |

Original CUDA implementation reports **50x speedup** on GPU.

<div align="right"><a href="#top">Back to top</a></div>

---

## Clause Implementations

This library provides four clause types optimized for different scenarios:

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
            │   N ≤ 64?     │               ▼
            └───────┬───────┘       ┌───────────────┐
                    │               │    Clause     │
            ┌───────┴───────┐       │ (heap, serde) │
            │               │       └───────────────┘
           YES              NO
            │               │
            ▼               ▼
    ┌───────────────┐ ┌─────────────────────┐
    │ SmallClause<N>│ │SmallBitwiseClause   │
    │ (fastest, no  │ │<N,W> (bitwise ops,  │
    │  heap)        │ │ no heap)            │
    └───────────────┘ └─────────────────────┘
```

### 1. `Clause` — Dynamic, Heap-Allocated

```rust
use tsetlin_rs::Clause;

let clause = Clause::new(64, 100, 1);  // 64 features, 100 states, +1 polarity
let fires = clause.evaluate(&input);
let vote = clause.vote(&input);  // polarity × fires × weight
```

**Memory Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Clause (64-byte aligned)                 │
├─────────────────────────────────────────────────────────────┤
│ automata: Vec<Automaton>  │ 24 bytes (ptr + len + cap)      │
│ n_features: usize         │ 8 bytes                         │
│ weight: f32               │ 4 bytes                         │
│ activations: u32          │ 4 bytes                         │
│ correct: u32              │ 4 bytes                         │
│ incorrect: u32            │ 4 bytes                         │
│ polarity: i8              │ 1 byte + 7 padding              │
├─────────────────────────────────────────────────────────────┤
│ + heap: [Automaton; 2N]   │ 8N bytes                        │
└─────────────────────────────────────────────────────────────┘
```

**Best for:** Unknown dimensions, serialization, large feature sets (1000+)

---

### 2. `SmallClause<N>` — Const Generic, Stack-Allocated

```rust
use tsetlin_rs::{SmallClause, Clause16};

let clause: Clause16 = SmallClause::new(100, 1);
let fires = clause.evaluate(&[1, 0, 1, 0, ...]);  // [u8; 16]
```

**Memory Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│              SmallClause<N> (stack-allocated)               │
├─────────────────────────────────────────────────────────────┤
│ include: [Automaton; N]   │ 4N bytes                        │
│ negated: [Automaton; N]   │ 4N bytes                        │
│ weight: f32               │ 4 bytes                         │
│ activations: u32          │ 4 bytes                         │
│ correct: u32              │ 4 bytes                         │
│ incorrect: u32            │ 4 bytes                         │
│ polarity: i8              │ 1 byte                          │
├─────────────────────────────────────────────────────────────┤
│ Total: 8N + 17 bytes (NO HEAP)                              │
└─────────────────────────────────────────────────────────────┘
```

**Type Aliases:**
```rust
type Clause2  = SmallClause<2>;   // XOR problems
type Clause4  = SmallClause<4>;   // Iris dataset
type Clause8  = SmallClause<8>;
type Clause16 = SmallClause<16>;
type Clause32 = SmallClause<32>;
type Clause64 = SmallClause<64>;  // Maximum before bitwise
```

**Best for:** Known dimensions, embedded/no_std, maximum performance

---

### 3. `BitwiseClause` — Packed Bitmask Evaluation

Evaluates 64 features per CPU instruction using bitwise AND.

```rust
use tsetlin_rs::{BitwiseClause, pack_input};

let mut clause = BitwiseClause::new(256, 100, 1);
// ... training ...
clause.rebuild_masks();  // Compile automata → bitmasks

let packed = pack_input(&input);  // Vec<u64>
let fires = clause.evaluate_packed(&packed);
```

**Evaluation Algorithm:**
```
Input:   x = [1,0,1,1,0,0,1,0, ...]  →  packed: 0b01001101...
Include mask: 0b00001101  (positions where x_k must be 1)
Negated mask: 0b00100010  (positions where x_k must be 0)

Clause fires iff: (include & !x) | (negated & x) == 0

Example:
  include = 0b00001101
  negated = 0b00100010
  x       = 0b01001101

  include & !x = 0b00001101 & 0b10110010 = 0b00000000
  negated & x  = 0b00100010 & 0b01001101 = 0b00000000
  result = 0b00000000 → fires = true
```

**Performance:**
```
64 features:   1 u64 AND    →   43x faster than scalar
256 features:  4 u64 ANDs   →   85x faster
1024 features: 16 u64 ANDs  →  116x faster
```

---

### 4. `SmallBitwiseClause<N, W>` — Const Generic Bitwise

```rust
use tsetlin_rs::{SmallBitwiseClause, BitwiseClause64, pack_input_small};

let mut clause: BitwiseClause64 = SmallBitwiseClause::new(100, 1);
clause.rebuild_masks();

let packed: [u64; 1] = pack_input_small(&input);
let fires = clause.evaluate_packed(&packed);
```

**Type Aliases:**
```rust
type BitwiseClause64  = SmallBitwiseClause<64, 1>;   // 1 word
type BitwiseClause128 = SmallBitwiseClause<128, 2>;  // 2 words
type BitwiseClause256 = SmallBitwiseClause<256, 4>;  // 4 words
```

---

### Performance Comparison

| Type | 16 features | 64 features | 256 features | Heap |
|------|-------------|-------------|--------------|:----:|
| `Clause` | 23 ns | 100 ns | 300 ns | Yes |
| `SmallClause<N>` | 15 ns | 60 ns | 240 ns | No |
| `BitwiseClause` | 8 ns | **3 ns** | **4.8 ns** | Yes |
| `SmallBitwiseClause<N,W>` | 6 ns | **2.5 ns** | **4 ns** | No |

<div align="right"><a href="#top">Back to top</a></div>

---

## Advanced Features

### Weighted Clauses

Clauses learn accuracy-based weights during training:

```rust
use tsetlin_rs::{Config, TsetlinMachine, AdvancedOptions};

let opts = AdvancedOptions {
    weight_lr: 0.08,    // Learning rate for weight updates
    weight_min: 0.3,    // Minimum weight (prevents dead clauses)
    weight_max: 1.5,    // Maximum weight (prevents dominance)
    ..Default::default()
};

let config = Config::builder().clauses(100).features(64).build().unwrap();
let mut tm = TsetlinMachine::with_advanced(config, 15, opts);
```

**Weight Update Rule:**
```
After each prediction:
  if clause_correct:
    weight += weight_lr × (weight_max - weight)
  else:
    weight -= weight_lr × (weight - weight_min)
```

---

### Adaptive Threshold

Dynamic T adjustment based on training error:

```rust
let opts = AdvancedOptions {
    adaptive_t: true,
    t_min: 5.0,         // Lower bound
    t_max: 30.0,        // Upper bound
    t_lr: 0.02,         // Learning rate
    ..Default::default()
};
```

**Adaptation Rule:**
```
if accuracy < target:
    T += t_lr × (t_max - T)   // Increase → more aggressive learning
else:
    T -= t_lr × (T - t_min)   // Decrease → more conservative
```

---

### Clause Pruning

Automatic reset of ineffective clauses:

```rust
let opts = AdvancedOptions {
    prune_threshold: 3,   // Min activations per epoch
    prune_weight: 0.25,   // Min weight to survive
    ..Default::default()
};
```

**Pruning Criteria:**
```
if clause.activations < prune_threshold OR clause.weight < prune_weight:
    reset_clause_to_initial_state()
```

---

### Complete Example

```rust
use tsetlin_rs::{Config, TsetlinMachine, AdvancedOptions};

let opts = AdvancedOptions {
    // Adaptive threshold
    adaptive_t: true,
    t_min: 5.0,
    t_max: 30.0,
    t_lr: 0.02,

    // Weighted clauses
    weight_lr: 0.08,
    weight_min: 0.3,
    weight_max: 1.5,

    // Clause pruning
    prune_threshold: 3,
    prune_weight: 0.25,
};

let config = Config::builder().clauses(40).features(8).build().unwrap();
let mut tm = TsetlinMachine::with_advanced(config, 15, opts);

tm.fit(&x_train, &y_train, 100, 42);
```

### When to Use Advanced Features

| Scenario | Standard | Advanced | Winner |
|----------|:--------:|:--------:|:------:|
| Clean data (0% noise) | 97.5% | **100%** | Advanced |
| High noise (30%+) | 55.9% | **58.8%** | Advanced |
| Large scale (32+ features) | 76.4% | **77.7%** | Advanced |
| Complex patterns (parity) | 49.1% | **50.5%** | Advanced |
| Simple patterns, low noise | **82.8%** | 81.1% | Standard |

<div align="right"><a href="#top">Back to top</a></div>

---

## Benchmarks

### Parallel Training

100 clauses, 64 features, 1 epoch:

| Samples | Sequential | Parallel v1 | Parallel v2 | Speedup |
|---------|------------|-------------|-------------|---------|
| 100 | 2.93 ms | 2.00 ms | **0.81 ms** | **3.6x** |
| 500 | 14.96 ms | 10.11 ms | **3.39 ms** | **4.4x** |
| 1000 | 29.15 ms | 19.92 ms | **6.58 ms** | **4.4x** |

### Bitwise Evaluation

| Features | Scalar | Bitwise | Speedup |
|----------|--------|---------|---------|
| 64 | 82 ns | **1.9 ns** | **43x** |
| 256 | 342 ns | **4 ns** | **85x** |
| 1024 | 1.32 µs | **11 ns** | **116x** |

### Storage Layouts

| Clauses | AoS (Vec<Clause>) | SoA (ClauseBank) | Speedup |
|---------|-------------------|------------------|---------|
| 50 | 3.64 µs | **3.50 µs** | 1.04x |
| 100 | 8.55 µs | **8.39 µs** | 1.02x |
| 200 | 18.50 µs | **16.02 µs** | **1.15x** |

### BitPlaneBank (Parallel State Updates)

| Operation | ClauseBank | BitPlaneBank | Speedup |
|-----------|------------|--------------|---------|
| Type II (1024 features) | 1.48 µs | **759 ns** | **~2x** |
| Type II (256 features) | 426 ns | **330 ns** | **~1.3x** |
| Type I (1024 features) | 5.05 µs | 5.25 µs | ~1x |

### Run Benchmarks

```bash
cargo bench --features parallel
cargo bench --features parallel -- parallel_vs_sequential
cargo bench -- bitwise
```

<div align="right"><a href="#top">Back to top</a></div>

---

## Algorithm Reference

<details>
<summary><strong>Tsetlin Automaton</strong></summary>

### Finite State Machine

A Tsetlin Automaton (TA) is a 2N-state FSM that learns a binary action through reinforcement:

```
         EXCLUDE ZONE              │              INCLUDE ZONE
    ◄─────────────────────────────│───────────────────────────────►
                                   │
    ┌───┬───┬───┬───┬─────────────┼───┬───────────┬───┬───┬───┐
    │ 1 │ 2 │ 3 │...│      N      │N+1│    ...    │...│2N-1│2N │
    └───┴───┴───┴───┴─────────────┼───┴───────────┴───┴───┴───┘
      ▲                           │                           ▲
      │                           │                           │
    floor                     threshold                    ceiling
    (min)                                                   (max)

    Action = EXCLUDE              │              Action = INCLUDE
    if state ∈ [1, N]             │              if state ∈ [N+1, 2N]
```

**State Transitions:**

| Feedback | Current State | Action |
|----------|---------------|--------|
| Reward | s ∈ [1, N] | s → max(1, s-1) |
| Reward | s ∈ [N+1, 2N] | s → min(2N, s+1) |
| Penalty | s ∈ [1, N] | s → min(N, s+1) |
| Penalty | s ∈ [N+1, 2N] | s → max(N+1, s-1) |

</details>

<details>
<summary><strong>Clause Structure</strong></summary>

### Conjunction of Literals

A clause is a conjunction (AND) of literals that votes for or against a class:

```
Clause_j = L₁ ∧ L₂ ∧ ... ∧ Lₘ

where each Lᵢ ∈ {xₖ, ¬xₖ} for some feature k
```

**Example (XOR):**
```
Clause₊₁: x₀ ∧ ¬x₁     (fires when x=[1,0])
Clause₊₂: ¬x₀ ∧ x₁     (fires when x=[1,0])
Clause₋₁: x₀ ∧ x₁      (fires when x=[1,1])
Clause₋₂: ¬x₀ ∧ ¬x₁    (fires when x=[0,0])
```

**Evaluation:**
```rust
fn evaluate(clause: &Clause, x: &[u8]) -> bool {
    for k in 0..n_features {
        // Check positive literal x_k
        if clause.include[k].action() == INCLUDE && x[k] == 0 {
            return false;  // Violation
        }
        // Check negative literal ¬x_k
        if clause.negated[k].action() == INCLUDE && x[k] == 1 {
            return false;  // Violation
        }
    }
    true  // All included literals satisfied
}
```

</details>

<details>
<summary><strong>Voting Mechanism</strong></summary>

### Weighted Majority Vote

Classification is determined by summing weighted clause votes:

```
         C/2
v(x) =   Σ   [polarityⱼ × weightⱼ × clauseⱼ(x)]
        j=1

ŷ = { 1  if v(x) ≥ 0
    { 0  if v(x) < 0
```

**Polarity Assignment:**
- Even-indexed clauses: polarity = +1 (vote for class 1)
- Odd-indexed clauses: polarity = -1 (vote for class 0)

</details>

<details>
<summary><strong>Type I Feedback</strong></summary>

### Reinforcing Target Patterns

Applied when the sample belongs to the clause's target class.

**When clause FIRES (satisfies input):**

| Input | Literal | Probability | Action |
|-------|---------|-------------|--------|
| xₖ = 1 | xₖ | (s-1)/s | Increment (strengthen) |
| xₖ = 1 | ¬xₖ | 1/s | Decrement (weaken) |
| xₖ = 0 | xₖ | 1/s | Decrement (weaken) |
| xₖ = 0 | ¬xₖ | (s-1)/s | Increment (strengthen) |

**When clause DOES NOT FIRE:**

All automata decremented with probability 1/s (drift toward exclusion).

**Effect:** Reinforces literals that match the input pattern.

</details>

<details>
<summary><strong>Type II Feedback</strong></summary>

### Blocking False Positives

Applied when a clause fires for the WRONG class.

**Algorithm:**
```rust
fn type_ii(clause: &mut Clause, x: &[u8]) {
    for k in 0..n_features {
        if x[k] == 0 {
            // x_k is FALSE → include x_k to block (will fail since x_k=0)
            if clause.include[k].state <= N {
                clause.include[k].increment();
            }
        } else {
            // x_k is TRUE → include ¬x_k to block (will fail since x_k=1)
            if clause.negated[k].state <= N {
                clause.negated[k].increment();
            }
        }
    }
}
```

**Effect:** Adds a contradicting literal that will fail on this input.

</details>

<details>
<summary><strong>Training Algorithm</strong></summary>

### Complete Training Loop

```rust
fn train_epoch(tm: &mut TsetlinMachine, X: &[Vec<u8>], Y: &[u8], T: f32, s: f32) {
    for (x, y) in X.iter().zip(Y.iter()) {
        // 1. Compute vote sum (clamped to [-T, T])
        let v = tm.sum_votes(x).clamp(-T, T);

        // 2. Compute feedback probability
        let p = if *y == 1 {
            (T - v) / (2.0 * T)   // Higher prob when v is low (need more +1 votes)
        } else {
            (T + v) / (2.0 * T)   // Higher prob when v is high (need more -1 votes)
        };

        // 3. Apply feedback to each clause
        for (j, clause) in tm.clauses.iter_mut().enumerate() {
            let polarity = if j % 2 == 0 { 1 } else { -1 };
            let fires = clause.evaluate(x);

            if random() < p {
                if (*y == 1 && polarity == 1) || (*y == 0 && polarity == -1) {
                    // Target class: Type I feedback
                    type_i(clause, x, fires, s);
                } else if fires {
                    // Wrong class AND clause fired: Type II feedback
                    type_ii(clause, x);
                }
            }
        }
    }
}
```

### Parameter Effects

| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| **Clauses (C)** | Fewer patterns, faster | More patterns, slower |
| **States (N)** | Fast adaptation, unstable | Slow adaptation, stable |
| **Threshold (T)** | Conservative learning | Aggressive learning |
| **Specificity (s)** | General patterns (more literals) | Specific patterns (fewer literals) |

</details>

<div align="right"><a href="#top">Back to top</a></div>

---

## API Reference

### Models

| Type | Description |
|------|-------------|
| `TsetlinMachine` | Binary classification |
| `MultiClass` | Multi-class (one-vs-all) |
| `Regressor` | Continuous output |
| `Convolutional` | 2D patch extraction |
| `SparseTsetlinMachine` | Memory-efficient inference (50-125x compression) |

### Clauses

| Type | Heap | Best For |
|------|:----:|----------|
| `Clause` | Yes | Dynamic dimensions, serde |
| `SmallClause<N>` | No | N ≤ 64, maximum speed |
| `BitwiseClause` | Yes | N ≥ 64, 25-92x speedup |
| `SmallBitwiseClause<N,W>` | No | 64-256 features |

### Storage

| Type | Description |
|------|-------------|
| `ClauseBank` | SoA layout for bulk operations |
| `BitPlaneBank` | Bit-plane for parallel updates |
| `SparseClause` | SmallVec-based sparse clause (≤32 literals inline) |
| `SparseClauseBank` | CSR format for memory-efficient batch inference |

### Parallel Training

| Type | Description |
|------|-------------|
| `LocalTally` | Cache-padded atomic accumulator |
| `ParallelBatch` | Training batch with tallies |

### Configuration

| Type | Description |
|------|-------------|
| `Config` | Basic TM configuration |
| `ConfigBuilder` | Fluent configuration builder |
| `ConvConfig` | Convolutional TM configuration |
| `AdvancedOptions` | Weights, adaptive T, pruning |
| `FitOptions` | Early stopping, callbacks |

### Utilities

| Function | Description |
|----------|-------------|
| `pack_input(&[u8])` | Pack input for BitwiseClause |
| `pack_input_small(&[u8])` | Pack for SmallBitwiseClause |
| `pack_batch(&[Vec<u8>])` | Pack multiple inputs |
| `rng_from_seed(u64)` | Deterministic RNG |
| `rng_from_entropy()` | Random RNG |

<div align="right"><a href="#top">Back to top</a></div>

---

## Coverage


<p align="center">
  <a href="https://codecov.io/gh/RAprogramm/tsetlin-rs">
    <img src="https://codecov.io/gh/RAprogramm/tsetlin-rs/graphs/icicle.svg?token=dSDoNSNudX" alt="Icicle"/>
  </a>
</p>


<div align="right"><a href="#top">Back to top</a></div>

---

## In Memory of Michael Tsetlin

<img src=".github/assets/tsetlin.png" alt="Michael Lvovich Tsetlin" width="150" align="right"/>

**Mikhail Lvovich Tsetlin** (Михаил Львович Цетлин, 1924–1966) — Soviet mathematician and one of the founders of cybernetics in the USSR.

A veteran of World War II who served as a scout and tank gunner, he later became a brilliant scientist. Working alongside I.M. Gelfand, he pioneered the theory of learning automata — finite state machines that learn optimal behavior through interaction with the environment.

His work on collective automata behavior laid the theoretical foundation for what we now call the Tsetlin Machine. Despite his life being cut short at 41, his ideas continue to influence machine learning research today.

> *This library honors his legacy by bringing his concepts to modern systems programming.*

<div align="right"><a href="#top">Back to top</a></div>

---

## References

### Original Paper

> **The Tsetlin Machine - A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic**
> Ole-Christoffer Granmo, 2018
> [arXiv:1804.01508](https://arxiv.org/abs/1804.01508)

### Parallel Training

> **Massively Parallel and Asynchronous Tsetlin Machine Architecture Supporting Almost Constant-Time Scaling**
> K. Darshana Abeyrathna et al., ICML 2021
> [arXiv:2009.04861](https://arxiv.org/abs/2009.04861)

### Sparse Representation

> **The Sparse Tsetlin Machine: Sparse Representation with Active Literals**
> Sebastian Østby, Tobias M. Brambo, Sondre Glimsdal, 2024
> [arXiv:2405.02375](https://arxiv.org/abs/2405.02375)

### CPU Inference Optimization

> **Fast and Compact Tsetlin Machine Inference on CPUs Using Instruction-Level Optimization**
> 2025
> [arXiv:2510.15653](https://arxiv.org/abs/2510.15653)

### Implementations

| Repository | Description |
|------------|-------------|
| [cair/TsetlinMachineC](https://github.com/cair/TsetlinMachineC) | Original C implementation |
| [cair/pyTsetlinMachine](https://github.com/cair/pyTsetlinMachine) | Python library |
| [cair/PyTsetlinMachineCUDA](https://github.com/cair/PyTsetlinMachineCUDA) | CUDA implementation |
| [cair/tmu](https://github.com/cair/tmu) | Unified TM framework |

<div align="right"><a href="#top">Back to top</a></div>

---

## License

MIT

<div align="right"><a href="#top">Back to top</a></div>
