//! # Tsetlin Machine
//!
//! A professional Rust implementation of the Tsetlin Machine algorithm
//! for interpretable machine learning.
//!
//! # Models
//!
//! - [`TsetlinMachine`] - Binary classification (dynamic)
//! - [`SmallTsetlinMachine`] - Binary classification (compile-time optimized)
//! - [`SparseTsetlinMachine`] - Memory-efficient inference (5-100x reduction)
//! - [`MultiClass`] - Multi-class classification (one-vs-all)
//! - [`Regressor`] - Regression
//! - [`Convolutional`] - Image classification with patch extraction
//!
//! # Clause Types
//!
//! | Type | Heap | Best For |
//! |------|------|----------|
//! | [`Clause`] | Yes | Dynamic dimensions, serde |
//! | [`SmallClause`] | No | N ≤ 64, maximum speed |
//! | [`BitwiseClause`] | Yes | N ≥ 64, 25-92x speedup |
//! | [`SmallBitwiseClause`] | No | 64-256 features, no heap |
//! | [`SparseClause`] | Inline | Inference, 5-100x compression |
//! | [`SparseClauseBank`] | Yes | CSR batch inference |
//!
//! # Advanced Features
//!
//! - **Weighted Clauses** - Clauses learn weights based on accuracy
//! - **Adaptive Threshold** - Dynamic T adjustment during training
//! - **Clause Pruning** - Automatic reset of dead/ineffective clauses
//! - **Const Generics** - Zero-allocation stack types with loop unrolling
//! - **Lock-Free Parallel Training** - Async local voting tallies (ICML 2021)
//!
//! # Feature Flags
//!
//! - `std` (default): Standard library support
//! - `simd`: SIMD-optimized evaluation (requires nightly)
//! - `parallel`: Parallel training via rayon
//! - `serde`: Serialization support
//!
//! # Quick Start
//!
//! ```
//! use tsetlin_rs::{Config, TsetlinMachine};
//!
//! let config = Config::builder().clauses(20).features(2).build().unwrap();
//! let mut tm = TsetlinMachine::new(config, 10);
//!
//! let x = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
//! let y = vec![0, 1, 1, 0];
//!
//! tm.fit(&x, &y, 200, 42);
//! assert!(tm.evaluate(&x, &y) >= 0.75);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "simd", feature(portable_simd))]

#[cfg(not(feature = "std"))]
extern crate alloc;

mod automaton;
mod binary;
mod bitplane;
mod bitwise;
mod clause;
mod clause_bank;
mod config;
mod convolutional;
pub mod error;
pub mod feedback;
mod model;
mod multiclass;
mod regression;
mod rule;
mod small;
mod sparse;
mod training;
pub mod utils;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "parallel")]
pub mod parallel_training;

#[cfg(feature = "simd")]
pub mod simd;

pub use automaton::Automaton;
pub use binary::{AdvancedOptions, TsetlinMachine};
pub use bitplane::BitPlaneBank;
pub use bitwise::{BitwiseClause, pack_batch, pack_input};
pub use clause::Clause;
pub use clause_bank::ClauseBank;
pub use config::{Config, ConfigBuilder};
pub use convolutional::{ConvConfig, Convolutional};
pub use error::{Error, Result};
pub use model::{TsetlinModel, VotingModel};
pub use multiclass::MultiClass;
#[cfg(feature = "parallel")]
pub use parallel_training::{LocalTally, ParallelBatch};
pub use regression::Regressor;
pub use rule::Rule;
pub use small::{
    BitwiseClause64, BitwiseClause128, BitwiseClause256, Clause2, Clause4, Clause8, Clause16,
    Clause32, Clause64, SmallBitwiseClause, SmallClause, SmallTsetlinMachine, TM2x20, TM4x40,
    TM8x80, TM16x160, pack_input_small
};
pub use sparse::{SparseClause, SparseClauseBank, SparseMemoryStats, SparseTsetlinMachine};
pub use training::{EarlyStop, FitOptions, FitResult};
