//! # Tsetlin Machine
//!
//! Professional Rust implementation of the Tsetlin Machine algorithm.
//!
//! # Features
//!
//! - `std` (default): Standard library support
//! - `simd`: SIMD-optimized evaluation (requires nightly)
//! - `parallel`: Parallel training via rayon
//! - `serde`: Serialization support
//!
//! # Examples
//!
//! ```
//! use tsetlin_rs::{Config, TsetlinMachine};
//!
//! let config = Config::builder().clauses(20).features(2).build().unwrap();
//!
//! let mut tm = TsetlinMachine::new(config, 10);
//!
//! let x = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
//! let y = vec![0, 1, 1, 0];
//!
//! tm.fit(&x, &y, 200, 42);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "simd", feature(portable_simd))]

#[cfg(not(feature = "std"))]
extern crate alloc;

mod automaton;
mod binary;
mod bitwise;
mod clause;
mod config;
mod convolutional;
pub mod error;
pub mod feedback;
mod multiclass;
mod regression;
mod rule;
mod small;
mod training;
pub mod utils;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "simd")]
pub mod simd;

pub use automaton::Automaton;
pub use binary::TsetlinMachine;
pub use bitwise::{BitwiseClause, pack_batch, pack_input};
pub use clause::Clause;
pub use config::{Config, ConfigBuilder};
pub use convolutional::{ConvConfig, Convolutional};
pub use error::{Error, Result};
pub use multiclass::MultiClass;
pub use regression::Regressor;
pub use rule::Rule;
pub use small::{Clause16, Clause2, Clause32, Clause4, Clause8, SmallClause};
pub use training::{EarlyStop, FitOptions, FitResult};
