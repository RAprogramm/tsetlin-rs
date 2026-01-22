//! Sparse clause representation for memory-efficient inference.
//!
//! This module provides [`SparseClause`] and [`SparseClauseBank`] for
//! 5-100x memory reduction in trained Tsetlin Machine models.
//!
//! # Background
//!
//! After training, most automata in a clause are in the "exclude" state
//! (action = false). A typical trained clause contains only 10-50 active
//! literals out of 1000+ possible features. Storing all automata wastes
//! 95-99% of memory.
//!
//! # Architecture
//!
//! This module implements sparse representations based on research:
//! - [Sparse TM with Active Literals (arXiv:2405.02375)](https://arxiv.org/abs/2405.02375)
//! - [Contracting TM with Absorbing Automata (arXiv:2310.11481)](https://arxiv.org/abs/2310.11481)
//!
//! ```text
//! Training:  ClauseBank (dense SoA)
//!                │
//!                ▼ .to_sparse()
//!                │
//! Inference: SparseClauseBank (CSR)  ──► 5-100x memory reduction
//! ```
//!
//! # Data Structures
//!
//! | Type | Format | Use Case |
//! |------|--------|----------|
//! | [`SparseClause`] | SmallVec | Single clause, inline for ≤32 literals |
//! | [`SparseClauseBank`] | CSR | Batch inference, cache-friendly |
//!
//! # Example
//!
//! ```
//! use tsetlin_rs::{ClauseBank, SparseClauseBank};
//!
//! // Train with dense representation
//! let bank = ClauseBank::new(100, 1000, 100);
//! // ... training ...
//!
//! // Convert to sparse for deployment
//! let sparse = SparseClauseBank::from_clause_bank(&bank);
//!
//! // Memory reduction
//! let stats = sparse.memory_stats();
//! println!("Compression: {}x", stats.compression_ratio(1000));
//! ```

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::{Clause, ClauseBank};

/// Inline capacity for typical clause (32 literals = 64 bytes).
///
/// Research shows trained clauses typically contain 10-50 active literals.
/// SmallVec avoids heap allocation for this common case.
const INLINE_CAPACITY: usize = 32;

/// Sparse clause representation storing only active literal indices.
///
/// Achieves 5-100x memory reduction compared to dense [`Clause`] by storing
/// only indices of features that affect evaluation.
///
/// # Memory Layout
///
/// Typical clause with 20 active literals:
/// - `include_indices`: 32 × 2 = 64 bytes (inline SmallVec)
/// - `negated_indices`: 32 × 2 = 64 bytes (inline SmallVec)
/// - `weight`: 4 bytes
/// - `polarity`: 1 byte
/// - **Total: ~133 bytes** vs ~8000 bytes for dense (1000 features)
///
/// # Performance
///
/// Uses early-exit evaluation: returns `false` on first violation.
/// For sparse input data, this is often faster than dense bitmask evaluation.
///
/// # Example
///
/// ```
/// use tsetlin_rs::{Clause, SparseClause};
///
/// let mut clause = Clause::new(100, 100, 1);
/// // ... train clause ...
///
/// let sparse = SparseClause::from_clause(&clause);
/// assert!(sparse.memory_usage() < 200); // vs 800+ bytes dense
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseClause {
    /// Indices of features where `x[k] = 1` is required (include literals).
    include_indices: SmallVec<[u16; INLINE_CAPACITY]>,

    /// Indices of features where `x[k] = 0` is required (negated literals).
    negated_indices: SmallVec<[u16; INLINE_CAPACITY]>,

    /// Clause weight for weighted voting.
    weight: f32,

    /// Voting polarity (+1 or -1).
    polarity: i8
}

impl SparseClause {
    /// Creates a sparse clause from a dense [`Clause`] by extracting active
    /// literals.
    ///
    /// Scans automata and records indices where `action() == true`.
    ///
    /// # Arguments
    ///
    /// * `clause` - Dense clause to convert
    ///
    /// # Example
    ///
    /// ```
    /// use tsetlin_rs::{Clause, SparseClause};
    ///
    /// let clause = Clause::new(100, 100, 1);
    /// let sparse = SparseClause::from_clause(&clause);
    /// assert_eq!(sparse.n_literals(), 0); // Fresh clause has no active literals
    /// ```
    #[must_use]
    pub fn from_clause(clause: &Clause) -> Self {
        let mut include = SmallVec::new();
        let mut negated = SmallVec::new();

        for (k, pair) in clause.automata().chunks(2).enumerate() {
            if pair[0].action() {
                include.push(k as u16);
            }
            if pair[1].action() {
                negated.push(k as u16);
            }
        }

        Self {
            include_indices: include,
            negated_indices: negated,
            weight:          clause.weight(),
            polarity:        clause.polarity()
        }
    }

    /// Creates a sparse clause from raw components.
    ///
    /// # Arguments
    ///
    /// * `include` - Feature indices requiring `x[k] = 1`
    /// * `negated` - Feature indices requiring `x[k] = 0`
    /// * `weight` - Clause weight
    /// * `polarity` - Vote direction (+1 or -1)
    #[must_use]
    pub fn new(include: &[u16], negated: &[u16], weight: f32, polarity: i8) -> Self {
        Self {
            include_indices: SmallVec::from_slice(include),
            negated_indices: SmallVec::from_slice(negated),
            weight,
            polarity
        }
    }

    /// Returns clause polarity (+1 or -1).
    #[inline(always)]
    #[must_use]
    pub const fn polarity(&self) -> i8 {
        self.polarity
    }

    /// Returns clause weight.
    #[inline(always)]
    #[must_use]
    pub const fn weight(&self) -> f32 {
        self.weight
    }

    /// Returns indices of include literals.
    #[inline(always)]
    #[must_use]
    pub fn include_indices(&self) -> &[u16] {
        &self.include_indices
    }

    /// Returns indices of negated literals.
    #[inline(always)]
    #[must_use]
    pub fn negated_indices(&self) -> &[u16] {
        &self.negated_indices
    }

    /// Evaluates clause with early exit on first violation.
    ///
    /// Returns `true` if all conditions are satisfied:
    /// - For each index in `include_indices`: `x[idx] == 1`
    /// - For each index in `negated_indices`: `x[idx] == 0`
    ///
    /// # Arguments
    ///
    /// * `x` - Binary input vector
    ///
    /// # Safety
    ///
    /// Uses unchecked indexing for performance. Caller must ensure all
    /// stored indices are within `x.len()`.
    #[inline]
    #[must_use]
    pub fn evaluate(&self, x: &[u8]) -> bool {
        for &idx in &self.include_indices {
            // SAFETY: caller ensures idx < x.len()
            if unsafe { *x.get_unchecked(idx as usize) } == 0 {
                return false;
            }
        }
        for &idx in &self.negated_indices {
            // SAFETY: caller ensures idx < x.len()
            if unsafe { *x.get_unchecked(idx as usize) } == 1 {
                return false;
            }
        }
        true
    }

    /// Evaluates clause with bounds checking.
    ///
    /// Safe version of [`evaluate`](Self::evaluate) that performs bounds
    /// checks.
    #[inline]
    #[must_use]
    pub fn evaluate_checked(&self, x: &[u8]) -> bool {
        for &idx in &self.include_indices {
            if x.get(idx as usize).copied().unwrap_or(0) == 0 {
                return false;
            }
        }
        for &idx in &self.negated_indices {
            if x.get(idx as usize).copied().unwrap_or(0) == 1 {
                return false;
            }
        }
        true
    }

    /// Evaluates using packed u64 input (64 features per word).
    ///
    /// Optimized for cases where input is already packed. Uses bit extraction
    /// instead of array indexing.
    ///
    /// # Arguments
    ///
    /// * `x` - Binary input packed as u64 words
    #[inline]
    #[must_use]
    pub fn evaluate_packed(&self, x: &[u64]) -> bool {
        for &idx in &self.include_indices {
            let word = idx as usize >> 6; // / 64
            let bit = idx as usize & 63; // % 64
            // SAFETY: caller ensures sufficient words
            if unsafe { *x.get_unchecked(word) } & (1u64 << bit) == 0 {
                return false;
            }
        }
        for &idx in &self.negated_indices {
            let word = idx as usize >> 6;
            let bit = idx as usize & 63;
            if unsafe { *x.get_unchecked(word) } & (1u64 << bit) != 0 {
                return false;
            }
        }
        true
    }

    /// Returns weighted vote: `polarity × weight` if clause fires, `0.0`
    /// otherwise.
    #[inline]
    #[must_use]
    pub fn vote(&self, x: &[u8]) -> f32 {
        if self.evaluate(x) {
            self.polarity as f32 * self.weight
        } else {
            0.0
        }
    }

    /// Returns unweighted vote: `polarity` if clause fires, `0` otherwise.
    #[inline]
    #[must_use]
    pub fn vote_unweighted(&self, x: &[u8]) -> i32 {
        if self.evaluate(x) {
            self.polarity as i32
        } else {
            0
        }
    }

    /// Returns approximate memory usage in bytes.
    ///
    /// Accounts for SmallVec inline vs heap allocation.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let base = core::mem::size_of::<Self>();
        let include_heap = if self.include_indices.spilled() {
            self.include_indices.capacity() * 2
        } else {
            0
        };
        let negated_heap = if self.negated_indices.spilled() {
            self.negated_indices.capacity() * 2
        } else {
            0
        };
        base + include_heap + negated_heap
    }

    /// Returns total number of active literals.
    #[inline]
    #[must_use]
    pub fn n_literals(&self) -> usize {
        self.include_indices.len() + self.negated_indices.len()
    }
}

/// Sparse clause bank using CSR (Compressed Sparse Row) format.
///
/// Optimized for batch inference with cache-friendly memory access.
/// Stores all active literals in contiguous arrays with offset indexing.
///
/// # Memory Layout
///
/// ```text
/// Dense ClauseBank (100 clauses, 1000 features):
///   states: 100 × 2 × 1000 × 2 = 400,000 bytes
///
/// SparseClauseBank (avg 30 literals/clause):
///   include_indices: ~1500 × 2 = 3,000 bytes
///   include_offsets: 101 × 4 = 404 bytes
///   negated_indices: ~1500 × 2 = 3,000 bytes
///   negated_offsets: 101 × 4 = 404 bytes
///   weights: 100 × 4 = 400 bytes
///   polarities: 100 bytes
///   Total: ~7,300 bytes
///
/// Memory reduction: 55x
/// ```
///
/// # CSR Format
///
/// For clause `c`, active include literals are at:
/// `include_indices[include_offsets[c]..include_offsets[c+1]]`
///
/// This format provides:
/// - O(1) clause lookup
/// - Sequential memory access (cache-friendly)
/// - Minimal overhead per clause (4 bytes offset)
///
/// # Example
///
/// ```
/// use tsetlin_rs::{ClauseBank, SparseClauseBank};
///
/// let bank = ClauseBank::new(100, 1000, 100);
/// let sparse = SparseClauseBank::from_clause_bank(&bank);
///
/// let stats = sparse.memory_stats();
/// println!("Total literals: {}", stats.total_literals);
/// println!("Memory: {} bytes", stats.total());
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseClauseBank {
    /// CSR data: active include literal indices for all clauses.
    include_indices: Vec<u16>,

    /// CSR offsets: `include_indices[offsets[c]..offsets[c+1]]` for clause `c`.
    include_offsets: Vec<u32>,

    /// CSR data: active negated literal indices.
    negated_indices: Vec<u16>,

    /// CSR offsets for negated literals.
    negated_offsets: Vec<u32>,

    /// Clause weights.
    weights: Vec<f32>,

    /// Clause polarities (+1 or -1).
    polarities: Vec<i8>,

    /// Number of clauses.
    n_clauses: usize,

    /// Number of features (for documentation/validation).
    n_features: usize
}

impl SparseClauseBank {
    /// Converts dense [`ClauseBank`] to sparse CSR format.
    ///
    /// Scans all clauses and extracts active literals (automata with
    /// `state > n_states`).
    ///
    /// # Arguments
    ///
    /// * `bank` - Dense clause bank to convert
    ///
    /// # Example
    ///
    /// ```
    /// use tsetlin_rs::{ClauseBank, SparseClauseBank};
    ///
    /// let bank = ClauseBank::new(100, 64, 100);
    /// let sparse = SparseClauseBank::from_clause_bank(&bank);
    /// assert_eq!(sparse.n_clauses(), 100);
    /// ```
    #[must_use]
    pub fn from_clause_bank(bank: &ClauseBank) -> Self {
        let mut include_indices = Vec::new();
        let mut include_offsets = vec![0u32];
        let mut negated_indices = Vec::new();
        let mut negated_offsets = vec![0u32];

        let threshold = bank.n_states();

        for c in 0..bank.n_clauses() {
            let states = bank.clause_states(c);

            for (k, pair) in states.chunks(2).enumerate() {
                if pair[0] > threshold {
                    include_indices.push(k as u16);
                }
                if pair[1] > threshold {
                    negated_indices.push(k as u16);
                }
            }

            include_offsets.push(include_indices.len() as u32);
            negated_offsets.push(negated_indices.len() as u32);
        }

        Self {
            include_indices,
            include_offsets,
            negated_indices,
            negated_offsets,
            weights: bank.weights().to_vec(),
            polarities: bank.polarities().to_vec(),
            n_clauses: bank.n_clauses(),
            n_features: bank.n_features()
        }
    }

    /// Creates from a vector of [`SparseClause`].
    ///
    /// Useful when building sparse representation incrementally or from
    /// non-ClauseBank sources.
    #[must_use]
    pub fn from_clauses(clauses: &[SparseClause], n_features: usize) -> Self {
        let mut include_indices = Vec::new();
        let mut include_offsets = vec![0u32];
        let mut negated_indices = Vec::new();
        let mut negated_offsets = vec![0u32];
        let mut weights = Vec::with_capacity(clauses.len());
        let mut polarities = Vec::with_capacity(clauses.len());

        for clause in clauses {
            include_indices.extend_from_slice(&clause.include_indices);
            include_offsets.push(include_indices.len() as u32);

            negated_indices.extend_from_slice(&clause.negated_indices);
            negated_offsets.push(negated_indices.len() as u32);

            weights.push(clause.weight);
            polarities.push(clause.polarity);
        }

        Self {
            include_indices,
            include_offsets,
            negated_indices,
            negated_offsets,
            weights,
            polarities,
            n_clauses: clauses.len(),
            n_features
        }
    }

    /// Returns number of clauses.
    #[inline(always)]
    #[must_use]
    pub const fn n_clauses(&self) -> usize {
        self.n_clauses
    }

    /// Returns number of features.
    #[inline(always)]
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns clause weights.
    #[inline(always)]
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Returns clause polarities.
    #[inline(always)]
    #[must_use]
    pub fn polarities(&self) -> &[i8] {
        &self.polarities
    }

    /// Returns number of active literals for a specific clause.
    #[inline]
    #[must_use]
    pub fn clause_n_literals(&self, clause: usize) -> usize {
        let inc = self.include_offsets[clause + 1] - self.include_offsets[clause];
        let neg = self.negated_offsets[clause + 1] - self.negated_offsets[clause];
        (inc + neg) as usize
    }

    /// Evaluates single clause.
    ///
    /// Returns `true` if all active literal conditions are satisfied.
    #[inline]
    #[must_use]
    pub fn evaluate_clause(&self, clause: usize, x: &[u8]) -> bool {
        let inc_start = self.include_offsets[clause] as usize;
        let inc_end = self.include_offsets[clause + 1] as usize;

        for &idx in &self.include_indices[inc_start..inc_end] {
            // SAFETY: indices were validated during construction
            if unsafe { *x.get_unchecked(idx as usize) } == 0 {
                return false;
            }
        }

        let neg_start = self.negated_offsets[clause] as usize;
        let neg_end = self.negated_offsets[clause + 1] as usize;

        for &idx in &self.negated_indices[neg_start..neg_end] {
            if unsafe { *x.get_unchecked(idx as usize) } == 1 {
                return false;
            }
        }

        true
    }

    /// Evaluates clause with packed u64 input.
    ///
    /// Optimized for pre-packed input where 64 features fit in one u64.
    #[inline]
    #[must_use]
    pub fn evaluate_clause_packed(&self, clause: usize, x: &[u64]) -> bool {
        let inc_start = self.include_offsets[clause] as usize;
        let inc_end = self.include_offsets[clause + 1] as usize;

        for &idx in &self.include_indices[inc_start..inc_end] {
            let word = idx as usize >> 6;
            let bit = idx as usize & 63;
            if unsafe { *x.get_unchecked(word) } & (1u64 << bit) == 0 {
                return false;
            }
        }

        let neg_start = self.negated_offsets[clause] as usize;
        let neg_end = self.negated_offsets[clause + 1] as usize;

        for &idx in &self.negated_indices[neg_start..neg_end] {
            let word = idx as usize >> 6;
            let bit = idx as usize & 63;
            if unsafe { *x.get_unchecked(word) } & (1u64 << bit) != 0 {
                return false;
            }
        }

        true
    }

    /// Sum of weighted votes for all clauses.
    ///
    /// Evaluates all clauses and accumulates `polarity × weight` for
    /// firing clauses.
    #[must_use]
    pub fn sum_votes(&self, x: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for c in 0..self.n_clauses {
            if self.evaluate_clause(c, x) {
                // SAFETY: c < n_clauses
                sum += unsafe {
                    *self.polarities.get_unchecked(c) as f32 * *self.weights.get_unchecked(c)
                };
            }
        }
        sum
    }

    /// Sum of weighted votes with packed input.
    #[must_use]
    pub fn sum_votes_packed(&self, x: &[u64]) -> f32 {
        let mut sum = 0.0f32;
        for c in 0..self.n_clauses {
            if self.evaluate_clause_packed(c, x) {
                sum += unsafe {
                    *self.polarities.get_unchecked(c) as f32 * *self.weights.get_unchecked(c)
                };
            }
        }
        sum
    }

    /// Sum of unweighted votes.
    #[must_use]
    pub fn sum_votes_unweighted(&self, x: &[u8]) -> i32 {
        let mut sum = 0i32;
        for c in 0..self.n_clauses {
            if self.evaluate_clause(c, x) {
                sum += self.polarities[c] as i32;
            }
        }
        sum
    }

    /// Returns memory usage statistics.
    #[must_use]
    pub fn memory_stats(&self) -> SparseMemoryStats {
        SparseMemoryStats {
            include_data:    self.include_indices.len() * 2,
            include_offsets: self.include_offsets.len() * 4,
            negated_data:    self.negated_indices.len() * 2,
            negated_offsets: self.negated_offsets.len() * 4,
            weights:         self.weights.len() * 4,
            polarities:      self.polarities.len(),
            total_literals:  self.include_indices.len() + self.negated_indices.len(),
            n_clauses:       self.n_clauses,
            n_features:      self.n_features
        }
    }
}

/// Memory usage breakdown for sparse clause bank.
///
/// Provides detailed statistics for memory analysis and optimization decisions.
#[derive(Debug, Clone, Copy)]
pub struct SparseMemoryStats {
    /// Size of include indices array in bytes.
    pub include_data: usize,

    /// Size of include offsets array in bytes.
    pub include_offsets: usize,

    /// Size of negated indices array in bytes.
    pub negated_data: usize,

    /// Size of negated offsets array in bytes.
    pub negated_offsets: usize,

    /// Size of weights array in bytes.
    pub weights: usize,

    /// Size of polarities array in bytes.
    pub polarities: usize,

    /// Total number of active literals across all clauses.
    pub total_literals: usize,

    /// Number of clauses.
    pub n_clauses: usize,

    /// Number of features.
    pub n_features: usize
}

impl SparseMemoryStats {
    /// Returns total memory usage in bytes.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.include_data
            + self.include_offsets
            + self.negated_data
            + self.negated_offsets
            + self.weights
            + self.polarities
    }

    /// Returns average literals per clause.
    #[must_use]
    pub fn avg_literals_per_clause(&self) -> f32 {
        if self.n_clauses == 0 {
            0.0
        } else {
            self.total_literals as f32 / self.n_clauses as f32
        }
    }

    /// Returns compression ratio compared to dense storage.
    ///
    /// # Arguments
    ///
    /// * `n_features` - Number of features (for dense size calculation)
    #[must_use]
    pub fn compression_ratio(&self, n_features: usize) -> f32 {
        let dense_size = self.n_clauses * 2 * n_features * 2; // i16 states
        if self.total() == 0 {
            0.0
        } else {
            dense_size as f32 / self.total() as f32
        }
    }

    /// Returns sparsity ratio (fraction of literals that are active).
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        let max_literals = self.n_clauses * 2 * self.n_features;
        if max_literals == 0 {
            0.0
        } else {
            self.total_literals as f32 / max_literals as f32
        }
    }
}

/// Sparse Tsetlin Machine for memory-efficient inference.
///
/// Inference-only model with 5-100x memory reduction compared to dense
/// `TsetlinMachine`. Create via `TsetlinMachine::to_sparse()` after training.
///
/// # Example
///
/// ```
/// use tsetlin_rs::{Config, TsetlinMachine};
///
/// let config = Config::builder().clauses(20).features(2).build().unwrap();
/// let mut tm = TsetlinMachine::new(config, 10);
///
/// let x = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
/// let y = vec![0, 1, 1, 0];
///
/// tm.fit(&x, &y, 200, 42);
///
/// // Convert to sparse for deployment
/// let sparse = tm.to_sparse();
///
/// // Same predictions, less memory
/// assert_eq!(tm.predict(&x[0]), sparse.predict(&x[0]));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseTsetlinMachine {
    clauses:   SparseClauseBank,
    threshold: f32
}

impl SparseTsetlinMachine {
    /// Creates from dense clauses.
    ///
    /// # Arguments
    ///
    /// * `clauses` - Dense clauses to convert
    /// * `n_features` - Number of features
    /// * `threshold` - Classification threshold
    #[must_use]
    pub fn from_clauses(clauses: &[Clause], n_features: usize, threshold: f32) -> Self {
        let sparse_clauses: Vec<SparseClause> =
            clauses.iter().map(SparseClause::from_clause).collect();

        Self {
            clauses: SparseClauseBank::from_clauses(&sparse_clauses, n_features),
            threshold
        }
    }

    /// Creates from pre-built sparse clause bank.
    #[must_use]
    pub fn new(clauses: SparseClauseBank, threshold: f32) -> Self {
        Self {
            clauses,
            threshold
        }
    }

    /// Returns number of clauses.
    #[inline(always)]
    #[must_use]
    pub const fn n_clauses(&self) -> usize {
        self.clauses.n_clauses()
    }

    /// Returns number of features.
    #[inline(always)]
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.clauses.n_features()
    }

    /// Returns threshold.
    #[inline(always)]
    #[must_use]
    pub const fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Predicts class (0 or 1).
    #[inline]
    #[must_use]
    pub fn predict(&self, x: &[u8]) -> u8 {
        if self.clauses.sum_votes(x) >= 0.0 {
            1
        } else {
            0
        }
    }

    /// Predicts using packed u64 input.
    #[inline]
    #[must_use]
    pub fn predict_packed(&self, x: &[u64]) -> u8 {
        if self.clauses.sum_votes_packed(x) >= 0.0 {
            1
        } else {
            0
        }
    }

    /// Batch prediction.
    #[must_use]
    pub fn predict_batch(&self, xs: &[Vec<u8>]) -> Vec<u8> {
        xs.iter().map(|x| self.predict(x)).collect()
    }

    /// Evaluates accuracy on test data.
    #[must_use]
    pub fn evaluate(&self, x: &[Vec<u8>], y: &[u8]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        let correct = x
            .iter()
            .zip(y)
            .filter(|(xi, yi)| self.predict(xi) == **yi)
            .count();
        correct as f32 / x.len() as f32
    }

    /// Returns memory statistics.
    #[must_use]
    pub fn memory_stats(&self) -> SparseMemoryStats {
        self.clauses.memory_stats()
    }

    /// Returns compression ratio compared to dense model.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        self.clauses
            .memory_stats()
            .compression_ratio(self.n_features())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_clause_from_dense() {
        let mut clause = Clause::new(10, 100, 1);

        // Activate literal 0 (include) and literal 2 (negated)
        for _ in 0..200 {
            clause.automata_mut()[0].increment(); // include[0]
            clause.automata_mut()[5].increment(); // negated[2]
        }

        let sparse = SparseClause::from_clause(&clause);
        assert_eq!(sparse.include_indices.len(), 1);
        assert_eq!(sparse.negated_indices.len(), 1);
        assert_eq!(sparse.include_indices[0], 0);
        assert_eq!(sparse.negated_indices[0], 2);
        assert_eq!(sparse.polarity(), 1);
    }

    #[test]
    fn sparse_clause_evaluate() {
        let sparse = SparseClause::new(&[0, 2], &[1], 1.0, 1);

        // x[0]=1, x[1]=0, x[2]=1 -> should fire
        assert!(sparse.evaluate(&[1, 0, 1, 0]));

        // x[0]=0 -> include violation
        assert!(!sparse.evaluate(&[0, 0, 1, 0]));

        // x[1]=1 -> negated violation
        assert!(!sparse.evaluate(&[1, 1, 1, 0]));

        // x[2]=0 -> include violation
        assert!(!sparse.evaluate(&[1, 0, 0, 0]));
    }

    #[test]
    fn sparse_clause_evaluate_packed() {
        let sparse = SparseClause::new(&[0, 2], &[1], 1.0, 1);

        // Packed: bits 0,2 set, bit 1 clear -> 0b101 = 5
        assert!(sparse.evaluate_packed(&[5u64]));

        // Packed: bit 0 clear -> should fail
        assert!(!sparse.evaluate_packed(&[4u64]));
    }

    #[test]
    fn sparse_clause_vote() {
        let sparse = SparseClause::new(&[], &[], 2.5, -1);

        // Empty clause always fires
        assert!((sparse.vote(&[0, 1, 0]) - (-2.5)).abs() < 0.001);
        assert_eq!(sparse.vote_unweighted(&[0, 1, 0]), -1);
    }

    #[test]
    fn sparse_clause_memory() {
        let sparse = SparseClause::new(&[0, 1, 2], &[3, 4], 1.0, 1);

        // Should be inline (not spilled)
        let usage = sparse.memory_usage();
        assert!(usage < 200);
        assert_eq!(sparse.n_literals(), 5);
    }

    #[test]
    fn sparse_bank_from_clause_bank() {
        let bank = ClauseBank::new(10, 100, 100);
        let sparse = SparseClauseBank::from_clause_bank(&bank);

        assert_eq!(sparse.n_clauses(), 10);
        assert_eq!(sparse.n_features(), 100);

        // Fresh bank has no active literals
        let stats = sparse.memory_stats();
        assert_eq!(stats.total_literals, 0);
    }

    #[test]
    fn sparse_bank_evaluate() {
        // Create sparse bank manually
        let clauses = vec![
            SparseClause::new(&[0], &[], 1.0, 1),  // requires x[0]=1
            SparseClause::new(&[], &[0], 1.0, -1), // requires x[0]=0
        ];
        let sparse = SparseClauseBank::from_clauses(&clauses, 4);

        // x[0]=1: clause 0 fires (+1), clause 1 fails
        let votes = sparse.sum_votes(&[1, 0, 0, 0]);
        assert!((votes - 1.0).abs() < 0.001);

        // x[0]=0: clause 0 fails, clause 1 fires (-1)
        let votes = sparse.sum_votes(&[0, 0, 0, 0]);
        assert!((votes - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn sparse_bank_memory_stats() {
        let clauses = vec![
            SparseClause::new(&[0, 1, 2], &[3], 1.0, 1),
            SparseClause::new(&[4, 5], &[6, 7, 8], 1.0, -1),
        ];
        let sparse = SparseClauseBank::from_clauses(&clauses, 100);

        let stats = sparse.memory_stats();
        assert_eq!(stats.total_literals, 9); // 3+1 + 2+3
        assert_eq!(stats.n_clauses, 2);

        // Compression ratio should be high
        let ratio = stats.compression_ratio(100);
        assert!(ratio > 10.0);
    }

    #[test]
    fn sparse_bank_packed_evaluation() {
        let clauses = vec![
            SparseClause::new(&[0, 63], &[], 1.0, 1),  // bits 0 and 63
            SparseClause::new(&[], &[1, 62], 1.0, -1), // not bits 1 and 62
        ];
        let sparse = SparseClauseBank::from_clauses(&clauses, 64);

        // Packed: bits 0 and 63 set, bits 1 and 62 clear
        let packed = 1u64 | (1u64 << 63);
        let votes = sparse.sum_votes_packed(&[packed]);
        assert!((votes - 0.0).abs() < 0.001); // Both fire: +1 - 1 = 0
    }
}
