//! Structure of Arrays (SoA) clause storage for cache-efficient operations.
//!
//! This module provides [`ClauseBank`], a cache-optimized storage layout for
//! Tsetlin Machine clauses. Instead of storing each clause as an independent
//! object (Array of Structures), all clause data is stored in contiguous arrays
//! (Structure of Arrays).
//!
//! # Performance
//!
//! SoA layout provides cache-efficient evaluation with speedup increasing
//! for larger clause counts:
//!
//! | Clauses | AoS (µs) | SoA (µs) | Speedup |
//! |---------|----------|----------|---------|
//! | 50      | 3.64     | 3.50     | 1.04x   |
//! | 100     | 8.55     | 8.39     | 1.02x   |
//! | 200     | 18.50    | 16.02    | 1.15x   |
//!
//! Benefits increase with SIMD vectorization (planned feature).
//!
//! # Memory Layout
//!
//! ```text
//! AoS (Vec<Clause>):           SoA (ClauseBank):
//! ┌─────────────────┐          ┌─────────────────────────────────┐
//! │ Clause 0        │          │ states: [c0a0, c0a1, ..., cNaN] │
//! │  ├─ automata[]  │          ├─────────────────────────────────┤
//! │  ├─ weight      │          │ weights: [w0, w1, ..., wN]      │
//! │  └─ polarity    │          ├─────────────────────────────────┤
//! ├─────────────────┤          │ polarities: [p0, p1, ..., pN]   │
//! │ Clause 1        │          └─────────────────────────────────┘
//! │  ├─ automata[]  │
//! │  └─ ...         │          Contiguous memory = better prefetch
//! └─────────────────┘
//! ```
//!
//! # Example
//!
//! ```
//! use tsetlin_rs::ClauseBank;
//!
//! // Create bank with 100 clauses, 64 features, 100 states per automaton
//! let bank = ClauseBank::new(100, 64, 100);
//!
//! // Evaluate all clauses and sum weighted votes
//! let input = vec![1u8; 64];
//! let vote_sum = bank.sum_votes(&input);
//!
//! // Access individual clause data
//! assert_eq!(bank.n_clauses(), 100);
//! assert_eq!(bank.polarity(0), 1); // Even clauses: +1
//! assert_eq!(bank.polarity(1), -1); // Odd clauses: -1
//! ```
//!
//! # When to Use
//!
//! | Scenario | Recommendation |
//! |----------|----------------|
//! | Bulk clause operations | **ClauseBank** |
//! | Per-clause access patterns | `Vec<Clause>` |
//! | SIMD vectorization (future) | **ClauseBank** |
//! | Serialization with serde | Both supported |

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod eval;
mod feedback;
mod filter;

pub use filter::{ClauseFilter, ClauseFilterStats};

#[cfg(test)]
mod tests;

/// Structure of Arrays storage for Tsetlin Machine clauses.
///
/// Stores all clause data in flat, contiguous arrays for optimal cache
/// utilization. Each array holds one field across all clauses.
///
/// # Memory Layout
///
/// | Field | Type | Size | Description |
/// |-------|------|------|-------------|
/// | `states` | `Vec<i16>` | `n_clauses × 2 × n_features` | Automata states |
/// | `weights` | `Vec<f32>` | `n_clauses` | Clause weights |
/// | `polarities` | `Vec<i8>` | `n_clauses` | Vote direction (+1/-1) |
/// | `activations` | `Vec<u32>` | `n_clauses` | Activation counters |
/// | `correct` | `Vec<u32>` | `n_clauses` | Correct prediction counts |
/// | `incorrect` | `Vec<u32>` | `n_clauses` | Incorrect prediction counts |
///
/// # States Array Layout
///
/// The `states` array is indexed as `states[clause * stride + automaton]`
/// where:
/// - `stride = 2 * n_features`
/// - `automaton = 2 * k` for literal `x_k` (include)
/// - `automaton = 2 * k + 1` for literal `¬x_k` (negated)
///
/// ```text
/// states: [c0_x0, c0_¬x0, c0_x1, c0_¬x1, ..., c1_x0, c1_¬x0, ...]
///          ├─────── clause 0 ───────────┤├─────── clause 1 ────...
/// ```
///
/// # Thread Safety
///
/// `ClauseBank` is `Send + Sync` when all contained types are. For parallel
/// training, use external synchronization or partition clauses across threads.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClauseBank {
    /// Flat array of all automata states.
    ///
    /// Layout: `[clause_0_automata..., clause_1_automata..., ...]`
    /// Size: `n_clauses * 2 * n_features` elements.
    /// Values range from `1` to `2 * n_states`.
    pub(crate) states: Vec<i16>,

    /// Weight for each clause, used in weighted voting.
    ///
    /// Initial value: `1.0`. Updated via
    /// [`update_weights`](Self::update_weights).
    pub(crate) weights: Vec<f32>,

    /// Voting polarity for each clause.
    ///
    /// - `+1`: Positive clause (votes for class 1)
    /// - `-1`: Negative clause (votes for class 0)
    ///
    /// Assigned alternating by clause index: even = +1, odd = -1.
    pub(crate) polarities: Vec<i8>,

    /// Activation counter for each clause.
    ///
    /// Incremented each time a clause fires during evaluation.
    /// Used for clause pruning (detecting "dead" clauses).
    pub(crate) activations: Vec<u32>,

    /// Correct prediction counter for each clause.
    ///
    /// Incremented when clause fires and prediction is correct.
    /// Used for weight learning.
    pub(crate) correct: Vec<u32>,

    /// Incorrect prediction counter for each clause.
    ///
    /// Incremented when clause fires and prediction is wrong.
    /// Used for weight learning.
    pub(crate) incorrect: Vec<u32>,

    /// Total number of clauses in the bank.
    pub(crate) n_clauses: usize,

    /// Number of input features.
    ///
    /// Each feature generates 2 automata: one for `x_k`, one for `¬x_k`.
    pub(crate) n_features: usize,

    /// States per automaton (threshold for include action).
    ///
    /// Automaton with `state > n_states` produces "include" action.
    pub(crate) n_states: i16,

    /// Stride for indexing into states array.
    ///
    /// Equal to `2 * n_features`. Used to locate clause data:
    /// `clause_start = clause_idx * stride`.
    pub(crate) stride: usize,

    /// Bitmap tracking which clauses fired in last evaluation.
    ///
    /// Each bit represents one clause. Used to skip inactive clauses
    /// during feedback, providing ~40% speedup on converged models.
    ///
    /// Layout: `fires_bitmap[clause / 64] & (1 << (clause % 64))`
    pub(crate) fires_bitmap: Vec<u64>
}

impl ClauseBank {
    /// Creates a new clause bank with the specified dimensions.
    ///
    /// Initializes all automata at the threshold state (`n_states`), which
    /// means no literals are initially included (all clauses fire on any
    /// input).
    ///
    /// # Arguments
    ///
    /// * `n_clauses` - Number of clauses in the bank
    /// * `n_features` - Number of input features
    /// * `n_states` - States per automaton (threshold for include action)
    ///
    /// # Polarity Assignment
    ///
    /// Clauses are assigned alternating polarities:
    /// - Even indices (0, 2, 4, ...): polarity = +1
    /// - Odd indices (1, 3, 5, ...): polarity = -1
    ///
    /// # Panics
    ///
    /// Debug-asserts that `n_clauses > 0` and `n_features > 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use tsetlin_rs::ClauseBank;
    ///
    /// let bank = ClauseBank::new(100, 64, 100);
    /// assert_eq!(bank.n_clauses(), 100);
    /// assert_eq!(bank.n_features(), 64);
    /// assert_eq!(bank.states().len(), 100 * 2 * 64);
    /// ```
    #[must_use]
    pub fn new(n_clauses: usize, n_features: usize, n_states: i16) -> Self {
        debug_assert!(n_clauses > 0 && n_features > 0);

        let stride = 2 * n_features;
        let states = vec![n_states; n_clauses * stride];
        let polarities = (0..n_clauses)
            .map(|i| if i % 2 == 0 { 1 } else { -1 })
            .collect();

        let bitmap_len = n_clauses.div_ceil(64);
        Self {
            states,
            weights: vec![1.0; n_clauses],
            polarities,
            activations: vec![0; n_clauses],
            correct: vec![0; n_clauses],
            incorrect: vec![0; n_clauses],
            n_clauses,
            n_features,
            n_states,
            stride,
            fires_bitmap: vec![0; bitmap_len]
        }
    }

    /// Returns the total number of clauses in the bank.
    #[inline(always)]
    #[must_use]
    pub const fn n_clauses(&self) -> usize {
        self.n_clauses
    }

    /// Returns the number of input features.
    #[inline(always)]
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the number of states per automaton (threshold).
    #[inline(always)]
    #[must_use]
    pub const fn n_states(&self) -> i16 {
        self.n_states
    }

    /// Returns a read-only view of all automata states.
    ///
    /// The array is indexed as `states[clause * stride + automaton]`.
    /// Use [`clause_states`](Self::clause_states) for per-clause access.
    #[inline(always)]
    #[must_use]
    pub fn states(&self) -> &[i16] {
        &self.states
    }

    /// Returns a read-only view of all clause weights.
    #[inline(always)]
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Returns a read-only view of all clause polarities.
    #[inline(always)]
    #[must_use]
    pub fn polarities(&self) -> &[i8] {
        &self.polarities
    }

    /// Returns a read-only view of all activation counters.
    #[inline(always)]
    #[must_use]
    pub fn activations(&self) -> &[u32] {
        &self.activations
    }

    /// Returns the weight of a specific clause.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_clauses`.
    #[inline(always)]
    #[must_use]
    pub fn weight(&self, idx: usize) -> f32 {
        self.weights[idx]
    }

    /// Returns the polarity of a specific clause (+1 or -1).
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_clauses`.
    #[inline(always)]
    #[must_use]
    pub fn polarity(&self, idx: usize) -> i8 {
        self.polarities[idx]
    }

    /// Returns the automata states for a specific clause.
    ///
    /// The returned slice has length `2 * n_features`:
    /// - Index `2*k`: state for literal `x_k`
    /// - Index `2*k+1`: state for literal `¬x_k`
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_clauses`.
    ///
    /// # Example
    ///
    /// ```
    /// use tsetlin_rs::ClauseBank;
    ///
    /// let bank = ClauseBank::new(10, 4, 100);
    /// let clause_0_states = bank.clause_states(0);
    /// assert_eq!(clause_0_states.len(), 8); // 2 * 4 features
    /// ```
    #[inline]
    #[must_use]
    pub fn clause_states(&self, idx: usize) -> &[i16] {
        let start = idx * self.stride;
        &self.states[start..start + self.stride]
    }

    /// Increments an automaton's state (moves toward "include" action).
    ///
    /// The state is capped at `2 * n_states`.
    ///
    /// # Arguments
    ///
    /// * `clause` - Clause index
    /// * `automaton` - Automaton index within the clause (0 to `2*n_features -
    ///   1`)
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    #[inline(always)]
    pub fn increment(&mut self, clause: usize, automaton: usize) {
        let idx = clause * self.stride + automaton;
        let max = 2 * self.n_states;
        if self.states[idx] < max {
            self.states[idx] += 1;
        }
    }

    /// Decrements an automaton's state (moves toward "exclude" action).
    ///
    /// The state is floored at `1`.
    ///
    /// # Arguments
    ///
    /// * `clause` - Clause index
    /// * `automaton` - Automaton index within the clause (0 to `2*n_features -
    ///   1`)
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    #[inline(always)]
    pub fn decrement(&mut self, clause: usize, automaton: usize) {
        let idx = clause * self.stride + automaton;
        if self.states[idx] > 1 {
            self.states[idx] -= 1;
        }
    }
}
