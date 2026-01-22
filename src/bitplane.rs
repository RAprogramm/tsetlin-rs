//! Bit-plane storage for Tsetlin Automata states.
//!
//! This module provides [`BitPlaneBank`], a transposed bit-level representation
//! of automata states optimized for parallel bitwise operations.
//!
//! # Memory Layout
//!
//! Traditional layout stores each state contiguously:
//! ```text
//! [state0: 8bits, state1: 8bits, state2: 8bits, ...]
//! ```
//!
//! Bit-plane layout transposes this, storing each bit position across all
//! states:
//!
//! ```text
//! plane[0]: [s0_b0, s1_b0, s2_b0, ...]  // LSB of all states
//! plane[1]: [s0_b1, s1_b1, s2_b1, ...]
//! ...
//! plane[7]: [s0_b7, s1_b7, s2_b7, ...]  // MSB = action bit
//! ```
//!
//! # Performance Benefits
//!
//! - Clause eval: O(n) comparisons vs O(n/64) bitwise AND
//! - Increment: O(n) additions vs O(8) ripple-carry
//! - Batch update: Sequential vs 64 states in parallel
//!
//! The MSB (bit 7) directly encodes the automaton action:
//! - MSB = 1: state > 127 = "include" action
//! - MSB = 0: state <= 127 = "exclude" action
//!
//! # References
//!
//! - [Fast CUDA TM](https://github.com/cair/fast-tsetlin-machine-in-cuda-with-imdb-demo)
//! - K.D. Abeyrathna et al., "Massively Parallel TM Architecture", ICML 2021

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Number of bits per automaton state.
const STATE_BITS: usize = 8;

/// Number of automata processed per u64 chunk.
const CHUNK_SIZE: usize = 64;

/// Bit-plane storage for automata states.
///
/// Stores `n_clauses × 2 × n_features` automata in transposed bit-plane format.
/// Each clause has `2 × n_features` automata (include + negated per feature).
///
/// # Example
///
/// ```
/// use tsetlin_rs::BitPlaneBank;
///
/// // 10 clauses, 32 features, initial state 100
/// let bank = BitPlaneBank::new(10, 32, 100);
///
/// assert_eq!(bank.n_clauses(), 10);
/// assert_eq!(bank.n_features(), 32);
/// assert_eq!(bank.get_state(0, 0), 100);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BitPlaneBank {
    /// Bit-planes for all automata states.
    /// Layout: `planes[bit][clause * chunks_per_clause + chunk]`
    planes: [Vec<u64>; STATE_BITS],

    /// Number of clauses.
    n_clauses: usize,

    /// Number of features per clause.
    n_features: usize,

    /// Number of u64 chunks per clause (ceil(2 * n_features / 64)).
    chunks_per_clause: usize,

    /// Total automata per clause (2 * n_features).
    automata_per_clause: usize,

    /// Clause polarities (+1 or -1).
    polarities: Vec<i8>,

    /// Clause weights for weighted voting.
    weights: Vec<f32>
}

impl BitPlaneBank {
    /// Creates a new bit-plane bank with all automata at initial state.
    ///
    /// # Arguments
    ///
    /// * `n_clauses` - Number of clauses
    /// * `n_features` - Number of input features
    /// * `initial_state` - Initial state for all automata (typically n_states)
    ///
    /// # Panics
    ///
    /// Panics if `n_clauses` is zero or `n_features` is zero.
    pub fn new(n_clauses: usize, n_features: usize, initial_state: u8) -> Self {
        assert!(n_clauses > 0, "n_clauses must be positive");
        assert!(n_features > 0, "n_features must be positive");

        let automata_per_clause = 2 * n_features;
        let chunks_per_clause = automata_per_clause.div_ceil(CHUNK_SIZE);
        let total_chunks = n_clauses * chunks_per_clause;

        // Initialize bit-planes from initial_state
        let planes = core::array::from_fn(|bit| {
            let bit_set = (initial_state >> bit) & 1 == 1;
            if bit_set {
                // All bits in this plane are 1
                vec![u64::MAX; total_chunks]
            } else {
                vec![0u64; total_chunks]
            }
        });

        // Mask out unused bits in last chunk of each clause
        let remainder = automata_per_clause % CHUNK_SIZE;
        let mut bank = Self {
            planes,
            n_clauses,
            n_features,
            chunks_per_clause,
            automata_per_clause,
            polarities: (0..n_clauses)
                .map(|i| if i % 2 == 0 { 1 } else { -1 })
                .collect(),
            weights: vec![1.0; n_clauses]
        };

        // Clear unused bits in remainder chunks
        if remainder > 0 {
            let mask = (1u64 << remainder) - 1;
            for clause in 0..n_clauses {
                let last_chunk = clause * chunks_per_clause + chunks_per_clause - 1;
                for plane in &mut bank.planes {
                    plane[last_chunk] &= mask;
                }
            }
        }

        bank
    }

    /// Returns number of clauses.
    #[inline]
    pub fn n_clauses(&self) -> usize {
        self.n_clauses
    }

    /// Returns number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns clause polarity (+1 or -1).
    #[inline]
    pub fn polarity(&self, clause: usize) -> i8 {
        self.polarities[clause]
    }

    /// Returns clause weight.
    #[inline]
    pub fn weight(&self, clause: usize) -> f32 {
        self.weights[clause]
    }

    /// Gets the state of a specific automaton.
    ///
    /// # Arguments
    ///
    /// * `clause` - Clause index
    /// * `automaton` - Automaton index within clause (0..2*n_features)
    pub fn get_state(&self, clause: usize, automaton: usize) -> u8 {
        let chunk_idx = clause * self.chunks_per_clause + automaton / CHUNK_SIZE;
        let bit_pos = automaton % CHUNK_SIZE;

        let mut state = 0u8;
        for (bit, plane) in self.planes.iter().enumerate() {
            if (plane[chunk_idx] >> bit_pos) & 1 == 1 {
                state |= 1 << bit;
            }
        }
        state
    }

    /// Sets the state of a specific automaton.
    ///
    /// # Arguments
    ///
    /// * `clause` - Clause index
    /// * `automaton` - Automaton index within clause
    /// * `state` - New state value
    pub fn set_state(&mut self, clause: usize, automaton: usize, state: u8) {
        let chunk_idx = clause * self.chunks_per_clause + automaton / CHUNK_SIZE;
        let bit_pos = automaton % CHUNK_SIZE;
        let mask = 1u64 << bit_pos;

        for (bit, plane) in self.planes.iter_mut().enumerate() {
            if (state >> bit) & 1 == 1 {
                plane[chunk_idx] |= mask;
            } else {
                plane[chunk_idx] &= !mask;
            }
        }
    }

    /// Returns the action for an automaton (true = include, false = exclude).
    ///
    /// Action is determined by MSB: state > 127 means include.
    #[inline]
    pub fn action(&self, clause: usize, automaton: usize) -> bool {
        let chunk_idx = clause * self.chunks_per_clause + automaton / CHUNK_SIZE;
        let bit_pos = automaton % CHUNK_SIZE;
        (self.planes[STATE_BITS - 1][chunk_idx] >> bit_pos) & 1 == 1
    }

    /// Returns the MSB chunks for a clause (for fast evaluation).
    #[inline]
    pub fn msb_chunks(&self, clause: usize) -> &[u64] {
        let start = clause * self.chunks_per_clause;
        &self.planes[STATE_BITS - 1][start..start + self.chunks_per_clause]
    }

    /// Evaluates whether a clause fires on the given input.
    ///
    /// A clause fires when all active literals are satisfied:
    /// - Include literal (automaton 2k): fires if x\[k\] = 1
    /// - Negated literal (automaton 2k+1): fires if x\[k\] = 0
    ///
    /// Uses bitwise operations for parallel evaluation of all literals.
    ///
    /// # Arguments
    ///
    /// * `clause` - Clause index
    /// * `x` - Binary input vector
    ///
    /// # Returns
    ///
    /// `true` if clause fires, `false` otherwise.
    pub fn evaluate(&self, clause: usize, x: &[u8]) -> bool {
        let msb = self.msb_chunks(clause);
        let n = x.len().min(self.n_features);

        // Process 32 features (64 automata) per chunk
        let full_chunks = n / 32;

        for (chunk_idx, &actions) in msb.iter().enumerate().take(full_chunks) {
            if actions == 0 {
                continue; // No active literals in this chunk
            }

            let x_offset = chunk_idx * 32;
            let x_interleaved = interleave_input(&x[x_offset..x_offset + 32]);

            // Check violations:
            // - Include (even bits): active AND x=0 → violation
            // - Negated (odd bits): active AND x=1 → violation
            // For interleaved x: bit 2k = x[k], bit 2k+1 = x[k]
            // Include violation: actions & EVEN_MASK & !x_interleaved
            // Negated violation: actions & ODD_MASK & x_interleaved
            let include_actions = actions & EVEN_BITS_MASK;
            let negated_actions = actions & ODD_BITS_MASK;

            if (include_actions & !x_interleaved) != 0 {
                return false;
            }
            if (negated_actions & x_interleaved) != 0 {
                return false;
            }
        }

        // Handle remaining features
        let start = full_chunks * 32;
        for (k, &xk) in x.iter().enumerate().skip(start).take(n - start) {
            let include_active = self.action(clause, 2 * k);
            let negated_active = self.action(clause, 2 * k + 1);

            if include_active && xk == 0 {
                return false;
            }
            if negated_active && xk == 1 {
                return false;
            }
        }

        true
    }

    /// Sums weighted votes for all clauses on the given input.
    ///
    /// # Arguments
    ///
    /// * `x` - Binary input vector
    ///
    /// # Returns
    ///
    /// Sum of `polarity * weight` for each firing clause.
    pub fn sum_votes(&self, x: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.n_clauses {
            if self.evaluate(i, x) {
                sum += self.polarities[i] as f32 * self.weights[i];
            }
        }
        sum
    }

    /// Increments states at positions specified by mask.
    ///
    /// Uses ripple-carry addition to increment up to 64 states in parallel.
    /// Saturates at 255 (no overflow).
    ///
    /// # Arguments
    ///
    /// * `clause` - Clause index
    /// * `chunk` - Chunk index within clause
    /// * `mask` - Bitmask of positions to increment
    ///
    /// # Algorithm
    ///
    /// Ripple-carry addition across bit-planes:
    /// ```text
    /// carry = mask
    /// for each bit-plane:
    ///     new_value = plane ^ carry
    ///     new_carry = plane & carry
    ///     plane = new_value
    ///     carry = new_carry
    /// ```
    #[inline]
    pub fn increment_masked(&mut self, clause: usize, chunk: usize, mask: u64) {
        if mask == 0 {
            return;
        }

        let chunk_idx = clause * self.chunks_per_clause + chunk;

        // Check for saturation (all bits set = 255)
        let saturated = self
            .planes
            .iter()
            .fold(u64::MAX, |acc, p| acc & p[chunk_idx]);
        let mask = mask & !saturated;

        if mask == 0 {
            return;
        }

        let mut carry = mask;
        for plane in &mut self.planes {
            let val = plane[chunk_idx];
            plane[chunk_idx] = val ^ carry;
            carry &= val;
            if carry == 0 {
                break;
            }
        }
    }

    /// Decrements states at positions specified by mask.
    ///
    /// Uses ripple-borrow subtraction to decrement up to 64 states in parallel.
    /// Saturates at 1 (no underflow below 1).
    ///
    /// # Arguments
    ///
    /// * `clause` - Clause index
    /// * `chunk` - Chunk index within clause
    /// * `mask` - Bitmask of positions to decrement
    #[inline]
    pub fn decrement_masked(&mut self, clause: usize, chunk: usize, mask: u64) {
        if mask == 0 {
            return;
        }

        let chunk_idx = clause * self.chunks_per_clause + chunk;

        // Check for floor (state = 1 = 0b00000001)
        let at_floor = self.planes[0][chunk_idx]
            & !self.planes[1][chunk_idx]
            & !self.planes[2][chunk_idx]
            & !self.planes[3][chunk_idx]
            & !self.planes[4][chunk_idx]
            & !self.planes[5][chunk_idx]
            & !self.planes[6][chunk_idx]
            & !self.planes[7][chunk_idx];
        let mask = mask & !at_floor;

        if mask == 0 {
            return;
        }

        let mut borrow = mask;
        for plane in &mut self.planes {
            let val = plane[chunk_idx];
            plane[chunk_idx] = val ^ borrow;
            borrow &= !val;
            if borrow == 0 {
                break;
            }
        }
    }

    /// Increments a single automaton state.
    ///
    /// Saturates at 255.
    #[inline]
    pub fn increment(&mut self, clause: usize, automaton: usize) {
        let chunk = automaton / CHUNK_SIZE;
        let bit_pos = automaton % CHUNK_SIZE;
        self.increment_masked(clause, chunk, 1u64 << bit_pos);
    }

    /// Decrements a single automaton state.
    ///
    /// Saturates at 1 (minimum state).
    #[inline]
    pub fn decrement(&mut self, clause: usize, automaton: usize) {
        let chunk = automaton / CHUNK_SIZE;
        let bit_pos = automaton % CHUNK_SIZE;
        self.decrement_masked(clause, chunk, 1u64 << bit_pos);
    }
}

/// Mask for even bits (0, 2, 4, ..., 62).
const EVEN_BITS_MASK: u64 = 0x5555_5555_5555_5555;

/// Mask for odd bits (1, 3, 5, ..., 63).
const ODD_BITS_MASK: u64 = 0xAAAA_AAAA_AAAA_AAAA;

/// Interleaves 32 input bytes into a u64 with pattern [x0,x0,x1,x1,...].
///
/// Each bit of the result has:
/// - Even positions (2k): x[k]
/// - Odd positions (2k+1): x[k]
#[inline]
fn interleave_input(x: &[u8]) -> u64 {
    let mut result = 0u64;
    for (k, &xk) in x.iter().enumerate().take(32) {
        if xk != 0 {
            // Set both even and odd bit for this feature
            result |= 3u64 << (2 * k);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_bank_initializes_states() {
        let bank = BitPlaneBank::new(4, 8, 100);

        assert_eq!(bank.n_clauses(), 4);
        assert_eq!(bank.n_features(), 8);

        // All automata should be at initial state
        for clause in 0..4 {
            for automaton in 0..16 {
                assert_eq!(bank.get_state(clause, automaton), 100);
            }
        }
    }

    #[test]
    fn set_and_get_state() {
        let mut bank = BitPlaneBank::new(2, 4, 100);

        bank.set_state(0, 0, 200);
        bank.set_state(0, 1, 50);
        bank.set_state(1, 3, 255);

        assert_eq!(bank.get_state(0, 0), 200);
        assert_eq!(bank.get_state(0, 1), 50);
        assert_eq!(bank.get_state(1, 3), 255);
        assert_eq!(bank.get_state(0, 2), 100); // unchanged
    }

    #[test]
    fn action_from_msb() {
        let mut bank = BitPlaneBank::new(1, 4, 100);

        // State 100 = 0b01100100 → MSB = 0 → exclude
        assert!(!bank.action(0, 0));

        // Set to 200 = 0b11001000 → MSB = 1 → include
        bank.set_state(0, 0, 200);
        assert!(bank.action(0, 0));

        // Set to 127 = 0b01111111 → MSB = 0 → exclude
        bank.set_state(0, 0, 127);
        assert!(!bank.action(0, 0));

        // Set to 128 = 0b10000000 → MSB = 1 → include
        bank.set_state(0, 0, 128);
        assert!(bank.action(0, 0));
    }

    #[test]
    fn alternating_polarity() {
        let bank = BitPlaneBank::new(4, 8, 100);

        assert_eq!(bank.polarity(0), 1);
        assert_eq!(bank.polarity(1), -1);
        assert_eq!(bank.polarity(2), 1);
        assert_eq!(bank.polarity(3), -1);
    }

    #[test]
    fn large_clause_chunks() {
        // 100 features = 200 automata = ceil(200/64) = 4 chunks
        let bank = BitPlaneBank::new(2, 100, 100);

        assert_eq!(bank.chunks_per_clause, 4);

        // Test automaton at boundary
        for clause in 0..2 {
            for automaton in 0..200 {
                assert_eq!(bank.get_state(clause, automaton), 100);
            }
        }
    }

    #[test]
    fn empty_clause_fires() {
        // Initial state 100 < 128 → all actions are exclude → clause fires
        let bank = BitPlaneBank::new(1, 8, 100);
        let x = vec![1, 0, 1, 0, 1, 0, 1, 0];
        assert!(bank.evaluate(0, &x));
    }

    #[test]
    fn include_violation_blocks() {
        let mut bank = BitPlaneBank::new(1, 8, 100);

        // Set include action for feature 0 (automaton 0)
        bank.set_state(0, 0, 200); // MSB = 1 → include active

        // x[0] = 1 → should fire
        assert!(bank.evaluate(0, &[1, 0, 0, 0, 0, 0, 0, 0]));

        // x[0] = 0 → should NOT fire (include violation)
        assert!(!bank.evaluate(0, &[0, 0, 0, 0, 0, 0, 0, 0]));
    }

    #[test]
    fn negated_violation_blocks() {
        let mut bank = BitPlaneBank::new(1, 8, 100);

        // Set negated action for feature 0 (automaton 1)
        bank.set_state(0, 1, 200); // MSB = 1 → negated active

        // x[0] = 0 → should fire
        assert!(bank.evaluate(0, &[0, 0, 0, 0, 0, 0, 0, 0]));

        // x[0] = 1 → should NOT fire (negated violation)
        assert!(!bank.evaluate(0, &[1, 0, 0, 0, 0, 0, 0, 0]));
    }

    #[test]
    fn sum_votes_with_polarity() {
        let mut bank = BitPlaneBank::new(4, 4, 100);
        let x = vec![1, 1, 1, 1];

        // All clauses fire (empty), polarities alternate
        // vote = 1 - 1 + 1 - 1 = 0
        assert_eq!(bank.sum_votes(&x), 0.0);

        // Disable clause 1 by adding include that violates
        bank.set_state(1, 0, 200); // include feature 0
        // Now clause 1 fires only if x[0]=1, which it is
        // So still all fire: 1 - 1 + 1 - 1 = 0
        assert_eq!(bank.sum_votes(&x), 0.0);

        // Test with x[0]=0: clause 1 doesn't fire
        // vote = 1 + 1 - 1 = 1
        assert_eq!(bank.sum_votes(&[0, 1, 1, 1]), 1.0);
    }

    #[test]
    fn interleave_input_correct() {
        let x = [
            1u8, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0
        ];
        let result = interleave_input(&x);

        // x[0]=1 → bits 0,1 set
        // x[2]=1 → bits 4,5 set
        // Expected: 0b110011 = 51
        assert_eq!(result & 0xFF, 0b00110011);
    }

    #[test]
    fn large_feature_evaluation() {
        // Test with 64 features (2 chunks)
        let mut bank = BitPlaneBank::new(1, 64, 100);
        let x: Vec<u8> = (0..64).map(|i| (i % 2) as u8).collect();

        // Empty clause fires
        assert!(bank.evaluate(0, &x));

        // Set include for feature 33 (second chunk)
        bank.set_state(0, 66, 200); // automaton 2*33 = 66

        // x[33] = 1 (odd index) → should fire
        assert!(bank.evaluate(0, &x));

        // Change x[33] to 0 → should not fire
        let mut x2 = x.clone();
        x2[33] = 0;
        assert!(!bank.evaluate(0, &x2));
    }

    #[test]
    fn increment_single() {
        let mut bank = BitPlaneBank::new(1, 4, 100);

        bank.increment(0, 0);
        assert_eq!(bank.get_state(0, 0), 101);

        bank.increment(0, 0);
        assert_eq!(bank.get_state(0, 0), 102);
    }

    #[test]
    fn decrement_single() {
        let mut bank = BitPlaneBank::new(1, 4, 100);

        bank.decrement(0, 0);
        assert_eq!(bank.get_state(0, 0), 99);

        bank.decrement(0, 0);
        assert_eq!(bank.get_state(0, 0), 98);
    }

    #[test]
    fn increment_saturates_at_255() {
        let mut bank = BitPlaneBank::new(1, 4, 250);

        for _ in 0..10 {
            bank.increment(0, 0);
        }
        assert_eq!(bank.get_state(0, 0), 255);
    }

    #[test]
    fn decrement_saturates_at_1() {
        let mut bank = BitPlaneBank::new(1, 4, 5);

        for _ in 0..10 {
            bank.decrement(0, 0);
        }
        assert_eq!(bank.get_state(0, 0), 1);
    }

    #[test]
    fn increment_masked_parallel() {
        let mut bank = BitPlaneBank::new(1, 64, 100);

        // Increment positions 0, 2, 4 (mask = 0b10101)
        bank.increment_masked(0, 0, 0b10101);

        assert_eq!(bank.get_state(0, 0), 101);
        assert_eq!(bank.get_state(0, 1), 100); // unchanged
        assert_eq!(bank.get_state(0, 2), 101);
        assert_eq!(bank.get_state(0, 3), 100); // unchanged
        assert_eq!(bank.get_state(0, 4), 101);
    }

    #[test]
    fn decrement_masked_parallel() {
        let mut bank = BitPlaneBank::new(1, 64, 100);

        // Decrement positions 0, 2, 4
        bank.decrement_masked(0, 0, 0b10101);

        assert_eq!(bank.get_state(0, 0), 99);
        assert_eq!(bank.get_state(0, 1), 100); // unchanged
        assert_eq!(bank.get_state(0, 2), 99);
        assert_eq!(bank.get_state(0, 3), 100); // unchanged
        assert_eq!(bank.get_state(0, 4), 99);
    }

    #[test]
    fn ripple_carry_correctness() {
        let mut bank = BitPlaneBank::new(1, 64, 127);

        // 127 = 0b01111111
        // +1 = 128 = 0b10000000
        bank.increment(0, 0);
        assert_eq!(bank.get_state(0, 0), 128);

        // Now action should be include (MSB = 1)
        assert!(bank.action(0, 0));
    }
}
