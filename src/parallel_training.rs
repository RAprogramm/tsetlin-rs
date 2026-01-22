//! Lock-free parallel training with asynchronous local voting tallies.
//!
//! This module implements the Massively Parallel and Asynchronous Tsetlin
//! Machine architecture from ICML 2021, adapted for CPU with Rust safety
//! guarantees.
//!
//! # Key Innovation
//!
//! Each training sample maintains its own atomic vote accumulator. Clauses
//! update tallies concurrently via `fetch_add` without any synchronization
//! barriers. This enables near-constant-time scaling with clause count.
//!
//! # Architecture
//!
//! ```text
//! Traditional (synchronous):
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ [Clause 0] ─┐                                                   │
//! │ [Clause 1] ─┼─► Barrier ─► Sum ─► Barrier ─► Feedback           │
//! │ [Clause N] ─┘                                                   │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! Async local tallies (this implementation):
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ [Clause 0] ──► eval ──► atomic_add ──► feedback (independent)   │
//! │ [Clause 1] ──► eval ──► atomic_add ──► feedback (independent)   │
//! │ [Clause N] ──► eval ──► atomic_add ──► feedback (independent)   │
//! │                    (no barriers, fully parallel)                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! - **50x speedup** reported in original CUDA paper
//! - **Constant-time scaling** for 20-7000 clauses
//! - **No accuracy loss** from working on stale data
//!
//! # References
//!
//! - [Massively Parallel TM (ICML 2021)](https://arxiv.org/abs/2009.04861)
//! - [PyTsetlinMachineCUDA](https://github.com/cair/PyTsetlinMachineCUDA)

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::sync::atomic::{AtomicI64, Ordering};

use crossbeam_utils::CachePadded;
use rayon::prelude::*;

use crate::ClauseBank;

/// Scale factor for converting f32 weights to i64 for atomic operations.
///
/// With SCALE = 10000, we get 4 decimal places of precision.
/// Weight range [-100, 100] maps to [-1_000_000, 1_000_000] i64.
const WEIGHT_SCALE: i64 = 10_000;

/// A training sample with cache-aligned atomic vote accumulator.
///
/// Uses `CachePadded` to prevent false sharing when multiple threads
/// update adjacent samples. Each sample occupies its own cache line.
///
/// # Thread Safety
///
/// The atomic tally uses `Relaxed` ordering for accumulation (sufficient
/// for commutative addition) and `Acquire` for final reads.
#[derive(Debug)]
pub struct LocalTally {
    /// Scaled vote accumulator (actual_vote * WEIGHT_SCALE).
    tally: CachePadded<AtomicI64>
}

impl LocalTally {
    /// Creates a new zeroed tally.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            tally: CachePadded::new(AtomicI64::new(0))
        }
    }

    /// Atomically adds a scaled vote.
    ///
    /// # Arguments
    ///
    /// * `delta` - Pre-scaled vote value (polarity * weight * WEIGHT_SCALE)
    #[inline]
    pub fn add(&self, delta: i64) {
        self.tally.fetch_add(delta, Ordering::Relaxed);
    }

    /// Atomically adds an unweighted vote (polarity only).
    ///
    /// # Arguments
    ///
    /// * `polarity` - Clause polarity (+1 or -1)
    #[inline]
    pub fn add_unweighted(&self, polarity: i8) {
        self.tally
            .fetch_add(polarity as i64 * WEIGHT_SCALE, Ordering::Relaxed);
    }

    /// Atomically adds a weighted vote.
    ///
    /// # Arguments
    ///
    /// * `polarity` - Clause polarity (+1 or -1)
    /// * `weight` - Clause weight (f32)
    #[inline]
    pub fn add_weighted(&self, polarity: i8, weight: f32) {
        let scaled = (polarity as f32 * weight * WEIGHT_SCALE as f32) as i64;
        self.tally.fetch_add(scaled, Ordering::Relaxed);
    }

    /// Reads the current tally as f32.
    #[inline]
    #[must_use]
    pub fn sum(&self) -> f32 {
        self.tally.load(Ordering::Acquire) as f32 / WEIGHT_SCALE as f32
    }

    /// Reads the raw scaled tally.
    #[inline]
    #[must_use]
    pub fn sum_scaled(&self) -> i64 {
        self.tally.load(Ordering::Acquire)
    }

    /// Resets the tally to zero.
    #[inline]
    pub fn reset(&self) {
        self.tally.store(0, Ordering::Relaxed);
    }
}

impl Default for LocalTally {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch of training samples with their local voting tallies.
///
/// Provides parallel iteration via rayon. Each sample's tally is
/// cache-aligned to prevent false sharing.
pub struct ParallelBatch<'a> {
    /// Input features for each sample.
    pub inputs: &'a [Vec<u8>],

    /// Target labels.
    pub labels: &'a [u8],

    /// Local vote tallies (one per sample, cache-aligned).
    tallies: Vec<LocalTally>
}

impl<'a> ParallelBatch<'a> {
    /// Creates a new batch from data.
    ///
    /// # Panics
    ///
    /// Panics if `inputs.len() != labels.len()`.
    #[must_use]
    pub fn new(inputs: &'a [Vec<u8>], labels: &'a [u8]) -> Self {
        assert_eq!(inputs.len(), labels.len());
        let tallies = (0..inputs.len()).map(|_| LocalTally::new()).collect();
        Self {
            inputs,
            labels,
            tallies
        }
    }

    /// Returns the number of samples.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Returns `true` if the batch is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    /// Resets all tallies to zero.
    pub fn reset_tallies(&self) {
        self.tallies.par_iter().for_each(LocalTally::reset);
    }

    /// Returns the tally for a specific sample.
    #[inline]
    #[must_use]
    pub fn tally(&self, idx: usize) -> &LocalTally {
        &self.tallies[idx]
    }

    /// Returns input features for a specific sample.
    #[inline]
    #[must_use]
    pub fn input(&self, idx: usize) -> &[u8] {
        &self.inputs[idx]
    }

    /// Returns the label for a specific sample.
    #[inline]
    #[must_use]
    pub fn label(&self, idx: usize) -> u8 {
        self.labels[idx]
    }

    /// Parallel iterator over (sample_idx, input, label, tally).
    pub fn par_iter(&self) -> impl ParallelIterator<Item = (usize, &[u8], u8, &LocalTally)> {
        (0..self.len()).into_par_iter().map(|i| {
            (
                i,
                self.inputs[i].as_slice(),
                self.labels[i],
                &self.tallies[i]
            )
        })
    }
}

impl ClauseBank {
    /// Evaluates a single clause and atomically updates the sample's tally.
    ///
    /// This is the core operation for lock-free parallel training.
    /// Multiple threads can call this concurrently on the same sample
    /// without synchronization.
    ///
    /// # Arguments
    ///
    /// * `clause_idx` - Index of the clause to evaluate
    /// * `x` - Binary input vector
    /// * `tally` - Atomic tally to update
    ///
    /// # Returns
    ///
    /// `true` if the clause fired.
    #[inline]
    pub fn eval_and_tally(&self, clause_idx: usize, x: &[u8], tally: &LocalTally) -> bool {
        let fires = self.evaluate_clause(clause_idx, x);
        if fires {
            tally.add_weighted(self.polarities[clause_idx], self.weights[clause_idx]);
        }
        fires
    }

    /// Parallel clause evaluation across all clauses for a single sample.
    ///
    /// Each clause is evaluated in parallel, atomically updating the
    /// sample's local tally. Returns a bitmap of firing clauses.
    ///
    /// # Arguments
    ///
    /// * `x` - Binary input vector
    /// * `tally` - Atomic tally to update
    ///
    /// # Returns
    ///
    /// Vector of firing clause indices.
    pub fn parallel_eval_sample(&self, x: &[u8], tally: &LocalTally) -> Vec<usize> {
        (0..self.n_clauses)
            .into_par_iter()
            .filter(|&clause| self.eval_and_tally(clause, x, tally))
            .collect()
    }

    /// Lock-free parallel training on a batch of samples.
    ///
    /// This implements the ICML 2021 massively parallel architecture:
    /// 1. All clauses evaluate all samples in parallel
    /// 2. Votes are accumulated via atomic operations
    /// 3. Feedback is applied independently per clause
    ///
    /// # Arguments
    ///
    /// * `batch` - Batch of samples with local tallies
    /// * `threshold` - Voting threshold T
    /// * `s` - Specificity parameter
    /// * `seed` - Random seed (each clause gets deterministic sub-seed)
    pub fn train_parallel(&mut self, batch: &ParallelBatch, threshold: f32, s: f32, seed: u64) {
        // Phase 1: Reset tallies
        batch.reset_tallies();

        let n_samples = batch.len();
        let n_clauses = self.n_clauses;
        let bitmap_words = n_clauses.div_ceil(64);

        // Phase 2: Parallel evaluation with bitmap per sample
        // Each sample gets a bitmap tracking which clauses fired
        let firing_bitmaps: Vec<Vec<u64>> = (0..n_samples)
            .into_par_iter()
            .map(|sample| {
                let x = batch.input(sample);
                let mut bitmap = vec![0u64; bitmap_words];

                for clause in 0..n_clauses {
                    let fires = self.evaluate_clause(clause, x);
                    if fires {
                        let polarity = self.polarities[clause];
                        let weight = self.weights[clause];
                        batch.tally(sample).add_weighted(polarity, weight);

                        // Set bit in bitmap
                        let word = clause / 64;
                        let bit = clause % 64;
                        bitmap[word] |= 1u64 << bit;
                    }
                }
                bitmap
            })
            .collect();

        // Phase 3: Apply feedback
        let inv_2t = 1.0 / (2.0 * threshold);

        for (sample_idx, bitmap) in firing_bitmaps.iter().enumerate() {
            let x = batch.input(sample_idx);
            let y = batch.label(sample_idx);
            let sum = batch.tally(sample_idx).sum().clamp(-threshold, threshold);

            let prob = if y == 1 {
                (threshold - sum) * inv_2t
            } else {
                (threshold + sum) * inv_2t
            };

            let mut rng = crate::utils::rng_from_seed(seed.wrapping_add(sample_idx as u64));

            for clause in 0..n_clauses {
                let p = self.polarities[clause];

                // O(1) lookup via bitmap
                let word = clause / 64;
                let bit = clause % 64;
                let fires = bitmap[word] & (1u64 << bit) != 0;

                if y == 1 {
                    if p == 1 && rand::Rng::random::<f32>(&mut rng) <= prob {
                        self.type_i(clause, x, fires, s, &mut rng);
                    } else if p == -1 && fires && rand::Rng::random::<f32>(&mut rng) <= prob {
                        self.type_ii(clause, x);
                    }
                } else if p == -1 && rand::Rng::random::<f32>(&mut rng) <= prob {
                    self.type_i(clause, x, fires, s, &mut rng);
                } else if p == 1 && fires && rand::Rng::random::<f32>(&mut rng) <= prob {
                    self.type_ii(clause, x);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_tally_atomic_operations() {
        let tally = LocalTally::new();

        assert!((tally.sum() - 0.0).abs() < 0.001);

        tally.add_weighted(1, 1.5);
        tally.add_weighted(-1, 0.5);

        assert!((tally.sum() - 1.0).abs() < 0.001);

        tally.reset();
        assert!((tally.sum() - 0.0).abs() < 0.001);
    }

    #[test]
    fn local_tally_unweighted() {
        let tally = LocalTally::new();

        tally.add_unweighted(1);
        tally.add_unweighted(1);
        tally.add_unweighted(-1);

        assert!((tally.sum() - 1.0).abs() < 0.001);
    }

    #[test]
    fn parallel_batch_creation() {
        let inputs = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
        let labels = vec![0, 1, 1, 0];

        let batch = ParallelBatch::new(&inputs, &labels);

        assert_eq!(batch.len(), 4);
        assert!(!batch.is_empty());
    }

    #[test]
    fn parallel_batch_iteration() {
        let inputs = vec![vec![0, 1], vec![1, 0]];
        let labels = vec![1, 0];

        let batch = ParallelBatch::new(&inputs, &labels);

        // Simulate parallel voting
        batch.tally(0).add_weighted(1, 1.0);
        batch.tally(1).add_weighted(-1, 1.0);

        assert!((batch.tally(0).sum() - 1.0).abs() < 0.001);
        assert!((batch.tally(1).sum() - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn clause_bank_eval_and_tally() {
        let bank = ClauseBank::new(4, 2, 100);
        let tally = LocalTally::new();

        // Fresh clauses always fire
        let fires = bank.eval_and_tally(0, &[1, 1], &tally);
        assert!(fires);
        assert!(tally.sum().abs() > 0.0); // Got a vote
    }
}
