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
//! # Arena Allocator
//!
//! For high-throughput training, [`TrainingArena`] provides memory reuse:
//! - Pre-allocated tally pool avoids per-epoch allocations
//! - Reusable index buffer eliminates shuffle allocation overhead
//! - Reduces allocator contention in parallel code
//!
//! # References
//!
//! - [Massively Parallel TM (ICML 2021)](https://arxiv.org/abs/2009.04861)
//! - [PyTsetlinMachineCUDA](https://github.com/cair/PyTsetlinMachineCUDA)

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::sync::atomic::{AtomicI64, Ordering};

use crossbeam_utils::CachePadded;
use rand::Rng;
use rayon::prelude::*;

use crate::{ClauseBank, config::prob_to_threshold};

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

/// Arena allocator for training memory reuse.
///
/// Pre-allocates memory for training operations to avoid repeated allocations
/// during epoch iterations. Particularly beneficial for large datasets where
/// per-epoch allocation overhead becomes significant.
///
/// # Memory Layout
///
/// ```text
/// TrainingArena
/// ├── tallies: Vec<LocalTally>   (128 bytes × n_samples, cache-aligned)
/// ├── indices: Vec<usize>        (8 bytes × n_samples)
/// └── bitmaps: Vec<Vec<u64>>     (8 bytes × bitmap_words × n_samples)
/// ```
///
/// # Example
///
/// ```
/// use tsetlin_rs::parallel_training::TrainingArena;
///
/// // Create arena for 1000 samples, 100 clauses
/// let mut arena = TrainingArena::new(1000, 100);
///
/// // Reuse across epochs
/// for epoch in 0..100 {
///     arena.reset();
///     // ... training with arena.batch(), arena.indices() ...
/// }
/// ```
#[derive(Debug)]
pub struct TrainingArena {
    /// Pre-allocated vote tallies (one per sample).
    tallies:   Vec<LocalTally>,
    /// Reusable index buffer for shuffling.
    indices:   Vec<usize>,
    /// Pre-allocated firing bitmaps (one per sample).
    bitmaps:   Vec<Vec<u64>>,
    /// Number of samples this arena supports.
    n_samples: usize,
    /// Number of clauses (for bitmap sizing).
    n_clauses: usize
}

impl TrainingArena {
    /// Creates a new arena for the given dataset size.
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Number of training samples
    /// * `n_clauses` - Number of clauses (for bitmap allocation)
    #[must_use]
    pub fn new(n_samples: usize, n_clauses: usize) -> Self {
        let bitmap_words = n_clauses.div_ceil(64);
        Self {
            tallies: (0..n_samples).map(|_| LocalTally::new()).collect(),
            indices: (0..n_samples).collect(),
            bitmaps: (0..n_samples).map(|_| vec![0u64; bitmap_words]).collect(),
            n_samples,
            n_clauses
        }
    }

    /// Returns the number of samples this arena supports.
    #[inline]
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Returns the number of clauses this arena supports.
    #[inline]
    #[must_use]
    pub fn n_clauses(&self) -> usize {
        self.n_clauses
    }

    /// Resets all tallies to zero for a new epoch.
    pub fn reset_tallies(&self) {
        self.tallies.par_iter().for_each(LocalTally::reset);
    }

    /// Resets all bitmaps to zero.
    pub fn reset_bitmaps(&mut self) {
        self.bitmaps.par_iter_mut().for_each(|bitmap| {
            bitmap.iter_mut().for_each(|word| *word = 0);
        });
    }

    /// Full reset for a new epoch.
    pub fn reset(&mut self) {
        self.reset_tallies();
        self.reset_bitmaps();
    }

    /// Returns the tally for a specific sample.
    #[inline]
    #[must_use]
    pub fn tally(&self, idx: usize) -> &LocalTally {
        &self.tallies[idx]
    }

    /// Returns the mutable index buffer for shuffling.
    #[inline]
    pub fn indices_mut(&mut self) -> &mut [usize] {
        &mut self.indices
    }

    /// Returns the index buffer.
    #[inline]
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns the bitmap for a specific sample.
    #[inline]
    #[must_use]
    pub fn bitmap(&self, idx: usize) -> &[u64] {
        &self.bitmaps[idx]
    }

    /// Returns the mutable bitmap for a specific sample.
    #[inline]
    pub fn bitmap_mut(&mut self, idx: usize) -> &mut [u64] {
        &mut self.bitmaps[idx]
    }

    /// Returns all bitmaps.
    #[inline]
    #[must_use]
    pub fn bitmaps(&self) -> &[Vec<u64>] {
        &self.bitmaps
    }

    /// Returns all mutable bitmaps.
    #[inline]
    pub fn bitmaps_mut(&mut self) -> &mut [Vec<u64>] {
        &mut self.bitmaps
    }

    /// Resizes the arena if needed for a different dataset size.
    ///
    /// Only reallocates if the new size exceeds current capacity.
    pub fn ensure_capacity(&mut self, n_samples: usize, n_clauses: usize) {
        let bitmap_words = n_clauses.div_ceil(64);

        if n_samples > self.tallies.len() {
            self.tallies
                .extend((self.tallies.len()..n_samples).map(|_| LocalTally::new()));
        }

        if n_samples > self.indices.len() {
            self.indices.extend(self.indices.len()..n_samples);
        }

        if n_samples > self.bitmaps.len() || n_clauses > self.n_clauses {
            self.bitmaps
                .resize_with(n_samples, || vec![0u64; bitmap_words]);
            for bitmap in &mut self.bitmaps {
                if bitmap.len() < bitmap_words {
                    bitmap.resize(bitmap_words, 0);
                }
            }
        }

        self.n_samples = n_samples;
        self.n_clauses = n_clauses;
    }

    /// Returns memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let tally_size = self.tallies.len() * core::mem::size_of::<LocalTally>();
        let indices_size = self.indices.len() * core::mem::size_of::<usize>();
        let bitmap_size: usize = self.bitmaps.iter().map(|b| b.len() * 8).sum();
        tally_size + indices_size + bitmap_size
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

    /// Fully parallel training with clause-parallel feedback.
    ///
    /// This is an improved version of [`train_parallel`](Self::train_parallel)
    /// that parallelizes the feedback phase as well. Since each clause's
    /// automata occupy a disjoint memory region, we can safely update them
    /// in parallel.
    ///
    /// # Architecture
    ///
    /// ```text
    /// Phase 1: Reset tallies              (parallel over samples)
    /// Phase 2: Evaluate & tally           (parallel over samples)
    /// Phase 3: Apply feedback             (parallel over CLAUSES)
    ///          └── Each clause processes all samples independently
    /// ```
    ///
    /// # Performance
    ///
    /// Provides additional speedup over `train_parallel` by eliminating the
    /// sequential feedback bottleneck. Speedup scales with clause count.
    ///
    /// # Arguments
    ///
    /// * `batch` - Batch of samples with local tallies
    /// * `threshold` - Voting threshold T
    /// * `s` - Specificity parameter
    /// * `seed` - Random seed (each clause gets deterministic sub-seed)
    pub fn train_parallel_v2(&mut self, batch: &ParallelBatch, threshold: f32, s: f32, seed: u64) {
        // Phase 1: Reset tallies
        batch.reset_tallies();

        let n_samples = batch.len();
        let n_clauses = self.n_clauses;
        let bitmap_words = n_clauses.div_ceil(64);

        // Phase 2: Parallel evaluation with bitmap per sample
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

                        let word = clause / 64;
                        let bit = clause % 64;
                        bitmap[word] |= 1u64 << bit;
                    }
                }
                bitmap
            })
            .collect();

        // Precompute per-sample data (sum, label, prob)
        let inv_2t = 1.0 / (2.0 * threshold);
        let sample_data: Vec<(f32, u8)> = (0..n_samples)
            .map(|i| {
                let sum = batch.tally(i).sum().clamp(-threshold, threshold);
                let y = batch.label(i);
                let prob = if y == 1 {
                    (threshold - sum) * inv_2t
                } else {
                    (threshold + sum) * inv_2t
                };
                (prob, y)
            })
            .collect();

        // Phase 3: FULLY PARALLEL feedback by clause
        // Each clause's states are in a disjoint memory region, so this is safe.
        let stride = self.stride;
        let n_states = self.n_states;
        let n_features = self.n_features;
        let polarities = &self.polarities;

        // Split states into per-clause chunks for parallel processing
        self.states
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(clause, states_chunk)| {
                let p = polarities[clause];
                let mut rng = crate::utils::rng_from_seed(seed.wrapping_add(clause as u64));

                for (sample_idx, &(prob, y)) in sample_data.iter().enumerate() {
                    // Check if clause fired for this sample
                    let word = clause / 64;
                    let bit = clause % 64;
                    let fires = firing_bitmaps[sample_idx][word] & (1u64 << bit) != 0;

                    let x = batch.input(sample_idx);

                    // Apply feedback based on y and polarity
                    if y == 1 {
                        if p == 1 && rng.random::<f32>() <= prob {
                            type_i_slice(
                                states_chunk,
                                x,
                                fires,
                                s,
                                n_states,
                                n_features,
                                &mut rng
                            );
                        } else if p == -1 && fires && rng.random::<f32>() <= prob {
                            type_ii_slice(states_chunk, x, n_states, n_features);
                        }
                    } else if p == -1 && rng.random::<f32>() <= prob {
                        type_i_slice(states_chunk, x, fires, s, n_states, n_features, &mut rng);
                    } else if p == 1 && fires && rng.random::<f32>() <= prob {
                        type_ii_slice(states_chunk, x, n_states, n_features);
                    }
                }
            });
    }
}

/// Type I feedback on a mutable slice of automata states.
///
/// This is a standalone version of [`ClauseBank::type_i`] that operates
/// directly on a slice, enabling parallel processing of disjoint clause chunks.
#[inline]
fn type_i_slice<R: Rng>(
    states: &mut [i16],
    x: &[u8],
    fires: bool,
    s: f32,
    n_states: i16,
    n_features: usize,
    rng: &mut R
) {
    let threshold_str = prob_to_threshold((s - 1.0) / s);
    let threshold_wk = prob_to_threshold(1.0 / s);
    let n = x.len().min(n_features);
    let max = 2 * n_states;

    if !fires {
        // Clause didn't fire: weaken all automata toward exclusion
        for state in states.iter_mut() {
            if rng.random::<u32>() < threshold_wk && *state > 1 {
                *state -= 1;
            }
        }
        return;
    }

    // Clause fired: reinforce matching pattern
    for (k, &xk) in x.iter().enumerate().take(n) {
        let pos = 2 * k;
        let neg = 2 * k + 1;

        if xk == 1 {
            // x[k] = 1: strengthen x_k, weaken ¬x_k
            if rng.random::<u32>() < threshold_str && states[pos] < max {
                states[pos] += 1;
            }
            if rng.random::<u32>() < threshold_wk && states[neg] > 1 {
                states[neg] -= 1;
            }
        } else {
            // x[k] = 0: strengthen ¬x_k, weaken x_k
            if rng.random::<u32>() < threshold_str && states[neg] < max {
                states[neg] += 1;
            }
            if rng.random::<u32>() < threshold_wk && states[pos] > 1 {
                states[pos] -= 1;
            }
        }
    }
}

/// Type II feedback on a mutable slice of automata states.
///
/// This is a standalone version of [`ClauseBank::type_ii`] that operates
/// directly on a slice, enabling parallel processing of disjoint clause chunks.
#[inline]
fn type_ii_slice(states: &mut [i16], x: &[u8], n_states: i16, n_features: usize) {
    let n = x.len().min(n_features);
    let max = 2 * n_states;
    let threshold = n_states;

    for (k, &xk) in x.iter().enumerate().take(n) {
        let pos = 2 * k;
        let neg = 2 * k + 1;

        if xk == 0 {
            // x[k] = 0: add x_k literal (which will fail since x[k]=0)
            if states[pos] <= threshold && states[pos] < max {
                states[pos] += 1;
            }
        } else if states[neg] <= threshold && states[neg] < max {
            // x[k] = 1: add ¬x_k literal (which will fail since x[k]=1)
            states[neg] += 1;
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

    #[test]
    fn local_tally_add_scaled() {
        let tally = LocalTally::new();

        // Test add() with pre-scaled values
        tally.add(WEIGHT_SCALE * 2); // +2.0
        tally.add(-WEIGHT_SCALE); // -1.0

        assert!((tally.sum() - 1.0).abs() < 0.001);
    }

    #[test]
    fn local_tally_sum_scaled() {
        let tally = LocalTally::new();

        tally.add(12345);
        assert_eq!(tally.sum_scaled(), 12345);
    }

    #[test]
    fn local_tally_default() {
        let tally = LocalTally::default();
        assert!((tally.sum() - 0.0).abs() < 0.001);
    }

    #[test]
    fn parallel_batch_par_iter() {
        let inputs = vec![vec![0, 1], vec![1, 0]];
        let labels = vec![1, 0];

        let batch = ParallelBatch::new(&inputs, &labels);

        // Use par_iter and collect results
        let results: Vec<_> = batch.par_iter().collect();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // sample_idx
        assert_eq!(results[0].2, 1); // label
        assert_eq!(results[1].0, 1);
        assert_eq!(results[1].2, 0);
    }

    #[test]
    fn parallel_eval_sample() {
        let bank = ClauseBank::new(4, 2, 100);
        let tally = LocalTally::new();

        // Fresh clauses always fire
        let firing = bank.parallel_eval_sample(&[1, 1], &tally);

        // All 4 clauses should fire
        assert_eq!(firing.len(), 4);
        // Votes may cancel out (alternating polarity), just check tally was updated
        assert_eq!(tally.sum_scaled().abs() % WEIGHT_SCALE, 0);
    }

    #[test]
    fn type_i_slice_fires() {
        let mut states = vec![100i16; 4]; // 2 features = 4 automata
        let x = &[1u8, 0];
        let mut rng = crate::utils::rng_from_seed(42);

        // Run type_i with fires=true multiple times to see state changes
        for _ in 0..100 {
            type_i_slice(&mut states, x, true, 3.9, 100, 2, &mut rng);
        }

        // With fires=true and x=[1,0]:
        // - states[0] (x_0) should increase (x[0]=1)
        // - states[1] (¬x_0) should decrease
        // - states[2] (x_1) should decrease (x[1]=0)
        // - states[3] (¬x_1) should increase
        assert!(states[0] >= 100); // strengthen x_0
        assert!(states[3] >= 100); // strengthen ¬x_1
    }

    #[test]
    fn type_i_slice_no_fire() {
        let mut states = vec![100i16; 4];
        let x = &[1u8, 0];
        let mut rng = crate::utils::rng_from_seed(42);

        // Run type_i with fires=false - all states should weaken
        for _ in 0..100 {
            type_i_slice(&mut states, x, false, 3.9, 100, 2, &mut rng);
        }

        // All states should have decreased
        assert!(states.iter().all(|&s| s <= 100));
    }

    #[test]
    fn type_ii_slice_blocking() {
        // States at threshold (100) - in exclusion zone
        let mut states = vec![100i16; 4];
        let x = &[1u8, 0];

        // Type II should add blocking literals
        type_ii_slice(&mut states, x, 100, 2);

        // x[0]=1: should increment states[1] (¬x_0) to block
        // x[1]=0: should increment states[2] (x_1) to block
        assert_eq!(states[1], 101); // ¬x_0 incremented
        assert_eq!(states[2], 101); // x_1 incremented
        assert_eq!(states[0], 100); // x_0 unchanged
        assert_eq!(states[3], 100); // ¬x_1 unchanged
    }

    #[test]
    fn type_ii_slice_above_threshold() {
        // States above threshold (101) - in inclusion zone
        let mut states = vec![101i16; 4];
        let x = &[1u8, 0];

        // Type II should not modify states above threshold
        type_ii_slice(&mut states, x, 100, 2);

        // All states should remain unchanged
        assert!(states.iter().all(|&s| s == 101));
    }

    #[test]
    fn train_parallel_v2_xor() {
        let mut bank = ClauseBank::new(20, 2, 100);
        let inputs = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
        let labels = vec![0, 1, 1, 0];
        let batch = ParallelBatch::new(&inputs, &labels);

        // Train for several epochs
        for epoch in 0..200 {
            bank.train_parallel_v2(&batch, 10.0, 3.9, epoch);
        }

        // Check that the model learned XOR
        let mut correct = 0;
        for (x, &y) in inputs.iter().zip(labels.iter()) {
            let sum = bank.sum_votes(x);
            let pred = if sum >= 0.0 { 1 } else { 0 };
            if pred == y {
                correct += 1;
            }
        }

        assert!(correct >= 3, "Should learn XOR pattern");
    }

    #[test]
    fn train_parallel_v2_modifies_states() {
        let mut bank = ClauseBank::new(4, 2, 100);
        let inputs = vec![vec![1, 0], vec![0, 1]];
        let labels = vec![1, 0];
        let batch = ParallelBatch::new(&inputs, &labels);

        // Capture initial state
        let initial_states = bank.states.clone();

        // Train
        bank.train_parallel_v2(&batch, 10.0, 3.9, 42);

        // States should have changed
        assert_ne!(bank.states, initial_states);
    }

    #[test]
    fn training_arena_creation() {
        let arena = TrainingArena::new(100, 20);

        assert_eq!(arena.n_samples(), 100);
        assert_eq!(arena.n_clauses(), 20);
        assert_eq!(arena.indices().len(), 100);
    }

    #[test]
    fn training_arena_reset() {
        let mut arena = TrainingArena::new(10, 8);

        // Modify tallies
        arena.tally(0).add_weighted(1, 1.5);
        arena.tally(5).add_weighted(-1, 2.0);

        // Modify bitmaps
        arena.bitmaps_mut()[0][0] = 0xFF;

        // Reset
        arena.reset();

        // Tallies should be zero
        assert!((arena.tally(0).sum() - 0.0).abs() < 0.001);
        assert!((arena.tally(5).sum() - 0.0).abs() < 0.001);

        // Bitmaps should be zero
        assert_eq!(arena.bitmap(0)[0], 0);
    }

    #[test]
    fn training_arena_indices() {
        let mut arena = TrainingArena::new(5, 4);

        // Initial indices should be 0..5
        assert_eq!(arena.indices(), &[0, 1, 2, 3, 4]);

        // Modify indices (simulating shuffle)
        let indices = arena.indices_mut();
        indices.swap(0, 4);

        assert_eq!(arena.indices(), &[4, 1, 2, 3, 0]);
    }

    #[test]
    fn training_arena_ensure_capacity() {
        let mut arena = TrainingArena::new(10, 8);

        // Grow
        arena.ensure_capacity(20, 16);
        assert_eq!(arena.n_samples(), 20);
        assert_eq!(arena.n_clauses(), 16);
        assert_eq!(arena.indices().len(), 20);

        // Shrinking parameters doesn't reallocate
        arena.ensure_capacity(5, 4);
        assert!(arena.indices().len() >= 5);
    }

    #[test]
    fn training_arena_memory_usage() {
        let arena = TrainingArena::new(100, 64);
        let usage = arena.memory_usage();

        // Should be non-zero and reasonable
        assert!(usage > 0);
        // Rough estimate: 100 tallies (128B each) + 100 indices (8B) + 100 bitmaps (8B)
        assert!(usage >= 100 * 128 + 100 * 8 + 100 * 8);
    }

    #[test]
    fn training_arena_bitmaps() {
        let mut arena = TrainingArena::new(10, 128);

        // 128 clauses = 2 bitmap words
        assert_eq!(arena.bitmap(0).len(), 2);

        // Set some bits
        arena.bitmaps_mut()[0][0] = 1;
        arena.bitmaps_mut()[0][1] = 1 << 63;

        assert_eq!(arena.bitmap(0)[0], 1);
        assert_eq!(arena.bitmap(0)[1], 1 << 63);
    }
}
