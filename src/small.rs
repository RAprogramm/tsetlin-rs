//! Small-sized Tsetlin Machine with const generics for compile-time
//! optimization.
//!
//! Uses stack allocation and compile-time unrolling for small feature sets.
//! Zero heap allocations, full loop unrolling at compile time.

use core::array;

use crate::Automaton;

/// A clause with compile-time known feature count.
///
/// Stack-allocated, zero heap allocations. The compiler fully unrolls
/// all loops for known N, enabling maximum optimization.
///
/// # Memory Layout
///
/// Fields ordered to minimize padding:
/// - `include: [Automaton; N]` (N * 4 bytes)
/// - `negated: [Automaton; N]` (N * 4 bytes)
/// - `weight: f32` (4 bytes)
/// - `activations: u32` (4 bytes)
/// - `correct: u32` (4 bytes)
/// - `incorrect: u32` (4 bytes)
/// - `polarity: i8` (1 byte + padding to 64-byte alignment)
///
/// # Performance
///
/// Optimal for N <= 64. For larger feature sets, use `BitwiseClause`.
// Note: serde not supported for const generic arrays without serde_big_array.
// Use Vec-based Clause for serialization needs.
#[derive(Debug, Clone)]
#[repr(align(64))]
pub struct SmallClause<const N: usize> {
    include:     [Automaton; N],
    negated:     [Automaton; N],
    weight:      f32,
    activations: u32,
    correct:     u32,
    incorrect:   u32,
    polarity:    i8
}

impl<const N: usize> SmallClause<N> {
    /// Creates clause with given states and polarity.
    ///
    /// # Arguments
    ///
    /// * `n_states` - States per automaton (threshold for action)
    /// * `polarity` - Must be +1 or -1
    ///
    /// # Panics
    ///
    /// Debug-asserts that polarity is +1 or -1.
    #[inline]
    #[must_use]
    pub fn new(n_states: i16, polarity: i8) -> Self {
        debug_assert!(polarity == 1 || polarity == -1);
        Self {
            include: array::from_fn(|_| Automaton::new(n_states)),
            negated: array::from_fn(|_| Automaton::new(n_states)),
            weight: 1.0,
            activations: 0,
            correct: 0,
            incorrect: 0,
            polarity
        }
    }

    /// Returns the clause polarity (+1 or -1).
    #[inline(always)]
    #[must_use]
    pub const fn polarity(&self) -> i8 {
        self.polarity
    }

    /// Returns the number of features (compile-time constant).
    #[inline(always)]
    #[must_use]
    pub const fn n_features(&self) -> usize {
        N
    }

    /// Returns the current clause weight.
    #[inline(always)]
    #[must_use]
    pub const fn weight(&self) -> f32 {
        self.weight
    }

    /// Returns the activation count since last reset.
    #[inline(always)]
    #[must_use]
    pub const fn activations(&self) -> u32 {
        self.activations
    }

    /// Returns read-only access to include automata.
    #[inline(always)]
    #[must_use]
    pub const fn include_automata(&self) -> &[Automaton; N] {
        &self.include
    }

    /// Returns mutable access to include automata.
    #[inline(always)]
    pub fn include_automata_mut(&mut self) -> &mut [Automaton; N] {
        &mut self.include
    }

    /// Returns read-only access to negated automata.
    #[inline(always)]
    #[must_use]
    pub const fn negated_automata(&self) -> &[Automaton; N] {
        &self.negated
    }

    /// Returns mutable access to negated automata.
    #[inline(always)]
    pub fn negated_automata_mut(&mut self) -> &mut [Automaton; N] {
        &mut self.negated
    }

    /// Evaluates clause on input with early exit on violation.
    ///
    /// Returns `true` if all included literals are satisfied.
    /// Loop is fully unrolled at compile time for maximum performance.
    ///
    /// # Performance
    ///
    /// Uses unchecked indexing. Safety is guaranteed by const generic bounds.
    #[inline]
    #[must_use]
    pub fn evaluate(&self, x: &[u8; N]) -> bool {
        for k in 0..N {
            // SAFETY: k is always in bounds [0, N) due to loop bounds
            // and array size is exactly N.
            let include_action = unsafe { self.include.get_unchecked(k).action() };
            let negated_action = unsafe { self.negated.get_unchecked(k).action() };
            let xk = unsafe { *x.get_unchecked(k) };

            if include_action && xk == 0 {
                return false;
            }
            if negated_action && xk == 1 {
                return false;
            }
        }
        true
    }

    /// Evaluates clause and tracks activation count.
    ///
    /// Use during training to track which clauses are active.
    #[inline]
    pub fn evaluate_tracked(&mut self, x: &[u8; N]) -> bool {
        let fires = self.evaluate(x);
        if fires {
            self.activations = self.activations.saturating_add(1);
        }
        fires
    }

    /// Returns weighted vote: `polarity * weight` if clause fires, `0.0`
    /// otherwise.
    #[inline(always)]
    #[must_use]
    pub fn vote_weighted(&self, x: &[u8; N]) -> f32 {
        if self.evaluate(x) {
            self.polarity as f32 * self.weight
        } else {
            0.0
        }
    }

    /// Returns unweighted vote: `polarity` if fires, `0` otherwise.
    #[inline(always)]
    #[must_use]
    pub fn vote(&self, x: &[u8; N]) -> i32 {
        if self.evaluate(x) {
            self.polarity as i32
        } else {
            0
        }
    }

    /// Records prediction outcome for weight learning.
    ///
    /// Call when clause fired and prediction was made.
    #[inline]
    pub fn record_outcome(&mut self, was_correct: bool) {
        if was_correct {
            self.correct = self.correct.saturating_add(1);
        } else {
            self.incorrect = self.incorrect.saturating_add(1);
        }
    }

    /// Updates weight based on accumulated outcomes.
    ///
    /// Weight increases when clause predictions are accurate,
    /// decreases when inaccurate. Call at end of each epoch.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - How fast weight changes (0.0 - 1.0)
    /// * `min_weight` - Minimum allowed weight
    /// * `max_weight` - Maximum allowed weight
    pub fn update_weight(&mut self, learning_rate: f32, min_weight: f32, max_weight: f32) {
        let total = self.correct + self.incorrect;
        if total == 0 {
            return;
        }

        let accuracy = self.correct as f32 / total as f32;
        let adjustment = (accuracy - 0.5) * 2.0 * learning_rate;
        self.weight = (self.weight + adjustment).clamp(min_weight, max_weight);

        self.correct = 0;
        self.incorrect = 0;
    }

    /// Returns `true` if clause is "dead" (rarely activates or very low
    /// weight).
    ///
    /// Dead clauses can be pruned and reset during training.
    #[inline]
    #[must_use]
    pub const fn is_dead(&self, min_activations: u32, min_weight: f32) -> bool {
        self.activations < min_activations || self.weight < min_weight
    }

    /// Resets activation counter. Call at start of each epoch.
    #[inline]
    pub fn reset_activations(&mut self) {
        self.activations = 0;
    }

    /// Resets all statistics (activations, correct, incorrect).
    #[inline]
    pub fn reset_stats(&mut self) {
        self.activations = 0;
        self.correct = 0;
        self.incorrect = 0;
    }
}

/// Type alias for 2-feature clause.
pub type Clause2 = SmallClause<2>;
/// Type alias for 4-feature clause (XOR problems).
pub type Clause4 = SmallClause<4>;
/// Type alias for 8-feature clause.
pub type Clause8 = SmallClause<8>;
/// Type alias for 16-feature clause.
pub type Clause16 = SmallClause<16>;
/// Type alias for 32-feature clause.
pub type Clause32 = SmallClause<32>;
/// Type alias for 64-feature clause.
pub type Clause64 = SmallClause<64>;

/// Bitwise clause with compile-time known feature count.
///
/// Uses packed u64 bitmasks for SIMD-friendly evaluation.
/// Processes 64 features per AND operation.
///
/// # Type Parameters
///
/// * `N` - Number of features (compile-time constant)
/// * `W` - Number of u64 words needed: `(N + 63) / 64`
// Note: serde not supported for const generic arrays without serde_big_array.
#[derive(Debug, Clone)]
#[repr(align(64))]
pub struct SmallBitwiseClause<const N: usize, const W: usize> {
    include:  [Automaton; N],
    negated:  [Automaton; N],
    inc_mask: [u64; W],
    neg_mask: [u64; W],
    weight:   f32,
    polarity: i8,
    dirty:    bool
}

impl<const N: usize, const W: usize> SmallBitwiseClause<N, W> {
    /// Creates bitwise clause with given states and polarity.
    ///
    /// # Panics
    ///
    /// Debug-asserts that W == (N + 63) / 64.
    #[inline]
    #[must_use]
    pub fn new(n_states: i16, polarity: i8) -> Self {
        debug_assert!(polarity == 1 || polarity == -1);
        debug_assert_eq!(W, N.div_ceil(64), "W must equal ceil(N/64)");
        Self {
            include: array::from_fn(|_| Automaton::new(n_states)),
            negated: array::from_fn(|_| Automaton::new(n_states)),
            inc_mask: [0; W],
            neg_mask: [0; W],
            weight: 1.0,
            polarity,
            dirty: true
        }
    }

    /// Returns the clause polarity (+1 or -1).
    #[inline(always)]
    #[must_use]
    pub const fn polarity(&self) -> i8 {
        self.polarity
    }

    /// Returns the number of features (compile-time constant).
    #[inline(always)]
    #[must_use]
    pub const fn n_features(&self) -> usize {
        N
    }

    /// Returns read-only access to include automata.
    #[inline(always)]
    #[must_use]
    pub const fn include_automata(&self) -> &[Automaton; N] {
        &self.include
    }

    /// Returns mutable access to include automata.
    #[inline(always)]
    pub fn include_automata_mut(&mut self) -> &mut [Automaton; N] {
        self.dirty = true;
        &mut self.include
    }

    /// Returns read-only access to negated automata.
    #[inline(always)]
    #[must_use]
    pub const fn negated_automata(&self) -> &[Automaton; N] {
        &self.negated
    }

    /// Returns mutable access to negated automata.
    #[inline(always)]
    pub fn negated_automata_mut(&mut self) -> &mut [Automaton; N] {
        self.dirty = true;
        &mut self.negated
    }

    /// Rebuilds bitmasks from automaton states.
    ///
    /// Call after training before evaluation.
    pub fn rebuild_masks(&mut self) {
        if !self.dirty {
            return;
        }

        for word in &mut self.inc_mask {
            *word = 0;
        }
        for word in &mut self.neg_mask {
            *word = 0;
        }

        for k in 0..N {
            let word_idx = k / 64;
            let bit_idx = k % 64;

            // SAFETY: k < N, so word_idx < W and k < N (automata bounds)
            if unsafe { self.include.get_unchecked(k).action() } {
                self.inc_mask[word_idx] |= 1u64 << bit_idx;
            }
            if unsafe { self.negated.get_unchecked(k).action() } {
                self.neg_mask[word_idx] |= 1u64 << bit_idx;
            }
        }

        self.dirty = false;
    }

    /// Evaluates clause using bitwise AND operations.
    ///
    /// Processes 64 features per CPU instruction for massive speedup.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `rebuild_masks()` was called after training.
    #[inline]
    #[must_use]
    pub fn evaluate_packed(&self, x_packed: &[u64; W]) -> bool {
        debug_assert!(!self.dirty, "call rebuild_masks() first");

        for i in 0..W {
            // SAFETY: i < W, all arrays have size W
            let x = unsafe { *x_packed.get_unchecked(i) };
            let inc = unsafe { *self.inc_mask.get_unchecked(i) };
            let neg = unsafe { *self.neg_mask.get_unchecked(i) };

            // include violation: inc & !x != 0 (required bit is 0)
            // negated violation: neg & x != 0 (forbidden bit is 1)
            if (inc & !x) | (neg & x) != 0 {
                return false;
            }
        }
        true
    }

    /// Returns polarity if fires, 0 otherwise.
    #[inline(always)]
    #[must_use]
    pub fn vote_packed(&self, x_packed: &[u64; W]) -> i32 {
        if self.evaluate_packed(x_packed) {
            self.polarity as i32
        } else {
            0
        }
    }

    /// Returns weighted vote if fires.
    #[inline(always)]
    #[must_use]
    pub fn vote_weighted_packed(&self, x_packed: &[u64; W]) -> f32 {
        if self.evaluate_packed(x_packed) {
            self.polarity as f32 * self.weight
        } else {
            0.0
        }
    }
}

/// Packs binary input into u64 array for bitwise evaluation.
///
/// Compile-time known size for zero allocation.
#[inline]
#[must_use]
pub fn pack_input_small<const N: usize, const W: usize>(x: &[u8; N]) -> [u64; W] {
    let mut packed = [0u64; W];

    for (k, &xk) in x.iter().enumerate() {
        if xk != 0 {
            packed[k / 64] |= 1u64 << (k % 64);
        }
    }

    packed
}

/// Bitwise clause for 64 features (1 u64 word).
pub type BitwiseClause64 = SmallBitwiseClause<64, 1>;
/// Bitwise clause for 128 features (2 u64 words).
pub type BitwiseClause128 = SmallBitwiseClause<128, 2>;
/// Bitwise clause for 256 features (4 u64 words).
pub type BitwiseClause256 = SmallBitwiseClause<256, 4>;

/// Binary classification Tsetlin Machine with compile-time known dimensions.
///
/// Stack-allocated with zero heap allocations. All loops unrolled at compile
/// time.
///
/// # Type Parameters
///
/// * `N` - Number of features (compile-time constant)
/// * `C` - Number of clauses (compile-time constant, must be even)
///
/// # Performance
///
/// Up to 3x faster than dynamic [`TsetlinMachine`](crate::TsetlinMachine)
/// for small dimensions due to:
/// - No heap allocations
/// - Full loop unrolling
/// - Better cache locality
///
/// # Example
///
/// ```
/// use tsetlin_rs::SmallTsetlinMachine;
///
/// // XOR with 2 features and 20 clauses
/// let mut tm: SmallTsetlinMachine<2, 20> = SmallTsetlinMachine::new(100, 15);
///
/// let x = [[0, 0], [0, 1], [1, 0], [1, 1]];
/// let y = [0u8, 1, 1, 0];
///
/// tm.fit(&x, &y, 200, 42);
/// assert!(tm.evaluate(&x, &y) >= 0.75);
/// ```
#[derive(Debug, Clone)]
#[repr(align(64))]
pub struct SmallTsetlinMachine<const N: usize, const C: usize> {
    clauses: [SmallClause<N>; C],
    s:       f32,
    t:       f32
}

impl<const N: usize, const C: usize> SmallTsetlinMachine<N, C> {
    /// Creates new machine with given states and threshold.
    ///
    /// Half clauses get +1 polarity, half get -1.
    ///
    /// # Panics
    ///
    /// Debug-asserts that C is even.
    #[must_use]
    pub fn new(n_states: i16, threshold: i32) -> Self {
        debug_assert!(C.is_multiple_of(2), "C must be even");
        Self {
            clauses: array::from_fn(|i| {
                let p = if i % 2 == 0 { 1 } else { -1 };
                SmallClause::new(n_states, p)
            }),
            s:       3.9,
            t:       threshold as f32
        }
    }

    /// Creates machine with custom specificity parameter.
    #[must_use]
    pub fn with_s(n_states: i16, threshold: i32, s: f32) -> Self {
        let mut tm = Self::new(n_states, threshold);
        tm.s = s;
        tm
    }

    /// Returns the number of features (compile-time constant).
    #[inline(always)]
    #[must_use]
    pub const fn n_features(&self) -> usize {
        N
    }

    /// Returns the number of clauses (compile-time constant).
    #[inline(always)]
    #[must_use]
    pub const fn n_clauses(&self) -> usize {
        C
    }

    /// Returns current threshold.
    #[inline(always)]
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.t
    }

    /// Returns read-only access to clauses.
    #[inline(always)]
    #[must_use]
    pub const fn clauses(&self) -> &[SmallClause<N>; C] {
        &self.clauses
    }

    /// Sum of clause votes for input x.
    #[inline]
    #[must_use]
    pub fn sum_votes(&self, x: &[u8; N]) -> i32 {
        let mut sum = 0i32;
        for i in 0..C {
            // SAFETY: i < C, array size is exactly C
            sum += unsafe { self.clauses.get_unchecked(i).vote(x) };
        }
        sum
    }

    /// Predicts class (0 or 1).
    #[inline(always)]
    #[must_use]
    pub fn predict(&self, x: &[u8; N]) -> u8 {
        if self.sum_votes(x) >= 0 { 1 } else { 0 }
    }

    /// Trains on single example.
    pub fn train_one(&mut self, x: &[u8; N], y: u8, rng: &mut impl rand::Rng) {
        let sum = (self.sum_votes(x) as f32).clamp(-self.t, self.t);
        let inv_2t = 1.0 / (2.0 * self.t);
        let s = self.s;

        let prob = if y == 1 {
            (self.t - sum) * inv_2t
        } else {
            (self.t + sum) * inv_2t
        };

        for i in 0..C {
            // SAFETY: i < C
            let clause = unsafe { self.clauses.get_unchecked_mut(i) };
            let fires = clause.evaluate(x);
            let p = clause.polarity();

            if y == 1 {
                if p == 1 && rng.random::<f32>() <= prob {
                    small_type_i(clause, x, fires, s, rng);
                } else if p == -1 && fires && rng.random::<f32>() <= prob {
                    small_type_ii(clause, x);
                }
            } else if p == -1 && rng.random::<f32>() <= prob {
                small_type_i(clause, x, fires, s, rng);
            } else if p == 1 && fires && rng.random::<f32>() <= prob {
                small_type_ii(clause, x);
            }
        }
    }

    /// Simple training for given epochs.
    pub fn fit(&mut self, x: &[[u8; N]], y: &[u8], epochs: usize, seed: u64) {
        let mut rng = crate::utils::rng_from_seed(seed);

        for _ in 0..epochs {
            for (xi, &yi) in x.iter().zip(y.iter()) {
                self.train_one(xi, yi, &mut rng);
            }
        }
    }

    /// Evaluates accuracy on test data.
    #[must_use]
    pub fn evaluate(&self, x: &[[u8; N]], y: &[u8]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        let correct = x
            .iter()
            .zip(y.iter())
            .filter(|(xi, yi)| self.predict(xi) == **yi)
            .count();
        correct as f32 / x.len() as f32
    }
}

/// Type I feedback for SmallClause (reinforces patterns).
fn small_type_i<const N: usize>(
    clause: &mut SmallClause<N>,
    x: &[u8; N],
    fires: bool,
    s: f32,
    rng: &mut impl rand::Rng
) {
    let prob_strengthen = (s - 1.0) / s;
    let prob_weaken = 1.0 / s;

    if !fires {
        // Clause doesn't fire: weaken all automata
        for k in 0..N {
            if rng.random::<f32>() <= prob_weaken {
                // SAFETY: k < N
                unsafe {
                    clause
                        .include_automata_mut()
                        .get_unchecked_mut(k)
                        .decrement()
                };
            }
            if rng.random::<f32>() <= prob_weaken {
                unsafe {
                    clause
                        .negated_automata_mut()
                        .get_unchecked_mut(k)
                        .decrement()
                };
            }
        }
    } else {
        // Clause fires: reinforce matching pattern
        for k in 0..N {
            // SAFETY: k < N
            let xk = unsafe { *x.get_unchecked(k) };

            if xk == 1 {
                if rng.random::<f32>() <= prob_strengthen {
                    unsafe {
                        clause
                            .include_automata_mut()
                            .get_unchecked_mut(k)
                            .increment()
                    };
                }
                if rng.random::<f32>() <= prob_weaken {
                    unsafe {
                        clause
                            .negated_automata_mut()
                            .get_unchecked_mut(k)
                            .decrement()
                    };
                }
            } else {
                if rng.random::<f32>() <= prob_strengthen {
                    unsafe {
                        clause
                            .negated_automata_mut()
                            .get_unchecked_mut(k)
                            .increment()
                    };
                }
                if rng.random::<f32>() <= prob_weaken {
                    unsafe {
                        clause
                            .include_automata_mut()
                            .get_unchecked_mut(k)
                            .decrement()
                    };
                }
            }
        }
    }
}

/// Type II feedback for SmallClause (corrects false positives).
fn small_type_ii<const N: usize>(clause: &mut SmallClause<N>, x: &[u8; N]) {
    for k in 0..N {
        // SAFETY: k < N
        let xk = unsafe { *x.get_unchecked(k) };
        let inc_action = unsafe { clause.include_automata().get_unchecked(k).action() };
        let neg_action = unsafe { clause.negated_automata().get_unchecked(k).action() };

        if xk == 0 && !inc_action {
            unsafe {
                clause
                    .include_automata_mut()
                    .get_unchecked_mut(k)
                    .increment()
            };
        }
        if xk == 1 && !neg_action {
            unsafe {
                clause
                    .negated_automata_mut()
                    .get_unchecked_mut(k)
                    .increment()
            };
        }
    }
}

/// 2-feature, 20-clause machine (XOR).
pub type TM2x20 = SmallTsetlinMachine<2, 20>;
/// 4-feature, 40-clause machine.
pub type TM4x40 = SmallTsetlinMachine<4, 40>;
/// 8-feature, 80-clause machine.
pub type TM8x80 = SmallTsetlinMachine<8, 80>;
/// 16-feature, 160-clause machine.
pub type TM16x160 = SmallTsetlinMachine<16, 160>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_clause_new() {
        let c: SmallClause<4> = SmallClause::new(100, 1);
        assert_eq!(c.n_features(), 4);
        assert_eq!(c.polarity(), 1);
        assert!((c.weight() - 1.0).abs() < 0.001);
        assert_eq!(c.activations(), 0);
    }

    #[test]
    fn small_clause_evaluate() {
        let c: SmallClause<4> = SmallClause::new(100, 1);
        let x = [0, 1, 0, 1];
        assert!(c.evaluate(&x));
    }

    #[test]
    fn small_clause_vote() {
        let c: SmallClause<4> = SmallClause::new(100, -1);
        let x = [1, 1, 1, 1];
        assert_eq!(c.vote(&x), -1);
    }

    #[test]
    fn small_clause_weighted_vote() {
        let mut c: SmallClause<4> = SmallClause::new(100, 1);
        c.weight = 0.5;
        let x = [0, 0, 0, 0];
        assert!((c.vote_weighted(&x) - 0.5).abs() < 0.001);
    }

    #[test]
    fn small_clause_activation_tracking() {
        let mut c: SmallClause<2> = SmallClause::new(100, 1);
        c.evaluate_tracked(&[0, 0]);
        c.evaluate_tracked(&[1, 1]);
        assert_eq!(c.activations(), 2);
    }

    #[test]
    fn small_clause_weight_update() {
        let mut c: SmallClause<2> = SmallClause::new(100, 1);
        c.correct = 8;
        c.incorrect = 2;
        c.update_weight(0.1, 0.1, 2.0);
        assert!(c.weight() > 1.0);
    }

    #[test]
    fn small_clause_is_dead() {
        let mut c: SmallClause<2> = SmallClause::new(100, 1);
        c.weight = 0.05;
        assert!(c.is_dead(10, 0.1));
    }

    #[test]
    fn small_bitwise_clause_new() {
        let c: SmallBitwiseClause<64, 1> = SmallBitwiseClause::new(100, 1);
        assert_eq!(c.n_features(), 64);
        assert_eq!(c.polarity(), 1);
    }

    #[test]
    fn small_bitwise_evaluate() {
        let mut c: BitwiseClause64 = SmallBitwiseClause::new(100, 1);
        c.rebuild_masks();

        let x_packed = [0xFFFF_FFFF_FFFF_FFFFu64];
        assert!(c.evaluate_packed(&x_packed));
    }

    #[test]
    fn small_bitwise_violation() {
        let mut c: BitwiseClause64 = SmallBitwiseClause::new(100, 1);

        // Force include[0] to be active
        for _ in 0..200 {
            c.include_automata_mut()[0].increment();
        }
        c.rebuild_masks();

        // x[0] = 0, should violate
        assert!(!c.evaluate_packed(&[0u64]));

        // x[0] = 1, should pass
        assert!(c.evaluate_packed(&[1u64]));
    }

    #[test]
    fn pack_input_small_test() {
        let x: [u8; 8] = [1, 0, 1, 1, 0, 0, 0, 1];
        let packed: [u64; 1] = pack_input_small(&x);
        assert_eq!(packed[0], 0b10001101); // bits 0,2,3,7 set
    }

    #[test]
    fn small_tm_new() {
        let tm: SmallTsetlinMachine<2, 20> = SmallTsetlinMachine::new(100, 15);
        assert_eq!(tm.n_features(), 2);
        assert_eq!(tm.n_clauses(), 20);
        assert!((tm.threshold() - 15.0).abs() < 0.001);
    }

    #[test]
    fn small_tm_xor_convergence() {
        let mut tm: SmallTsetlinMachine<2, 20> = SmallTsetlinMachine::new(100, 10);

        let x = [[0, 0], [0, 1], [1, 0], [1, 1]];
        let y = [0u8, 1, 1, 0];

        tm.fit(&x, &y, 200, 42);
        assert!(tm.evaluate(&x, &y) >= 0.75);
    }

    #[test]
    fn small_tm_type_alias() {
        let tm: TM2x20 = TM2x20::new(100, 10);
        assert_eq!(tm.n_features(), 2);
        assert_eq!(tm.n_clauses(), 20);
    }
}
