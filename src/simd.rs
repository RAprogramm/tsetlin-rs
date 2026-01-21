//! SIMD-optimized clause evaluation.
//!
//! Requires the `simd` feature and nightly Rust for `portable_simd`.

#![cfg(feature = "simd")]

use std::simd::{Simd, cmp::SimdPartialEq, u8x32};

/// # Overview
///
/// Evaluates clause using SIMD operations for vectorized processing.
///
/// Processes 32 features at a time using AVX2/SSE instructions.
/// Falls back to scalar for remaining features.
///
/// # Arguments
///
/// * `actions_include` - Byte array: 1 if include[k] is active, 0 otherwise
/// * `actions_negated` - Byte array: 1 if negated[k] is active, 0 otherwise
/// * `x` - Input features (binary 0/1)
///
/// # Examples
///
/// ```ignore
/// use tsetlin_rs::simd::evaluate_simd;
///
/// let include = vec![0u8; 64];
/// let negated = vec![0u8; 64];
/// let x = vec![1u8; 64];
///
/// let result = evaluate_simd(&include, &negated, &x);
/// ```
pub fn evaluate_simd(actions_include: &[u8], actions_negated: &[u8], x: &[u8]) -> bool {
    debug_assert_eq!(actions_include.len(), x.len());
    debug_assert_eq!(actions_negated.len(), x.len());

    let n = x.len();
    let chunks = n / 32;
    let zeros = u8x32::splat(0);
    let ones = u8x32::splat(1);

    for chunk in 0..chunks {
        let offset = chunk * 32;

        let xi = u8x32::from_slice(&x[offset..offset + 32]);
        let inc = u8x32::from_slice(&actions_include[offset..offset + 32]);
        let neg = u8x32::from_slice(&actions_negated[offset..offset + 32]);

        let x_is_zero = xi.simd_eq(zeros);
        let x_is_one = xi.simd_eq(ones);

        let inc_active = inc.simd_eq(ones);
        let neg_active = neg.simd_eq(ones);

        let inc_violation = inc_active & x_is_zero;
        let neg_violation = neg_active & x_is_one;

        if inc_violation.any() || neg_violation.any() {
            return false;
        }
    }

    for k in (chunks * 32)..n {
        let include_active = actions_include[k] == 1;
        let negated_active = actions_negated[k] == 1;

        if include_active && x[k] == 0 {
            return false;
        }
        if negated_active && x[k] == 1 {
            return false;
        }
    }

    true
}

/// # Overview
///
/// Converts automaton states to action bytes for SIMD processing.
///
/// Returns two vectors: (include_actions, negated_actions).
/// Each byte is 1 if the action is active, 0 otherwise.
pub fn states_to_actions(states: &[i16], n_states: i16) -> (Vec<u8>, Vec<u8>) {
    let n_features = states.len() / 2;
    let mut include = Vec::with_capacity(n_features);
    let mut negated = Vec::with_capacity(n_features);

    for k in 0..n_features {
        include.push(if states[2 * k] > n_states { 1 } else { 0 });
        negated.push(if states[2 * k + 1] > n_states { 1 } else { 0 });
    }

    (include, negated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_empty_clause() {
        let include = vec![0u8; 64];
        let negated = vec![0u8; 64];
        let x = vec![1u8; 64];

        assert!(evaluate_simd(&include, &negated, &x));
    }

    #[test]
    fn evaluate_with_violation() {
        let mut include = vec![0u8; 64];
        let negated = vec![0u8; 64];
        let mut x = vec![1u8; 64];

        include[10] = 1;
        x[10] = 0;

        assert!(!evaluate_simd(&include, &negated, &x));
    }

    #[test]
    fn evaluate_no_violation() {
        let mut include = vec![0u8; 64];
        let negated = vec![0u8; 64];
        let x = vec![1u8; 64];

        include[10] = 1;

        assert!(evaluate_simd(&include, &negated, &x));
    }

    #[test]
    fn states_conversion() {
        let states = vec![50, 150, 150, 50];
        let (include, negated) = states_to_actions(&states, 100);

        assert_eq!(include, vec![0, 1]);
        assert_eq!(negated, vec![1, 0]);
    }
}
