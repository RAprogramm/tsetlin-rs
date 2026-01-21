//! Utility functions for random number generation and helpers.

use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Fast RNG suitable for no_std environments.
pub type FastRng = Xoshiro256PlusPlus;

/// # Overview
///
/// Creates a fast RNG seeded from a u64 value.
///
/// # Examples
///
/// ```
/// use tsetlin_rs::utils::rng_from_seed;
///
/// let mut rng = rng_from_seed(42);
/// ```
#[inline]
pub fn rng_from_seed(seed: u64) -> FastRng {
    use rand::SeedableRng;
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// # Overview
///
/// Creates a fast RNG with entropy from thread-local RNG.
///
/// # Examples
///
/// ```
/// use tsetlin_rs::utils::rng_from_entropy;
///
/// let mut rng = rng_from_entropy();
/// ```
#[cfg(feature = "std")]
#[inline]
pub fn rng_from_entropy() -> FastRng {
    use rand::SeedableRng;
    Xoshiro256PlusPlus::from_rng(&mut rand::rng())
}

/// # Overview
///
/// Generates a random f32 in [0, 1).
#[inline]
pub fn random_f32<R: Rng>(rng: &mut R) -> f32 {
    rng.random::<f32>()
}

/// # Overview
///
/// Performs a Bernoulli trial with given probability.
#[inline]
pub fn bernoulli<R: Rng>(rng: &mut R, probability: f32) -> bool {
    random_f32(rng) < probability
}

/// # Overview
///
/// Shuffles a slice in-place using Fisher-Yates algorithm.
#[inline]
pub fn shuffle<T, R: Rng>(slice: &mut [T], rng: &mut R) {
    let len = slice.len();
    for i in (1..len).rev() {
        let j = rng.random_range(0..=i as u64) as usize;
        slice.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_deterministic() {
        let mut rng1 = rng_from_seed(42);
        let mut rng2 = rng_from_seed(42);

        for _ in 0..100 {
            assert_eq!(rng1.random::<u64>(), rng2.random::<u64>());
        }
    }

    #[test]
    fn bernoulli_probability() {
        let mut rng = rng_from_seed(42);
        let mut count = 0;
        let trials = 10000;

        for _ in 0..trials {
            if bernoulli(&mut rng, 0.5) {
                count += 1;
            }
        }

        let ratio = count as f64 / trials as f64;
        assert!((ratio - 0.5).abs() < 0.05);
    }

    #[test]
    fn shuffle_preserves_elements() {
        let mut data = vec![1, 2, 3, 4, 5];
        let original = data.clone();
        let mut rng = rng_from_seed(42);

        shuffle(&mut data, &mut rng);

        data.sort();
        assert_eq!(data, original);
    }
}
