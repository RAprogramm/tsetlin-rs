//! Configuration and builder for Tsetlin Machine.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// # Overview
///
/// Configuration parameters for a Tsetlin Machine.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[must_use]
pub struct Config {
    pub n_clauses:  usize,
    pub n_features: usize,
    pub n_states:   i16,
    pub s:          f32
}

impl Config {
    /// # Overview
    ///
    /// Creates a new ConfigBuilder.
    #[inline]
    #[must_use]
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }

    /// # Overview
    ///
    /// Validates configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.n_clauses == 0 {
            return Err(Error::MissingClauses);
        }
        if !self.n_clauses.is_multiple_of(2) {
            return Err(Error::OddClauses);
        }
        if self.n_features == 0 {
            return Err(Error::MissingFeatures);
        }
        if self.s <= 1.0 {
            return Err(Error::InvalidSpecificity);
        }
        Ok(())
    }

    /// # Overview
    ///
    /// Pre-computed probability for strengthening: (s-1)/s.
    #[inline]
    #[must_use]
    pub fn prob_strengthen(&self) -> f32 {
        (self.s - 1.0) / self.s
    }

    /// # Overview
    ///
    /// Pre-computed probability for weakening: 1/s.
    #[inline]
    #[must_use]
    pub fn prob_weaken(&self) -> f32 {
        1.0 / self.s
    }

    /// # Overview
    ///
    /// Pre-computed integer threshold for strengthening.
    ///
    /// Converts float probability to u32 for faster comparison:
    /// `rng.random::<u32>() < threshold` is ~2x faster than float comparison.
    #[inline]
    #[must_use]
    pub fn threshold_strengthen(&self) -> u32 {
        prob_to_threshold(self.prob_strengthen())
    }

    /// # Overview
    ///
    /// Pre-computed integer threshold for weakening.
    #[inline]
    #[must_use]
    pub fn threshold_weaken(&self) -> u32 {
        prob_to_threshold(self.prob_weaken())
    }
}

/// Converts probability [0.0, 1.0] to integer threshold for fast comparison.
///
/// Usage: `rng.random::<u32>() < threshold` is equivalent to
/// `rng.random::<f32>() < probability` but ~2x faster.
#[inline]
#[must_use]
pub fn prob_to_threshold(prob: f32) -> u32 {
    (prob as f64 * u32::MAX as f64) as u32
}

/// # Overview
///
/// Builder for Config with validation.
#[derive(Debug, Default)]
pub struct ConfigBuilder {
    n_clauses:  Option<usize>,
    n_features: Option<usize>,
    n_states:   Option<i16>,
    s:          Option<f32>
}

impl ConfigBuilder {
    /// # Overview
    ///
    /// Sets the number of clauses (must be even).
    pub fn clauses(mut self, n: usize) -> Self {
        self.n_clauses = Some(n);
        self
    }

    /// # Overview
    ///
    /// Sets the number of input features.
    pub fn features(mut self, n: usize) -> Self {
        self.n_features = Some(n);
        self
    }

    /// # Overview
    ///
    /// Sets states per automaton action (default: 100).
    pub fn states(mut self, n: i16) -> Self {
        self.n_states = Some(n);
        self
    }

    /// # Overview
    ///
    /// Sets specificity parameter s (default: 3.9).
    pub fn specificity(mut self, s: f32) -> Self {
        self.s = Some(s);
        self
    }

    /// # Overview
    ///
    /// Builds and validates the Config.
    pub fn build(self) -> Result<Config> {
        let config = Config {
            n_clauses:  self.n_clauses.ok_or(Error::MissingClauses)?,
            n_features: self.n_features.ok_or(Error::MissingFeatures)?,
            n_states:   self.n_states.unwrap_or(100),
            s:          self.s.unwrap_or(3.9)
        };
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_with_defaults() {
        let config = Config::builder().clauses(20).features(4).build().unwrap();

        assert_eq!(config.n_clauses, 20);
        assert_eq!(config.n_features, 4);
        assert_eq!(config.n_states, 100);
        assert!((config.s - 3.9).abs() < 0.01);
    }

    #[test]
    fn builder_rejects_odd_clauses() {
        let result = Config::builder().clauses(21).features(4).build();

        assert_eq!(result, Err(Error::OddClauses));
    }

    #[test]
    fn prob_precomputed() {
        let config = Config::builder()
            .clauses(20)
            .features(4)
            .specificity(4.0)
            .build()
            .unwrap();

        assert!((config.prob_strengthen() - 0.75).abs() < 0.001);
        assert!((config.prob_weaken() - 0.25).abs() < 0.001);
    }

    #[test]
    fn integer_thresholds() {
        let config = Config::builder()
            .clauses(20)
            .features(4)
            .specificity(4.0)
            .build()
            .unwrap();

        // 0.75 * u32::MAX ≈ 3221225471
        assert!(config.threshold_strengthen() > 3_000_000_000);
        assert!(config.threshold_strengthen() < 3_500_000_000);

        // 0.25 * u32::MAX ≈ 1073741823
        assert!(config.threshold_weaken() > 1_000_000_000);
        assert!(config.threshold_weaken() < 1_200_000_000);
    }

    #[test]
    fn prob_to_threshold_boundaries() {
        assert_eq!(prob_to_threshold(0.0), 0);
        assert_eq!(prob_to_threshold(1.0), u32::MAX);
        assert_eq!(prob_to_threshold(0.5), u32::MAX / 2);
    }
}
