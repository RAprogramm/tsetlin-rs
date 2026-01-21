//! Error types for Tsetlin Machine.

use core::fmt;

/// # Overview
///
/// Errors that can occur when building or using a Tsetlin Machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    MissingClauses,
    MissingFeatures,
    OddClauses,
    InvalidSpecificity,
    InvalidThreshold,
    EmptyDataset,
    DimensionMismatch { expected: usize, got: usize }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingClauses => write!(f, "n_clauses is required"),
            Self::MissingFeatures => write!(f, "n_features is required"),
            Self::OddClauses => write!(f, "n_clauses must be even"),
            Self::InvalidSpecificity => write!(f, "s must be > 1.0"),
            Self::InvalidThreshold => write!(f, "threshold must be > 0"),
            Self::EmptyDataset => write!(f, "dataset cannot be empty"),
            Self::DimensionMismatch {
                expected,
                got
            } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// # Overview
///
/// Result type for Tsetlin Machine operations.
pub type Result<T> = core::result::Result<T, Error>;
