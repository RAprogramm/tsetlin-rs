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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_all_variants() {
        assert_eq!(Error::MissingClauses.to_string(), "n_clauses is required");
        assert_eq!(Error::MissingFeatures.to_string(), "n_features is required");
        assert_eq!(Error::OddClauses.to_string(), "n_clauses must be even");
        assert_eq!(Error::InvalidSpecificity.to_string(), "s must be > 1.0");
        assert_eq!(Error::InvalidThreshold.to_string(), "threshold must be > 0");
        assert_eq!(Error::EmptyDataset.to_string(), "dataset cannot be empty");
        assert_eq!(
            Error::DimensionMismatch {
                expected: 10,
                got:      5
            }
            .to_string(),
            "dimension mismatch: expected 10, got 5"
        );
    }

    #[test]
    fn error_debug() {
        let err = Error::DimensionMismatch {
            expected: 100,
            got:      50
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("DimensionMismatch"));
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("50"));
    }

    #[test]
    fn error_eq() {
        assert_eq!(Error::MissingClauses, Error::MissingClauses);
        assert_ne!(Error::MissingClauses, Error::MissingFeatures);
        assert_eq!(
            Error::DimensionMismatch {
                expected: 5,
                got:      3
            },
            Error::DimensionMismatch {
                expected: 5,
                got:      3
            }
        );
        assert_ne!(
            Error::DimensionMismatch {
                expected: 5,
                got:      3
            },
            Error::DimensionMismatch {
                expected: 5,
                got:      4
            }
        );
    }

    #[test]
    fn error_clone() {
        let err = Error::DimensionMismatch {
            expected: 10,
            got:      5
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }
}
