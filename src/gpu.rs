//! GPU acceleration support for Tsetlin Machine operations.
//!
//! This module provides the foundation for GPU-accelerated training and
//! inference. The Tsetlin Machine is embarrassingly parallel - each clause
//! can be evaluated independently, making it well-suited for GPU computation.
//!
//! # Current Status
//!
//! This module provides:
//! - [`GpuBackend`] trait for abstracting different GPU libraries
//! - [`GpuConfig`] for GPU-specific configuration
//! - Data transfer utilities for clause bank ↔ GPU memory
//!
//! # Supported Backends (Planned)
//!
//! | Backend | Crate | Platforms | Status |
//! |---------|-------|-----------|--------|
//! | WebGPU | `wgpu` | Cross-platform | Planned |
//! | Vulkan | `vulkano` | Linux, Windows | Planned |
//! | CUDA | `cudarc` | NVIDIA GPUs | Planned |
//!
//! # Performance Considerations
//!
//! GPU acceleration is beneficial when:
//! - Dataset has > 100,000 samples
//! - Number of clauses > 1000
//! - Batch size > 1000
//!
//! For smaller workloads, CPU (especially with rayon parallelization) may
//! be faster due to data transfer overhead.
//!
//! # Example (Future API)
//!
//! ```ignore
//! use tsetlin_rs::gpu::{GpuClauseBank, WgpuBackend};
//!
//! // Create GPU-accelerated clause bank
//! let backend = WgpuBackend::new()?;
//! let gpu_bank = GpuClauseBank::from_cpu(&bank, &backend)?;
//!
//! // Train on GPU
//! gpu_bank.train_batch(&inputs, &labels, threshold, s)?;
//!
//! // Transfer back to CPU for inference
//! let cpu_bank = gpu_bank.to_cpu()?;
//! ```
//!
//! # References
//!
//! - [PyTsetlinMachineCUDA](https://github.com/cair/PyTsetlinMachineCUDA)
//! - [Massively Parallel TM (ICML 2021)](https://arxiv.org/abs/2009.04861)

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// GPU backend abstraction trait.
///
/// Implement this trait to add support for a new GPU library.
/// The trait provides the minimal interface needed for clause bank operations.
pub trait GpuBackend {
    /// Error type for GPU operations.
    type Error;

    /// GPU buffer handle type.
    type Buffer;

    /// Returns the name of this backend (e.g., "wgpu", "vulkano", "cuda").
    fn name(&self) -> &'static str;

    /// Allocates a buffer on the GPU.
    fn allocate(&self, size: usize) -> Result<Self::Buffer, Self::Error>;

    /// Copies data from CPU to GPU buffer.
    fn upload<T: Copy>(&self, buffer: &Self::Buffer, data: &[T]) -> Result<(), Self::Error>;

    /// Copies data from GPU buffer to CPU.
    fn download<T: Copy>(&self, buffer: &Self::Buffer, data: &mut [T]) -> Result<(), Self::Error>;

    /// Returns maximum buffer size in bytes.
    fn max_buffer_size(&self) -> usize;

    /// Returns available GPU memory in bytes.
    fn available_memory(&self) -> usize;
}

/// Configuration for GPU operations.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Minimum batch size to use GPU (smaller batches use CPU).
    pub min_batch_size:         usize,
    /// Maximum samples per GPU kernel launch.
    pub max_samples_per_launch: usize,
    /// Whether to keep clause states on GPU between epochs.
    pub persistent_state:       bool,
    /// Enable GPU timing for profiling.
    pub enable_timing:          bool
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            min_batch_size:         1000,
            max_samples_per_launch: 100_000,
            persistent_state:       true,
            enable_timing:          false
        }
    }
}

impl GpuConfig {
    /// Creates a new GPU configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the minimum batch size for GPU operations.
    #[must_use]
    pub fn with_min_batch_size(mut self, size: usize) -> Self {
        self.min_batch_size = size;
        self
    }

    /// Enables persistent GPU state between epochs.
    #[must_use]
    pub fn with_persistent_state(mut self, enabled: bool) -> Self {
        self.persistent_state = enabled;
        self
    }

    /// Enables GPU timing for profiling.
    #[must_use]
    pub fn with_timing(mut self, enabled: bool) -> Self {
        self.enable_timing = enabled;
        self
    }
}

/// GPU memory layout for clause bank data.
///
/// Describes how clause bank data is arranged in GPU memory for
/// efficient parallel access.
#[derive(Debug, Clone)]
pub struct GpuMemoryLayout {
    /// Number of clauses.
    pub n_clauses:       usize,
    /// Number of features.
    pub n_features:      usize,
    /// Stride between clause state arrays.
    pub clause_stride:   usize,
    /// Total size of states buffer in bytes.
    pub states_size:     usize,
    /// Total size of weights buffer in bytes.
    pub weights_size:    usize,
    /// Total size of polarities buffer in bytes.
    pub polarities_size: usize
}

impl GpuMemoryLayout {
    /// Computes the memory layout for a clause bank.
    #[must_use]
    pub fn from_dimensions(n_clauses: usize, n_features: usize) -> Self {
        let clause_stride = 2 * n_features;
        Self {
            n_clauses,
            n_features,
            clause_stride,
            states_size: n_clauses * clause_stride * core::mem::size_of::<i16>(),
            weights_size: n_clauses * core::mem::size_of::<f32>(),
            polarities_size: n_clauses * core::mem::size_of::<i8>()
        }
    }

    /// Returns total GPU memory required in bytes.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.states_size + self.weights_size + self.polarities_size
    }
}

/// Estimates whether GPU acceleration would be beneficial.
///
/// Returns `true` if the workload is large enough to benefit from GPU,
/// accounting for data transfer overhead.
#[must_use]
pub fn should_use_gpu(n_samples: usize, n_clauses: usize, n_features: usize) -> bool {
    // Rough heuristic based on typical transfer overhead
    let compute_work = n_samples * n_clauses * n_features;
    let transfer_overhead = n_clauses * n_features * 2; // Upload states + download

    // GPU beneficial when compute work >> transfer overhead
    // Typical crossover point around 10M operations
    compute_work > 10_000_000 && compute_work > transfer_overhead * 100
}

/// GPU timing information for profiling.
#[derive(Debug, Clone, Default)]
pub struct GpuTiming {
    /// Time spent uploading data to GPU (microseconds).
    pub upload_us:   u64,
    /// Time spent in GPU computation (microseconds).
    pub compute_us:  u64,
    /// Time spent downloading results (microseconds).
    pub download_us: u64
}

impl GpuTiming {
    /// Returns total time in microseconds.
    #[must_use]
    pub fn total_us(&self) -> u64 {
        self.upload_us + self.compute_us + self.download_us
    }

    /// Returns the fraction of time spent on data transfer.
    #[must_use]
    pub fn transfer_fraction(&self) -> f32 {
        let total = self.total_us();
        if total == 0 {
            0.0
        } else {
            (self.upload_us + self.download_us) as f32 / total as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.min_batch_size, 1000);
        assert!(config.persistent_state);
        assert!(!config.enable_timing);
    }

    #[test]
    fn gpu_config_builder() {
        let config = GpuConfig::new()
            .with_min_batch_size(500)
            .with_persistent_state(false)
            .with_timing(true);

        assert_eq!(config.min_batch_size, 500);
        assert!(!config.persistent_state);
        assert!(config.enable_timing);
    }

    #[test]
    fn memory_layout() {
        let layout = GpuMemoryLayout::from_dimensions(100, 64);

        assert_eq!(layout.n_clauses, 100);
        assert_eq!(layout.n_features, 64);
        assert_eq!(layout.clause_stride, 128);
        assert_eq!(layout.states_size, 100 * 128 * 2); // i16 = 2 bytes
        assert_eq!(layout.weights_size, 100 * 4); // f32 = 4 bytes
        assert_eq!(layout.polarities_size, 100 * 1); // i8 = 1 byte
    }

    #[test]
    fn memory_layout_total() {
        let layout = GpuMemoryLayout::from_dimensions(100, 64);
        let total = layout.total_size();

        assert_eq!(total, 100 * 128 * 2 + 100 * 4 + 100);
    }

    #[test]
    fn should_use_gpu_small_workload() {
        // Small workload - CPU better
        assert!(!should_use_gpu(100, 20, 64));
        assert!(!should_use_gpu(1000, 100, 64));
    }

    #[test]
    fn should_use_gpu_large_workload() {
        // Large workload - GPU beneficial
        assert!(should_use_gpu(100_000, 1000, 64));
        assert!(should_use_gpu(1_000_000, 100, 64));
    }

    #[test]
    fn gpu_timing_total() {
        let timing = GpuTiming {
            upload_us:   100,
            compute_us:  500,
            download_us: 50
        };

        assert_eq!(timing.total_us(), 650);
    }

    #[test]
    fn gpu_timing_transfer_fraction() {
        let timing = GpuTiming {
            upload_us:   100,
            compute_us:  400,
            download_us: 100
        };

        let fraction = timing.transfer_fraction();
        assert!((fraction - 0.333).abs() < 0.01);
    }

    #[test]
    fn gpu_timing_empty() {
        let timing = GpuTiming::default();
        assert_eq!(timing.total_us(), 0);
        assert!((timing.transfer_fraction() - 0.0).abs() < 0.001);
    }
}
