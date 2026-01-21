//! Convolutional Tsetlin Machine for image-like data.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    Clause,
    feedback::{type_i, type_ii},
    utils::rng_from_seed
};

/// # Overview
///
/// Configuration for Convolutional Tsetlin Machine.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConvConfig {
    pub n_clauses:   usize,
    pub n_states:    i16,
    pub s:           f32,
    pub patch_size:  usize,
    pub image_width: usize
}

/// # Overview
///
/// Convolutional Tsetlin Machine for 2D binary images.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Convolutional {
    clauses:   Vec<Vec<Clause>>,
    config:    ConvConfig,
    threshold: i32
}

impl Convolutional {
    pub fn new(config: ConvConfig, n_classes: usize, threshold: i32) -> Self {
        let pf = config.patch_size * config.patch_size;
        let clauses = (0..n_classes)
            .map(|_| {
                (0..config.n_clauses)
                    .map(|i| Clause::new(pf, config.n_states, if i % 2 == 0 { 1 } else { -1 }))
                    .collect()
            })
            .collect();
        Self {
            clauses,
            config,
            threshold
        }
    }

    pub fn class_votes(&self, image: &[u8]) -> Vec<i32> {
        let (rows, cols, ps, w) = self.patch_dims(image);
        self.clauses
            .iter()
            .map(|cls| {
                let mut sum = 0i32;
                for r in 0..rows {
                    for c in 0..cols {
                        let patch = extract_patch(image, r, c, ps, w);
                        sum += cls.iter().map(|cl| cl.vote(&patch)).sum::<i32>();
                    }
                }
                sum
            })
            .collect()
    }

    pub fn predict(&self, image: &[u8]) -> usize {
        self.class_votes(image)
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn fit(&mut self, images: &[Vec<u8>], labels: &[usize], epochs: usize, seed: u64) {
        let mut rng = rng_from_seed(seed);
        let mut idx: Vec<usize> = (0..images.len()).collect();
        for _ in 0..epochs {
            crate::utils::shuffle(&mut idx, &mut rng);
            for &i in &idx {
                self.train_one(&images[i], labels[i], &mut rng);
            }
        }
    }

    fn train_one<R: Rng>(&mut self, image: &[u8], label: usize, rng: &mut R) {
        let votes = self.class_votes(image);
        let (rows, cols, ps, w) = self.patch_dims(image);
        let t = self.threshold as f32;

        for (ci, cls) in self.clauses.iter_mut().enumerate() {
            let is_target = ci == label;
            let sum = votes[ci].clamp(-self.threshold, self.threshold) as f32;

            for r in 0..rows {
                for c in 0..cols {
                    let patch = extract_patch(image, r, c, ps, w);
                    for clause in cls.iter_mut() {
                        let fires = clause.evaluate(&patch);
                        let p = clause.polarity();
                        if is_target {
                            let prob = (t - sum) / (2.0 * t);
                            if p == 1 && rng.random::<f32>() <= prob {
                                type_i(clause, &patch, fires, self.config.s, rng);
                            } else if p == -1 && fires && rng.random::<f32>() <= prob {
                                type_ii(clause, &patch);
                            }
                        } else {
                            let prob = (t + sum) / (2.0 * t);
                            if p == -1 && rng.random::<f32>() <= prob {
                                type_i(clause, &patch, fires, self.config.s, rng);
                            } else if p == 1 && fires && rng.random::<f32>() <= prob {
                                type_ii(clause, &patch);
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn evaluate(&self, images: &[Vec<u8>], labels: &[usize]) -> f32 {
        let correct = images
            .iter()
            .zip(labels)
            .filter(|(i, l)| self.predict(i) == **l)
            .count();
        correct as f32 / images.len() as f32
    }

    fn patch_dims(&self, image: &[u8]) -> (usize, usize, usize, usize) {
        let h = image.len() / self.config.image_width;
        let rows = h.saturating_sub(self.config.patch_size - 1);
        let cols = self
            .config
            .image_width
            .saturating_sub(self.config.patch_size - 1);
        (rows, cols, self.config.patch_size, self.config.image_width)
    }
}

fn extract_patch(image: &[u8], row: usize, col: usize, ps: usize, w: usize) -> Vec<u8> {
    let mut p = Vec::with_capacity(ps * ps);
    for dr in 0..ps {
        for dc in 0..ps {
            p.push(image.get((row + dr) * w + col + dc).copied().unwrap_or(0));
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_valid() {
        let config = ConvConfig {
            n_clauses:   10,
            n_states:    100,
            s:           3.0,
            patch_size:  2,
            image_width: 4
        };
        let tm = Convolutional::new(config, 3, 15);
        assert!(tm.predict(&[0u8; 16]) < 3);
    }
}
