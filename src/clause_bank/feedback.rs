//! Feedback operations for ClauseBank training.

use rand::Rng;

use super::ClauseBank;

impl ClauseBank {
    /// Type I feedback: reinforces patterns when y=1.
    ///
    /// When clause fires: strengthen matching, weaken non-matching.
    /// When doesn't fire: weaken all toward exclusion.
    pub fn type_i<R: Rng>(&mut self, clause: usize, x: &[u8], fires: bool, s: f32, rng: &mut R) {
        let prob_str = (s - 1.0) / s;
        let prob_wk = 1.0 / s;
        let n = x.len().min(self.n_features);
        let base = clause * self.stride;
        let max = 2 * self.n_states;

        if !fires {
            for i in 0..self.stride {
                if rng.random::<f32>() <= prob_wk && self.states[base + i] > 1 {
                    self.states[base + i] -= 1;
                }
            }
            return;
        }

        for (k, &xk) in x.iter().enumerate().take(n) {
            let pos = base + 2 * k;
            let neg = base + 2 * k + 1;

            if xk == 1 {
                if rng.random::<f32>() <= prob_str && self.states[pos] < max {
                    self.states[pos] += 1;
                }
                if rng.random::<f32>() <= prob_wk && self.states[neg] > 1 {
                    self.states[neg] -= 1;
                }
            } else {
                if rng.random::<f32>() <= prob_str && self.states[neg] < max {
                    self.states[neg] += 1;
                }
                if rng.random::<f32>() <= prob_wk && self.states[pos] > 1 {
                    self.states[pos] -= 1;
                }
            }
        }
    }

    /// Type II feedback: corrects false positives when y=0.
    ///
    /// Activates blocking literals to prevent future misfires.
    pub fn type_ii(&mut self, clause: usize, x: &[u8]) {
        let n = x.len().min(self.n_features);
        let base = clause * self.stride;
        let max = 2 * self.n_states;
        let threshold = self.n_states;

        for (k, &xk) in x.iter().enumerate().take(n) {
            let pos = base + 2 * k;
            let neg = base + 2 * k + 1;

            if xk == 0 {
                if self.states[pos] <= threshold && self.states[pos] < max {
                    self.states[pos] += 1;
                }
            } else if self.states[neg] <= threshold && self.states[neg] < max {
                self.states[neg] += 1;
            }
        }
    }
}
