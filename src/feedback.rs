//! Type I and Type II feedback mechanisms.

use rand::Rng;

use crate::Clause;

/// # Overview
///
/// Type I feedback: reinforces patterns when y=1.
///
/// When clause fires: strengthen matching, weaken non-matching.
/// When clause doesn't fire: weaken all toward exclusion.
pub fn type_i<R: Rng>(clause: &mut Clause, x: &[u8], fires: bool, s: f32, rng: &mut R) {
    let prob_strengthen = (s - 1.0) / s;
    let prob_weaken = 1.0 / s;
    let automata = clause.automata_mut();
    let n = x.len();

    if !fires {
        for a in automata.iter_mut() {
            if rng.random::<f32>() <= prob_weaken {
                a.decrement();
            }
        }
        return;
    }

    for k in 0..n {
        if x[k] == 1 {
            if rng.random::<f32>() <= prob_strengthen {
                automata[2 * k].increment();
            }
            if rng.random::<f32>() <= prob_weaken {
                automata[2 * k + 1].decrement();
            }
        } else {
            if rng.random::<f32>() <= prob_strengthen {
                automata[2 * k + 1].increment();
            }
            if rng.random::<f32>() <= prob_weaken {
                automata[2 * k].decrement();
            }
        }
    }
}

/// # Overview
///
/// Type II feedback: corrects false positives when y=0.
///
/// Activates blocking literals to prevent future misfires.
pub fn type_ii(clause: &mut Clause, x: &[u8]) {
    let automata = clause.automata_mut();

    for (k, &xk) in x.iter().enumerate() {
        if xk == 0 {
            if !automata[2 * k].action() {
                automata[2 * k].increment();
            }
        } else if !automata[2 * k + 1].action() {
            automata[2 * k + 1].increment();
        }
    }
}

/// # Overview
///
/// Boosted Type I (Type Ia): always strengthens matching when firing.
pub fn type_ia<R: Rng>(clause: &mut Clause, x: &[u8], fires: bool, s: f32, rng: &mut R) {
    if !fires {
        type_i(clause, x, fires, s, rng);
        return;
    }

    let prob_weaken = 1.0 / s;
    let automata = clause.automata_mut();

    for (k, &xk) in x.iter().enumerate() {
        if xk == 1 {
            automata[2 * k].increment();
            if rng.random::<f32>() <= prob_weaken {
                automata[2 * k + 1].decrement();
            }
        } else {
            automata[2 * k + 1].increment();
            if rng.random::<f32>() <= prob_weaken {
                automata[2 * k].decrement();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::rng_from_seed;

    #[test]
    fn type_i_not_firing() {
        let mut clause = Clause::new(3, 100, 1);
        let mut rng = rng_from_seed(42);

        for _ in 0..100 {
            type_i(&mut clause, &[1, 0, 1], false, 3.0, &mut rng);
        }

        assert!(clause.automata().iter().any(|a| a.state() < 100));
    }

    #[test]
    fn type_i_firing() {
        let mut clause = Clause::new(3, 50, 1);
        let mut rng = rng_from_seed(42);

        for _ in 0..200 {
            type_i(&mut clause, &[1, 0, 1], true, 3.0, &mut rng);
        }

        assert!(clause.automata()[0].action());
        assert!(clause.automata()[3].action());
    }

    #[test]
    fn type_ii_blocks() {
        let mut clause = Clause::new(3, 50, 1);

        for _ in 0..100 {
            type_ii(&mut clause, &[1, 0, 1]);
        }

        assert!(clause.automata()[1].action());
        assert!(clause.automata()[2].action());
        assert!(clause.automata()[5].action());
    }
}
