//! The single-flip Metropolis update.

use rand::Rng;

use crate::lattice::Lattice;

/// Accepted proposals and proposals made in one sweep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SweepStats {
    pub accepted: u64,
    pub proposals: u64,
}

/// `A = min(1, exp(-dE / T))`.
pub fn acceptance_probability(delta_energy: i32, temperature: f64) -> f64 {
    todo!()
}

/// The same rule, decided by a supplied uniform draw in `[0, 1)`.
pub fn proposal_accepted(delta_energy: i32, temperature: f64, uniform: f64) -> bool {
    todo!()
}

/// One sweep: `l * l` proposals, each site drawn with replacement.
pub fn sweep(lattice: &mut Lattice, temperature: f64, rng: &mut impl Rng) -> SweepStats {
    todo!()
}
