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
///
/// A downhill or neutral proposal is accepted outright, so at a finite
/// temperature the rule has three distinct values, one per uphill `dE`.
pub fn acceptance_probability(delta_energy: i32, temperature: f64) -> f64 {
    if delta_energy <= 0 {
        1.0
    } else {
        (-(delta_energy as f64) / temperature).exp().min(1.0)
    }
}

/// The same rule, decided by a supplied uniform draw in `[0, 1)`.
pub fn proposal_accepted(delta_energy: i32, temperature: f64, uniform: f64) -> bool {
    uniform < acceptance_probability(delta_energy, temperature)
}

/// One sweep: `l * l` proposals, each site drawn with replacement.
///
/// A proposal with `dE <= 0` is taken without drawing from the stream, so one
/// sweep at `T -> 0` consumes exactly `l * l` small integers.
pub fn sweep(lattice: &mut Lattice, temperature: f64, rng: &mut impl Rng) -> SweepStats {
    let l = lattice.l();
    let sites = l * l;
    let mut accepted = 0u64;
    for _ in 0..sites {
        let index = rng.random_range(0..sites);
        let (row, col) = (index / l, index % l);
        let delta = lattice.delta_energy(row, col);
        let take = delta <= 0 || proposal_accepted(delta, temperature, rng.random::<f64>());
        if take {
            lattice.flip(row, col);
            accepted += 1;
        }
    }
    SweepStats {
        accepted,
        proposals: sites as u64,
    }
}
