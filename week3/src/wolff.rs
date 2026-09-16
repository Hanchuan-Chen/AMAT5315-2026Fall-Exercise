//! The Wolff single-cluster update.

use rand::Rng;

use crate::lattice::Lattice;

/// The bond probability `p = 1 - exp(-2 / T)` of the sheet's Equation 16.
///
/// The cluster grows along the bonds of the Fortuin-Kasteleyn representation
/// of the model, and this is the probability that makes flipping the whole
/// cluster with acceptance one leave the Boltzmann distribution invariant.
/// Colder lattices bond more readily: `p -> 1` as `T -> 0` and `p -> 0` as
/// `T -> inf`.
pub fn add_probability(temperature: f64) -> f64 {
    1.0 - (-2.0 / temperature).exp()
}

/// The bond decision for one aligned neighbour, from a uniform in `[0, 1)`.
///
/// A uniform of exactly `p` declines the bond, the same convention the
/// Metropolis rule uses when it compares a uniform with `min(1, exp(-dE/T))`.
pub fn bond_added(temperature: f64, uniform: f64) -> bool {
    uniform < add_probability(temperature)
}

/// The cluster grown from `seed`, in the order its sites joined.
///
/// The seed is first. A neighbour joins only when its spin equals the spin of
/// the site it was reached from -- which, because only the seed's spin is ever
/// admitted, is the same test as the seed's spin -- and the drawn uniform is
/// below [`add_probability`]. An opposite-spin neighbour and an
/// already-clustered neighbour are skipped without consuming a draw.
///
/// At `T -> 0` the probability is exactly one and the whole aligned component
/// joins; at `T -> inf` it is zero and the cluster is the seed alone. For
/// `l = 2` the same neighbour index appears twice in the four-neighbour list,
/// because the torus of that size glues each pair of sites with two bonds; the
/// two appearances are drawn independently, as those `2 l^2` bonds need.
///
/// # Panics
///
/// Panics when `seed >= l * l`.
pub fn grow_cluster<R: Rng + ?Sized>(
    lattice: &Lattice,
    seed: usize,
    temperature: f64,
    rng: &mut R,
) -> Vec<usize> {
    let l = lattice.l();
    let sites = l * l;
    let seed_spin = lattice.spins()[seed];
    let probability = add_probability(temperature);
    let mut in_cluster = vec![false; sites];
    let mut cluster = Vec::new();
    let mut stack = vec![seed];
    in_cluster[seed] = true;

    while let Some(site) = stack.pop() {
        cluster.push(site);
        let (row, col) = (site / l, site % l);
        let neighbours = [
            ((row + l - 1) % l) * l + col,
            ((row + 1) % l) * l + col,
            row * l + (col + l - 1) % l,
            row * l + (col + 1) % l,
        ];
        for neighbour in neighbours {
            if in_cluster[neighbour] || lattice.spins()[neighbour] != seed_spin {
                continue;
            }
            if probability >= 1.0 || rng.random::<f64>() < probability {
                in_cluster[neighbour] = true;
                stack.push(neighbour);
            }
        }
    }
    cluster
}

/// One Wolff move: pick a site uniformly, grow its cluster, flip all of it.
///
/// There is no acceptance test; Equation 16 already makes the flip exact. The
/// move's first draw is `random_range(0..l * l)`, the seed site, so a caller
/// that makes the same first draw from a copy of the stream grows the same
/// cluster. Returns the number of spins flipped, which is the move's
/// `cluster_size`.
pub fn cluster_flip<R: Rng + ?Sized>(
    lattice: &mut Lattice,
    temperature: f64,
    rng: &mut R,
) -> usize {
    let seed = rng.random_range(0..lattice.site_count());
    let cluster = grow_cluster(lattice, seed, temperature, rng);
    for &site in &cluster {
        lattice.flip_index(site);
    }
    cluster.len()
}
