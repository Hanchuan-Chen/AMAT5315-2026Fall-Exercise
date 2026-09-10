use rand::Rng;

use crate::Lattice;

pub fn wolff_add_probability(temperature: f64) -> f64 {
    1.0 - (-2.0 / temperature).exp()
}

pub fn wolff_cluster_flip<R: Rng + ?Sized>(
    lattice: &mut Lattice,
    temperature: f64,
    rng: &mut R,
) -> usize {
    let seed = rng.random_range(0..lattice.site_count());
    let cluster_spin = lattice.spin(seed);
    let probability = wolff_add_probability(temperature);
    let mut in_cluster = vec![false; lattice.site_count()];
    let mut cluster = Vec::new();
    let mut stack = vec![seed];
    in_cluster[seed] = true;

    while let Some(site) = stack.pop() {
        cluster.push(site);
        for neighbour in lattice.neighbours(site) {
            if !in_cluster[neighbour]
                && lattice.spin(neighbour) == cluster_spin
                && (probability >= 1.0 || rng.random::<f64>() < probability)
            {
                in_cluster[neighbour] = true;
                stack.push(neighbour);
            }
        }
    }
    for site in cluster.iter().copied() {
        lattice.flip(site);
    }
    cluster.len()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WolffSweepStats {
    pub touched: usize,
    pub clusters: usize,
}

pub fn wolff_sweep<R: Rng + ?Sized>(
    lattice: &mut Lattice,
    temperature: f64,
    rng: &mut R,
) -> WolffSweepStats {
    let target = lattice.site_count();
    let mut stats = WolffSweepStats::default();
    while stats.touched < target {
        stats.touched += wolff_cluster_flip(lattice, temperature, rng);
        stats.clusters += 1;
    }
    stats
}
