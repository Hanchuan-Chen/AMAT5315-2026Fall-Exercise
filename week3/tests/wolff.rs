use ising::{Lattice, wolff_add_probability, wolff_cluster_flip, wolff_sweep};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn wolff_bond_probability_matches_the_ising_rule() {
    let expected = 1.0 - (-2.0_f64 / 2.3).exp();
    assert!((wolff_add_probability(2.3) - expected).abs() < 1e-15);
}

#[test]
fn zero_temperature_limit_grows_through_periodic_all_up_lattice() {
    let mut lattice = Lattice::all_up(4);
    let mut rng = StdRng::seed_from_u64(2026);
    let size = wolff_cluster_flip(&mut lattice, 1e-12, &mut rng);
    assert_eq!(size, 16);
    assert_eq!(lattice.magnetization(), -1.0);
}

#[test]
fn one_wolff_sweep_touches_at_least_one_lattice_worth_of_spins() {
    let mut lattice = Lattice::all_up(8);
    let mut rng = StdRng::seed_from_u64(2026);
    let stats = wolff_sweep(&mut lattice, 2.3, &mut rng);
    assert!(stats.touched >= 64);
    assert!(stats.clusters >= 1);
}

#[test]
fn seeded_wolff_sweeps_are_reproducible() {
    let mut first = Lattice::all_up(16);
    let mut second = first.clone();
    let mut rng_a = StdRng::seed_from_u64(77);
    let mut rng_b = StdRng::seed_from_u64(77);
    for _ in 0..20 {
        wolff_sweep(&mut first, 2.3, &mut rng_a);
        wolff_sweep(&mut second, 2.3, &mut rng_b);
    }
    assert_eq!(first, second);
}
