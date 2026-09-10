use std::collections::BTreeSet;

use ising::{AcceptanceTable, Lattice, sweep};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn local_delta_matches_recomputed_total_energy_on_random_lattices() {
    let mut rng = StdRng::seed_from_u64(5315);
    let mut observed = BTreeSet::new();

    for _ in 0..32 {
        let lattice = Lattice::random(8, &mut rng);
        for site in 0..lattice.site_count() {
            let before = lattice.energy();
            let delta = lattice.delta_energy_at(site);
            let after = lattice.flipped(site).energy();
            assert_eq!(after - before, delta, "wrong delta at site {site}");
            observed.insert(delta);
        }
    }

    assert_eq!(observed, BTreeSet::from([-8, -4, 0, 4, 8]));
}

#[test]
fn periodic_boundaries_give_four_neighbours_and_correct_all_up_energy() {
    let lattice = Lattice::all_up(5);
    assert_eq!(lattice.energy(), -50);
    assert_eq!(lattice.delta_energy_at(0), 8);
    assert_eq!(lattice.delta_energy_at(24), 8);
}

#[test]
fn one_sweep_makes_exactly_one_proposal_per_site_on_average() {
    let mut lattice = Lattice::all_up(7);
    let table = AcceptanceTable::new(2.3);
    let mut rng = StdRng::seed_from_u64(2026);
    let stats = sweep(&mut lattice, &table, &mut rng);
    assert_eq!(stats.proposals, 49);
    assert!(stats.accepted <= stats.proposals);
}

