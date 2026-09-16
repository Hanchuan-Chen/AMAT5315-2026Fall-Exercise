//! Physics: the periodic energy, the five flip energies, and the accept rule.

use std::collections::BTreeSet;

use ising::lattice::Lattice;
use ising::metropolis::{acceptance_probability, proposal_accepted, sweep};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Independent recomputation: sum over the `2 * l^2` bonds of the torus.
fn brute_force_energy(l: usize, spins: &[i8]) -> i64 {
    let mut total = 0i64;
    for row in 0..l {
        for col in 0..l {
            let here = spins[row * l + col] as i64;
            let right = spins[row * l + (col + 1) % l] as i64;
            let down = spins[((row + 1) % l) * l + col] as i64;
            total -= here * right;
            total -= here * down;
        }
    }
    total
}

fn random_spins(l: usize, rng: &mut ChaCha8Rng) -> Vec<i8> {
    (0..l * l)
        .map(|_| if rng.random::<f64>() < 0.5 { -1 } else { 1 })
        .collect()
}

#[test]
fn energy_matches_a_brute_force_bond_sum() {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for l in [2usize, 3, 5, 8] {
        let up = Lattice::all_up(l);
        assert_eq!(up.energy(), -2 * (l * l) as i64);
        assert_eq!(up.energy_per_site(), -2.0);
        assert_eq!(up.energy(), brute_force_energy(l, up.spins()));

        let down = Lattice::from_spins(l, vec![-1; l * l]).unwrap();
        assert_eq!(down.energy(), brute_force_energy(l, down.spins()));
        assert_eq!(down.energy(), -2 * (l * l) as i64);

        // An even checkerboard disagrees on every bond, so every bond costs +1.
        // An odd side has a frustrated wrap seam: those bonds agree instead.
        let checker: Vec<i8> = (0..l * l)
            .map(|k| if (k / l + k % l) % 2 == 0 { 1 } else { -1 })
            .collect();
        let checker = Lattice::from_spins(l, checker).unwrap();
        if l % 2 == 0 {
            assert_eq!(checker.energy(), 2 * (l * l) as i64);
            assert_eq!(checker.energy_per_site(), 2.0);
        }
        assert_eq!(checker.energy(), brute_force_energy(l, checker.spins()));

        for _ in 0..8 {
            let spins = random_spins(l, &mut rng);
            let lattice = Lattice::from_spins(l, spins.clone()).unwrap();
            assert_eq!(lattice.energy(), brute_force_energy(l, &spins));
            assert_eq!(
                lattice.energy_per_site(),
                lattice.energy() as f64 / (l * l) as f64
            );
        }
    }
}

#[test]
fn the_corner_bonds_wrap_around_the_torus() {
    let l = 8;
    let mut spins = vec![1i8; l * l];
    spins[0] = -1;
    let lattice = Lattice::from_spins(l, spins).unwrap();
    // 128 bonds; the four touching the flipped corner now disagree.
    assert_eq!(lattice.energy(), -128 + 2 * 4);
    assert_eq!(lattice.energy_per_site(), -120.0 / 64.0);
    assert_eq!(lattice.neighbor_sum(0, 0), 4);
    assert_eq!(lattice.delta_energy(0, 0), -8);
    assert_eq!(lattice.magnetization(), 62.0 / 64.0);
}

#[test]
fn a_single_flip_takes_one_of_five_delta_energies() {
    let l = 3;
    let neighbours = [(0usize, 1usize), (2, 1), (1, 0), (1, 2)];
    for (down, expected) in [(0usize, 8i32), (1, 4), (2, 0), (3, -4), (4, -8)] {
        let mut spins = vec![1i8; l * l];
        for (index, (row, col)) in neighbours.iter().enumerate() {
            if index < down {
                spins[row * l + col] = -1;
            }
        }
        let lattice = Lattice::from_spins(l, spins).unwrap();
        assert_eq!(lattice.neighbor_sum(1, 1), 4 - 2 * down as i32);
        assert_eq!(lattice.delta_energy(1, 1), expected);

        // Flipping a down spin with the same neighbours flips the sign.
        let mut flipped = lattice.clone();
        flipped.flip(1, 1);
        assert_eq!(flipped.delta_energy(1, 1), -expected);
    }
}

#[test]
fn delta_energy_predicts_the_measured_energy_change() {
    let l = 8;
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let lattice = Lattice::from_spins(l, random_spins(l, &mut rng)).unwrap();
    let mut board = lattice.clone();
    let mut seen = BTreeSet::new();
    for row in 0..l {
        for col in 0..l {
            let delta = board.delta_energy(row, col);
            assert!(matches!(delta, -8 | -4 | 0 | 4 | 8), "delta {delta}");
            seen.insert(delta);

            let before = board.energy();
            board.flip(row, col);
            assert_eq!(board.energy() - before, delta as i64);
            board.flip(row, col);
            assert_eq!(board, lattice);
        }
    }
    assert!(seen.len() >= 3, "only saw {seen:?}");
}

#[test]
fn acceptance_probability_is_min_one_exponential() {
    let t = 2.3;
    assert_eq!(acceptance_probability(-8, t), 1.0);
    assert_eq!(acceptance_probability(-4, t), 1.0);
    assert_eq!(acceptance_probability(0, t), 1.0);
    assert!((acceptance_probability(4, t) - (-4.0f64 / t).exp()).abs() < 1e-15);
    assert!((acceptance_probability(8, t) - (-8.0f64 / t).exp()).abs() < 1e-15);
    // The sheet's table at T = 2.3.
    assert!((acceptance_probability(4, t) - 0.1757).abs() < 5e-4);
    assert!((acceptance_probability(8, t) - 0.0309).abs() < 5e-4);
    // Hotter lattices accept uphill moves more often, and the rule never exceeds one.
    assert!(acceptance_probability(4, 1.8) < acceptance_probability(4, 3.0));
    assert!(acceptance_probability(4, 1e-9) < 1e-9);
    assert!(acceptance_probability(-8, 1e-9) == 1.0);
}

#[test]
fn proposal_acceptance_compares_the_uniform_to_the_probability() {
    let t = 2.3;
    let p = acceptance_probability(4, t);
    assert!(proposal_accepted(4, t, 0.0));
    assert!(proposal_accepted(4, t, p - 1e-12));
    assert!(!proposal_accepted(4, t, p));
    assert!(!proposal_accepted(4, t, p + 1e-12));
    assert!(!proposal_accepted(4, t, 0.999_999));
    for delta in [-8, -4, 0] {
        for uniform in [0.0, 0.25, 0.5, 0.999_999] {
            assert!(proposal_accepted(delta, t, uniform));
        }
    }
}

#[test]
fn many_uphill_proposals_accept_at_the_predicted_rate() {
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let t = 2.3;
    let draws = 200_000u32;
    let accepted = (0..draws)
        .filter(|_| proposal_accepted(4, t, rng.random::<f64>()))
        .count();
    let rate = accepted as f64 / draws as f64;
    assert!(
        (rate - (-4.0f64 / t).exp()).abs() < 0.01,
        "acceptance rate {rate}"
    );
}

#[test]
fn a_sweep_proposes_l_squared_sites_with_replacement() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut lattice = Lattice::all_up(4);
    let stats = sweep(&mut lattice, 2.3, &mut rng);
    assert_eq!(stats.proposals, 16);
    assert!(stats.accepted <= 16);
}

#[test]
fn a_cold_sweep_never_accepts_an_uphill_flip_from_all_up() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let mut lattice = Lattice::all_up(6);
    let stats = sweep(&mut lattice, 1e-9, &mut rng);
    assert_eq!(stats.accepted, 0);
    assert_eq!(stats.proposals, 36);
    assert_eq!(lattice, Lattice::all_up(6));
}

#[test]
fn a_cold_ramp_takes_the_one_downhill_flip_and_stops() {
    let l = 8;
    let mut spins = vec![1i8; l * l];
    spins[0] = -1;
    let mut lattice = Lattice::from_spins(l, spins).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let mut accepted = 0u64;
    for _ in 0..50 {
        accepted += sweep(&mut lattice, 0.05, &mut rng).accepted;
    }
    assert_eq!(accepted, 1);
    assert_eq!(lattice, Lattice::all_up(l));
    assert_eq!(lattice.energy(), -2 * (l * l) as i64);
}

#[test]
fn magnetization_is_the_signed_mean_spin() {
    let l = 4;
    let mut spins = vec![1i8; l * l];
    spins[0] = -1;
    spins[5] = -1;
    let lattice = Lattice::from_spins(l, spins).unwrap();
    // Sixteen spins, two of them down: the mean is 14/16 minus 2/16.
    assert_eq!(lattice.magnetization(), 12.0 / 16.0);
    assert_eq!(lattice.abs_magnetization(), 12.0 / 16.0);

    let mut flipped = lattice.clone();
    for row in 0..l {
        for col in 0..l {
            flipped.flip(row, col);
        }
    }
    assert_eq!(flipped.magnetization(), -12.0 / 16.0);
    assert_eq!(flipped.abs_magnetization(), 12.0 / 16.0);
}

#[test]
fn the_lattice_constructor_rejects_bad_spins() {
    assert!(Lattice::from_spins(1, vec![1]).is_err());
    assert!(Lattice::from_spins(2, vec![1, 1, 1]).is_err());
    assert!(Lattice::from_spins(2, vec![1, 1, 0, 1]).is_err());
    assert!(Lattice::from_spins(2, vec![1, 1, 1, 1]).is_ok());
}
