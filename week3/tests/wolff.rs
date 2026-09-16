//! The Wolff single-cluster update: Equation 16, the growth, and the flip.

use std::collections::HashSet;

use ising::lattice::Lattice;
use ising::wolff::{add_probability, bond_added, cluster_flip, grow_cluster};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Equation 16 written out again, so the test does not repeat the code.
fn p(temperature: f64) -> f64 {
    1.0 - (-2.0 / temperature).exp()
}

fn random_spins(l: usize, rng: &mut ChaCha8Rng) -> Vec<i8> {
    (0..l * l)
        .map(|_| if rng.random::<f64>() < 0.5 { -1 } else { 1 })
        .collect()
}

/// The four wrapped neighbour indices of `index`, duplicates included.
fn neighbours(l: usize, index: usize) -> [usize; 4] {
    let (row, col) = (index / l, index % l);
    [
        ((row + l - 1) % l) * l + col,
        ((row + 1) % l) * l + col,
        row * l + (col + l - 1) % l,
        row * l + (col + 1) % l,
    ]
}

fn one_down(l: usize, down: usize) -> Lattice {
    let mut spins = vec![1i8; l * l];
    spins[down] = -1;
    Lattice::from_spins(l, spins).unwrap()
}

fn checkerboard(l: usize) -> Lattice {
    let spins: Vec<i8> = (0..l * l)
        .map(|k| if (k / l + k % l) % 2 == 0 { 1 } else { -1 })
        .collect();
    Lattice::from_spins(l, spins).unwrap()
}

#[test]
fn add_probability_is_one_minus_exp_minus_two_over_t() {
    for temperature in [1.5, 2.0, 2.26919, 2.3, 2.6, 3.0, 3.5] {
        assert!(
            (add_probability(temperature) - p(temperature)).abs() < 1e-15,
            "T = {temperature}"
        );
    }
    // The sheet's own temperatures, to the four decimals the report prints.
    assert!((add_probability(2.3) - 0.5809).abs() < 5e-5);
    assert!((add_probability(2.26919) - 0.5858).abs() < 5e-5);
    assert!((add_probability(2.0) - 0.6321).abs() < 5e-5);
    assert!((add_probability(3.0) - 0.4866).abs() < 5e-5);

    // A cold lattice bonds almost surely, a hot one almost never.
    assert_eq!(add_probability(1e-9), 1.0);
    assert!(add_probability(1e12) < 1e-11);

    // Monotone in T: the colder the lattice the likelier a bond joins.
    let grid = [1.5, 2.0, 2.3, 2.6, 3.0, 3.5];
    for pair in grid.windows(2) {
        assert!(
            add_probability(pair[0]) > add_probability(pair[1]),
            "p({}) <= p({})",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn bond_added_compares_the_uniform_to_the_probability() {
    let t = 2.3;
    let probability = add_probability(t);
    assert!(bond_added(t, 0.0));
    assert!(bond_added(t, probability - 1e-12));
    assert!(!bond_added(t, probability));
    assert!(!bond_added(t, probability + 1e-12));
    assert!(!bond_added(t, 0.999_999));
    // A cold bond is never declined; a very hot one is declined almost surely.
    for uniform in [0.0, 0.25, 0.5, 0.999_999] {
        assert!(bond_added(1e-9, uniform), "uniform {uniform}");
    }
    assert!(!bond_added(1e12, 0.5));
}

#[test]
fn many_aligned_bonds_join_at_the_predicted_rate() {
    let mut rng = ChaCha8Rng::seed_from_u64(16);
    for temperature in [1.8, 2.3, 3.0] {
        let draws = 200_000u32;
        let joined = (0..draws)
            .filter(|_| bond_added(temperature, rng.random::<f64>()))
            .count();
        let rate = joined as f64 / draws as f64;
        assert!(
            (rate - p(temperature)).abs() < 0.01,
            "T = {temperature}: rate {rate}, expected {}",
            p(temperature)
        );
    }
}

#[test]
fn a_checkerboard_never_grows_past_its_seed() {
    // Every neighbour of a checkerboard site has the opposite spin, so no
    // bond is ever aligned: the cluster is the seed alone, whatever the draws.
    let l = 6;
    let lattice = checkerboard(l);
    for temperature in [1.0, 2.26919, 2.3, 3.0, 5.0] {
        for seed in 0..l * l {
            let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
            let cluster = grow_cluster(&lattice, seed, temperature, &mut rng);
            assert_eq!(cluster, vec![seed], "seed {seed} at T = {temperature}");
        }
    }
}

#[test]
fn a_cold_cluster_swallows_the_whole_aligned_component() {
    // At T -> 0 Equation 16 is exactly one, so the cluster is the connected
    // component of equal spins: the whole lattice when they all agree.
    let l = 4;
    let all_up = Lattice::all_up(l);
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let cluster = grow_cluster(&all_up, 5, 1e-9, &mut rng);
    let set: HashSet<usize> = cluster.iter().copied().collect();
    assert_eq!(set, (0..l * l).collect::<HashSet<usize>>());
    assert_eq!(cluster[0], 5, "the seed is the first site grown");

    // One down spin: an up seed cannot cross it, the down seed cannot leave.
    let lattice = one_down(l, 0);
    let up_seed = l + 1;
    let cluster = grow_cluster(&lattice, up_seed, 1e-9, &mut rng);
    assert_eq!(cluster.len(), l * l - 1);
    assert!(!cluster.contains(&0));
    assert!(cluster.iter().all(|site| lattice.spins()[*site] == 1));

    let cluster = grow_cluster(&lattice, 0, 1e-9, &mut rng);
    assert_eq!(cluster, vec![0]);
}

#[test]
fn the_grown_cluster_only_holds_the_seed_spin_and_is_connected() {
    let l = 8;
    let mut rng = ChaCha8Rng::seed_from_u64(8);
    let lattice = Lattice::from_spins(l, random_spins(l, &mut rng)).unwrap();
    for seed in 0..l * l {
        let temperature = 2.3;
        let cluster = grow_cluster(&lattice, seed, temperature, &mut rng);
        let set: HashSet<usize> = cluster.iter().copied().collect();
        assert_eq!(set.len(), cluster.len(), "seed {seed} repeats a site");
        assert_eq!(cluster[0], seed, "seed {seed} is not grown first");
        let spin = lattice.spins()[seed];
        assert!(
            cluster.iter().all(|site| lattice.spins()[*site] == spin),
            "seed {seed} grew an opposite-spin site"
        );
        for site in &cluster {
            assert!(
                *site == seed
                    || neighbours(l, *site)
                        .iter()
                        .any(|neighbour| set.contains(neighbour)),
                "site {site} has no neighbour inside the cluster"
            );
        }
    }
}

#[test]
fn a_flipped_cluster_is_exactly_the_cluster_that_was_grown() {
    let l = 8;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let before = Lattice::from_spins(l, random_spins(l, &mut rng)).unwrap();
    for _ in 0..25 {
        let temperature = rng.random_range(1.5..3.5);

        // The move's first draw picks the seed, so a clone that makes the same
        // first draw reconstructs the cluster the move is about to grow.
        let mut probe = rng.clone();
        let seed = probe.random_range(0..l * l);
        let grown: HashSet<usize> = grow_cluster(&before, seed, temperature, &mut probe)
            .into_iter()
            .collect();

        let mut after = before.clone();
        let size = cluster_flip(&mut after, temperature, &mut rng);
        let changed: HashSet<usize> = (0..l * l)
            .filter(|index| before.spins()[*index] != after.spins()[*index])
            .collect();

        assert_eq!(size, grown.len());
        assert_eq!(changed, grown, "T = {temperature}, seed = {seed}");
        assert!(!changed.is_empty(), "a cluster flip moved no spin");
    }
}

#[test]
fn the_whole_cluster_flips_at_every_temperature() {
    // There is no acceptance test: whatever the move grew is what moved.
    let l = 8;
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let before = Lattice::from_spins(l, random_spins(l, &mut rng)).unwrap();
    for temperature in [1.0, 2.26919, 2.3, 3.0, 10.0] {
        let mut probe = rng.clone();
        let seed = probe.random_range(0..l * l);
        let grown: HashSet<usize> = grow_cluster(&before, seed, temperature, &mut probe)
            .into_iter()
            .collect();

        let mut after = before.clone();
        cluster_flip(&mut after, temperature, &mut rng);
        for index in 0..l * l {
            let flipped = before.spins()[index] != after.spins()[index];
            assert_eq!(
                flipped,
                grown.contains(&index),
                "T = {temperature}, site {index}: flipped {flipped}"
            );
        }
    }
}

#[test]
fn every_cluster_size_is_between_one_and_l_squared() {
    let l = 6;
    let mut rng = ChaCha8Rng::seed_from_u64(6);
    let mut lattice = Lattice::all_up(l);
    for step in 0..500 {
        let size = cluster_flip(&mut lattice, 2.3, &mut rng);
        assert!((1..=l * l).contains(&size), "step {step}: size {size}");
    }
}

#[test]
fn the_same_seed_reproduces_the_cluster_size_sequence() {
    let l = 8;
    let sequence = |seed: u64| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut lattice = Lattice::all_up(l);
        let sizes: Vec<usize> = (0..200)
            .map(|_| cluster_flip(&mut lattice, 2.3, &mut rng))
            .collect();
        (sizes, lattice)
    };
    let (first, first_lattice) = sequence(42);
    let (second, second_lattice) = sequence(42);
    assert_eq!(first, second);
    assert_eq!(first_lattice, second_lattice);
    let (other, _) = sequence(1042);
    assert_ne!(first, other, "seed 1042 produced the same cluster sizes as 42");
}
