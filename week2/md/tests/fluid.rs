use md::fluid::{RunConfig, initial_velocities, simulate, triangular_lattice};

#[test]
fn triangular_lattice_has_requested_density() {
    let lattice = triangular_lattice(100, 0.8).unwrap();
    assert_eq!(lattice.positions.len(), 100);
    let measured = 100.0 / (lattice.box_size[0] * lattice.box_size[1]);

    assert!((measured - 0.8).abs() < 1.0e-12);
    assert!(lattice.positions.iter().all(|position| {
        0.0 <= position[0]
            && position[0] < lattice.box_size[0]
            && 0.0 <= position[1]
            && position[1] < lattice.box_size[1]
    }));
}

#[test]
fn invalid_lattice_atom_counts_are_rejected() {
    assert!(triangular_lattice(15, 0.8).is_err());
    assert!(triangular_lattice(25, 0.8).is_err());
}

#[test]
fn seeded_velocities_are_reproducible_zero_momentum_and_on_temperature() {
    let first = initial_velocities(100, 0.5, 2026).unwrap();
    let same = initial_velocities(100, 0.5, 2026).unwrap();
    let different = initial_velocities(100, 0.5, 2027).unwrap();
    assert_eq!(first, same);
    assert_ne!(first, different);

    let sum = first.iter().fold([0.0, 0.0], |mut sum, velocity| {
        sum[0] += velocity[0];
        sum[1] += velocity[1];
        sum
    });
    assert!(sum[0].abs() < 1.0e-12);
    assert!(sum[1].abs() < 1.0e-12);

    let temperature = first
        .iter()
        .map(|velocity| velocity[0].powi(2) + velocity[1].powi(2))
        .sum::<f64>()
        / 198.0;
    assert!((temperature - 0.5).abs() < 1.0e-12);
}

#[test]
fn short_run_records_only_production_sample_steps() {
    let config = RunConfig {
        n: 16,
        rho: 0.2,
        eq_steps: 50,
        steps: 20,
        sample_every: 5,
        ..RunConfig::default()
    };

    let output = simulate(&config).unwrap();
    let steps: Vec<usize> = output.frames.iter().map(|frame| frame.step).collect();

    assert_eq!(steps, vec![5, 10, 15, 20]);
    assert_eq!(output.frames[0].t, 0.05);
    assert_eq!(output.run.integrator, "velocity-verlet");
}
