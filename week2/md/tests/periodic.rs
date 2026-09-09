use md::simulation::{Integrator, System, VelocityVerlet};

const RC: f64 = 2.5;

#[test]
fn periodic_pair_force_is_equal_and_opposite_across_boundary() {
    let system = System::new_periodic(
        vec![[0.25, 3.0], [7.75, 3.0]],
        vec![[0.0, 0.0]; 2],
        [8.0, 8.0],
        RC,
    )
    .unwrap();
    let acceleration = system.accelerations();

    assert!(acceleration[0][0] > 0.0);
    for (first, second) in acceleration[0].iter().zip(&acceleration[1]) {
        assert!((first + second).abs() < 1.0e-12);
    }
}

#[test]
fn shifted_potential_is_zero_at_and_beyond_cutoff() {
    let energy_at = System::new_periodic(
        vec![[1.0, 1.0], [1.0 + RC, 1.0]],
        vec![[0.0, 0.0]; 2],
        [8.0, 8.0],
        RC,
    )
    .unwrap()
    .potential_energy();
    let energy_beyond = System::new_periodic(
        vec![[1.0, 1.0], [1.0 + RC + 0.1, 1.0]],
        vec![[0.0, 0.0]; 2],
        [8.0, 8.0],
        RC,
    )
    .unwrap()
    .potential_energy();
    let energy_inside = System::new_periodic(
        vec![[1.0, 1.0], [1.0 + RC - 1.0e-8, 1.0]],
        vec![[0.0, 0.0]; 2],
        [8.0, 8.0],
        RC,
    )
    .unwrap()
    .potential_energy();

    assert_eq!(energy_at, 0.0);
    assert_eq!(energy_beyond, 0.0);
    assert!(energy_inside.abs() < 1.0e-8);
}

#[test]
fn verlet_wraps_positions_without_changing_free_velocity() {
    let mut system =
        System::new_periodic(vec![[7.5, 1.0]], vec![[1.0, 0.0]], [8.0, 8.0], RC).unwrap();

    VelocityVerlet.step(&mut system, 1.0);

    assert!((system.positions()[0][0] - 0.5).abs() < 1.0e-12);
    assert_eq!(system.velocities()[0], [1.0, 0.0]);
}
