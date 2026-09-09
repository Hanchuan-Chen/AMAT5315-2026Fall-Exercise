use md::fluid::triangular_lattice;
use md::simulation::{ForceMethod, System};

fn compare_methods(positions: Vec<[f64; 2]>, box_size: [f64; 2], cutoff: f64) {
    let velocities = vec![[0.0, 0.0]; positions.len()];
    let naive = System::new_periodic_with_method(
        positions.clone(),
        velocities.clone(),
        box_size,
        cutoff,
        ForceMethod::Naive,
    )
    .unwrap();
    let cells = System::new_periodic_with_method(
        positions,
        velocities,
        box_size,
        cutoff,
        ForceMethod::Cells,
    )
    .unwrap();

    for (left, right) in naive.accelerations().iter().zip(cells.accelerations()) {
        for axis in 0..2 {
            assert!((left[axis] - right[axis]).abs() < 1.0e-10);
        }
    }
    assert!((naive.potential_energy() - cells.potential_energy()).abs() < 1.0e-10);
}

#[test]
fn cells_match_naive_on_perturbed_lattice() {
    let mut lattice = triangular_lattice(100, 0.8).unwrap();
    for (index, position) in lattice.positions.iter_mut().enumerate() {
        position[0] = (position[0] + 0.013 * (index as f64).sin()).rem_euclid(lattice.box_size[0]);
        position[1] =
            (position[1] + 0.011 * (index as f64).cos()).rem_euclid(lattice.box_size[1]);
    }
    compare_methods(lattice.positions, lattice.box_size, 2.5);
}

#[test]
fn cells_match_naive_for_boundary_cutoff_and_two_cell_box() {
    let positions = vec![
        [0.10, 0.20],
        [5.10, 0.25],
        [1.00, 3.00],
        [3.50, 3.00],
        [2.60, 0.20],
    ];
    compare_methods(positions, [5.2, 5.2], 2.5);
}

