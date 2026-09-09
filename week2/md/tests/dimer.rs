use md::simulation::{Euler, System, VelocityVerlet, run_dimer};

const DT: f64 = 0.01;
const TOLERANCE: f64 = 1.0e-12;

fn max_abs(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    assert!(values.iter().all(|value| value.is_finite()));
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

#[test]
fn dimer_accelerations_are_equal_opposite_and_attractive() {
    let system = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0, 0.0], [0.0, 0.0]]);
    let acceleration = system.accelerations();

    assert!(acceleration[0][0] > 0.0);
    assert!(acceleration[1][0] < 0.0);
    assert!(acceleration[0][1].abs() < TOLERANCE);
    assert!(acceleration[1][1].abs() < TOLERANCE);
    for (first, second) in acceleration[0].iter().zip(&acceleration[1]) {
        assert!((first + second).abs() < TOLERANCE);
    }
}

#[test]
fn same_dimer_run_distinguishes_euler_from_verlet() {
    let euler = run_dimer(&Euler, 500, DT);
    let verlet = run_dimer(&VelocityVerlet, 500, DT);

    assert_eq!(euler.times().len(), 500);
    assert_eq!(euler.relative_errors().len(), 500);
    assert_eq!(verlet.times().len(), 500);
    assert_eq!(verlet.relative_errors().len(), 500);
    assert!((euler.times()[0] - 0.01).abs() < TOLERANCE);
    assert!((euler.times()[499] - 5.0).abs() < TOLERANCE);
    assert!((verlet.times()[0] - 0.01).abs() < TOLERANCE);
    assert!((verlet.times()[499] - 5.0).abs() < TOLERANCE);
    assert!(max_abs(verlet.relative_errors()) < 1.0e-3);
    assert!(*euler.relative_errors().last().unwrap() > 0.5);
}

#[test]
fn verlet_error_stays_bounded_for_ten_times_longer() {
    let trace = run_dimer(&VelocityVerlet, 5000, DT);

    assert_eq!(trace.times().len(), 5000);
    assert_eq!(trace.relative_errors().len(), 5000);
    assert!((trace.times()[0] - 0.01).abs() < TOLERANCE);
    assert!((trace.times()[4999] - 50.0).abs() < TOLERANCE);
    assert!(max_abs(trace.relative_errors()) < 1.0e-3);
}
