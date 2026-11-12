use thermo::{boundary, ising, trg};

#[test]
fn boundary_matches_open_four_by_four_enumeration() {
    let exact = ising::exact_log_z(0.44, 4, false);
    let result = boundary::solve(4, 0.44, 8);
    assert!((result.log_z - exact).abs() < 1e-8);
}

#[test]
fn trg_is_exact_for_two_by_two() {
    let exact = ising::exact_log_z(0.44, 2, true);
    let result = trg::solve(2, 0.44, 16);
    assert!((result.log_z - exact).abs() < 1e-8);
}

#[test]
fn trg_four_by_four_is_within_requirement() {
    let exact = ising::exact_log_z(0.44, 4, true);
    let result = trg::solve(4, 0.44, 8);
    assert!((result.log_z - exact).abs() < 2e-3 * 16.0);
}

#[test]
fn periodic_two_by_two_counts_parallel_bonds() {
    let expected = (12.0f64 + 4.0 * (8.0f64 * 0.44).cosh()).ln();
    let exact = ising::exact_log_z(0.44, 2, true);
    assert!((exact - expected).abs() < 1e-12);
}
