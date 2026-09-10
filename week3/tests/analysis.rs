use ising::{critical_temperature, quadratic_peak, susceptibility};

#[test]
fn susceptibility_uses_signed_second_moment_and_absolute_first_moment() {
    let magnetizations = [-0.8, -0.4, 0.4, 0.8];
    let chi = susceptibility(4, 2.0, &magnetizations);
    assert!((chi - 0.32).abs() < 1e-12);
}

#[test]
fn five_point_quadratic_fit_returns_the_vertex() {
    let points: Vec<_> = [2.2_f64, 2.25, 2.3, 2.35, 2.4]
        .into_iter()
        .map(|temperature| (temperature, 10.0 - (temperature - 2.31).powi(2)))
        .collect();
    assert!((quadratic_peak(&points).unwrap() - 2.31).abs() < 1e-9);
}

#[test]
fn two_size_extrapolation_cancels_the_leading_inverse_size_shift() {
    assert!((critical_temperature(2.35, 2.31) - 2.27).abs() < 1e-12);
}
