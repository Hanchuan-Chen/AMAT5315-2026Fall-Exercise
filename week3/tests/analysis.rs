use ising::{
    blocked_error, critical_temperature, integrated_autocorrelation_time, naive_error,
    quadratic_peak, susceptibility,
};

#[test]
fn susceptibility_uses_signed_second_moment_and_absolute_first_moment() {
    let magnetizations = [-0.8, -0.4, 0.4, 0.8];
    let chi = susceptibility(4, 2.0, &magnetizations);
    assert!((chi - 0.32).abs() < 1e-12);
}

#[test]
fn naive_and_blocked_errors_follow_the_sample_standard_error_formulas() {
    let values = [1.0, 2.0, 3.0, 4.0];
    assert!((naive_error(&values) - 0.6454972243679028).abs() < 1e-12);
    assert!((blocked_error(&values, 2) - 1.0).abs() < 1e-12);
}

#[test]
fn autocorrelation_stops_before_a_negative_first_lag() {
    let values = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    assert_eq!(integrated_autocorrelation_time(&values), 0.5);
}

#[test]
fn autocorrelation_detects_a_slow_block_signal() {
    let values: Vec<_> = (0..400)
        .map(|index| if (index / 100) % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    assert!(integrated_autocorrelation_time(&values) > 10.0);
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
