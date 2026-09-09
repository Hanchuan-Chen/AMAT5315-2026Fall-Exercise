/// Lennard-Jones pair energy in reduced units.
pub fn energy(r: f64) -> f64 {
    assert!(r.is_finite() && r > 0.0, "separation must be positive");
    let inv_r6 = r.powi(-6);
    4.0 * (inv_r6 * inv_r6 - inv_r6)
}

/// Radial Lennard-Jones pair force in reduced units.
///
/// Positive values are repulsive and negative values are attractive.
pub fn force(r: f64) -> f64 {
    assert!(r.is_finite() && r > 0.0, "separation must be positive");
    let inv_r = r.recip();
    let inv_r6 = inv_r.powi(6);
    24.0 * inv_r * (2.0 * inv_r6 * inv_r6 - inv_r6)
}

#[cfg(test)]
mod tests {
    use super::{energy, force};

    #[test]
    fn well_depth() {
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        assert!((energy(r0) + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn force_is_negative_energy_slope() {
        let h = 1.0e-5;
        let r0 = 2.0_f64.powf(1.0 / 6.0);

        for r in [0.9, 1.05, r0, 1.25, 2.0] {
            let numerical = -(energy(r + h) - energy(r - h)) / (2.0 * h);
            let analytical = force(r);
            let tolerance = 1.0e-6 * analytical.abs().max(1.0);

            assert!(
                (analytical - numerical).abs() < tolerance,
                "r={r}: force={analytical}, -dU/dr={numerical}, tolerance={tolerance}"
            );
        }
    }
}
