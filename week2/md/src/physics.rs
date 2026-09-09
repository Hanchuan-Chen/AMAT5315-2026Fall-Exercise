/// Lennard-Jones pair energy in reduced units.
pub fn energy(_r: f64) -> f64 {
    todo!("implement the Lennard-Jones pair energy")
}

/// Radial Lennard-Jones pair force in reduced units.
///
/// Positive values are repulsive and negative values are attractive.
pub fn force(_r: f64) -> f64 {
    todo!("implement the radial Lennard-Jones pair force")
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
