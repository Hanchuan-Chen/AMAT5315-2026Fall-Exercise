use md::analysis::{check_run, radial_distribution, rayleigh_cdf, rayleigh_quantile};
use md::fluid::{RunConfig, simulate};

#[test]
fn rayleigh_cdf_and_quantile_are_inverses() {
    for probability in [0.01, 0.25, 0.5, 0.95, 0.999] {
        let speed = rayleigh_quantile(probability, 0.5).unwrap();
        assert!((rayleigh_cdf(speed, 0.5).unwrap() - probability).abs() < 1.0e-13);
    }
}

#[test]
fn default_contract_run_passes_every_physics_gate() {
    let artifacts = simulate(&RunConfig::default()).unwrap();
    let report = check_run(&artifacts).unwrap();

    assert!(report.max_energy_consistency < 1.0e-6);
    assert!(report.secular_drift < 2.0e-3);
    assert!((report.temperature - 0.5).abs() < 0.05);
    assert!(report.chi2_per_dof < 2.0);

    let radial = radial_distribution(&artifacts.frames[..20], &artifacts.run, 64).unwrap();
    assert_eq!(radial.r.len(), 64);
    assert_eq!(radial.g.len(), 64);
    assert!(
        radial
            .g
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    );
}
