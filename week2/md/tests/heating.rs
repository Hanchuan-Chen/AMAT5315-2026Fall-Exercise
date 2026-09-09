use std::fs;

use md::artifacts::write_artifacts;
use md::fluid::{RunConfig, production_target_temperature, simulate};
use md::simulation::ForceMethod;
use serde_json::Value;

#[test]
fn production_target_is_linear_and_reaches_both_endpoints() {
    assert_eq!(production_target_temperature(0.2, 1.2, 0, 20_000), 0.2);
    assert!((production_target_temperature(0.2, 1.2, 10_000, 20_000) - 0.7).abs() < 1.0e-12);
    assert_eq!(production_target_temperature(0.2, 1.2, 20_000, 20_000), 1.2);
}

#[test]
fn heated_samples_follow_target_and_record_ramp() {
    let config = RunConfig {
        n: 16,
        rho: 0.2,
        temperature: 0.2,
        dt: 0.002,
        eq_steps: 50,
        steps: 100,
        sample_every: 50,
        seed: 9,
        force: ForceMethod::Cells,
        ramp_to: Some(0.6),
    };
    let artifacts = simulate(&config).unwrap();
    let temperatures = artifacts
        .frames
        .iter()
        .map(|frame| 2.0 * frame.e_kin / (2 * config.n - 2) as f64)
        .collect::<Vec<_>>();
    assert!((temperatures[0] - 0.4).abs() < 1.0e-12);
    assert!((temperatures[1] - 0.6).abs() < 1.0e-12);
    assert_eq!(artifacts.run.ramp_to, Some(0.6));

    let temp = tempfile::tempdir().unwrap();
    write_artifacts(temp.path(), &artifacts).unwrap();
    let run: Value =
        serde_json::from_str(&fs::read_to_string(temp.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(run["ramp_to"], 0.6);
}
