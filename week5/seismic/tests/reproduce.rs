//! End-to-end checks on the published reflector experiment.

use seismic::experiment::Experiment;
use seismic::npy;
use seismic::sim::Model;
use std::path::{Path, PathBuf};

fn reflector() -> Model {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inputs/reflector.json");
    Model::new(Experiment::load(&path))
}

#[test]
fn forward_matches_the_published_trace_norm_and_peak() {
    let model = reflector();
    let mut norm = 0.0;
    let mut peaks = Vec::new();
    for shot in 0..model.exp.shots.len() {
        let run = model.forward(shot, &model.exp.background, None, false, None);
        norm += run.traces.iter().map(|v| v * v).sum::<f64>();
        let mut peak = 0.0f64;
        for value in &run.traces {
            peak = peak.max(value.abs());
        }
        peaks.push(peak);
    }
    let norm = norm.sqrt();
    assert!((norm - 11.574770).abs() / 11.574770 < 1e-6, "trace norm {norm}");
    assert!((peaks[0] - 0.60809514).abs() / 0.60809514 < 1e-6);
    assert!((peaks[1] - 0.59271397).abs() / 0.59271397 < 1e-6);
}

#[test]
fn born_and_full_adjoint_obey_the_transpose_identity() {
    let model = reflector();
    let born = model.born();
    let mut norm = 0.0;
    for data in &born {
        norm += data.iter().map(|v| v * v).sum::<f64>();
    }
    assert!((norm.sqrt() - 0.18667590).abs() / 0.18667590 < 1e-6);
    let migrated = model.migrate_full(&born, None);
    let mut lhs = 0.0;
    let mut rhs = 0.0;
    for data in &born {
        lhs += data.iter().map(|v| v * v).sum::<f64>();
    }
    for (m, image) in model.exp.perturbation.iter().zip(&migrated.image) {
        rhs += m * image;
    }
    assert!((lhs - rhs).abs() / lhs.abs().max(rhs.abs()) < 1e-12, "{lhs} vs {rhs}");
    let profile: Vec<f64> = (10..34)
        .map(|z| {
            let row = &migrated.image[z * model.exp.nx + 7..z * model.exp.nx + 34];
            row.iter().map(|v| v * v).sum::<f64>().sqrt()
        })
        .collect();
    let peak = 10 + profile.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(peak, 21);
}

#[test]
fn npy_round_trip_preserves_shape_and_values() {
    let dir = std::env::temp_dir().join("seismic-npy-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path: PathBuf = dir.join("sample.npy");
    let data = vec![1.0f64, -2.5, 3.25, 0.0, 4.5, 6.0];
    npy::write_f64(&path, &[2, 3], &data).unwrap();
    let (shape, back) = npy::read_f64(&path).unwrap();
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(back, data);
}
