use std::fs;

use md::artifacts::{Frame, RunArtifacts, RunMetadata, read_artifacts, write_artifacts};
use serde_json::Value;

fn sample() -> RunArtifacts {
    RunArtifacts {
        run: RunMetadata {
            n: 2,
            rho: 0.03125,
            box_size: [8.0, 8.0],
            dt: 0.01,
            temperature: 0.5,
            eq_steps: 0,
            steps: 5,
            sample_every: 5,
            seed: 7,
            integrator: "velocity-verlet".into(),
            ramp_to: None,
        },
        frames: vec![Frame {
            step: 5,
            t: 0.05,
            pos: vec![[1.0, 1.0], [2.0, 2.0]],
            vel: vec![[0.1, 0.0], [-0.1, 0.0]],
            e_pot: -0.1,
            e_kin: 0.01,
        }],
    }
}

fn sorted_keys(value: &Value) -> Vec<String> {
    let mut keys: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    keys
}

#[test]
fn exact_schema_round_trips() {
    let temp = tempfile::tempdir().unwrap();
    let expected = sample();
    write_artifacts(temp.path(), &expected).unwrap();
    let run: Value =
        serde_json::from_str(&fs::read_to_string(temp.path().join("run.json")).unwrap()).unwrap();
    let line = fs::read_to_string(temp.path().join("traj.jsonl")).unwrap();
    let frame: Value = serde_json::from_str(line.trim()).unwrap();

    assert_eq!(
        sorted_keys(&run),
        [
            "box",
            "dt",
            "eq_steps",
            "integrator",
            "n",
            "rho",
            "sample_every",
            "seed",
            "steps",
            "temperature"
        ]
    );
    assert_eq!(
        sorted_keys(&frame),
        ["E_kin", "E_pot", "pos", "step", "t", "vel"]
    );
    assert_eq!(read_artifacts(temp.path()).unwrap(), expected);
}

#[test]
fn malformed_frame_dimensions_are_rejected_without_panic() {
    let temp = tempfile::tempdir().unwrap();
    write_artifacts(temp.path(), &sample()).unwrap();
    let malformed = r#"{"step":5,"t":0.05,"pos":[[0.0,0.0]],"vel":[[0.0,0.0],[0.0,0.0]],"E_pot":0.0,"E_kin":0.0}"#;
    fs::write(temp.path().join("traj.jsonl"), malformed).unwrap();

    let message = read_artifacts(temp.path()).unwrap_err().to_string();

    assert!(message.contains("frame 0"), "{message}");
    assert!(message.contains("n = 2"), "{message}");
}
