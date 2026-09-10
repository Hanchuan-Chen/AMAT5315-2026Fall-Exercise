use std::fs;

use ising::{SnapshotConfig, write_snapshots};
use serde_json::Value;
use tempfile::tempdir;

fn small_config(path: std::path::PathBuf) -> SnapshotConfig {
    SnapshotConfig {
        l: 6,
        t_start: 1.5,
        t_end: 1.7,
        t_step: 0.1,
        equilibration_sweeps: 2,
        recording_sweeps: 4,
        frame_interval: 2,
        seed: 2026,
        output: path,
    }
}

#[test]
fn snapshot_ramp_obeys_the_frame_contract_and_cadence() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("spins.jsonl");
    let summary = write_snapshots(&small_config(path.clone())).unwrap();
    assert_eq!(summary.frames, 6);

    let text = fs::read_to_string(path).unwrap();
    let rows: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 6);
    assert_eq!(
        rows.iter()
            .map(|row| row["T"].as_f64().unwrap())
            .collect::<Vec<_>>(),
        [1.5, 1.5, 1.6, 1.6, 1.7, 1.7]
    );
    assert_eq!(
        rows.iter()
            .map(|row| row["sweep"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [4, 6, 10, 12, 16, 18]
    );
    for row in rows {
        let object = row.as_object().unwrap();
        let mut keys: Vec<_> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["L", "T", "m", "spins", "sweep"]);
        assert_eq!(row["L"], 6);
        let spins = row["spins"].as_str().unwrap();
        assert_eq!(spins.len(), 36);
        assert!(spins.bytes().all(|b| b == b'0' || b == b'1'));
        assert!((-1.0..=1.0).contains(&row["m"].as_f64().unwrap()));
    }
}

#[test]
fn snapshot_ramp_is_byte_reproducible() {
    let directory = tempdir().unwrap();
    let first = directory.path().join("first.jsonl");
    let second = directory.path().join("second.jsonl");
    write_snapshots(&small_config(first.clone())).unwrap();
    write_snapshots(&small_config(second.clone())).unwrap();
    assert_eq!(fs::read(first).unwrap(), fs::read(second).unwrap());
}
