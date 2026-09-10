use std::fs;

use ising::{
    TemperatureSweepConfig, course_temperature_grid, write_temperature_sweep,
    write_wolff_temperature_sweep,
};
use serde_json::Value;
use tempfile::tempdir;

#[test]
fn course_grid_has_the_fixed_coarse_and_critical_points() {
    let grid = course_temperature_grid();
    assert_eq!(grid.len(), 27);
    assert_eq!(grid.first(), Some(&1.5));
    assert_eq!(grid.last(), Some(&3.5));
    assert!(grid.contains(&2.05));
    assert!(grid.contains(&2.6));
    assert!(!grid.contains(&2.65));
}

fn small_config(output_dir: std::path::PathBuf) -> TemperatureSweepConfig {
    TemperatureSweepConfig {
        sizes: vec![4],
        temperatures: vec![1.5, 2.0],
        equilibration_sweeps: 1,
        measurement_sweeps: 2,
        critical_measurement_sweeps: 3,
        critical_low: 2.0,
        critical_high: 2.6,
        seed: 42,
        output_dir,
    }
}

#[test]
fn sweep_writes_exact_contract_in_protocol_order_and_six_decimals() {
    let directory = tempdir().unwrap();
    let summary = write_temperature_sweep(&small_config(directory.path().into())).unwrap();
    assert_eq!(summary.rows, 5);

    let run: Value =
        serde_json::from_str(&fs::read_to_string(directory.path().join("run.json")).unwrap())
            .unwrap();
    let mut run_keys: Vec<_> = run
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    run_keys.sort_unstable();
    assert_eq!(
        run_keys,
        [
            "algorithm",
            "eq_sweeps",
            "meas_sweeps",
            "meas_sweeps_critical",
            "sample_every",
            "seed",
            "sizes",
            "t_grid"
        ]
    );
    assert_eq!(run["algorithm"], "metropolis");
    assert_eq!(run["sample_every"], 1);

    let text = fs::read_to_string(directory.path().join("series.jsonl")).unwrap();
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(lines.len(), 5);
    assert!(lines.iter().all(|line| {
        let m = line
            .split("\"M\":")
            .nth(1)
            .unwrap()
            .split(',')
            .next()
            .unwrap();
        let e = line.split("\"E\":").nth(1).unwrap().trim_end_matches('}');
        m.split('.').nth(1).unwrap().len() == 6 && e.split('.').nth(1).unwrap().len() == 6
    }));
    let rows: Vec<Value> = lines
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        rows.iter()
            .map(|row| row["T"].as_f64().unwrap())
            .collect::<Vec<_>>(),
        [1.5, 1.5, 2.0, 2.0, 2.0]
    );
    assert_eq!(
        rows.iter()
            .map(|row| row["sweep"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [0, 1, 0, 1, 2]
    );
}

#[test]
fn sweep_is_byte_reproducible() {
    let directory = tempdir().unwrap();
    let first = directory.path().join("first");
    let second = directory.path().join("second");
    write_temperature_sweep(&small_config(first.clone())).unwrap();
    write_temperature_sweep(&small_config(second.clone())).unwrap();
    assert_eq!(
        fs::read(first.join("series.jsonl")).unwrap(),
        fs::read(second.join("series.jsonl")).unwrap()
    );
}

#[test]
fn wolff_sweep_reuses_the_contract_and_labels_the_algorithm() {
    let directory = tempdir().unwrap();
    write_wolff_temperature_sweep(&small_config(directory.path().into())).unwrap();
    let run: Value =
        serde_json::from_str(&fs::read_to_string(directory.path().join("run.json")).unwrap())
            .unwrap();
    assert_eq!(run["algorithm"], "wolff");
    assert_eq!(
        fs::read_to_string(directory.path().join("series.jsonl"))
            .unwrap()
            .lines()
            .count(),
        5
    );
}
