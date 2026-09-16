//! One random stream per run: the seed decides every byte the run writes.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

const SHARED: [&str; 16] = [
    "--update",
    "metropolis",
    "--l",
    "8",
    "--t-from",
    "2.0",
    "--t-to",
    "2.2",
    "--t-step",
    "0.1",
    "--discard",
    "5",
    "--measure",
    "10",
    "--every",
    "5",
];

/// The same run with the cluster rule: one step is one cluster flip.
const WOLFF_SHARED: [&str; 16] = [
    "--update",
    "wolff",
    "--l",
    "8",
    "--t-from",
    "2.0",
    "--t-to",
    "2.2",
    "--t-step",
    "0.1",
    "--discard",
    "5",
    "--measure",
    "10",
    "--every",
    "5",
];

fn run_with(shared: &[&str], seed: &str) -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("temp dir");
    let out = dir.path().join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args(shared)
        .args(["--seed", seed])
        .arg("--out")
        .arg(&out)
        .output()
        .expect("ising runs");
    assert!(
        output.status.success(),
        "seed {seed} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    (dir, out)
}

fn run(seed: &str) -> (TempDir, PathBuf) {
    run_with(&SHARED, seed)
}

fn run_wolff(seed: &str) -> (TempDir, PathBuf) {
    run_with(&WOLFF_SHARED, seed)
}

fn read(dir: &Path, name: &str) -> Vec<u8> {
    fs::read(dir.join(name)).unwrap_or_else(|error| panic!("read {name}: {error}"))
}

#[test]
fn the_same_seed_reproduces_every_artifact_byte_for_byte() {
    let (_first_dir, first) = run("2026");
    let (_second_dir, second) = run("2026");
    for name in ["run.json", "series.jsonl", "spins.jsonl"] {
        assert_eq!(
            read(&first, name),
            read(&second, name),
            "{name} differs between two seed-2026 runs"
        );
    }
}

#[test]
fn a_different_seed_changes_the_series() {
    let (_dir_2026, first) = run("2026");
    let (_dir_2027, second) = run("2027");
    assert_ne!(
        read(&first, "series.jsonl"),
        read(&second, "series.jsonl"),
        "seed 2027 produced the same series as seed 2026"
    );
    assert_ne!(read(&first, "spins.jsonl"), read(&second, "spins.jsonl"));
}

#[test]
fn the_seed_is_the_only_difference_in_run_json() {
    let (_dir_2026, first) = run("2026");
    let (_dir_2027, second) = run("2027");
    let first: serde_json::Value = serde_json::from_slice(&read(&first, "run.json")).unwrap();
    let second: serde_json::Value = serde_json::from_slice(&read(&second, "run.json")).unwrap();
    assert_eq!(first["seed"], 2026);
    assert_eq!(second["seed"], 2027);
    let mut first = first.as_object().unwrap().clone();
    let mut second = second.as_object().unwrap().clone();
    first.remove("seed");
    second.remove("seed");
    assert_eq!(first, second);
}

#[test]
fn the_same_seed_reproduces_every_wolff_artifact_byte_for_byte() {
    let (_first_dir, first) = run_wolff("42");
    let (_second_dir, second) = run_wolff("42");
    for name in ["run.json", "series.jsonl", "spins.jsonl"] {
        assert_eq!(
            read(&first, name),
            read(&second, name),
            "{name} differs between two seed-42 wolff runs"
        );
    }
}

#[test]
fn a_different_seed_changes_the_wolff_series() {
    let (_dir_42, first) = run_wolff("42");
    let (_dir_1042, second) = run_wolff("1042");
    assert_ne!(
        read(&first, "series.jsonl"),
        read(&second, "series.jsonl"),
        "seed 1042 produced the same wolff series as seed 42"
    );
    assert_ne!(read(&first, "spins.jsonl"), read(&second, "spins.jsonl"));
}

#[test]
fn the_wolff_seed_is_the_only_difference_in_run_json() {
    let (_dir_42, first) = run_wolff("42");
    let (_dir_1042, second) = run_wolff("1042");
    let first: serde_json::Value = serde_json::from_slice(&read(&first, "run.json")).unwrap();
    let second: serde_json::Value = serde_json::from_slice(&read(&second, "run.json")).unwrap();
    assert_eq!(first["seed"], 42);
    assert_eq!(second["seed"], 1042);
    let mut first = first.as_object().unwrap().clone();
    let mut second = second.as_object().unwrap().clone();
    first.remove("seed");
    second.remove("seed");
    assert_eq!(first, second);
}
