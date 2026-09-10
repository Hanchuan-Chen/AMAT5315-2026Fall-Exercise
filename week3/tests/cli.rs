use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

#[test]
fn relax_prints_one_summary_and_one_character_per_spin() {
    let output = Command::cargo_bin("ising")
        .expect("binary")
        .args([
            "relax",
            "--l",
            "4",
            "--t",
            "2.3",
            "--sweeps",
            "2",
            "--measure",
            "3",
            "--seed",
            "2026",
        ])
        .output()
        .expect("run command");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 output");
    let lines: Vec<_> = stdout.lines().collect();
    assert_eq!(lines.len(), 5);
    assert!(lines[0].contains("L=4 T=2.300000 sweeps=2 measure=3"));
    assert!(lines[0].contains("mean_abs_m="));
    assert!(lines[0].contains("accept="));
    assert!(lines[1..].iter().all(|line| line.chars().count() == 4));
}

#[test]
fn snapshots_accepts_tunable_protocol_values() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("frames.jsonl");
    Command::cargo_bin("ising")
        .expect("binary")
        .args([
            "snapshots",
            "--l",
            "4",
            "--t-start",
            "1.5",
            "--t-end",
            "1.6",
            "--t-step",
            "0.1",
            "--equilibrate",
            "1",
            "--record",
            "2",
            "--frame-every",
            "1",
            "--seed",
            "9",
            "--output",
        ])
        .arg(&output)
        .assert()
        .success()
        .stdout(predicate::str::contains("frames=4"));
    assert_eq!(std::fs::read_to_string(output).unwrap().lines().count(), 4);
}

#[test]
fn sweep_accepts_tunable_protocol_values() {
    let directory = tempdir().unwrap();
    Command::cargo_bin("ising")
        .expect("binary")
        .args([
            "sweep",
            "--sizes",
            "4",
            "--temperatures",
            "1.5,2.0",
            "--equilibrate",
            "1",
            "--measure",
            "2",
            "--measure-critical",
            "3",
            "--critical-low",
            "2.0",
            "--critical-high",
            "2.6",
            "--seed",
            "9",
            "--output-dir",
        ])
        .arg(directory.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("rows=5"));
    assert_eq!(
        std::fs::read_to_string(directory.path().join("series.jsonl"))
            .unwrap()
            .lines()
            .count(),
        5
    );
}

#[test]
fn plot_reads_a_saved_run_and_writes_both_pngs() {
    let directory = tempdir().unwrap();
    Command::cargo_bin("ising")
        .unwrap()
        .args([
            "sweep",
            "--sizes",
            "4",
            "--temperatures",
            "1.5,1.6,1.7,1.8,1.9",
            "--equilibrate",
            "1",
            "--measure",
            "2",
            "--measure-critical",
            "2",
            "--output-dir",
        ])
        .arg(directory.path())
        .assert()
        .success();
    Command::cargo_bin("ising")
        .unwrap()
        .arg("plot")
        .arg(directory.path())
        .assert()
        .success();
    assert!(directory.path().join("magnetization.png").is_file());
    assert!(directory.path().join("susceptibility.png").is_file());
}

#[test]
fn analyze_prints_both_errors_ratio_and_autocorrelation() {
    let directory = tempdir().unwrap();
    Command::cargo_bin("ising")
        .unwrap()
        .args([
            "sweep",
            "--sizes",
            "4",
            "--temperatures",
            "1.5,1.6,1.7,1.8,1.9",
            "--equilibrate",
            "1",
            "--measure",
            "20",
            "--measure-critical",
            "20",
            "--output-dir",
        ])
        .arg(directory.path())
        .assert()
        .success();
    Command::cargo_bin("ising")
        .unwrap()
        .arg("analyze")
        .arg(directory.path())
        .args(["--blocks", "5"])
        .assert()
        .success()
        .stdout(predicate::str::contains("naive="))
        .stdout(predicate::str::contains("blocked="))
        .stdout(predicate::str::contains("ratio="))
        .stdout(predicate::str::contains("tau_int="));
    assert!(directory.path().join("tau.png").is_file());
}

#[test]
fn sweep_wolff_switches_the_saved_algorithm() {
    let directory = tempdir().unwrap();
    Command::cargo_bin("ising")
        .unwrap()
        .args([
            "sweep",
            "--wolff",
            "--sizes",
            "4",
            "--temperatures",
            "2.0,2.1",
            "--equilibrate",
            "1",
            "--measure-critical",
            "3",
            "--output-dir",
        ])
        .arg(directory.path())
        .assert()
        .success();
    let run: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(directory.path().join("run.json")).unwrap())
            .unwrap();
    assert_eq!(run["algorithm"], "wolff");
}

#[test]
fn compare_writes_the_shared_size_tau_chart() {
    let directory = tempdir().unwrap();
    let metropolis = directory.path().join("metropolis");
    let wolff = directory.path().join("wolff");
    for (output, algorithm_flag) in [(&metropolis, None), (&wolff, Some("--wolff"))] {
        let mut command = Command::cargo_bin("ising").unwrap();
        command.arg("sweep");
        if let Some(flag) = algorithm_flag {
            command.arg(flag);
        }
        command
            .args([
                "--sizes",
                "4",
                "--temperatures",
                "2.0,2.1",
                "--equilibrate",
                "1",
                "--measure",
                "20",
                "--measure-critical",
                "20",
                "--output-dir",
            ])
            .arg(output)
            .assert()
            .success();
    }
    let output = directory.path().join("tau-compare.png");
    Command::cargo_bin("ising")
        .unwrap()
        .arg("compare")
        .arg(&metropolis)
        .arg(&wolff)
        .arg("--output")
        .arg(&output)
        .assert()
        .success()
        .stdout(predicate::str::contains("L=4"));
    assert!(output.is_file());
}

#[test]
fn relax_rejects_invalid_physical_parameters() {
    for args in [
        vec!["relax", "--l", "0", "--t", "2.3", "--measure", "1"],
        vec!["relax", "--l", "4", "--t", "0", "--measure", "1"],
        vec!["relax", "--l", "4", "--t", "2.3", "--measure", "0"],
    ] {
        Command::cargo_bin("ising")
            .expect("binary")
            .args(args)
            .assert()
            .failure()
            .stderr(predicate::str::contains("must"));
    }
}
