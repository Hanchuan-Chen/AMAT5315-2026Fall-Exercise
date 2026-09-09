use std::{fs, process::Command};

use md::artifacts::read_artifacts;

#[test]
fn run_writes_short_contract_and_check_rejects_corruption() {
    let temp = tempfile::tempdir().unwrap();
    let output_dir = temp.path().join("artifacts");
    let run = Command::new(env!("CARGO_BIN_EXE_md"))
        .args([
            "run",
            "--n",
            "16",
            "--rho",
            "0.2",
            "--eq-steps",
            "50",
            "--steps",
            "20",
            "--sample-every",
            "5",
            "--seed",
            "7",
            "--out",
        ])
        .arg(&output_dir)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(output_dir.join("run.json").is_file());
    assert!(output_dir.join("traj.jsonl").is_file());

    let artifacts = read_artifacts(&output_dir).unwrap();
    assert_eq!(
        artifacts
            .frames
            .iter()
            .map(|frame| frame.step)
            .collect::<Vec<_>>(),
        [5, 10, 15, 20]
    );

    fs::write(output_dir.join("traj.jsonl"), "{}\n").unwrap();
    let check = Command::new(env!("CARGO_BIN_EXE_md"))
        .arg("check")
        .arg(&output_dir)
        .output()
        .unwrap();

    assert_eq!(check.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&check.stderr).contains("contract"));
}
