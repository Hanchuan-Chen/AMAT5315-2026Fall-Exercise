use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn relax_prints_one_summary_and_one_character_per_spin() {
    let output = Command::cargo_bin("ising")
        .expect("binary")
        .args([
            "relax", "--l", "4", "--t", "2.3", "--sweeps", "2", "--measure", "3",
            "--seed", "2026",
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
