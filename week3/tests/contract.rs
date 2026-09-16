//! The contract: the stdout table, the three artifact files, and the T grid.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use ising::temperature_grid;
use serde_json::Value;
use tempfile::TempDir;

/// One finished run in its own temporary output folder.
struct Run {
    _dir: TempDir,
    out: PathBuf,
    output: Output,
}

impl Run {
    fn code(&self) -> Option<i32> {
        self.output.status.code()
    }

    fn stdout(&self) -> String {
        String::from_utf8(self.output.stdout.clone()).expect("utf8 stdout")
    }

    fn stderr(&self) -> String {
        String::from_utf8(self.output.stderr.clone()).expect("utf8 stderr")
    }

    fn exists(&self, name: &str) -> bool {
        self.out.join(name).exists()
    }

    fn text(&self, name: &str) -> String {
        let path = self.out.join(name);
        fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path:?}: {error}"))
    }

    fn json(&self, name: &str) -> Value {
        serde_json::from_str(&self.text(name)).expect("valid json")
    }

    fn jsonl(&self, name: &str) -> Vec<Value> {
        self.text(name)
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("valid json line"))
            .collect()
    }
}

fn run(args: &[&str]) -> Run {
    let dir = TempDir::new().expect("temp dir");
    let out = dir.path().join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args(args)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("ising runs");
    Run {
        _dir: dir,
        out,
        output,
    }
}

/// The text between `"key":` and the next comma or brace on one JSON line.
fn field_text<'a>(line: &'a str, key: &str) -> &'a str {
    let needle = format!("\"{key}\":");
    let start = line.find(&needle).expect("key is present") + needle.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).expect("field ends");
    &rest[..end]
}

fn decimal_digits(text: &str) -> usize {
    match text.split_once('.') {
        Some((_, fraction)) => fraction.len(),
        None => 0,
    }
}

fn sorted_keys(value: &Value) -> Vec<String> {
    let mut keys: Vec<String> = value.as_object().expect("object").keys().cloned().collect();
    keys.sort();
    keys
}

#[test]
fn stdout_has_the_contract_header_and_one_row_per_temperature() {
    let finished = run(&[
        "--update",
        "metropolis",
        "--l",
        "4",
        "--t-from",
        "2.0",
        "--t-to",
        "2.2",
        "--t-step",
        "0.1",
        "--discard",
        "3",
        "--measure",
        "5",
        "--seed",
        "7",
    ]);
    assert_eq!(finished.code(), Some(0), "stderr: {}", finished.stderr());

    let stdout = finished.stdout();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 4, "stdout was {stdout:?}");
    assert_eq!(lines[0], "T\tmean|M|\tacceptance");
    for (index, line) in lines[1..].iter().enumerate() {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 3, "row {line:?}");
        let t: f64 = fields[0].parse().expect("temperature is a number");
        assert!(
            (t - (2.0 + 0.1 * index as f64)).abs() < 1e-9,
            "row {line:?}"
        );
        assert_eq!(decimal_digits(fields[0]), 3, "row {line:?}");
        let mean_m: f64 = fields[1].parse().expect("mean |M| is a number");
        let acceptance: f64 = fields[2].parse().expect("acceptance is a number");
        assert!((0.0..=1.0).contains(&mean_m), "row {line:?}");
        assert!((0.0..=1.0).contains(&acceptance), "row {line:?}");
        assert_eq!(decimal_digits(fields[1]), 4, "row {line:?}");
        assert_eq!(decimal_digits(fields[2]), 4, "row {line:?}");
    }
}

#[test]
fn run_json_carries_exactly_the_contract_fields() {
    let finished = run(&[
        "--update",
        "metropolis",
        "--l",
        "4",
        "--t-from",
        "2.0",
        "--t-to",
        "2.2",
        "--t-step",
        "0.1",
        "--discard",
        "3",
        "--measure",
        "5",
        "--seed",
        "7",
    ]);
    let value = finished.json("run.json");
    assert_eq!(
        sorted_keys(&value),
        [
            "L",
            "discard",
            "measure",
            "sample_every",
            "seed",
            "t_grid",
            "time_unit",
            "update"
        ]
    );
    assert_eq!(value["L"], 4);
    assert_eq!(value["update"], "metropolis");
    assert_eq!(value["discard"], 3);
    assert_eq!(value["measure"], 5);
    assert_eq!(value["seed"], 7);
    assert_eq!(value["sample_every"], 1);
    assert_eq!(value["time_unit"], "sweep");
    let grid: Vec<f64> = value["t_grid"]
        .as_array()
        .expect("t_grid is a list")
        .iter()
        .map(|t| t.as_f64().expect("temperature"))
        .collect();
    assert_eq!(grid, vec![2.0, 2.1, 2.2]);
}

#[test]
fn series_rows_restart_at_one_at_each_temperature() {
    let finished = run(&[
        "--update",
        "metropolis",
        "--l",
        "4",
        "--t-from",
        "2.0",
        "--t-to",
        "2.2",
        "--t-step",
        "0.1",
        "--discard",
        "3",
        "--measure",
        "5",
        "--seed",
        "7",
    ]);
    let rows = finished.jsonl("series.jsonl");
    assert_eq!(rows.len(), 15);

    let mut blocks: Vec<(f64, Vec<u64>)> = Vec::new();
    for row in &rows {
        assert_eq!(sorted_keys(row), ["E", "L", "M", "T", "sweep"]);
        assert_eq!(row["L"], 4);
        let t = row["T"].as_f64().expect("temperature");
        let sweep = row["sweep"].as_u64().expect("sweep");
        let m = row["M"].as_f64().expect("M");
        let e = row["E"].as_f64().expect("E");
        assert!((-1.0..=1.0).contains(&m), "M = {m}");
        assert!((-2.0..=2.0).contains(&e), "E = {e}");
        match blocks.last_mut() {
            Some((last_t, sweeps)) if *last_t == t => sweeps.push(sweep),
            _ => blocks.push((t, vec![sweep])),
        }
    }

    assert_eq!(blocks.len(), 3);
    for (t, sweeps) in &blocks {
        assert_eq!(sweeps, &(1..=5).collect::<Vec<u64>>(), "at T = {t}");
    }
    let temperatures: Vec<f64> = blocks.iter().map(|(t, _)| *t).collect();
    assert!(temperatures.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn series_text_rounds_m_and_e_to_six_decimals() {
    let finished = run(&[
        "--update",
        "metropolis",
        "--l",
        "4",
        "--t-from",
        "2.0",
        "--t-to",
        "2.1",
        "--t-step",
        "0.1",
        "--discard",
        "2",
        "--measure",
        "4",
        "--seed",
        "11",
    ]);
    let text = finished.text("series.jsonl");
    assert_eq!(text.lines().count(), 8);
    for line in text.lines() {
        assert!(line.starts_with("{\"L\":4,\"T\":"), "line {line}");
        assert!(line.ends_with('}'), "line {line}");
        assert_eq!(decimal_digits(field_text(line, "M")), 6, "line {line}");
        assert_eq!(decimal_digits(field_text(line, "E")), 6, "line {line}");
        let m: f64 = field_text(line, "M").parse().expect("M");
        let e: f64 = field_text(line, "E").parse().expect("E");
        assert!((m * 1e6 - (m * 1e6).round()).abs() < 1e-6, "line {line}");
        assert!((e * 1e6 - (e * 1e6).round()).abs() < 1e-6, "line {line}");
    }
}

#[test]
fn spins_rows_carry_the_contract_keys_and_l_squared_spins() {
    let finished = run(&[
        "--update",
        "metropolis",
        "--l",
        "4",
        "--t-from",
        "2.0",
        "--t-to",
        "2.1",
        "--t-step",
        "0.1",
        "--discard",
        "3",
        "--measure",
        "8",
        "--every",
        "2",
        "--seed",
        "2026",
    ]);
    let text = finished.text("spins.jsonl");
    assert_eq!(text.lines().count(), 8);
    for line in text.lines() {
        let row: Value = serde_json::from_str(line).expect("valid json line");
        assert_eq!(sorted_keys(&row), ["L", "T", "m", "spins", "sweep"]);
        assert_eq!(row["L"], 4);
        let spins: Vec<i64> = row["spins"]
            .as_array()
            .expect("spins is an array")
            .iter()
            .map(|spin| spin.as_i64().expect("integer spin"))
            .collect();
        assert_eq!(spins.len(), 16);
        assert!(spins.iter().all(|spin| *spin == 1 || *spin == -1));
        let mean = spins.iter().sum::<i64>() as f64 / 16.0;
        let m = row["m"].as_f64().expect("m");
        assert!((m - mean).abs() < 5e-5, "m {m} vs mean {mean}");
        assert_eq!(decimal_digits(field_text(line, "m")), 4, "line {line}");
    }
}

#[test]
fn spins_jsonl_is_written_only_when_every_is_positive() {
    let shared = [
        "--update",
        "metropolis",
        "--l",
        "4",
        "--t-from",
        "2.0",
        "--t-to",
        "2.1",
        "--t-step",
        "0.1",
        "--discard",
        "2",
        "--measure",
        "6",
        "--seed",
        "2026",
    ];

    let default_every = run(&shared);
    assert!(default_every.exists("series.jsonl"));
    assert!(!default_every.exists("spins.jsonl"));

    let mut zero = shared.to_vec();
    zero.extend(["--every", "0"]);
    let zero_every = run(&zero);
    assert!(!zero_every.exists("spins.jsonl"));

    let mut two = shared.to_vec();
    two.extend(["--every", "2"]);
    let every_two = run(&two);
    assert_eq!(every_two.jsonl("spins.jsonl").len(), 6);
}

#[test]
fn spins_sweep_counts_the_whole_ramp_including_discard() {
    let finished = run(&[
        "--update",
        "metropolis",
        "--l",
        "4",
        "--t-from",
        "2.0",
        "--t-to",
        "2.2",
        "--t-step",
        "0.1",
        "--discard",
        "3",
        "--measure",
        "8",
        "--every",
        "2",
        "--seed",
        "2026",
    ]);
    let frames: Vec<u64> = finished
        .jsonl("spins.jsonl")
        .iter()
        .map(|frame| frame["sweep"].as_u64().expect("sweep"))
        .collect();
    assert_eq!(frames, vec![5, 7, 9, 11, 16, 18, 20, 22, 27, 29, 31, 33]);

    let series: Vec<u64> = finished
        .jsonl("series.jsonl")
        .iter()
        .map(|row| row["sweep"].as_u64().expect("sweep"))
        .collect();
    let expected: Vec<u64> = (0..3).flat_map(|_| 1..=8).collect();
    assert_eq!(series, expected);
}

#[test]
fn frame_spins_agree_with_the_series_row_at_the_same_step() {
    let finished = run(&[
        "--update",
        "metropolis",
        "--l",
        "4",
        "--t-from",
        "2.0",
        "--t-to",
        "2.0",
        "--t-step",
        "0.1",
        "--discard",
        "3",
        "--measure",
        "8",
        "--every",
        "2",
        "--seed",
        "2026",
    ]);
    let series = finished.jsonl("series.jsonl");
    let frames = finished.jsonl("spins.jsonl");
    assert_eq!(frames.len(), 4);
    for frame in &frames {
        let step = frame["sweep"].as_u64().expect("sweep") - 3;
        let row = series
            .iter()
            .find(|row| row["sweep"].as_u64() == Some(step))
            .expect("the same measured step is in the series");
        let frame_m = frame["m"].as_f64().expect("m");
        let series_m = row["M"].as_f64().expect("M");
        assert!(
            (frame_m - series_m).abs() < 1e-4,
            "step {step}: {frame_m} vs {series_m}"
        );
    }
}

#[test]
fn the_ramp_includes_t_to_only_when_the_step_reaches_it() {
    let reached = run(&[
        "--update",
        "metropolis",
        "--l",
        "2",
        "--t-from",
        "1.5",
        "--t-to",
        "3.5",
        "--t-step",
        "0.05",
        "--discard",
        "0",
        "--measure",
        "1",
        "--seed",
        "1",
    ]);
    let grid = reached.json("run.json")["t_grid"].clone();
    let grid = grid.as_array().expect("t_grid is a list");
    assert_eq!(grid.len(), 41);
    assert_eq!(grid[0].as_f64(), Some(1.5));
    assert_eq!(grid[40].as_f64(), Some(3.5));
    assert_eq!(reached.stdout().lines().count(), 42);

    let overshoot = run(&[
        "--update",
        "metropolis",
        "--l",
        "2",
        "--t-from",
        "1.5",
        "--t-to",
        "3.55",
        "--t-step",
        "0.1",
        "--discard",
        "0",
        "--measure",
        "1",
        "--seed",
        "1",
    ]);
    let grid = overshoot.json("run.json")["t_grid"].clone();
    let grid = grid.as_array().expect("t_grid is a list");
    assert_eq!(grid.len(), 21);
    assert_eq!(grid[20].as_f64(), Some(3.5));
}

#[test]
fn temperature_grid_follows_the_inclusion_rule() {
    let grid = temperature_grid(1.5, 3.5, 0.05);
    assert_eq!(grid.len(), 41);
    assert_eq!(grid[0], 1.5);
    assert_eq!(grid[16], 2.3, "no floating point dust in the grid");
    assert_eq!(*grid.last().expect("last temperature"), 3.5);
    assert!(grid.windows(2).all(|pair| pair[0] < pair[1]));

    assert_eq!(temperature_grid(1.5, 3.55, 0.1).len(), 21);
    assert_eq!(*temperature_grid(1.5, 3.55, 0.1).last().unwrap(), 3.5);
    assert_eq!(temperature_grid(1.5, 1.8, 0.4), vec![1.5]);
    assert_eq!(temperature_grid(2.0, 2.0, 0.1), vec![2.0]);
    assert!(temperature_grid(3.0, 2.0, 0.1).is_empty());
    assert!(temperature_grid(1.5, 3.5, 0.0).is_empty());
}

#[test]
fn invalid_settings_fail_before_any_file_is_written() {
    let cases: [&[&str]; 5] = [
        &[
            "--update",
            "metropolis",
            "--l",
            "1",
            "--t-from",
            "2.0",
            "--t-to",
            "2.0",
            "--t-step",
            "0.1",
            "--discard",
            "0",
            "--measure",
            "1",
            "--seed",
            "1",
        ],
        &[
            "--update",
            "metropolis",
            "--l",
            "4",
            "--t-from",
            "2.0",
            "--t-to",
            "2.0",
            "--t-step",
            "0",
            "--discard",
            "0",
            "--measure",
            "1",
            "--seed",
            "1",
        ],
        &[
            "--update",
            "metropolis",
            "--l",
            "4",
            "--t-from",
            "3.0",
            "--t-to",
            "2.0",
            "--t-step",
            "0.1",
            "--discard",
            "0",
            "--measure",
            "1",
            "--seed",
            "1",
        ],
        &[
            "--update",
            "metropolis",
            "--l",
            "4",
            "--t-from",
            "2.0",
            "--t-to",
            "2.0",
            "--t-step",
            "0.1",
            "--discard",
            "0",
            "--measure",
            "0",
            "--seed",
            "1",
        ],
        &[
            "--update",
            "metropolis",
            "--l",
            "4",
            "--t-from",
            "0.0",
            "--t-to",
            "2.0",
            "--t-step",
            "0.1",
            "--discard",
            "0",
            "--measure",
            "1",
            "--seed",
            "1",
        ],
    ];
    for case in cases {
        let finished = run(case);
        assert_eq!(finished.code(), Some(1), "case {case:?}");
        assert!(
            finished.stderr().starts_with("error: "),
            "case {case:?}: {}",
            finished.stderr()
        );
        assert!(
            !finished.out.exists(),
            "case {case:?} created its output folder"
        );
    }
}

#[test]
fn wolff_run_json_names_the_cluster_flip_time_unit() {
    let finished = run(&[
        "--update",
        "wolff",
        "--l",
        "8",
        "--t-from",
        "2.0",
        "--t-to",
        "2.1",
        "--t-step",
        "0.1",
        "--discard",
        "0",
        "--measure",
        "5",
        "--seed",
        "7",
    ]);
    assert_eq!(finished.code(), Some(0), "stderr: {}", finished.stderr());
    let value = finished.json("run.json");
    assert_eq!(
        sorted_keys(&value),
        [
            "L",
            "discard",
            "measure",
            "sample_every",
            "seed",
            "t_grid",
            "time_unit",
            "update"
        ]
    );
    assert_eq!(value["update"], "wolff");
    assert_eq!(value["time_unit"], "cluster_flip");
    assert_eq!(value["L"], 8);
    assert_eq!(value["sample_every"], 1);
}

#[test]
fn wolff_series_rows_carry_the_flipped_cluster_size() {
    let finished = run(&[
        "--update",
        "wolff",
        "--l",
        "8",
        "--t-from",
        "2.0",
        "--t-to",
        "2.1",
        "--t-step",
        "0.1",
        "--discard",
        "0",
        "--measure",
        "5",
        "--seed",
        "7",
    ]);
    let rows = finished.jsonl("series.jsonl");
    assert_eq!(rows.len(), 10);
    let mut sizes = Vec::new();
    for row in &rows {
        assert_eq!(
            sorted_keys(row),
            ["E", "L", "M", "T", "cluster_size", "sweep"]
        );
        assert_eq!(row["L"], 8);
        let size = row["cluster_size"].as_u64().expect("integer cluster size");
        assert!((1..=64).contains(&size), "cluster size {size}");
        sizes.push(size);
    }
    assert!(sizes.iter().any(|size| *size > 1), "sizes were {sizes:?}");

    // With no discard, the printed mean cluster size is exactly the mean of
    // the recorded sizes.
    let stdout = finished.stdout();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines[0], "T\tmean|M|\tmean_cluster_size");
    let printed: Vec<f64> = lines[1..]
        .iter()
        .map(|line| line.split('\t').nth(2).expect("third column").parse().unwrap())
        .collect();
    for (index, printed) in printed.iter().enumerate() {
        let block = &sizes[index * 5..(index + 1) * 5];
        let mean = block.iter().sum::<u64>() as f64 / block.len() as f64;
        assert!(
            (printed - mean).abs() < 1e-4,
            "T block {index}: printed {printed}, recorded {mean}"
        );
        assert!(
            (1.0..=64.0).contains(printed),
            "printed cluster size {printed}"
        );
    }
}

#[test]
fn wolff_stdout_counts_discarded_cluster_moves_too() {
    let finished = run(&[
        "--update",
        "wolff",
        "--l",
        "8",
        "--t-from",
        "2.0",
        "--t-to",
        "2.0",
        "--t-step",
        "0.1",
        "--discard",
        "40",
        "--measure",
        "20",
        "--seed",
        "7",
    ]);
    assert_eq!(finished.code(), Some(0), "stderr: {}", finished.stderr());
    let line = finished.stdout().lines().nth(1).expect("one row").to_string();
    let third: f64 = line.split('\t').nth(2).unwrap().parse().unwrap();
    assert!((1.0..=64.0).contains(&third), "printed {third}");
    let recorded: Vec<u64> = finished
        .jsonl("series.jsonl")
        .iter()
        .map(|row| row["cluster_size"].as_u64().unwrap())
        .collect();
    let measured = recorded.iter().sum::<u64>() as f64 / recorded.len() as f64;
    assert!(
        (third - measured).abs() < 0.5 * measured,
        "printed {third} is far from the measured {measured}"
    );
}

#[test]
fn wolff_sweep_counters_follow_the_metropolis_rules() {
    let finished = run(&[
        "--update",
        "wolff",
        "--l",
        "4",
        "--t-from",
        "2.0",
        "--t-to",
        "2.2",
        "--t-step",
        "0.1",
        "--discard",
        "3",
        "--measure",
        "8",
        "--every",
        "2",
        "--seed",
        "2026",
    ]);
    let series: Vec<u64> = finished
        .jsonl("series.jsonl")
        .iter()
        .map(|row| row["sweep"].as_u64().expect("sweep"))
        .collect();
    assert_eq!(series, (0..3).flat_map(|_| 1..=8).collect::<Vec<u64>>());

    let frames: Vec<u64> = finished
        .jsonl("spins.jsonl")
        .iter()
        .map(|frame| frame["sweep"].as_u64().expect("sweep"))
        .collect();
    // One step is one cluster flip, so the global counter moves by one per
    // recorded frame just as it does for a Metropolis sweep.
    assert_eq!(frames, vec![5, 7, 9, 11, 16, 18, 20, 22, 27, 29, 31, 33]);

    let mut zero = vec![
        "--update",
        "wolff",
        "--l",
        "4",
        "--t-from",
        "2.0",
        "--t-to",
        "2.1",
        "--t-step",
        "0.1",
        "--discard",
        "2",
        "--measure",
        "6",
        "--seed",
        "2026",
    ];
    zero.extend(["--every", "0"]);
    let zero_every = run(&zero);
    assert!(zero_every.exists("series.jsonl"));
    assert!(!zero_every.exists("spins.jsonl"));
}

#[test]
fn the_observation_interval_does_not_depend_on_the_cluster_sizes() {
    let shared = [
        "--update",
        "wolff",
        "--l",
        "8",
        "--t-from",
        "2.3",
        "--t-to",
        "2.3",
        "--t-step",
        "0.1",
        "--discard",
        "0",
        "--seed",
        "2026",
    ];
    let short = run(&[shared.as_slice(), &["--measure", "20"]].concat());
    let long = run(&[shared.as_slice(), &["--measure", "40"]].concat());
    let short_rows: Vec<String> = short.text("series.jsonl").lines().map(str::to_string).collect();
    let long_rows: Vec<String> = long.text("series.jsonl").lines().map(str::to_string).collect();
    assert_eq!(short_rows.len(), 20);
    assert_eq!(long_rows.len(), 40);
    assert_eq!(
        short_rows,
        long_rows[..20],
        "the first twenty measured moves changed with the run length"
    );
}

#[test]
fn an_unknown_update_is_rejected() {
    let unknown = run(&[
        "--update",
        "glauber",
        "--l",
        "4",
        "--t-from",
        "2.0",
        "--t-to",
        "2.0",
        "--t-step",
        "0.1",
        "--discard",
        "0",
        "--measure",
        "1",
        "--seed",
        "1",
    ]);
    assert_eq!(unknown.code(), Some(2), "{}", unknown.stderr());
}
