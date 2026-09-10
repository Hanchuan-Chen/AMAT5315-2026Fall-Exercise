use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

use crate::{AcceptanceTable, Lattice, sweep};

#[derive(Clone, Debug)]
pub struct SnapshotConfig {
    pub l: usize,
    pub t_start: f64,
    pub t_end: f64,
    pub t_step: f64,
    pub equilibration_sweeps: usize,
    pub recording_sweeps: usize,
    pub frame_interval: usize,
    pub seed: u64,
    pub output: PathBuf,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            l: 64,
            t_start: 1.5,
            t_end: 3.5,
            t_step: 0.05,
            equilibration_sweeps: 2000,
            recording_sweeps: 200,
            frame_interval: 20,
            seed: 2026,
            output: PathBuf::from("artifacts/spins.jsonl"),
        }
    }
}

#[allow(non_snake_case)]
#[derive(Serialize)]
struct SnapshotFrame {
    L: usize,
    T: f64,
    sweep: usize,
    m: f64,
    spins: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotSummary {
    pub frames: usize,
    pub sweeps: usize,
}

pub fn write_snapshots(config: &SnapshotConfig) -> Result<SnapshotSummary, Box<dyn Error>> {
    if let Some(parent) = config.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(&config.output)?);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut lattice = Lattice::all_up(config.l);
    let steps = ((config.t_end - config.t_start) / config.t_step).round() as usize;
    let mut cumulative_sweeps = 0;
    let mut frames = 0;

    for temperature_index in 0..=steps {
        let temperature = config.t_start + temperature_index as f64 * config.t_step;
        let table = AcceptanceTable::new(temperature);
        for _ in 0..config.equilibration_sweeps {
            sweep(&mut lattice, &table, &mut rng);
            cumulative_sweeps += 1;
        }
        for recorded_sweep in 1..=config.recording_sweeps {
            sweep(&mut lattice, &table, &mut rng);
            cumulative_sweeps += 1;
            if recorded_sweep % config.frame_interval == 0 {
                let frame = SnapshotFrame {
                    L: config.l,
                    T: temperature,
                    sweep: cumulative_sweeps,
                    m: lattice.magnetization(),
                    spins: lattice.spins_binary(),
                };
                serde_json::to_writer(&mut writer, &frame)?;
                writer.write_all(b"\n")?;
                frames += 1;
            }
        }
    }
    writer.flush()?;
    Ok(SnapshotSummary {
        frames,
        sweeps: cumulative_sweeps,
    })
}
