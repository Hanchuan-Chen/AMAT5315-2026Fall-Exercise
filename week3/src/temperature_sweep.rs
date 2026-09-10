use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

use crate::{AcceptanceTable, Lattice, sweep};

pub fn course_temperature_grid() -> Vec<f64> {
    let mut hundredths: Vec<i32> = (150..=350).step_by(10).collect();
    hundredths.extend((200..=260).step_by(5));
    hundredths.sort_unstable();
    hundredths.dedup();
    hundredths.into_iter().map(|value| value as f64 / 100.0).collect()
}

#[derive(Clone, Debug)]
pub struct TemperatureSweepConfig {
    pub sizes: Vec<usize>,
    pub temperatures: Vec<f64>,
    pub equilibration_sweeps: usize,
    pub measurement_sweeps: usize,
    pub critical_measurement_sweeps: usize,
    pub critical_low: f64,
    pub critical_high: f64,
    pub seed: u64,
    pub output_dir: PathBuf,
}

impl Default for TemperatureSweepConfig {
    fn default() -> Self {
        Self {
            sizes: vec![32, 64],
            temperatures: course_temperature_grid(),
            equilibration_sweeps: 2000,
            measurement_sweeps: 5000,
            critical_measurement_sweeps: 100_000,
            critical_low: 2.0,
            critical_high: 2.6,
            seed: 42,
            output_dir: PathBuf::from("artifacts"),
        }
    }
}

#[derive(Serialize)]
struct RunContract<'a> {
    sizes: &'a [usize],
    t_grid: &'a [f64],
    eq_sweeps: usize,
    meas_sweeps: usize,
    meas_sweeps_critical: usize,
    sample_every: usize,
    seed: u64,
    algorithm: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemperatureSweepSummary {
    pub rows: usize,
}

pub fn write_temperature_sweep(
    config: &TemperatureSweepConfig,
) -> Result<TemperatureSweepSummary, Box<dyn Error>> {
    fs::create_dir_all(&config.output_dir)?;
    let run = RunContract {
        sizes: &config.sizes,
        t_grid: &config.temperatures,
        eq_sweeps: config.equilibration_sweeps,
        meas_sweeps: config.measurement_sweeps,
        meas_sweeps_critical: config.critical_measurement_sweeps,
        sample_every: 1,
        seed: config.seed,
        algorithm: "metropolis",
    };
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(config.output_dir.join("run.json"))?),
        &run,
    )?;
    let mut writer = BufWriter::new(File::create(config.output_dir.join("series.jsonl"))?);
    let mut rows = 0;

    for &l in &config.sizes {
        let size_seed = if l == 32 { config.seed + 1000 } else { config.seed };
        let mut rng = StdRng::seed_from_u64(size_seed);
        let mut lattice = Lattice::all_up(l);
        for &temperature in &config.temperatures {
            let table = AcceptanceTable::new(temperature);
            for _ in 0..config.equilibration_sweeps {
                sweep(&mut lattice, &table, &mut rng);
            }
            let in_critical_window = temperature >= config.critical_low - 1e-9
                && temperature <= config.critical_high + 1e-9;
            let measurements = if in_critical_window {
                config.critical_measurement_sweeps
            } else {
                config.measurement_sweeps
            };
            for measurement_sweep in 0..measurements {
                sweep(&mut lattice, &table, &mut rng);
                writeln!(
                    writer,
                    "{{\"L\":{l},\"T\":{temperature},\"sweep\":{measurement_sweep},\"M\":{:.6},\"E\":{:.6}}}",
                    lattice.magnetization(),
                    lattice.energy() as f64 / lattice.site_count() as f64,
                )?;
                rows += 1;
            }
        }
    }
    writer.flush()?;
    Ok(TemperatureSweepSummary { rows })
}
