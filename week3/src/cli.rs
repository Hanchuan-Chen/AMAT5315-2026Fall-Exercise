use clap::{Parser, Subcommand};

use std::path::PathBuf;

use crate::{
    RelaxConfig, SnapshotConfig, TemperatureSweepConfig, course_temperature_grid, relax,
    write_snapshots, write_temperature_sweep,
};

#[derive(Debug, Parser)]
#[command(name = "ising", about = "Two-dimensional Ising Monte Carlo simulator")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Equilibrate and measure one lattice at one temperature.
    Relax {
        #[arg(long)]
        l: usize,
        #[arg(long = "t")]
        temperature: f64,
        #[arg(long, default_value_t = 2000)]
        sweeps: usize,
        #[arg(long, default_value_t = 2000)]
        measure: usize,
        #[arg(long, default_value_t = 2026)]
        seed: u64,
    },
    /// Record an ascending temperature ramp for the supplied viewer.
    Snapshots {
        #[arg(long, default_value_t = 64)]
        l: usize,
        #[arg(long, default_value_t = 1.5)]
        t_start: f64,
        #[arg(long, default_value_t = 3.5)]
        t_end: f64,
        #[arg(long, default_value_t = 0.05)]
        t_step: f64,
        #[arg(long, default_value_t = 2000)]
        equilibrate: usize,
        #[arg(long, default_value_t = 200)]
        record: usize,
        #[arg(long, default_value_t = 20)]
        frame_every: usize,
        #[arg(long, default_value_t = 2026)]
        seed: u64,
        #[arg(long, default_value = "artifacts/spins.jsonl")]
        output: PathBuf,
    },
    /// Measure raw magnetization and energy across a temperature grid.
    Sweep {
        #[arg(long, value_delimiter = ',', default_value = "32,64")]
        sizes: Vec<usize>,
        #[arg(long, value_delimiter = ',')]
        temperatures: Option<Vec<f64>>,
        #[arg(long, default_value_t = 2000)]
        equilibrate: usize,
        #[arg(long, default_value_t = 5000)]
        measure: usize,
        #[arg(long, default_value_t = 100_000)]
        measure_critical: usize,
        #[arg(long, default_value_t = 2.0)]
        critical_low: f64,
        #[arg(long, default_value_t = 2.6)]
        critical_high: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "artifacts")]
        output_dir: PathBuf,
    },
    /// Draw thermodynamic plots and print the finite-size critical estimate.
    Plot {
        #[arg(default_value = "artifacts")]
        directory: PathBuf,
    },
}

pub fn run() -> Result<(), String> {
    match Cli::parse().command {
        Command::Relax {
            l,
            temperature,
            sweeps,
            measure,
            seed,
        } => {
            if l < 2 {
                return Err("lattice side --l must be at least 2".into());
            }
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err("temperature --t must be finite and positive".into());
            }
            if measure == 0 {
                return Err("--measure must be greater than zero".into());
            }
            let result = relax(RelaxConfig {
                l,
                temperature,
                equilibration_sweeps: sweeps,
                measurement_sweeps: measure,
                seed,
            });
            print!("{}", result.render());
            Ok(())
        }
        Command::Snapshots {
            l,
            t_start,
            t_end,
            t_step,
            equilibrate,
            record,
            frame_every,
            seed,
            output,
        } => {
            if l < 2 {
                return Err("lattice side --l must be at least 2".into());
            }
            if !t_start.is_finite()
                || !t_end.is_finite()
                || !t_step.is_finite()
                || t_start <= 0.0
                || t_end < t_start
                || t_step <= 0.0
            {
                return Err("temperature ramp must be finite, positive, and ascending".into());
            }
            if record == 0 || frame_every == 0 || record % frame_every != 0 {
                return Err("record sweeps must be positive and divisible by --frame-every".into());
            }
            let summary = write_snapshots(&SnapshotConfig {
                l,
                t_start,
                t_end,
                t_step,
                equilibration_sweeps: equilibrate,
                recording_sweeps: record,
                frame_interval: frame_every,
                seed,
                output: output.clone(),
            })
            .map_err(|error| error.to_string())?;
            println!(
                "snapshots frames={} sweeps={} output={}",
                summary.frames,
                summary.sweeps,
                output.display()
            );
            Ok(())
        }
        Command::Sweep {
            sizes,
            temperatures,
            equilibrate,
            measure,
            measure_critical,
            critical_low,
            critical_high,
            seed,
            output_dir,
        } => {
            if sizes.is_empty() || sizes.iter().any(|&l| l < 2) {
                return Err("all lattice sizes must be at least 2".into());
            }
            let temperatures = temperatures.unwrap_or_else(course_temperature_grid);
            if temperatures.is_empty()
                || temperatures
                    .iter()
                    .any(|temperature| !temperature.is_finite() || *temperature <= 0.0)
                || temperatures.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err("temperatures must be finite, positive, and strictly ascending".into());
            }
            if measure == 0 || measure_critical == 0 {
                return Err("measurement sweep counts must be positive".into());
            }
            let summary = write_temperature_sweep(&TemperatureSweepConfig {
                sizes,
                temperatures,
                equilibration_sweeps: equilibrate,
                measurement_sweeps: measure,
                critical_measurement_sweeps: measure_critical,
                critical_low,
                critical_high,
                seed,
                output_dir: output_dir.clone(),
            })
            .map_err(|error| error.to_string())?;
            println!("sweep rows={} output={}", summary.rows, output_dir.display());
            Ok(())
        }
        Command::Plot { directory } => {
            let summary = crate::load_analysis(&directory).map_err(|error| error.to_string())?;
            crate::write_thermodynamic_plots(&summary, &directory)
                .map_err(|error| error.to_string())?;
            for (&l, points) in &summary.by_size {
                if let Some(point) = points.first() {
                    println!(
                        "ordered L={l} T={:.2} mean_abs_m={:.6}",
                        point.temperature, point.mean_abs_m
                    );
                }
            }
            for (&l, &peak) in &summary.peaks {
                println!("peak L={l} T={peak:.6}");
            }
            if let Some(tc) = summary.critical_temperature {
                let deviation = 100.0 * (tc - 2.26919) / 2.26919;
                println!("T_c={tc:.6} Onsager=2.269190 deviation={deviation:+.2}%");
            }
            Ok(())
        }
    }
}
