use clap::{Parser, Subcommand};

use std::path::PathBuf;

use crate::{RelaxConfig, SnapshotConfig, relax, write_snapshots};

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
    }
}
