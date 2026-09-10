use clap::{Parser, Subcommand};

use crate::{RelaxConfig, relax};

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
    }
}

