use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use md::analysis::{CheckError, check_run};
use md::artifacts::{read_artifacts, write_artifacts};
use md::fluid::{RunConfig, simulate};
use md::video::render_video;

#[derive(Parser)]
#[command(
    name = "md",
    about = "Two-dimensional Lennard-Jones molecular dynamics"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Run {
        #[arg(long, default_value_t = 100)]
        n: usize,
        #[arg(long, default_value_t = 0.8)]
        rho: f64,
        #[arg(long, default_value_t = 0.5)]
        temperature: f64,
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        #[arg(long, default_value_t = 2000)]
        eq_steps: usize,
        #[arg(long, default_value_t = 10_000)]
        steps: usize,
        #[arg(long, default_value_t = 50)]
        sample_every: usize,
        #[arg(long, default_value_t = 2026)]
        seed: u64,
        #[arg(long, default_value = "artifacts")]
        out: PathBuf,
    },
    Check {
        artifacts: PathBuf,
    },
    Video {
        artifacts: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

struct CliFailure {
    code: u8,
    message: String,
}

fn execute(command: Command) -> Result<(), CliFailure> {
    match command {
        Command::Run {
            n,
            rho,
            temperature,
            dt,
            eq_steps,
            steps,
            sample_every,
            seed,
            out,
        } => {
            let config = RunConfig {
                n,
                rho,
                temperature,
                dt,
                eq_steps,
                steps,
                sample_every,
                seed,
            };
            let artifacts = simulate(&config).map_err(|error| CliFailure {
                code: 2,
                message: format!("contract: {error}"),
            })?;
            write_artifacts(&out, &artifacts).map_err(|error| CliFailure {
                code: 2,
                message: format!("contract: {error}"),
            })?;
            println!("wrote {}", out.join("run.json").display());
            println!("wrote {}", out.join("traj.jsonl").display());
            Ok(())
        }
        Command::Check { artifacts } => {
            let artifacts = read_artifacts(&artifacts).map_err(|error| CliFailure {
                code: 2,
                message: format!("contract: {error}"),
            })?;
            let report = check_run(&artifacts).map_err(|error| match error {
                CheckError::Contract(message) => CliFailure {
                    code: 2,
                    message: format!("contract: {message}"),
                },
                CheckError::Physics(message) => CliFailure {
                    code: 1,
                    message: format!("physics: {message}"),
                },
            })?;
            println!(
                "max energy-consistency relative error = {:.3e} (< 1e-6)",
                report.max_energy_consistency
            );
            println!(
                "secular energy drift = {:.3e} (< 2e-3)",
                report.secular_drift
            );
            println!(
                "max energy oscillation = {:.3e} (not gated)",
                report.max_energy_oscillation
            );
            println!("temperature = {:.4} (|T - 0.5| < 0.05)", report.temperature);
            println!("Rayleigh chi2/dof = {:.3} (< 2.0)", report.chi2_per_dof);
            println!("PASS");
            Ok(())
        }
        Command::Video { artifacts, out } => {
            let artifacts = read_artifacts(&artifacts).map_err(|error| CliFailure {
                code: 2,
                message: format!("contract: {error}"),
            })?;
            render_video(&artifacts, &out).map_err(|error| CliFailure {
                code: 1,
                message: format!("video: {error}"),
            })?;
            println!(
                "wrote {} frames to {}",
                artifacts.frames.len(),
                out.display()
            );
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match execute(Cli::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("FAIL: {}", error.message);
            ExitCode::from(error.code)
        }
    }
}
