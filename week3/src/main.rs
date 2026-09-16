//! `ising`: sample the Ising model along one temperature ramp.

use std::io::Write;
use std::process::ExitCode;

use clap::Parser;
use ising::Update;
use ising::cli::Args;
use ising::ramp::run_ramp;

fn main() -> ExitCode {
    match execute(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn execute(args: Args) -> Result<(), String> {
    let config = args.into_config()?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    // The third column is the acceptance rate for metropolis and the mean
    // cluster size for wolff; the header names whichever the run prints.
    writeln!(out, "T\tmean|M|\t{}", config.update.statistic()).map_err(write_error)?;
    out.flush().map_err(write_error)?;

    let mut table_failed = false;
    run_ramp(&config, |result| {
        let third = match config.update {
            Update::Metropolis => result.acceptance,
            Update::Wolff => result.mean_cluster_size,
        };
        if writeln!(
            out,
            "{:.3}\t{:.4}\t{:.4}",
            result.t, result.mean_abs_m, third
        )
        .is_err()
        {
            table_failed = true;
        }
    })?;

    if table_failed {
        return Err("could not write the stdout table".to_string());
    }
    Ok(())
}

fn write_error(error: std::io::Error) -> String {
    format!("write stdout: {error}")
}
