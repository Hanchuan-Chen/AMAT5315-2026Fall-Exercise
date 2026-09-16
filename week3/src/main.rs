//! `ising`: sample the Ising model along one temperature ramp.

use std::io::Write;
use std::process::ExitCode;

use clap::Parser;
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
    writeln!(out, "T\tmean|M|\tacceptance").map_err(write_error)?;
    out.flush().map_err(write_error)?;

    let mut table_failed = false;
    run_ramp(&config, |result| {
        if writeln!(
            out,
            "{:.3}\t{:.4}\t{:.4}",
            result.t, result.mean_abs_m, result.acceptance
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
