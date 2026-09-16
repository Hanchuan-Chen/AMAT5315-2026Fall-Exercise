//! Command-line arguments.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use crate::ramp::{RunConfig, Update};

/// `ising` samples the Ising model along one temperature ramp.
#[derive(Debug, Parser)]
#[command(
    name = "ising",
    version,
    about = "sample the Ising model along a temperature ramp; no unstated defaults"
)]
pub struct Args {
    /// metropolis or wolff; one step is an l*l-proposal sweep or a cluster flip.
    #[arg(long, value_enum)]
    pub update: UpdateArg,

    /// Integer lattice side, at least 2.
    #[arg(long)]
    pub l: usize,

    /// Lowest temperature; the ramp ascends from here, from an all-up lattice.
    #[arg(long)]
    pub t_from: f64,

    /// Temperature upper bound; included only if reached by the temperature step.
    #[arg(long)]
    pub t_to: f64,

    /// Temperature step; each temperature starts from the previous lattice.
    #[arg(long)]
    pub t_step: f64,

    /// Equilibration steps discarded at each temperature.
    #[arg(long)]
    pub discard: u64,

    /// Measured steps at each temperature.
    #[arg(long)]
    pub measure: u64,

    /// Random seed; one random stream per run, carried through the ramp.
    #[arg(long)]
    pub seed: u64,

    /// Record at measured steps every, 2*every, ... at each temperature; 0 records none.
    #[arg(long, default_value_t = 0)]
    pub every: u64,

    /// Output folder; may exist; overwrite same-named files written by this run.
    #[arg(long)]
    pub out: PathBuf,
}

/// The `--update` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum UpdateArg {
    Metropolis,
    Wolff,
}

impl Args {
    /// Validate the arguments and build the run settings.
    pub fn into_config(self) -> Result<RunConfig, String> {
        todo!()
    }
}

impl From<UpdateArg> for Update {
    fn from(value: UpdateArg) -> Self {
        match value {
            UpdateArg::Metropolis => Update::Metropolis,
            UpdateArg::Wolff => Update::Wolff,
        }
    }
}
