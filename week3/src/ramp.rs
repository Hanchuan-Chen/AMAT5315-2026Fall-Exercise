//! The ascending temperature ramp.

use std::path::PathBuf;

/// The update rule named by `--update`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Update {
    Metropolis,
    Wolff,
}

/// The validated settings of one run.
#[derive(Clone, Debug)]
pub struct RunConfig {
    pub update: Update,
    pub l: usize,
    pub t_from: f64,
    pub t_to: f64,
    pub t_step: f64,
    pub discard: u64,
    pub measure: u64,
    pub seed: u64,
    pub every: u64,
    pub out: PathBuf,
}

/// What one temperature contributed to the printed table.
#[derive(Clone, Debug, PartialEq)]
pub struct TemperatureResult {
    pub t: f64,
    pub mean_abs_m: f64,
    pub acceptance: f64,
    pub frames: u64,
    pub measured: u64,
}

/// `{ t_from + k * t_step <= t_to }` rounded to nine decimals.
pub fn temperature_grid(t_from: f64, t_to: f64, t_step: f64) -> Vec<f64> {
    todo!()
}

impl RunConfig {
    /// The temperatures this run visits.
    pub fn t_grid(&self) -> Vec<f64> {
        todo!()
    }
}

/// Run the whole ramp, writing the artifacts and reporting each temperature.
pub fn run_ramp<F>(config: &RunConfig, on_temperature: F) -> Result<Vec<TemperatureResult>, String>
where
    F: FnMut(&TemperatureResult),
{
    todo!()
}
