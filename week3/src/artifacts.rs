//! The three files a run writes, in the exact formats the contract fixes.

use std::path::Path;

use crate::ramp::RunConfig;

/// Streams `series.jsonl` and `spins.jsonl` for one run.
pub struct Recorder {
    _private: (),
}

impl Recorder {
    /// Open the writers for `out_dir`; `spins.jsonl` only when `every > 0`.
    pub fn create(out_dir: &Path, every: u64) -> Result<Self, String> {
        todo!()
    }

    /// One measured step.
    pub fn write_series_row(
        &mut self,
        l: usize,
        temperature: f64,
        sweep: u64,
        m: f64,
        energy_per_site: f64,
    ) -> Result<(), String> {
        todo!()
    }

    /// One recorded frame.
    pub fn write_spin_frame(
        &mut self,
        temperature: f64,
        sweep: u64,
        m: f64,
        spins: &[i8],
    ) -> Result<(), String> {
        todo!()
    }
}

/// Write `<out>/run.json`.
pub fn write_run_json(config: &RunConfig, t_grid: &[f64]) -> Result<(), String> {
    todo!()
}
