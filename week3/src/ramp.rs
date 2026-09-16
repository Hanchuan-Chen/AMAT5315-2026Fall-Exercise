//! The ascending temperature ramp: warm starts, one random stream, one clock.

use std::path::PathBuf;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::artifacts::{Recorder, write_run_json};
use crate::lattice::Lattice;
use crate::metropolis::sweep;

/// Temperatures closer than this are the same grid point.
const GRID_EPSILON: f64 = 1e-9;

/// Grids longer than this are an argument mistake, not a run.
pub const MAX_GRID: usize = 100_000;

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

/// `{ t_from + k * t_step <= t_to }`, ascending, rounded to nine decimals.
///
/// `t_to` is included exactly when the step lands on it. The rounding removes
/// binary floating point dust (`1.5 + 40 * 0.05` is `3.5000000000000004`)
/// before the bound is compared, so the last grid point is never dropped.
pub fn temperature_grid(t_from: f64, t_to: f64, t_step: f64) -> Vec<f64> {
    let mut grid = Vec::new();
    if !t_step.is_finite() || t_step <= 0.0 || !t_from.is_finite() || !t_to.is_finite() {
        return grid;
    }
    let mut k = 0u64;
    while grid.len() <= MAX_GRID {
        let temperature = round_nine(t_from + k as f64 * t_step);
        if temperature > t_to + GRID_EPSILON {
            break;
        }
        grid.push(temperature);
        k += 1;
    }
    grid
}

/// Nine decimals is finer than any temperature step this command needs.
fn round_nine(value: f64) -> f64 {
    (value * 1e9).round() / 1e9
}

impl RunConfig {
    /// Reject settings that cannot produce a contract-conforming run.
    pub fn validate(&self) -> Result<(), String> {
        if self.l < 2 {
            return Err(format!(
                "--l must be an integer of at least 2, got {}",
                self.l
            ));
        }
        if self.t_from <= 0.0 || !self.t_from.is_finite() {
            return Err(format!("--t-from must be positive, got {}", self.t_from));
        }
        if !self.t_step.is_finite() || self.t_step <= 0.0 {
            return Err(format!("--t-step must be positive, got {}", self.t_step));
        }
        if !self.t_to.is_finite() {
            return Err(format!("--t-to must be finite, got {}", self.t_to));
        }
        if self.measure == 0 {
            return Err("--measure must be at least 1".to_string());
        }
        let grid = self.t_grid();
        if grid.is_empty() {
            return Err(format!(
                "--t-to {} is not reachable from --t-from {} with --t-step {}",
                self.t_to, self.t_from, self.t_step
            ));
        }
        if grid.len() > MAX_GRID {
            return Err(format!(
                "--t-step {} gives more than {MAX_GRID} temperatures",
                self.t_step
            ));
        }
        Ok(())
    }

    /// The temperatures this run visits.
    pub fn t_grid(&self) -> Vec<f64> {
        temperature_grid(self.t_from, self.t_to, self.t_step)
    }
}

/// Run the whole ramp, write the artifacts, and report on each temperature.
///
/// The first temperature starts from an all-up lattice; every later one starts
/// from the lattice the previous temperature finished on. One `ChaCha8Rng`
/// seeded from `config.seed` supplies the whole ramp, so two runs with the same
/// arguments write identical bytes.
pub fn run_ramp<F>(
    config: &RunConfig,
    mut on_temperature: F,
) -> Result<Vec<TemperatureResult>, String>
where
    F: FnMut(&TemperatureResult),
{
    if config.update != Update::Metropolis {
        return Err(
            "--update wolff lands in a later phase; this build implements metropolis only"
                .to_string(),
        );
    }
    config.validate()?;
    let grid = config.t_grid();
    std::fs::create_dir_all(&config.out)
        .map_err(|error| format!("create {}: {error}", config.out.display()))?;
    write_run_json(config, &grid)?;
    let mut recorder = Recorder::create(&config.out, config.l, config.every)?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut lattice = Lattice::all_up(config.l);
    let mut global_sweep = 0u64;
    let mut results = Vec::with_capacity(grid.len());

    for &temperature in &grid {
        let mut accepted = 0u64;
        let mut proposals = 0u64;
        for _ in 0..config.discard {
            let stats = sweep(&mut lattice, temperature, &mut rng);
            accepted += stats.accepted;
            proposals += stats.proposals;
            global_sweep += 1;
        }

        let mut sum_abs_m = 0.0;
        let mut frames = 0u64;
        for step in 1..=config.measure {
            let stats = sweep(&mut lattice, temperature, &mut rng);
            accepted += stats.accepted;
            proposals += stats.proposals;
            global_sweep += 1;

            let m = lattice.magnetization();
            sum_abs_m += m.abs();
            recorder.write_series_row(temperature, step, m, lattice.energy_per_site())?;
            if config.every > 0 && step % config.every == 0 {
                recorder.write_spin_frame(temperature, global_sweep, m, lattice.spins())?;
                frames += 1;
            }
        }

        let result = TemperatureResult {
            t: temperature,
            mean_abs_m: sum_abs_m / config.measure as f64,
            acceptance: accepted as f64 / proposals as f64,
            frames,
            measured: config.measure,
        };
        on_temperature(&result);
        results.push(result);
    }
    recorder.finish()?;
    Ok(results)
}
