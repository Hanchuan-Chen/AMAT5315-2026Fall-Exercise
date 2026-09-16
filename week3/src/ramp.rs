//! The ascending temperature ramp: warm starts, one random stream, one clock.

use std::path::PathBuf;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::artifacts::{Recorder, write_run_json};
use crate::lattice::Lattice;
use crate::metropolis::sweep;
use crate::wolff::cluster_flip;

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

impl Update {
    /// The name this rule writes into `run.json`.
    pub fn name(self) -> &'static str {
        match self {
            Update::Metropolis => "metropolis",
            Update::Wolff => "wolff",
        }
    }

    /// The unit of one step, which is also what `run.json` calls the clock.
    pub fn time_unit(self) -> &'static str {
        match self {
            Update::Metropolis => "sweep",
            Update::Wolff => "cluster_flip",
        }
    }

    /// The quantity in the third column of the printed table.
    pub fn statistic(self) -> &'static str {
        match self {
            Update::Metropolis => "acceptance",
            Update::Wolff => "mean_cluster_size",
        }
    }
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
    /// Metropolis: accepted proposals per proposal over discard + measure.
    /// Zero for a Wolff run, which has no proposals.
    pub acceptance: f64,
    /// Wolff: spins flipped per step over discard + measure. Zero for a
    /// Metropolis run, whose steps flip one spin per accepted proposal.
    pub mean_cluster_size: f64,
    pub frames: u64,
    pub measured: u64,
}

/// What one step of the chosen rule cost.
#[derive(Clone, Copy, Debug, Default)]
struct StepStats {
    /// Accepted proposals (Metropolis only).
    accepted: u64,
    /// Proposals made (Metropolis only).
    proposals: u64,
    /// Spins flipped by this step (Wolff only).
    cluster_size: u64,
}

/// One step: an `l * l`-proposal sweep, or one cluster flip.
fn one_step(
    update: Update,
    lattice: &mut Lattice,
    temperature: f64,
    rng: &mut ChaCha8Rng,
) -> StepStats {
    match update {
        Update::Metropolis => {
            let stats = sweep(lattice, temperature, rng);
            StepStats {
                accepted: stats.accepted,
                proposals: stats.proposals,
                cluster_size: 0,
            }
        }
        Update::Wolff => StepStats {
            accepted: 0,
            proposals: 0,
            cluster_size: cluster_flip(lattice, temperature, rng) as u64,
        },
    }
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
///
/// One step is a full sweep under Metropolis and one cluster flip under Wolff;
/// `discard` and `measure` count whatever the chosen rule calls a step, and a
/// Wolff move is recorded after every move, never after an accumulated number
/// of flipped spins, so the observation interval cannot depend on the cluster
/// sizes just seen.
pub fn run_ramp<F>(
    config: &RunConfig,
    mut on_temperature: F,
) -> Result<Vec<TemperatureResult>, String>
where
    F: FnMut(&TemperatureResult),
{
    config.validate()?;
    let grid = config.t_grid();
    std::fs::create_dir_all(&config.out)
        .map_err(|error| format!("create {}: {error}", config.out.display()))?;
    write_run_json(config, &grid)?;
    let mut recorder = Recorder::create(&config.out, config.l, config.update, config.every)?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut lattice = Lattice::all_up(config.l);
    let mut global_step = 0u64;
    let mut results = Vec::with_capacity(grid.len());

    for &temperature in &grid {
        let mut accepted = 0u64;
        let mut proposals = 0u64;
        let mut steps = 0u64;
        let mut flipped = 0u64;
        for _ in 0..config.discard {
            let stats = one_step(config.update, &mut lattice, temperature, &mut rng);
            accepted += stats.accepted;
            proposals += stats.proposals;
            steps += 1;
            flipped += stats.cluster_size;
            global_step += 1;
        }

        let mut sum_abs_m = 0.0;
        let mut frames = 0u64;
        for step in 1..=config.measure {
            let stats = one_step(config.update, &mut lattice, temperature, &mut rng);
            accepted += stats.accepted;
            proposals += stats.proposals;
            steps += 1;
            flipped += stats.cluster_size;
            global_step += 1;

            let m = lattice.magnetization();
            sum_abs_m += m.abs();
            recorder.write_series_row(
                temperature,
                step,
                m,
                lattice.energy_per_site(),
                stats.cluster_size,
            )?;
            if config.every > 0 && step % config.every == 0 {
                recorder.write_spin_frame(temperature, global_step, m, lattice.spins())?;
                frames += 1;
            }
        }

        let result = TemperatureResult {
            t: temperature,
            mean_abs_m: sum_abs_m / config.measure as f64,
            acceptance: if proposals == 0 {
                0.0
            } else {
                accepted as f64 / proposals as f64
            },
            mean_cluster_size: flipped as f64 / steps as f64,
            frames,
            measured: config.measure,
        };
        on_temperature(&result);
        results.push(result);
    }
    recorder.finish()?;
    Ok(results)
}
