use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::artifacts::{Frame, RunArtifacts, RunMetadata};
use crate::simulation::{ForceMethod, Integrator, System, SystemError, VelocityVerlet};

pub const CUTOFF: f64 = 2.5;
pub const THERMOSTAT_EVERY: usize = 50;

#[derive(Debug, Clone, PartialEq)]
pub struct RunConfig {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub force: ForceMethod,
    pub ramp_to: Option<f64>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2000,
            steps: 10_000,
            sample_every: 50,
            seed: 2026,
            force: ForceMethod::Cells,
            ramp_to: None,
        }
    }
}

impl RunConfig {
    pub fn validate(&self) -> Result<(), FluidError> {
        square_side(self.n)?;
        if !self.rho.is_finite() || self.rho <= 0.0 {
            return Err(FluidError::Configuration(
                "rho must be finite and positive".into(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(FluidError::Configuration(
                "temperature must be finite and positive".into(),
            ));
        }
        if self
            .ramp_to
            .is_some_and(|temperature| !temperature.is_finite() || temperature <= 0.0)
        {
            return Err(FluidError::Configuration(
                "ramp_to must be finite and positive".into(),
            ));
        }
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(FluidError::Configuration(
                "dt must be finite and positive".into(),
            ));
        }
        if self.steps == 0 || self.sample_every == 0 || self.sample_every > self.steps {
            return Err(FluidError::Configuration(
                "steps and sample_every must define at least one sample".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lattice {
    pub positions: Vec<[f64; 2]>,
    pub box_size: [f64; 2],
}

#[derive(Debug, thiserror::Error)]
pub enum FluidError {
    #[error("configuration: {0}")]
    Configuration(String),
    #[error(transparent)]
    System(#[from] SystemError),
}

fn square_side(n: usize) -> Result<usize, FluidError> {
    let side = (n as f64).sqrt() as usize;
    if side == 0 || side * side != n || !side.is_multiple_of(2) {
        return Err(FluidError::Configuration(
            "n must be a perfect square with an even square root".into(),
        ));
    }
    Ok(side)
}

pub fn triangular_lattice(n: usize, rho: f64) -> Result<Lattice, FluidError> {
    let side = square_side(n)?;
    if !rho.is_finite() || rho <= 0.0 {
        return Err(FluidError::Configuration(
            "rho must be finite and positive".into(),
        ));
    }
    let spacing = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let row_height = 3.0_f64.sqrt() * spacing / 2.0;
    let box_size = [side as f64 * spacing, side as f64 * row_height];
    if box_size.iter().any(|length| *length < 2.0 * CUTOFF) {
        return Err(FluidError::Configuration(format!(
            "box sides must be at least {}; got [{}, {}]",
            2.0 * CUTOFF,
            box_size[0],
            box_size[1]
        )));
    }

    let mut positions = Vec::with_capacity(n);
    for row in 0..side {
        for column in 0..side {
            positions.push([
                (column as f64 + 0.5 * (row % 2) as f64) * spacing,
                row as f64 * row_height,
            ]);
        }
    }
    Ok(Lattice {
        positions,
        box_size,
    })
}

pub fn initial_velocities(
    n: usize,
    temperature: f64,
    seed: u64,
) -> Result<Vec<[f64; 2]>, FluidError> {
    if n < 2 {
        return Err(FluidError::Configuration(
            "at least two atoms are required".into(),
        ));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(FluidError::Configuration(
            "temperature must be finite and positive".into(),
        ));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let scale = temperature.sqrt();
    let mut velocities = (0..n)
        .map(|_| {
            let x: f64 = rng.sample(StandardNormal);
            let y: f64 = rng.sample(StandardNormal);
            [scale * x, scale * y]
        })
        .collect::<Vec<_>>();

    let mut mean = [0.0, 0.0];
    for velocity in &velocities {
        for axis in 0..2 {
            mean[axis] += velocity[axis] / n as f64;
        }
    }
    for velocity in &mut velocities {
        for axis in 0..2 {
            velocity[axis] -= mean[axis];
        }
    }
    let current = velocities
        .iter()
        .map(|velocity| velocity[0].powi(2) + velocity[1].powi(2))
        .sum::<f64>()
        / (2 * n - 2) as f64;
    if !current.is_finite() || current <= 0.0 {
        return Err(FluidError::Configuration(
            "generated velocities have no finite thermal energy".into(),
        ));
    }
    let rescale = (temperature / current).sqrt();
    for velocity in &mut velocities {
        for component in velocity {
            *component *= rescale;
        }
    }
    Ok(velocities)
}

pub fn simulate(config: &RunConfig) -> Result<RunArtifacts, FluidError> {
    config.validate()?;
    let lattice = triangular_lattice(config.n, config.rho)?;
    let velocities = initial_velocities(config.n, config.temperature, config.seed)?;
    let box_size = lattice.box_size;
    let mut system = System::new_periodic_with_method(
        lattice.positions,
        velocities,
        box_size,
        CUTOFF,
        config.force,
    )?;

    for step in 1..=config.eq_steps {
        VelocityVerlet.step(&mut system, config.dt);
        if step % THERMOSTAT_EVERY == 0 {
            system.remove_center_of_mass_velocity();
            system.rescale_temperature(config.temperature)?;
        }
    }
    system.remove_center_of_mass_velocity();

    let mut frames = Vec::with_capacity(config.steps / config.sample_every);
    for step in 1..=config.steps {
        VelocityVerlet.step(&mut system, config.dt);
        if let Some(final_temperature) = config.ramp_to
            && step % THERMOSTAT_EVERY == 0
        {
            system.remove_center_of_mass_velocity();
            system.rescale_temperature(production_target_temperature(
                config.temperature,
                final_temperature,
                step,
                config.steps,
            ))?;
        }
        if step % config.sample_every == 0 {
            frames.push(Frame {
                step,
                t: step as f64 * config.dt,
                pos: system.positions().to_vec(),
                vel: system.velocities().to_vec(),
                e_pot: system.potential_energy(),
                e_kin: system.kinetic_energy(),
            });
        }
    }

    Ok(RunArtifacts {
        run: RunMetadata {
            n: config.n,
            rho: config.rho,
            box_size,
            dt: config.dt,
            temperature: config.temperature,
            eq_steps: config.eq_steps,
            steps: config.steps,
            sample_every: config.sample_every,
            seed: config.seed,
            integrator: "velocity-verlet".into(),
            ramp_to: config.ramp_to,
        },
        frames,
    })
}

pub fn production_target_temperature(start: f64, end: f64, step: usize, steps: usize) -> f64 {
    start + (end - start) * step as f64 / steps as f64
}
