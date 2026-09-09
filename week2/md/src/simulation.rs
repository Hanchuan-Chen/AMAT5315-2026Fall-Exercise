use crate::physics::{energy, force};

#[derive(Clone, Copy, Debug)]
enum InteractionModel {
    PlainOpen,
    PeriodicShifted { box_size: [f64; 2], cutoff: f64 },
}

#[derive(Debug, thiserror::Error)]
pub enum SystemError {
    #[error("system must contain at least one atom")]
    Empty,
    #[error("positions and velocities must have equal lengths")]
    LengthMismatch,
    #[error("all positions and velocities must be finite")]
    NonFiniteState,
    #[error("box sides must be finite and at least twice cutoff")]
    InvalidBox,
    #[error("cutoff must be finite and positive")]
    InvalidCutoff,
    #[error("temperature must be finite and positive")]
    InvalidTemperature,
    #[error("at least two atoms with nonzero kinetic energy are required")]
    NoThermalDegreesOfFreedom,
}

pub struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>,
    interaction: InteractionModel,
}

impl System {
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self {
        assert!(
            !positions.is_empty(),
            "system must contain at least one atom"
        );
        assert_eq!(
            positions.len(),
            velocities.len(),
            "positions and velocities must have equal lengths"
        );
        let interaction = InteractionModel::PlainOpen;
        let accelerations = accelerations(&positions, interaction);
        Self {
            positions,
            velocities,
            accelerations,
            interaction,
        }
    }

    pub fn new_periodic(
        mut positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        box_size: [f64; 2],
        cutoff: f64,
    ) -> Result<Self, SystemError> {
        if positions.is_empty() {
            return Err(SystemError::Empty);
        }
        if positions.len() != velocities.len() {
            return Err(SystemError::LengthMismatch);
        }
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(SystemError::InvalidCutoff);
        }
        if box_size
            .iter()
            .any(|side| !side.is_finite() || *side < 2.0 * cutoff)
        {
            return Err(SystemError::InvalidBox);
        }
        if positions
            .iter()
            .chain(&velocities)
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(SystemError::NonFiniteState);
        }

        for position in &mut positions {
            for axis in 0..2 {
                position[axis] = position[axis].rem_euclid(box_size[axis]);
            }
        }
        let interaction = InteractionModel::PeriodicShifted { box_size, cutoff };
        let accelerations = accelerations(&positions, interaction);
        Ok(Self {
            positions,
            velocities,
            accelerations,
            interaction,
        })
    }

    pub fn positions(&self) -> &[[f64; 2]] {
        &self.positions
    }

    pub fn velocities(&self) -> &[[f64; 2]] {
        &self.velocities
    }

    pub fn accelerations(&self) -> &[[f64; 2]] {
        &self.accelerations
    }

    pub fn kinetic_energy(&self) -> f64 {
        self.velocities
            .iter()
            .map(|velocity| 0.5 * (velocity[0].powi(2) + velocity[1].powi(2)))
            .sum()
    }

    pub fn potential_energy(&self) -> f64 {
        match self.interaction {
            InteractionModel::PlainOpen => {
                let mut potential = 0.0;
                for i in 0..self.positions.len() {
                    for j in (i + 1)..self.positions.len() {
                        let dx = self.positions[i][0] - self.positions[j][0];
                        let dy = self.positions[i][1] - self.positions[j][1];
                        potential += energy(dx.hypot(dy));
                    }
                }
                potential
            }
            InteractionModel::PeriodicShifted { box_size, cutoff } => {
                let shift = energy(cutoff);
                let cutoff_squared = cutoff * cutoff;
                let mut potential = 0.0;
                for i in 0..self.positions.len() {
                    for j in (i + 1)..self.positions.len() {
                        let dx =
                            minimum_image(self.positions[i][0] - self.positions[j][0], box_size[0]);
                        let dy =
                            minimum_image(self.positions[i][1] - self.positions[j][1], box_size[1]);
                        let distance_squared = dx * dx + dy * dy;
                        if distance_squared < cutoff_squared {
                            potential += energy(distance_squared.sqrt()) - shift;
                        }
                    }
                }
                potential
            }
        }
    }

    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    pub fn thermodynamic_temperature(&self) -> f64 {
        assert!(
            self.positions.len() >= 2,
            "temperature requires at least two atoms"
        );
        2.0 * self.kinetic_energy() / (2 * self.positions.len() - 2) as f64
    }

    pub fn remove_center_of_mass_velocity(&mut self) {
        let count = self.velocities.len() as f64;
        let mut mean = [0.0, 0.0];
        for velocity in &self.velocities {
            for axis in 0..2 {
                mean[axis] += velocity[axis] / count;
            }
        }
        for velocity in &mut self.velocities {
            for axis in 0..2 {
                velocity[axis] -= mean[axis];
            }
        }
    }

    pub fn rescale_temperature(&mut self, target: f64) -> Result<(), SystemError> {
        if !target.is_finite() || target <= 0.0 {
            return Err(SystemError::InvalidTemperature);
        }
        if self.positions.len() < 2 {
            return Err(SystemError::NoThermalDegreesOfFreedom);
        }
        let current = self.thermodynamic_temperature();
        if !current.is_finite() || current <= 0.0 {
            return Err(SystemError::NoThermalDegreesOfFreedom);
        }
        let scale = (target / current).sqrt();
        for velocity in &mut self.velocities {
            for component in velocity {
                *component *= scale;
            }
        }
        Ok(())
    }

    fn wrap_positions(&mut self) {
        if let InteractionModel::PeriodicShifted { box_size, .. } = self.interaction {
            for position in &mut self.positions {
                for axis in 0..2 {
                    position[axis] = position[axis].rem_euclid(box_size[axis]);
                }
            }
        }
    }
}

fn minimum_image(delta: f64, side: f64) -> f64 {
    delta - side * (delta / side).round()
}

fn accelerations(positions: &[[f64; 2]], interaction: InteractionModel) -> Vec<[f64; 2]> {
    match interaction {
        InteractionModel::PlainOpen => {
            let mut result = vec![[0.0, 0.0]; positions.len()];
            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    let dx = positions[i][0] - positions[j][0];
                    let dy = positions[i][1] - positions[j][1];
                    add_pair_acceleration(&mut result, i, j, dx, dy);
                }
            }
            result
        }
        InteractionModel::PeriodicShifted { box_size, cutoff } => {
            let mut result = vec![[0.0, 0.0]; positions.len()];
            let cutoff_squared = cutoff * cutoff;
            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    let dx = minimum_image(positions[i][0] - positions[j][0], box_size[0]);
                    let dy = minimum_image(positions[i][1] - positions[j][1], box_size[1]);
                    if dx * dx + dy * dy < cutoff_squared {
                        add_pair_acceleration(&mut result, i, j, dx, dy);
                    }
                }
            }
            result
        }
    }
}

fn add_pair_acceleration(accelerations: &mut [[f64; 2]], i: usize, j: usize, dx: f64, dy: f64) {
    let distance = dx.hypot(dy);
    let scale = force(distance) / distance;
    let pair = [scale * dx, scale * dy];
    for axis in 0..2 {
        accelerations[i][axis] += pair[axis];
        accelerations[j][axis] -= pair[axis];
    }
}

fn refresh_accelerations(system: &mut System) {
    system.accelerations = accelerations(&system.positions, system.interaction);
}

fn drift_and_wrap(system: &mut System, dt: f64) {
    for atom in 0..system.positions.len() {
        for axis in 0..2 {
            system.positions[atom][axis] += dt * system.velocities[atom][axis];
        }
    }
    system.wrap_positions();
}

fn require_positive_dt(dt: f64) {
    assert!(dt.is_finite() && dt > 0.0, "dt must be finite and positive");
}

pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

pub struct Euler;

impl Integrator for Euler {
    fn step(&self, system: &mut System, dt: f64) {
        require_positive_dt(dt);
        for atom in 0..system.positions.len() {
            for axis in 0..2 {
                system.positions[atom][axis] += dt * system.velocities[atom][axis];
                system.velocities[atom][axis] += dt * system.accelerations[atom][axis];
            }
        }
        system.wrap_positions();
        refresh_accelerations(system);
    }
}

pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) {
        require_positive_dt(dt);
        for atom in 0..system.positions.len() {
            for axis in 0..2 {
                system.velocities[atom][axis] += 0.5 * dt * system.accelerations[atom][axis];
            }
        }
        drift_and_wrap(system, dt);
        let new_accelerations = accelerations(&system.positions, system.interaction);
        for (velocity, acceleration) in system.velocities.iter_mut().zip(&new_accelerations) {
            for axis in 0..2 {
                velocity[axis] += 0.5 * dt * acceleration[axis];
            }
        }
        system.accelerations = new_accelerations;
    }
}

pub fn advance(method: &impl Integrator, system: &mut System, dt: f64) {
    method.step(system, dt);
}

pub struct EnergyTrace {
    times: Vec<f64>,
    relative_errors: Vec<f64>,
}

impl EnergyTrace {
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    pub fn relative_errors(&self) -> &[f64] {
        &self.relative_errors
    }
}

pub fn run_dimer(method: &impl Integrator, steps: usize, dt: f64) -> EnergyTrace {
    require_positive_dt(dt);
    let mut system = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0, 0.0], [0.0, 0.0]]);
    let initial_energy = system.total_energy();
    let mut times = Vec::with_capacity(steps);
    let mut relative_errors = Vec::with_capacity(steps);

    for step in 1..=steps {
        advance(method, &mut system, dt);
        times.push(step as f64 * dt);
        relative_errors.push((system.total_energy() - initial_energy) / initial_energy.abs());
    }

    EnergyTrace {
        times,
        relative_errors,
    }
}
