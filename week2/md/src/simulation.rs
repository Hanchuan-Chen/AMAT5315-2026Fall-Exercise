use crate::physics::{energy, force};

pub struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>,
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
        let accelerations = accelerations(&positions);
        Self {
            positions,
            velocities,
            accelerations,
        }
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

    pub fn total_energy(&self) -> f64 {
        let kinetic = self
            .velocities
            .iter()
            .map(|velocity| 0.5 * (velocity[0].powi(2) + velocity[1].powi(2)))
            .sum::<f64>();
        let mut potential = 0.0;
        for i in 0..self.positions.len() {
            for j in (i + 1)..self.positions.len() {
                let dx = self.positions[i][0] - self.positions[j][0];
                let dy = self.positions[i][1] - self.positions[j][1];
                potential += energy(dx.hypot(dy));
            }
        }
        kinetic + potential
    }
}

fn accelerations(positions: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut result = vec![[0.0, 0.0]; positions.len()];
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let dx = positions[i][0] - positions[j][0];
            let dy = positions[i][1] - positions[j][1];
            let r = dx.hypot(dy);
            let scale = force(r) / r;
            let pair = [scale * dx, scale * dy];
            for axis in 0..2 {
                result[i][axis] += pair[axis];
                result[j][axis] -= pair[axis];
            }
        }
    }
    result
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
        system.accelerations = accelerations(&system.positions);
    }
}

pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) {
        require_positive_dt(dt);
        for atom in 0..system.positions.len() {
            for axis in 0..2 {
                system.velocities[atom][axis] += 0.5 * dt * system.accelerations[atom][axis];
                system.positions[atom][axis] += dt * system.velocities[atom][axis];
            }
        }
        let new_accelerations = accelerations(&system.positions);
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
