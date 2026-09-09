use std::f64::consts::PI;

use crate::artifacts::{Frame, RunArtifacts, RunMetadata, validate_artifacts};
use crate::physics;

const CUTOFF: f64 = 2.5;
const CONSISTENCY_TOLERANCE: f64 = 1.0e-6;
const SECULAR_TOLERANCE: f64 = 2.0e-3;
const TEMPERATURE_TOLERANCE: f64 = 0.05;
const CHI2_DOF_TOLERANCE: f64 = 2.0;
const RAYLEIGH_BINS: usize = 24;

#[derive(Debug, Clone)]
pub struct CheckReport {
    pub max_energy_consistency: f64,
    pub secular_drift: f64,
    pub max_energy_oscillation: f64,
    pub temperature: f64,
    pub chi2_per_dof: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum CheckError {
    #[error("contract: {0}")]
    Contract(String),
    #[error("physics: {0}")]
    Physics(String),
}

#[derive(Debug, Clone)]
pub struct RadialDistribution {
    pub r: Vec<f64>,
    pub g: Vec<f64>,
}

fn minimum_image(delta: f64, side: f64) -> f64 {
    delta - side * (delta / side).round()
}

fn recompute_potential(positions: &[[f64; 2]], box_size: [f64; 2]) -> Result<f64, CheckError> {
    let cutoff_squared = CUTOFF * CUTOFF;
    let shift = physics::energy(CUTOFF);
    let mut potential = 0.0;
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let dx = minimum_image(positions[i][0] - positions[j][0], box_size[0]);
            let dy = minimum_image(positions[i][1] - positions[j][1], box_size[1]);
            let distance_squared = dx * dx + dy * dy;
            if distance_squared == 0.0 {
                return Err(CheckError::Physics(
                    "overlapping particles give non-finite energy".into(),
                ));
            }
            if distance_squared < cutoff_squared {
                potential += physics::energy(distance_squared.sqrt()) - shift;
            }
        }
    }
    if !potential.is_finite() {
        return Err(CheckError::Physics(
            "recomputed potential energy is non-finite".into(),
        ));
    }
    Ok(potential)
}

fn recompute_kinetic(velocities: &[[f64; 2]]) -> Result<f64, CheckError> {
    let kinetic = velocities
        .iter()
        .map(|velocity| 0.5 * (velocity[0].powi(2) + velocity[1].powi(2)))
        .sum::<f64>();
    if !kinetic.is_finite() {
        return Err(CheckError::Physics(
            "recomputed kinetic energy is non-finite".into(),
        ));
    }
    Ok(kinetic)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn rayleigh_cdf(speed: f64, temperature: f64) -> Result<f64, CheckError> {
    if !speed.is_finite() || speed < 0.0 {
        return Err(CheckError::Contract(
            "speed must be finite and non-negative".into(),
        ));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(CheckError::Contract(
            "temperature must be finite and positive".into(),
        ));
    }
    Ok(-(-speed * speed / (2.0 * temperature)).exp_m1())
}

pub fn rayleigh_quantile(probability: f64, temperature: f64) -> Result<f64, CheckError> {
    if !probability.is_finite() || !(0.0..1.0).contains(&probability) {
        return Err(CheckError::Contract(
            "probability must be finite and in [0, 1)".into(),
        ));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(CheckError::Contract(
            "temperature must be finite and positive".into(),
        ));
    }
    Ok((-2.0 * temperature * (-probability).ln_1p()).sqrt())
}

fn rayleigh_chi2(speeds: &[f64], temperature: f64) -> Result<f64, CheckError> {
    let mut edges = (0..RAYLEIGH_BINS)
        .map(|index| rayleigh_quantile(index as f64 / RAYLEIGH_BINS as f64, temperature))
        .collect::<Result<Vec<_>, _>>()?;
    edges.push(f64::INFINITY);
    let mut observed = [0_usize; RAYLEIGH_BINS];
    for speed in speeds {
        let mut bin = 0;
        while bin + 1 < RAYLEIGH_BINS && *speed >= edges[bin + 1] {
            bin += 1;
        }
        observed[bin] += 1;
    }
    let expected = speeds.len() as f64 / RAYLEIGH_BINS as f64;
    let chi2 = observed
        .iter()
        .map(|count| (*count as f64 - expected).powi(2) / expected)
        .sum::<f64>();
    Ok(chi2 / (RAYLEIGH_BINS - 2) as f64)
}

fn require_locked_contract(artifacts: &RunArtifacts) -> Result<(), CheckError> {
    let run = &artifacts.run;
    if run.n != 100 {
        return Err(CheckError::Physics(format!("n = {}; required 100", run.n)));
    }
    if (run.dt - 0.01).abs() > 1.0e-9 {
        return Err(CheckError::Physics(format!(
            "dt = {}; required 0.01",
            run.dt
        )));
    }
    if run.box_size.iter().any(|side| *side < 2.0 * CUTOFF) {
        return Err(CheckError::Physics(
            "each box side must be at least 2*cutoff".into(),
        ));
    }
    let measured_density = run.n as f64 / (run.box_size[0] * run.box_size[1]);
    if (run.rho - 0.8).abs() > 0.008 || (measured_density - 0.8).abs() > 0.008 {
        return Err(CheckError::Physics(format!(
            "rho = {}, measured density = {measured_density}; required 0.8 +/- 0.008",
            run.rho
        )));
    }
    if (run.temperature - 0.5).abs() > 1.0e-9 {
        return Err(CheckError::Physics(format!(
            "declared temperature = {}; required 0.5",
            run.temperature
        )));
    }
    if run.eq_steps < 2000 || run.steps < 10_000 {
        return Err(CheckError::Physics(format!(
            "eq_steps/steps = {}/{}; required at least 2000/10000",
            run.eq_steps, run.steps
        )));
    }
    if run.sample_every > 100 {
        return Err(CheckError::Physics(format!(
            "sample_every = {}; required <= 100",
            run.sample_every
        )));
    }
    if run.integrator != "velocity-verlet" {
        return Err(CheckError::Physics(format!(
            "integrator = {:?}; required \"velocity-verlet\"",
            run.integrator
        )));
    }
    if artifacts.frames.len() < 100 {
        return Err(CheckError::Contract(format!(
            "{} frames found; required at least 100",
            artifacts.frames.len()
        )));
    }
    Ok(())
}

pub fn check_run(artifacts: &RunArtifacts) -> Result<CheckReport, CheckError> {
    validate_artifacts(artifacts).map_err(|error| CheckError::Contract(error.to_string()))?;
    require_locked_contract(artifacts)?;
    let frame_count = artifacts.frames.len();
    let stride = usize::max(1, frame_count / 25);
    let mut max_energy_consistency: f64 = 0.0;
    for index in (0..frame_count).step_by(stride).take(25) {
        let frame = &artifacts.frames[index];
        let potential = recompute_potential(&frame.pos, artifacts.run.box_size)?;
        let kinetic = recompute_kinetic(&frame.vel)?;
        let potential_relative = (potential - frame.e_pot).abs() / frame.e_pot.abs().max(1.0);
        let kinetic_relative = (kinetic - frame.e_kin).abs() / frame.e_kin.abs().max(1.0);
        max_energy_consistency =
            max_energy_consistency.max(potential_relative.max(kinetic_relative));
    }
    if max_energy_consistency >= CONSISTENCY_TOLERANCE {
        return Err(CheckError::Physics(format!(
            "energy consistency {max_energy_consistency:.3e} >= {CONSISTENCY_TOLERANCE:.0e}"
        )));
    }

    let totals = artifacts
        .frames
        .iter()
        .map(|frame| {
            Ok(recompute_potential(&frame.pos, artifacts.run.box_size)?
                + recompute_kinetic(&frame.vel)?)
        })
        .collect::<Result<Vec<_>, CheckError>>()?;
    let initial = totals[0];
    if initial == 0.0 {
        return Err(CheckError::Physics(
            "initial total energy is exactly zero".into(),
        ));
    }
    let end_count = usize::max(1, frame_count / 10);
    let secular_drift = (mean(&totals[frame_count - end_count..]) - mean(&totals[..end_count]))
        .abs()
        / initial.abs();
    let max_energy_oscillation = totals
        .iter()
        .map(|energy| (energy - initial).abs() / initial.abs())
        .fold(0.0, f64::max);
    if secular_drift >= SECULAR_TOLERANCE {
        return Err(CheckError::Physics(format!(
            "secular energy drift {secular_drift:.3e} >= {SECULAR_TOLERANCE:.0e}"
        )));
    }

    let speeds = artifacts
        .frames
        .iter()
        .flat_map(|frame| {
            frame
                .vel
                .iter()
                .map(|velocity| velocity[0].hypot(velocity[1]))
        })
        .collect::<Vec<_>>();
    let temperature =
        speeds.iter().map(|speed| speed * speed).sum::<f64>() / (2.0 * speeds.len() as f64);
    if (temperature - 0.5).abs() >= TEMPERATURE_TOLERANCE {
        return Err(CheckError::Physics(format!(
            "measured temperature {temperature:.4} is outside 0.5 +/- {TEMPERATURE_TOLERANCE}"
        )));
    }
    let chi2_per_dof = rayleigh_chi2(&speeds, temperature)?;
    if chi2_per_dof >= CHI2_DOF_TOLERANCE {
        return Err(CheckError::Physics(format!(
            "Rayleigh chi2/dof {chi2_per_dof:.3} >= {CHI2_DOF_TOLERANCE}"
        )));
    }

    Ok(CheckReport {
        max_energy_consistency,
        secular_drift,
        max_energy_oscillation,
        temperature,
        chi2_per_dof,
    })
}

pub fn radial_distribution(
    frames: &[Frame],
    run: &RunMetadata,
    bins: usize,
) -> Result<RadialDistribution, CheckError> {
    if frames.is_empty() || bins == 0 {
        return Err(CheckError::Contract(
            "radial distribution requires frames and at least one bin".into(),
        ));
    }
    let radial_max = 0.5 * run.box_size[0].min(run.box_size[1]);
    if !radial_max.is_finite() || radial_max <= 0.0 || !run.rho.is_finite() || run.rho <= 0.0 {
        return Err(CheckError::Contract(
            "radial distribution requires a finite positive box and density".into(),
        ));
    }
    let width = radial_max / bins as f64;
    let mut counts = vec![0.0; bins];
    for (frame_index, frame) in frames.iter().enumerate() {
        if frame.pos.len() != run.n {
            return Err(CheckError::Contract(format!(
                "frame {frame_index} position count differs from n"
            )));
        }
        for i in 0..frame.pos.len() {
            for j in (i + 1)..frame.pos.len() {
                let dx = minimum_image(frame.pos[i][0] - frame.pos[j][0], run.box_size[0]);
                let dy = minimum_image(frame.pos[i][1] - frame.pos[j][1], run.box_size[1]);
                let distance = dx.hypot(dy);
                if distance < radial_max {
                    let bin = (distance / width).floor() as usize;
                    counts[bin] += 2.0;
                }
            }
        }
    }

    let mut radii = Vec::with_capacity(bins);
    let mut distribution = Vec::with_capacity(bins);
    for (index, count) in counts.into_iter().enumerate() {
        let inner = index as f64 * width;
        let outer = (index + 1) as f64 * width;
        let normalization =
            frames.len() as f64 * run.n as f64 * run.rho * PI * (outer * outer - inner * inner);
        radii.push(0.5 * (inner + outer));
        distribution.push(count / normalization);
    }
    Ok(RadialDistribution {
        r: radii,
        g: distribution,
    })
}
