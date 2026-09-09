use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunMetadata {
    pub n: usize,
    pub rho: f64,
    #[serde(rename = "box")]
    pub box_size: [f64; 2],
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    #[serde(rename = "E_pot")]
    pub e_pot: f64,
    #[serde(rename = "E_kin")]
    pub e_kin: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunArtifacts {
    pub run: RunMetadata,
    pub frames: Vec<Frame>,
}

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("JSON error at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("trajectory contains no frames")]
    EmptyTrajectory,
    #[error("contract: {0}")]
    Contract(String),
}

fn io_error(path: &Path, source: std::io::Error) -> ArtifactError {
    ArtifactError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn json_error(path: &Path, source: serde_json::Error) -> ArtifactError {
    ArtifactError::Json {
        path: path.to_path_buf(),
        source,
    }
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

pub fn validate_artifacts(artifacts: &RunArtifacts) -> Result<(), ArtifactError> {
    let run = &artifacts.run;
    if run.n == 0 {
        return Err(ArtifactError::Contract(
            "run.json n must be a positive integer".into(),
        ));
    }
    if !finite_positive(run.rho) || !finite_positive(run.dt) || !finite_positive(run.temperature) {
        return Err(ArtifactError::Contract(
            "run.json rho, dt, and temperature must be finite and positive".into(),
        ));
    }
    if run.box_size.iter().any(|side| !finite_positive(*side)) {
        return Err(ArtifactError::Contract(
            "run.json box must contain two finite positive sides".into(),
        ));
    }
    if run.steps == 0 || run.sample_every == 0 || run.sample_every > run.steps {
        return Err(ArtifactError::Contract(
            "run.json steps and sample_every must define at least one sample".into(),
        ));
    }
    if run.integrator.trim().is_empty() {
        return Err(ArtifactError::Contract(
            "run.json integrator must be a non-empty string".into(),
        ));
    }
    if artifacts.frames.is_empty() {
        return Err(ArtifactError::EmptyTrajectory);
    }
    let expected_frames = run.steps / run.sample_every;
    if artifacts.frames.len() != expected_frames {
        return Err(ArtifactError::Contract(format!(
            "trajectory has {} frames; expected {expected_frames}",
            artifacts.frames.len()
        )));
    }

    for (index, frame) in artifacts.frames.iter().enumerate() {
        if frame.pos.len() != run.n || frame.vel.len() != run.n {
            return Err(ArtifactError::Contract(format!(
                "frame {index} pos/vel lengths must equal n = {}",
                run.n
            )));
        }
        let expected_step = (index + 1) * run.sample_every;
        if frame.step != expected_step {
            return Err(ArtifactError::Contract(format!(
                "frame {index} step is {}; expected {expected_step}",
                frame.step
            )));
        }
        let expected_time = expected_step as f64 * run.dt;
        if !frame.t.is_finite() || (frame.t - expected_time).abs() > 1.0e-9 {
            return Err(ArtifactError::Contract(format!(
                "frame {index} time must equal step*dt = {expected_time}"
            )));
        }
        if !frame.e_pot.is_finite() || !frame.e_kin.is_finite() || frame.e_kin < 0.0 {
            return Err(ArtifactError::Contract(format!(
                "frame {index} energies must be finite and E_kin non-negative"
            )));
        }

        for (particle, position) in frame.pos.iter().enumerate() {
            if position.iter().any(|value| !value.is_finite()) {
                return Err(ArtifactError::Contract(format!(
                    "frame {index} pos[{particle}] must be finite"
                )));
            }
            if !(0.0 <= position[0]
                && position[0] < run.box_size[0]
                && 0.0 <= position[1]
                && position[1] < run.box_size[1])
            {
                return Err(ArtifactError::Contract(format!(
                    "frame {index} pos[{particle}] is not wrapped into the box"
                )));
            }
        }
        for (particle, velocity) in frame.vel.iter().enumerate() {
            if velocity.iter().any(|value| !value.is_finite()) {
                return Err(ArtifactError::Contract(format!(
                    "frame {index} vel[{particle}] must be finite"
                )));
            }
        }
    }
    Ok(())
}

pub fn write_artifacts(directory: &Path, artifacts: &RunArtifacts) -> Result<(), ArtifactError> {
    validate_artifacts(artifacts)?;
    fs::create_dir_all(directory).map_err(|source| io_error(directory, source))?;
    let run_path = directory.join("run.json");
    let trajectory_path = directory.join("traj.jsonl");
    let mut run_temp =
        NamedTempFile::new_in(directory).map_err(|source| io_error(&run_path, source))?;
    let mut trajectory_temp =
        NamedTempFile::new_in(directory).map_err(|source| io_error(&trajectory_path, source))?;

    serde_json::to_writer_pretty(&mut run_temp, &artifacts.run)
        .map_err(|source| json_error(&run_path, source))?;
    writeln!(run_temp).map_err(|source| io_error(&run_path, source))?;
    for frame in &artifacts.frames {
        serde_json::to_writer(&mut trajectory_temp, frame)
            .map_err(|source| json_error(&trajectory_path, source))?;
        writeln!(trajectory_temp).map_err(|source| io_error(&trajectory_path, source))?;
    }
    run_temp
        .flush()
        .map_err(|source| io_error(&run_path, source))?;
    trajectory_temp
        .flush()
        .map_err(|source| io_error(&trajectory_path, source))?;
    run_temp
        .as_file()
        .sync_all()
        .map_err(|source| io_error(&run_path, source))?;
    trajectory_temp
        .as_file()
        .sync_all()
        .map_err(|source| io_error(&trajectory_path, source))?;

    trajectory_temp
        .persist(&trajectory_path)
        .map_err(|error| io_error(&trajectory_path, error.error))?;
    run_temp
        .persist(&run_path)
        .map_err(|error| io_error(&run_path, error.error))?;
    Ok(())
}

pub fn read_artifacts(directory: &Path) -> Result<RunArtifacts, ArtifactError> {
    let run_path = directory.join("run.json");
    let trajectory_path = directory.join("traj.jsonl");
    let run_file = File::open(&run_path).map_err(|source| io_error(&run_path, source))?;
    let run = serde_json::from_reader(run_file).map_err(|source| json_error(&run_path, source))?;
    let trajectory_file =
        File::open(&trajectory_path).map_err(|source| io_error(&trajectory_path, source))?;
    let mut frames = Vec::new();

    for line in BufReader::new(trajectory_file).lines() {
        let line = line.map_err(|source| io_error(&trajectory_path, source))?;
        if line.trim().is_empty() {
            continue;
        }
        let frame =
            serde_json::from_str(&line).map_err(|source| json_error(&trajectory_path, source))?;
        frames.push(frame);
    }

    let artifacts = RunArtifacts { run, frames };
    validate_artifacts(&artifacts)?;
    Ok(artifacts)
}
