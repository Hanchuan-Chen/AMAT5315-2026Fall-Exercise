//! Two-dimensional Ising model sampler.
//!
//! The binary `ising` samples the model along one temperature ramp and writes
//! the artifacts fixed by `week3/ising.design.toml`. The library modules are
//! public so the integration tests can drive the physics directly.

pub mod artifacts;
pub mod cli;
pub mod lattice;
pub mod metropolis;
pub mod ramp;

pub use lattice::Lattice;
pub use ramp::{RunConfig, TemperatureResult, Update, temperature_grid};
