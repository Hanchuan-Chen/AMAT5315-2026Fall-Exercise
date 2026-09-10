//! Week 3 two-dimensional Ising simulator.

pub mod cli;
mod lattice;
mod metropolis;

pub use lattice::Lattice;
pub use metropolis::{AcceptanceTable, RelaxConfig, RelaxResult, SweepStats, relax, sweep};
