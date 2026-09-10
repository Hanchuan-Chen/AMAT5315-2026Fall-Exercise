//! Week 3 two-dimensional Ising simulator.

mod lattice;
mod metropolis;

pub use lattice::Lattice;
pub use metropolis::{AcceptanceTable, RelaxConfig, RelaxResult, SweepStats, relax, sweep};

