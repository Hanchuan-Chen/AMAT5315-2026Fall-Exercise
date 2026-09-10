//! Week 3 two-dimensional Ising simulator.

pub mod cli;
mod lattice;
mod metropolis;
mod snapshots;

pub use lattice::Lattice;
pub use metropolis::{AcceptanceTable, RelaxConfig, RelaxResult, SweepStats, relax, sweep};
pub use snapshots::{SnapshotConfig, SnapshotSummary, write_snapshots};
