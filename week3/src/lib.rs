//! Week 3 two-dimensional Ising simulator.

pub mod cli;
mod lattice;
mod metropolis;
mod snapshots;
mod temperature_sweep;

pub use lattice::Lattice;
pub use metropolis::{AcceptanceTable, RelaxConfig, RelaxResult, SweepStats, relax, sweep};
pub use snapshots::{SnapshotConfig, SnapshotSummary, write_snapshots};
pub use temperature_sweep::{
    TemperatureSweepConfig, TemperatureSweepSummary, course_temperature_grid,
    write_temperature_sweep,
};
