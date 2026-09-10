//! Week 3 two-dimensional Ising simulator.

mod analysis;
pub mod cli;
mod lattice;
mod metropolis;
mod plot;
mod snapshots;
mod temperature_sweep;
mod wolff;

pub use analysis::{
    AnalysisSummary, ObservablePoint, blocked_error, critical_temperature,
    integrated_autocorrelation_time, load_analysis, naive_error, quadratic_peak, susceptibility,
};
pub use lattice::Lattice;
pub use metropolis::{AcceptanceTable, RelaxConfig, RelaxResult, SweepStats, relax, sweep};
pub use plot::{write_tau_comparison, write_tau_plot, write_thermodynamic_plots};
pub use snapshots::{SnapshotConfig, SnapshotSummary, write_snapshots};
pub use temperature_sweep::{
    TemperatureSweepConfig, TemperatureSweepSummary, course_temperature_grid,
    write_temperature_sweep, write_wolff_temperature_sweep,
};
pub use wolff::{WolffSweepStats, wolff_add_probability, wolff_cluster_flip, wolff_sweep};
