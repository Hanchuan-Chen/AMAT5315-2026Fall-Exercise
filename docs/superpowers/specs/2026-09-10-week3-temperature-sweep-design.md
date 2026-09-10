# Week 3 Temperature Sweep Design

## Goal

Measure the magnetization and energy of `L=32,64` lattices across the fixed
temperature grid, retain every raw measurement sweep, plot magnetization and
susceptibility, and estimate the infinite-lattice critical temperature within 2%
of Onsager's value.

## Protocol and artifacts

The Metropolis `sweep` subcommand anneals each size from an all-up lattice through
the union of the coarse `1.5..3.5:0.1` grid and fine `2.0..2.6:0.05` grid. It
discards 2000 sweeps per temperature, then writes 5000 measurements outside and
100000 inside the critical window. `L=64` uses seed 42 and `L=32` seed 1042.

`artifacts/run.json` carries exactly the required run settings. The streamed
`artifacts/series.jsonl` rows carry `L,T,sweep,M,E` in protocol order, with signed
mean spin and energy per site formatted to six decimals. Generated artifacts are
ignored by Git.

`src/temperature_sweep.rs` owns protocol construction and streaming generation.
`src/analysis.rs` reads rows by `(L,T)`, computes mean absolute magnetization and
susceptibility, fits a quadratic through five points around each maximum, and
calculates `T_c = 2 T_peak(64) - T_peak(32)`. `src/plot.rs` draws PNG charts from
those summaries. The CLI exposes `sweep` and `plot`; protocol sizes, grid, sweep
counts, seeds, and output directory remain tunable for tests and experiments.

## Verification

Red tests fix the temperature grid, schemas, row order/count, six-decimal format,
seed reproducibility, susceptibility formula, quadratic vertex, and extrapolation.
A Makefile `reproduce` target runs snapshots, Metropolis sweep, plots, and summary.
The full default produces 2,740,000 rows, identical checksums on repeated runs,
ordered-phase means above 0.9, susceptibility peaks that drift downward from
`L=32` to `L=64`, and a critical estimate within 2% of 2.26919. The supplied
course checker independently recomputes these claims from the raw rows.
