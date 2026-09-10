# Week 3 Part 1 Metropolis Design

## Goal

Build the first working slice of the `week3/ising` Rust program: a reproducible
random-site Metropolis sampler for the periodic two-dimensional Ising model and a
`relax` command that reports a measured mean absolute magnetization, acceptance
fraction, and final lattice.

## Fixed physics

- The lattice is `L x L`, with spins in `{-1, +1}`, coupling `J = 1`, no field,
  and periodic boundaries.
- The initial lattice is all `+1`.
- One proposal chooses one site uniformly at random. A proposed flip has
  `delta_E = 2 s_i sum_neighbours s_j` and is accepted with probability
  `min(1, exp(-delta_E / T))`.
- One sweep is exactly `L * L` proposals, accepted or rejected.
- The five possible acceptance probabilities are precomputed once per
  temperature.
- Every random choice comes from one `StdRng` constructed with
  `StdRng::seed_from_u64(seed)` so equal commands and seeds reproduce byte-for-byte
  output.

## Structure

- `week3/src/lattice.rs` owns row-major spin storage, periodic neighbour lookup,
  total energy, signed magnetization, proposed energy change, flips, and character
  rendering.
- `week3/src/metropolis.rs` owns the acceptance table, one proposal, one sweep,
  and the equilibration/measurement loop.
- `week3/src/cli.rs` owns command-line parsing, validation, execution, and stable
  text formatting.
- `week3/src/lib.rs` exposes testable interfaces; `week3/src/main.rs` is a thin
  executable entry point.

This separation keeps the physics independently testable and leaves the same
lattice and sampler available to later `snapshots`, `sweep`, and `analyze`
subcommands.

## Command contract

The command

```text
ising relax --l 64 --t 1.8 --sweeps 2000 --measure 2000 --seed 2026
```

equilibrates for `sweeps`, measures `abs(m)` after each of `measure` further
sweeps, and prints a first line containing `L`, `T`, both sweep counts,
`mean_abs_m`, and the acceptance fraction over all equilibration and measurement
proposals. It then prints exactly `L` rows of `L` spin characters.

Invalid zero lattice size, non-positive/non-finite temperature, or zero
measurement count is rejected with a useful message and non-zero exit status.

## Tests and evidence

Tests are written and observed failing before production implementation.

- On many seeded random lattices and every site, the local proposed `delta_E`
  must equal the difference between total energies recomputed before and after a
  flip. The collected cases must contain all five values `-8, -4, 0, 4, 8`.
- Equal seeds must reproduce an entire relaxation result and formatted output;
  a different seed must differ.
- Periodic neighbours, energy, magnetization, render dimensions, acceptance
  table values, sweep proposal count, and CLI validation receive focused tests.
- Release tests must pass.
- The required `L=64` runs at `T=1.8` and `T=3.0` must land in the learning
  sheet's magnetization and acceptance windows.

## Scope

This Part does not yet write JSON artifacts, plot data, publish a viewer, analyze
autocorrelation, or implement Wolff updates. Those are separate Part designs that
reuse this tested core.
