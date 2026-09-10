# Week 3 Snapshot Ramp Design

## Goal

Extend the tested Ising core with a reproducible ascending temperature ramp that
writes the supplied viewer's JSON Lines frame contract and publish that recording
under `docs/week3/` without disturbing the Week 2 page.

## Design

`snapshots` keeps one all-up lattice and one seeded RNG for the entire run. It
visits inclusive temperatures from 1.5 to 3.5 in increments of 0.05, equilibrates
for 2000 sweeps at each temperature, then records after every twentieth sweep of
200 measurement sweeps. Each temperature starts from the preceding final lattice.

`src/artifacts.rs` owns streaming JSONL output and directory creation.
`src/snapshots.rs` owns the protocol. The CLI exposes tunable lattice size,
temperature endpoints and step, equilibration/recording sweeps, frame interval,
seed, and output path, with the course protocol as defaults.

Each frame has exactly `L`, `T`, `sweep`, `m`, and `spins`. The cumulative sweep
counts every equilibration and recording sweep across the ramp. `spins` contains
exactly `L * L` row-major ASCII `1`/`0` characters.

## Verification

Tests first assert exact field names, spin encoding, inclusive-grid construction,
warm-start continuity, frame cadence, and seeded reproducibility on a small run.
The default run must write exactly 410 lines. The supplied viewer and a copy of
the default recording are tracked at `docs/week3/index.html` and
`docs/week3/spins.jsonl`; the README carries their public Pages URL.
