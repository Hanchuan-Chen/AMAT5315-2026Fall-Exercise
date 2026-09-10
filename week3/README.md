# Week 3: Monte Carlo simulation

This crate simulates the periodic two-dimensional Ising model with reproducible
Monte Carlo updates.

## Relax one lattice

```bash
cargo run --release -- relax --l 64 --t 1.8 --sweeps 2000 --measure 2000 --seed 2026
```

The first line reports the time-averaged absolute magnetization and proposal
acceptance fraction. The following rows draw the final lattice (`█` is spin +1,
`·` is spin -1).

## Test

```bash
cargo test --release
```

The tests independently compare each proposed local energy change with total
energies recomputed from scratch on random lattices and verify seeded
reproducibility.

## Recorded temperature ramp

```bash
cargo run --release -- snapshots
```

The default command writes 410 JSON Lines frames to `artifacts/spins.jsonl` for
the ascending `T=1.5` to `3.5` protocol.

## Pages

The published viewer is available at:

https://hanchuan-chen.github.io/AMAT5315-2026Fall-Exercise/week3/

Append `?T=1.8`, `?T=2.3`, or `?T=3.0` to inspect the ordered, critical, and
disordered regimes.
