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

## Reproduce the measured transition

```bash
make reproduce
```

This writes the viewer ramp plus the raw two-size Metropolis run under
`artifacts/`, then generates `magnetization.png` and `susceptibility.png` and
prints the two fitted susceptibility peaks and the extrapolated critical
temperature. The raw files follow the course gate's fixed contracts:

- `artifacts/run.json`: sizes, grid, sweep counts, seed, sampling interval, and
  `algorithm="metropolis"`.
- `artifacts/series.jsonl`: one `L,T,sweep,M,E` row after every measurement
  sweep. The full default contains 2,740,000 rows and remains untracked.

The susceptibility is computed from the signed per-sweep magnetizations,

```text
chi(T) = L^2 * (<M^2> - <|M|>^2) / T,
```

and a five-point quadratic fit locates each finite-size peak. The reported
infinite-size estimate is `T_c = 2 T_peak(64) - T_peak(32)` and is compared with
Onsager's exact `2.26919`.

## Correlations and honest error bars

```bash
cargo run --release -- analyze artifacts --blocks 50
```

For every size and temperature this reports the mean absolute magnetization, the
naive independent-sample standard error, the standard error of 50 contiguous
block means, their ratio, and the integrated autocorrelation time. The latter is
the positive-prefix windowed estimate

```text
tau_int = 1/2 + sum_t rho(t), stopping by t >= 6 tau_int.
```

The command also writes `artifacts/tau.png` with a logarithmic vertical axis. In
the default run, `L=64,T=2.3` has `tau_int=939.80` and only about
`100000/(2*939.80) = 53` effective samples. Its 50-block error is 32.59 times the
naive error. At `T=3.5`, `tau_int=2.44` and the ratio is only 2.22. Thus blocks of
2000 sweeps at the transition span only about two autocorrelation times and have
not fully reached the error plateau; the reported blocked error there is still
optimistic.
