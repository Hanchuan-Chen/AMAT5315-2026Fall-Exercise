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

## Wolff clusters

```bash
cargo run --release -- sweep --wolff
cargo run --release -- analyze artifacts-wolff --blocks 50
make compare
```

The Wolff update grows like-spin clusters with bond probability
`1-exp(-2/T)` and always flips the completed cluster. One comparable sweep is
enough whole cluster flips to touch at least `L^2` spins. The default cluster run
covers `T=2.0..2.6:0.05`, writes the same raw contract under
`artifacts-wolff/` with `algorithm="wolff"`, and produces 2,600,000 rows.

At `L=64,T=2.3`, Wolff gives `mean_abs_m=0.524456`, error ratio `1.45`, and
`tau_int=0.86`, versus Metropolis `0.443748`, `32.59`, and `939.80`. The same
number of stored sweeps is therefore worth roughly a thousand times more under
clusters. The Metropolis result differs because its 100000 sweeps contain only
about 53 effective samples and its 2000-sweep blocks are only about two
autocorrelation times long; the blocked error has not reached its plateau.

Away from the peak, the two methods agree much more closely: at `T=2.0` their
means are `0.911601` and `0.911492`; at `T=2.6` they are `0.075293` and
`0.078100`, with the slow Metropolis series retaining the larger uncertainty.
The Wolff susceptibility peaks give `T_c=2.295180`, a `+1.15%` deviation from
Onsager and within the required 2%. `tau-compare.png` shows the Metropolis peak
near 940 sweeps while Wolff stays below one sweep throughout the window.
