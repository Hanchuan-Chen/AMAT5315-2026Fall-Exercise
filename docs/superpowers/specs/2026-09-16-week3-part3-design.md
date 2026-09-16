# Week 3 Part 3 Uncertainty Analysis Design

## Purpose

Phase C finishes Part 3 of the week-3 sheet, "How much to trust Part 2". Phase
B produced the susceptibility curve, the two five-point peak fits and the
extrapolated `T_c` without a single error bar, on the sheet's own instruction.
Part 3 studies the rows those runs already wrote and asks how far the Part 2
numbers can be trusted:

- `week3/scripts/errors.py`, which turns the raw per-sweep `|M|` series into
  `week3/evidence/errors.txt`: one row per size and temperature with the mean,
  the naive standard error, the 50-block standard error, their ratio and the
  integrated autocorrelation time of `|M|`;
- `week3/scripts/chi_bootstrap.py`, a block bootstrap of the two susceptibility
  peaks and of `T_c = 2 T_peak(64) - T_peak(32)` at block lengths 2000, 4000 and
  8000 sweeps, drawn into `week3/evidence/chi-bootstrap.png`;
- three diagnostic charts: `trace.png` (the raw `|m|` series at two
  temperatures), `acf-binning.png` (the autocorrelation and the error against
  block length) and `tau.png` (`tau_int` against temperature for both sizes);
- the two extension numbers the sheet asks for, recorded in this document and
  in `week3/README.md` by the wrap-up phase.

Nothing is added to the command-line contract. Part 3 reads
`week3/artifacts/` and nothing else.

## Scope

In scope for Phase C:

- the estimator conventions of the sheet: the autocorrelation function of
  Equation 12, the integrated autocorrelation time of Equation 13 with the
  six-times-running-total truncation, the effective sample count of
  Equation 14 and the correlation-corrected error of Equation 15;
- `errors.txt`, the bootstrap envelope chart, and the three diagnostic charts;
- the extension estimate: `n_eff` at `L = 64`, `T = 2.3`, the sweep count that
  would make the honest error as small as today's naive error, and that count
  in hours at the measured sweep rate;
- a block-length stability verdict for the bootstrap `T_c` error and a plateau
  verdict for the binning error.

Out of scope, because the sheet puts it in Part 4: the Wolff update, the
cluster runs, the sampler-agreement comparison, `magnetization-compare.png`,
`tau-compare.png`, and `week3/README.md` itself (a later phase writes it).
`week3/scripts/compare.py` is a Part 4 deliverable and is not written here.

## Inputs

The four Metropolis ramps that Part 2 wrote into the git-ignored
`week3/artifacts/`, exactly as they stand: `coarse-l32` and `coarse-l64`
cover `1.5 .. 3.5` step `0.1` with 2000 discarded and 5000 measured sweeps per
temperature, and `window-l32` and `window-l64` cover `2.0 .. 2.6` step `0.05`
with 2000 discarded and 100000 measured sweeps per temperature. Every row of
`series.jsonl` carries `L`, `T`, `sweep`, `M` (signed mean spin) and `E`.

The analysis of a single temperature uses one series, and the analysed
observable is `a_k = |M_k|`, the absolute magnetization, because that is the
quantity whose error bar the week is about. The signed `m` also carries rare
global sign flips between the two ordered orientations, which the
absolute-magnetization susceptibility of Equation 10 already subtracts away;
mixing the two would make `tau_int` describe the sign-flip process instead of
the fluctuation process.

## Estimators

### Autocorrelation and `tau_int`

For a series `a` of `n` measured sweeps the sheet's autocorrelation is

```text
rho(t) = (<a_k a_{k+t}> - <a_k>^2) / (<a_k^2> - <a_k>^2)      (Equation 12)
```

estimated with the biased (single-pass, sum over `n - t` pairs divided by `n`)
estimator that the course's own diagnostic uses. The integrated
autocorrelation time is

```text
tau_int = 1/2 + sum_{t>=1} rho(t)                              (Equation 13)
```

and the sum is truncated exactly as the sheet describes and as
`week3/checker/tau` (the instructor diagnostic shipped with the course
resources) implements it: start from `1/2`, add `rho(t)` for increasing `t`,
stop before adding the first negative `rho(t)`, and stop once `t >= 6 * tau`
with `tau` the running total, never looking past lag `min(n // 4, 5000) - 1`.
The user-facing rule is "summed until the lag exceeds six times the sum,
stopping early if `rho` goes negative"; the checker's exact loop is

```text
tau = 0.5
for t in 1 .. min(n // 4, 5000) - 1:
    if rho(t) < 0: break
    tau += rho(t)
    if t >= 6 * tau: break
```

so the term at the breaking lag is included and the cap is `min(n // 4, 5000)`
lags. Matching that loop bit for bit is what makes an independent cross-check
reproduce the table.

The autocorrelation itself is computed by FFT: zero-pad the demeaned series to
`>= 2n - 1`, take `irfft(|rfft(x)|^2)[:n]`, and divide by `var * n` with the
population variance. That is exactly the checker's `sum_{k} x_k x_{k+t} / (n v)`
in `O(n log n)` instead of the `O(n^2)` direct sum, which matters for
100000-sweep series; the table is checked against the direct sum in the unit
tests.

### Error bars

```text
s_naive   = std(a, ddof=1) / sqrt(n)                            (Equation 15)
s_50block = std(block means, ddof=1) / sqrt(50),
            block means of the first 50 * floor(n / 50) sweeps
ratio     = s_50block / s_naive
```

and the correlation-corrected prediction is `sqrt(2 tau_int)`. The naive and
blocked errors and the definition of the blocks are the same in every script
here, so `errors.txt`, the binning chart and the bootstrap share one
implementation in `errors.py`.

### Merged grid

One row per size and temperature on the same 27-temperature merged grid that
Part 2 uses: inside `[2.0, 2.6]` the window rows (100000 sweeps each, twenty
times the coarse sampling) win, outside the coarse rows (5000 sweeps each)
stand alone. `errors.txt` and `tau.png` therefore read both runs per size,
while the bootstrap and the trace read the window runs inside the window.

## Bootstrap

`chi_bootstrap.py` block-bootstraps the two window runs. For each size and each
temperature of the 13-point window grid, the 100000 measured sweeps are cut
into `k = floor(n / b)` whole blocks of `b` sweeps and `k` blocks are drawn
with replacement; because the blocks all have the same length, the resampled
`<|m|>` and `<m^2>` are the means of the drawn block means, so the resampling
is exact and never materializes a 100000-row copy. `L = 32` and `L = 64` are
resampled independently at every temperature, with a fixed seed per replicate
so the run is reproducible.

Each replicate recomputes `chi(T) = L^2 (<m^2> - <|m|>^2) / T` (Equation 10
imported from `peaks.py`), the five-point parabola through the grid points
around the largest `chi` for each size (`peaks.fit_peak`, the same fit Part 2
used), and `T_c = 2 T_peak(64) - T_peak(32)` (Equation 11). A replicate is a
**failed fit** when its parabola does not bend downward (`curvature >= 0`, the
error `peaks.fit_peak` already raises) or when its vertex falls outside the
five fitted temperatures. The reported bootstrap error is the sample standard
deviation (`ddof=1`) of the successful replicates' `T_c`.

Block lengths 2000, 4000 and 8000 sweeps, 500 replicates each, are the sheet's
values. Their errors are called **stable** when the three values agree within a
tenth of their mean, `max - min <= mean / 10`; otherwise the deliverable
reports "sampling error unresolved".

This error covers sampling only. It excludes finite-size corrections and the
bias of fitting a five-point parabola to an asymmetric peak, which more sweeps
do not shrink.

## Charts

- `trace.png`: `|m|` against the first 2000 recorded sweeps at `T = 2.3` and
  `T = 3.0` for `L = 64`. The window run covers `2.0 .. 2.6`, so `T = 2.3` comes
  from `window-l64` and `T = 3.0` from `coarse-l64`; the chart and its report
  say which source each trace used.
- `acf-binning.png`: two panels at `L = 64`, `T = 2.3`. Left, `rho(t)` against
  lag from the window run, the sheet's Equation 12 curve. Right, the standard
  error of `<|m|>` against block length on a logarithmic axis, from block
  length one (which is by construction the naive error of `errors.txt`) up to
  5000 and beyond, with the number of surviving blocks shown. A rising curve
  with too few blocks for a plateau means the honest error is unresolved.
- `tau.png`: `tau_int` against temperature for both sizes on the merged grid
  with a logarithmic vertical axis and the dashed line at `T_c`.
- `chi-bootstrap.png`: `chi(T)` for both sizes, their five-point fits, the
  shaded envelope of the 500 fitted parabolas at each block length, and the
  marked `T_c = 2.26919`.

Every chart is matplotlib `Agg`, reads only `week3/artifacts/`, and is written
by a script that also prints the numbers it drew, so the committed PNG and the
reported numbers come from the same command.

## Extension: how long is long enough

At `L = 64`, `T = 2.3`, with `n = 100000` recorded sweeps and the measured
`tau_int`:

```text
n_eff = n / (2 tau_int)                                        (Equation 14)
N     = 2 tau_int * n     sweeps to reach today's naive error,
                          from s_naive sqrt(2 tau_int) = s_naive of a
                          run with 2 tau_int times the samples
hours = N / (measured sweep rate in sweeps per second) / 3600
```

The second number assumes `tau_int` stays fixed, so it is a lower bound on the
run length: on a larger lattice or a longer run the autocorrelation time near
`T_c` does not shrink. The sweep rate is measured in this phase with a
dedicated timed `ising` run at `L = 64` and cross-checked against the wall
clock of the committed `window-l64` ramp (1326000 sweeps including its
discarded ones).

The measured values are recorded under "Measured extension numbers" below.

## Uncertainties

The sheet asks for this discussion at every Part 3 step; the user is not
present, so it is recorded here and repeated in the report.

- **The error bar itself is not resolved.** At `L = 64`, `T = 2.3` the 50-block
  error is about thirty times the naive error and `tau_int` is of order six
  hundred sweeps, so 2000-sweep blocks hold only about three autocorrelation
  times: the blocked estimate is still short of the plateau, and Equation 15
  predicts `sqrt(2 tau_int)` above 30. The binning curve is the check that
  decides this, and the sheet expects it still rising at 5000-sweep blocks,
  where only twenty blocks remain. The honest verdict is therefore "sampling
  error unresolved" unless the curve flattens, and any number quoted from the
  50-block estimate at the transition is provisional.
- **`tau_int` from a finite window is biased low.** The estimator truncates
  once the running total is six times the lag, so a noisy short series
  truncates early and reports a small `tau_int`; the same observable measured
  with 5000 coarse sweeps reports a much smaller time than with 100000 window
  sweeps. That is why `tau.png` uses the window rows wherever they exist and
  why the coarse points outside the window are trustworthy only because
  `tau_int` there is a few sweeps.
- **A single seed.** Every run here is one random stream. The bootstrap
  resamples rows that already exist and therefore measures the sampling error
  of this run, not the run-to-run scatter of a different seed; it cannot see a
  bias common to all 100000 rows, such as residual equilibration.
- **Equilibration.** Each window temperature discards only 2000 sweeps before
  measuring 100000, far less than `tau_int` at the transition. The chain comes
  from the previous, colder temperature, so the first measured sweeps are
  closer to the ordered phase than equilibrium; at `T = 2.3` the reported mean
  `|m|` is a little above the answer key's, and a slowly decaying transient is
  one candidate. The trace chart is the visual check, and the bootstrap is
  blind to it.
- **`tau_int` is not the only correlation.** Only `|M|` is analysed;
  `chi` also needs `<m^2>`, whose autocorrelation time is not the same. The
  bootstrap handles that by resampling the rows, so `chi` and both peak fits
  see the same correlated data as the run did.
- **Bootstrap bias.** The block bootstrap assumes the block length is long
  enough to carry the correlations between blocks. At 2000 sweeps with
  `tau_int` near 600 that assumption is marginal, which is exactly why the
  same bootstrap is repeated at 4000 and 8000 sweeps and the three errors are
  compared. Agreement across block lengths is evidence that the resampling is
  no longer dominated by the block length; it is not evidence of correctness.
- **Peak-fit bias.** Equation 11 cancels the leading `1/L` shift but not the
  five-point parabola's own bias, which is a fixed offset in `T_c` that more
  sweeps do not remove, and the sheet says so explicitly. The bootstrap error
  and the finite-size bias are different quantities and are reported
  separately.
- **Sweep rate.** The sweep rate is a wall-clock measurement on this machine
  under whatever load was present, so the hours figure is an estimate for this
  laptop, not a property of the model.

## Risks and decisions

- **One estimator, three consumers.** `errors.py` owns the autocorrelation,
  `tau_int`, the naive error and the 50-block error; `acf_binning.py`,
  `tau.py` and `chi_bootstrap.py` import them. There is no second copy of the
  truncation rule to drift.
- **Matching the checker's rule exactly.** The truncation loop is a
  transcription of `week3/checker/tau`'s `tau_int`, including the
  `min(n // 4, 5000)` cap, the "break on negative `rho`" clause and the
  inclusion of the term at the breaking lag, so an independent recomputation
  with the shipped diagnostic reproduces the table instead of merely
  resembling it.
- **FFT autocorrelation.** `np.correlate(x, x, "full")` on a 100000-sweep
  series is the direct `O(n^2)` sum and is far too slow; the FFT form is
  algebraically identical and is pinned against the direct sum in the tests.
- **Testability without the 160 MB of rows.** The red tests build synthetic
  series whose autocorrelation function and `tau_int` are known in closed form
  (an AR(1)-like geometric decay), check the truncation rule against hand
  computed sums, and pin the bootstrap on a synthetic two-size window whose
  susceptibility is an exact parabola, so no test reads
  `week3/artifacts/`.
- **Runtime.** The bootstrap is minutes on the sheet's own protocol; here the
  block means are precomputed once per temperature and block length and each
  replicate only resamples them, so the run is dominated by reading 2.6
  million JSON rows. It is still launched in the background with a log, as the
  sheet instructs.
- **Committed size.** The four PNGs and the text table are all far below 5 MB;
  `week3/artifacts/` stays ignored.

## Measured extension numbers

(Filled in by the evidence commit that follows `errors.py`; see
`/tmp/amat5315-w3/part3-extension.md` for the copy the wrap-up phase quotes.)
