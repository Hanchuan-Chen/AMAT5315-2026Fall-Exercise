# Week 3 Part 2 Analysis Design

## Purpose

Phase B finishes Part 2 of the week-3 sheet, "The magnet dies". The sampler is
already built and installed; nothing is added to the command-line contract.
Part 2 is an analysis phase on top of raw rows that the sampler writes:

- four Metropolis ramps, two lattice sizes and two temperature grids, into
  `week3/artifacts/`;
- `week3/scripts/peaks.py`, which turns those rows into a susceptibility curve
  per size, a fitted peak per size, and an extrapolated critical temperature;
- `week3/scripts/plot_magnetization.py` and
  `week3/scripts/plot_susceptibility.py`, which draw the two evidence charts;
- `week3/evidence/peaks.txt`, `magnetization.png`, `susceptibility.png`.

The point of the phase is the physics claim, not the tooling: the measured
magnetization must follow Onsager's infinite-lattice curve below the transition
and round it off near `T_c`, and the susceptibility peaks of two sizes must
cancel their leading `1/L` drift into a critical temperature within 2% of
Onsager's exact `2.26919`.

## Scope

In scope for Phase B:

- the four command lines the sheet spells out, run verbatim into
  `week3/artifacts/` with a release build installed with
  `cargo install --path . --quiet` first;
- a small importable analysis module (inside `peaks.py`) plus a CLI that writes
  `week3/evidence/peaks.txt`;
- the two matplotlib charts, drawn by scripts that read only the artifacts;
- the Part 2 verification numbers, reported against the sheet's stated pass
  conditions.

Out of scope, because the sheet puts them in Part 3 and 4: `errors.py`, the
autocorrelation time, the block-bootstrap envelopes
(`errors.txt`, `chi-bootstrap.png`), the size comparison
(`magnetization-compare.png`, `tau-compare.png`), and the Wolff update. The
Part 2 scripts are written so Part 3 can import them rather than re-derive the
susceptibility.

## Inputs: the four ramps

Every value is spelled out; the sheet adds nothing to the contract. Release
build, binary on `PATH`, one process per ramp:

```text
ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 \
  --discard 2000 --measure 5000  --seed 1042 --out artifacts/coarse-l32
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 \
  --discard 2000 --measure 5000  --seed 42   --out artifacts/coarse-l64
ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32
ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 2000 --measure 100000 --seed 42   --out artifacts/window-l64
```

Shapes, from the contract's grid rule `t_from + k * t_step <= t_to`:
the coarse ramps hold 21 temperatures (`1.5 .. 3.5` step `0.1`), the window
ramps hold 13 (`2.0 .. 2.6` step `0.05`), and the merged analysis grid holds the
sheet's 27 temperatures: the 21 coarse points minus the five interior points
`2.1 .. 2.5` that the window grid replaces, plus the eleven refined points
`2.05 .. 2.55`.

| folder | temperatures | measured rows | expectation |
| --- | --- | --- | --- |
| `coarse-l32` | 21 | 105000 | 5000 per temperature outside the window |
| `coarse-l64` | 21 | 105000 | 5000 per temperature outside the window |
| `window-l32` | 13 | 1300000 | 100000 per temperature inside `[2.0, 2.6]` |
| `window-l64` | 13 | 1300000 | 100000 per temperature inside `[2.0, 2.6]` |

`week3/artifacts/` stays out of git (already ignored); stdout goes to a log
beside each folder so the wall-clock times can be quoted in the README later.

## Analysis model

Rows are read from `series.jsonl`: one object per measured sweep with `L`, `T`,
`sweep`, `M` (the signed mean spin `m`) and `E` (energy per site). Within a
folder the rows are already grouped by temperature in ascending order, so the
reader accumulates per temperature and never needs to sort.

For one size `L` and one temperature `T`, over the measured sweeps:

```text
mean_abs_m(T) = <|m|>
m_sq(T)       = <m^2>
chi(T)        = L^2 (m_sq(T) - mean_abs_m(T)^2) / T          (Equation 10)
```

This is the absolute-magnetization susceptibility the sheet fixes: it subtracts
`<|m|>^2` so that a global sign reversal between two ordered orientations does
not masquerade as a fluctuation. The zero-field response form, subtracting
`<m>^2`, is deliberately not used; the two agree everywhere except deep in the
ordered phase, and the sheet's peak-location comparison is defined with
Equation 10.

Two grids cover different temperature ranges, so the analysis merges them per
size into the sheet's single 27-temperature grid: inside `[2.0, 2.6]` the window
rows are used, so the coarse rows at `2.1 .. 2.5` are dropped in favour of the
20-times-longer measurements at the same temperatures, while `2.0` and `2.6`
are the shared endpoints and the coarse temperatures outside the window stand
alone. Merging is by exact temperature value; the contract's nine-decimal grid
rounding makes `2.0`, `2.3`, `2.6` identical across the two folders.

## Peak fit and extrapolation

`chi(T)` is evaluated on the merged grid, which is uniform at `0.05` inside the
window. The largest `chi` point is found, the five consecutive grid points
centred on it are taken (two either side), and an exact parabola is fitted
through those five `(T, chi)` samples by least squares in the monomial basis.
`T_peak` is the vertex of that parabola, `T_peak = T_v - b / (2 a)` for
`chi = a x^2 + b x + c` with `x = T - T_v`. The fit is done on the centred
variable so the small `0.05` spacing does not make the normal equations
ill-conditioned.

The finite-size shift is `T_peak(L) ~ T_c + a / L + O(1/L^2)`, so

```text
T_c = 2 T_peak(64) - T_peak(32)                              (Equation 11)
```

cancels the leading term because `L = 64` carries half the error of `L = 32`.
The reported deviation is `(T_c - 2.26919) / 2.26919`.

`peaks.py` prints, per size, the cold mean `|m|` at the lowest temperature
`1.5`; then each peak; then `T_c` and its deviation. It saves the identical text
to `week3/evidence/peaks.txt`, so a reader who never sees the code can still
check the claim against the raw rows.

## Charts

Both charts are matplotlib (`Agg`), read only `week3/artifacts/`, and use one
shared helper module so the susceptibility is defined once.

`plot_magnetization.py` draws mean `|m|` against `T` for `L = 64` over the
merged grid, the Onsager infinite-lattice curve

```text
<|m|>(T) = (1 - sinh(2/T)^-4)^(1/8)   for T < T_c
         = 0                          for T >= T_c
```

(Equation 3 of the sheet, `T_c = 2 / ln(1 + sqrt 2) = 2.26919`), and a dashed
vertical line at `T_c`. The measured points should track the exact curve below
the transition and lift off it near `T_c`: that gap is the finite-size
rounding, not an error, and the chart is the evidence for it.

`plot_susceptibility.py` draws `chi(T)` for both sizes on the merged grid, the
dashed line at `T_c`, and dotted vertical lines at the two fitted `T_peak`
values, so the shift with size is visible against the shared horizontal axis.

## Uncertainties

The sheet asks for this discussion at every Part 2 step; the user is not
present, so the discussion is recorded here and repeated in the report.

- **Sampling error.** Every point is a finite average over correlated sweeps.
  Near `T_c` the Metropolis chain decorrelates over hundreds of sweeps, so
  100000 measured sweeps are worth far fewer independent samples, roughly
  `100000 / (2 tau_int)`; the coarse grid's 5000 sweeps are worse by the same
  factor. The susceptibility is built from the difference of two noisy near-equal
  averages (`<m^2>` and `<|m|>^2`), which amplifies that error. Part 3 exists
  precisely to put a number on it; Part 2 quotes none, on the sheet's own
  instruction.
- **Equilibration.** Each temperature starts from the previous one's lattice
  (warm start), so the 2000 discarded sweeps at a temperature are enough only if
  the chain crosses that temperature quickly. Both sizes start from all-up at
  `1.5`, deep in the ordered phase, and heat up; the ordered phase is the slow
  one for a local update, so a residual bias in the first measured sweeps above
  `T_c` is possible. A constant bias cancels in the two-size extrapolation
  only to the extent that it is size independent, which it is not.
- **Peak-fit bias.** A parabola through five points of a rounded, asymmetric
  peak is a model, not a measurement. The true finite-size peak is not exactly
  quadratic, the residual bias is largest when the true vertex falls between
  grid points, and the five-point window has only five degrees of freedom. The
  shift between the two sizes is `0.028` of a temperature, comparable to one
  grid step of `0.05`, so a fit bias of a fraction of a grid step in either size
  moves `T_c` by half that fraction after the `2 x` amplification.
- **Finite-size extrapolation.** Equation 11 assumes `T_peak(L) = T_c + a / L`
  and discards higher-order `1/L^2` corrections and the logarithmic corrections
  the 2D Ising transition actually carries. Two sizes cannot test the
  assumption; they only cancel the leading term. With `a` of order `1`, the
  neglected `1/L^2` term is of order `0.001` in `T_c`, small against the 2% gate
  but not against a careful error bar.
- **Discretization of the ramp.** `0.05` grid steps at a peak width of a few
  tenths of a temperature put only five to seven points on the peak. The
  fit interpolates between them, but the reported `T_peak` is only as good as
  the grid allows.

## Risks and decisions

- **Merging the two grids.** The sheet asks for a magnetization curve at
  `L = 64` and susceptibility for both sizes, "reading the metropolis runs".
  Inside `[2.0, 2.6]` the coarse run is 20 times noisier at the same
  temperatures, so the analysis prefers the window rows there; outside the
  window it uses the coarse rows. This is stated in `peaks.txt` so the choice is
  auditable.
- **One susceptibility definition.** Equation 10 is implemented once, in
  `peaks.py`, and imported by both chart scripts; there is no second copy to
  drift.
- **Testability without the 160 MB of rows.** The red test drives the analysis
  functions on a synthetic `run.json`/`series.jsonl` pair in a temporary
  directory with an analytically known susceptibility and an exactly known
  parabola vertex, so the test is fast, deterministic and independent of the
  ramps.
- **Committed size.** The three evidence outputs are small: two PNGs of a few
  hundred kilobytes and a text file of a few hundred bytes. The rows themselves
  stay in `week3/artifacts/`, ignored by git.
