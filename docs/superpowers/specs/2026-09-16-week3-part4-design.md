# Week 3 Part 4 Cluster Sampler Design

## Purpose

Phase D finishes Part 4 of the week-3 sheet, "Beat critical slowing down".
Part 3 measured the price of single-flip Metropolis at the transition: at
`L = 64`, `T = 2.3`, `tau_int` is about 600 sweeps, so 100000 recorded sweeps
hold only about 80 independent samples and the honest error bar is unresolved.
Part 4 changes the update rule instead of buying more sweeps:

- `ising --update wolff` in `week3/`, the single-cluster update of
  Wolf (1989) exactly as `week3/ising.design.toml` already describes it;
- two cluster runs of the same 13-temperature critical window, `artifacts/
  wolff-l32` and `artifacts/wolff-l64`, written by the installed release
  binary and kept out of git;
- `week3/scripts/magnetization_compare.py` and
  `week3/evidence/magnetization-compare.png`: `mean |M|` at `L = 64` from the
  Metropolis window run and from the cluster run with block-bootstrap error
  bars, the cluster susceptibility with its five-point peak fits and the
  extrapolated `T_c`, and the agreement statistic of Equation 18 at `T = 2.3`
  with the verdict the sheet's stability clause demands;
- `week3/scripts/compare.py` and `week3/evidence/tau-compare.png`: the
  work-normalized autocorrelation time of Equation 17 for both update rules at
  `L = 64`.

## Scope

In scope for Phase D:

- the Wolff single-cluster update, its physics tests and its contract
  consequences (no new flag: `--update wolff` is already in the contract);
- the two cluster runs and their wall-clock log;
- `magnetization_compare.py`, the `T_c` bootstrap over the cluster runs, the
  sampler-agreement statistic and verdict, and `compare.py`;
- the systematic-versus-sampling discussion the sheet asks for at each step.

Out of scope:

- `week3/README.md`. The sheet puts it under "Wrap up the work"; the run
  commands and timings are recorded in `week3/evidence/part4-runs.txt` so that
  the wrap-up phase can quote them.
- any change to `ising.design.toml`. `diff` against the reference copy stays
  empty.

## The update rule

One Wolff move is:

1. pick a site uniformly at random from the `L^2` sites;
2. grow the cluster outward from it: for each neighbour of a site already in
   the cluster, if the neighbour has the same spin as the site it was reached
   from, add it with probability

   ```text
   p = 1 - exp(-2 / T)                                          (Equation 16)
   ```

   and never add a neighbour of opposite spin;
3. repeat step 2 for the neighbours of every newly added site until no site is
   added;
4. flip every spin in the cluster. There is no acceptance test: the flip
   always happens.

Because the cluster only ever contains sites with the seed's spin, "the same
spin as the site it was reached from" is the same test as "the same spin as the
seed". Opposite-spin neighbours are rejected outright and consume no random
draw, which mirrors the Metropolis rule's treatment of `dE <= 0`.

Equation 16 is exactly the bond probability of the Fortuin-Kasteleyn
representation of the Ising model, so the grown cluster is a percolation
cluster drawn with the right weight and flipping all of its spins with
acceptance 1 leaves the Boltzmann distribution invariant (Sandvik's notes,
section 6.4). It is also why the rule has no acceptance test: the rejected
bonds are the boundary, and Equation 16 already contains the Boltzmann factor
of moving the boundary.

Two implementation notes the tests pin:

- One step is **one cluster move**, not an accumulated `L^2` of flipped spins.
  Recording after every move keeps the observation interval independent of the
  cluster sizes just seen; a "stop when the accumulated size reaches `L^2`"
  rule would change the observed cluster-size distribution, which is exactly
  what the sheet warns against.
- For `L = 2` the torus has two bonds between each pair of sites, so the same
  neighbour index appears twice in the growth loop. The two appearances are
  two distinct bonds and are drawn independently, which is what the `2 L^2`
  bonds of that torus need.

## Contract consequences

The CLI is unchanged; the contract already names `wolff` and says that one
step is then a cluster flip. What changes is the meaning of the existing
outputs:

| Output | Metropolis | Wolff |
| --- | --- | --- |
| `run.json.time_unit` | `"sweep"` | `"cluster_flip"` |
| `series.jsonl` row | `L, T, sweep, M, E` | those five and `cluster_size` |
| stdout third column | `accepted / proposals` | mean cluster size over discard + measure |
| one step | `L^2` proposals | one cluster move |
| `discard`, `measure`, `--every` | counts steps | counts cluster moves |

`series.jsonl`'s `sweep` still restarts at 1 after the discard at each
temperature and `spins.jsonl`'s `sweep` still counts steps cumulatively across
the whole ramp including discarded ones; only the unit of a step changes, and
`run.json.time_unit` says which unit the run used. The stdout header names the
third quantity (`acceptance` or `mean_cluster_size`) so the number cannot be
misread; the tab-separated shape is unchanged.

## Implementation

- `week3/src/wolff.rs`: `add_probability(T) = 1 - exp(-2/T)`,
  `bond_added(T, uniform)`, `grow_cluster(lattice, seed, T, rng) -> Vec<usize>`
  (the sites in the order they joined, the seed first) and
  `cluster_flip(lattice, T, rng) -> usize` (draw a uniform seed with
  `random_range(0..L^2)`, grow, flip all of it, return the size). The seed draw
  is the move's first draw; the tests use that to reproduce the chosen cluster
  from a cloned stream.
- `week3/src/lattice.rs`: `site_count()` and `flip_index(index)`, the latter
  used by the flip loop; `flip(row, col)` delegates to it.
- `week3/src/ramp.rs`: the per-temperature loop runs `discard` then `measure`
  steps of whichever rule `--update` names. `TemperatureResult` carries the
  acceptance rate (Metropolis) and the mean cluster size (Wolff) separately,
  so neither is mislabelled, and `Update::name()` / `Update::time_unit()`
  replace the mapping that lived in `artifacts.rs`.
- `week3/src/artifacts.rs`: the recorder knows the update rule and appends
  `cluster_size` to `series.jsonl` rows exactly when the run is Wolff.
- `week3/src/main.rs`: the header names the third column for the rule in use.

One `ChaCha8Rng` seeded from `--seed` still supplies the whole ramp, so the
same arguments write byte-identical artifacts, and `--every` still writes
`spins.jsonl` frames every `every` measured steps at each temperature.

## Tests

Rust, `week3/tests/wolff.rs` (new) and the Wolff cases added to
`week3/tests/contract.rs`:

- `p(T) = 1 - exp(-2/T)`: exact values at `T = 2.3` (0.5808) and
  `T = 2.26919` (0.5858), monotone in `T`, limits 0 at large `T` and 1 at
  small `T`, and an empirical bond rate within a Monte Carlo tolerance.
- a hand-checkable small case: an `L = 4` checkerboard has only opposite-spin
  neighbours, so every seed grows the single-site cluster whatever the draws;
  an all-up lattice at `T -> 0` (where `p = 1` exactly) grows the whole
  `L^2`, and a one-down lattice at `T -> 0` grows exactly the 15 up sites from
  an up seed and exactly the one down site from the down seed.
- opposite-spin neighbours are never added, and the grown set only ever holds
  the seed's spin.
- a flipped cluster is exactly the connected set that was grown: the changed
  sites after `cluster_flip` equal the sites `grow_cluster` returns from the
  same lattice, seed and stream (the seed is the move's first draw), and the
  changed sites form a connected set.
- the whole cluster flips with no acceptance test: every site in the grown set
  changes spin and no site outside it does, for every temperature drawn.
- reproducible seeding: the same seed reproduces the cluster-size sequence
  and the artifacts byte for byte; a different seed differs.
- contract: `run.json.time_unit == "cluster_flip"`, `series.jsonl` rows carry
  exactly `L, T, sweep, M, E, cluster_size` with an integer `1 <=
  cluster_size <= L^2`, stdout's third column is the mean of the recorded
  `cluster_size` values over discard + measure, `series.jsonl` `sweep`
  restarts at 1 at each temperature, and `spins.jsonl` `sweep` counts cluster
  moves cumulatively across the ramp including discard.

Python, `week3/scripts/test_part4.py` (new, stdlib `unittest`, no artifact
reads):

- the agreement statistic `d = |m1 - m2| / sqrt(s1^2 + s2^2)` (Equation 18) on
  hand values, and the verdict rule: agreement when both errors are stable and
  `d <= 3`; "agreement provisional" when either error still depends on the
  block length; a discrepancy when `d > 3`.
- the block-bootstrap error of a mean on synthetic series with known answers
  (an independent series tends to the naive error, a two-block series hits the
  exact resampling distribution).
- Equation 17 on the sheet's reference numbers: `4.736 * 921.6 / 64^2 =
  1.066`, and the `701.8 / 1.066 = 658` work ratio.
- the cluster reader: a synthetic Wolff run folder is read as
  `(L, [(T, signed M, cluster_size), ...])`.

The RED commit adds these tests before `src/wolff.rs` exists, so they fail to
compile (the module is missing) and `test_part4.py` fails at import.

## Runs

After the implementation is committed and installed with
`cargo install --path . --quiet`, the two cluster runs are launched in the
background with logs, exactly as the sheet writes them:

```text
ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32
```

Each writes 13 temperatures x 100000 rows = 1.3 million rows in
`artifacts/`, which stays git-ignored. The wall clock, the exact commands and
the row counts are recorded in `week3/evidence/part4-runs.txt`.

## The agreement statistic and its verdict

Both rules sample the same Boltzmann distribution, so `<|m|>` at the same `L`
and `T` must agree within its sampling errors:

```text
d = | <|m|>_1 - <|m|>_2 | / sqrt(sigma_1^2 + sigma_2^2)       (Equation 18)
```

The errors are block-bootstrap standard deviations of the mean of `|m|` at
block lengths 2000, 4000 and 8000 moves (Metropolis: sweeps), the same three
lengths Part 3's `chi_bootstrap.py` uses and its shared 500 replicates
(`chi_bootstrap.REPLICATES`, the count this phase's committed evidence was
produced with; Part 3's own default is a separate, larger constant), so the
stability verdict is the same test: the three values must agree within a tenth
of their mean. The sheet's rule is then:

- both errors stable and `d <= 3` -> report agreement;
- either error still depends on the block length -> "agreement provisional";
- `d > 3` -> a discrepancy to investigate, whatever the stability.

Part 3 already found the Metropolis error at `L = 64`, `T = 2.3` block-length
sensitive -- `evidence/errors.txt` reports a 50-block error of 0.0185 against
a naive error of 0.0006, and `evidence/acf-binning.png` has no plateau -- so
the Metropolis error is unresolved and the verdict is expected to be
"agreement provisional" even when `d` comes out small. The two samplers differ
in one more way that the statistic does not cover: the Metropolis window run
discards only 2000 sweeps at `T = 2.3`, about three autocorrelation times,
while each cluster temperature discards 20000 moves, about four thousand
`tau_moves`. A difference larger than the quoted errors is therefore a
statement about equilibration as much as about the update rule, and the report
says so.

## The cluster critical temperature

The second panel of `magnetization-compare.png` uses the same absolute
magnetization susceptibility as Part 2,

```text
chi(T) = L^2 (<m^2> - <|m|>^2) / T                            (Equation 10)
```

computed from the cluster runs on the 13-point window grid, the same
five-point parabola around the largest `chi` (`peaks.fit_peak`), and the same
two-size extrapolation

```text
T_c = 2 T_peak(64) - T_peak(32)                                (Equation 11)
```

`chi_bootstrap.py` is reused unchanged to block-bootstrap the cluster `T_c` at
2000, 4000 and 8000 moves with 500 replicates each and to apply the same
stability verdict, so the cluster error is computed by the same code as the
Part 3 error and can be compared with it.

## Work per independent sample

One cluster move flips `<c>` spins, so in units of `L^2` flipped spins (one
Metropolis sweep) it costs `<c> / L^2`:

```text
tau_work = tau_moves * <c> / L^2                               (Equation 17)
```

`compare.py` draws `tau_work` against temperature for `L = 64` from
`artifacts/window-l64` (Metropolis, already in sweeps) and from
`artifacts/wolff-l64` (`tau_moves` from the same Equation 13 estimator
`errors.py` uses, times the measured `<c> / L^2` of the same temperature), with
a logarithmic vertical axis and `T_c` marked.

The conversion compares spin-update counts, not elapsed time: cluster growth
and a single-spin proposal have different costs, so the factor is not a
wall-clock speedup. It is also the honest count of work per independent
sample, which is what critical slowing down is about.

## Why more samples do not remove every error

The sheet asks for this explanation with the comparison: the cluster run's
bootstrap `T_c` error is expected to be much smaller than the single-flip one
because the chain decorrelates in a few moves instead of hundreds of sweeps,
and the sampling error of any mean shrinks as `1 / sqrt(n_eff)` with the number
of independent samples. Three errors do **not** shrink with more data:

- the finite-size shift, which the two-size extrapolation only cancels to
  leading order in `1/L`; a residual `O(1/L^2)` term remains and is a property
  of the finite lattices, not of the sample size;
- the five-point parabola fit, which puts its vertex slightly off the true
  maximum of an asymmetric peak; that offset is fixed by the grid and the
  shape of `chi(T)`, and more sweeps at the same 13 temperatures do not move
  it;
- the single random stream and its equilibration history, which a bootstrap
  over the rows of that run cannot see.

The bootstrap error is therefore a **lower bound** on the uncertainty of the
central `T_c`, and the report separates the sampling error (which shrinks)
from the systematic bias (which does not).

## Uncertainties

- **The Metropolis error bar is unresolved**, so the sampler agreement is
  provisional by the sheet's own rule even if `d` is small. The cluster error
  is checked with the same three block lengths; if it is stable, only the
  Metropolis side makes the verdict provisional.
- **Cluster `tau_moves` from a finite window.** Equation 13 truncates early
  for a noisy short series, so `tau_moves` is a lower bound; at `T = 2.3` the
  cluster chain decorrelates in a few moves, so the truncation is far less
  consequential than for Metropolis.
- **One stream per sampler.** The two runs have different seeds (42 and 1042
  for the two sizes) and the bootstrap sees only the sampling error of those
  rows, not the scatter between seeds.
- **Equilibration of the two runs differs** (2000 sweeps versus 20000 moves at
  each temperature), so a difference in the means is not purely a sampling
  fluctuation.
- **Work units are not seconds.** Equation 17 counts flipped spins; the
  cluster move's bookkeeping costs more per spin than a Metropolis proposal,
  so the factor is an upper bound on the wall-clock speedup.
- **Peak-fit and finite-size bias** remain in the cluster `T_c` exactly as in
  Part 2 and are not part of the bootstrap error.

## Risks and decisions

- **No contract change.** `--update wolff` already parses; the phase removes
  the "later phase" error and implements what the design toml describes.
- **Per-rule stdout header.** The header names the third quantity
  (`acceptance` or `mean_cluster_size`); the contract fixes the shape, not the
  word, and a mean cluster size labelled "acceptance" would mislead.
- **Two fields, not one.** `TemperatureResult` reports the acceptance rate and
  the mean cluster size separately so the Wolff value cannot be printed as an
  acceptance rate by accident.
- **One estimator, reused.** `errors.tau_int`, `peaks.fit_peak` and
  `chi_bootstrap.bootstrap_tc` keep a single definition of Equation 13,
  Equation 10 and the block bootstrap; `magnetization_compare.py` loads the
  cluster rows into the same shapes `chi_bootstrap.py` already consumes.
- **Cluster rows are large.** The two runs write about 250 MB into the
  git-ignored `artifacts/`; the evidence that is committed is the run log, the
  two PNGs and the text reports.
- **Runtime.** The sheet's reference run took about 95 s on an Apple M2; the
  runs here are launched in the background with logs and polled, and the wall
  clock is recorded rather than predicted.

## Measured numbers

The measured agreement statistic and verdict, the cluster peaks and `T_c` with
its bootstrap error, and the work ratio at `T = 2.3` are recorded in
`week3/evidence/magnetization-compare.txt`, `week3/evidence/tau-compare.txt`
and `week3/evidence/part4-runs.txt` by the commands that produced them. This
document states no number it has not measured.
