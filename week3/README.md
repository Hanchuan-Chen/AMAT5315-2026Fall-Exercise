# Week 3: the Ising sampler, the critical window, and the cluster update

This directory holds the Week 3 exercise: a Rust command `ising` that samples
the two-dimensional Ising model, the analysis that turns its raw rows into
evidence about the critical temperature, the error bars on that evidence, and
the Wolff cluster update that beats single-flip critical slowing down. The
whole contract is `ising.design.toml`, copied from the course's published
`week3-resources.zip`; the learning sheet fixes the physics and the checks.

This README regenerates every committed file from a clean clone. Run every
command from `week3/` unless it says otherwise.

## Repository layout

```text
week3/
  Cargo.toml, Cargo.lock        Rust crate `ising`
  ising.design.toml             the published command-line contract
  README.md                     this document
  src/                          lattice, metropolis, wolff, ramp, artifacts, cli
  tests/                        physics, contract, reproducibility, wolff tests
  scripts/                      analysis and chart scripts, plus their tests
  spins.jsonl                   the committed ramp recording the viewer reads
  evidence/                     every committed chart, table and proof
  artifacts/                    raw rows of the four Metropolis runs and two
                                cluster runs (generated, not committed)
  runs/                         Part 1 runs (generated, not committed)
  target/, .venv/, .viewer/     build, Python and viewer-capture scratch
```

Everything tracked under `week3/` is source (`Cargo.toml`, `Cargo.lock`,
`ising.design.toml`, `src/`, `tests/`, `scripts/`, `README.md`), the recording
(`spins.jsonl`), or final evidence (`evidence/`). `artifacts/`, `runs/`,
`target/`, `.venv/` and `.viewer/` are ignored by `week3/.gitignore` and are
rebuilt by the commands below.

## Prerequisites

| tool | version this README was verified with | why |
| --- | --- | --- |
| `cargo` / `rustc` | 1.98.1 | builds and installs `ising` (edition 2024) |
| `uv` | 0.12.9 | creates the Python virtual environment |
| `python3` | 3.12 | the interpreter `uv` uses |
| `node` | 24.20.0 | only for `scripts/capture_viewer.mjs` |
| Google Chrome | `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` | headless capture of the viewer proof |
| `python3 -m pytest` | any recent pytest | the Week 1 regression from the repository root |

Network access is needed twice: the contract comparison downloads
`week3-resources.zip`, and the viewer capture downloads the hosted viewer and
`puppeteer-core`. Everything else is local.

## Install

Build and install the release binary, then create the analysis environment:

```bash
cargo install --path . --quiet
uv venv .venv && uv pip install numpy matplotlib
.venv/bin/python -c "import numpy, matplotlib; print(numpy.__version__, matplotlib.__version__)"
```

`cargo install --path .` leaves `ising` on `PATH` (`~/.cargo/bin/ising`);
`uv venv` creates the git-ignored `week3/.venv`. Every Python command below is
run as `.venv/bin/python ...`. Verified with numpy 2.5.3 and matplotlib
3.11.2; the charts are byte-identical for that pair of versions.

## The contract

Download the current reference copy from the course site and compare it with
the committed one, section for section:

```bash
curl -fsSL -o /tmp/week3-resources.zip \
  https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week3-resources.zip
unzip -p /tmp/week3-resources.zip week3/ising.design.toml > /tmp/week3-ising.design.toml
diff /tmp/week3-ising.design.toml ising.design.toml && echo "contract identical"
```

The last line prints `contract identical` and nothing else. The usage line at
the top of the file is the command the Part 1 section runs verbatim:

```text
usage = "ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 --discard 2000 --measure 200 --every 20 --seed 2026 --out runs/ramp"
```

## Part 1: the single-spin sampler and its evidence

### The runs

All seven runs use `--update metropolis` on an `L = 64` lattice, start from an
all-up lattice, discard 2000 sweeps and measure 2000 sweeps at each
temperature:

```bash
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.05 \
  --discard 2000 --measure 2000 --seed 2026 --out runs/T1.8
ising --update metropolis --l 64 --t-from 3.0 --t-to 3.0 --t-step 0.05 \
  --discard 2000 --measure 2000 --seed 2026 --out runs/T3.0
ising --update metropolis --l 64 --t-from 3.1 --t-to 3.1 --t-step 0.05 \
  --discard 2000 --measure 2000 --seed 2026 --out runs/T3.1
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.05 \
  --discard 2000 --measure 2000 --seed 2026 --out runs/a
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.05 \
  --discard 2000 --measure 2000 --seed 2026 --out runs/b
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.05 \
  --discard 2000 --measure 2000 --seed 2027 --out runs/c
```

The cold, hot and Boltzmann runs are the checks the sheet asks for; `runs/a`
and `runs/b` are the same seed twice and `runs/c` is the different-seed
control. Each run writes `run.json` and `series.jsonl`. With the same seed the
two files are byte-identical:

```bash
diff runs/a/series.jsonl runs/b/series.jsonl && echo "seed 2026 reproduces byte for byte"
diff -q runs/a/series.jsonl runs/c/series.jsonl || echo "seed 2027 differs"
```

Measured stdout (the second column is `mean |M|` over the 2000 measured
sweeps):

| run | T | mean \|M\| | check |
| --- | ---: | ---: | --- |
| `runs/T1.8` | 1.8 | 0.9569 | inside `[0.95, 0.965]`; Onsager's exact value is 0.9569 |
| `runs/T3.0` | 3.0 | 0.0458 | below 0.06, so the ordered phase is gone |
| `runs/T3.1` | 3.1 | 0.0405 | the Boltzmann partner of `T = 3.0` |

The committed recording is the contract's own usage line, with `--every 20` so
the viewer has frames, and its `spins.jsonl` is copied to the tracked
`week3/spins.jsonl`:

```bash
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 \
  --discard 2000 --measure 200 --every 20 --seed 2026 --out runs/ramp
cp runs/ramp/spins.jsonl spins.jsonl
```

The ramp records 410 frames, one every 20 measured sweeps at each of the 41
temperatures; `spins.jsonl` is the largest tracked file at about 3.9 MB.

### Boltzmann chart

```bash
.venv/bin/python scripts/boltzmann.py
```

reads `runs/T3.0` and `runs/T3.1` and writes `evidence/boltzmann.png`. For two
temperatures on one lattice the density of states cancels, so
`ln(P_3.1(E) / P_3.0(E))` is linear in the total energy with slope
`1/3.0 - 1/3.1 = 0.0107527`. Measured with 40-unit bins: 25 bins, 15 kept;
free least-squares slope 0.0104153 against the predicted 0.0107527; the ratio
points sit at most 0.454 (rms 0.205) off the predicted line where both bins
hold at least 20 sweeps.

### Viewer proofs

The three proofs are the published viewer's own `window.composeProofPNG()`
capture of the committed recording, at the ordered, critical and disordered
temperatures. The script serves a folder holding a copy of the viewer as
`index.html` beside `spins.jsonl` and `runs/ramp/run.json`, so it needs the
hosted viewer (downloaded on first use), headless Chrome, and `puppeteer-core`
installed in the git-ignored `.viewer/`:

```bash
mkdir -p .viewer && cd .viewer && npm --cache /tmp/npmcache i puppeteer-core && cd ..
node scripts/capture_viewer.mjs 1.8 evidence/viewer-T1.8.png
node scripts/capture_viewer.mjs 2.3 evidence/viewer-T2.3.png
node scripts/capture_viewer.mjs 3.0 evidence/viewer-T3.0.png
```

Each proof is the viewer's own stamped caption over the lattice and the
`m`-against-`sweep` trace, so it records the temperature, the sweep, `m`, the
frame index, `L = 64`, the `spins.jsonl` size, the frame count and the capture
time. A later capture of the same recording is the same picture with a new
timestamp; the committed stamps were taken on 2026-09-16.

## Part 2: the magnetization and the critical temperature

### The four ramps

Two lattice sizes on two temperature grids, each written into the git-ignored
`artifacts/`. The coarse ramps cover the ordered and disordered phases, the
window ramps spend 20 times as many sweeps inside `[2.0, 2.6]`:

```bash
ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 \
  --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 \
  --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64
ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32
ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64
```

Measured on the machine this README was verified on: 2.80 s, 10.54 s, 22.25 s
and 84.59 s when the four ran one process per ramp; 105000 rows per coarse
folder (21 temperatures x 5000 sweeps) and 1300000 rows per window folder (13
temperatures x 100000 sweeps), about 160 MB in total.

### Analysis

```bash
.venv/bin/python scripts/peaks.py                 # -> evidence/peaks.txt
.venv/bin/python scripts/plot_magnetization.py    # -> evidence/magnetization.png
.venv/bin/python scripts/plot_susceptibility.py   # -> evidence/susceptibility.png
```

`peaks.py` defines the sheet's absolute-magnetization susceptibility
`chi(T) = L^2 (<m^2> - <|m|>^2) / T`, fits a parabola through the five grid
points around the largest `chi` of each size, and extrapolates
`T_c = 2 T_peak(64) - T_peak(32)` to cancel the leading `1/L` shift. The two
chart scripts import that module, so the definition exists once. Measured:

```text
L = 32   cold mean |M| at T = 1.5 : 0.9867   T_peak(32) = 2.3495
L = 64   cold mean |M| at T = 1.5 : 0.9866   T_peak(64) = 2.3108
T_c = 2.2720,  +0.12% from Onsager's 2.26919 (gate: within 2%)
```

`magnetization.png` draws the `L = 64` points against Onsager's exact curve
with `T_c` marked; the lattice rounds the transition instead of meeting it.
`susceptibility.png` draws both sizes with their fitted peaks.

## Part 3: how much to trust Part 2

```bash
.venv/bin/python scripts/errors.py          # -> evidence/errors.txt
.venv/bin/python scripts/trace.py           # -> evidence/trace.png
.venv/bin/python scripts/acf_binning.py     # -> evidence/acf-binning.png
.venv/bin/python scripts/tau.py             # -> evidence/tau.png
.venv/bin/python scripts/chi_bootstrap.py   # -> evidence/chi-bootstrap.png
```

All five read `artifacts/` and nothing else (about 8 s in total). `errors.py`
owns the estimators: the autocorrelation of `|M|`, the integrated
autocorrelation time `tau_int = 1/2 + sum rho(t)` truncated at the sheet's
six-times rule, the naive standard error, and the 50-block error. Measured
rows of `evidence/errors.txt`:

| L | T | mean \|M\| | naive | 50-block | ratio | tau_int | n |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2.30 | 0.5452 | 0.0006875 | 0.0128429 | 18.68 | 229.40 | 100000 |
| 64 | 1.50 | 0.9866 | 0.0000431 | 0.0000869 | 2.02 | 1.65 | 5000 |
| 64 | 2.30 | 0.4724 | 0.0006023 | 0.0184699 | 30.66 | 607.71 | 100000 |
| 64 | 3.50 | 0.0287 | 0.0003087 | 0.0006154 | 1.99 | 2.26 | 5000 |

At `L = 64`, `T = 2.3` the ratio is 30.66 against the sheet's reference 31.16
and `tau_int` is 607.71 sweeps against 669.22; away from the transition the
ratio is about 2 and `tau_int` a couple of sweeps. The binning chart has no
plateau up to 20000-sweep blocks, so the honest error bar on `<|M|>` there is
at least 0.0242 and the sampling error is reported as unresolved.

The binning curve itself, from `evidence/acf-binning.png` (`|M|` at `L = 64`,
`T = 2.3`, error on the mean against block length): 1 sweep -> 0.000602 (the
naive error of `errors.txt`, which the script asserts equals the block-length-one
point), 512 -> 0.0121, 2048 -> 0.0184, 5000 -> 0.0180 with 20 blocks left, and
20000 -> 0.0242 with 5 blocks left; `rho(t)` first goes negative at lag 2441.
`evidence/tau.png` peaks at `tau_int = 229.40` sweeps for `L = 32` and 607.71
sweeps for `L = 64`, both at `T = 2.30`, against the sheet's reference 190 and
670; the flat ends are 1.6 to 2.9 sweeps, a rise of more than a factor of a
hundred, and the `L = 64` spike is the taller one.

`chi_bootstrap.py` block-bootstraps the two window ramps, 500 replicates each
at block lengths 2000, 4000 and 8000 sweeps: the bootstrap error of `T_c` is
0.0097 / 0.0100 / 0.0100 against the sheet's reference 0.0096 / 0.0095 /
0.0092, stable within a tenth of its mean, with no failed fits. As an extra,
non-committed cross-check, the course's own diagnostic (the shipped
`week3/checker/check` and `week3/checker/tau` run against a legacy-layout
aggregate rebuilt from the same rows) reproduced `T_c = 2.2720` and every
`tau_int`, naive, binned and ratio value of `evidence/errors.txt` exactly, for
example `L = 64`, `T = 2.30`: `tau_int = 607.71`, `naive = 6.023e-04`,
`binned = 1.847e-02`, `ratio = 30.66`.

### Extension: how long is long enough

From the `L = 64`, `T = 2.30` row of `evidence/errors.txt` (`tau_int =
607.71` sweeps over `n = 100000` recorded sweeps), Equation 14 gives

```text
n_eff = n / (2 tau_int) = 100000 / (2 * 607.71) = 82.3 independent samples
N     = 2 tau_int * n   = 121542000 sweeps       about 1.22e8, 1215x today's run
```

The sweep rate is measured, not assumed. Three timed runs at `L = 64`,
`T = 2.3`:

```bash
ising --update metropolis --l 64 --t-from 2.3 --t-to 2.3 --t-step 0.05 \
  --discard 2000 --measure 100000 --seed 7 --out /tmp/rate-l64
```

gave 6.49 s, 6.52 s and 6.48 s for 102000 sweeps each, a mean of 15700.5
sweeps/s; the committed `window-l64` ramp independently implies 15675.6
sweeps/s (1326000 sweeps in 84.59 s), within 0.2%. So

```text
hours = 121542000 / 15700.5 / 3600 = 2.15 hours   about 2 h 9 min
```

That converts the run length, assuming `tau_int` stays fixed, so 2.15 hours is
a lower bound: the honest error at `T = 2.3` is unresolved, and the longer run
buys about 82 effective samples for the whole length, not 100000. The rate is
a wall-clock measurement of one machine under whatever load it carried (a
repeat inside the fresh clone gave 6.12, 6.13 and 6.16 s, about 16600
sweeps/s), so the hours figure is an estimate for this laptop, not a property
of the model.

## Part 4: beat critical slowing down

### The two cluster runs

```bash
ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
  --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32
```

One step is one cluster flip, so `discard`, `measure` and `--every` count
cluster moves, `run.json` says `time_unit = "cluster_flip"`, each row adds
`cluster_size`, and stdout's third column is the mean cluster size over
discard plus measure. Measured: 53.37 s and 16.09 s; 1300000 rows each.
Below `T_c` the mean cluster covers most of the lattice (3397.2 of 4096 spins
at `T = 2.0`); above it the cluster shrinks (35.7 spins at `T = 2.6`).

### Analysis

```bash
.venv/bin/python scripts/magnetization_compare.py   # -> evidence/magnetization-compare.png
                                                    #    evidence/magnetization-compare.txt
.venv/bin/python scripts/compare.py                 # -> evidence/tau-compare.png
                                                    #    evidence/tau-compare.txt
```

Both read the Metropolis window and the cluster runs. `magnetization_compare.py`
draws `<|M|>` at `L = 64` for both rules with block-bootstrap errors, tests
their agreement with `d = |m1 - m2| / sqrt(s1^2 + s2^2)`, and fits the cluster
susceptibility peaks. Measured at `L = 64`, `T = 2.30`: Metropolis `<|m|> =
0.4724` and cluster `<|m|> = 0.4350`, so `d = 2.00 <= 3`, but the Metropolis
error is still block-length sensitive (0.0178 / 0.0198 / 0.0186 at 2000 / 4000
/ 8000 steps), so the verdict is **agreement provisional** by the sheet's own
rule. The cluster peaks are `T_peak(32) = 2.3489` and `T_peak(64) = 2.3121`,
giving `T_c = 2.2752`, `+0.26%` from 2.26919 (inside Part 2's 2% gate), with a
bootstrap error of 0.0006 at every block length.

`compare.py` converts the cluster's `tau_moves` into spin-update work with
`tau_work = tau_moves * <c> / L^2` (Equation 17). At `L = 64`, `T = 2.30` the
cluster chain has `tau_moves = 4.476` and `<c> = 935.6` spins per move, so
`tau_work = 1.022` sweeps against Metropolis's 607.7: **594 times less
spin-update work per independent sample** at that temperature, and 1236 times
at `T = 2.35`, the largest ratio on the window grid. The conversion counts
flipped and proposed spins, not seconds, so it is an upper bound on the
elapsed-time speedup.

## Script order

The order the scripts must run, each beside the file it writes. Everything
reads `runs/` or `artifacts/` produced by the runs above.

| # | command | writes |
| ---: | --- | --- |
| 1 | `.venv/bin/python scripts/boltzmann.py` | `evidence/boltzmann.png` |
| 2 | `node scripts/capture_viewer.mjs 1.8 evidence/viewer-T1.8.png` | `evidence/viewer-T1.8.png` |
| 3 | `node scripts/capture_viewer.mjs 2.3 evidence/viewer-T2.3.png` | `evidence/viewer-T2.3.png` |
| 4 | `node scripts/capture_viewer.mjs 3.0 evidence/viewer-T3.0.png` | `evidence/viewer-T3.0.png` |
| 5 | `.venv/bin/python scripts/peaks.py` | `evidence/peaks.txt` |
| 6 | `.venv/bin/python scripts/plot_magnetization.py` | `evidence/magnetization.png` |
| 7 | `.venv/bin/python scripts/plot_susceptibility.py` | `evidence/susceptibility.png` |
| 8 | `.venv/bin/python scripts/errors.py` | `evidence/errors.txt` |
| 9 | `.venv/bin/python scripts/trace.py` | `evidence/trace.png` |
| 10 | `.venv/bin/python scripts/acf_binning.py` | `evidence/acf-binning.png` |
| 11 | `.venv/bin/python scripts/tau.py` | `evidence/tau.png` |
| 12 | `.venv/bin/python scripts/chi_bootstrap.py` | `evidence/chi-bootstrap.png` |
| 13 | `.venv/bin/python scripts/magnetization_compare.py` | `evidence/magnetization-compare.png`, `evidence/magnetization-compare.txt` |
| 14 | `.venv/bin/python scripts/compare.py` | `evidence/tau-compare.png`, `evidence/tau-compare.txt` |
| 15 | `cp runs/ramp/spins.jsonl spins.jsonl` | `spins.jsonl` |

## Evidence inventory

Every file in `week3/evidence/`, beside the command that produces it. The
scripts take `--out`/`--report` arguments, so the paths can be redirected;
these are the committed names, which are also the defaults.

| file | produced by |
| --- | --- |
| `boltzmann.png` | `.venv/bin/python scripts/boltzmann.py` |
| `magnetization.png` | `.venv/bin/python scripts/plot_magnetization.py` |
| `susceptibility.png` | `.venv/bin/python scripts/plot_susceptibility.py` |
| `trace.png` | `.venv/bin/python scripts/trace.py` |
| `acf-binning.png` | `.venv/bin/python scripts/acf_binning.py` |
| `tau.png` | `.venv/bin/python scripts/tau.py` |
| `chi-bootstrap.png` | `.venv/bin/python scripts/chi_bootstrap.py` |
| `magnetization-compare.png` | `.venv/bin/python scripts/magnetization_compare.py` |
| `tau-compare.png` | `.venv/bin/python scripts/compare.py` |
| `viewer-T1.8.png` | `node scripts/capture_viewer.mjs 1.8 evidence/viewer-T1.8.png` |
| `viewer-T2.3.png` | `node scripts/capture_viewer.mjs 2.3 evidence/viewer-T2.3.png` |
| `viewer-T3.0.png` | `node scripts/capture_viewer.mjs 3.0 evidence/viewer-T3.0.png` |
| `peaks.txt` | `.venv/bin/python scripts/peaks.py` |
| `errors.txt` | `.venv/bin/python scripts/errors.py` |
| `magnetization-compare.txt` | `.venv/bin/python scripts/magnetization_compare.py` |
| `tau-compare.txt` | `.venv/bin/python scripts/compare.py` |

The tracked files at the `week3/` root (`git ls-files week3`):

```text
Cargo.toml, Cargo.lock        Rust crate and its locked dependency graph
ising.design.toml             the contract, byte-identical to the published copy
README.md                     this document
src/                          library and binary sources
tests/                        Rust integration tests
scripts/                      analysis scripts and their Python tests
spins.jsonl                   the committed viewer recording
evidence/                     the inventory table above
```

## What stays out of git

`week3/.gitignore` ignores `/target/`, `/artifacts/`, `/runs/`, `/.venv/` and
`/.viewer/`, plus `__pycache__/`. None of those paths is tracked:

```bash
git ls-files week3 | grep -E 'artifacts|runs|target|\.venv|\.viewer' || echo "no generated path tracked"
```

The raw rows are the bulk of the week (about 160 MB of Metropolis rows and 250
MB of cluster rows) and are regenerated by the runs above; the viewer scratch
folder holds `puppeteer-core` and the downloaded viewer copy. Every tracked
file is below 5 MB, the largest being `spins.jsonl` at about 3.9 MB.

## Tests

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release
.venv/bin/python -m unittest scripts.test_peaks scripts.test_part3 scripts.test_part4
cd .. && python3 -m pytest week1/
```

The Rust suite is 46 tests (17 contract, 12 physics, 6 reproducibility, 11
Wolff), the Python suites are 50 tests, and the Week 1 regression is 1 test;
all pass. The Rust tests cover the physics (energy, the five `dE` values, the
acceptance rule), the contract (`run.json`, `series.jsonl`, `spins.jsonl`, the
stdout table, the grid-inclusion rule), reproducibility (same seed identical
bytes, different seed differs) and the Wolff update (the exact Boltzmann
distribution on a 3x3 torus, the hand-checkable clusters, Equation 16). The
Python suites are self-contained: they build synthetic run folders, so they
never read `artifacts/`.

## Regeneration checklist

From a clean clone, in order: install; contract `diff`; the seven Part 1 runs
and `cp runs/ramp/spins.jsonl spins.jsonl`; `boltzmann.py`; the viewer setup
and three `capture_viewer.mjs` runs; the four Part 2 ramps; `peaks.py`,
`plot_magnetization.py`, `plot_susceptibility.py`; `errors.py`, `trace.py`,
`acf_binning.py`, `tau.py`, `chi_bootstrap.py`; the two cluster runs;
`magnetization_compare.py`, `compare.py`; then the tests above. At that point
every tracked file under `week3/` exists, `git status` shows only the
git-ignored generated folders, and the evidence tables in this README can be
checked line by line against `evidence/peaks.txt`, `evidence/errors.txt`,
`evidence/magnetization-compare.txt` and `evidence/tau-compare.txt`.
