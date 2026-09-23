# Week 4: the advection-diffusion line and the two-dimensional flow solver

This directory holds the Week 4 exercise: three explicit integrators behind one
interface, the periodic line they are tested on where every answer is exact, the
two-dimensional incompressible vorticity equation solved with Fourier
pseudospectral derivatives, and the measurements that fix the step: the
stability map of the line, the two blow-up boundaries of the flow, the decay of
a random flow, and RK4's accuracy order on the fluid. The contract is
`field.design.toml` and `fluid.design.toml`, copied block by block from the
learning sheet; the sheet fixes the physics and the checks.

This README regenerates every committed file from a clean clone. Run every
command from `week4/`.

## Repository layout

```text
week4/
  Cargo.toml, Cargo.lock        Rust crate `week4`, binaries `field` and `fluid`
  Makefile                      `make reproduce`: the artifacts the gate reads
  field.design.toml             the contract of `field`
  fluid.design.toml             the contract of `fluid`
  README.md                     this document
  src/                          integrator.rs, line.rs, solver.rs, lib.rs, and the
                                two command lines bin/field.rs and bin/fluid.rs
  scripts/                      the line, comparison, order, refinement and plot scripts
  evidence/                     the nine committed figures and the convergence json
  artifacts/                    raw runs the figures are drawn from (generated)
  target/, .venv/               build and Python scratch
```

Everything tracked under `week4/` is source (`Cargo.toml`, `Cargo.lock`, the two
design files, `src/`, `scripts/`, `README.md`) or final evidence (`evidence/`).
`artifacts/`, `target/` and `.venv/` are ignored by `week4/.gitignore` and are
rebuilt by the commands below.

## Prerequisites

| tool | version this README was verified with | why |
| --- | --- | --- |
| `cargo` / `rustc` | 1.98.1 | builds and installs `field` and `fluid` (edition 2024) |
| `uv` | 0.12.9 | creates the Python virtual environment |
| `python3` | 3.12 | the interpreter `uv` uses |

Network access is needed twice: the design-file comparison downloads
`week4-resources.zip`, and the optional cross-check downloads the same archive.
Everything else is local.

## Install

```bash
cargo install --path . --quiet
uv venv .venv && uv pip install --python .venv/bin/python numpy matplotlib
.venv/bin/python -c "import numpy, matplotlib; print(numpy.__version__, matplotlib.__version__)"
```

`cargo install --path .` leaves `field` and `fluid` on `PATH`
(`~/.cargo/bin/`); `uv venv` creates the git-ignored `week4/.venv`. Every Python
command below is run as `.venv/bin/python ...`. Verified with numpy 2.5.3 and
matplotlib 3.11.2.

## The two design files

The sheet's green panels are the source: `field.design.toml` and
`fluid.design.toml` were copied block by block from them. The course also
publishes a reference copy, which differs only in whitespace (it aligns the `=`
signs into a column and keeps blank lines between groups), so comparing the
blocks means comparing the files with the alignment and the blank lines
normalised away:

```bash
curl -fsSL -o /tmp/week4-resources.zip \
  https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week4-resources.zip
unzip -p /tmp/week4-resources.zip week4/field.design.toml > /tmp/w4-field.design.toml
unzip -p /tmp/week4-resources.zip week4/fluid.design.toml > /tmp/w4-fluid.design.toml
diff -w /tmp/w4-field.design.toml field.design.toml && echo "field block-identical"
diff -w <(grep -v '^[[:space:]]*$' /tmp/w4-fluid.design.toml) \
        <(grep -v '^[[:space:]]*$' fluid.design.toml) && echo "fluid block-identical"
```

Both lines print `... block-identical` and nothing else, so every key, every
value, and every block of both files agrees with the published reference. The
two tools communicate through one JSON object: `field` writes a velocity field
to stdout, `fluid` reads it, copies its metadata into `run.json`, and writes
`fields.jsonl` frames with six decimals.

## Part 1: integrators on a line

`src/line.rs` holds the periodic line of Equation 6 with Fourier multipliers and
with centred differences, and `src/integrator.rs` holds forward Euler, the
midpoint rule and classical RK4 behind one trait. The two Part 1 figures are
drawn from measurements the crate itself takes: `scripts/line.py` runs
`cargo test --release line_dump -- --nocapture` and reads the one JSON line the
crate prints.

```bash
.venv/bin/python scripts/line.py            # -> evidence/line-stability.png
                                            #    evidence/line-accuracy.png
```

The printed numbers are the mode test and the four error slopes:

```text
modes at dt = 0.045: max |R| = 0.9978 over the travelling modes, 0 of 63 outside
modes at dt = 0.056: max |R| = 1.8505 over the travelling modes, 7 of 63 outside
exact stability limit of the line: h = 0.049386
|R| at the axis crossings: real 0.9996, imaginary 1.0040
Part 1 (b), maximum error at t = 2 pi:
  euler_fourier  2.116e-01
  rk4_fd         3.073e-01
  rk4_fourier    1.804e-05
Part 1 (b), fitted log-log slopes: 1.033 (euler), 2.005 (rk2), 4.004 (rk4), 2.003 (rk4, equal weights)
```

## Part 2: discretize the flow and build the solver

The derivative comparison of check (1) asks the solver itself:

```bash
.venv/bin/python scripts/comparison.py      # prints the table and the ratios
```

Measured on `g = sin(3x) cos(2y)`, `n = 32` and `n = 64`, full precision:

| derivative | FD, n=32 | FD, n=64 | Fourier, n=32 |
| --- | ---: | ---: | ---: |
| dx g | 0.17050 | 0.04318 | < 1e-10 |
| dxx g | 0.25724 | 0.06487 | < 1e-10 |
| dxdy g | 0.48534 | 0.12429 | < 1e-10 |
| Laplacian g | 0.30838 | 0.07771 | < 1e-10 |

The finite-difference columns fall by 3.90 to 3.97 when the grid is halved, the
second-order law; the Fourier column is at roundoff.

## The two pipelines

`make reproduce` is the gate's entry point. The published checker reads the raw
artifacts of a checkout, which the sheet keeps untracked, so a fresh clone has
none until this runs; it installs the crate and writes exactly the runs the gate
reads, honouring `SEED` as `${SEED:-2026}` for the random field:

```bash
make reproduce          # SEED defaults to 2026
SEED=7 make reproduce   # the same runs on another random field
```

It writes `artifacts/taylor-green/`, `artifacts/random/` (`n = 128`,
`nu = 0.004`, `t_end = 10`), the three `artifacts/order/rk4-dt*/` runs
(`taylor-green`, `n = 8`, `nu = 0.5`, `t_end = 2`, `dt = 0.4, 0.25, 0.2`) and
`artifacts/unstable/taylor-green/` (`rk4`, `n = 64`, `nu = 0.1`, `dt = 0.04`),
and nothing else: the tracked `evidence/` figures are redrawn by the scripts
below, so a different `SEED` cannot dirty a committed figure.

The Taylor-Green case of check (2), and the random flow of Part 3, each from the
generator into the solver:

```bash
mkdir -p artifacts
field taylor-green --n 64 \
  | fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 \
      --out artifacts/taylor-green > artifacts/taylor-green.tsv
field taylor-green --n 64 --nu 0.1 --t 1 > artifacts/taylor-green/exact-t1.json
field random --n 128 --seed 2026 --k-min 2 --k-max 6 \
  | fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 \
      --out artifacts/random > artifacts/random.tsv
```

`head -2 artifacts/taylor-green.tsv` and `tail -1` give the six-decimal decay
`0.0 0.250000 0.500000` and `1.0 0.167580 0.335160`; the random run prints
`0.0 0.500000 6.634685` and `10.0 0.277588 1.048184`.

```bash
.venv/bin/python scripts/taylor_green.py    # -> evidence/taylor-green.png
.venv/bin/python scripts/random_flow.py     # -> evidence/random.png
```

`taylor_green.py` prints the relative velocity error of the last stored frame
against the exact field, `7.039e-07` (required below `1e-5`), and draws the
vorticity at `t = 0` and `t = 1` with velocity arrows on one colour scale
(`max |omega| = 2.000` and `1.637`). `random_flow.py` prints the first and last
lines of `artifacts/random.tsv` and draws the vorticity at `t = 0, 2, 5, 10` on
one colour scale, each frame labelled with the energy and enstrophy recomputed
from its own fields.

The random field's spectrum is flat inside the band and scaled to `E(0) = 0.5`,
as the design file fixes it, so its enstrophy is
`Z(0) = N / (2 sum 1/|k|^2) = 104 / 15.675198 = 6.6347` and the committed run
then falls to `Z(10) = 1.0482`, a factor of 6.33 where the sheet's Expected
output prints `6.6567`, a factor of seven. The two cannot both hold: for any
equal-amplitude integer ring `Z/E = N / sum(1/|k|^2) = 104 / 7.837599 = 13.2694`
forces `Z(0) = 6.6347` at `E(0) = 0.5`, while the key's `6.6567` implies
`Z/E = 13.3134`, a non-flat spectrum of the kind its own published data shows
(`week4/data/transfer.json`, per-shell amplitudes 0.3587, 0.2677, 0.2825,
0.3294, 0.1682 for `|k| = 2..6`) and whose generator is not published. This
follows the design file, the sheet's diagnostic ("an initial enstrophy far from
6.66 means the band or the amplitude is wrong") is not triggered at 0.33%, the
seed-to-seed spread of this generator (seeds 3, 11, 99, 2027: `E(10)` in
`[0.278, 0.292]`, `Z(10)` in `[0.921, 1.140]`) brackets the key's
`0.2939 / 0.9337`, and the published gate does not read these values. The
`random.png` colour scale is the sheet's rule applied to this field,
`+/- max |omega(0)| = +/- 11.08`, where the key's own field gives `+/- 10.97`.

## Part 3: find the stability limit

The run past the diffusive limit, and the scan of both cases on either side of
their limits, all under `artifacts/`:

```bash
mkdir -p artifacts/unstable artifacts/scan
field taylor-green --n 64 \
  | fluid --method rk4 --nu 0.1 --dt 0.04 --t-end 4 --every 0.1 \
      --out artifacts/unstable/taylor-green > artifacts/unstable/taylor-green.tsv

field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.032 --t-end 8 \
  --every 0.5 --out artifacts/scan/tg-rk4-0.032 > artifacts/scan/tg-rk4-0.032.tsv
field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.033 --t-end 8 \
  --every 0.5 --out artifacts/scan/tg-rk4-0.033 > artifacts/scan/tg-rk4-0.033.tsv
```

The Taylor-Green boundary sits between `0.032` and `0.033`, within 5% of the
predicted `2.785/88.2 = 0.0316`; the `0.033` run stores frames to `t = 7.8` and
then prints a non-finite line. The sheet then asks for the random flow at `0.038`
and `0.040`, and for smaller steps if both blow up:

```bash
for dt in 0.038 0.040 0.034 0.03125; do field random --n 128 --seed 2026 \
  --k-min 2 --k-max 6 | fluid --method rk4 --nu 0.004 --dt $dt --t-end 10 \
  --every 0.5 --out artifacts/scan/random-rk4-$dt > artifacts/scan/random-rk4-$dt.tsv
done
field random --n 128 --seed 2026 --k-min 2 --k-max 6 \
  | fluid --method euler --nu 0.004 --dt 0.01 --t-end 10 --every 0.5 \
      --out artifacts/scan/random-euler-0.010 > artifacts/scan/random-euler-0.010.tsv
```

Both required steps blew up (`0.038` and `0.040` stop at `t = 0.8`), so the
bracketing pair moved down: `0.034` stops at `t = 1.1` and `0.03125` reaches
`t = 10` with a stored frame there, since it is `0.5/16`: 320 steps land exactly
on `t = 10` and every 16th step on a snapshot. That is `1.6x` to `1.7x` the Equation 17 bound
`2.83 / (U_max |k|_max) = 2.83 / (2.4176 * 59.40) = 0.0197`, inside the sheet's
allowed factor of one to three. Forward Euler at `0.01`, which has no imaginary
stability interval, stops at `t = 1.1`.

```bash
.venv/bin/python scripts/blowup.py          # -> evidence/blowup.png
```

prints the largest speed of the random initial field, the bound, and each
stopping time, and draws the energy against time with a logarithmic energy axis,
one panel per case, the exact Taylor-Green decay dashed.

The sensitivity pair needs a perturbation the command line does not offer, so
the script builds it: each case is run twice at `dt = 0.01` to `t = 20`, the
second from the same field plus `-7e-5 M cos(3x) cos(4y)` in the vorticity, with
`M` the largest absolute initial `u` or `v` of that case.

```bash
.venv/bin/python scripts/sensitivity.py     # -> evidence/sensitivity.png
                                            #    artifacts/sensitivity/ (both runs of both cases)
```

Measured: the random pair grows from `1.976e-05` to `1.759e-03` (x89) against
`M = 2.0569`, while the Taylor-Green pair decays from `3.497e-05` to the storage
floor, so the blow-up of the scan is not the flow's own sensitivity.

## Part 4: measure the accuracy order and choose a step

Three Taylor-Green runs on the coarse `8 x 8` grid, then the random refinement
series with its `dt = 0.0025` reference on the same grid:

```bash
mkdir -p artifacts/order artifacts/convergence
for dt in 0.4 0.25 0.2; do field taylor-green --n 8 \
  | fluid --method rk4 --nu 0.5 --dt $dt --t-end 2 --every 2 \
      --out artifacts/order/rk4-dt$dt > artifacts/order/rk4-dt$dt.tsv
done
for dt in 0.02 0.0125 0.01 0.0025; do field random --n 128 --seed 2026 \
  --k-min 2 --k-max 6 | fluid --method rk4 --nu 0.004 --dt $dt --t-end 2 \
  --every 2 --out artifacts/convergence/rk4-dt$dt > artifacts/convergence/rk4-dt$dt.tsv
done
```

```bash
.venv/bin/python scripts/order.py           # -> evidence/order.png
.venv/bin/python scripts/refinement.py      # -> evidence/convergence.json
                                            #    evidence/convergence.png
```

`order.py` prints the three relative velocity errors `5.987e-04`, `7.880e-05`
and `3.574e-05`, their fitted slope `4.10` (required within 15% of 4), and the
six-decimal storage floor `4.266e-06`. `refinement.py` prints the relative
error of `omega` at `t = 2` for every run (`2.026e-05`, `3.044e-06`, `1.243e-06`
against the `dt = 0.0025` reference), the fitted slope `4.027` (required
`3.7 <= q <= 4.3`), the Richardson estimate `1.270e-06` at `dt = 0.01`, the
predicted errors at the three candidate steps, and the choice: `dt = 0.0125`,
predicted `3.100e-06` and measured `3.044e-06`, both below `5e-6`, while
`dt = 0.02` fails. Those two figures differ from the sheet's `3.46e-6 / 3.40e-6`
because every error in this series is measured against this generator's own
`dt = 0.0025` reference field, not the answer key's. The full record is
`evidence/convergence.json`.

## Script order

Each script beside the file it writes. Everything reads `artifacts/` produced by
the runs above.

| # | command | writes |
| ---: | --- | --- |
| 1 | `.venv/bin/python scripts/line.py` | `evidence/line-stability.png`, `evidence/line-accuracy.png` |
| 2 | `.venv/bin/python scripts/comparison.py` | nothing; the derivative table goes to the terminal |
| 3 | `.venv/bin/python scripts/taylor_green.py` | `evidence/taylor-green.png` |
| 4 | `.venv/bin/python scripts/random_flow.py` | `evidence/random.png` |
| 5 | `.venv/bin/python scripts/blowup.py` | `evidence/blowup.png` |
| 6 | `.venv/bin/python scripts/sensitivity.py` | `evidence/sensitivity.png`, `artifacts/sensitivity/` |
| 7 | `.venv/bin/python scripts/order.py` | `evidence/order.png` |
| 8 | `.venv/bin/python scripts/refinement.py` | `evidence/convergence.json`, `evidence/convergence.png` |

## Evidence inventory

Every file in `week4/evidence/`, beside the command that produces it.

| file | produced by |
| --- | --- |
| `line-stability.png` | `.venv/bin/python scripts/line.py` |
| `line-accuracy.png` | `.venv/bin/python scripts/line.py` |
| `taylor-green.png` | `.venv/bin/python scripts/taylor_green.py` |
| `random.png` | `.venv/bin/python scripts/random_flow.py` |
| `blowup.png` | `.venv/bin/python scripts/blowup.py` |
| `sensitivity.png` | `.venv/bin/python scripts/sensitivity.py` |
| `order.png` | `.venv/bin/python scripts/order.py` |
| `convergence.png` | `.venv/bin/python scripts/refinement.py` |
| `convergence.json` | `.venv/bin/python scripts/refinement.py` |

The tracked files at the `week4/` root (`git ls-files week4`):

```text
Cargo.toml, Cargo.lock        the Rust crate and its locked dependency graph
field.design.toml             the contract of `field`, block-identical to the sheet
fluid.design.toml             the contract of `fluid`, block-identical to the sheet
README.md                     this document
src/                          library and binary sources
scripts/                      the measurement and plot scripts
evidence/                     the inventory table above
```

## A cross-check with the course checker

The published `week4-resources.zip` also carries `week4/checker/check`, a
stdlib-only gate over the raw artifacts. It reads
`artifacts/{taylor-green,random}/`, the three `artifacts/order/` runs and
`artifacts/unstable/taylor-green/`, so `make reproduce` has to run first; the
gate can also pin the seed, which the same run must honour:

```bash
unzip -p /tmp/week4-resources.zip week4/checker/check > /tmp/w4check.py
make reproduce && python3 /tmp/w4check.py .
SEED=7 make reproduce && SEED=7 python3 /tmp/w4check.py .
```

It prints its own measurements and `PASS`; measured here, `taylor-green-field
7.038587e-07 < 1e-05`, `taylor-green-energy 9.565108e-08 < 1e-06`, `order
4.104406 >= 3.5`, and `unstable inf >= 10` with `dt = 0.04` at 1.27x the
predicted diffusive limit.

## What stays out of git

`week4/.gitignore` ignores `/target/`, `/artifacts/`, `/.venv/` and
`__pycache__/`. None of those paths is tracked:

```bash
git ls-files week4 | grep -E 'artifacts|target|\.venv' || echo "no generated path tracked"
```

## Regeneration checklist

From a clean clone, in order: install; the design-file comparison; the two
pipelines; the unstable run and the scan; the three order runs and the four
convergence runs; then the eight scripts in the table above. `make reproduce`
covers the install and every run the gate reads in one step. At that point every
tracked file under `week4/` exists, `git status` shows only the git-ignored
generated folders, and the numbers quoted in this README can be checked line by
line against the printed output of the scripts.
