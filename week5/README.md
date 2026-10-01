# Week 5: automatic differentiation and checkpointing

This directory holds the Week 5 exercise: a hand-written JAX forward and reverse
pass for the Lennard-Jones pair energy, a Rust damped acoustic wave solver whose
one-timestep derivatives come from Enzyme, its Born forward mode and reverse-time
migration adjoint, and a Rust port of Treeverse that reverses the 240-step
reflector simulation from six saved states and images the Marmousi model. The
contract is `seismic.design.toml`, copied block by block from the learning
sheet; the sheet fixes the physics and the checks.

This README regenerates every committed file from a clean clone. Run every
command from `week5/`.

## Repository layout

```text
week5/
  rust-toolchain.toml           pins nightly-2026-09-05 with the enzyme component
  seismic/                      Rust crate `seismic`
    Cargo.toml, Cargo.lock      the crate and its locked dependency graph
    build.rs                    compiles src/kernel.rs with -Zautodiff=Enable
    rust-toolchain.toml         the crate-root pin the enzyme-setup skill asks for
    src/kernel.rs               isolated no_std Enzyme kernel (input to build.rs only)
    src/deriv.rs                safe C-ABI wrappers, one derivative call per timestep
    src/experiment.rs           experiment parser
    src/sim.rs                  the Rust loop that chains the timestep derivatives
    src/treeverse.rs            Treeverse schedule (port of TreeverseAlgorithm.jl)
    src/npy.rs, src/cli.rs, src/bin/seismic.rs
    tests/                      kernel transpose, Treeverse work, end-to-end checks
  scripts/                      the Python measurement and plot scripts
  tests/                        the JAX Part 1 checks
  seismic.design.toml           the contract of `seismic`
  pyproject.toml, uv.lock       the pinned Python environment
  Makefile                      `make reproduce`
  README.md                     this document
  inputs/                       reflector.json, marmousi.json (downloads, ignored)
  artifacts/                    the runs the figures are drawn from, and the evidence
```

Everything tracked under `week5/` is source (the crate, the two toolchain files,
the design file, `scripts/`, `tests/`, the Python project files, `README.md`,
`MARMOUSI-LICENSE`) or committed evidence. `inputs/`, `.npy` arrays, `seismic/target/`
and `.venv/` are ignored by `week5/.gitignore` and are rebuilt by the commands
below.

## Prerequisites

| tool | version this README was verified with | why |
| --- | --- | --- |
| `cargo` / `rustc` | stable 1.98.1 for the crate, `nightly-2026-09-05` for Enzyme | builds and installs `seismic` (edition 2024) |
| `uv` | 0.12.9 | creates the Python virtual environment |
| `python3` | 3.12.14 | the interpreter `uv` uses |

Network access is needed twice: the ignored inputs come from
`week5-inputs.zip`, and `cargo install` fetches `serde_json`.

## Install

The Enzyme toolchain is already present on this machine; a fresh one is one
command:

```bash
rustup toolchain install nightly-2026-09-05 --profile minimal --component enzyme
cargo install --path seismic --quiet          # leaves `seismic` on PATH
uv venv .venv && uv pip install --python .venv/bin/python numpy matplotlib jax pytest
.venv/bin/python -c "import numpy, matplotlib, jax; print(numpy.__version__, matplotlib.__version__, jax.__version__)"
```

Verified with numpy 2.5.3, matplotlib 3.11.2 and jax 0.11.2. Every Python
command below is run as `.venv/bin/python ...`.

The `enzyme-setup` skill is installed at the repository root,
`.agents/skills/enzyme-setup/SKILL.md`. It fixes the isolation this crate uses:
`build.rs` compiles only `seismic/src/kernel.rs` with `-Zautodiff=Enable` into a
`no_std` static library; the rest of the program stays ordinary Rust and reaches
the kernel through a C ABI, one local derivative call per timestep.

## Fetch the inputs

`inputs/` and the `.npy` arrays stay out of git. Re-download the two experiments
with Part 2's command:

```bash
curl -fLO https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-inputs.zip
unzip -o week5-inputs.zip && rm week5-inputs.zip
```

`MARMOUSI-LICENSE` is committed at the `week5/` root.

## The design file

`seismic.design.toml` contains the four green panels of the sheet — forward,
recording, reverse and checkpoint — copied block by block as the program grew.
The course publishes the same file, so comparing them means comparing with the
alignment normalised away:

```bash
curl -fsSL -o /tmp/week5-resources.zip \
  https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-resources.zip
unzip -p /tmp/week5-resources.zip week5/seismic.design.toml > /tmp/w5-design.toml
diff -w /tmp/w5-design.toml seismic.design.toml && echo "design block-identical"
```

## Part 1: hand-written JAX forward and reverse passes

`scripts/ad_modes.py` writes `artifacts/ad/derivatives.json` and draws
`artifacts/ad/modes.png`. The two passes are written one node at a time with
`jax.jvp` and `jax.vjp`; the sample of 601 separations uses the same chain rule
in vector form.

```bash
.venv/bin/python scripts/ad_modes.py
```

Measured: `energy = -0.6570169144600471`, `tangents.U = 2.239979929791144`,
`adjoints.r = jax_grad = 2.239979929791143` and
`adjoints.a = -2.342590311735973`, each inside `1e-12` of the sheet's values
(`a` collects both paths through the shared node). The maximum absolute errors
over the sample are

| method | maximum absolute error |
| --- | ---: |
| forward mode | 2.13e-14 |
| reverse mode | 1.42e-14 |
| centred finite difference, h = 1e-6 | 5.84e-09 |

Both AD errors are below `1e-12`; the finite-difference error is the largest, and
the curves cross zero at `r = 2^(1/6)`.

`scripts/ad_graph.py` records the jaxpr of `U` and of `grad(U)` at `r = 1.3` and
draws one node per operation, labelled with the operation's name, with edges
following the data dependencies.

```bash
.venv/bin/python scripts/ad_graph.py
```

`artifacts/ad/graph.png` shows the four operations `integer_pow[y=-6]`,
`integer_pow[y=2]`, `sub`, `mul`. `artifacts/ad/grad-graph.png` computes
`-6 r^-7` and `2 a` on its forward side and joins the two contributions to `a`
at one `add_any` node (highlighted), exactly as the sheet's reference does.

`scripts/ad_scaling.py` times a Lennard-Jones cluster energy and its gradient for
`N = 64, 128, 256, 512, 1024` atoms on a cubic lattice with spacing `2^(1/6)`,
each coordinate moved by a Gaussian of standard deviation 0.05.

```bash
.venv/bin/python scripts/ad_scaling.py
```

Measured on this machine in one `make reproduce` run:

| N | P = 3N | energy (ms) | forward (ms) | reverse (ms) | forward/energy | reverse/energy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 192 | 0.075 | 29.6 | 0.077 | 396.0 | 1.03 |
| 128 | 384 | 0.169 | 116.7 | 0.189 | 690.7 | 1.12 |
| 256 | 768 | 0.416 | 429.6 | 0.475 | 1033.3 | 1.14 |
| 512 | 1536 | 1.100 | 2174.5 | 1.324 | 1976.9 | 1.20 |
| 1024 | 3072 | 3.487 | 14332.1 | 4.813 | 4110.0 | 1.38 |

The forward ratio grows in proportion to `P` while the reverse ratio stays flat
near one; at `P = 3072` the forward ratio is 2978 times the reverse ratio, far
past the required factor of 100. Both modes' largest relative error against the
analytic forces is below `3e-15`. Wall-clock times fluctuate a few percent
between runs, so the ratios, not the milliseconds, are the check.

## Part 2: the Rust wave solver in forward mode

`seismic/src/sim.rs` carries the Rust loop; `seismic/src/kernel.rs` is the one
timestep Enzyme differentiates. The kernel implements
`u^{n+1} = (2u^n - (1 - dt sigma) u^{n-1} + dt^2 (c^2 L u^n + q)) / (1 + dt sigma)`
with the five-point Laplacian, the sponge `sigma`, a zero outer boundary and a
Ricker pulse times a unit-peak Gaussian footprint.

```bash
cargo install --path seismic --quiet
seismic --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward
.venv/bin/python scripts/inputs_figure.py
.venv/bin/python scripts/forward_figures.py
```

`scripts/inputs_figure.py` writes `artifacts/inputs.png`: the 1.8 km/s
background, the reflector at 2.1 km, three shots and fourteen receivers at
0.8 km, the dashed sponge inner edge at 0.6 km, and the Ricker pulse against
time. The pulse peaks at 1.00 at 1.5 s, troughs at -0.446 at 1.02 s and 1.98 s,
and is negligible after 2.6 s of the 4.8 s run.

`scripts/forward_figures.py` draws `artifacts/forward/gathers.png` and prints the
amplitudes. The L2 norm of all traces is `11.574770`, a relative error of
`4.3e-08` against the sheet's reference. Each shot's largest pressure falls at
trace index 83, one grid cell from its source:

| shot | source | peak | receiver | reference |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0 km | 0.60809514 | 1 | 0.60809514 |
| 1 | 2.0 km | 0.59271397 | 6 | 0.59271397 |
| 2 | 3.0 km | 0.60809514 | 12 | 0.60809514 |

The same script draws `artifacts/forward/wavefield.png` and
`artifacts/forward/echo.png` from step 150 (`t = 3.00 s`). The direct wave at
that step reaches 0.255 while the echo reaches 0.006, 2.4% of it, which is why
the recording also runs the first shot in background plus perturbation and keeps
the difference.

## Part 3: Enzyme differentiation and reverse-time migration

`born` chains Enzyme's timestep JVPs; `adjoint` stores the full trajectory and
chains the timestep VJPs backward, injecting the receiver weights from `--data`.

```bash
seismic --experiment inputs/reflector.json --mode born --out artifacts/born
seismic --experiment inputs/reflector.json --mode adjoint \
  --data artifacts/born/born_data.npy --every 3 --out artifacts/adjoint
.venv/bin/python scripts/adjoint_figures.py
```

The Born data have L2 norm `0.18667590`, a relative error of `2.0e-08` against
the reference. The transpose identity with `w = Jm` gives
`<Jm, Jm> = 0.03484789021516389` and `<m, J^T Jm> = 0.034847890215163865`, a
relative difference of `8.0e-16` (required below `1e-9`). The full history keeps
`N + 1 = 241` complete states, `6,481,936` bytes.

`scripts/adjoint_figures.py` draws `artifacts/adjoint/image.png`: the known
reflector, the raw signed image with its positive band and two negative side
lobes, and the row-L2 depth profile. The profile peaks at `z = 21`, 2.10 km, the
input depth; the depth difference is 0.0 km. It also draws
`artifacts/adjoint/wavefield.png` at step 132 (`t = 2.64 s`), where the receivers
re-emit the scattered data and the adjoint field converges on the reflector.

## Part 4: Treeverse checkpointing

`seismic/src/treeverse.rs` ports `src/treeverse.jl` of
`GiggleLiu/TreeverseAlgorithm.jl` (MIT, commit `72f22fe`), including the split
`ceil((d s + t p)/(t + d))`, the `max(s+1, p-1)` endpoint rule and the base case
that reverses the first step from its saved input. `s_0` is saved from the start
and never freed; steps `N-1, ..., 0` are each reversed once, in that order.

```bash
cargo install --path seismic --quiet
for b in 1 3 5 10; do
  seismic --experiment inputs/reflector.json --mode adjoint \
    --data artifacts/born/born_data.npy --storage treeverse --checkpoints $b \
    --out artifacts/checkpoint-$b
done
seismic --experiment inputs/marmousi.json --mode born --out artifacts/marmousi-born
seismic --experiment inputs/marmousi.json --mode adjoint \
  --data artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 \
  --out artifacts/marmousi-image
.venv/bin/python scripts/checkpoint_figures.py
.venv/bin/python scripts/audit.py
.venv/bin/python scripts/marmousi_figure.py
```

Marmousi is never run with full history: one shot would hold
`1201 * 2 * 805 * 269 * 8 = 4,161,128,720` bytes, 4.16 GB.

`scripts/checkpoint_figures.py` compares each checkpointed image with the
full-history image and draws `artifacts/checkpoint-actions.png` (budget 5, first
shot: the sawtooth of the schedule) and `artifacts/checkpoint-work.png`.

| budget | relative L2 error | peak saved states | peak bytes | forward steps per shot |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 | 2 | 53,792 | 28,680 |
| 3 | 0.0 | 4 | 107,584 | 1,695 |
| 5 | 0.0 | 6 | 161,376 | 990 |
| 10 | 0.0 | 11 | 295,856 | 642 |
| full | — | 241 | 6,481,936 | 240 |

Every checkpointed image is bit-identical to the full-history image (relative
error exactly 0), and the forward-step counts reproduce the sheet's reference
28,680 / 1,695 / 990 / 642.

`scripts/audit.py` replays every action file, including Marmousi's, and prints
the three counts the sheet asks for. All are zero:

```text
TOTAL audit: grad out of order/missing/extra=0, invalid restores=0, budget overruns=0
```

`scripts/marmousi_figure.py` draws `artifacts/marmousi.png`: the smoothed
background, the short-wavelength perturbation, the Born gather of the shot at
x = 10 km and the raw checkpointed image on one amplitude scale across depth.
The image L2 norm is `6.7037741e-04`, a relative error of `5.9e-09` against the
sheet's value; the run peaks at six saved states holding `20,788,320` bytes.

Why an accurate derivative still gives an image that differs from `m`: the
image is `J^T J m`, not `m`. `J` is the band-limited derivative of the receiver
data, so the migration squares the source's bandwidth and convolves with the
source–receiver geometry. The result locates a reflector but changes its
amplitude and width, and the same amplitude scale shows only the upper layers,
because the 3.6 s acquisition barely illuminates the deep model.

## `make reproduce`

`make reproduce` is the entry point. It installs the crate (Enzyme kernel
included), writes every raw run into `artifacts/`, then redraws every committed
figure and prints every numeric check.

```bash
make reproduce          # install, all runs, all figures, all checks
make test               # cargo test --release, then pytest
```

The published gate reads the raw artifacts:

```bash
curl -fsSL -o /tmp/week5-resources.zip \
  https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-resources.zip
unzip -p /tmp/week5-resources.zip week5/checker/check > /tmp/w5check.py
# the gate resolves week5/inputs relative to its own parents[2]; run it from a
# checkout that carries this directory, or symlink inputs beside it.
python3 /tmp/w5check.py .
```

Measured here, every line prints `PASS` and the gate ends with
`PASS: Week 5 visible numerical checks and checkpoint evidence`: AD-energy
`1.1e-16`, forward-trace-norm `4.3e-08`, forward peaks `5.6e-09` and `5.8e-09`,
Born-data-norm `2.0e-08`, Born-adjoint-dot-product `8.0e-16`,
reflector-depth-profile `z = 21`, the four checkpoint image errors `0`,
checkpoint action traces `2/4/6/11` states, Marmousi-image-norm `5.9e-09` and
Marmousi `6` states / `20,788,320` bytes.

## Evidence inventory

Every committed evidence file under `week5/artifacts/`, beside the command or
script that produces it. The `.npy` arrays in the same folders are generated but
ignored by git, as the sheet requires.

| file | produced by |
| --- | --- |
| `inputs.png` | `.venv/bin/python scripts/inputs_figure.py` |
| `ad/derivatives.json` | `.venv/bin/python scripts/ad_modes.py` |
| `ad/modes.png` | `.venv/bin/python scripts/ad_modes.py` |
| `ad/graph.png` | `.venv/bin/python scripts/ad_graph.py` |
| `ad/grad-graph.png` | `.venv/bin/python scripts/ad_graph.py` |
| `ad/scaling.png` | `.venv/bin/python scripts/ad_scaling.py` |
| `forward/run.json`, `forward/result.json` | `seismic --mode forward ...` |
| `forward/gathers.png` | `.venv/bin/python scripts/forward_figures.py` |
| `forward/wavefield.png`, `forward/echo.png` | `.venv/bin/python scripts/forward_figures.py` |
| `born/run.json`, `born/result.json` | `seismic --mode born ...` |
| `adjoint/run.json`, `adjoint/result.json` | `seismic --mode adjoint ...` |
| `adjoint/image.png` | `.venv/bin/python scripts/adjoint_figures.py` |
| `adjoint/wavefield.png` | `.venv/bin/python scripts/adjoint_figures.py` |
| `checkpoint-{1,3,5,10}/run.json`, `result.json`, `actions-<shot>.json` | `seismic --storage treeverse --checkpoints <b> ...` |
| `checkpoint-actions.png`, `checkpoint-work.png` | `.venv/bin/python scripts/checkpoint_figures.py` |
| `marmousi-born/run.json`, `result.json` | `seismic --experiment inputs/marmousi.json --mode born ...` |
| `marmousi-image/run.json`, `result.json`, `actions-<shot>.json` | `seismic --experiment inputs/marmousi.json --mode adjoint --storage treeverse --checkpoints 5 ...` |
| `marmousi.png` | `.venv/bin/python scripts/marmousi_figure.py` |

The zero-count audit is printed by `.venv/bin/python scripts/audit.py`; the
derivative checks are printed by `scripts/ad_modes.py` and `scripts/ad_scaling.py`;
the image and work checks are printed by the figure scripts themselves.

## Regeneration checklist

From a clean clone: fetch the inputs; install the toolchain, the crate and the
virtual environment; then `make reproduce`. At that point every tracked file
under `week5/` exists, the only untracked paths are the ignored `inputs/`,
`*.npy`, `seismic/target/` and `.venv/`, and every number quoted above can be
read line by line from the printed output of the scripts.

## What stays out of git

`week5/.gitignore` ignores `/inputs/`, `*.npy`, `/seismic/target/`, `/.venv/`,
`__pycache__/` and `.pytest_cache/`. None of those paths is tracked:

```bash
git ls-files week5 | grep -E '\.npy$|/inputs/|/target/|\.venv' || echo "no generated path tracked"
```

The exercise was committed locally and is **not pushed**, per the session
instruction.
