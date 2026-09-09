# Equilibrium Fluid CLI Design

## Purpose

Extend the existing `week2/md/` crate into a command-line molecular-dynamics
tool that prepares and records a two-dimensional Lennard-Jones equilibrium
fluid, independently checks the saved physics, and renders the trajectory as a
short video. This is the Week 2 Part 4 deliverable and must preserve every Part
1-3 behavior and test.

The default command must produce the exact raw-artifact contract required by
the learning sheet and pass both the Rust `md check` implementation and the
instructor-provided external Python checker.

## Scope and repository boundary

Part 4 includes:

- periodic minimum-image geometry;
- a potential-shifted Lennard-Jones cutoff;
- triangular-lattice initialization and deterministic Gaussian velocities;
- thermostatted equilibration followed by unthermostatted NVE production;
- `md run`, `md check`, and `md video` subcommands;
- JSON metadata and JSON-lines trajectory artifacts;
- energy, temperature, speed-shape, and radial-distribution analysis;
- `week2/Makefile` with `make reproduce`; and
- a generated `week2/fluid.mp4` for manual verification.

Part 5 features are out of scope: cell lists, naive/cells selection, heating
ramps, performance tables, scaling plots, profiles, GitHub Pages, and hot/cold
comparison videos.

The supplied Python checker and NumPy reference solver remain outside the
public exercise repository. They are independent development oracles, not
student deliverables. The repository contains the student's Rust
implementation, tests, Makefile, generated `fluid.mp4`, and documentation.
Generated `week2/artifacts/` data and temporary video frames remain ignored by
Git.

## Module architecture

### `simulation.rs`

`System` remains the sole owner of positions, velocities, and accelerations.
It gains an internal interaction model:

```rust
enum InteractionModel {
    PlainOpen,
    PeriodicShifted { box_size: [f64; 2], cutoff: f64 },
}
```

`System::new` continues to create the Part 3 open-boundary plain-LJ model with
the same signature and behavior. A new validated constructor creates a
periodic system. The existing `Integrator`, `Euler`, `VelocityVerlet`,
`advance`, and `run_dimer` interfaces remain intact. Both integrators use the
system's selected interaction model; velocity-Verlet is not duplicated for the
fluid.

The interaction model is matched once per force/energy calculation, outside
the pair loop. No trait object or per-pair dynamic dispatch is introduced.
Part 4 uses the all-pairs force calculation; Part 5 will introduce alternative
pair-search algorithms without changing the physical model.

The periodic system exposes shared-borrow accessors plus focused methods for
kinetic energy, thermodynamic temperature, removal of center-of-mass velocity,
and velocity rescaling. Position wrapping occurs during drift inside the
integrator and never changes velocity.

### `fluid.rs`

This module defines and validates `RunConfig`, constructs the triangular
lattice and deterministic velocities, runs equilibration and production, and
returns recorded metadata and frames. It owns experiment orchestration but not
pair-force formulas, artifact parsing, statistical checks, plotting, or CLI
argument parsing.

### `artifacts.rs`

This module defines Serde data structures for `run.json` and each
`traj.jsonl` frame. It writes atomically enough to avoid reporting success with
only one artifact present, reads one JSON object per nonempty trajectory line,
and returns descriptive errors for missing, malformed, non-finite, or
dimensionally inconsistent data.

### `analysis.rs`

This module consumes saved metadata and frames. It independently recomputes
shifted periodic potential energy and kinetic energy from raw positions and
velocities, evaluates the acceptance metrics, and calculates radial
distribution functions. It does not advance a simulation and does not trust
recorded energies as gate values.

The canonical scalar plain-LJ `physics::energy` remains shared, so the physical
formula has one implementation. Independence means the checker starts from
saved raw states and recomputes every total energy rather than consuming the
simulator's cached or logged totals. The external Python checker protects
against a common Rust formula error.

### `video.rs`

This module reads artifacts, computes recent-frame radial structure through
`analysis`, renders temporary PNG frames with Plotters, invokes the system
FFmpeg executable, validates that an MP4 was produced, and lets the temporary
directory clean itself up.

FFmpeg is an explicit runtime dependency for `md video`. If it is missing or
encoding fails, the command returns a concise nonzero error. The Rust crate
does not embed a separate video codec.

### `main.rs`

The binary uses Clap to parse the three subcommands and maps library errors to
human-readable stderr plus a nonzero process status. It contains no pair-force,
integration, statistical, or plotting formulas.

## Periodic physical model

For a coordinate difference `d` along a box side of length `L`, the minimum
image is

```text
d_min = d - L * round(d / L)
```

After every position update, periodic coordinates are wrapped with
`rem_euclid(L)` so every saved coordinate satisfies `0 <= x < Lx` and
`0 <= y < Ly`. Velocities are unchanged by wrapping.

The cutoff is fixed at `rc = 2.5`. For nearest-image distance `r`:

```text
U_cut(r) = physics::energy(r) - physics::energy(rc),  r < rc
U_cut(r) = 0,                                          r >= rc

F_cut(r) = physics::force(r),                          r < rc
F_cut(r) = 0,                                          r >= rc
```

The potential is continuous at the cutoff. The force is deliberately not
shifted and has a small jump there. Every unordered pair `i < j` is visited
once; its vector force is accumulated equally and oppositely, preserving zero
total internal force up to floating-point rounding. A periodic run requires
both box sides to be at least `2 * rc`.

## Lattice and run configuration

`n` must be a perfect square whose square root is even. With `nx = ny =
sqrt(n)` and density `rho`:

```text
a  = sqrt(2 / (sqrt(3) * rho))
h  = sqrt(3) * a / 2
Lx = nx * a
Ly = ny * h
x(i,j) = (i + 0.5 * (j mod 2)) * a
y(i,j) = j * h
```

The default configuration is:

```text
n = 100
rho = 0.8
temperature = 0.5
dt = 0.01
eq_steps = 2000
steps = 10000
sample_every = 50
seed = 2026
integrator = "velocity-verlet"
```

Validation rejects a non-square or odd-row atom count, non-finite or nonpositive
density/temperature/timestep, zero production or sampling counts, a sampling
interval larger than the production length, and any derived box incompatible
with the cutoff.

## Deterministic velocity preparation

Use a seeded ChaCha random-number generator with independent standard-normal
draws scaled by `sqrt(temperature)` for every x and y component. A named ChaCha
generator is used instead of an implementation-dependent default RNG so a
fixed crate version and seed reproduce the same initial state.

After drawing velocities:

1. subtract the mean x and y velocity;
2. compute `E_kin = 0.5 * sum_i |v_i|^2`;
3. compute `T_thermo = 2 * E_kin / (2 * n - 2)`; and
4. multiply every component by `sqrt(T_target / T_thermo)`.

The initial velocities are rescaled immediately. Equilibration then performs
2000 velocity-Verlet steps and rescales after steps 50, 100, ..., 2000.
Center-of-mass velocity is removed once more after equilibration. Production
performs 10000 velocity-Verlet steps with the thermostat off.

## Sampling and artifact contract

Production time starts at zero after equilibration. A frame is saved after
each production step divisible by `sample_every`, excluding step zero. Thus the
default run records exactly 200 frames at steps 50 through 10000.

`run.json` is one JSON object containing exactly the required public fields:

```text
n
rho
box = [Lx, Ly]
dt
temperature
eq_steps
steps
sample_every
seed
integrator = "velocity-verlet"
```

`traj.jsonl` contains one JSON object per frame:

```text
step
t = step * dt
pos = [[x, y]; n]
vel = [[vx, vy]; n]
E_pot
E_kin
```

`E_pot` uses the shifted periodic potential and `E_kin` uses the serialized
velocities. Serde JSON's round-trip representation retains sufficient `f64`
precision; the values are calculated from the same state that is serialized.
Every numeric value written must be finite.

## Command-line interface

The supported interface is:

```text
md run --n 100 --rho 0.8 --temperature 0.5 --dt 0.01 \
       --eq-steps 2000 --steps 10000 --sample-every 50 \
       --seed 2026 --out artifacts
md run --out artifacts
md check artifacts
md video artifacts --out artifacts/run.mp4
```

The two `run` commands are equivalent. `run` creates the output directory when
needed and replaces only its own `run.json` and `traj.jsonl`. `check` never
modifies artifacts. `video` replaces the explicitly named output file only
after FFmpeg successfully encodes a temporary result.

## Independent physics checks

`md check` mirrors the supplied contract and reports measured values next to
their limits.

### Schema, configuration, and chronology

The checker requires finite two-dimensional arrays of length `n`, wrapped
positions, nonnegative kinetic energies, at least 100 frames, the default
`n/rho/temperature/dt/integrator`, at least 2000 equilibration and 10000
production steps, `sample_every <= 100`, exactly
`floor(steps / sample_every)` frames, and the exact step/time sequence.

### Logged-energy consistency

For 25 evenly spaced frames, recompute shifted periodic potential energy and
kinetic energy from raw `pos/vel`. The maximum relative disagreement with the
logged fields, normalized by `max(1, abs(logged))`, must be below `1e-6`.

### Secular energy drift

Recompute total energy from raw state for every frame. With first-frame energy
`E0` and `k = max(1, floor(frame_count / 10))`:

```text
drift = abs(mean(last k energies) - mean(first k energies)) / abs(E0)
```

Require `drift < 2e-3`.

### Temperature

Pool all saved speeds and calculate

```text
T_speed = mean(v^2) / 2
```

Require `abs(T_speed - 0.5) < 0.05`.

### Speed shape

Use the two-dimensional Maxwell-Boltzmann/Rayleigh distribution at the measured
temperature:

```text
f(v) = (v / T_speed) * exp(-v^2 / (2 * T_speed))
F(v) = 1 - exp(-v^2 / (2 * T_speed))
```

Create 24 equal-probability bins. For `k = 0..23`, finite lower edges are

```text
b_k = sqrt(-2 * T_speed * ln(1 - k / 24))
```

and `b_24 = infinity`. With `M` speeds, expected count `M / 24`, and 22 degrees
of freedom:

```text
chi2_per_dof = sum_bins((observed - expected)^2 / expected) / 22
```

Require `chi2_per_dof < 2`.

All three physical conditions must pass. A malformed artifact is a controlled
data error, not a panic. Any failed gate causes nonzero exit. Successful output
ends with `PASS`.

## Radial distribution and video

For a frame, compute all minimum-image pair distances up to half the shorter
box side. Each unordered pair contributes two neighbors. For ring edges `r1`
and `r2`, normalize accumulated counts by

```text
frame_count * n * rho * pi * (r2^2 - r1^2)
```

to obtain `g(r)`. Video frames use the current frame plus up to the preceding
19 trajectory frames, so the curve is readable while still responding to
structural change. The radial range is split into 64 equal-width bins.

Each 960-by-480 video frame contains the periodic particle box on the left and
labeled `g(r)` axes on the right. One saved trajectory frame produces one video
frame. FFmpeg encodes H.264 with `yuv420p` at 20 frames per second, the `medium`
preset, CRF 30, and fast-start metadata. If that result exceeds 2 MB, the
command retries once at CRF 34; exceeding 2 MB after the retry is a controlled
error. No trajectory frames are dropped.

## Makefile and ignored outputs

From `week2/`, `make reproduce` runs the release binary with default flags and
writes to `week2/artifacts/`:

```make
.PHONY: reproduce

reproduce:
	cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts
```

`week2/.gitignore` continues to exclude `artifacts/` and `md/target/`. Temporary
video frames are created outside the repository and cleaned automatically.

## Test-first development

Before implementing Part 4 production code, commit tests that fail because the
fluid, artifact, analysis, and CLI interfaces do not yet exist. The red suite
must include the four required behaviors:

1. Periodic pair forces sum to zero within numerical tolerance.
2. The shifted potential approaches zero immediately inside `rc` and equals
   zero at and outside `rc`.
3. The default contract run satisfies drift `< 2e-3`, temperature error `<
   0.05`, and speed-shape `chi2/22 < 2`.
4. The built binary runs a shortened configuration and produces readable
   `run.json` and `traj.jsonl` with exact fields, dimensions, frame count, and
   chronology.

Additional focused tests cover lattice density and box dimensions,
minimum-image separation and wrapping, deterministic seeded initialization,
Rayleigh CDF/quantile behavior, and controlled rejection of malformed
artifacts. Tests use real implementations and temporary directories rather
than mocks.

The implementation proceeds in independently reviewable green stages:

1. periodic force model and state operations;
2. lattice, deterministic initialization, equilibration, and production;
3. artifact I/O and CLI `run`;
4. independent analysis and CLI `check`;
5. Makefile reproduction and external Python-checker validation; and
6. video rendering and `fluid.mp4`.

No physical threshold is weakened to make a run pass. If the contract fails,
diagnose initialization, thermostat timing, force sign, cutoff, wrapping, or
integration before changing any test.

## Acceptance

Part 4 is ready for the user's independent verification when:

- all earlier tests and all new unit/integration tests pass in release mode;
- `make reproduce` creates exactly the two required ignored artifacts;
- Rust `md check artifacts` prints measured drift, temperature, and speed-shape
  values within their bounds followed by `PASS`;
- the external supplied Python checker independently passes the same artifacts;
- `md video artifacts --out fluid.mp4` creates a video under 2 MB with one
  rendered frame per saved frame;
- visual inspection shows atoms moving within the periodic box and `g(r)`
  settling to a liquid-like first peak with a nearly flat tail;
- the README records commands that reproduce and inspect these outputs;
- an independent code review finds no unresolved critical or important issue;
  and
- Git history preserves the red tests before implementation and remains linear
  without rewriting or pushing until the user's manual checkpoint passes.
