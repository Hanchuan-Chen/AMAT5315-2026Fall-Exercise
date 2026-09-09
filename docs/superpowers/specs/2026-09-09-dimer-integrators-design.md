# Dimer Integrators Design

## Purpose

Extend the existing `week2/md/` crate with a two-atom, two-dimensional
molecular-dynamics experiment that compares forward Euler with
velocity-Verlet through one Rust trait. The experiment must isolate integration
error: it uses the plain Lennard-Jones potential from Part 2, open boundaries,
unit mass, and no cutoff, thermostat, or periodic wrapping.

## Scope

This design covers only the Week 2 Part 3 dimer experiment. Periodic
boundaries, a shifted cutoff, many-atom initialization, equilibration, command
line flags, trajectory files, and video rendering belong to Part 4 or later.

The fixed experiment starts two atoms at `[0.0, 0.0]` and `[1.2, 0.0]`, both
with zero velocity. Both integrators use `dt = 0.01`. The comparison run lasts
500 steps, and the long velocity-Verlet run lasts 5000 steps.

## State ownership and public interface

Create `week2/md/src/simulation.rs`. Its `System` struct is the sole owner of
three equally sized arrays:

```rust
positions: Vec<[f64; 2]>
velocities: Vec<[f64; 2]>
accelerations: Vec<[f64; 2]>
```

The intended public interface is:

```rust
pub struct System { /* private fields */ }

impl System {
    pub fn new(
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
    ) -> Self;
    pub fn positions(&self) -> &[[f64; 2]];
    pub fn velocities(&self) -> &[[f64; 2]];
    pub fn accelerations(&self) -> &[[f64; 2]];
    pub fn total_energy(&self) -> f64;
}

pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

pub struct Euler;
pub struct VelocityVerlet;

pub fn advance(method: &impl Integrator, system: &mut System, dt: f64);

pub struct EnergyTrace { /* private fields */ }

impl EnergyTrace {
    pub fn times(&self) -> &[f64];
    pub fn relative_errors(&self) -> &[f64];
}

pub fn run_dimer(
    method: &impl Integrator,
    steps: usize,
    dt: f64,
) -> EnergyTrace;
```

`System::new` asserts that positions and velocities have the same nonzero
length and computes the initial accelerations from the positions. Accessors
return shared borrows. Integrators receive one exclusive mutable borrow of the
system, so they can update the owned arrays without copying the whole state.

`week2/md/src/lib.rs` publicly exposes the new `simulation` module. The
existing `physics` module remains the single source of the scalar pair energy
and radial force.

## Pair forces and energy

For every unordered pair `i < j`, define

```text
d = positions[i] - positions[j]
r = |d|
F_i = force(r) * d / r
F_j = -F_i
```

The pair is visited once, and equal and opposite contributions are accumulated
into the acceleration array. Unit mass makes acceleration equal to force. The
calculation calls `physics::force(r)` and rejects overlapping atoms through the
existing positive-separation assertion.

Total energy is recomputed independently from the current state:

```text
E_kin = 0.5 * sum_i |v_i|^2
E_pot = sum_(i<j) physics::energy(|x_i - x_j|)
E = E_kin + E_pot
```

No recorded or cached energy is used in the acceptance tests.

## Integration algorithms

Forward Euler uses the old position, velocity, and acceleration for both state
updates:

```text
x_new = x_old + v_old * dt
v_new = v_old + a_old * dt
a_new = accelerations(x_new)
```

Velocity-Verlet performs a half kick, drift, one new force evaluation, and a
second half kick:

```text
v_half = v_old + 0.5 * a_old * dt
x_new = x_old + v_half * dt
a_new = accelerations(x_new)
v_new = v_half + 0.5 * a_new * dt
```

The new acceleration remains in `System` for the next step. After the initial
acceleration computed by `System::new`, each Verlet step performs exactly one
new force calculation.

`advance` is the sole driver entry point and delegates to the trait method.
`run_dimer` creates the standard initial state on every call and advances it
only through `advance`. Thus the only difference between an Euler trace and a
velocity-Verlet trace is the integrator value passed to the shared driver.

## Measurement

Before the first step, `run_dimer` records the initial energy as `E0`. After
each step `n = 1..steps`, it records

```text
t_n = n * dt
relative_error_n = (E_n - E0) / |E0|
```

The trace contains exactly `steps` times and `steps` relative errors. Invalid
programmer inputs are rejected by assertions with clear messages: state arrays
must have equal nonzero length, and `dt` must be finite and positive. A
recoverable error type is deferred until Part 4 introduces user-controlled CLI
input.

## Test-first development

Create `week2/md/tests/dimer.rs` before `simulation.rs` exists. The first
committed test run must fail to compile because the public simulation API is
missing. The integration tests exercise real crate behavior without mocks.

Tests cover these behaviors:

1. A two-atom `System` initialized at separation `1.2` has equal and opposite
   accelerations, and both accelerations point toward the other atom.
2. One test calls the same `run_dimer` function for `Euler` and
   `VelocityVerlet` with 500 steps and `dt = 0.01`. Velocity-Verlet's maximum
   absolute relative energy error is below `1e-3`; Euler's final relative
   energy error is above `0.5`.
3. A 5000-step velocity-Verlet trace at `dt = 0.01` has maximum absolute
   relative energy error below `1e-3`.
4. All existing greeting, well-depth, and force-derivative tests continue to
   pass.

After implementation, `cargo test --manifest-path week2/md/Cargo.toml` and
`cargo clippy --manifest-path week2/md/Cargo.toml --all-targets -- -D warnings`
must complete without failures or warnings.

## Figure and documentation

Create `week2/md/examples/dimer.rs`. It calls `run_dimer` rather than
reimplementing either integrator or the energy calculation. It writes
`week2/dimer.png` with two panels:

- Left: Euler and velocity-Verlet relative energy error for 500 steps,
  `0 <= t <= 5`.
- Right: velocity-Verlet alone for 5000 steps, `0 <= t <= 50`, with displayed
  errors multiplied by 1000.

Both curves and both axes are labeled, and the figure remains legible at its
native resolution. `week2/README.md` records the exact command, run from
`week2/`, that regenerates the image.

## Acceptance

Part 3 is ready for the user's independent verification when:

- the Git history contains a committed red test before its implementation;
- all tests pass through the shared trait-driven experiment;
- the 500-step and 5000-step numerical bounds above hold;
- `dimer.png` visually shows Euler's secular growth and Verlet's bounded,
  oscillatory error; and
- the user can identify `System` as the array owner, shared borrows in its
  accessors, the mutable borrow in `Integrator::step`, and the trait used by the
  common driver.
