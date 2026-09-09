# Dimer Integrators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trait-driven open-boundary Lennard-Jones dimer experiment that demonstrates forward Euler energy drift and bounded velocity-Verlet energy error.

**Architecture:** `System` owns position, velocity, and acceleration arrays in a focused simulation module. A single generic driver delegates each step to either `Euler` or `VelocityVerlet`, while one shared dimer runner independently recomputes total energy after every step.

**Tech Stack:** Rust 1.98.1, Cargo, the existing `md::physics` module, Plotters 0.3.7, Rust integration tests, Clippy.

**Spec:** `docs/superpowers/specs/2026-09-09-dimer-integrators-design.md`

## Global Constraints

- Work in the existing `AMAT5315-2026Fall-Exercise` repository on `main`; do not rewrite published history.
- Keep all coursework implementation under `week2/` and all design/plan documents under `docs/superpowers/`.
- Use two atoms in two dimensions with mass `m = 1`, positions `[0.0, 0.0]` and `[1.2, 0.0]`, and zero initial velocities.
- Use the plain Part 2 Lennard-Jones energy and force with open boundaries: no cutoff, periodic wrapping, or thermostat.
- Use `dt = 0.01`; compare both integrators for 500 steps and velocity-Verlet alone for 5000 steps.
- Require velocity-Verlet maximum absolute relative energy error `< 1e-3`, Euler final relative energy error `> 0.5`, and long-run velocity-Verlet maximum absolute relative error `< 1e-3`.
- Write and commit failing tests before production code, keep the tests unchanged while making them pass, and preserve separate red and green commits.
- Keep all earlier Week 1 and Week 2 tests passing and keep Clippy free of warnings.

---

### Task 1: Commit the failing public-contract tests

**Files:**

- Create: `week2/md/tests/dimer.rs`
- Test: `week2/md/tests/dimer.rs`

**Interfaces:**

- Consumes: the approved design and the existing `md` crate.
- Produces: executable expectations for `System`, `Euler`, `VelocityVerlet`, and `run_dimer` before those symbols exist.

- [ ] **Step 1: Write the failing integration tests**

Create `week2/md/tests/dimer.rs` with exactly these behavior tests:

```rust
use md::simulation::{run_dimer, Euler, System, VelocityVerlet};

const DT: f64 = 0.01;

fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

#[test]
fn dimer_accelerations_are_equal_opposite_and_attractive() {
    let system = System::new(
        vec![[0.0, 0.0], [1.2, 0.0]],
        vec![[0.0, 0.0], [0.0, 0.0]],
    );
    let acceleration = system.accelerations();

    assert!(acceleration[0][0] > 0.0);
    assert!(acceleration[1][0] < 0.0);
    for axis in 0..2 {
        assert!((acceleration[0][axis] + acceleration[1][axis]).abs() < 1.0e-12);
    }
}

#[test]
fn same_dimer_run_distinguishes_euler_from_verlet() {
    let euler = run_dimer(&Euler, 500, DT);
    let verlet = run_dimer(&VelocityVerlet, 500, DT);

    assert_eq!(euler.times().len(), 500);
    assert_eq!(verlet.times().len(), 500);
    assert!(max_abs(verlet.relative_errors()) < 1.0e-3);
    assert!(*euler.relative_errors().last().unwrap() > 0.5);
}

#[test]
fn verlet_error_stays_bounded_for_ten_times_longer() {
    let trace = run_dimer(&VelocityVerlet, 5000, DT);

    assert_eq!(trace.times().len(), 5000);
    assert!(max_abs(trace.relative_errors()) < 1.0e-3);
}
```

Mutation check: reversing the vector-force sign breaks the first and energy tests; routing Euler through Verlet breaks the Euler bound; dropping either Verlet half-kick breaks the Verlet bounds.

- [ ] **Step 2: Verify the intended red state**

Run from the repository root:

```bash
cargo test --manifest-path week2/md/Cargo.toml --test dimer
```

Expected: compilation fails with an unresolved import for `md::simulation`. This is the intended failure because the tested public API does not exist yet, not because of a typo in the test.

- [ ] **Step 3: Commit the red state**

```bash
git add week2/md/tests/dimer.rs
git commit -m "test: add failing dimer integrator checks"
```

Record the commit hash and the failing command output before continuing.

---

### Task 2: Implement the shared state, force accumulation, and integrators

**Files:**

- Create: `week2/md/src/simulation.rs`
- Modify: `week2/md/src/lib.rs`
- Test: `week2/md/tests/dimer.rs`

**Interfaces:**

- Consumes: `md::physics::{energy, force}` and the tests from Task 1.
- Produces: `System`, `Integrator`, `Euler`, `VelocityVerlet`, `advance`, `EnergyTrace`, and `run_dimer` with the signatures fixed by the design.

- [ ] **Step 1: Expose the new module**

Add this line to `week2/md/src/lib.rs` next to the existing physics module:

```rust
pub mod simulation;
```

- [ ] **Step 2: Implement the minimal simulation module**

Create `week2/md/src/simulation.rs`:

```rust
use crate::physics::{energy, force};

pub struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>,
}

impl System {
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self {
        assert!(!positions.is_empty(), "system must contain at least one atom");
        assert_eq!(
            positions.len(),
            velocities.len(),
            "positions and velocities must have equal lengths"
        );
        let accelerations = accelerations(&positions);
        Self {
            positions,
            velocities,
            accelerations,
        }
    }

    pub fn positions(&self) -> &[[f64; 2]] {
        &self.positions
    }

    pub fn velocities(&self) -> &[[f64; 2]] {
        &self.velocities
    }

    pub fn accelerations(&self) -> &[[f64; 2]] {
        &self.accelerations
    }

    pub fn total_energy(&self) -> f64 {
        let kinetic = self
            .velocities
            .iter()
            .map(|velocity| 0.5 * (velocity[0].powi(2) + velocity[1].powi(2)))
            .sum::<f64>();
        let mut potential = 0.0;
        for i in 0..self.positions.len() {
            for j in (i + 1)..self.positions.len() {
                let dx = self.positions[i][0] - self.positions[j][0];
                let dy = self.positions[i][1] - self.positions[j][1];
                potential += energy(dx.hypot(dy));
            }
        }
        kinetic + potential
    }
}

fn accelerations(positions: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut result = vec![[0.0, 0.0]; positions.len()];
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let dx = positions[i][0] - positions[j][0];
            let dy = positions[i][1] - positions[j][1];
            let r = dx.hypot(dy);
            let scale = force(r) / r;
            let pair = [scale * dx, scale * dy];
            for axis in 0..2 {
                result[i][axis] += pair[axis];
                result[j][axis] -= pair[axis];
            }
        }
    }
    result
}

fn require_positive_dt(dt: f64) {
    assert!(dt.is_finite() && dt > 0.0, "dt must be finite and positive");
}

pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

pub struct Euler;

impl Integrator for Euler {
    fn step(&self, system: &mut System, dt: f64) {
        require_positive_dt(dt);
        for atom in 0..system.positions.len() {
            for axis in 0..2 {
                system.positions[atom][axis] += dt * system.velocities[atom][axis];
                system.velocities[atom][axis] += dt * system.accelerations[atom][axis];
            }
        }
        system.accelerations = accelerations(&system.positions);
    }
}

pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) {
        require_positive_dt(dt);
        for atom in 0..system.positions.len() {
            for axis in 0..2 {
                system.velocities[atom][axis] +=
                    0.5 * dt * system.accelerations[atom][axis];
                system.positions[atom][axis] += dt * system.velocities[atom][axis];
            }
        }
        let new_accelerations = accelerations(&system.positions);
        for (velocity, acceleration) in system.velocities.iter_mut().zip(&new_accelerations) {
            for axis in 0..2 {
                velocity[axis] += 0.5 * dt * acceleration[axis];
            }
        }
        system.accelerations = new_accelerations;
    }
}

pub fn advance(method: &impl Integrator, system: &mut System, dt: f64) {
    method.step(system, dt);
}

pub struct EnergyTrace {
    times: Vec<f64>,
    relative_errors: Vec<f64>,
}

impl EnergyTrace {
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    pub fn relative_errors(&self) -> &[f64] {
        &self.relative_errors
    }
}

pub fn run_dimer(method: &impl Integrator, steps: usize, dt: f64) -> EnergyTrace {
    require_positive_dt(dt);
    let mut system = System::new(
        vec![[0.0, 0.0], [1.2, 0.0]],
        vec![[0.0, 0.0], [0.0, 0.0]],
    );
    let initial_energy = system.total_energy();
    let mut times = Vec::with_capacity(steps);
    let mut relative_errors = Vec::with_capacity(steps);

    for step in 1..=steps {
        advance(method, &mut system, dt);
        times.push(step as f64 * dt);
        relative_errors.push(
            (system.total_energy() - initial_energy) / initial_energy.abs(),
        );
    }

    EnergyTrace {
        times,
        relative_errors,
    }
}
```

- [ ] **Step 3: Verify the green state**

Run:

```bash
cargo fmt --manifest-path week2/md/Cargo.toml
cargo test --manifest-path week2/md/Cargo.toml
cargo clippy --manifest-path week2/md/Cargo.toml --all-targets -- -D warnings
```

Expected: the three dimer integration tests and all three existing library tests pass; Clippy finishes without warnings. If a numerical bound fails, inspect the update order and vector-force sign without changing the approved test thresholds.

- [ ] **Step 4: Commit the green implementation**

```bash
git add week2/md/src/lib.rs week2/md/src/simulation.rs
git commit -m "feat: add trait-driven dimer integrators"
```

---

### Task 3: Generate and document the dimer energy-error figure

**Files:**

- Create: `week2/md/examples/dimer.rs`
- Create: `week2/dimer.png`
- Modify: `week2/README.md`

**Interfaces:**

- Consumes: `run_dimer`, `Euler`, `VelocityVerlet`, and Plotters 0.3.7 already present as a development dependency.
- Produces: a two-panel PNG and its exact reproduction command; no duplicate physics or integration formula.

- [ ] **Step 1: Add the plotting example**

Create `week2/md/examples/dimer.rs`:

```rust
use std::env;
use std::error::Error;

use md::simulation::{run_dimer, Euler, VelocityVerlet};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;

type Chart<'a> = ChartContext<
    'a,
    BitMapBackend<'a>,
    Cartesian2d<RangedCoordf64, RangedCoordf64>,
>;

fn draw_curve(
    chart: &mut Chart<'_>,
    times: &[f64],
    errors: &[f64],
    scale: f64,
    color: RGBColor,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    chart
        .draw_series(LineSeries::new(
            times.iter().copied().zip(errors.iter().map(|error| scale * error)),
            color.stroke_width(2),
        ))?
        .label(label)
        .legend(move |(x, y)| PathElement::new([(x, y), (x + 28, y)], color.stroke_width(2)));
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let output = env::args()
        .nth(1)
        .unwrap_or_else(|| "../dimer.png".to_owned());
    let euler = run_dimer(&Euler, 500, 0.01);
    let verlet = run_dimer(&VelocityVerlet, 500, 0.01);
    let verlet_long = run_dimer(&VelocityVerlet, 5000, 0.01);

    let root = BitMapBackend::new(&output, (1200, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 2));

    let mut left = ChartBuilder::on(&panels[0])
        .caption("First 500 steps", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(0.0..5.0, -0.1..2.2)?;
    left.configure_mesh()
        .x_desc("time t")
        .y_desc("(E(t) - E0) / |E0|")
        .draw()?;
    draw_curve(
        &mut left,
        euler.times(),
        euler.relative_errors(),
        1.0,
        RGBColor(190, 45, 45),
        "forward Euler",
    )?;
    draw_curve(
        &mut left,
        verlet.times(),
        verlet.relative_errors(),
        1.0,
        RGBColor(35, 90, 175),
        "velocity-Verlet",
    )?;
    left.configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    let long_scaled: Vec<f64> = verlet_long
        .relative_errors()
        .iter()
        .map(|error| 1000.0 * error)
        .collect();
    let limit = long_scaled
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(0.5)
        * 1.1;
    let mut right = ChartBuilder::on(&panels[1])
        .caption("Velocity-Verlet: 5000 steps", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(0.0..50.0, -limit..limit)?;
    right.configure_mesh()
        .x_desc("time t")
        .y_desc("relative energy error x 1000")
        .draw()?;
    draw_curve(
        &mut right,
        verlet_long.times(),
        verlet_long.relative_errors(),
        1000.0,
        RGBColor(35, 90, 175),
        "velocity-Verlet",
    )?;
    right.configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("wrote {output}");
    Ok(())
}
```

- [ ] **Step 2: Generate the figure**

Run from `week2/`:

```bash
cargo run --manifest-path md/Cargo.toml --release --example dimer -- dimer.png
```

Expected: `week2/dimer.png` is a 1200 by 520 PNG. The left panel shows Euler growing above `0.5` while velocity-Verlet stays close to zero; the right panel shows a bounded oscillatory velocity-Verlet error over `t = 50`.

- [ ] **Step 3: Record the command and interpretation**

Append this section to `week2/README.md`:

````markdown
## Dimer integration

From `week2/`, reproduce `dimer.png` with:

```bash
cargo run --manifest-path md/Cargo.toml --release --example dimer -- dimer.png
```

Both curves start from the same two stationary atoms at separation `1.2` and
use `dt = 0.01` through the same `Integrator` trait driver. Forward Euler shows
secular energy growth, while velocity-Verlet's error remains bounded and
oscillatory over the ten-times-longer run.
````

- [ ] **Step 4: Re-run all checks and inspect the image**

Run:

```bash
cargo fmt --manifest-path week2/md/Cargo.toml -- --check
cargo test --manifest-path week2/md/Cargo.toml
cargo clippy --manifest-path week2/md/Cargo.toml --all-targets -- -D warnings
file week2/dimer.png
```

Open `week2/dimer.png` and verify labels, legend, axes, and unclipped curves before committing.

- [ ] **Step 5: Commit the figure and reproduction instructions**

```bash
git add week2/md/examples/dimer.rs week2/dimer.png week2/README.md
git commit -m "docs: add dimer energy comparison"
```

---

### Task 4: Review the Part 3 implementation and prepare the user checkpoint

**Files:**

- Inspect: `docs/superpowers/specs/2026-09-09-dimer-integrators-design.md`
- Inspect: `docs/superpowers/plans/2026-09-09-dimer-integrators.md`
- Inspect: `week2/md/src/simulation.rs`
- Inspect: `week2/md/tests/dimer.rs`
- Inspect: `week2/md/examples/dimer.rs`
- Modify only if review identifies a concrete defect, with a failing regression test before any production fix.

**Interfaces:**

- Consumes: the full Part 3 Git range beginning after the plan commit.
- Produces: reviewed, passing work plus an exact manual verification handoff for the user.

- [ ] **Step 1: Record the review range**

```bash
git log --oneline --decorate -12
git status --short --branch
```

Use the plan commit as the base and the figure commit as the head.

- [ ] **Step 2: Review against the approved spec**

Check the range for vector-force sign, old-state Euler updates, both Verlet half-kicks, exactly one new Verlet force calculation per step, independent total-energy recomputation, identical dimer initialization, unchanged thresholds, test quality, and README reproducibility.

If a production defect is found, add a minimal failing regression test, verify the intended failure, implement only the fix, re-run all checks, and commit the test and fix without amending earlier commits.

- [ ] **Step 3: Run the final automated verification**

```bash
cargo test --manifest-path week2/md/Cargo.toml
cargo clippy --manifest-path week2/md/Cargo.toml --all-targets -- -D warnings
git diff --check
git status --short --branch
```

Expected: six tests pass, Clippy is clean, there are no whitespace errors, the worktree is clean, and `main` remains ahead of `origin/main` without rewritten history.

- [ ] **Step 4: Hand off the required independent check**

Ask the user to run from `week2/` in a second terminal:

```bash
cargo test --manifest-path md/Cargo.toml
cargo run --manifest-path md/Cargo.toml --release --example dimer -- dimer.png
open dimer.png
```

The user confirms both integrators are exercised by the same test driver, Euler exceeds the required final error, velocity-Verlet stays bounded, and the figure matches the reference shape before Part 4 starts.
