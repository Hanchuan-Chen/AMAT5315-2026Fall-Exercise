# Week 3 Part 1 Metropolis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a seeded random-site Metropolis Ising sampler with the required `relax` CLI.

**Architecture:** A library separates periodic lattice physics, Metropolis dynamics, and CLI formatting. The executable delegates to the library so tests exercise the same code used by the command.

**Tech Stack:** Rust 2024, Cargo, `rand = "0.9"`, `clap = "4"`, `assert_cmd = "2"`.

**Spec:** `docs/superpowers/specs/2026-09-10-week3-part1-metropolis-design.md`

## Global Constraints

- Use the periodic two-dimensional Ising model with `J = 1` and no field.
- Use random-site Metropolis; one sweep is exactly `L * L` proposals.
- Seed every stochastic path from the user-provided `u64` seed.
- Keep all Week 3 source under `week3/`.
- Preserve red and green states as separate chronological commits.

---

### Task 1: Red physics and reproducibility tests

**Files:**
- Create: `week3/Cargo.toml`
- Create: `week3/src/lib.rs`
- Create: `week3/tests/physics.rs`
- Create: `week3/tests/reproducibility.rs`

**Interfaces:**
- Consumes: no Week 3 code.
- Produces: required public interfaces `Lattice`, `AcceptanceTable`, `sweep`, `relax`, `RelaxConfig`, and `RelaxResult`.

- [ ] **Step 1: Create the minimal crate manifest and test files**

The physics test constructs seeded random spin vectors, calls
`Lattice::delta_energy_at`, flips through `Lattice::flipped`, and compares the
reported delta to `energy_after - energy_before`. It records the observed deltas
and asserts equality with `{-8,-4,0,4,8}`. The reproducibility test calls `relax`
twice with seed 2026 and once with 2027 and compares `RelaxResult::render()`.

- [ ] **Step 2: Verify the tests fail for missing production interfaces**

Run: `cargo test --release`

Expected: compilation fails because the tested types and functions do not exist.

- [ ] **Step 3: Commit the red state**

```bash
git add week3
git commit -m "test: specify week 3 Ising sampler"
```

### Task 2: Green lattice and Metropolis core

**Files:**
- Create: `week3/src/lattice.rs`
- Create: `week3/src/metropolis.rs`
- Modify: `week3/src/lib.rs`

**Interfaces:**
- Consumes: the interfaces fixed by Task 1.
- Produces: `Lattice::{all_up,from_spins,random,energy,magnetization,delta_energy_at,flip,flipped,render}`; `AcceptanceTable::new`; `sweep`; `relax`.

- [ ] **Step 1: Implement periodic lattice physics minimally**

Store `i8` spins row-major. Count each energy bond once using right and down
neighbours. Compute the local neighbour sum with modular indexing.

- [ ] **Step 2: Implement the five-entry acceptance table and seeded run**

Map `delta_E` to index `(delta_E + 8) / 4`. Accept non-positive changes directly;
otherwise compare one uniform `f64` draw with the table entry. Count proposals and
acceptances exactly.

- [ ] **Step 3: Run the release suite**

Run: `cargo test --release`

Expected: all physics and reproducibility tests pass.

- [ ] **Step 4: Commit the green core**

```bash
git add week3/src
git commit -m "feat: implement seeded Metropolis sampler"
```

### Task 3: Relax command and numerical verification

**Files:**
- Create: `week3/src/cli.rs`
- Create: `week3/src/main.rs`
- Create: `week3/tests/cli.rs`
- Create: `week3/README.md`

**Interfaces:**
- Consumes: `relax(RelaxConfig) -> RelaxResult`.
- Produces: the specified `ising relax` command and stable human-readable output.

- [ ] **Step 1: Write CLI tests first**

Assert that a small valid command exits successfully with one summary line and
exactly `L` lattice lines. Assert invalid `L`, temperature, and measurement count
exit unsuccessfully with useful diagnostics.

- [ ] **Step 2: Run the CLI tests and observe the missing binary failure**

Run: `cargo test --release --test cli`

Expected: failure because the CLI has not been implemented.

- [ ] **Step 3: Implement and validate the thin CLI**

Use `clap` derive for `relax` and validate numerical constraints before calling
the library. Print `RelaxResult::render()` without nondeterministic metadata.

- [ ] **Step 4: Run all tests and the two required numerical checks**

```bash
cargo test --release
cargo run --release -- relax --l 64 --t 1.8 --sweeps 2000 --measure 2000 --seed 2026
cargo run --release -- relax --l 64 --t 3.0 --sweeps 2000 --measure 2000 --seed 2026
```

Expected: all tests pass; cold `mean_abs_m` is 0.95-0.965 with about 4% acceptance;
hot `mean_abs_m` is below 0.06 with about 46% acceptance.

- [ ] **Step 5: Commit Part 1**

```bash
git add week3
git commit -m "feat: add Ising relax command"
```
