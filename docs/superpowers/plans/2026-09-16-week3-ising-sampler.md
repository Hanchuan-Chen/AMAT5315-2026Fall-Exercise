# Week 3 Part 1 Ising Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ising` in `week3/`, a Rust CLI that samples the two-dimensional Ising model along one ascending temperature ramp with the `metropolis` update, prints the contract's tab-separated table, writes `run.json`, `series.jsonl`, and `spins.jsonl` exactly as `week3/ising.design.toml` fixes them, and ships the Part 1 evidence produced by running it.

**Architecture:** One binary, five small modules. `lattice` owns spins and the energy of a flip; `metropolis` owns the accept rule and one sweep; `ramp` owns the temperature grid, the single random stream, the global sweep counter, and the warm-start loop; `artifacts` owns the exact file formats; `cli`/`main` own argument parsing, validation, the stdout table, and exit codes. Integration tests drive both the library and the installed binary.

**Tech Stack:** Rust 1.98.1, Cargo, Clap 4 (derive), rand 0.9 with rand_chacha 0.9 (`ChaCha8Rng`), serde/serde_json 1, tempfile 3 plus Cargo's `CARGO_BIN_EXE_ising` for the binary tests, Python 3.12 in `week3/.venv` via `uv` with numpy and matplotlib for `scripts/boltzmann.py`, and puppeteer-core driving headless Google Chrome for the viewer proofs (`window.composeProofPNG()`).

**Spec:** docs/superpowers/specs/2026-09-16-week3-ising-sampler-design.md

## Global Constraints

- Work only in `/Users/eureka/Documents/Codex/2026-09-02/ban/AMAT5315-2026Fall-Exercise` on `main`; check `git status` before every commit.
- Preserve chronological history: never amend, squash, rebase, reset, or force-push; never push to `origin` at all in this phase.
- Keep every new path under `week3/` or `docs/superpowers/`; `/target/`, `/artifacts/`, `/runs/`, `/.venv/` under `week3/` stay out of git through `week3/.gitignore`.
- Copy `week3/ising.design.toml` byte-for-byte from `/tmp/w3site.HE6C/week3/ising.design.toml`; `diff` must print nothing.
- No unstated defaults: every flag is required except `--every`, whose documented default is `0`.
- Physics: `E = -sum_<ij> s_i s_j`, `E` per site `E/L^2`, `dE = 2 s_i (s_1+..+s_4)`, `A = min(1, exp(-dE/T))`, one sweep is `L*L` proposals with replacement, periodic boundaries, `J = 1`.
- One `ChaCha8Rng` seeded from `--seed` per run, carried through the ramp; same arguments must reproduce `series.jsonl` byte for byte.
- `series.jsonl` `sweep` restarts at 1 at each temperature; `spins.jsonl` `sweep` is the global counter including discard.
- `spins.jsonl` exists only when `--every > 0`.
- Each committed file stays below 5 MB.
- Red tests are committed before production code; each green stage is independently reviewable; `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --release` must all pass before evidence work.
- Evidence numbers come from actually running the binary; never write a measured value by hand.

## File map

- Create `week3/.gitignore`: ignore `/target/`, `/artifacts/`, `/runs/`, `/.venv/`.
- Create `week3/ising.design.toml`: byte-identical contract.
- Create `week3/Cargo.toml`, `week3/Cargo.lock`: crate `ising`, edition 2024, binary `ising`.
- Create `week3/src/lattice.rs`: spins, periodic neighbour sum, `dE`, total and per-site energy, magnetization, flip.
- Create `week3/src/metropolis.rs`: `acceptance_probability`, `proposal_accepted`, `sweep`, `SweepStats`.
- Create `week3/src/ramp.rs`: `RunConfig` validation, `temperature_grid`, `run_ramp`, `TemperatureResult`.
- Create `week3/src/artifacts.rs`: `Recorder` writing `run.json`, `series.jsonl`, `spins.jsonl`.
- Create `week3/src/cli.rs`: Clap struct, `Update` enum, `RunConfig` conversion.
- Create `week3/src/lib.rs`, `week3/src/main.rs`: module wiring, header, per-temperature rows, exit codes.
- Create `week3/tests/physics.rs`: energy vs brute force, the five `dE` values, accept-rule correctness.
- Create `week3/tests/contract.rs`: stdout table, `run.json`, `series.jsonl` schema and rounding, sweep restart, `spins.jsonl` schema, frames only when `every > 0`, cumulative sweep, temperature-grid inclusion.
- Create `week3/tests/reproducibility.rs`: same seed identical bytes, different seed differs.
- Create `week3/scripts/boltzmann.py`: energy histograms and their log ratio.
- Create `week3/evidence/boltzmann.png`, `week3/evidence/viewer-T1.8.png`, `viewer-T2.3.png`, `viewer-T3.0.png`.
- Create `week3/spins.jsonl`: committed ramp recording.
- Create `week3/scripts/capture_viewer.mjs`: headless-Chrome capture of `window.composeProofPNG()`.
- Create `week3/scripts/serve_and_capture.sh` or equivalent small helper if the capture needs a served folder.

## Task 1: Contract file (commit)

**Files:** create `week3/ising.design.toml`, `week3/.gitignore`.

- [ ] `mkdir -p week3` and copy `/tmp/w3site.HE6C/week3/ising.design.toml` to `week3/ising.design.toml`.
- [ ] `diff /tmp/w3site.HE6C/week3/ising.design.toml week3/ising.design.toml` prints nothing.
- [ ] Write `week3/.gitignore`.
- [ ] Commit `docs: add the week 3 ising contract and its design`.

## Task 2: Plan doc (commit)

**Files:** create `docs/superpowers/plans/2026-09-16-week3-ising-sampler.md`.

- [ ] Write this plan.
- [ ] Commit `docs: plan the week 3 ising sampler`.

## Task 3: RED tests (commit)

**Files:** create `week3/Cargo.toml`, `week3/Cargo.lock`, `week3/src/lib.rs`, `week3/src/{lattice,metropolis,ramp,artifacts,cli,main}.rs` skeletons, `week3/tests/{physics,contract,reproducibility}.rs`.

- [ ] Add the manifest and a compiling skeleton: every public function exists with the signature the tests call and a `todo!()` body.
- [ ] Write the physics tests: energy against an independent brute-force bond loop on `L = 2, 3, 8` and random configurations; the five `dE` values realized by explicit 3x3 patterns; `dE` equal to `energy_after - energy_before`; acceptance probabilities `1, 1, 1, exp(-4/T), exp(-8/T)`; a boundary case per value; a statistical check that many `dE = +4` proposals at `T = 2.3` accept at `exp(-4/2.3)` within tolerance; a flip twice restores spins and energy.
- [ ] Write the contract tests, driving the built binary through `env!("CARGO_BIN_EXE_ising")`: header exactly `T\tmean|M|\tacceptance` then one tab-separated row per temperature; `run.json` fields and values; `series.jsonl` keys, ramp order, `sweep` restart at 1, `M`/`E` six-decimal fields, `M` equal to the frame's mean spin when both are recorded; `spins.jsonl` keys, `L^2` entries of `+1/-1`, only created when `every > 0`, and the exact cumulative global `sweep` list for a small ramp; grid inclusion for `1.5..3.5` step `0.05` (41 values ending at 3.5) and `1.5..3.55` step `0.1` (21 values ending at 3.5), plus a unit test of `temperature_grid` for the `t_to` rule.
- [ ] Write the reproducibility tests: two runs with seed 2026 give identical `series.jsonl` bytes; seed 2027 differs; different `--l` gives different rows.
- [ ] Run `cargo test` and record the expected failures; commit `test: specify the ising sampler contract`.

## Task 4: GREEN lattice and Metropolis (commit)

**Files:** modify `week3/src/lattice.rs`, `week3/src/metropolis.rs`, `week3/src/lib.rs`.

- [ ] Implement the periodic lattice: row-major `Vec<i8>`, wrapping index helper, `neighbor_sum`, `delta_energy = 2*s*(sum)`, `energy` over `2*L^2` bonds, `energy_per_site`, `magnetization`, `flip`.
- [ ] Implement `acceptance_probability`, `proposal_accepted`, and `sweep` with `rng.random_range(0..L*L)` and no uniform draw when `dE <= 0`.
- [ ] `cargo test --test physics` passes; commit `feat: implement the periodic Ising lattice and Metropolis sweep`.

## Task 5: GREEN ramp, artifacts, CLI (commit)

**Files:** modify `week3/src/ramp.rs`, `week3/src/artifacts.rs`, `week3/src/cli.rs`, `week3/src/main.rs`.

- [ ] Implement `temperature_grid` with nine-decimal rounding and the `t_to` inclusion rule; return a usage error for an empty grid.
- [ ] Implement the ramp: all-up start, warm start between temperatures, discard then measure, per-temperature accepted/proposal counters, global sweep counter, mean of `|m|`.
- [ ] Implement `Recorder`: `run.json` with exactly the contract's fields; `series.jsonl` rows `{"L":..,"T":..,"sweep":..,"M":%.6,"E":%.6}`; `spins.jsonl` rows with `m` at four decimals and a compact integer array, only when `every > 0`.
- [ ] Implement the CLI: required flags, `every` default 0, validation, `wolff` accepted by the parser and rejected at runtime with exit code 1.
- [ ] `cargo test --release` passes, `cargo clippy --all-targets --all-features -- -D warnings` is clean, `cargo fmt --check` is clean; commit `feat: drive the temperature ramp and write the ising artifacts`.

## Task 6: Part 1 verification evidence (commits)

**Files:** create `week3/scripts/boltzmann.py`, `week3/evidence/boltzmann.png`, `week3/spins.jsonl`, `week3/evidence/viewer-*.png`, `week3/scripts/capture_viewer.mjs`.

- [ ] `cargo install --path . --quiet`; `ising --version` and a real run prove the binary is on `PATH`.
- [ ] Cold/hot runs into `runs/T1.8`, `runs/T3.0`; record stdout and compare with the thresholds.
- [ ] `T = 3.1` run into `runs/T3.1`; `uv venv .venv && uv pip install numpy matplotlib`; write and run `scripts/boltzmann.py`; save `evidence/boltzmann.png`; report the fitted intercept, the residual spread where bins hold more than 20 sweeps, and the predicted slope.
- [ ] Repeatability: seeds 2026 into `runs/a` and `runs/b`, `diff` prints nothing and `echo IDENTICAL`; seed 2027 into `runs/c` differs.
- [ ] The contract's usage line into `runs/ramp`, then copy `runs/ramp/spins.jsonl` to `week3/spins.jsonl` and commit it with its size.
- [ ] Commit the Boltzmann script and chart, and the recording.
- [ ] Serve `week3/` over localhost, load the recording in the published viewer, capture `window.composeProofPNG()` at `T = 1.8, 2.3, 3.0` with headless Chrome, save the three evidence PNGs, and commit them.
- [ ] Final gate: `cargo clippy --all-targets --all-features -- -D warnings` clean and `cargo test --release` green on the committed tree.

## Self-check before reporting

- [ ] `diff` of the contract is empty and the md5 matches the reference.
- [ ] Every commit is conventional, chronological, and unpushed.
- [ ] `git ls-files week3 | xargs -I{} du -k {}` shows every committed file below 5 MB.
- [ ] The reported numbers are copied from the command output in this session, not from the sheet.
