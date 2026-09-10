# Week 3 Temperature Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the raw two-size temperature sweep, plots, and a checked critical-temperature estimate.

**Architecture:** Stream raw samples from the existing sampler, then read them into grouped summaries for plotting and peak fitting. Keep generation and analysis independent so the course gate can distrust reported summaries and recompute the raw physics.

**Tech Stack:** Rust 2024, Serde JSON Lines, Plotters PNG backend, Make.

**Spec:** `docs/superpowers/specs/2026-09-10-week3-temperature-sweep-design.md`

## Global Constraints

- Defaults must produce 2,740,000 ordered rows and the exact artifact field names.
- `M` and `E` are per-site values formatted to six decimals.
- Every stochastic path is seeded and generated artifacts remain ignored.

---

### Task 1: Raw sweep contract

**Files:** Create `week3/tests/temperature_sweep.rs` and `week3/src/temperature_sweep.rs`; modify `week3/src/lib.rs` and `week3/src/cli.rs`.

**Interfaces:** Produce `TemperatureSweepConfig`, `course_temperature_grid`, `write_temperature_sweep`, and CLI `sweep`.

- [ ] Write small-run tests for the 27-point default grid, exact run keys, row keys/order/count, six decimals, and reproducibility; run and commit the expected missing-interface failure.
- [ ] Implement streaming generation and CLI validation; run all release tests and commit green.

### Task 2: Physics summaries and plots

**Files:** Create `week3/tests/analysis.rs`, `week3/src/analysis.rs`, and `week3/src/plot.rs`; modify `week3/src/lib.rs` and `week3/src/cli.rs`.

**Interfaces:** Produce grouped observables, five-point quadratic peak fits, critical extrapolation, PNG plots, and CLI `plot`.

- [ ] Write mathematical fixture tests and observe the missing-interface failure; commit red.
- [ ] Implement the reader, formulas, fit, plotting, and printed summary; run all tests and commit green.

### Task 3: Reproduction and independent gate

**Files:** Create `week3/Makefile`; modify `week3/README.md`.

- [ ] Make `reproduce` run snapshots, sweep, plots, and analysis with defaults.
- [ ] Run it; verify 2,740,000 rows and deterministic checksums.
- [ ] Run the supplied checker and inspect both plots; document commands and commit.
