# Week 3 Autocorrelation Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add honest error bars and efficient autocorrelation analysis to saved runs.

**Architecture:** Extend the existing grouped raw-series analysis with small pure estimator functions and an FFT-backed autocorrelation path, then expose one CLI and one log-scale plot.

**Tech Stack:** Rust 2024, RustFFT, Plotters.

**Spec:** `docs/superpowers/specs/2026-09-10-week3-autocorrelation-design.md`

## Global Constraints

- Default to exactly 50 contiguous blocks.
- Match the course checker's `tau = 0.5 + positive ACF prefix`, `6*tau` window, and 5000-lag cap.
- Analyze existing rows without rerunning the simulation.

---

### Task 1: Estimators

**Files:** Modify `week3/tests/analysis.rs`, `week3/src/analysis.rs`, and `week3/Cargo.toml`.

**Interfaces:** Produce `naive_error`, `blocked_error`, and `integrated_autocorrelation_time`.

- [ ] Add exact fixture tests, run the missing-interface failure, and commit red.
- [ ] Implement the estimators, run all release tests, and commit green.

### Task 2: Analysis CLI and logarithmic chart

**Files:** Modify `week3/tests/cli.rs`, `week3/src/cli.rs`, `week3/src/plot.rs`, `week3/README.md`, and `week3/Makefile`.

**Interfaces:** Produce `ising analyze [DIRECTORY] --blocks 50` and `tau.png`.

- [ ] Add a small saved-run CLI test and observe the missing-command failure; commit red.
- [ ] Print every required field, write the log chart, and run all tests; commit green.
- [ ] Analyze the full Metropolis artifact and verify the required critical/remote values and chart shape.
