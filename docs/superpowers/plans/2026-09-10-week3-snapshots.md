# Week 3 Snapshot Ramp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and publish the 410-frame Week 3 Ising temperature ramp.

**Architecture:** Reuse the tested lattice and Metropolis sweep, add a streaming JSONL artifact writer and one protocol runner, then expose it through the existing thin CLI.

**Tech Stack:** Rust 2024, `serde`, `serde_json`, supplied HTML viewer, GitHub Pages.

**Spec:** `docs/superpowers/specs/2026-09-10-week3-snapshots-design.md`

## Global Constraints

- Preserve the exact five-key frame contract.
- Defaults are `L=64`, `T=1.5..3.5:0.05`, 2000 equilibration sweeps, 200 recorded sweeps, interval 20, seed 2026.
- Keep generated `week3/artifacts/` out of Git and track the published copy in `docs/week3/`.

---

### Task 1: Snapshot protocol, red then green

**Files:** Create `week3/tests/snapshots.rs`, `week3/src/artifacts.rs`, and `week3/src/snapshots.rs`; modify `week3/src/lib.rs` and `week3/Cargo.toml`.

**Interfaces:** Produce `SnapshotConfig`, `SnapshotFrame`, and `write_snapshots(&SnapshotConfig)`.

- [ ] Write a small-run test that requires an inclusive three-temperature ramp, two frames per temperature, cumulative sweeps, exact JSON keys, valid spin strings, and byte-identical seeded output.
- [ ] Run it and observe failure because the interfaces do not exist; commit the red state.
- [ ] Implement streaming output by reusing one lattice and RNG across temperatures.
- [ ] Run all release tests; commit the green state.

### Task 2: CLI and publication

**Files:** Modify `week3/src/cli.rs` and `week3/README.md`; create `docs/week3/index.html` and `docs/week3/spins.jsonl`.

**Interfaces:** Produce `ising snapshots` with all protocol values tunable and course defaults.

- [ ] Extend the CLI test with a tiny configurable snapshots command and observe it fail.
- [ ] Implement argument validation and invocation; run all tests.
- [ ] Run the default command and verify 410 lines, field names, size, and deterministic checksum.
- [ ] Place the supplied viewer and generated recording under `docs/week3/`, document the Pages URL, and commit.
