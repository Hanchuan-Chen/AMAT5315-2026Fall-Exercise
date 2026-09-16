# Week 3 Wrap-Up Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax. The plan's
> acceptance test is a literal run of the new README inside a throw-away clone.

**Goal:** `week3/README.md` regenerates every committed file of week 3 from a
clean clone, `week3/evidence/` has no undocumented loose file, and both claims
are proved by following the README in `/tmp/w3-fresh` and running the gates on
the committed tree.

**Spec:** docs/superpowers/specs/2026-09-16-week3-wrapup-design.md

## Constraints

- Work only in the canonical repository on `main`; never push, never rewrite
  history; conventional commits, one per logical change.
- Keep `artifacts/`, `runs/`, `target/`, `.venv/`, `.viewer/` out of git.
- Report only numbers a command in this session printed.

## Task 1: Design and plan (commit)

- [ ] Write `docs/superpowers/specs/2026-09-16-week3-wrapup-design.md` and this
  plan.
- [ ] Commit `docs: design and plan the week 3 wrap-up`.

## Task 2: The README (commit)

- [ ] Write `week3/README.md` with the layout, prerequisites, install, contract
  check, the seven Part 1 runs, the four Part 2 ramps, the two Part 4 cluster
  runs, every script beside its output, the extension numbers, the evidence
  inventory and the git-ignored paths.
- [ ] Commit `docs: add the week 3 reproduction guide`.

## Task 3: Evidence hygiene (commit)

- [ ] Fold `evidence/part2-runs.txt`, `part3-runs.txt`, `part4-runs.txt` into
  the README and delete them.
- [ ] Make `scripts/magnetization_compare.py` and `scripts/compare.py` write
  the chart path relative to `week3/`; regenerate
  `evidence/magnetization-compare.txt` and `evidence/tau-compare.txt`.
- [ ] Commit `chore: drop the loose week 3 run logs`.

## Task 4: Prove the README in a fresh clone

- [ ] `git clone` the committed state to `/tmp/w3-fresh`.
- [ ] Run every command in order, backgrounding the long ramps with logs.
- [ ] Confirm every listed file appears; `diff` the fresh `spins.jsonl` and
  the Part 1 series against the committed recording.
- [ ] Fix the README (or a script) for anything that failed, and re-run the
  affected steps.

## Task 5: Gates and commit (commit)

- [ ] `cargo test --release`, `cargo clippy --all-targets --all-features --
  -D warnings`, `cargo fmt --check`, the Python suites, `python3 -m pytest
  week1/`, an empty contract `diff`, a clean `git status`, no tracked
  generated path, every tracked file below 5 MB.
- [ ] Commit any fix the proof produced.
