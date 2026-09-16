# Week 3 Wrap-Up (README and Evidence Hygiene) Design

## Purpose

Phases A-D built the sampler, the critical-window analysis, the uncertainty
estimates and the cluster comparison, and committed their evidence. Phase E
closes the week-3 sheet's "Wrap up the work" section: one document that lets a
reader reproduce every committed artifact from a clean clone, and an
`week3/evidence/` folder in which every file has a producing command.

The deliverable is `week3/README.md`. It is a procedure, not a narrative: the
commands are meant to be pasted, in order, into a fresh clone, and every file
the sheet's tree lists has to appear when they are.

## What the README must contain

In the order the sheet fixes:

1. repository layout and prerequisites (Rust, `uv`, Node, Chrome);
2. the install commands, `cargo install --path . --quiet` from `week3/` and
   `uv venv .venv && uv pip install numpy matplotlib`, with `.venv/bin/python`
   used for every script afterwards;
3. every `ising` run with its values: the seven Part 1 runs (cold `T = 1.8`,
   hot `T = 3.0`, the `T = 3.1` Boltzmann partner, the two seed-2026 repeats,
   the seed-2027 control, and the contract's own usage line whose
   `runs/ramp/spins.jsonl` is copied to the tracked `week3/spins.jsonl`), the
   four Part 2 ramps, the two Part 4 cluster runs, and the three timed
   sweep-rate runs behind the extension answer;
4. every script beside the file it writes, in the order it must run;
5. the sheet's contract comparison against the published
   `week3-resources.zip`, with the `diff` that proves section-for-section
   equality;
6. the Part 3 extension numbers (`n_eff`, the longer run in sweeps and in
   hours) re-derived from `evidence/errors.txt`;
7. an inventory of every tracked file under `week3/`, and of every file in
   `week3/evidence/` beside its producing command;
8. what stays out of git (`artifacts/`, `runs/`, `target/`, `.venv/`,
   `.viewer/`) and the fact that the viewer capture needs the hosted viewer
   plus headless Chrome.

## Evidence hygiene

The sheet's tree lists twelve PNGs and separately asks for `peaks.txt` and
`errors.txt`. The tree currently also holds three hand-written run logs
(`part2-runs.txt`, `part3-runs.txt`, `part4-runs.txt`) and two script reports
(`magnetization-compare.txt`, `tau-compare.txt`).

- The three run logs are written by no command. Their content (commands, wall
  clocks, row counts, measured verdicts) moves into the README and the files
  are deleted, so the folder holds no hand-maintained loose file.
- The two script reports *are* the documented default output of
  `scripts/magnetization_compare.py --report` and `scripts/compare.py
  --report`. They stay, and the README lists them beside those commands.

The two report scripts currently print the chart's absolute path into the
report, which embeds one machine's directory into committed evidence. They are
changed to print the path relative to `week3/`, so a regenerated report is
identical on any clone. Nothing else in either script changes.

## Verification

The README is proved by following it literally in a throw-away clone:

```bash
git clone /Users/eureka/Documents/Codex/2026-09-02/ban/AMAT5315-2026Fall-Exercise /tmp/w3-fresh
```

The clone is the committed state, with no `artifacts/`, `runs/`, `.venv/` or
`.viewer/`. Every command runs as written; every listed file must exist
afterwards. The Part 1 runs are additionally checked byte for byte against the
files the committed `spins.jsonl` came from, so the README's command lines are
the ones that really produced the committed recording and not merely
equivalent-looking ones.

Final gates on the committed tree: `cargo test --release`,
`cargo clippy --all-targets --all-features -- -D warnings`,
`cargo fmt --check`, the Python suite through `.venv/bin/python -m unittest`,
`python3 -m pytest week1/` from the repository root, an empty `diff` of the
contract against the published copy, a clean `git status`, no tracked
generated path, and every tracked file below 5 MB.

## Risks and decisions

- **The README must stay true.** A command that needs a flag the README omits,
  or a file that no listed command writes, is a failure of this phase, so the
  fresh-clone run is the acceptance test, not a formality.
- **Run order.** The Part 1 runs must precede both the Boltzmann chart (it
  reads `runs/T3.0` and `runs/T3.1`) and the viewer capture (which copies
  `runs/ramp/run.json` into the served folder); Part 2 must precede Part 3,
  and the cluster runs must precede the Part 4 charts. The README states the
  order explicitly and the inventory repeats it.
- **Recomputation time.** The full regeneration is a few minutes of wall clock
  dominated by the `L = 64` window and cluster ramps, so the runs are launched
  in the background with logs and polled rather than blocking a single call.
- **Network and Chrome.** The contract comparison and the viewer proof need
  the course site; the viewer proof additionally needs headless Chrome and
  `puppeteer-core` in the git-ignored `.viewer/`. The README says so instead
  of pretending the step is offline.
- **PNG bytes.** The charts are deterministic for a fixed numpy/matplotlib
  version; the README pins the versions it was verified with and treats the
  file's existence, not its hash, as the reproducibility claim for the
  images.
