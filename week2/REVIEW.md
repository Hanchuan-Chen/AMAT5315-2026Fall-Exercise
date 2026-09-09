# Week 2 review

Scope: Week 2 implementation and evidence were compared with the designs and
plans in `docs/superpowers/` and with the learning-sheet acceptance checks.

## Findings

1. **Fixed — Pages artifacts were one directory too deep.** The viewer and
   heating data had initially been placed in `week2/docs/`, which GitHub Pages
   would not publish from the required repository-root `/docs` source. Commit
   `6666c54` moved all three files to `docs/` and corrected the reproduction
   command.

2. **Fixed — cell-list allocation erased the N=400 profile gain.** The first
   implementation allocated a vector per bin and a neighbor vector per atom on
   every force evaluation. Its 112 ms profile was slower than the 109 ms naive
   profile. Commit `6666c54` replaced this with prefix offsets, one flat atom
   array, and a fixed nine-entry neighbor array. The repeated profile is 95 ms,
   and the N=1600 three-run median improves from 0.4243 s to 0.1360 s (3.12x).

3. **Fixed — artifact writing masked the force hot spot.** Unbuffered JSONL
   writes accounted for most samples in the first measurement, so the profile
   did not answer the scientific performance question. Commit `8c919e1`
   added buffered output and release debug symbols. The final naive profile
   attributes 95.7% of simulation samples to force evaluation.

4. **Fixed — a malformed JSONL test fixture did not contain valid JSON.** This
   weakened the intended dimension-contract test by failing during parsing.
   Commit `9a4c81a` made the fixture valid so the test reaches and checks the
   dimension error.

5. **Fixed — periodic-boundary and cutoff assertions were too weak.** The
   original boundary case only asserted a force direction that could also occur
   without minimum-image wrapping, and the just-inside-cutoff assertion allowed
   an incorrectly early zero cutoff to pass. Commit `4d20219` now compares the
   cross-boundary force with the analytic force at the known minimum-image
   separation and compares the just-inside energy with the shifted
   Lennard-Jones value. The existing perturbed-lattice and two-cell equality
   tests remain in place.

6. **Not fixed — the required two-minute screen recording and GitHub release
   link are missing.** No recording file or release URL is present in the
   repository, and the README has no recording link. The learning sheet
   requires the student to make the recording in one take, narrate the cold and
   hot `g(r)` views, attach it to a GitHub release rather than Git, and add that
   release URL beside the Pages link. This requires the student's screen,
   voice, and authenticated GitHub release action, so it cannot be completed by
   this reviewer.

7. **No other unresolved code, physics, artifact, media, or performance
   finding.** Naive and cells share the exact pair physics and agree in the
   strengthened boundary/cutoff checks plus perturbed-lattice and two-cell
   tests. Heating is isolated from the NVE run. Timing/profile claims match the
   recorded tables and screenshots; all three videos and the deployed page data
   match their documented configurations.

## Verification after fixes

Verification was run from a new local clone at commit `6666c54`:

- `cargo test --manifest-path md/Cargo.toml --release`: all 22 non-ignored
  tests passed, including CLI, cell-list equality, and heating schedule tests.
- `make reproduce` followed by the Rust checker: PASS; secular drift
  `5.521e-4`, temperature `0.5136`, chi2/dof `1.119`.
- Supplied independent checker: PASS with energy consistency `7.381e-16`.
- The ignored external FFmpeg integration test was run explicitly and passed.
- `cold.mp4` and `hot.mp4` decode fully and are under 2 MB.
- The local HTTP page loads 400 atoms and 200 frames; long-range contrast falls
  from `0.354` to `0.096`, while the temperature trace reaches `1.197`.

## Independent fresh-agent re-verification

An independent fresh session re-read the 20-page learning sheet, all three
designs and plans, and the Week 2 source/tests without relying on the earlier
review. It then verified commit `4d20219` as follows:

- `cargo test --manifest-path week2/md/Cargo.toml --release`: all 22
  non-ignored tests passed; the external-video test remains intentionally
  ignored in the default suite.
- `cargo clippy --manifest-path week2/md/Cargo.toml --all-targets
  --all-features -- -D warnings`: passed with no warnings.
- `make reproduce` and the Rust checker: PASS; energy consistency `9.828e-16`,
  secular drift `5.521e-4`, temperature `0.5136`, and chi2/dof `1.119`.
- Supplied checker: PASS; energy consistency `7.381e-16` and the same drift,
  temperature, and speed-shape values.
- The deployed page returned HTTP 200 without authentication. Its viewer,
  `run.json`, and `traj.jsonl` exactly match the tracked files; metadata reports
  400 atoms, 20000 steps, sampling every 100 steps (200 frames), and a 0.2 to
  1.2 ramp.
- `fluid.mp4`, `cold.mp4`, and `hot.mp4` each decoded through all 200 frames,
  are 960x480 yuv420p at 20 fps, and are below 2 MB. Visual inspection confirms
  the documented ordered cold and flattened-tail hot `g(r)` behavior.
- Required paths are tracked, generated `week2/artifacts/` remains ignored,
  and a tracked-file scan found no student number, credential, token, identity
  document, or private note.
