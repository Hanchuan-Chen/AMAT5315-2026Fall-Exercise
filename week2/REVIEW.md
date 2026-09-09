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

5. **No unresolved code or evidence finding.** Naive and cells share the exact
   pair physics and agree in the boundary, cutoff, perturbed-lattice, and
   two-cell tests. Heating is isolated from the NVE run. Timing/profile claims
   match the recorded tables and images; both videos and the page data match
   their documented configurations.

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

The course sheet separately asks for a fresh-agent review. This file records
the completed inline review; an independent fresh session should confirm or
amend these findings before the final push.
