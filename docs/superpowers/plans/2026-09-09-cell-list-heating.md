# Cell-list Heating Implementation Plan

> Apply strict red-green TDD and preserve one commit per evidence-bearing stage.

**Goal:** Implement and measure the approved cell-list and heating design.

**Spec:** docs/superpowers/specs/2026-09-09-cell-list-heating-design.md

## Constraints

- Preserve the plain open dimer and the verified Part 4 NVE contract.
- Naive and cells share one pair-physics calculation.
- Cell widths are at least rc; wrapped neighbor-cell IDs are deduplicated.
- Cells is the CLI default; naive remains selectable.
- Ramp rescaling occurs every 50 production steps and sampling occurs after it.
- Never commit temporary benchmarks, build outputs, or the supplied checker.

### Task 1: Red tests

- Create tests/cells.rs comparing accelerations and energy for perturbed
  lattices, cross-boundary/cutoff pairs, and a two-cell-wide box.
- Create tests/heating.rs checking the linear endpoints, intermediate target,
  saved temperatures, and optional ramp_to JSON field.
- Run both targets and confirm missing interfaces, then commit:

~~~bash
git commit -m "test: add failing cell-list and heating checks"
~~~

### Task 2: Cell list

- Add ForceMethod and new_periodic_with_method in simulation.rs.
- Build bins every force evaluation; for each i visit deduplicated wrapped
  neighboring cells and j>i; call the existing cutoff/force accumulation.
- Add --force to RunConfig and CLI, defaulting to cells.
- Run focused equality tests, all release tests, and Clippy; commit:

~~~bash
git commit -m "feat: add cell-list force search"
~~~

### Task 3: Heating

- Add RunConfig.ramp_to and optional RunMetadata.ramp_to.
- Add production_target_temperature and rescale after every 50th heated
  production step before sampling.
- Add --ramp-to, run tests/Clippy, and commit:

~~~bash
git commit -m "feat: add production temperature ramp"
~~~

### Task 4: Measurements

- Record three-run timing and N=100/400/1600 method benchmarks.
- Render scaling.png from recorded medians.
- Record naive and cells samply profiles and save screenshots.
- Update README with exact commands, medians, ranges, profile values, and an
  explanation of quadratic versus fixed-neighbor search; commit:

~~~bash
git commit -m "docs: add cell-list performance evidence"
~~~

### Task 5: Melting evidence and page

- Copy the supplied viewer to docs/index.html.
- Generate the 400-atom 0.2-to-1.2 ramp into docs/.
- Generate fixed 100-atom cold and hot runs and videos.
- Compute and document cold/hot final long-range contrast.
- Commit page data, videos, and README:

~~~bash
git commit -m "docs: add heating and melting evidence"
~~~

### Task 6: Review and final verification

- Write week2/REVIEW.md with each finding marked fixed with a commit hash or
  not fixed with a concrete reason.
- Verify release tests, make reproduce, Rust/Python physics checks, all media,
  page fields, tracked-file scope, and clean Git status.
- Commit review without rewriting history. Push only after the manual
  screenshot/recording and repository checkpoint are approved.
