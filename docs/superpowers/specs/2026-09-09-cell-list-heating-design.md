# Cell-list Heating Design

## Goal

Complete Week 2 Part 5 by retaining the verified naive force path, adding a
default cell-list path with identical physics, adding an optional production
temperature ramp, measuring both paths, and publishing a 400-atom heating
trajectory for the supplied viewer.

## Force selection and shared state

System remains the only owner of positions, velocities, and accelerations.
PeriodicShifted stores a ForceMethod value. ForceMethod has Naive and Cells;
System::new_periodic selects Cells, while new_periodic_with_method permits
tests and benchmarks to select either path. The open dimer remains unchanged.

Naive visits every i<j pair. Cells uses nx=floor(Lx/rc) and
ny=floor(Ly/rc), so cell widths are at least rc. Each step rebuilds bins from
wrapped positions. For every atom i, search its wrapped 3x3 neighbor-cell
indices, deduplicate those indices, and consider only j>i. Apply the same
minimum-image, strict r<rc, shifted energy, and equal/opposite force code after
candidate discovery. This handles boundary pairs, exact-cutoff pairs, and
two-cell-wide boxes without duplicates.

## CLI and artifacts

md run adds --force naive|cells (default cells) and optional --ramp-to.
RunConfig carries both values. run.json adds ramp_to only for heated runs so
the original Part 4 contract remains unchanged for make reproduce. Force
choice is a runtime implementation detail documented by the command.

Without ramp_to, production stays NVE. With ramp_to, after each Verlet step
divisible by 50, remove center-of-mass velocity and rescale to

~~~text
T(step) = T_start + (T_final - T_start) * step / production_steps
~~~

Sampling follows rescaling, so the last saved frame reaches T_final. Heating
intentionally changes total energy and is never passed to the Part 4 NVE
checker.

## Evidence

- Timing table: three default runs each for supplied NumPy, Rust debug, and
  Rust release; record median and range.
- Profile screenshots: N=400, eq=200, production=1000 for naive and cells.
- Benchmark table: N=100,400,1600, eq=100, production=500, three runs per
  method; record median, range, and naive/cells speedup.
- scaling.png: log-log seconds per step from the six medians.
- cold.mp4 and hot.mp4: fixed T=0.2 and T=1.0, each under 2 MB.
- docs/index.html plus docs/run.json and docs/traj.jsonl: supplied viewer and
  N=400 ramp 0.2 to 1.2 over 20000 steps, sampled every 100 (200 frames).

## Acceptance

All earlier tests pass. Naive and cell accelerations and energies agree within
rounding tolerance on perturbed, boundary, cutoff, and two-cell cases. The
heating schedule test reaches both endpoints. Release is below one third of
debug time; cell-list speedup rises with N and exceeds 2 at N=1600. The cold
video retains distant g(r) peaks; the hot video has only short-range order; the
heating viewer shows falling long-range contrast and rising temperature.

