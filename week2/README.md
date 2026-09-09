# Week 2: Agentic coding with Rust

This directory contains a two-dimensional molecular-dynamics program built in
stages from tested Lennard-Jones physics.

## Pair field

From `week2/`, reproduce `field.png` with:

```bash
cargo run --manifest-path md/Cargo.toml --release --example field -- field.png
```

The color is the pair energy capped to the interval `[-1, 1]`. The arrows show
the radial force, and the dashed circle marks the zero-force separation
`r0 = 2^(1/6)`.

## Dimer integration

From `week2/`, reproduce `dimer.png` with:

```bash
cargo run --manifest-path md/Cargo.toml --release --example dimer -- dimer.png
```

Both curves start from the same two stationary atoms at separation `1.2` and
use `dt = 0.01` through the same `Integrator` trait driver. Forward Euler shows
secular energy growth, while velocity-Verlet's error remains bounded and
oscillatory over the ten-times-longer run.

## Equilibrium fluid

Generate the deterministic default run from the week2 directory:

~~~bash
make reproduce
~~~

This runs 100 particles at density 0.8 and temperature 0.5. It uses a
potential-shifted Lennard-Jones cutoff at 2.5, minimum-image periodic
boundaries, 2000 thermostatted equilibration steps, and 10000 unthermostatted
velocity-Verlet production steps.

## Independent check

Run the Rust checker from the repository root:

~~~bash
cargo run --manifest-path week2/md/Cargo.toml --release -- check week2/artifacts
~~~

For seed 2026, the Rust checker measured energy-consistency error 9.828e-16,
secular energy drift 5.521e-4, temperature 0.5136, and Rayleigh chi2/dof =
1.119. The supplied independent Python checker measured energy-consistency
error 7.381e-16 and the same remaining metrics. Both reported PASS.

## Raw artifacts

make reproduce writes artifacts/run.json and artifacts/traj.jsonl. The
trajectory contains exactly 200 production frames at steps 50 through 10000.
The artifacts/ directory is ignored by Git because these files are generated
inputs for the check and video commands.

## Fluid video

With FFmpeg available on PATH, render the saved trajectory from the repository
root:

~~~bash
cargo run --manifest-path week2/md/Cargo.toml --release -- \
  video week2/artifacts --out week2/fluid.mp4
~~~

The committed fluid.mp4 is H.264/yuv420p at 960x480 and 20 frames per second.
It contains all 200 trajectory frames (10 seconds), with no duplicated or
dropped frames, and is 738759 bytes. Each frame shows the periodic particle box
beside a 64-bin radial distribution averaged over up to 20 recent samples.

## Timing

The supplied NumPy solver and Rust's retained naive path were each run three
times with the default 100-particle, 2000-step equilibration and 10000-step
production workload. Times are end-to-end wall seconds; each run wrote its
artifacts to a fresh temporary directory.

| implementation | three runs (s) | median (s) | range (s) |
|---|---:|---:|---:|
| supplied NumPy | 5.361, 5.291, 5.255 | 5.291 | 5.255–5.361 |
| Rust debug, naive | 1.699, 1.263, 1.262 | 1.263 | 1.262–1.699 |
| Rust release, naive | 0.413, 0.079, 0.079 | 0.079 | 0.079–0.413 |

The release median is 6.3% of the debug median, below the required one-third
threshold. The slower first release invocation is retained rather than hidden;
the median makes the cold-start effect explicit without discarding a run.

The timed commands were:

~~~bash
time /tmp/amat5315-numpy-venv-20260909/bin/python week2-sim.py
cargo build --manifest-path md/Cargo.toml
time ./md/target/debug/md run --force naive --out /tmp/md-debug
cargo build --manifest-path md/Cargo.toml --release
time ./md/target/release/md run --force naive --out /tmp/md-release
~~~

## Benchmark: cell-list force search

Select either implementation with `--force naive` or `--force cells`; cells is
the default. The cell grid uses `floor(L/rc)` cells per direction, making every
cell at least `rc` wide. It rebuilds wrapped bins each step, searches the
deduplicated 3x3 neighboring cells, and visits only `j > i`. Both paths then use
the same minimum-image displacement, strict cutoff, shifted energy, and pair
force code. Tests cover perturbed lattices, cross-boundary pairs, an exact
cutoff, and a two-cell-wide box.

The prescribed scaling command shape was run three times per row:

~~~bash
md run --n N --eq-steps 100 --steps 500 --force METHOD --out /tmp/run
~~~

| N | naive median (range), s | cells median (range), s | speedup |
|---:|---:|---:|---:|
| 100 | 0.0206 (0.0176–0.3101) | 0.0208 (0.0203–0.0242) | 0.99× |
| 400 | 0.0449 (0.0436–0.0455) | 0.0406 (0.0398–0.0407) | 1.10× |
| 1600 | 0.4243 (0.4223–0.4243) | 0.1360 (0.1353–0.1371) | 3.12× |

At small N, process startup, JSON output, bin construction, and allocation hide
the force-search saving. At N=1600 the all-pairs candidate count grows as
O(N²), while fixed density and cutoff leave a roughly constant number of nearby
particles per atom, so the cell-list force work grows approximately as O(N).
The measured speedup therefore rises with N and exceeds 2 at N=1600.

Reproduce the log-log figure with:

~~~bash
cargo run --manifest-path md/Cargo.toml --release --example scaling -- scaling.png
~~~

The figure reports each full-command median divided by the 500 production
steps. See `scaling.png`.

## Profile

Both profiles use N=400, 200 equilibration steps, and 1000 production steps:

~~~bash
samply record md run --n 400 --eq-steps 200 --steps 1000 \
  --force naive --out /tmp/profile-naive
samply record md run --n 400 --eq-steps 200 --steps 1000 \
  --force cells --out /tmp/profile-cells
~~~

| Version | Force share (%) | Elapsed time (s) |
|---|---:|---:|
| Naive | 95.7 | 0.109 |
| Cell list | 96.4 | 0.095 |

The symbolicated profiles span 109 ms (naive) and 95 ms (cells). In the naive
profile, `accelerations` accounts for 90 of 94 samples inside `simulate`
(95.7%; 82.6% of the complete command). In the cells profile it accounts for
81 of 84 simulation samples (96.4%; 85.3% of the command), now including flat
bin construction and neighbor lookup. The optimized elapsed time is below the
naive elapsed time. The evidence is saved as `profile-naive.png` and
`profile-cells.png`.

## Heating and melting

`--ramp-to` linearly raises the production target from `--temperature` to the
requested final temperature. Every 50 steps the program removes center-of-mass
velocity and rescales; sampling follows rescaling. `run.json` records
`ramp_to`, and the unheated path remains NVE.

The fixed-temperature comparison is reproducible with:

~~~bash
md run --n 100 --temperature 0.2 --out /tmp/cold
md video /tmp/cold --out cold.mp4
md run --n 100 --temperature 1.0 --out /tmp/hot
md video /tmp/hot --out hot.mp4
~~~

Both videos contain 200 H.264/yuv420p frames and are under 2 MB (`cold.mp4` is
605 KiB; `hot.mp4` is 785 KiB). Over the last 20 frames and radii at least 3,
the RMS deviation of g(r) from 1 is 0.549 for the cold solid and 0.109 for the
hot fluid: the cold trajectory retains distant peaks, while the hot trajectory
retains mainly short-range order.

## Pages

The supplied viewer is committed at the repository root's `docs/index.html`.
Its adjacent data is a
400-particle ramp from 0.2 to 1.2 over 20000 production steps, sampled every
100 steps:

~~~bash
cd ..
md run --n 400 --temperature 0.2 --ramp-to 1.2 --steps 20000 \
  --sample-every 100 --out docs
~~~

Local HTTP verification loaded all 200 frames and `run.json`. The viewer's
long-range contrast falls from 0.354 on the first frame to 0.096 over the last
20-frame window; the final temperature trace is 1.197. After GitHub Pages is
enabled from `main` and `/docs`, the page is:

<https://hanchuan-chen.github.io/AMAT5315-2026Fall-Exercise/>
