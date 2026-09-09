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

For seed 2026, the Rust checker measured energy-consistency error 8.515e-16,
secular energy drift 2.806e-4, temperature 0.5134, and Rayleigh chi2/dof =
0.448. The supplied independent Python checker measured energy-consistency
error 6.816e-16 and the same remaining metrics. Both reported PASS.

## Raw artifacts

make reproduce writes artifacts/run.json and artifacts/traj.jsonl. The
trajectory contains exactly 200 production frames at steps 50 through 10000.
The artifacts/ directory is ignored by Git because these files are generated
inputs for the check and video commands.
