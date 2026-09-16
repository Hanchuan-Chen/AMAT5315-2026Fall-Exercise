# Week 3 Part 1 Ising Sampler Design

## Purpose

Build `ising`, a Rust command that samples the two-dimensional Ising model along
one ascending temperature ramp, prints one tab-separated summary line per
temperature, and writes the raw artifacts the course viewer and the course gate
read. Phase A covers Part 1 of the week-3 sheet: the contract file, the
`metropolis` update rule, the run artifacts, and the Part 1 verification
evidence. The cluster update `wolff` is named by the contract but is implemented
in a later phase.

The contract is `week3/ising.design.toml`, copied byte-for-byte from the course's
published `week3-ising.design.toml`. It is the whole specification of the
command line and of what the command writes; the physics of the model is fixed
by the learning sheet and repeated below so the design is self-contained.

## Scope

In scope for Phase A:

- `week3/ising.design.toml`, byte-identical to the course reference;
- a crate `week3/` building the binary `ising` with exactly the flags in the
  contract and no unstated defaults;
- the `metropolis` update: one sweep is `L*L` single-site proposals, each site
  drawn independently and uniformly with replacement;
- the temperature ramp with warm starts and the `--every` frame recording;
- `<out>/run.json`, `<out>/series.jsonl`, `<out>/spins.jsonl`;
- the stdout table;
- the Part 1 evidence: the cold/hot check, the Boltzmann ratio chart, the
  repeatability check, the committed ramp recording, and the three viewer
  proofs.

Out of scope for Phase A: the `wolff` cluster update, the magnetization and
susceptibility charts, the critical-temperature estimate, autocorrelation and
bootstrap errors, and the Part 2-4 artifact runs. `--update wolff` parses as a
valid value and exits with a clear unimplemented error rather than being
rejected by the argument parser, so the later phase only fills in a function
body and never touches the command line.

## Model and conventions

- Lattice: `L x L` spins `s_i in {-1, +1}` on a periodic square lattice
  (`s_{i+L} = s_i` in both directions), `J = 1`, no external field, `T` in
  units of `J`.
- Energy: `E(s) = -sum_<ij> s_i s_j` over nearest-neighbour bonds, so the energy
  per site is `E / L^2`, with a minimum of `-2` for a fully aligned lattice.
- The sum runs over each unordered bond once. For a `L x L` torus that is
  `2 L^2` bonds (`L = 2` included, where each site has two distinct neighbours).
- Magnetization: `m = (1/L^2) sum_i s_i`, signed; the printed observable is the
  mean of `|m|` over the measured steps.
- Energy change of one flip: flipping a single site changes only its four
  bonds, so `dE = 2 s_i (s_1 + s_2 + s_3 + s_4)`. The neighbour sum is one of
  `-4, -2, 0, 2, 4`, so `dE` is one of `-8, -4, 0, +4, +8`.
- Metropolis acceptance: `A = min(1, exp(-dE / T))`. `dE <= 0` always passes and
  consumes no uniform draw; `dE > 0` passes when a fresh uniform in `[0, 1)` is
  below `exp(-dE / T)`.
- Time: one sweep is `L*L` proposals, accepted or not. Simulation time is
  counted in sweeps for `metropolis`.
- Ramp: the first temperature starts from an all-up lattice; every later
  temperature starts from the previous temperature's final lattice (warm
  start); each temperature discards `--discard` sweeps and then takes
  `--measure` measured sweeps.
- Randomness: one `ChaCha8Rng` seeded from `--seed` is created per run and
  carried through the whole ramp, so the same arguments always reproduce the
  same artifacts byte for byte and a different seed diverges.

## Temperature grid

`t_grid = { t_from + k * t_step : k = 0, 1, 2, ... ; value <= t_to }`, evaluated
in ascending order and stopped at the first value above `t_to`. `t_to` is
therefore included exactly when the step lands on it: `1.5 .. 3.5` step `0.05`
gives 41 temperatures ending at `3.5`, while `1.5 .. 3.55` step `0.1` gives 21
temperatures ending at `3.5` because `3.55` is never reached.

Each candidate is rounded to nine decimals before the comparison so that binary
floating point cannot drop the last point (`1.5 + 40 * 0.05` is
`3.5000000000000004` before rounding) and cannot print as
`2.3000000000000003`. Nine decimals is far finer than any temperature step this
command needs and far coarser than f64 noise over the ranges involved. An empty
grid (for example `--t-to` below `--t-from`) is a usage error.

## Command line

`clap` derive, one command, no subcommands. Long flags only: `--update`, `--l`,
`--t-from`, `--t-to`, `--t-step`, `--discard`, `--measure`, `--seed`, `--every`
(default `0`), `--out`. Every flag except `--every` is required, which is what
"no unstated defaults" means in the contract; `--every 0` is the documented
default and records no frames. Validation rejects `--l < 2`, a non-positive
`--t-step`, an empty grid, and `--measure 0` before any file is touched.

`--update` is a value enum `metropolis | wolff`. Phase A accepts both spellings
and returns exit code 1 with `error: --update wolff lands in a later phase; this
build implements metropolis only` after the argument parse succeeds.

## Outputs

All paths are relative to the working directory. `--out` may exist; files this
run writes are overwritten, nothing else in the folder is read or removed.

### stdout

A header line, then one line per temperature as it finishes:

```text
T	mean|M|	acceptance
1.800	0.9568	0.0410
```

Tab separated. `T` with three decimals, `mean|M|` and acceptance with four.
Metropolis prints `mean |m|` over its measured steps and the accepted flips per
proposal over `discard + measure` at that temperature.

### `<out>/run.json`

One JSON object with exactly the fields the contract lists, in that order:
`L`, `update` (`"metropolis"`), `t_grid` (the list above), `discard`,
`measure`, `seed`, `sample_every` (`1`), `time_unit` (`"sweep"`). The viewer
fetches this file beside `spins.jsonl` when it exists.

### `<out>/series.jsonl`

One compact JSON object per measured step, in ramp order, every temperature
ascending: `{"L":64,"T":1.5,"sweep":1,"M":0.956789,"E":-1.900123}`. `sweep`
restarts at `1` after the discard block at each temperature (it indexes measured
steps, not the global clock). `M` is the signed mean spin `m` and `E` is the
energy per site, both rounded to six decimals. `sample_every = 1` in `run.json`
states the rule: every measured step is a row.

### `<out>/spins.jsonl`

Written only when `--every > 0`; with `--every 0` no file is created. One
compact object per recorded frame, at measured steps
`every, 2*every, 3*every, ...` at each temperature:
`{"L":64,"T":1.5,"sweep":2020,"m":0.9839,"spins":[1,-1,...]}`.
Here `sweep` is the global step counter, incremented for every sweep performed
in the whole ramp including discarded ones, so a frame at measured step
`k` of temperature `j` carries
`sum_{i<j} (discard + measure) + discard + k`. `m` is the signed magnetization
of that frame rounded to four decimals (the precision the viewer stamps).
`spins` is the row-major array of `L^2` integers, `+1` for up and `-1` for down,
which is the array form the published viewer validates.

## Module architecture

### `src/lattice.rs`

`Lattice` owns `l: usize` and `spins: Vec<i8>` in row-major order. It exposes
`all_up`, a validated `from_spins`, `l`, `spins`, `spin(row, col)`,
`neighbor_sum(row, col)`, `delta_energy(row, col)`, `energy`,
`energy_per_site`, `magnetization`, `abs_magnetization`, and `flip(row, col)`.
`energy` is `O(L^2)` and exists for tests and for an initial value; the ramp
does not need it, because the Metropolis step only ever evaluates one flip's
`dE`. Index arithmetic wraps both edges so there is no boundary branch.

### `src/metropolis.rs`

Pure and testable: `acceptance_probability(delta_energy, temperature)`,
`proposal_accepted(delta_energy, temperature, uniform)`, and
`sweep(lattice, temperature, rng) -> SweepStats { accepted, proposals }`. The
sweep draws `L*L` sites with `rng.random_range(0..L*L)`, flips on acceptance.
The rule is factored out of the loop so the acceptance test can call it with a
supplied uniform instead of relying on statistics alone.

### `src/ramp.rs`

`RunConfig` (validated), `temperature_grid(t_from, t_to, t_step)`,
`run_ramp(config) -> Vec<TemperatureResult>` where `TemperatureResult` carries
`t`, `mean_abs_m`, `acceptance`, and `frames`. The ramp is the only place that
owns the random stream and the global sweep counter; it drives the writers
through a small `Recorder` so the three artifacts stay in step.

### `src/artifacts.rs`

`Recorder` wraps the optional `series.jsonl` and `spins.jsonl` writers plus the
counters, formats the rows, and writes `run.json`. Row formatting is explicit
(`{:.6}` for `M` and `E`, `{:.4}` for frame `m`, compact JSON) rather than
`serde_json` float output, so "six decimals" is a property of the writer and
not of a float printer.

### `src/cli.rs`, `src/main.rs`

`cli.rs` holds the `clap` struct, the `Update` enum, and the conversion from
parsed arguments to `RunConfig`. `main.rs` parses, prints the header, runs the
ramp while printing one row per temperature, and maps errors to exit code 1
with a message on stderr. `lib.rs` re-exports the modules so integration tests
can drive the physics directly.

## Verification and evidence

Phase A ships these checks, all produced by running the built binary:

1. `cargo install --path . --quiet`, then `ising` runs by name.
2. Cold/hot: `T = 1.8` gives `mean |m| in [0.95, 0.965]` and acceptance near
   `0.041`; `T = 3.0` gives `mean |m| < 0.06` and acceptance near `0.46`.
3. Boltzmann ratio: energy histograms of the total energy (`E * 4096`) from the
   `T = 3.0` and `T = 3.1` runs, bin width 40, log ratio where both bins hold at
   least five rows, against slope `1/3.0 - 1/3.1 = 0.0107527`. Saved by
   `scripts/boltzmann.py` as `evidence/boltzmann.png`.
4. Repeatability: seeds `2026` twice give byte-identical `series.jsonl`; seed
   `2027` differs in the last digits.
5. The contract's usage line exactly, warm start from all-up: its
   `spins.jsonl` is committed as `week3/spins.jsonl`.
6. The committed recording loaded in the published viewer, with the viewer's
   own `window.composeProofPNG()` capture saved for `T = 1.8, 2.3, 3.0` into
   `evidence/`.

## Risks and decisions

- **Format of `spins`.** The older 19-page sheet's reference recording packed
  the lattice as a string of `0`/`1` characters; the published viewer now
  validates `spins` as an array of `L^2` integers in `{+1, -1}`. This design
  follows the published viewer, which is the only reader the sheet names.
- **Committed recording size.** With `--every 20` the usage line records 410
  frames, about 4 MB of compact JSON. It stays below the 5 MB per-file limit but
  is the largest tracked file in the week; nothing else generated is committed.
- **Acceptance draws.** Skipping the uniform draw when `dE <= 0` shrinks the
  stream per sweep. That choice is part of the reproducibility contract, so it
  is fixed here and covered by the byte-identical same-seed test.
- **`m` precision in `spins.jsonl`.** The contract does not fix it; the viewer
  renders `m.toFixed(4)`, so four decimals is enough for every reader that
  names this file, while `series.jsonl` keeps the six the contract asks for.
