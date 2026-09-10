# Week 3 Wolff Cluster Design

## Goal

Replace critical single-spin crawling with a seeded Wolff single-cluster update,
measure it under the same raw artifact and analysis contracts, and compare both
algorithms directly at `L=64`.

## Algorithm

Choose a random seed site and retain its spin. Grow a cluster through periodic
nearest neighbours: each not-yet-considered neighbour with the same spin joins
with probability `p_add = 1 - exp(-2/T)`. Each candidate is decided at most once.
After growth finishes, flip every cluster site; there is no acceptance test.

One comparable Wolff sweep performs cluster flips until the cumulative number of
touched spins is at least `L*L`. A single cluster may overshoot; it is never split.
All site choices and bond decisions use one explicitly seeded `StdRng`.

## Protocol and integration

`src/wolff.rs` implements cluster growth and comparable sweeps. `sweep --wolff`
selects it behind the existing sweep interface and defaults to `L=32,64`, the 13
temperatures `2.0..2.6:0.05`, 2000 equilibration and 100000 measurement sweeps,
and `artifacts-wolff/`. Its `run.json` says `algorithm="wolff"`; its row schema is
identical, so `analyze artifacts-wolff` runs unchanged.

The comparison command reads both runs at their largest shared size, writes the
tracked `week3/tau-compare.png`, and places both autocorrelation curves on one
logarithmic axis.

## Verification

Tests first cover the bond probability, periodic cluster growth, touched-spin
sweep definition, seeded reproducibility, Wolff artifact label, and CLI defaults.
The full run must have `tau_int < 2` and blocked/naive ratio below 2 at
`L=64,T=2.3`, remain nearly flat through the window, and estimate `T_c` within 2%
of 2.26919. Both algorithms must agree closely away from the transition; any
critical disagreement is interpreted using their effective sample counts and
block plateau lengths.
