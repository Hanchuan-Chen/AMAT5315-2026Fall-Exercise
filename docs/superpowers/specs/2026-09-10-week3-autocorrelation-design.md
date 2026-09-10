# Week 3 Autocorrelation Analysis Design

## Goal

Quantify how much each saved magnetization mean can be trusted by reporting the
naive independent-sample error, a configurable blocked error, their ratio, and a
windowed integrated autocorrelation time for every size and temperature.

## Design

The existing raw `M` rows remain the only input. `analysis.rs` computes sample
standard deviation divided by `sqrt(n)` for the naive standard error and divides
the series into 50 equal contiguous blocks for the default blocked standard error.
It evaluates the autocovariance for all required lags with an FFT convolution,
normalizes exactly as the course checker does, and accumulates positive ACF terms
until the lag reaches `6 tau_int`, with a 5000-lag cap.

The `analyze [DIRECTORY] --blocks 50` command prints one stable line per `(L,T)`
with `mean_abs_m`, `naive`, `blocked`, `ratio`, and `tau_int`, followed by the
ordered-phase and critical-temperature summary already used in Part 3. It also
writes `tau.png`, showing both sizes against temperature on a logarithmic vertical
axis.

## Verification

Tests first fix the naive and blocked formulas, negative-ACF stopping rule, and a
strongly correlated fixture. The full run must show at `L=64,T=2.3` a blocked to
naive ratio of at least 10 and autocorrelation in the hundreds, while at `T=3.5`
the ratio is below 3 and the time is a few sweeps. The `L=64` peak must clear the
gate floor of five sweeps, and the plot must rise by over fifty-fold near the
transition with the larger lattice higher.
