#!/usr/bin/env python3
"""Block bootstrap of the susceptibility peaks and of the extrapolated T_c.

Part 3 of the week-3 sheet, the sampling-error check. Part 2's `T_c` came from
one random stream with no error bar; a block bootstrap puts one on it. Each
replicate rebuilds the critical window by drawing whole blocks of consecutive
sweeps with replacement, so the correlations inside a block survive, and then
recomputes everything Part 2 computed:

    chi(T) = L^2 (<m^2> - <|m|>^2) / T                          (Equation 10)
    T_peak = the vertex of the five-point parabola around the largest chi
    T_c    = 2 T_peak(64) - T_peak(32)                           (Equation 11)

Each size and temperature is resampled separately, at block lengths 2000, 4000
and 8000 sweeps, the sheet's block lengths, with 4000 replicates each. The
spread of the replicated `T_c` values is its sampling error; the sheet calls
that error stable when the three block lengths agree within a tenth of their
mean, and reports it as unresolved otherwise.

A bootstrap `sigma_Tc` carries a Monte Carlo error of its own, about
`1 / sqrt(2 (B - 1))` of itself: 3.2% at 500 replicates and 1.1% at 4000,
against a sheet line a tenth of the mean away. The first version of this
script drew 500 replicates from one seed and reported that draw's verdict as
the answer, which made the verdict a property of the seed 3 times in 10; the
replicate count is now 4000, and the seed sweep below runs at that count, so
the verdict of the committed draw and of the estimate pooled over the sweep
are both reported, with the single-draw crossing rate between them.

Two kinds of replicate are failed fits and are counted, not silently dropped:
a parabola that does not bend downward (a positive curvature, which
`peaks.fit_peak` rejects) and a vertex that falls outside the five fitted
temperatures.

The bootstrap error covers sampling only. It excludes the finite-size
correction Equation 11 does not cancel and the bias of fitting a parabola to a
peak that is not one, both of which more sweeps do not shrink.

Blocks are handled through their means: a resampled series is the mean of the
drawn block means, so a replicate costs `k` additions rather than a 100000-row
copy, and the whole run -- the committed draw plus the default 50-seed sweep,
3 * 4000 replicates each -- costs about a minute once the rows are read.

Run:  .venv/bin/python scripts/chi_bootstrap.py
Writes:  week3/evidence/chi-bootstrap.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import errors  # noqa: E402
import peaks  # noqa: E402

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent

#: The sheet's block lengths.
BLOCK_LENGTHS = [2000, 4000, 8000]

#: Part 3's replicate count, and the count the committed chart is drawn at.
#: A bootstrap `sigma_Tc` carries a Monte Carlo error of about
#: `1 / sqrt(2 (B - 1))` of itself -- 3.2% of it at the 500 replicates the
#: first version of this script used, 1.1% at 4000 -- and the sheet's
#: stability line sits a tenth of the mean away, so 4000 puts the Monte Carlo
#: error well under the line; the residual single-draw sensitivity is then
#: measured by the sweep below rather than assumed away. See `SeedSweep`.
DEFAULT_REPLICATES = 4000

#: Part 4's sampler comparison reads this name for its own replicate count and
#: its committed evidence was produced at 500, so the value stays 500 here.
REPLICATES = 500

#: Seeds the single-draw sensitivity check sweeps, starting at the default
#: seed. Every swept seed is a complete bootstrap draw at the same replicate
#: count, so the sweep measures the scatter of the estimator itself.
SEED_SWEEP = 50

#: Sizes and the number of grid points the peak fit sees.
SIZES = (32, 64)
FIT_POINTS = 5

COLORS = {32: "tab:green", 64: "tab:blue"}


@dataclass(frozen=True)
class FitOutcome:
    """One five-point parabola: where it peaks, how high, and its curvature."""

    t_peak: float
    t_min: float
    t_max: float
    curvature: float
    chi_peak: float = 0.0


def fit_is_acceptable(outcome: FitOutcome) -> bool:
    """Does the fitted parabola bend downward and peak inside its five points?"""
    if not outcome.curvature < 0.0:
        return False
    return outcome.t_min <= outcome.t_peak <= outcome.t_max


def block_means(series: np.ndarray, block_length: int) -> np.ndarray:
    """Means of the first `n // block_length` whole blocks of `series`."""
    series = np.asarray(series, dtype=float)
    nblocks = series.size // block_length
    if nblocks < 1:
        raise ValueError("block longer than the series")
    return series[: nblocks * block_length].reshape(nblocks, block_length).mean(axis=1)


def resample_chi(
    abs_blocks: np.ndarray,
    sq_blocks: np.ndarray,
    lattice: int,
    t: float,
    rng: np.random.Generator,
) -> float:
    """Equation 10 on one bootstrap resample of one (L, T) series.

    Drawing `k` blocks with replacement and averaging their means is exactly
    the mean of the concatenated resampled series, because every block has the
    same length.
    """
    nblocks = abs_blocks.size
    drawn = rng.integers(0, nblocks, nblocks)
    mean_abs = float(abs_blocks[drawn].mean())
    mean_sq = float(sq_blocks[drawn].mean())
    variance = mean_sq - mean_abs * mean_abs
    return lattice * lattice * variance / t


@dataclass
class BootstrapResult:
    """One block length's worth of replicated critical temperatures."""

    block_length: int
    replicates: int
    tc_values: np.ndarray
    failures: int
    central_tc: float
    central_peaks: dict[int, float] = field(default_factory=dict)
    parabolas: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def successes(self) -> int:
        return int(self.tc_values.size)

    @property
    def error(self) -> float:
        if self.tc_values.size < 2:
            return float("nan")
        return float(self.tc_values.std(ddof=1))


def fits_for_window(
    window: dict[int, dict[float, np.ndarray]],
    block_means_by_size: dict[int, dict[str, dict[float, np.ndarray]]],
    rngs: dict[tuple[int, float], np.random.Generator],
) -> tuple[dict[int, FitOutcome], list[str]]:
    """Recompute one replicate's five-point fit for every size.

    `block_means_by_size` carries the precomputed block means of `|m|` and
    `m^2`; each call resamples all of them once. The returned outcomes are the
    fits per size, and the second value names the sizes whose fit failed so
    the caller can count the replicate as failed rather than drop it.
    """
    outcomes: dict[int, FitOutcome] = {}
    failed: list[str] = []
    for lattice in SIZES:
        abs_blocks = block_means_by_size[lattice]["abs"]
        sq_blocks = block_means_by_size[lattice]["sq"]
        ts = sorted(window[lattice])
        chis = [
            resample_chi(abs_blocks[t], sq_blocks[t], lattice, t, rngs[(lattice, t)])
            for t in ts
        ]
        try:
            fit = peaks.fit_peak(ts, chis)
        except ValueError:
            failed.append(f"L={lattice} has no downward parabola")
            continue
        outcome = FitOutcome(
            t_peak=fit.t_peak,
            t_min=fit.t_min,
            t_max=fit.t_max,
            curvature=fit.curvature,
            chi_peak=fit.chi_peak,
        )
        if not fit_is_acceptable(outcome):
            failed.append(f"L={lattice} peak {fit.t_peak:.4f} outside its five points")
            continue
        outcomes[lattice] = outcome
    return outcomes, failed


def central_fits(
    window: dict[int, dict[float, np.ndarray]]
) -> tuple[dict[int, peaks.PeakFit | None], float]:
    """The unbootstrapped five-point fits and `T_c` of the same window rows.

    A size whose sampled susceptibility has no downward parabola has no
    central fit either; `T_c` is then undefined (`nan`) rather than invented,
    which is the same verdict the bootstrap replicates get.
    """
    fits: dict[int, peaks.PeakFit | None] = {}
    for lattice in SIZES:
        ts = sorted(window[lattice])
        chis = []
        for t in ts:
            m = window[lattice][t]
            mean_abs = float(np.abs(m).mean())
            mean_sq = float((m * m).mean())
            chis.append(peaks.chi_at(lattice, peaks.TemperatureStats(t, mean_abs, mean_sq, m.size))[1])
        try:
            fits[lattice] = peaks.fit_peak(ts, chis)
        except ValueError:
            fits[lattice] = None
    if any(fit is None for fit in fits.values()):
        return fits, float("nan")
    return fits, 2.0 * fits[64].t_peak - fits[32].t_peak


def bootstrap_tc(
    window: dict[int, dict[float, np.ndarray]],
    block_length: int,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = 2026,
) -> BootstrapResult:
    """Block-bootstrap `T_c` of the window runs at one block length."""
    means: dict[int, dict[str, dict[float, np.ndarray]]] = {}
    rngs: dict[tuple[int, float], np.random.Generator] = {}
    for lattice in SIZES:
        abs_blocks: dict[float, np.ndarray] = {}
        sq_blocks: dict[float, np.ndarray] = {}
        for index, t in enumerate(sorted(window[lattice])):
            m = window[lattice][t]
            abs_blocks[t] = block_means(np.abs(m), block_length)
            sq_blocks[t] = block_means(m * m, block_length)
            rngs[(lattice, t)] = np.random.default_rng([seed, lattice, index])
        means[lattice] = {"abs": abs_blocks, "sq": sq_blocks}

    fits, central_tc = central_fits(window)
    tc_values: list[float] = []
    failures = 0
    parabolas: dict[int, list[tuple[float, float, float]]] = {32: [], 64: []}
    for _ in range(replicates):
        outcomes, failed = fits_for_window(window, means, rngs)
        if failed or len(outcomes) != len(SIZES):
            failures += 1
            continue
        tc = 2.0 * outcomes[64].t_peak - outcomes[32].t_peak
        tc_values.append(tc)
        for lattice, outcome in outcomes.items():
            parabolas[lattice].append(
                (outcome.t_peak, outcome.chi_peak, outcome.curvature)
            )
    parabolas = {k: (np.array(v) if v else np.zeros((0, 3))) for k, v in parabolas.items()}
    return BootstrapResult(
        block_length=block_length,
        replicates=replicates,
        tc_values=np.array(tc_values),
        failures=failures,
        central_tc=central_tc,
        central_peaks={
            lattice: fits[lattice].t_peak for lattice in SIZES if fits[lattice] is not None
        },
        parabolas=parabolas,
    )


def stability_verdict(errors_by_length: dict[int, float]) -> tuple[bool, str]:
    """Stable when the three errors agree within a tenth of their mean."""
    values = np.array([errors_by_length[b] for b in BLOCK_LENGTHS], dtype=float)
    mean = float(values.mean())
    spread = float(values.max() - values.min())
    if not np.isfinite(values).all():
        return False, "sampling error unresolved: not enough successful replicates"
    if spread <= mean / 10.0:
        return True, (
            f"stable: the three errors span {spread:.4f}, within a tenth of their "
            f"mean {mean:.4f}"
        )
    return False, (
        f"sampling error unresolved: the three errors span {spread:.4f}, more than "
        f"a tenth of their mean {mean:.4f}"
    )


@dataclass(frozen=True)
class SeedSweep:
    """One complete bootstrap draw per seed, and what the sweep says.

    Every seed bootstraps the same rows, so all of them estimate the same
    sampling error and the scatter between them is the Monte Carlo noise of
    the estimator itself. `pooled` averages those draws, which cuts the
    residual Monte Carlo error from 1.1% of `sigma` to about 0.16% of it at 50
    seeds, and is why the pooled verdict is a statement about the estimator
    rather than about the seed; `crossings` counts the single draws that would
    have reported the other verdict on their own.
    """

    seeds: list[int]
    replicates: int
    errors: dict[int, np.ndarray]

    @property
    def total(self) -> int:
        return len(self.seeds)

    @property
    def stabilities(self) -> np.ndarray:
        """Each seed's own verdict, in seed order."""
        return np.array(
            [
                stability_verdict({b: self.errors[b][i] for b in BLOCK_LENGTHS})[0]
                for i in range(self.total)
            ],
            dtype=bool,
        )

    @property
    def crossings(self) -> int:
        """How many single draws cross the sheet's tenth-of-the-mean line."""
        return int((~self.stabilities).sum())

    @property
    def ratios(self) -> np.ndarray:
        """Each seed's `span / mean` of its three errors, in seed order."""
        values = np.stack([self.errors[b] for b in BLOCK_LENGTHS], axis=1)
        mean = values.mean(axis=1)
        span = values.max(axis=1) - values.min(axis=1)
        if not np.isfinite(values).all() or not np.isfinite(mean).all():
            return np.full(self.total, np.nan)
        return np.where(mean > 0.0, span / np.where(mean > 0.0, mean, 1.0), 0.0)

    @property
    def pooled(self) -> dict[int, float]:
        """The mean of the swept seeds' errors at each block length."""
        return {b: float(self.errors[b].mean()) for b in BLOCK_LENGTHS}

    def pooled_summary(self) -> tuple[np.ndarray, float, float]:
        """`(errors, span, span / mean)` of the pooled estimate."""
        values = np.array([self.pooled[b] for b in BLOCK_LENGTHS], dtype=float)
        mean = float(values.mean())
        span = float(values.max() - values.min())
        return values, span, (span / mean if mean > 0.0 else 0.0)


def sweep_seeds(
    window: dict[int, dict[float, np.ndarray]],
    replicates: int,
    seeds: list[int],
) -> SeedSweep:
    """Bootstrap the whole stability check once per seed.

    The seeds are swept rather than averaged over inside one bootstrap so that
    each one stays a complete, reproducible draw: the sweep answers "would a
    reader who chose another seed get the committed verdict?", which is the
    question the replicate count alone cannot settle.
    """
    errors = {b: np.empty(len(seeds), dtype=float) for b in BLOCK_LENGTHS}
    for index, seed in enumerate(seeds):
        for length in BLOCK_LENGTHS:
            errors[length][index] = bootstrap_tc(window, length, replicates, seed).error
    return SeedSweep(seeds=list(seeds), replicates=replicates, errors=errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts")
    parser.add_argument("--out", type=Path, default=WEEK / "evidence" / "chi-bootstrap.png")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--seed-sweep",
        type=int,
        default=SEED_SWEEP,
        help="seeds to sweep from --seed for the single-draw sensitivity check; 0 disables it",
    )
    return parser.parse_args()


def load_window(artifacts: Path) -> dict[int, dict[float, np.ndarray]]:
    """`{L: {T: signed M series}}` for the two window runs."""
    window: dict[int, dict[float, np.ndarray]] = {}
    for lattice in SIZES:
        run_lattice, series = errors.read_series(artifacts / peaks.WINDOW_RUNS[lattice])
        if run_lattice != lattice:
            raise SystemExit(f"{peaks.WINDOW_RUNS[lattice]} holds L = {run_lattice}")
        window[lattice] = {
            t: m
            for t, m in series
            if peaks.WINDOW_LOW - peaks.T_TOL <= t <= peaks.WINDOW_HIGH + peaks.T_TOL
        }
    return window


def main() -> None:
    args = parse_args()
    window = load_window(args.artifacts)
    temperatures = sorted(window[64])

    results = [
        bootstrap_tc(window, length, args.replicates, args.seed)
        for length in BLOCK_LENGTHS
    ]
    errors_by_length = {r.block_length: r.error for r in results}
    stable, verdict = stability_verdict(errors_by_length)
    this_span = float(
        max(errors_by_length.values()) - min(errors_by_length.values())
    )
    this_mean = float(np.mean(list(errors_by_length.values())))

    sweep = None
    if args.seed_sweep > 0:
        sweep = sweep_seeds(
            window,
            args.replicates,
            [args.seed + k for k in range(args.seed_sweep)],
        )

    fits, central_tc = central_fits(window)
    if not np.isfinite(central_tc):
        raise SystemExit("the window runs have no five-point peak fit for both sizes")

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), dpi=160, gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    shade = {2000: 0.28, 4000: 0.20, 8000: 0.14}
    for lattice in SIZES:
        ts = np.array(temperatures)
        chis = np.array(
            [
                peaks.chi_at(
                    lattice,
                    peaks.TemperatureStats(
                        t,
                        float(np.abs(window[lattice][t]).mean()),
                        float((window[lattice][t] ** 2).mean()),
                        window[lattice][t].size,
                    ),
                )[1]
                for t in temperatures
            ]
        )
        left.plot(
            ts,
            chis,
            "o",
            color=COLORS[lattice],
            markersize=4.5,
            label=f"$L$ = {lattice}: $\\chi(T)$",
        )
        fit = fits[lattice]
        grid = np.linspace(fit.t_min, fit.t_max, 101)
        left.plot(
            grid,
            fit.chi_peak + fit.curvature * (grid - fit.t_peak) ** 2,
            color=COLORS[lattice],
            linewidth=1.2,
            linestyle="-",
            label=f"$L$ = {lattice}: five-point fit, $T_{{peak}}$ = {fit.t_peak:.4f}",
        )
        left.axvline(fit.t_peak, color=COLORS[lattice], linestyle=":", linewidth=1.0)
        for result in results:
            parabolas = result.parabolas[lattice]
            if parabolas.size == 0:
                continue
            # The fitted parabola is only a model of the five grid points it
            # was fitted through, where the peak is narrow; drawing it across
            # the whole window would show the extrapolation, not the envelope.
            fine = np.linspace(fit.t_min, fit.t_max, 101)
            curves = parabolas[:, 1][:, None] + parabolas[:, 2][:, None] * (
                fine[None, :] - parabolas[:, 0][:, None]
            ) ** 2
            low = np.nanmin(curves, axis=0)
            high = np.nanmax(curves, axis=0)
            left.fill_between(
                fine,
                low,
                high,
                color=COLORS[lattice],
                alpha=shade[result.block_length],
                linewidth=0.0,
                label=(
                    f"$L$ = {lattice}: {result.replicates} replicates,"
                    f" {result.block_length}-sweep blocks"
                    if lattice == 64
                    else None
                ),
            )
    left.axvline(
        peaks.T_C,
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label=f"$T_c$ = {peaks.T_C:.5f}",
    )
    left.axvline(
        central_tc,
        color="0.35",
        linestyle="-.",
        linewidth=1.0,
        label=f"Part 2 $T_c$ = {central_tc:.4f}",
    )
    left.set_xlabel("temperature $T$")
    left.set_ylabel(r"$\chi(T)$")
    left.set_xlim(min(temperatures) - 0.02, max(temperatures) + 0.02)
    left.set_ylim(0.0, 95.0)
    left.set_title("Susceptibility peaks and the bootstrap envelope", fontsize=10)
    left.grid(alpha=0.25, linewidth=0.6)
    left.legend(loc="upper right", fontsize=7.5, frameon=True)

    for result in results:
        right.hist(
            result.tc_values,
            bins=30,
            histtype="step",
            linewidth=1.3,
            label=(
                f"{result.block_length}-sweep blocks: "
                f"$\\sigma_{{T_c}}$ = {result.error:.4f}, {result.failures} failed"
            ),
        )
    right.axvline(peaks.T_C, color="0.35", linestyle="--", linewidth=1.2, label="$T_c$ = 2.26919")
    right.axvline(central_tc, color="0.35", linestyle="-.", linewidth=1.0, label=f"Part 2 $T_c$ = {central_tc:.4f}")
    right.set_xlabel("replicated $T_c$")
    right.set_ylabel("replicates")
    right.set_title("Bootstrap critical temperatures", fontsize=10)
    right.grid(alpha=0.25, linewidth=0.6)
    right.legend(loc="upper left", fontsize=8)

    figure.suptitle(
        "Block bootstrap of the susceptibility peaks over the window runs,"
        f" {args.replicates} replicates at 2000 / 4000 / 8000 sweeps"
        + (f", {sweep.total}-seed sweep" if sweep is not None else ""),
        y=0.985,
        fontsize=11,
    )
    caption = []
    if sweep is not None:
        values, span, ratio = sweep.pooled_summary()
        pooled_stable, _ = stability_verdict(sweep.pooled)
        caption.append(
            f"pooled over {sweep.total} seeds: $\\sigma_{{T_c}}$ = "
            + " / ".join(f"{v:.4f}" for v in values)
            + f", span {span:.4f} = {ratio:.1%} of their mean {values.mean():.4f}"
            + f" -> {'stable' if pooled_stable else 'sampling error unresolved'}"
            " by the sheet's tenth-of-the-mean line"
        )
        caption.append(
            f"this draw (seed {args.seed}): span {this_span:.4f} = "
            f"{this_span / this_mean:.1%} of the mean; {sweep.crossings} of"
            f" {sweep.total} single draws cross the line"
            f" (span/mean {sweep.ratios.min():.3f} .. {sweep.ratios.max():.3f})"
        )
    if caption:
        figure.text(
            0.5,
            0.945,
            "\n".join(caption),
            ha="center",
            va="top",
            fontsize=8.5,
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86 if caption else 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)

    print(f"wrote {args.out}")
    print(f"  window temperatures: {temperatures[0]:.2f} .. {temperatures[-1]:.2f},"
          f" {len(temperatures)} points, n = {window[64][temperatures[0]].size} sweeps at each")
    print(f"  Part 2 central peaks: L=32 {fits[32].t_peak:.4f}, L=64 {fits[64].t_peak:.4f},"
          f" T_c = {central_tc:.4f}")
    for result in results:
        print(f"  block length {result.block_length:5d} sweeps:"
              f" {result.successes}/{result.replicates} successful replicates,"
              f" {result.failures} failed fits, T_c = {result.tc_values.mean():.4f}"
              f" +- {result.error:.4f} (sample sd, ddof=1)")
    print(f"  {'STABLE' if stable else 'UNSTABLE'} (seed {args.seed}): {verdict}")
    print(f"    span {this_span:.4f} = {this_span / this_mean:.1%} of the mean"
          f" {this_mean:.4f}; replicates carry a Monte Carlo error near"
          f" {1.0 / np.sqrt(2.0 * (args.replicates - 1)):.1%} of sigma")
    if sweep is not None:
        values, span, ratio = sweep.pooled_summary()
        pooled_stable, pooled_verdict = stability_verdict(sweep.pooled)
        print(f"  seed sweep, {sweep.total} seeds from {args.seed} at"
              f" {args.replicates} replicates each:")
        print(f"    single draws crossing the tenth-of-the-mean line:"
              f" {sweep.crossings}/{sweep.total}"
              f" (span/mean {sweep.ratios.min():.4f} .. {sweep.ratios.max():.4f})")
        print(f"    pooled errors: {' / '.join(f'{v:.4f}' for v in values)},"
              f" span {span:.4f}, span/mean {ratio:.4f}")
        print(f"  {'STABLE' if pooled_stable else 'UNSTABLE'} (pooled over"
              f" {sweep.total} seeds): {pooled_verdict}")
    print("  comparison with the sheet's Rust reference 0.0096 / 0.0095 / 0.0092")


if __name__ == "__main__":
    main()
