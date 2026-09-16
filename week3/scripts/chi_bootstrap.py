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
and 8000 sweeps with 500 replicates each, both as the sheet specifies. The
spread of the replicated `T_c` values is its sampling error; the sheet calls
that error stable when the three block lengths agree within a tenth of their
mean, and reports it as unresolved otherwise.

Two kinds of replicate are failed fits and are counted, not silently dropped:
a parabola that does not bend downward (a positive curvature, which
`peaks.fit_peak` rejects) and a vertex that falls outside the five fitted
temperatures.

The bootstrap error covers sampling only. It excludes the finite-size
correction Equation 11 does not cancel and the bias of fitting a parabola to a
peak that is not one, both of which more sweeps do not shrink.

Blocks are handled through their means: a resampled series is the mean of the
drawn block means, so a replicate costs `k` additions rather than a 100000-row
copy, and all 39000 of them run in seconds once the rows are read.

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

#: The sheet's block lengths and replicate count.
BLOCK_LENGTHS = [2000, 4000, 8000]
REPLICATES = 500

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
    replicates: int = REPLICATES,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts")
    parser.add_argument("--out", type=Path, default=WEEK / "evidence" / "chi-bootstrap.png")
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--seed", type=int, default=2026)
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
                    f"$L$ = {lattice}: 500 replicates, {result.block_length}-sweep blocks"
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
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
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
    print(f"  {'STABLE' if stable else 'UNSTABLE'}: {verdict}")
    print("  comparison with the sheet's Rust reference 0.0096 / 0.0095 / 0.0092")


if __name__ == "__main__":
    main()
