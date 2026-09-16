#!/usr/bin/env python3
"""Susceptibility peaks and the extrapolated critical temperature.

Part 2 of the week-3 sheet. The four Metropolis ramps in `week3/artifacts/`
cover two lattice sizes on two temperature grids; this script reads their raw
rows and answers three questions.

1. How ordered is each size at the lowest temperature?  The mean `|m|` at
   `T = 1.5`, which the gate wants at or above 0.9.
2. Where does the susceptibility peak for each size?  With the sheet's
   absolute-magnetization definition

       chi(T) = L^2 (<m^2> - <|m|>^2) / T                       (Equation 10)

   the peak is found by fitting a parabola through the five grid points
   centred on the largest chi and taking its vertex.
3. What is the critical temperature?  The finite lattice shifts the peak by
   about 1/L, so two sizes cancel the leading term:

       T_c = 2 T_peak(64) - T_peak(32)                          (Equation 11)

The script prints the numbers and writes the identical text to
`week3/evidence/peaks.txt`. The chart scripts import this module so that
Equation 10 is defined exactly once.

Run:  .venv/bin/python scripts/peaks.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent

#: Onsager's exact critical temperature for the square lattice,
#: 2 / ln(1 + sqrt 2); the analysis never rounds it.
TC_EXACT = 2.0 / math.log(1.0 + math.sqrt(2.0))
T_C = TC_EXACT

#: The sheet's refinement window: inside it the long window runs are used, and
#: the coarse runs stand alone outside it.
WINDOW_LOW = 2.0
WINDOW_HIGH = 2.6

#: Folder names the four contract runs write, per lattice size.
COARSE_RUNS = {32: "coarse-l32", 64: "coarse-l64"}
WINDOW_RUNS = {32: "window-l32", 64: "window-l64"}

#: Tolerance for matching a temperature printed by two different ramps.
T_TOL = 1e-9


@dataclass(frozen=True)
class TemperatureStats:
    """Measured averages at one temperature of one run."""

    t: float
    mean_abs_m: float
    mean_m_sq: float
    n: int


@dataclass(frozen=True)
class PeakFit:
    """The parabola through five susceptibility samples and its vertex."""

    t_peak: float
    chi_peak: float
    curvature: float
    t_min: float
    t_max: float
    n_points: int
    grid_peak_t: float
    grid_peak_chi: float

    @property
    def grid_offset(self) -> float:
        """How far the fitted vertex sits from the largest sampled point."""
        return self.t_peak - self.grid_peak_t


def read_run(folder: Path) -> tuple[int, list[TemperatureStats]]:
    """Return `(L, per-temperature averages)` for one run folder.

    `series.jsonl` holds one object per measured sweep in ramp order, so the
    file is streamed once and only running sums are kept: the window runs are
    1.3 million rows each and never need to be held in memory.
    """
    run = json.loads((folder / "run.json").read_text())
    lattice = int(run["L"])
    order: list[float] = []
    sums: dict[float, list[float]] = {}
    with (folder / "series.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = float(row["T"])
            m = float(row["M"])
            acc = sums.get(t)
            if acc is None:
                order.append(t)
                acc = sums[t] = [0.0, 0.0, 0.0]
            acc[0] += abs(m)
            acc[1] += m * m
            acc[2] += 1.0
    stats = [
        TemperatureStats(t, sums[t][0] / sums[t][2], sums[t][1] / sums[t][2], int(sums[t][2]))
        for t in order
    ]
    return lattice, stats


def chi_series(lattice: int, stats: list[TemperatureStats]) -> list[tuple[float, float]]:
    """Equation 10 on each temperature of one size: `(T, chi)`."""
    return [chi_at(lattice, s) for s in stats]


def chi_at(lattice: int, stat: TemperatureStats) -> tuple[float, float]:
    """Equation 10 at one temperature."""
    variance = stat.mean_m_sq - stat.mean_abs_m * stat.mean_abs_m
    if variance < 0.0:
        # A negative sample variance can only be round-off in the two means;
        # clamp it so the log-free arithmetic downstream stays sane.
        variance = 0.0
    return stat.t, lattice * lattice * variance / stat.t


def merge_grid(
    coarse: list[TemperatureStats], window: list[TemperatureStats]
) -> list[TemperatureStats]:
    """Merge the two grids into the sheet's single 27-temperature grid.

    Inside `[2.0, 2.6]` the window rows are used wherever they exist, because
    they hold twenty times the sampling at the same contract temperatures;
    outside, and at any temperature the window grid does not carry, the coarse
    rows stand alone. Matching is on the printed temperature value.
    """
    merged: dict[float, TemperatureStats] = {}
    for stat in coarse:
        merged[_key(stat.t)] = stat
    for stat in window:
        if WINDOW_LOW - T_TOL <= stat.t <= WINDOW_HIGH + T_TOL:
            merged[_key(stat.t)] = stat
    return [merged[key] for key in sorted(merged)]


def _key(t: float) -> float:
    """Round a printed temperature to the tolerance used for matching."""
    return round(t, 9)


def fit_peak(ts: list[float], chis: list[float]) -> PeakFit:
    """Fit a parabola through the five points around the largest chi.

    The vertex of that parabola is `T_peak`. The fit is done on `T - T_v`
    with `T_v` the largest sampled temperature, so the normal equations stay
    well conditioned at the `0.05` grid spacing.
    """
    if len(ts) != len(chis):
        raise ValueError("temperature and chi lists must have the same length")
    if len(ts) < 5:
        raise ValueError("a five-point parabola needs at least five points")
    top = max(range(len(chis)), key=lambda i: chis[i])
    # Five consecutive points centred on the largest chi; when the largest chi
    # sits on the edge of the grid the window slides inside rather than
    # shrinking, so the fit always sees the same five points.
    lo = min(max(top - 2, 0), len(ts) - 5)
    hi = lo + 5
    x = np.array(ts[lo:hi], dtype=float) - ts[top]
    y = np.array(chis[lo:hi], dtype=float)
    a, b, c = np.polyfit(x, y, 2)
    if a >= 0.0:
        raise ValueError("the sampled chi has no downward parabola around its peak")
    x_vertex = -b / (2.0 * a)
    return PeakFit(
        t_peak=float(ts[top] + x_vertex),
        chi_peak=float(a * x_vertex * x_vertex + b * x_vertex + c),
        curvature=float(a),
        t_min=float(ts[lo]),
        t_max=float(ts[hi - 1]),
        n_points=hi - lo,
        grid_peak_t=float(ts[top]),
        grid_peak_chi=float(chis[top]),
    )


def onsager_abs_m(t: float) -> float:
    """Onsager's infinite-lattice `<|m|>`, Equation 3 of the sheet.

    `(1 - sinh(2/T)^-4)^(1/8)` below `T_c`, and zero at and above it. The
    argument of the root is exactly zero at `T_c`.
    """
    if t >= TC_EXACT:
        return 0.0
    return (1.0 - math.sinh(2.0 / t) ** -4) ** (1.0 / 8.0)


def size_analysis(
    artifacts: Path, lattice: int
) -> tuple[list[TemperatureStats], PeakFit, float]:
    """Load one size's two runs, merge them, and fit the susceptibility peak.

    Returns the merged per-temperature averages, the fitted peak, and the cold
    `mean |m|` at the lowest temperature of the merged grid.
    """
    _, coarse = read_run(artifacts / COARSE_RUNS[lattice])
    _, window = read_run(artifacts / WINDOW_RUNS[lattice])
    merged = merge_grid(coarse, window)
    ts = [s.t for s in merged]
    chis = [chi for _, chi in chi_series(lattice, merged)]
    fit = fit_peak(ts, chis)
    return merged, fit, merged[0].mean_abs_m


def build_report(artifacts: Path) -> str:
    """The `peaks.txt` body: cold means, fitted peaks, extrapolated `T_c`."""
    lines: list[str] = []
    lines.append("week3 part 2 - susceptibility peaks and the extrapolated critical temperature")
    lines.append("")
    lines.append(
        "chi(T) = L^2 (<m^2> - <|m|>^2) / T; window rows are used for "
        f"{WINDOW_LOW:.2f} <= T <= {WINDOW_HIGH:.2f}, coarse rows elsewhere."
    )
    lines.append("")
    peaks_by_size: dict[int, PeakFit] = {}
    for lattice in (32, 64):
        merged, fit, cold = size_analysis(artifacts, lattice)
        peaks_by_size[lattice] = fit
        lines.append(f"L = {lattice} ({len(merged)} temperatures on the merged grid)")
        lines.append(f"  cold mean |M| at T = {merged[0].t:.3f} : {cold:.4f}")
        lines.append(
            f"  largest chi on the grid at T = {fit.grid_peak_t:.3f}"
            f" : chi = {fit.grid_peak_chi:.4f}"
        )
        lines.append(
            f"  fitted peak T_peak({lattice}) over T = {fit.t_min:.3f} .. {fit.t_max:.3f}"
            f" : {fit.t_peak:.4f} (chi = {fit.chi_peak:.4f})"
        )
        lines.append("")
    t_c = 2.0 * peaks_by_size[64].t_peak - peaks_by_size[32].t_peak
    deviation = (t_c - TC_EXACT) / TC_EXACT * 100.0
    lines.append("extrapolated critical temperature")
    lines.append(f"  T_c = 2 T_peak(64) - T_peak(32) = {t_c:.4f}")
    lines.append(f"  Onsager exact T_c                = {TC_EXACT:.4f}")
    lines.append(f"  deviation                        = {deviation:+.2f}%")
    lines.append("")
    lines.append(
        "  gate: cold mean |M| >= 0.9 for each size, |T_c - 2.26919| / 2.26919 < 0.02"
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts")
    parser.add_argument("--out", type=Path, default=WEEK / "evidence" / "peaks.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.artifacts)
    print(report, end="")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)


if __name__ == "__main__":
    main()
