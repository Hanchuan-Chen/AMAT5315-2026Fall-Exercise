#!/usr/bin/env python3
"""Susceptibility against temperature for L = 32 and L = 64.

Part 2 of the week-3 sheet. `chi(T) = L^2 (<m^2> - <|m|>^2) / T` (Equation 10)
is imported from `peaks.py` so the chart and the reported numbers share one
definition. The dashed line is Onsager's exact `T_c`; the dotted lines are the
two fitted peaks, which move down and grow as `L` doubles.

Run:  .venv/bin/python scripts/plot_susceptibility.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import peaks  # noqa: E402

COLORS = {32: "tab:green", 64: "tab:blue"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=peaks.WEEK / "artifacts")
    parser.add_argument(
        "--out", type=Path, default=peaks.WEEK / "evidence" / "susceptibility.png"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure, axes = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    axes.axvline(
        peaks.T_C,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label=f"$T_c$ = {peaks.T_C:.4f}",
    )

    report: list[str] = []
    for lattice in (32, 64):
        merged, fit, _ = peaks.size_analysis(args.artifacts, lattice)
        ts = np.array([s.t for s in merged])
        chis = np.array([chi for _, chi in peaks.chi_series(lattice, merged)])
        axes.plot(
            ts,
            chis,
            "o-",
            color=COLORS[lattice],
            markersize=4.0,
            linewidth=1.3,
            label=f"$L = {lattice}$, peak {fit.t_peak:.4f}",
        )
        axes.axvline(fit.t_peak, color=COLORS[lattice], linestyle=":", linewidth=1.1)
        report.append(
            f"  L = {lattice}: grid peak {fit.grid_peak_chi:.3f} at T = {fit.grid_peak_t:.3f},"
            f" fitted peak {fit.chi_peak:.3f} at T_peak = {fit.t_peak:.4f},"
            f" curvature {fit.curvature:.1f}"
        )

    axes.set_xlabel("temperature $T$")
    axes.set_ylabel(r"$\chi(T)$")
    axes.set_xlim(1.9, 2.9)
    axes.set_title("Susceptibility: the peak sharpens and moves toward $T_c$ with size")
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.legend(loc="upper right", frameon=True, fontsize=9)
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)
    print(f"wrote {args.out}")
    for line in report:
        print(line)


if __name__ == "__main__":
    main()
