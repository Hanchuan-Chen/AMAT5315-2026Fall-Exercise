#!/usr/bin/env python3
"""Magnetization against temperature, measured at L = 64 against Onsager.

Part 2 of the week-3 sheet. The measured points are the merged 27-temperature
grid of the contract runs, the solid line is Onsager's infinite-lattice result
(Equation 3), and the dashed vertical line is his exact critical temperature.
The finite lattice rounds the transition: the points follow the exact curve
below `T_c` and sit above it, with a tail, near and above `T_c`.

Run:  .venv/bin/python scripts/plot_magnetization.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import peaks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=peaks.WEEK / "artifacts")
    parser.add_argument(
        "--out", type=Path, default=peaks.WEEK / "evidence" / "magnetization.png"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged, fit, _ = peaks.size_analysis(args.artifacts, 64)
    ts = np.array([s.t for s in merged])
    measured = np.array([s.mean_abs_m for s in merged])

    exact_ts = np.linspace(ts.min(), peaks.T_C, 400)
    exact = np.array([peaks.onsager_abs_m(t) for t in exact_ts])

    figure, axes = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    axes.axvline(
        peaks.T_C,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label=f"$T_c$ = {peaks.T_C:.4f}",
    )
    axes.plot(exact_ts, exact, color="tab:orange", linewidth=2.0, label="Onsager, infinite lattice")
    axes.plot(ts, measured, "o-", color="tab:blue", markersize=4.0, linewidth=1.3, label="measured, $L = 64$")

    axes.set_xlabel("temperature $T$")
    axes.set_ylabel(r"$\langle |m| \rangle$")
    axes.set_xlim(1.45, 3.55)
    axes.set_ylim(0.0, 1.02)
    axes.set_title("Magnetization: the finite lattice rounds the transition")
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.legend(loc="upper right", frameon=True, fontsize=9)
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)
    print(f"wrote {args.out}")
    print(f"  {len(ts)} measured points, {ts.min():.2f} <= T <= {ts.max():.2f}")
    print(f"  Onsager curve drawn for T < T_c = {peaks.T_C:.4f}")
    for t in (2.25, 2.30, 2.60):
        i = int(np.argmin(np.abs(ts - t)))
        print(
            f"  T = {ts[i]:.2f}: measured {measured[i]:.4f}, "
            f"Onsager {peaks.onsager_abs_m(ts[i]):.4f}"
        )


if __name__ == "__main__":
    main()
