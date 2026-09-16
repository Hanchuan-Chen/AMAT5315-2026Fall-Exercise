#!/usr/bin/env python3
"""Integrated autocorrelation time of |M| against temperature, both sizes.

Part 3 of the week-3 sheet, check (4). The critical slowing down chart:
`tau_int` of `|M|` from Equation 13 against temperature for `L = 32` and
`L = 64`, on a logarithmic vertical axis so the flat order-one parts and the
spike at the transition can share one panel.

The autocorrelation time is the one `errors.py` computes, on the same
27-temperature merged grid the error table uses: window rows inside
`[2.0, 2.6]`, coarse rows outside. Away from the transition single-flip
Metropolis is nearly uncorrelated sweep to sweep; at the transition the chain
remembers for hundreds of sweeps, and it remembers longer on the larger
lattice, which is why Part 2 spends twenty times more sweeps in the window.

Run:  .venv/bin/python scripts/tau.py
Writes:  week3/evidence/tau.png  (and prints the two peaks)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import errors  # noqa: E402
import peaks  # noqa: E402

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent

COLORS = {32: "tab:green", 64: "tab:blue"}

#: The sheet's reference peaks, quoted only to check "within a factor of two".
REFERENCE_PEAKS = {32: 190.0, 64: 670.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts")
    parser.add_argument("--out", type=Path, default=WEEK / "evidence" / "tau.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = errors.merged_rows(args.artifacts)

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
        size_rows = [row for row in rows if row.L == lattice]
        ts = np.array([row.T for row in size_rows])
        taus = np.array([row.tau for row in size_rows])
        axes.plot(
            ts,
            taus,
            "o-",
            color=COLORS[lattice],
            markersize=4.0,
            linewidth=1.3,
            label=f"$L$ = {lattice}",
        )
        peak = size_rows[int(np.argmax(taus))]
        far = [row.tau for row in size_rows if row.T <= 1.6 or row.T >= 3.4]
        report.append(
            f"  L = {lattice}: peak tau_int = {peak.tau:.2f} sweeps at T = {peak.T:.2f};"
            f" flat value near the ends = {min(far):.2f} to {max(far):.2f} sweeps;"
            f" rise = {peak.tau / max(far):.0f}x; reference peak"
            f" {REFERENCE_PEAKS[lattice]:.0f}, ratio {peak.tau / REFERENCE_PEAKS[lattice]:.2f}"
        )

    axes.set_yscale("log")
    axes.set_xlabel("temperature $T$")
    axes.set_ylabel(r"$\tau_{int}$ (sweeps)")
    axes.set_xlim(1.45, 3.55)
    axes.set_title("Integrated autocorrelation time of $|M|$: critical slowing down")
    axes.grid(alpha=0.25, linewidth=0.6, which="both")
    axes.legend(loc="upper right", frameon=True, fontsize=9)
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)
    print(f"wrote {args.out}")
    for line in report:
        print(line)
    print("  required: flat order-one values away from T_c, a rise by more than a")
    print("  factor of a hundred near it, the L = 64 spike taller, and both peaks")
    print("  within a factor of two of the sheet's reference (670 and 190 sweeps).")


if __name__ == "__main__":
    main()
