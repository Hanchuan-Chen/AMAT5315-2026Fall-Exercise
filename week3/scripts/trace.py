#!/usr/bin/env python3
"""The raw |m| series at the critical and hot temperatures, L = 64.

Part 3 of the week-3 sheet, check (1). The absolute magnetization of the first
2000 recorded sweeps at T = 2.3 and T = 3.0, drawn so the two memory regimes
can be compared by eye: away from the transition the trace is noise around its
mean, at the transition it drifts in excursions of hundreds of sweeps.

Sources, stated because the critical window and the coarse ramp cover
different temperatures:

    T = 2.3  -> window-l64 (the window grid is 2.0 .. 2.6)
    T = 3.0  -> coarse-l64 (3.0 is outside the window grid)

The script also prints the mean of each 2000-sweep window, the long-run mean
over the whole measured series, and the effective sample count
`n_eff = n / (2 tau_int)` of Equation 14, so the claim is not left to the eye.

Run:  .venv/bin/python scripts/trace.py
Writes:  week3/evidence/trace.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import errors  # noqa: E402

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent

#: Which run folder holds each drawn temperature, and the colour to draw it in.
SOURCES = [
    (2.3, "window-l64", "tab:blue", "the window run covers 2.0 .. 2.6"),
    (3.0, "coarse-l64", "tab:red", "3.0 is outside the window grid"),
]

#: How many recorded sweeps the sheet asks for.
NSHOW = 2000


def series_at(artifacts: Path, folder: str, t_wanted: float) -> tuple[int, np.ndarray]:
    """`(L, signed M series)` at one temperature of one run folder."""
    lattice, series = errors.read_series(artifacts / folder)
    for t, m in series:
        if abs(t - t_wanted) < 1e-9:
            return lattice, m
    raise SystemExit(f"no T = {t_wanted} in {folder}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts")
    parser.add_argument("--out", type=Path, default=WEEK / "evidence" / "trace.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure, axes = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    report: list[str] = []
    for t_wanted, folder, color, why in SOURCES:
        lattice, m = series_at(args.artifacts, folder, t_wanted)
        trace = np.abs(m)[:NSHOW]
        axes.plot(
            np.arange(1, trace.size + 1),
            trace,
            color=color,
            linewidth=0.8,
            label=f"$T$ = {t_wanted}",
        )
        tau = errors.tau_int(np.abs(m))
        report.append(
            f"  T = {t_wanted}: source {folder} (L = {lattice}, {why}),"
            f" mean |m| over the first {trace.size} sweeps = {trace.mean():.4f},"
            f" over all {m.size} = {np.abs(m).mean():.4f}, tau_int = {tau:.2f} sweeps,"
            f" n_eff = {m.size / (2.0 * tau):.1f}"
        )

    axes.set_xlabel("measurement sweep")
    axes.set_ylabel(r"$|m|$")
    axes.set_xlim(0, NSHOW)
    axes.set_ylim(0.0, 0.8)
    axes.set_title(
        "$|m|$ sweep by sweep at $L$ = 64,"
        f" the first {NSHOW} measured sweeps"
    )
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
