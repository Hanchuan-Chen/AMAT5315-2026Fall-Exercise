#!/usr/bin/env python3
"""Energy on either side of both stability limits, Part 3, check (2).

Part 3 of the week-4 sheet. The runs under `artifacts/scan/` are the two
Taylor-Green steps 0.032 and 0.033, the two random RK4 steps that bracket the
measured advective limit, and forward Euler at 0.01 on the same random field.
Every one of them saves a snapshot every 0.5 time units; the unstable ones stop
at their first non-finite energy.

The script prints the largest speed of the random initial field, the Equation 17
bound it implies, and the stopping time of each run that stopped early, then
draws the energy against time with a logarithmic energy axis, one panel per
case, labelling each curve with its method and time step.

Run:  .venv/bin/python scripts/blowup.py
Writes:  week4/evidence/blowup.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent


def read_tsv(path: Path) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Time, energy, and the stopping time if the run ended non-finite."""
    times, energies, stopped = [], [], None
    with path.open() as handle:
        next(handle)
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            t = float(parts[0])
            energy = float(parts[1])
            if not np.isfinite(energy):
                stopped = t
                continue
            times.append(t)
            energies.append(energy)
    return np.array(times), np.array(energies), stopped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts")
    parser.add_argument("--out", type=Path, default=WEEK / "evidence" / "blowup.png")
    args = parser.parse_args()

    field = subprocess.run(
        ["field", "random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    initial = json.loads(field)
    u = np.array(initial["u"])
    v = np.array(initial["v"])
    largest_speed = float(np.sqrt(u * u + v * v).max())
    n = initial["n"]
    longest = np.sqrt(2.0) * (n // 3)
    bound = 2.83 / (largest_speed * longest)
    print(f"largest speed of the random initial field: {largest_speed:.4f}")
    print(f"|k|max = sqrt(2) * floor({n}/3) = {longest:.2f}")
    print(f"Equation 17 bound: 2.83 / ({largest_speed:.4f} * {longest:.2f}) = {bound:.4f}")

    scan = args.artifacts / "scan"
    taylor = [
        ("RK4, $\\Delta t = 0.032$", scan / "tg-rk4-0.032.tsv", "C0", "-"),
        ("RK4, $\\Delta t = 0.033$", scan / "tg-rk4-0.033.tsv", "C3", "-"),
    ]
    random = [
        ("RK4, $\\Delta t = 0.030$", scan / "random-rk4-0.030.tsv", "C0", "-"),
        ("RK4, $\\Delta t = 0.034$", scan / "random-rk4-0.034.tsv", "C3", "-"),
        ("Euler, $\\Delta t = 0.010$", scan / "random-euler-0.010.tsv", "C2", ":"),
    ]
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    stopped_at = {}
    for axis, series, name, title in (
        (
            left,
            taylor,
            "Taylor-Green",
            r"Taylor-Green, $n = 64$, $\nu = 0.1$, predicted $\Delta t = 0.0316$",
        ),
        (
            right,
            random,
            "random",
            r"random flow, $n = 128$, $\nu = 0.004$, predicted $\Delta t = "
            + f"{bound:.4f}$",
        ),
    ):
        for label, path, color, style in series:
            times, energies, stopped = read_tsv(path)
            axis.semilogy(times, energies, style, color=color, label=label, lw=1.4)
            if stopped is not None:
                axis.axvline(stopped, color=color, ls=":", lw=0.8)
                clean = label.replace("$", "").replace("\\Delta t", "dt").replace("\\", "")
                stopped_at.setdefault(name, []).append((stopped, color, clean))
                print(f"{name}: {clean} stopped at t = {stopped:g}")
        axis.set_xlabel(r"time $t$")
        axis.set_ylabel(r"energy $E(t)$")
        axis.set_title(title, fontsize=9)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8, loc="lower left")
        for index, (stopped, color, clean) in enumerate(stopped_at.get(name, [])):
            axis.annotate(
                f"{clean}: non-finite at t = {stopped:g}",
                xy=(stopped, 0.45),
                xycoords=("data", "axes fraction"),
                xytext=(0.32, 0.62 - 0.08 * index),
                textcoords="axes fraction",
                fontsize=7,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.6),
            )
    exact_t = np.linspace(0.0, 8.0, 200)
    left.semilogy(
        exact_t,
        0.25 * np.exp(-0.4 * exact_t),
        "k--",
        lw=1.0,
        label="exact $E = \\frac{1}{4}e^{-0.4t}$",
    )
    left.legend(fontsize=8, loc="lower left")
    figure.tight_layout()
    figure.savefig(args.out, dpi=140)


if __name__ == "__main__":
    main()
