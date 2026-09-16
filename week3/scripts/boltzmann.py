#!/usr/bin/env python3
"""Energy histograms of two hot runs and the log ratio that tests Boltzmann.

For two temperatures on the same lattice the density of states cancels in the
probability ratio, so

    ln(P_{T_b}(E) / P_{T_a}(E)) = (1/T_a - 1/T_b) E + const.

Only the slope is predicted: 1/3.0 - 1/3.1 = 0.0107527 per energy unit at
L = 64. The script bins the total energy (E from series.jsonl times L^2) with
40-unit bins, plots both histograms and the log ratio, and draws the predicted
slope as a dashed line through the ratio points (least-squares intercept, slope
held at the prediction).

Run:  .venv/bin/python scripts/boltzmann.py \
          --cold runs/T3.0 --hot runs/T3.1 --out evidence/boltzmann.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent

BIN_WIDTH = 40.0
MIN_ROWS = 5
SOLID_ROWS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold", type=Path, default=WEEK / "runs" / "T3.0")
    parser.add_argument("--hot", type=Path, default=WEEK / "runs" / "T3.1")
    parser.add_argument("--out", type=Path, default=WEEK / "evidence" / "boltzmann.png")
    return parser.parse_args()


def read_run(folder: Path) -> tuple[int, float, np.ndarray]:
    """Return (L, T, total energy per sweep) for one run folder."""
    run = json.loads((folder / "run.json").read_text())
    lattice = int(run["L"])
    energies = []
    temperature = None
    with (folder / "series.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            temperature = float(row["T"])
            energies.append(float(row["E"]) * lattice * lattice)
    return lattice, temperature, np.asarray(energies)


def bin_edges(energies: np.ndarray) -> np.ndarray:
    low = math.floor(energies.min() / BIN_WIDTH) * BIN_WIDTH
    high = math.ceil(energies.max() / BIN_WIDTH) * BIN_WIDTH
    return np.arange(low, high + BIN_WIDTH, BIN_WIDTH)


def main() -> None:
    args = parse_args()
    lattice, t_cold, e_cold = read_run(args.cold)
    hot_lattice, t_hot, e_hot = read_run(args.hot)
    assert lattice == hot_lattice, "both runs must use the same lattice"

    edges = bin_edges(np.concatenate([e_cold, e_hot]))
    centres = 0.5 * (edges[:-1] + edges[1:])
    n_cold, _ = np.histogram(e_cold, bins=edges)
    n_hot, _ = np.histogram(e_hot, bins=edges)

    predicted = 1.0 / t_cold - 1.0 / t_hot
    both = (n_cold >= MIN_ROWS) & (n_hot >= MIN_ROWS)
    x = centres[both]
    y = np.log(n_hot[both] / n_cold[both])

    # Least-squares slope for information; the drawn line holds the prediction.
    free_slope, free_intercept = np.polyfit(x, y, 1)
    intercept = float(np.mean(y - predicted * x))
    residual = y - (predicted * x + intercept)
    solid = (n_cold[both] >= SOLID_ROWS) & (n_hot[both] >= SOLID_ROWS)

    print(f"L = {lattice}, measured sweeps per temperature: {len(e_cold)} and {len(e_hot)}")
    print(
        f"T = {t_cold} and {t_hot}: total energy "
        f"{e_cold.mean():.1f} +/- {e_cold.std(ddof=1):.1f} and "
        f"{e_hot.mean():.1f} +/- {e_hot.std(ddof=1):.1f}"
    )
    print(f"bins of {BIN_WIDTH:g} energy units: {len(edges) - 1} total, {int(both.sum())} kept")
    print(f"predicted slope 1/{t_cold} - 1/{t_hot} = {predicted:.7f}")
    print(f"free least-squares slope over the kept bins = {free_slope:.7f} (intercept {free_intercept:.3f})")
    print(f"ratio-point residual against the predicted line: max {np.abs(residual).max():.3f}")
    print(
        f"where both bins hold >= {SOLID_ROWS} sweeps: max "
        f"{np.abs(residual[solid]).max():.3f}, rms {np.sqrt(np.mean(residual[solid] ** 2)):.3f}"
    )

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)

    left.step(centres, n_cold, where="mid", label=f"T = {t_cold:g}", color="#3b7dd8")
    left.step(centres, n_hot, where="mid", label=f"T = {t_hot:g}", color="#d1495b")
    left.set_xlabel("total energy $E$")
    left.set_ylabel(f"sweeps in a {BIN_WIDTH:g}-wide bin")
    left.set_title("Energy histograms")
    left.legend()

    right.axhline(0.0, color="#999999", linewidth=0.8)
    right.plot(x, y, "o", markersize=6, color="#1f2a44", label="log ratio (both bins $\\geq$ 5)")
    line_x = np.array([x.min(), x.max()])
    right.plot(
        line_x,
        predicted * line_x + intercept,
        "--",
        color="#d1495b",
        label=f"slope {predicted:.7f} (predicted)",
    )
    right.set_xlabel("total energy $E$")
    right.set_ylabel(r"$\ln(P_{3.1}/P_{3.0})$")
    right.set_title("Log ratio of the histograms")
    right.legend()

    figure.suptitle(
        f"L = {lattice}, {len(e_cold)} sweeps each: slope "
        f"$1/{t_cold:g} - 1/{t_hot:g} = {predicted:.7f}$",
        fontsize=10,
    )
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
