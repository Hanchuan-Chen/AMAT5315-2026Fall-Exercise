#!/usr/bin/env python3
"""The random flow at four times, Part 3, check (1).

Part 3 of the week-4 sheet. The run in `artifacts/random/` was produced by

    field random --n 128 --seed 2026 --k-min 2 --k-max 6 \\
        | fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 \\
            --out artifacts/random > artifacts/random.tsv

The script prints the first two lines and the last line of `artifacts/random.tsv`
and draws the vorticity at `t = 0, 2, 5, 10` in one row with one colour scale.
Each frame is labelled with the energy and enstrophy recomputed from its own
stored fields, Equation 14.

Run:  .venv/bin/python scripts/random.py
Writes:  week4/evidence/random.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent

WANTED = (0.0, 2.0, 5.0, 10.0)


def read_frames(path: Path) -> dict[float, dict]:
    wanted = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for target in WANTED:
                if abs(row["t"] - target) < 1e-6:
                    wanted[target] = row
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts")
    parser.add_argument("--out", type=Path, default=WEEK / "evidence" / "random.png")
    args = parser.parse_args()

    lines = (args.artifacts / "random.tsv").read_text().splitlines()
    print("artifacts/random.tsv, head -2:")
    for line in lines[:2]:
        print("  " + line)
    print("artifacts/random.tsv, tail -1:")
    print("  " + lines[-1])

    frames = read_frames(args.artifacts / "random" / "fields.jsonl")
    n = int(round(np.sqrt(len(frames[0.0]["omega"]))))
    scale = float(np.abs(np.array(frames[0.0]["omega"])).max())
    figure, axes = plt.subplots(1, 4, figsize=(16.0, 4.4))
    mesh = None
    for axis, t in zip(axes, WANTED):
        frame = frames[t]
        omega = np.array(frame["omega"]).reshape(n, n)
        u = np.array(frame["u"])
        v = np.array(frame["v"])
        energy = 0.5 * float(np.mean(u * u + v * v))
        enstrophy = 0.5 * float(np.mean(omega * omega))
        mesh = axis.pcolormesh(
            np.linspace(0, 2 * np.pi, n, endpoint=False),
            np.linspace(0, 2 * np.pi, n, endpoint=False),
            omega,
            shading="auto",
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
        )
        axis.set_title(f"$t = {t:g}$\n$E = {energy:.3f}$, $Z = {enstrophy:.3f}$")
        axis.set_xticks([0, np.pi, 2 * np.pi])
        axis.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
        axis.set_yticks([0, np.pi, 2 * np.pi])
        axis.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
        axis.set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    if mesh is not None:
        figure.colorbar(
            mesh,
            ax=axes,
            label=rf"$\omega$, fixed scale $\pm {scale:.2f}$",
            fraction=0.02,
        )
    figure.suptitle(
        r"the random flow at $n = 128$, $\nu = 0.004$, seed 2026, band $2 \leq |k| \leq 6$",
        y=1.04,
    )
    figure.savefig(args.out, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
