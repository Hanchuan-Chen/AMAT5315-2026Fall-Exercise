#!/usr/bin/env python3
"""The Taylor-Green decay of Part 2, check (2).

Part 2 of the week-4 sheet. The run in `artifacts/taylor-green/` was produced by

    field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.01 \\
        --t-end 1 --every 0.1 --out artifacts/taylor-green

and the exact field at `t = 1` by

    field taylor-green --n 64 --nu 0.1 --t 1 > artifacts/taylor-green/exact-t1.json

The script prints the relative error of the last stored velocity against that
exact field, and draws the vorticity at `t = 0` and `t = 1` with velocity
arrows and one colour scale for both panels.

Run:  .venv/bin/python scripts/taylor_green.py
Writes:  week4/evidence/taylor-green.png
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


def frames(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def relative_error(computed: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(computed - exact) / np.linalg.norm(exact))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts", type=Path, default=WEEK / "artifacts" / "taylor-green"
    )
    parser.add_argument(
        "--out", type=Path, default=WEEK / "evidence" / "taylor-green.png"
    )
    args = parser.parse_args()

    stored = frames(args.artifacts / "fields.jsonl")
    exact = json.loads((args.artifacts / "exact-t1.json").read_text())
    last = stored[-1]
    velocity = np.concatenate([np.array(last["u"]), np.array(last["v"])])
    reference = np.concatenate([np.array(exact["u"]), np.array(exact["v"])])
    error = relative_error(velocity, reference)
    print(f"last stored frame: t = {last['t']:.6f}")
    print(f"relative velocity error against the exact field: {error:.3e}  (required < 1e-5)")

    n = int(round(np.sqrt(len(last["omega"]))))
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    keep = np.zeros((n, n), dtype=bool)
    keep[::4, ::4] = True
    fields = [stored[0], last]
    scale = float(np.abs(np.array(last["omega"])).max())
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for axis, frame in zip(axes, fields):
        omega = np.array(frame["omega"]).reshape(n, n)
        mesh = axis.pcolormesh(
            x, x, omega, shading="auto", cmap="RdBu_r", vmin=-scale, vmax=scale
        )
        u = np.array(frame["u"]).reshape(n, n)
        v = np.array(frame["v"]).reshape(n, n)
        axis.quiver(
            xx[keep],
            yy[keep],
            u[keep],
            v[keep],
            color="k",
            scale=28,
            width=0.004,
        )
        axis.set_title(rf"$t = {frame['t']:.1f}$, max $|\omega| = {np.abs(omega).max():.3f}$")
        axis.set_xlabel(r"$x$")
        axis.set_xticks([0, np.pi, 2 * np.pi])
        axis.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
        axis.set_yticks([0, np.pi, 2 * np.pi])
        axis.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    axes[0].set_ylabel(r"$y$")
    figure.colorbar(mesh, ax=axes, label=r"$\omega$", fraction=0.046)
    figure.suptitle(r"Taylor-Green at $\nu = 0.1$: the exact solution, colour on a fixed scale")
    figure.savefig(args.out, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
