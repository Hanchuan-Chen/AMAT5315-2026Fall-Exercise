#!/usr/bin/env python3
"""RK4's accuracy order on the fluid, Part 4, check (1).

Part 4 of the week-4 sheet. The three runs under `artifacts/order/` are
Taylor-Green on the `n = 8` grid, `nu = 0.5`, to `t = 2`, with `dt = 0.4, 0.25`
and `0.2`; every run is stable, since `nu |k|^2 dt = 1.6 < 2.785`. The script
compares each final velocity field with the exact solution at `t = 2` and fits
the log-log slope of the error against the step.

Run:  .venv/bin/python scripts/order.py
Writes:  week4/evidence/order.png
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

STEPS = (0.4, 0.25, 0.2)


def last_frame(path: Path) -> dict:
    frame = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                frame = json.loads(line)
    return frame


def exact_taylor_green(n: int, nu: float, t: float):
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    decay = np.exp(-2.0 * nu * t)
    u = np.outer(np.sin(x), np.cos(x)) * decay
    v = -np.outer(np.cos(x), np.sin(x)) * decay
    return u.ravel(), v.ravel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts" / "order")
    parser.add_argument("--out", type=Path, default=WEEK / "evidence" / "order.png")
    args = parser.parse_args()

    n, nu, t_end = 8, 0.5, 2.0
    exact_u, exact_v = exact_taylor_green(n, nu, t_end)
    reference = np.concatenate([exact_u, exact_v])
    errors = []
    for dt in STEPS:
        frame = last_frame(args.artifacts / f"rk4-dt{dt}" / "fields.jsonl")
        velocity = np.concatenate([np.array(frame["u"]), np.array(frame["v"])])
        errors.append(float(np.linalg.norm(velocity - reference) / np.linalg.norm(reference)))
    errors = np.array(errors)
    slope = float(np.polyfit(np.log(STEPS), np.log(errors), 1)[0])
    for dt, error in zip(STEPS, errors):
        print(f"dt = {dt:<5g} relative velocity error at t = 2: {error:.3e}")
    print(f"fitted log-log slope: {slope:.2f}  (required within 15% of 4)")

    # The frames are stored to six decimals, so the relative error cannot fall
    # much below this floor.
    floor = (
        np.sqrt(reference.size)
        * (1e-6 / np.sqrt(12.0))
        / np.linalg.norm(reference)
    )
    print(f"six-decimal storage floor: {floor:.3e}")
    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.loglog(STEPS, errors, "o-", label=f"RK4, slope {slope:.2f}")
    axis.axhline(
        floor, color="0.5", ls="--", lw=1.0, label="six-decimal storage floor"
    )
    axis.set_xlabel(r"time step $\Delta t$")
    axis.set_ylabel(r"relative velocity error at $t = 2$")
    axis.set_title(r"Taylor-Green on $8 \times 8$, $\nu = 0.5$")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.out, dpi=140)


if __name__ == "__main__":
    main()
