#!/usr/bin/env python3
"""The step refinement and the chosen step, Part 4, check (2).

Part 4 of the week-4 sheet. The random flow on `128 x 128`, `nu = 0.004`, to
`t = 2` (seed 2026, wavenumbers 2 to 6) is run with RK4 at `dt = 0.02, 0.0125`
and `0.01`, and compared with a reference run at `dt = 0.0025` on the same
grid. The script reports the relative error of omega at `t = 2` for every run
and the log-log slope, estimates the relative temporal error at `dt = 0.01`
with the fourth-order Richardson formula (Equation 18), predicts the error at
the three candidate steps, and chooses the largest one below `5e-6`.

Run:  .venv/bin/python scripts/refinement.py
Writes:  week4/evidence/convergence.json
         week4/evidence/convergence.png
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

CANDIDATES = (0.02, 0.0125, 0.01)
REFERENCE = 0.0025
TARGET = 5e-6


def last_frame(path: Path) -> dict:
    frame = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                frame = json.loads(line)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts", type=Path, default=WEEK / "artifacts" / "convergence"
    )
    parser.add_argument(
        "--json", type=Path, default=WEEK / "evidence" / "convergence.json"
    )
    parser.add_argument(
        "--out", type=Path, default=WEEK / "evidence" / "convergence.png"
    )
    args = parser.parse_args()

    def load(dt: float) -> np.ndarray:
        return np.array(last_frame(args.artifacts / f"rk4-dt{dt}" / "fields.jsonl")["omega"])

    reference = load(REFERENCE)
    fields = {dt: load(dt) for dt in CANDIDATES}
    errors = {
        dt: float(np.linalg.norm(fields[dt] - reference) / np.linalg.norm(reference))
        for dt in CANDIDATES
    }
    steps = np.array(CANDIDATES)
    series = np.array([errors[dt] for dt in CANDIDATES])
    slope = float(np.polyfit(np.log(steps), np.log(series), 1)[0])

    # Equation 18: the difference of two steps estimates the finer one's error.
    difference = float(
        np.linalg.norm(fields[0.02] - fields[0.01]) / np.linalg.norm(fields[0.01])
    )
    richardson = difference / (2.0**4 - 1.0)
    predicted = {dt: richardson * (dt / 0.01) ** 4 for dt in CANDIDATES}
    choice = max(
        (dt for dt in CANDIDATES if predicted[dt] < TARGET), default=None
    )

    print("relative error of omega at t = 2 against the dt = 0.0025 reference:")
    for dt in CANDIDATES:
        print(f"  dt = {dt:<7g} {errors[dt]:.3e}")
    print(f"fitted log-log slope: {slope:.3f}  (required 3.7 <= q <= 4.3)")
    print(
        "Richardson estimate at dt = 0.01: "
        f"||w(0.02) - w(0.01)|| / (15 ||w(0.01)||) = {richardson:.3e}"
    )
    for dt in CANDIDATES:
        verdict = "below" if predicted[dt] < TARGET else "above"
        print(
            f"  dt = {dt:<7g} predicted {predicted[dt]:.3e} ({verdict} {TARGET:.0e}), "
            f"measured {errors[dt]:.3e}"
        )
    if choice is not None:
        print(
            f"chosen step: dt = {choice:g}, predicted {predicted[choice]:.3e}, "
            f"measured {errors[choice]:.3e}"
        )

    report = {
        "reference_dt": REFERENCE,
        "errors": {f"{dt:g}": errors[dt] for dt in CANDIDATES},
        "slope": slope,
        "richardson_estimate_at_0.01": richardson,
        "predicted_errors": {f"{dt:g}": predicted[dt] for dt in CANDIDATES},
        "target": TARGET,
        "chosen_dt": choice,
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.loglog(
        steps,
        series,
        "o",
        label=f"RK4, slope {slope:.3f}",
    )
    fit = np.exp(np.polyval(np.polyfit(np.log(steps), np.log(series), 1), np.log(steps)))
    axis.loglog(steps, fit, "-", color="C0", lw=1.0)
    if choice is not None:
        axis.plot([choice], [errors[choice]], "*", ms=14, color="C3", label=f"chosen $\\Delta t = {choice:g}$")
    axis.set_xlabel(r"time step $\Delta t$")
    axis.set_ylabel(r"relative error of $\omega$ at $t = 2$")
    axis.set_title(r"random flow, $128 \times 128$, $\nu = 0.004$, reference $\Delta t = 0.0025$")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.out, dpi=140)


if __name__ == "__main__":
    main()
