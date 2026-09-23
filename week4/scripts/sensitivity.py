#!/usr/bin/env python3
"""Two runs a perturbation apart, Part 3, check (2).

Part 3 of the week-4 sheet. Each case is run twice with RK4 at `dt = 0.01` to
`t = 20`, snapshots every 0.5 time units. The first run starts from the field
the crate writes; the second starts from the same field plus

    delta omega(x, y) = -7e-5 * M * cos(3x) * cos(4y)

with `M` the largest absolute value among that case's initial `u` and `v`. No
command-line flag injects a perturbation, so the script does what the sheet
describes by hand: it takes the vorticity of the field the crate writes,
adds the ripple, keeps the Poisson solve consistent, and pipes the perturbed
field into `fluid`. Both runs write their frames under `artifacts/sensitivity/`.

The figure is `||omega_1 - omega_2|| / ||omega_1||` against time on a log scale.

Run:  .venv/bin/python scripts/sensitivity.py
Writes:  week4/evidence/sensitivity.png
         week4/artifacts/sensitivity/ (both runs of both cases)
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

CASES = {
    "taylor-green": {
        "field": ["field", "taylor-green", "--n", "64"],
        "nu": 0.1,
    },
    "random": {
        "field": [
            "field", "random", "--n", "128", "--seed", "2026", "--k-min", "2",
            "--k-max", "6",
        ],
        "nu": 0.004,
    },
}


def wavenumbers(n: int):
    k = np.fft.fftfreq(n) * n
    return np.meshgrid(k, k)


def vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    kx, ky = wavenumbers(u.shape[0])
    return np.real(
        np.fft.ifft2(1j * kx * np.fft.fft2(v) - 1j * ky * np.fft.fft2(u))
    )


def velocity(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kx, ky = wavenumbers(omega.shape[0])
    k2 = kx * kx + ky * ky
    psi = np.fft.fft2(omega) / np.where(k2 == 0.0, 1.0, k2)
    return (
        np.real(np.fft.ifft2(1j * ky * psi)),
        np.real(np.fft.ifft2(-1j * kx * psi)),
    )


def run_fluid(field: dict, out: Path, nu: float) -> Path:
    tsv = out.parent / (out.name + ".tsv")
    with tsv.open("w") as handle:
        subprocess.run(
            [
                "fluid", "--method", "rk4", "--nu", str(nu), "--dt", "0.01",
                "--t-end", "20", "--every", "0.5", "--out", str(out),
            ],
            input=json.dumps(field),
            text=True,
            stdout=handle,
            check=True,
        )
    return out / "fields.jsonl"


def distances(a: Path, b: Path) -> tuple[np.ndarray, np.ndarray]:
    times, values = [], []
    with a.open() as first, b.open() as second:
        for left, right in zip(first, second):
            one, two = json.loads(left), json.loads(right)
            omega_1 = np.array(one["omega"])
            omega_2 = np.array(two["omega"])
            times.append(one["t"])
            values.append(
                float(np.linalg.norm(omega_1 - omega_2) / np.linalg.norm(omega_1))
            )
    return np.array(times), np.array(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts", type=Path, default=WEEK / "artifacts" / "sensitivity"
    )
    parser.add_argument(
        "--out", type=Path, default=WEEK / "evidence" / "sensitivity.png"
    )
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name, case in CASES.items():
        base = json.loads(
            subprocess.run(case["field"], check=True, capture_output=True, text=True).stdout
        )
        (args.artifacts / f"{name}.json").write_text(json.dumps(base))
        n = base["n"]
        u = np.array(base["u"]).reshape(n, n)
        v = np.array(base["v"]).reshape(n, n)
        scale = float(max(np.abs(u).max(), np.abs(v).max()))
        omega = vorticity(u, v)
        x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        xx, yy = np.meshgrid(x, x)
        ripple = -7e-5 * scale * np.cos(3.0 * xx) * np.cos(4.0 * yy)
        u2, v2 = velocity(omega + ripple)
        perturbed = dict(base, u=u2.ravel().tolist(), v=v2.ravel().tolist())
        (args.artifacts / f"{name}-perturbed.json").write_text(json.dumps(perturbed))

        plain = run_fluid(base, args.artifacts / name, case["nu"])
        shaken = run_fluid(perturbed, args.artifacts / f"{name}-perturbed", case["nu"])
        times, values = distances(plain, shaken)
        print(
            f"{name}: M = {scale:.4f}, "
            f"||d omega|| / ||omega|| = {values[0]:.3e} at t = 0 and "
            f"{values[-1]:.3e} at t = {times[-1]:g} "
            f"(x{values[-1] / values[0]:.1f})"
        )
        axis.semilogy(times, values, label=name)
    axis.set_xlabel(r"time $t$")
    axis.set_ylabel(r"$\|\omega_1 - \omega_2\| / \|\omega_1\|$")
    axis.set_title("the same step, a physical perturbation apart")
    axis.set_ylim(1e-8, 2e-2)
    axis.annotate(
        "dips to zero: the two stored frames agree at six decimals",
        xy=(0.33, 0.06),
        xycoords="axes fraction",
        fontsize=7,
        color="0.35",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.out, dpi=140)


if __name__ == "__main__":
    main()
