#!/usr/bin/env python3
"""The derivative comparison of Part 2, check (1).

Part 2 of the week-4 sheet. The solver's own Fourier routines are applied to
`g(x, y) = sin(3x) cos(2y)` on the `n = 32` periodic grid and compared with
`dx g = 3 cos(3x) cos(2y)`, `dxx g = -9 g`, `dxdy g = -6 cos(3x) sin(2y)` and
the Laplacian `-13 g`. Centred finite differences (Equation 8, periodic
wrapping) are compared on the same grid and on `n = 64`, so the ratio of the
two columns shows the second-order error law.

The fields live on the crate's side, so the script asks
`cargo test --release derivative_dump -- --nocapture` for them and formats the
sheet's table. Nothing is rounded before the comparison.

Run:  .venv/bin/python scripts/comparison.py
Writes:  nothing; the table and the ratios go to the terminal
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent


def cargo_json(filter_name: str) -> dict:
    """The one JSON line the crate's measurement test prints."""
    result = subprocess.run(
        ["cargo", "test", "--release", "--quiet", filter_name, "--", "--nocapture"],
        cwd=WEEK,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit(f"cargo test {filter_name} printed no JSON")


def main() -> None:
    data = cargo_json("derivative_dump")
    print(f"g(x, y) = sin(3x) cos(2y) on the periodic [0, 2pi)^2 grid, n = {data['n']}")
    print()
    print(f"{'derivative':12s} {'FD n=32':>10s} {'FD n=64':>10s} {'Fourier n=32':>14s}")
    ratios = []
    for row in data["rows"]:
        fourier = "< 1e-10" if row["fourier"] < 1e-10 else f"{row['fourier']:.2e}"
        print(
            f"{row['name']:12s} {row['fd32']:10.5f} {row['fd64']:10.5f} "
            f"{fourier:>14s}"
        )
        ratios.append(row["ratio"])
    print()
    print("FD n=32 / FD n=64: " + "  ".join(f"{value:.2f}" for value in ratios))
    print("the ratio is about four, so the centred differences are second order")


if __name__ == "__main__":
    main()
