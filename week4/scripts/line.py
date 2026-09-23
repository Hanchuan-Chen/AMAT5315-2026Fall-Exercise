#!/usr/bin/env python3
"""The stability map and the pulse experiments of Part 1.

Part 1 of the week-4 sheet. The measurements come from the crate itself:
`scripts/line.py` asks `cargo test --release line_dump -- --nocapture` for the
growth factor RK4 measures on `y' = z y` at `h = 1`, for the line's modes
`lambda_k h` at the steps 0.045 and 0.056, and for the pulse runs.

The left panel of `line-stability.png` is that measured growth factor on a log
colour scale, the solid black curve is `|R_RK4(z)| = 1` (Equation 11) and the
dashed ones are Euler's and the midpoint rule's, with the modes of the line at
`nu = 0.05`, `n = 64`, `c = 1` as dots. The other two panels are the Gaussian
pulse of standard deviation 0.35, centred at `pi/2` and periodic, integrated
by the crate's RK4 to `t = 6` on either side of the exact limit 0.0494.

`line-accuracy.png` holds the two checks of Part 1 (b): one lap of the
`sigma = 0.25` pulse with the exact solution, and the maximum error at `t = 1`
against the step for forward Euler, the midpoint rule, RK4 and the
equal-weight RK4.

Run:  .venv/bin/python scripts/line.py
Writes:  week4/evidence/line-stability.png
         week4/evidence/line-accuracy.png
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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


def stability(z):
    """Equation 11: the stability function of each method."""
    return {
        "euler": 1.0 + z,
        "midpoint": 1.0 + z + z**2 / 2.0,
        "rk4": 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0,
    }


def stability_figure(data: dict) -> plt.Figure:
    z = data["z"]
    re = np.linspace(z["re_min"], z["re_max"], z["n_re"])
    im = np.linspace(z["im_min"], z["im_max"], z["n_im"])
    growth = np.array(z["growth"]).reshape(z["n_im"], z["n_re"])
    x, y = np.meshgrid(re, im)
    curves = stability(x + 1j * y)

    fig = plt.figure(figsize=(13.0, 4.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.30)
    ax = fig.add_subplot(grid[0, 0])
    mesh = ax.pcolormesh(
        re,
        im,
        np.log10(np.maximum(growth, 1e-12)),
        shading="nearest",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    fig.colorbar(mesh, ax=ax, label=r"$\log_{10}|R(z)|$, measured by RK4")
    ax.contour(x, y, np.abs(curves["rk4"]), levels=[1.0], colors="black")
    ax.contour(
        x, y, np.abs(curves["euler"]), levels=[1.0], colors="cyan", linestyles="--"
    )
    ax.contour(
        x, y, np.abs(curves["midpoint"]), levels=[1.0], colors="lime", linestyles="--"
    )
    modes45 = np.array(data["modes"]["0.045"]).reshape(-1, 2)
    modes56 = np.array(data["modes"]["0.056"]).reshape(-1, 2)
    ax.plot(
        modes45[:, 0],
        modes45[:, 1],
        "o",
        mfc="none",
        mec="white",
        ms=3.0,
        label=r"modes, $\Delta t = 0.045$",
    )
    ax.plot(
        modes56[:, 0],
        modes56[:, 1],
        "s",
        mfc="none",
        mec="white",
        ms=2.6,
        label=r"modes, $\Delta t = 0.056$",
    )
    ax.plot([], [], "k-", label=r"RK4, $|R|=1$")
    ax.plot([], [], "c--", label=r"Euler, $|R|=1$")
    ax.plot([], [], "g--", label=r"midpoint, $|R|=1$")
    ax.axhline(0.0, color="0.8", lw=0.6)
    ax.axvline(0.0, color="0.8", lw=0.6)
    ax.plot([-2.785], [0.0], "wo", ms=4)
    ax.annotate(
        r"$-2.785$",
        xy=(-2.785, 0.0),
        xytext=(-4.2, 0.35),
        color="white",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="white", lw=0.7),
    )
    ax.plot([0.0, 0.0], [2.83, -2.83], "wo", ms=4)
    ax.annotate(
        r"$\pm 2.83i$",
        xy=(0.0, 2.83),
        xytext=(-1.6, 2.6),
        color="white",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="white", lw=0.7),
    )
    ax.set_xlim(re[0], re[-1])
    ax.set_ylim(-3.3, 3.3)
    ax.set_xlabel(r"$\mathrm{Re}\, z = \mathrm{Re}\,\lambda_k h$")
    ax.set_ylabel(r"$\mathrm{Im}\, z$")
    ax.set_title("the stability map, measured")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    panels = []
    for column, step in ((1, "0.045"), (2, "0.056")):
        axis = fig.add_subplot(grid[0, column])
        pulse = data["pulses"][step]
        times = np.array(pulse["t"])
        fields = np.array(pulse["u"]).reshape(times.size, -1)
        mesh = axis.imshow(
            fields,
            aspect="auto",
            origin="upper",
            extent=(0.0, 2.0 * np.pi, times[-1], 0.0),
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
        )
        fig.colorbar(mesh, ax=axis, label=r"$u(x,t)$")
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$t$")
        axis.set_title(rf"RK4, $\Delta t = {step}$")
        panels.append(axis)
    fig.suptitle(
        "the line at $\\nu = 0.05$, $n = 64$, $c = 1$: measured growth and the pulse"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def accuracy_figure(data: dict) -> tuple[plt.Figure, dict, dict]:
    panel_a = data["panel_a"]
    x = np.array(panel_a["x"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    left.plot(x, panel_a["exact"], "k-", lw=2.5, label="exact")
    left.plot(x, panel_a["rk4_fourier"], "C0--", label=r"RK4, Fourier, $\Delta t=0.02$")
    left.plot(x, panel_a["rk4_fd"], "C3-", lw=1.2, label=r"RK4, centred, $\Delta t=0.02$")
    left.plot(
        x, panel_a["euler_fourier"], "C2:", label=r"Euler, Fourier, $\Delta t=0.005$"
    )
    left.set_xlabel(r"$x$")
    left.set_ylabel(r"$u$ at $t = 2\pi$")
    left.set_title("one lap: spatial against time error")
    left.legend(fontsize=8)

    panel_b = data["panel_b"]
    steps = np.array(panel_b["steps"])
    errors = {name: np.array(values) for name, values in panel_b["errors"].items()}
    labels = {
        "euler": "Euler",
        "rk2": "midpoint",
        "rk4": "RK4",
        "rk4_equal": "RK4, equal weights",
    }
    colors = {"euler": "C2", "rk2": "C1", "rk4": "C0", "rk4_equal": "C4"}
    slopes = {}
    for name, error in errors.items():
        slope = float(np.polyfit(np.log(steps), np.log(error), 1)[0])
        slopes[name] = slope
        right.loglog(
            steps,
            error,
            "o-",
            color=colors[name],
            label=f"{labels[name]}, slope {slope:.2f}",
        )
    right.set_xlabel(r"time step $\Delta t$")
    right.set_ylabel(r"max error at $t = 1$")
    right.set_title("the four error slopes")
    right.legend(fontsize=8)
    fig.tight_layout()
    return fig, panel_a["errors"], slopes


def main() -> None:
    data = cargo_json("line_dump")
    evidence = WEEK / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    curves = stability
    for step in ("0.045", "0.056"):
        modes = np.array(data["modes"][step]).reshape(-1, 2)
        z = modes[:, 0] + 1j * modes[:, 1]
        magnitude = np.abs(curves(z)["rk4"])
        moving = magnitude[np.abs(z) > 0.0]
        outside = int((moving > 1.0).sum())
        print(
            f"modes at dt = {step}: max |R| = {moving.max():.4f} over the "
            f"travelling modes, {outside} of {moving.size} outside the region"
        )
    # The exact limit: the map above is the parabola stretched by h, so
    # bisect h until its tip touches |R| = 1.
    modes = np.array(data["modes"]["0.045"]).reshape(-1, 2)
    lam = modes[:, 0] / 0.045 + 1j * modes[:, 1] / 0.045
    lam = lam[np.abs(lam) > 0.0]

    def worst(h: float) -> float:
        return float(np.abs(curves(lam * h)["rk4"]).max())

    low, high = 0.0, 0.2
    for _ in range(60):
        middle = 0.5 * (low + high)
        if worst(middle) <= 1.0:
            low = middle
        else:
            high = middle
    print(f"exact stability limit of the line: h = {low:.6f}")
    print(
        "|R| at the axis crossings: "
        f"real {np.abs(curves(-2.785 + 0j)['rk4']):.4f}, "
        f"imaginary {np.abs(curves(2.83j)['rk4']):.4f}"
    )
    stability_figure(data).savefig(evidence / "line-stability.png", dpi=140)
    plt.close("all")
    figure, errors, slopes = accuracy_figure(data)
    figure.savefig(evidence / "line-accuracy.png", dpi=140)
    print("Part 1 (b), maximum error at t = 2 pi:")
    for name, value in errors.items():
        print(f"  {name:14s} {value:.3e}")
    print("Part 1 (b), fitted log-log slopes:")
    for name, value in slopes.items():
        print(f"  {name:14s} {value:.3f}")


if __name__ == "__main__":
    main()
