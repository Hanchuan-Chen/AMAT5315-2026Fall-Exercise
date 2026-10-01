#!/usr/bin/env python3
"""Part 2, checks (2) and (3): shot gathers, wavefield and echo frames."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEEK = Path(__file__).resolve().parents[1]
FOLDER = WEEK / "artifacts" / "forward"


def main():
    run = json.loads((FOLDER / "run.json").read_text())
    result = json.loads((FOLDER / "result.json").read_text())
    cfg = run["experiment"]
    traces = np.load(FOLDER / "traces.npy")
    assert traces.shape == (len(cfg["shots"]), cfg["steps"], len(cfg["receivers"]))
    assert traces.dtype == np.float64 and np.isfinite(traces).all()
    scale = cfg["length_unit_m"] / 1000.0
    spacing = cfg["dx"] * scale
    dt = cfg["dt"] * cfg["time_unit_s"]
    receiver_x = np.asarray(cfg["receivers"])[:, 0] * spacing
    time = (np.arange(cfg["steps"]) + 1) * dt
    step_x = float(np.diff(receiver_x)[0])
    extent = (
        receiver_x[0] - step_x / 2,
        receiver_x[-1] + step_x / 2,
        time[-1] + dt / 2,
        time[0] - dt / 2,
    )
    limit = float(np.max(np.abs(traces)))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(traces), figsize=(9.2, 3.5), sharex=True, sharey=True, layout="constrained")
    for shot, ax in enumerate(np.atleast_1d(axes)):
        image = ax.imshow(
            traces[shot], extent=extent, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest"
        )
        ax.set(
            xlabel="Receiver position (km)",
            title=f"Shot {shot}; source x = {cfg['shots'][shot][0] * spacing:.1f} km",
        )
        if shot == 0:
            ax.set_ylabel("Time (s)")
        step, receiver = np.unravel_index(np.argmax(np.abs(traces[shot])), traces[shot].shape)
        print(
            f"shot {shot}: L2 {np.linalg.norm(traces[shot]):.6f}; peak {traces[shot, step, receiver]:.8f} "
            f"at trace index {step}, receiver {receiver} (t = {time[step]:.2f} s)"
        )
    fig.colorbar(image, ax=list(np.atleast_1d(axes)), label="Pressure (common scale; arbitrary units)", shrink=0.9)
    fig.savefig(FOLDER / "gathers.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"forward L2 norm of all traces: {np.linalg.norm(traces):.6f} (reference 11.574770)")

    steps = run["recording"]["steps"]
    wavefield = np.load(FOLDER / "wavefield.npy")
    echo = np.load(FOLDER / "echo.npy")
    assert wavefield.shape == echo.shape
    frame = steps.index(150)
    extent_km = (0.0, cfg["nx"] * spacing, cfg["nz"] * spacing, 0.0)
    for field, name, title, unit in (
        (wavefield[frame], "wavefield.png", "Forward wavefield; shot 0, t = 3.00 s", "Pressure (arbitrary units)"),
        (echo[frame], "echo.png", "Reflector echo; shot 0, t = 3.00 s", "Pressure difference (arbitrary units)"),
    ):
        fig, ax = plt.subplots(figsize=(4.9, 4.2), layout="constrained")
        limit = float(np.max(np.abs(field)))
        image = ax.imshow(
            field, extent=extent_km, origin="upper", cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="equal"
        )
        ax.plot(np.asarray(cfg["shots"])[:, 0] * spacing, np.asarray(cfg["shots"])[:, 1] * spacing, "v", color="#111111", ms=6)
        ax.plot(
            np.asarray(cfg["receivers"])[:, 0] * spacing,
            np.asarray(cfg["receivers"])[:, 1] * spacing,
            ".",
            color="#333333",
            ms=4,
        )
        ax.axhline(21 * spacing, color="#555555", ls="--", lw=0.8)
        ax.set(xlabel="Horizontal position (km)", ylabel="Depth (km)", title=title)
        fig.colorbar(image, ax=ax, label=unit, shrink=0.85)
        fig.savefig(FOLDER / name, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"{name}: step 150, max |field| {limit:.4f}")
    print(f"echo/direct ratio at step 150: {np.max(np.abs(echo[frame])) / np.max(np.abs(wavefield[frame])):.3f}")


if __name__ == "__main__":
    main()
