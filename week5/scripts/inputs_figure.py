#!/usr/bin/env python3
"""Part 2, check (1): plot the reflector experiment and its source pulse."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEEK = Path(__file__).resolve().parents[1]
ART = WEEK / "artifacts"


def main():
    data = json.loads((WEEK / "inputs" / "reflector.json").read_text())
    nx, nz = data["nx"], data["nz"]
    scale = data["length_unit_m"] / 1000.0
    background = np.asarray(data["background"])
    perturbation = np.asarray(data["perturbation"])
    shots = np.asarray(data["shots"])
    receivers = np.asarray(data["receivers"])
    x = (np.arange(nx) + 0.5) * data["dx"] * scale
    z = (np.arange(nz) + 0.5) * data["dx"] * scale
    extent = (x[0], x[-1], z[-1], z[0])
    sponge = data["sponge_width"] * data["dx"] * scale

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")
    image = axes[0].imshow(
        background + perturbation, extent=extent, origin="upper", cmap="viridis", aspect="equal"
    )
    axes[0].contour(
        perturbation, levels=[0.05], colors="#f2f2f2", linewidths=1.0, extent=extent, origin="upper"
    )
    axes[0].plot(shots[:, 0] * scale, shots[:, 1] * scale, "v", color="#ffffff", mec="#111111", ms=8, label="Sources")
    axes[0].plot(receivers[:, 0] * scale, receivers[:, 1] * scale, ".", color="#ffd166", mec="#111111", ms=6, label="Receivers")
    axes[0].axhline(sponge, color="#dddddd", ls="--", lw=0.9)
    axes[0].text(
        0.08,
        sponge - 0.12,
        "dashed: sponge inner edge",
        color="#eeeeee",
        fontsize=6.5,
        bbox=dict(facecolor="#3a3a3a", edgecolor="none", pad=1.2),
    )
    axes[0].set(xlabel="Horizontal position (km)", ylabel="Depth (km)", title="Seismic acquisition")
    axes[0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.colorbar(image, ax=axes[0], label="Speed (km/s)", shrink=0.85)

    steps = np.arange(data["steps"])
    time = steps * data["dt"] * data["time_unit_s"]
    a = np.pi * data["source_frequency"] * (steps * data["dt"] - data["source_peak_time"])
    pulse = data["source_amplitude"] * (1 - 2 * a * a) * np.exp(-a * a)
    axes[1].plot(time, pulse, color="#24557a", lw=1.4)
    axes[1].axhline(0.0, color="#cccccc", lw=0.6)
    axes[1].set(
        xlabel="Time (s)",
        ylabel="Source pulse $g(t)$",
        title=f"Ricker pulse; peak frequency {data['source_frequency'] / data['time_unit_s']:.1f} Hz, "
        f"peak time {data['source_peak_time'] * data['time_unit_s']:.1f} s",
    )
    fig.savefig(ART / "inputs.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    reflector_depth = int(np.argmax(np.abs(perturbation).sum(axis=1))) * data["dx"] * scale
    print(f"reflector depth: {reflector_depth:.1f} km; sponge inner edge {sponge:.1f} km")
    print(f"shots: {shots[:, 0] * scale}; receivers: {receivers.shape[0]} at depth {receivers[0, 1] * scale:.1f} km")
    strong = np.flatnonzero(np.abs(pulse) >= 1e-3)
    print(f"pulse peak {pulse.max():.4f} at {time[int(np.argmax(pulse))]:.2f} s; "
          f"troughs {pulse.min():.4f}; last sample above 1e-3 at {time[strong[-1]]:.1f} s")


if __name__ == "__main__":
    main()
