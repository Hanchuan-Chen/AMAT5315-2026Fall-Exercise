#!/usr/bin/env python3
"""Part 3, check (1): the migrated image, its depth profile and the adjoint field."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEEK = Path(__file__).resolve().parents[1]
FOLDER = WEEK / "artifacts" / "adjoint"
EXPERIMENT = json.loads((WEEK / "inputs" / "reflector.json").read_text())


def main():
    run = json.loads((FOLDER / "run.json").read_text())
    cfg = run["experiment"]
    image = np.load(FOLDER / "image.npy")
    perturbation = np.asarray(EXPERIMENT["perturbation"])
    spacing = cfg["dx"] * cfg["length_unit_m"] / 1000.0
    x0, x1, z0, z1 = 7, 34, 10, 24
    window = image[z0:z1, x0:x1]
    profile = np.linalg.norm(window, axis=1)
    peak_row = z0 + int(np.argmax(profile))
    true_row = int(np.argmax(np.abs(perturbation).sum(axis=1)))
    difference = abs(peak_row - true_row) * spacing
    print(f"true reflector depth {true_row * spacing:.2f} km; profile peak {peak_row * spacing:.2f} km; "
          f"difference {difference:.2f} km")
    assert abs(peak_row - 21) <= 1

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), layout="constrained")
    extent = (x0 * spacing, (x1 - 1) * spacing, (z1 - 1) * spacing, z0 * spacing)
    artists = [
        axes[0].imshow(perturbation[z0:z1, x0:x1], extent=extent, origin="upper", cmap="RdBu_r", aspect="equal"),
        axes[1].imshow(window, extent=extent, origin="upper", cmap="RdBu_r", aspect="equal"),
    ]
    axes[0].set_title("1. Known reflector")
    axes[1].set_title("2. Raw signed RTM image")
    for ax in axes[:2]:
        ax.axhline(21 * spacing, color="#333333", ls="--", lw=0.9)
    axes[0].set(
        xlabel="Horizontal position (km)",
        ylabel=f"Depth (km)\nKnown depth: {true_row * spacing:.1f} km",
    )
    axes[1].set(xlabel="Horizontal position (km)", ylabel="Depth (km)")
    fig.colorbar(artists[0], ax=axes[0], label="Velocity change (km/s)", shrink=0.85)
    fig.colorbar(artists[1], ax=axes[1], label="Image (arbitrary units)", shrink=0.85)
    depth = (np.arange(z0, z1)) * spacing
    axes[2].plot(profile, depth, color="#24557a", lw=1.4)
    axes[2].axhline(peak_row * spacing, color="#ba4b37", ls="--", lw=1.0, label=f"Peak: {peak_row * spacing:.1f} km")
    axes[2].invert_yaxis()
    axes[2].set(xlabel="Row L2 norm\n(arbitrary units)", ylabel="Depth (km)", title="3. Depth profile")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"RTM locates the reflector; depth error: {difference:.1f} km", fontsize=10)
    fig.savefig(FOLDER / "image.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    steps = run["recording"]["steps"]
    field = np.load(FOLDER / "wavefield.npy")
    frame = steps.index(132)
    depth_km = (0.0, cfg["nz"] * spacing)
    fig, ax = plt.subplots(figsize=(4.9, 4.2), layout="constrained")
    limit = float(np.max(np.abs(field[frame])))
    artist = ax.imshow(
        field[frame],
        extent=(0.0, cfg["nx"] * spacing, depth_km[1], 0.0),
        origin="upper",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )
    ax.axhline(21 * spacing, color="#555555", ls="--", lw=0.8)
    ax.set(xlabel="Horizontal position (km)", ylabel="Depth (km)", title="Adjoint field at step 132; 2.64 s")
    fig.colorbar(artist, ax=ax, label="Pressure adjoint (arbitrary units)", shrink=0.85)
    fig.savefig(FOLDER / "wavefield.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"adjoint wavefield: step 132, max |field| {limit:.4f}")


if __name__ == "__main__":
    main()
