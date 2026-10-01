#!/usr/bin/env python3
"""Part 4, check (3): four Marmousi panels and the image norm."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEEK = Path(__file__).resolve().parents[1]
ART = WEEK / "artifacts"


def main():
    experiment = json.loads((WEEK / "inputs" / "marmousi.json").read_text())
    born = np.load(ART / "marmousi-born" / "born_data.npy")
    image = np.load(ART / "marmousi-image" / "image.npy")
    stats = json.loads((ART / "marmousi-image" / "result.json").read_text())["statistics"]
    background = np.asarray(experiment["background"])
    perturbation = np.asarray(experiment["perturbation"])
    spacing = experiment["dx"] * experiment["length_unit_m"] / 1000.0
    extent = (0.0, experiment["nx"] * spacing, experiment["nz"] * spacing, 0.0)
    shots = np.asarray(experiment["shots"])
    shot = int(np.argmin(np.abs(shots[:, 0] - 10.0 / spacing)))
    receivers = np.asarray(experiment["receivers"])
    receiver_km = receivers[:, 0] * spacing
    time = (np.arange(experiment["steps"]) + 1) * experiment["dt"] * experiment["time_unit_s"]

    limit = float(np.max(np.abs(image)))
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.1), layout="constrained")
    panels = [
        (background, "Smoothed Marmousi background", "Speed (km/s)", "viridis"),
        (perturbation, "Short-wavelength perturbation", "Velocity perturbation (km/s)", "RdBu_r"),
        (image, "Checkpointed migration image (one amplitude scale)", "Adjoint image (arbitrary units)", "RdBu_r"),
    ]
    for ax, (field, title, label, cmap) in zip(axes[:3], panels):
        artist = ax.imshow(field, extent=extent, origin="upper", cmap=cmap, aspect="auto")
        ax.set(xlabel="Horizontal position (km)", ylabel="Depth (km)", title=title)
        fig.colorbar(artist, ax=ax, label=label, shrink=0.85)
    axes[2].set_title("Checkpointed migration image")

    gather = born[shot]
    step_x = float(np.diff(receiver_km)[0])
    gather_extent = (
        receiver_km[0] - step_x / 2,
        receiver_km[-1] + step_x / 2,
        time[-1] + time[1] / 2,
        time[1] - time[1] / 2,
    )
    glimit = float(np.max(np.abs(gather)))
    artist = axes[3].imshow(
        gather, extent=gather_extent, origin="upper", cmap="RdBu_r", vmin=-glimit, vmax=glimit, aspect="auto"
    )
    axes[3].set(
        xlabel="Receiver position (km)",
        ylabel="Time (s)",
        title=f"Born gather; source x = {shots[shot, 0] * spacing:.1f} km",
    )
    fig.colorbar(artist, ax=axes[3], label="Scattered pressure (arbitrary units)", shrink=0.85)
    fig.savefig(ART / "marmousi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    norm = float(np.linalg.norm(image))
    print(f"Marmousi image L2 norm: {norm:.7e} (reference 6.7037741e-4, relative {abs(norm - 6.7037741e-4) / 6.7037741e-4:.2e})")
    print(
        f"peak saved states {stats['peak_saved_states']}, peak saved bytes {stats['peak_saved_bytes']:,}; "
        f"full history would hold {(experiment['steps'] + 1) * 2 * experiment['nx'] * experiment['nz'] * 8:,} bytes per shot"
    )
    cosine = float(np.sum(image * perturbation) / np.linalg.norm(image) / np.linalg.norm(perturbation))
    print(f"whole-domain cosine similarity with the perturbation: {cosine:.3f}")
    assert abs(norm - 6.7037741e-4) / 6.7037741e-4 < 1e-4


if __name__ == "__main__":
    main()
