"""Part 4, VERIFY (3): modern retrieval survives loads where classical recall fails.

Plots retrieval success against P for the classical recall and for the modern
update at every beta, and adds a gallery of targets, queries and both
retrievals at P = 100.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figlib

SCAN = "artifacts/modern/scan.npz"
RUN = "artifacts/modern/run.json"
REFERENCE = {"beta_0.3_minimum": 0.90, "beta_0.1_at_1000": 0.91}


def as_image(pattern: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(pattern), -1.0, 1.0) + 1.0) / 2.0


def main() -> None:
    data = np.load(SCAN)
    with open(RUN) as handle:
        record = json.load(handle)
    ps = data["ps"]
    betas = data["betas"]
    success = data["success_modern"]
    classical = data["success_classical"]
    targets = data["sample_targets"]
    patterns = data["sample_patterns"]
    queries = data["sample_queries"]
    classical_retrieval = data["sample_classical"]
    modern = data["sample_modern"]

    fig = plt.figure(figsize=(12.0, 5.2))
    ax_gallery = fig.add_axes([0.135, 0.10, 0.26, 0.74])
    mosaic = [
        [as_image(patterns[:, i]).reshape(28, 28) for i in range(targets.size)],
        [as_image(queries[:, i]).reshape(28, 28) for i in range(targets.size)],
        [as_image(classical_retrieval[:, i]).reshape(28, 28)
         for i in range(targets.size)],
        [as_image(modern[:, i, betas.tolist().index(0.3)]).reshape(28, 28)
         for i in range(targets.size)],
    ]
    y_centers, x_centers = figlib.grid_centers(mosaic, gap=2)
    ax_gallery.imshow(figlib.compose_grid(mosaic, gap=2), cmap="gray_r",
                      interpolation="none", vmin=0, vmax=1)
    ax_gallery.set_yticks(y_centers)
    ax_gallery.set_yticklabels(
        ["target", "query", "classical recall", "modern, beta = 0.3"], fontsize=8
    )
    ax_gallery.set_xticks([])
    ax_gallery.set_title("at P = 100: targets, queries and both\nretrievals",
                         fontsize=9)

    ax = fig.add_axes([0.50, 0.13, 0.47, 0.72])
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(betas)))
    for j, beta in enumerate(betas):
        ax.plot(ps, success[:, j], "-o", ms=4, color=colours[j],
                label=f"modern, beta = {beta:g}")
    ax.plot(ps, classical, "--s", ms=4, color="black", label="classical Hopfield")
    ax.set_xscale("log")
    ax.set_xticks(list(ps))
    ax.set_xticklabels([str(p) for p in ps])
    ax.set_xlabel("number of stored patterns P")
    ax.set_ylabel("retrieval success fraction")
    ax.set_ylim(-0.04, 1.05)
    ax.legend(fontsize=7)
    ax.set_title(
        "modern retrieval survives the load; classical recall never succeeds",
        fontsize=9,
    )
    fig.savefig("evidence/modern-retrieval.png", dpi=160)
    plt.close(fig)
    print("wrote evidence/modern-retrieval.png")

    index = int(np.argmin(np.abs(betas - 0.3)))
    minimum = float(success[:, index].min())
    print(
        f"modern success at beta=0.3: "
        + ", ".join(f"P={p}:{value:.2f}" for p, value in zip(ps, success[:, index]))
        + f"; minimum {minimum:.2f}, "
        + ("at least 0.90 for every P as required"
           if minimum >= REFERENCE["beta_0.3_minimum"] else "BELOW 0.90")
    )
    index = int(np.argmin(np.abs(betas - 0.1)))
    print(
        f"modern success at beta=0.1: "
        + ", ".join(f"P={p}:{value:.2f}" for p, value in zip(ps, success[:, index]))
        + f" (the sheet's reference falls to {REFERENCE['beta_0.1_at_1000']:.0%} "
        "at P = 1000)"
    )
    print(
        f"classical success: "
        + ", ".join(f"P={p}:{value:.2f}" for p, value in zip(ps, classical))
        + ("; zero at every P as required" if classical.max() == 0.0
           else "; NOT zero")
    )
    print(
        "seed and settings: " + json.dumps(
            {k: record[k] for k in ("queries", "query_seed", "recall_seed")}
        )
    )


if __name__ == "__main__":
    main()
