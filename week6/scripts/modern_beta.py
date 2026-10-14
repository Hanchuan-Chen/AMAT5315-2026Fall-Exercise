"""Part 4, VERIFY (2): inverse temperature separates averaging from retrieval.

Several queries with their modern retrievals across beta, beside Part 3's
classical recall of the same queries, all on the 100 stored digits.  Small beta
returns the same blurred average digit for every query, large beta returns the
stored digit, and classical recall returns a fragment of strokes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figlib

SCAN = "artifacts/modern/scan.npz"


def as_image(pattern: np.ndarray) -> np.ndarray:
    """A spin pattern or a real update in [-1, 1] as an image on [0, 1]."""
    return (np.clip(np.asarray(pattern), -1.0, 1.0) + 1.0) / 2.0


def main() -> None:
    data = np.load(SCAN)
    betas = data["betas"]
    targets = data["sample_targets"]
    patterns = data["sample_patterns"]
    queries = data["sample_queries"]
    classical = data["sample_classical"]
    modern = data["sample_modern"]
    n_queries = targets.size

    mosaic = [
        [as_image(queries[:, i]).reshape(28, 28) for i in range(n_queries)],
        [as_image(classical[:, i]).reshape(28, 28) for i in range(n_queries)],
    ] + [
        [as_image(modern[:, i, j]).reshape(28, 28) for i in range(n_queries)]
        for j in range(len(betas))
    ]
    labels = ["query (lower half random)", "classical Hopfield recall"] + [
        f"modern update, beta = {beta:g}" for beta in betas
    ]

    y_centers, x_centers = figlib.grid_centers(mosaic, gap=2)
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    ax.imshow(figlib.compose_grid(mosaic, gap=2), cmap="gray_r",
              interpolation="none", vmin=0, vmax=1)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"digit {t}" for t in targets], fontsize=8)
    ax.set_yticks(y_centers)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(
        "one modern update per beta beside the classical recall, on 100 stored "
        "digits:\nsmall beta averages every stored digit, large beta retrieves "
        "one, classical recall returns a fragment",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig("evidence/modern-beta.png", dpi=160)
    plt.close(fig)
    print("wrote evidence/modern-beta.png")

    print("overlap of each signed retrieval with its target:")
    for i in range(n_queries):
        target = patterns[:, i]
        modern_overlap = [
            float(np.mean(np.sign(modern[:, i, j]) * target))
            for j in range(len(betas))
        ]
        print(
            f"  digit {targets[i]}: classical "
            f"{float(np.mean(np.sign(classical[:, i]) * target)):.3f}; modern "
            + " ".join(f"{b:g}:{v:.2f}" for b, v in zip(betas, modern_overlap))
        )
    at_small = modern[:, :, 0]
    blurred = np.mean(
        [float(np.mean(np.abs(at_small[:, i, None] - at_small[:, j, None])))
         for i in range(n_queries) for j in range(n_queries) if i != j]
    )
    print(
        f"at beta=0.001 any two queries' updates differ by only {blurred:.3f} on "
        "average, so every query returns the same blurred average digit"
    )


if __name__ == "__main__":
    main()
