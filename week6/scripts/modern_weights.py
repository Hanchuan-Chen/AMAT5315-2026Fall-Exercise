"""Part 4, VERIFY (1): the modern update is one softmax-attention step.

Plots the softmax weights over the stored patterns for one query at three beta
values and prints the largest difference between the modern update and JAX's
softmax-attention output over all queries, P and beta.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCAN = "artifacts/modern/scan.npz"
RUN = "artifacts/modern/run.json"
REFERENCE_WEIGHTS = {0.003: 0.01, 0.03: 0.25, 0.3: 1.00}


def main() -> None:
    data = np.load(SCAN)
    with open(RUN) as handle:
        record = json.load(handle)
    weights = data["weights"]
    betas = data["weight_betas"]
    target = int(data["weight_target"])
    query = int(data["weight_query"])

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    colours = {0.003: "tab:blue", 0.03: "tab:orange", 0.3: "tab:red"}
    for j, beta in enumerate(betas):
        ax.plot(np.arange(weights.shape[0]), weights[:, j], "-o", ms=3,
                color=colours[float(beta)], label=f"beta = {beta:g}")
    ax.axvline(target, color="black", ls="--", lw=1.0,
               label=f"queried pattern {target}")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 3)
    ax.set_xlabel("stored pattern index")
    ax.set_ylabel("softmax weight over stored patterns")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(
        "weights of the modern update for one corrupted query at three betas:\n"
        "uniform blending, the queried digit emerging, then one-hot retrieval",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig("evidence/modern-weights.png", dpi=160)
    plt.close(fig)
    print("wrote evidence/modern-weights.png")

    print(
        f"query {query} targets stored pattern {target}; its weight is "
        + ", ".join(
            f"{float(weights[target, j]):.4f} at beta={float(betas[j]):g}"
            for j in range(len(betas))
        )
    )
    for j, beta in enumerate(betas):
        reference = REFERENCE_WEIGHTS[float(beta)]
        print(
            f"  beta={float(beta):g}: sheet's reference weight "
            f"{reference:.2f}, measured {float(weights[target, j]):.4f}, "
            f"{int(np.sum(weights[:, j] > 0.1))} patterns above 0.1"
        )
    print(
        "largest |modern update - softmax attention| over all queries, P and "
        f"beta: {record['attention_max_abs_difference']:.2e} (required below 1e-5; "
        "the library default scale 1/sqrt(784) would give about 2, so scale=beta)"
    )


if __name__ == "__main__":
    main()
