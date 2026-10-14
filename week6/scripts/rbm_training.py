"""Part 2, VERIFY (2): the CD surrogate and the held-out one-step
reconstruction error measure different things.

Reads ``artifacts/rbm/run.json`` written by ``scripts/rbm_train.py`` and plots
the epoch mean of the batch CD surrogates, Eq. 10, and the held-out one-step
reconstruction error (the Hamming fraction of changed pixels), for CD-1 and
CD-10, with the value at initialization marked.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = "artifacts/rbm/run.json"


def main() -> None:
    with open(RUN) as handle:
        record = json.load(handle)
    runs = record["runs"]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    recon_ax = ax.twinx()
    colours = {"cd1": "tab:blue", "cd10": "tab:red"}
    for label in ("cd1", "cd10"):
        history = runs[label]["history"]
        epochs = np.asarray(history["epoch"], dtype=float)
        surrogate = np.asarray(history["surrogate"], dtype=float)
        recon = np.asarray(history["reconstruction_error"], dtype=float)
        ax.plot(
            epochs[1:], surrogate[1:], "-o", ms=3, color=colours[label],
            label=f"{label.upper()} CD surrogate",
        )
        recon_ax.plot(
            epochs, recon, "--s", ms=3, color=colours[label], alpha=0.6,
            label=f"{label.upper()} held-out one-step error (Hamming fraction)",
        )
        recon_ax.plot(
            epochs[0], recon[0], "*", ms=13, color=colours[label],
            markeredgecolor="black", markeredgewidth=0.4,
        )

    init_errors = [runs[label]["history"]["reconstruction_error"][0] for label in ("cd1", "cd10")]
    final_errors = [runs[label]["history"]["reconstruction_error"][-1] for label in ("cd1", "cd10")]
    ax.axhline(0.0, color="0.8", lw=0.7)
    ax.set_xlabel("epoch (0 = initialization, the star)")
    ax.set_ylabel("CD surrogate  mean(F(v0) - F(stop_gradient(v10)))")
    recon_ax.set_ylabel("held-out one-step reconstruction error")
    recon_ax.set_ylim(0, 0.2)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = recon_ax.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2, labels1 + labels2, fontsize=7, loc="upper right"
    )
    ax.set_title(
        "CD-1 and CD-10: the surrogate need not decrease while held-out\n"
        "one-step reconstruction improves (star = initialization)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig("evidence/rbm-training.png", dpi=160)
    plt.close(fig)
    print("wrote evidence/rbm-training.png")

    print(
        f"shared initialization: the two models start from reconstruction error "
        f"{init_errors[0]:.5f} and {init_errors[1]:.5f} "
        f"(difference {abs(init_errors[0] - init_errors[1]):.2e})"
    )
    print(
        "same training settings except the chain length: "
        f"images {record['rbm_toml']['training']['images']}, "
        f"epochs {record['rbm_toml']['training']['epochs']}, "
        f"batch {record['rbm_toml']['training']['batch_size']}, "
        f"SGD lr {record['rbm_toml']['training']['learning_rate']}, "
        f"seed {record['rbm_toml']['training']['seed']}, "
        f"init seed {record['rbm_toml']['initialization']['seed']}, "
        f"chains {runs['cd1']['chain_length']} and {runs['cd10']['chain_length']}"
    )
    for label, init, final in zip(("cd1", "cd10"), init_errors, final_errors):
        surrogate = np.asarray(runs[label]["history"]["surrogate"][1:])
        print(
            f"{label}: reconstruction error {init:.5f} -> {final:.5f} "
            f"({100 * final / init:.1f}% of its initialization value, "
            f"{'below' if final < 0.5 * init else 'NOT below'} half); "
            f"surrogate rises from {surrogate[0]:.3f} to {surrogate.max():.3f} "
            f"at epoch {int(surrogate.argmax()) + 1} and ends at {surrogate[-1]:.3f}"
        )


if __name__ == "__main__":
    main()
