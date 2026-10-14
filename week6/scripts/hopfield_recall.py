"""Part 3, VERIFY (2): recall collapses near the storage capacity.

Plots the mean final overlap after recall and the fraction of recalls ending at
overlap at least 0.9, against the load, one curve per N, with alpha = 0.138
marked, and reports where each mean-overlap curve falls below 0.9.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCAN = "artifacts/hopfield/scan.npz"
REFERENCE = {100: 0.186, 200: 0.170, 400: 0.156, 800: 0.150}
ALPHA_C = 0.138


def crossing(alphas, values, level: float = 0.9):
    """Load where the curve drops through ``level``, by linear interpolation."""
    for i in range(len(alphas) - 1):
        if values[i] >= level > values[i + 1]:
            frac = (values[i] - level) / (values[i] - values[i + 1])
            return alphas[i] + frac * (alphas[i + 1] - alphas[i])
    return None


def main() -> None:
    data = np.load(SCAN)
    alphas = data["alphas"]
    columns = [str(name) for name in data["columns"]]
    ns = [int(n) for n in data["ns"]]
    overlap = columns.index("overlap_mean")
    fraction = columns.index("overlap_fraction_ge_0.9")

    colours = {100: "tab:blue", 200: "tab:orange", 400: "tab:green", 800: "tab:red"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for n in ns:
        rows = data[f"n{n}"]
        ax.plot(alphas, rows[:, overlap], "-o", ms=3.5, color=colours[n],
                label=f"N = {n}")
        ax2.plot(alphas, rows[:, fraction], "-o", ms=3.5, color=colours[n],
                 label=f"N = {n}")
    for axis in (ax, ax2):
        axis.set_xlabel(r"load $\alpha = P/N$")
        axis.tick_params(labelsize=8)
    ax.axvline(ALPHA_C, color="black", ls="--", lw=1.0,
               label=r"$\alpha_c = 0.138$")
    ax.set_ylabel("mean final overlap after recall")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("mean overlap collapses past the capacity", fontsize=9)
    ax2.axvline(ALPHA_C, color="black", ls="--", lw=1.0)
    ax2.set_ylabel("fraction of recalls with overlap >= 0.9")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7.5, loc="lower left")
    ax2.set_title("the recall state stops being reached", fontsize=9)
    fig.suptitle(
        "recall of random patterns against the load; the drop sharpens as N grows",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("evidence/hopfield-recall.png", dpi=160)
    plt.close(fig)
    print("wrote evidence/hopfield-recall.png")

    crossings = {}
    for n in ns:
        rows = data[f"n{n}"]
        value = crossing(alphas, rows[:, overlap])
        crossings[n] = value
        reference = REFERENCE[n]
        within = "within" if abs(value - reference) <= 0.02 else "OUTSIDE"
        print(
            f"N={n:4d}: mean overlap falls below 0.9 at alpha={value:.4f} "
            f"(sheet's reference {reference:.3f}, difference "
            f"{abs(value - reference):.4f}, {within} 0.02)"
        )
    ordered = [crossings[n] for n in ns]
    decreasing = all(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1))
    listed = ", ".join(f"{value:.4f}" for value in ordered)
    print(f"crossings decrease with N: {decreasing} ({listed})")
    for n in ns:
        rows = data[f"n{n}"]
        at_012 = rows[np.argmin(np.abs(alphas - 0.12)), overlap]
        print(
            f"N={n:4d}: overlap {at_012:.4f} at alpha=0.12, settles near "
            f"{rows[-1, overlap]:.2f} at alpha=0.25"
        )


if __name__ == "__main__":
    main()
