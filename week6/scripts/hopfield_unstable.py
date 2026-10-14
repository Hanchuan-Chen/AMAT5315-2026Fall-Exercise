"""Part 3, VERIFY (1): the unstable-bit fraction follows the crosstalk curve.

Plots the unstable-bit fraction against the load for each N, with standard
errors across the 20 pattern sets, beside the Gaussian crosstalk prediction
``Phi(-1/sqrt(alpha))`` of Eq. 15 and the finite-N form
``Phi(-sqrt((N-1)/(P-1)))``.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCAN = "artifacts/hopfield/scan.npz"
REFERENCE = {"N=800, alpha=0.20": 1.25e-2, "N=100, alpha=0.10": 3.5e-4}


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def main() -> None:
    data = np.load(SCAN)
    alphas = data["alphas"]
    columns = [str(name) for name in data["columns"]]
    ns = [int(n) for n in data["ns"]]
    mean = columns.index("unstable_mean")
    se = columns.index("unstable_se")

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    curve_alpha = np.linspace(0.05, 0.25, 400)
    ax.plot(curve_alpha, [phi(-1 / math.sqrt(a)) for a in curve_alpha], "k-",
            lw=1.4, label=r"large-$N$ prediction $\Phi(-1/\sqrt{\alpha})$")
    colours = {100: "tab:blue", 200: "tab:orange", 400: "tab:green", 800: "tab:red"}
    for n in ns:
        rows = data[f"n{n}"]
        ax.errorbar(alphas, rows[:, mean], yerr=rows[:, se], color=colours[n],
                    marker="o", ms=4, capsize=2.5, lw=1.0, label=f"N = {n}")
        finite = [
            phi(-math.sqrt((n - 1) / max(a * n - 1, 1))) for a in curve_alpha
        ]
        ax.plot(curve_alpha, finite, color=colours[n], ls=":", lw=1.0)
    ax.plot([], [], color="0.4", ls=":", lw=1.0,
            label=r"finite-$N$ form $\Phi(-\sqrt{(N-1)/(P-1)})$")
    ax.set_xlabel(r"load $\alpha = P/N$")
    ax.set_ylabel("unstable-bit fraction")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.0)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(
        "unstable bits follow the Gaussian crosstalk prediction;\n"
        "the finite-N points sit below it at small N",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig("evidence/hopfield-unstable.png", dpi=160)
    plt.close(fig)
    print("wrote evidence/hopfield-unstable.png")

    print(r"prediction Phi(-1/sqrt(alpha)) and the finite-N form per N:")
    for n in ns:
        rows = data[f"n{n}"]
        for target in (0.10, 0.20):
            index = int(np.argmin(np.abs(alphas - target)))
            alpha = float(alphas[index])
            p = alpha * n
            print(
                f"  N={n:4d} alpha={alpha:.2f}: measured "
                f"{rows[index, mean]:.3e}+/-{rows[index, se]:.1e}, "
                f"large-N {phi(-1 / math.sqrt(alpha)):.3e}, "
                f"finite-N {phi(-math.sqrt((n - 1) / (p - 1))):.3e}"
            )
    for name, reference in REFERENCE.items():
        n = int(name.split(",")[0].split("=")[1])
        alpha = float(name.split("alpha=")[1])
        index = int(np.argmin(np.abs(alphas - alpha)))
        print(
            f"sheet's reference {name}: {reference:.2e}, measured "
            f"{data[f'n{n}'][index, mean]:.3e}, ratio "
            f"{data[f'n{n}'][index, mean] / reference:.3f}"
        )
    print(
        "at alpha=0.25 the four N read "
        + ", ".join(
            f"N={n}: {data[f'n{n}'][-1, mean]:.3e}" for n in ns
        )
    )


if __name__ == "__main__":
    main()
