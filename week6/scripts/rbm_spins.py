"""Part 2, VERIFY (1): the image is a configuration of visible spins.

Left: the first 16 training images binarized at half intensity, drawn as
visible bits.  Right: one enlarged spin configuration s = 2v - 1, where ink is
+1 and background is -1.  Bottom: the RBM's two layers and their couplings,
784 visible bits joined to 128 hidden units and nothing inside a layer.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import datasets
import figlib
import rbm_lib


def main() -> None:
    cfg = rbm_lib.load_config()
    x_train, _, _, _ = datasets.read_mnist()
    v = np.asarray(rbm_lib.binarize(x_train[:16]))
    print(
        f"visible layer {rbm_lib.N_VISIBLE} bits = 28x28 pixels, hidden layer "
        f"{rbm_lib.N_HIDDEN} units, couplings {rbm_lib.N_VISIBLE * rbm_lib.N_HIDDEN}"
    )
    print(f"binarized ink fraction of the first 16 training images: {v.mean():.4f}")
    spins = rbm_lib.spins_of(v).reshape(-1, 28, 28)
    print(f"spin values present: {sorted(set(np.unique(spins).tolist()))}")

    fig = plt.figure(figsize=(11.0, 5.6))
    ax_gallery = fig.add_axes([0.02, 0.56, 0.42, 0.36])
    ax_gallery.imshow(
        figlib.compose_grid([list(row) for row in v.reshape(4, 4, 28, 28)]),
        cmap="gray_r",
        interpolation="none",
        vmin=0,
        vmax=1,
    )
    ax_gallery.set_xticks([])
    ax_gallery.set_yticks([])
    ax_gallery.set_title(
        "first 16 training images, binarized visible bits v (v=1 ink, v=0 background)",
        fontsize=8,
    )

    ax_spin = fig.add_axes([0.52, 0.56, 0.30, 0.36])
    image = ax_spin.imshow(
        spins[0], cmap="coolwarm", interpolation="none", vmin=-1, vmax=1
    )
    ax_spin.set_xticks([])
    ax_spin.set_yticks([])
    ax_spin.set_title("one enlarged spin configuration s = 2v - 1", fontsize=8)
    bar = fig.colorbar(image, ax=ax_spin, fraction=0.046, pad=0.03)
    bar.set_ticks([-1, 1])
    bar.set_ticklabels(["-1 background", "+1 ink"])
    bar.ax.tick_params(labelsize=7)

    ax_rbm = fig.add_axes([0.06, 0.06, 0.88, 0.34])
    visible = np.linspace(0.04, 0.96, 8)
    hidden = np.linspace(0.1, 0.9, 6)
    rng = np.random.default_rng(cfg["initialization"]["seed"])
    for x in visible:
        for y in hidden:
            ax_rbm.plot(
                [x, y],
                [0.32, 0.68],
                color="tab:blue",
                alpha=0.35 + 0.25 * rng.random(),
                lw=0.6,
                zorder=1,
            )
    ax_rbm.scatter(visible, np.full_like(visible, 0.32), s=170, c="tab:orange", zorder=2)
    ax_rbm.scatter(hidden, np.full_like(hidden, 0.68), s=170, c="tab:green", zorder=2)
    ax_rbm.text(
        0.5, 0.14,
        "visible layer v: 784 bits (28x28 pixels), each with bias a_i",
        ha="center", fontsize=8,
    )
    ax_rbm.text(
        0.5, 0.86,
        "hidden layer h: 128 units, each with bias b_j",
        ha="center", fontsize=8,
    )
    ax_rbm.text(
        0.005, 0.5,
        "couplings W_ij, one per\nvisible-hidden pair\n(no within-layer edges)",
        ha="left", va="center", fontsize=7, color="tab:blue",
    )
    ax_rbm.set_xlim(-0.02, 1.02)
    ax_rbm.set_ylim(0, 1)
    ax_rbm.axis("off")
    ax_rbm.set_title(
        "the restricted Boltzmann machine: 784x128 = 100352 visible-hidden couplings",
        fontsize=8,
    )

    fig.savefig("evidence/rbm-spins.png", dpi=160)
    plt.close(fig)
    print("wrote evidence/rbm-spins.png")


if __name__ == "__main__":
    main()
