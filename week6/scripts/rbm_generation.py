"""Part 2, VERIFY (3): generation develops structure from random spins.

Top: reconstruction of held-out digits and fresh generation from random spin
configurations, both models, at 1, 10, 100 and 1000 Gibbs steps from matched
starts (the same held-out images and the same random spins for CD-1 and CD-10).
Bottom: the two models' learning curves, the held-out one-step reconstruction
error against epoch.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

import datasets
import figlib
import rbm_lib

RUN = "artifacts/rbm/run.json"
STEPS = (1, 10, 100, 1000)
CHAINS = 2


def advance(params, v, n_steps, key):
    """``n_steps`` Gibbs steps under fixed parameters, jitted per step count."""
    fn = jax.jit(lambda p, x, k: rbm_lib.gibbs(p, x, n_steps, k))
    return fn(params, v, key)


def chain_frames(params, v0, key):
    """Frames of one Gibbs chain at the sheet's four steps, from any start."""
    frames = {0: np.asarray(v0)}
    total = 0
    for segment, step in enumerate(STEPS):
        v0 = advance(params, v0, step - total, jax.random.fold_in(key, segment))
        total = step
        frames[step] = np.asarray(v0)
    return frames


def load_params(label: str) -> rbm_lib.Params:
    with np.load(f"artifacts/rbm/{label}-params.npz") as data:
        return rbm_lib.numpy_to_params(data)


def main() -> None:
    cfg = rbm_lib.load_config()
    _, _, x_test, _ = datasets.read_mnist()
    v_test = rbm_lib.binarize(x_test[: int(cfg["evaluation"]["test_images"])])
    recon_start = v_test[:CHAINS]

    eval_key = jax.random.key(cfg["evaluation"]["seed"])
    generation_key = jax.random.fold_in(eval_key, 1)
    spin_keys = jax.random.split(generation_key, CHAINS)
    random_spins = jnp.stack(
        [2.0 * jax.random.bernoulli(key, 0.5, (rbm_lib.N_VISIBLE,)) - 1.0
         for key in spin_keys]
    )
    gen_start = (random_spins + 1.0) / 2.0

    frames = {}
    for label in ("cd1", "cd10"):
        params = load_params(label)
        frames[(label, "recon")] = chain_frames(params, recon_start, eval_key)
        frames[(label, "gen")] = chain_frames(params, gen_start, generation_key)

    same = max(
        float(jnp.max(jnp.abs(frames[(label, "gen")][0] - frames[("cd1", "gen")][0])))
        for label in ("cd1", "cd10")
    )
    print(
        f"both models start from the same {CHAINS} random spin configurations "
        f"(largest entry difference {same:.1e}) and the same held-out images"
    )
    for step in STEPS:
        print(
            f"fresh generation, CD-10, {step:4d} steps: ink fraction "
            f"{float(jnp.mean(frames[('cd10', 'gen')][step])):.4f}"
        )

    rows = [
        ("CD-1 reconstruction\nfrom held-out digits", "cd1", "recon"),
        ("CD-1 fresh generation\nfrom random spins", "cd1", "gen"),
        ("CD-10 reconstruction\nfrom held-out digits", "cd10", "recon"),
        ("CD-10 fresh generation\nfrom random spins", "cd10", "gen"),
    ]
    gallery = [
        [frames[(label, kind)][step][chain].reshape(28, 28)
         for step in [0, *STEPS] for chain in range(CHAINS)]
        for _, label, kind in rows
    ]

    fig = plt.figure(figsize=(9.6, 6.6))
    ax = fig.add_axes([0.235, 0.30, 0.75, 0.64])
    ax.imshow(figlib.compose_grid(gallery, gap=2), cmap="gray_r",
              interpolation="none", vmin=0, vmax=1)
    ax.set_xticks([CHAINS * 30 * i + 29 for i in range(len([0, *STEPS]))])
    ax.set_xticklabels(["0 (start)", "1", "10", "100", "1000"], fontsize=8)
    ax.set_xlabel("Gibbs steps", fontsize=9)
    ax.set_yticks([i * 30 + 14 for i in range(len(rows))])
    ax.set_yticklabels([title for title, _, _ in rows], fontsize=8)
    ax.set_title(
        "reconstruction keeps the starting digit; generation must develop "
        "structure without one (two chains per cell)",
        fontsize=9,
    )

    with open(RUN) as handle:
        record = json.load(handle)
    ax_curves = fig.add_axes([0.235, 0.07, 0.75, 0.15])
    for label, colour in (("cd1", "tab:blue"), ("cd10", "tab:red")):
        history = record["runs"][label]["history"]
        ax_curves.plot(history["epoch"], history["reconstruction_error"], "-o",
                       ms=3, color=colour, label=label.upper())
    ax_curves.set_yscale("log")
    ax_curves.set_xlabel("epoch", fontsize=8)
    ax_curves.set_ylabel("held-out one-step\nreconstruction error", fontsize=7)
    ax_curves.tick_params(labelsize=7)
    ax_curves.legend(fontsize=7, title="learning curves", title_fontsize=7)
    ax_curves.set_title(
        "CD-1 and CD-10 agree on held-out one-step reconstruction", fontsize=8
    )
    fig.savefig("evidence/rbm-generation.png", dpi=160)
    plt.close(fig)
    print("wrote evidence/rbm-generation.png")


if __name__ == "__main__":
    main()
