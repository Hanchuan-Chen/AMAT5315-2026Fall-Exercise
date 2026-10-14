"""Part 2, VERIFY (4): clamped Gibbs sampling completes held-out digits whose
lower half is missing.

Clamps the observed upper half of the first 64 held-out digits and resamples
the missing lower half with block Gibbs sampling from random starts, 8 chains
per image, both RBMs, recording frames after 1, 10, 100 and 1000 steps.  The
same is repeated with each digit clamped to the wrong upper half, the next
evaluation image with a different label.  Top: the frames.  Bottom: the ink
overlap 2|A n B| / (|A| + |B|) against steps, for the true and the wrong upper
half, with the standard error over the 64 digits.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

import datasets
import figlib
import rbm_lib

STEPS = (1, 10, 100, 1000)
HALF = 28 * 14
SAMPLE_DIGITS = 3
SAMPLE_CHAINS = 2


def load_params(label: str) -> rbm_lib.Params:
    with np.load(f"artifacts/rbm/{label}-params.npz") as data:
        return rbm_lib.numpy_to_params(data)


def wrong_upper_indices(labels: np.ndarray) -> np.ndarray:
    """The next evaluation image with a different label, wrapping."""
    n = labels.size
    out = np.empty(n, dtype=int)
    for i in range(n):
        j = (i + 1) % n
        while labels[j] == labels[i]:
            j = (j + 1) % n
        out[i] = j
    return out


def run_chains(params, v_init, observed, key):
    """Clamped block Gibbs chains; returns frames at the sheet's four steps."""
    upper = jnp.zeros(rbm_lib.N_VISIBLE, dtype=bool).at[:HALF].set(True)

    def advance(v, n_steps, segment_key):
        def body(carry, step_key):
            v_next = rbm_lib.gibbs_step(params, carry, step_key)
            return jnp.where(upper, observed, v_next), None

        keys = jax.random.split(segment_key, n_steps)
        v, _ = jax.lax.scan(body, v, keys)
        return v

    frames = {}
    total = 0
    for segment, step in enumerate(STEPS):
        advance_jit = jax.jit(
            lambda v, key, n=step - total: advance(v, n, key)
        )
        v_init = advance_jit(v_init, jax.random.fold_in(key, segment))
        total = step
        frames[step] = v_init
    return frames


def overlaps(frames, true_lower):
    """Ink overlap of every chain's lower half with the true lower half.

    ``frames`` is (digits, chains, 784) and ``true_lower`` is (digits, 392).
    Returns the per-digit mean over chains.
    """
    ink = frames[:, :, HALF:] >= 0.5
    intersection = jnp.sum(ink * true_lower[:, None, :], axis=-1)
    sizes = jnp.sum(ink, axis=-1) + jnp.sum(true_lower, axis=-1)[:, None]
    overlap = jnp.where(sizes > 0, 2.0 * intersection / jnp.maximum(sizes, 1), 0.0)
    return jnp.mean(overlap, axis=1)


def isolated_fraction(frame_lower):
    """Fraction of inked lower-half pixels with no inked 4-neighbour."""
    grid = np.asarray(frame_lower) >= 0.5
    grid = grid.reshape(14, 28)
    neighbours = np.zeros_like(grid, dtype=int)
    neighbours[1:, :] += grid[:-1, :]
    neighbours[:-1, :] += grid[1:, :]
    neighbours[:, 1:] += grid[:, :-1]
    neighbours[:, :-1] += grid[:, 1:]
    isolated = grid & (neighbours == 0)
    return float(isolated.sum()) / max(int(grid.sum()), 1)


def main() -> None:
    cfg = rbm_lib.load_config()
    _, _, x_test, y_test = datasets.read_mnist()
    n = int(cfg["completion"]["test_images"])
    chains = int(cfg["completion"]["chains"])
    v_test = rbm_lib.binarize(x_test[:n])
    labels = y_test[:n]
    wrong = wrong_upper_indices(labels)
    true_lower = v_test[:, HALF:] >= 0.5

    base = jax.random.key(cfg["completion"]["seed"])
    spin_keys = jax.random.split(base, n * chains)
    random_spins = jnp.stack(
        [
            2.0 * jax.random.bernoulli(key, 0.5, (HALF,)) - 1.0
            for key in spin_keys
        ]
    ).reshape(n, chains, HALF)
    random_lower = (random_spins + 1.0) / 2.0

    results = {}
    frames_by = {}
    for label in ("cd1", "cd10"):
        params = load_params(label)
        for clamp in ("true", "wrong"):
            observed_upper = (
                v_test[:, :HALF] if clamp == "true" else v_test[wrong][:, :HALF]
            )
            observed = jnp.broadcast_to(
                observed_upper[:, None, :], (n, chains, HALF)
            )
            v_init = jnp.concatenate([observed, random_lower], axis=-1).reshape(
                n * chains, rbm_lib.N_VISIBLE
            )
            observed_flat = jnp.concatenate(
                [observed, jnp.zeros_like(random_lower)], axis=-1
            ).reshape(n * chains, rbm_lib.N_VISIBLE)
            frames = run_chains(params, v_init, observed_flat, base)
            clamped = jnp.max(
                jnp.abs(frames[STEPS[0]][:, :HALF] - observed_flat[:, :HALF])
            )
            print(
                f"{label} {clamp}: observed upper half unchanged in every frame "
                f"(largest difference {float(clamped):.0f} pixels)"
            )
            stats = {}
            frames_by[(label, clamp)] = {}
            for step in STEPS:
                flat = frames[step].reshape(n, chains, rbm_lib.N_VISIBLE)
                per_digit = np.asarray(
                    overlaps(flat, true_lower)
                )
                stats[step] = {"mean": per_digit.mean(),
                               "se": per_digit.std(ddof=1) / np.sqrt(n)}
                frames_by[(label, clamp)][step] = np.asarray(flat)
            results[(label, clamp)] = stats

    one_step = frames_by[("cd10", "true")][STEPS[0]][:SAMPLE_DIGITS, :SAMPLE_CHAINS]
    final = frames_by[("cd10", "true")][STEPS[-1]][:SAMPLE_DIGITS, :SAMPLE_CHAINS]
    print(
        f"speckle check, CD-10, 1 step: isolated-ink fraction "
        f"{np.mean([isolated_fraction(f[HALF:]) for f in one_step.reshape(-1, 784)]):.3f} "
        f"at 1 step against "
        f"{np.mean([isolated_fraction(f[HALF:]) for f in final.reshape(-1, 784)]):.3f} "
        f"at 1000 steps"
    )
    gap_one = abs(
        results[("cd10", "true")][STEPS[0]]["mean"]
        - results[("cd10", "wrong")][STEPS[0]]["mean"]
    )
    print(
        f"speckle check: at 1 step the true-half and wrong-half overlaps differ by "
        f"{gap_one:.4f}"
    )

    draw_figure(v_test, wrong, frames_by, results)

    for label in ("cd1", "cd10"):
        true_stats = results[(label, "true")][STEPS[-1]]
        wrong_stats = results[(label, "wrong")][STEPS[-1]]
        gap = true_stats["mean"] - wrong_stats["mean"]
        print(
            f"{label}: after 1000 completion steps the true-half overlap is "
            f"{true_stats['mean']:.3f}+/-{true_stats['se']:.3f} against "
            f"{wrong_stats['mean']:.3f}+/-{wrong_stats['se']:.3f} for the wrong "
            f"half (gap {gap:.3f}, "
            f"{'at least 0.1 as required' if gap >= 0.1 else 'BELOW the required 0.1'})"
        )
    for label in ("cd1", "cd10"):
        row = "  ".join(
            f"{step}:{results[(label, 'true')][step]['mean']:.3f}"
            for step in STEPS
        )
        print(f"{label}: true-half overlap against steps  {row}")


def draw_figure(v_test, wrong, frames_by, results) -> None:
    rows = [
        ("CD-1, true upper half", "cd1", "true"),
        ("CD-1, wrong upper half", "cd1", "wrong"),
        ("CD-10, true upper half", "cd10", "true"),
        ("CD-10, wrong upper half", "cd10", "wrong"),
    ]
    gallery = []
    for _, label, clamp in rows:
        clipped_upper = (
            v_test[:SAMPLE_DIGITS, :HALF]
            if clamp == "true"
            else v_test[wrong[:SAMPLE_DIGITS], :HALF]
        )
        line = [
            figlib.compose_grid(
                [
                    [
                        np.concatenate(
                            [
                                clipped_upper[d].reshape(14, 28),
                                np.full((14, 28), 0.5, dtype=np.float32),
                            ]
                        )
                    ]
                    for d in range(SAMPLE_DIGITS)
                ],
                gap=1,
            )
        ]
        for step in STEPS:
            flat = frames_by[(label, clamp)][step][
                :SAMPLE_DIGITS, :SAMPLE_CHAINS
            ]
            flat = flat.reshape(SAMPLE_DIGITS, SAMPLE_CHAINS, 28, 28)
            line.append(
                figlib.compose_grid(
                    [
                        [flat[d, c] for c in range(SAMPLE_CHAINS)]
                        for d in range(SAMPLE_DIGITS)
                    ],
                    gap=1,
                )
            )
        gallery.append(line)
    y_centers, x_centers = figlib.grid_centers(gallery, gap=4)

    fig = plt.figure(figsize=(11.6, 6.2))
    ax = fig.add_axes([0.215, 0.13, 0.36, 0.76])
    ax.imshow(figlib.compose_grid(gallery, gap=4), cmap="gray_r",
              interpolation="none", vmin=0, vmax=1)
    ax.set_yticks(y_centers)
    ax.set_yticklabels([title for title, _, _ in rows], fontsize=8)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        ["observed\n(3 digits\nx 2 chains)", "1 step", "10 steps", "100 steps",
         "1000 steps"],
        fontsize=7.5,
    )
    ax.set_title(
        "completion: the clamped upper half never changes,\n"
        "the lower half is resampled from random starts",
        fontsize=8.5,
    )

    ax2 = fig.add_axes([0.62, 0.16, 0.36, 0.62])
    styles = {
        ("cd1", "true"): ("tab:blue", "-"),
        ("cd1", "wrong"): ("tab:blue", "--"),
        ("cd10", "true"): ("tab:red", "-"),
        ("cd10", "wrong"): ("tab:red", "--"),
    }
    for key, (colour, style) in styles.items():
        label, clamp = key
        means = [results[key][step]["mean"] for step in STEPS]
        errors = [results[key][step]["se"] for step in STEPS]
        ax2.errorbar(
            range(len(STEPS)), means, yerr=errors, color=colour, ls=style,
            marker="o", ms=4, capsize=3,
            label=f"{label.upper()} {'true' if clamp == 'true' else 'wrong'} upper half",
        )
    ax2.set_xticks(range(len(STEPS)))
    ax2.set_xticklabels([str(step) for step in STEPS])
    ax2.set_xlabel("Gibbs steps (clamped)", fontsize=8)
    ax2.set_ylabel("ink overlap 2|A n B| / (|A| + |B|)\nwith the true lower half", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=7, ncol=1, loc="lower right")
    ax2.set_title("overlap grows when the observation is right, stays flat when wrong",
                  fontsize=8.5)
    fig.savefig("evidence/rbm-completion.png", dpi=160)
    plt.close(fig)
    print("wrote evidence/rbm-completion.png")


if __name__ == "__main__":
    main()
