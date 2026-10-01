#!/usr/bin/env python3
"""Part 1, check (3): time a cluster-energy gradient in both modes.

Writes `artifacts/ad/scaling.png` and prints the gradient/energy time ratios and
each mode's largest relative error against the analytic forces.
"""

import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEEK = Path(__file__).resolve().parents[1]
OUT = WEEK / "artifacts" / "ad"
SIZES = [64, 128, 256, 512, 1024]
SPACING = 2 ** (1 / 6)
NOISE = 0.05


def lattice(n, seed):
    side = int(np.ceil(n ** (1 / 3)))
    sites = np.array(
        [(i, j, k) for i in range(side) for j in range(side) for k in range(side)][:n],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    return (sites * SPACING + rng.normal(0.0, NOISE, sites.shape)).astype(np.float64)


def energy(coordinates):
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    r2 = jnp.sum(difference**2, axis=-1)
    index = jnp.triu_indices(coordinates.shape[0], 1)
    r = jnp.sqrt(r2[index])
    inv6 = r**-6
    return jnp.sum(4 * (inv6**2 - inv6))


def energy_flat(z):
    return energy(z.reshape(-1, 3))


def analytic_gradient(x):
    difference = x[:, None, :] - x[None, :, :]
    r2 = np.sum(difference**2, axis=-1)
    r = np.sqrt(r2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dudr = 24 * (r**-7 - 2 * r**-13)
        scale = np.where(r2 > 0, dudr / np.where(r > 0, r, 1.0), 0.0)
    return np.sum(scale[:, :, None] * difference, axis=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in SIZES:
        coordinates = lattice(n, 2026)
        p = 3 * n
        x = jnp.asarray(coordinates.reshape(-1))
        exact = analytic_gradient(coordinates).reshape(-1)
        energy_jit = jax.jit(energy_flat)
        grad_jit = jax.jit(jax.grad(energy_flat))
        tangent = jax.jit(lambda z, v: jax.jvp(energy_flat, (z,), (v,))[1])
        basis = jnp.zeros(p, dtype=jnp.float64)
        # Warm the caches; the first call of each shape pays compilation.
        energy_jit(x).block_until_ready()
        grad_jit(x).block_until_ready()
        tangent(x, basis.at[0].set(1.0)).block_until_ready()

        start = time.perf_counter()
        energy_jit(x).block_until_ready()
        energy_time = time.perf_counter() - start
        start = time.perf_counter()
        reference = grad_jit(x)
        jax.block_until_ready(reference)
        reverse_time = time.perf_counter() - start
        forward = np.empty(p)
        start = time.perf_counter()
        for j in range(p):
            direction = basis.at[j].set(1.0)
            forward[j] = float(tangent(x, direction))
        forward_time = time.perf_counter() - start

        reverse = np.asarray(reference)
        forward_error = float(
            np.max(np.abs(forward - exact)) / np.max(np.abs(exact))
        )
        reverse_error = float(
            np.max(np.abs(reverse - exact)) / np.max(np.abs(exact))
        )
        rows.append(
            {
                "N": n,
                "P": p,
                "energy_time": energy_time,
                "forward_time": forward_time,
                "reverse_time": reverse_time,
                "forward_ratio": forward_time / energy_time,
                "reverse_ratio": reverse_time / energy_time,
                "forward_error": forward_error,
                "reverse_error": reverse_error,
            }
        )
        print(
            f"N={n:5d} P={p:5d} energy={energy_time * 1e3:8.3f} ms "
            f"forward={forward_time * 1e3:9.2f} ms reverse={reverse_time * 1e3:8.3f} ms "
            f"ratios forward={rows[-1]['forward_ratio']:8.1f} reverse={rows[-1]['reverse_ratio']:5.2f} "
            f"errors forward={forward_error:.2e} reverse={reverse_error:.2e}"
        )
        assert max(forward_error, reverse_error) < 1e-12

    last = rows[-1]
    assert last["forward_ratio"] > 100 * last["reverse_ratio"]

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"}
    )
    p = np.array([row["P"] for row in rows])
    fig, ax = plt.subplots(figsize=(5.6, 3.6), layout="constrained")
    ax.loglog(p, [row["forward_ratio"] for row in rows], "o-", color="#ba4b37", label="forward mode")
    ax.loglog(p, [row["reverse_ratio"] for row in rows], "s-", color="#24557a", label="reverse mode")
    ax.loglog(p, p / rows[0]["P"] * rows[0]["forward_ratio"], ":", color="#8b55a1", lw=1, label=r"$\propto P$")
    ax.set(
        xlabel="Number of inputs $P = 3N$",
        ylabel="Gradient time / energy time",
        title="Cluster-energy gradient cost",
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.savefig(OUT / "scaling.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(
        f"at P={last['P']}: forward ratio {last['forward_ratio']:.1f}, "
        f"reverse ratio {last['reverse_ratio']:.2f}, "
        f"forward/reverse = {last['forward_ratio'] / last['reverse_ratio']:.1f}x"
    )


if __name__ == "__main__":
    main()
