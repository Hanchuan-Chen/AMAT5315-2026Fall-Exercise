#!/usr/bin/env python3
"""Part 1, check (1): hand-written JAX forward and reverse modes.

Writes `artifacts/ad/derivatives.json` and `artifacts/ad/modes.png`, and prints
the maximum absolute error of each method against `dU/dr = 24(r^-7 - 2 r^-13)`.
"""

import json
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


def energy(r):
    a = r**-6
    b = a**2
    c = b - a
    return 4 * c


def analytic(r):
    return 24 * (r**-7 - 2 * r**-13)


def hand_forward(r_value):
    """Forward mode, one node at a time: seed dr = 1 and carry tangents."""
    r = jnp.float64(r_value)
    one = jnp.float64(1.0)
    a, da = jax.jvp(lambda x: x**-6, (r,), (one,))
    b, db = jax.jvp(lambda x: x**2, (a,), (da,))
    c, dc = jax.jvp(lambda x, y: x - y, (b, a), (db, da))
    u, du = jax.jvp(lambda x: 4.0 * x, (c,), (dc,))
    return {
        "r": float(r),
        "energy": float(u),
        "tangents": {
            "r": float(one),
            "a": float(da),
            "b": float(db),
            "c": float(dc),
            "U": float(du),
        },
    }


def hand_reverse(r_value):
    """Reverse mode, one node at a time: seed Ubar = 1 and sweep back."""
    r = jnp.float64(r_value)
    a = r**-6
    b = a**2
    c = b - a
    u = 4.0 * c
    ubar = jnp.float64(1.0)
    _, pullback = jax.vjp(lambda x: 4.0 * x, c)
    (cbar,) = pullback(ubar)
    _, pullback = jax.vjp(lambda x, y: x - y, b, a)
    bbar, abar_from_c = pullback(cbar)
    _, pullback = jax.vjp(lambda x: x**2, a)
    (abar_from_b,) = pullback(bbar)
    abar = abar_from_c + abar_from_b
    _, pullback = jax.vjp(lambda x: x**-6, r)
    (rbar,) = pullback(abar)
    return {
        "energy": float(u),
        "adjoints": {
            "U": float(ubar),
            "c": float(cbar),
            "b": float(bbar),
            "a": float(abar),
            "r": float(rbar),
        },
    }


def sweep(count, low, high, h):
    """Vectorised form of the same hand-written chain rule over a sample."""
    r = jnp.linspace(low, high, count)
    a = r**-6
    b = a**2
    c = b - a
    u = 4 * c
    da = -6 * r**-7
    db = 2 * a * da
    dc = db - da
    forward = 4 * dc
    cbar = 4.0 * jnp.ones_like(r)
    bbar = cbar
    abar = -cbar + bbar * 2 * a
    reverse = abar * (-6 * r**-7)
    exact = analytic(r)
    plus = 4 * ((r + h) ** -12 - (r + h) ** -6)
    minus = 4 * ((r - h) ** -12 - (r - h) ** -6)
    finite_difference = (plus - minus) / (2 * h)
    return (
        np.asarray(r),
        np.asarray(u),
        np.asarray(forward),
        np.asarray(reverse),
        np.asarray(exact),
        np.asarray(finite_difference),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    forward_mode = hand_forward(1.3)
    reverse_mode = hand_reverse(1.3)
    jax_grad = float(jax.grad(energy)(jnp.float64(1.3)))
    evidence = {
        "r": forward_mode["r"],
        "energy": forward_mode["energy"],
        "tangents": forward_mode["tangents"],
        "adjoints": reverse_mode["adjoints"],
        "jax_grad": jax_grad,
    }
    (OUT / "derivatives.json").write_text(json.dumps(evidence, indent=2) + "\n")

    r, u, forward, reverse, exact, finite_difference = sweep(601, 0.95, 2.5, 1e-6)
    errors = {
        "forward": float(np.max(np.abs(forward - exact))),
        "reverse": float(np.max(np.abs(reverse - exact))),
        "finite difference": float(np.max(np.abs(finite_difference - exact))),
    }
    for name, value in errors.items():
        print(f"{name} max abs error: {value:.6e}")
    assert errors["forward"] < 1e-12 and errors["reverse"] < 1e-12
    assert errors["finite difference"] > max(errors["forward"], errors["reverse"])

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), layout="constrained")
    axes[0].plot(r, exact, "-", color="#24557a", lw=1.6, label="analytic")
    axes[0].plot(r, forward, "--", color="#ba4b37", lw=1.0, label="forward AD")
    axes[0].plot(r, reverse, ":", color="#34845a", lw=1.2, label="reverse AD")
    axes[0].axhline(0.0, color="#bbbbbb", lw=0.6)
    axes[0].axvline(2 ** (1 / 6), color="#999999", lw=0.7, ls="--")
    axes[0].set(
        xlabel="Separation $r$",
        ylabel="$\\mathrm{d}U/\\mathrm{d}r$",
        title="Lennard-Jones force",
        ylim=(-8, 8),
    )
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].semilogy(r, np.abs(forward - exact), color="#ba4b37", lw=1.0, label="forward AD")
    axes[1].semilogy(r, np.abs(reverse - exact), color="#34845a", lw=1.2, ls=":", label="reverse AD")
    axes[1].semilogy(
        r,
        np.abs(finite_difference - exact),
        color="#8b55a1",
        lw=0.9,
        ls="--",
        label="finite difference",
    )
    axes[1].set(xlabel="Separation $r$", ylabel="Absolute error", title="Error against the analytic derivative")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Pair energy: both AD modes beat the finite difference", fontsize=10)
    fig.savefig(OUT / "modes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(
        "derivative check: tangents.U={:.15f} adjoints.r={:.15f} jax_grad={:.15f} "
        "adjoints.a={:.15f} energy={:.16f}".format(
            forward_mode["tangents"]["U"],
            reverse_mode["adjoints"]["r"],
            jax_grad,
            reverse_mode["adjoints"]["a"],
            forward_mode["energy"],
        )
    )


if __name__ == "__main__":
    main()
