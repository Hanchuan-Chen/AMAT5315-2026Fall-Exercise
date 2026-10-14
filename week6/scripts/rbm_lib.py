"""Part 2's restricted Boltzmann machine: energy, free energy, block Gibbs
sampling and contrastive-divergence training, all from ``rbm.toml``.

The visible layer holds 784 binary pixel bits (``v = 1`` at pixel value >= 128)
and the hidden layer 128 binary units.  The energy is Eq. 3 of the sheet,

    E(v, h) = -a^T v - b^T h - v^T W h,

the visible free energy is Eq. 4,

    F(v) = -a^T v - sum_j softplus(b_j + (v^T W)_j),

and one Gibbs step draws the whole hidden layer then the whole visible layer
from their conditional Bernoulli laws, Eqs. 6-7.  Training minimizes CD-1's or
CD-10's surrogate, Eq. 10, ``mean(F(v_0) - F(stop_gradient(v_k)))``, whose
gradient is the CD update of Eqs. 8-9.
"""

from __future__ import annotations

import json
import time
import tomllib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

N_VISIBLE = 28 * 28
N_HIDDEN = 128
TOML_PATH = "rbm.toml"


class Params(NamedTuple):
    W: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray


def load_config(path: str = TOML_PATH) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def binarize(images: np.ndarray) -> jnp.ndarray:
    """Half-intensity binarization: v = 1 for a pixel value at least 128."""
    flat = images.reshape(images.shape[0], -1)
    return jnp.asarray(flat >= 128, dtype=jnp.float32)


def spins_of(visible: np.ndarray) -> np.ndarray:
    """The equivalent spin configuration s = 2v - 1, ink +1 and background -1."""
    return 2.0 * visible - 1.0


def init_params(cfg: dict, pixel_frequency: jnp.ndarray) -> Params:
    """rbm.toml's initialization: normal weights, zero hidden biases, and the
    clipped training pixel frequency's logit as each visible bias."""
    init = cfg["initialization"]
    key = jax.random.key(init["seed"])
    W = (init["weight_std"] * jax.random.normal(key, (N_VISIBLE, N_HIDDEN))).astype(
        jnp.float32
    )
    b = jnp.full((N_HIDDEN,), float(init["hidden_bias"]), dtype=jnp.float32)
    low, high = init["pixel_frequency_clip"]
    p = jnp.clip(pixel_frequency, low, high)
    a = jnp.log(p / (1.0 - p)).astype(jnp.float32)
    return Params(W=W, a=a, b=b)


def energy(params: Params, v: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """Eq. 3, one scalar per (v, h) pair."""
    return -(v @ params.a) - (h @ params.b) - jnp.sum((v @ params.W) * h, axis=-1)


def free_energy(params: Params, v: jnp.ndarray) -> jnp.ndarray:
    """Eq. 4, one scalar per image."""
    return -(v @ params.a) - jnp.sum(
        jax.nn.softplus(params.b + v @ params.W), axis=-1
    )


def hidden_probability(params: Params, v: jnp.ndarray) -> jnp.ndarray:
    """Eq. 6: p(h_j = 1 | v), one row per image."""
    return jax.nn.sigmoid(params.b + v @ params.W)


def visible_probability(params: Params, h: jnp.ndarray) -> jnp.ndarray:
    """Eq. 7: p(v_i = 1 | h), one row per configuration."""
    return jax.nn.sigmoid(params.a + h @ params.W.T)


def sample_hidden(params: Params, v: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    """Draw the whole hidden layer given v.  The bits are conditionally
    independent, so one key draws every entry in parallel."""
    return jax.random.bernoulli(key, hidden_probability(params, v)).astype(jnp.float32)


def sample_visible(params: Params, h: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    """Draw the whole visible layer given h."""
    return jax.random.bernoulli(key, visible_probability(params, h)).astype(jnp.float32)


def gibbs_step(params: Params, v: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    """One Gibbs step, also called a sweep: h | v then v | h."""
    hidden_key, visible_key = jax.random.split(key)
    h = sample_hidden(params, v, hidden_key)
    return sample_visible(params, h, visible_key)


def gibbs(params: Params, v: jnp.ndarray, steps: int, key: jnp.ndarray) -> jnp.ndarray:
    """``steps`` Gibbs steps from v, each with its own split of ``key``."""
    keys = jax.random.split(key, steps)

    def body(carry, step_key):
        return gibbs_step(params, carry, step_key), None

    v, _ = jax.lax.scan(body, v, keys)
    return v


def surrogate(params: Params, v0: jnp.ndarray, vk: jnp.ndarray) -> jnp.ndarray:
    """Eq. 10: mean over the batch of F(v0) - F(stop_gradient(vk))."""
    return jnp.mean(
        free_energy(params, v0) - free_energy(params, jax.lax.stop_gradient(vk))
    )


def cd_gradient(
    params: Params, v0: jnp.ndarray, vk: jnp.ndarray
) -> tuple[jnp.ndarray, Params]:
    """The surrogate's value and its gradient, the CD update direction."""
    value, grads = jax.value_and_grad(surrogate)(params, v0, vk)
    return value, grads


def reconstruction_error(
    params: Params, v0: jnp.ndarray, key: jnp.ndarray
) -> jnp.ndarray:
    """Held-out one-step reconstruction error: fraction of changed pixels."""
    v1 = gibbs_step(params, v0, key)
    return jnp.mean(v1 != v0)


def training_keys(cfg: dict, epoch: int, n_batches: int) -> jnp.ndarray:
    """The shared Gibbs keys of one epoch, one per batch.

    Both models draw these from the same ``[training] seed``, so CD-1 and CD-10
    see identical data order and identical Gibbs randomness.
    """
    base = jax.random.key(cfg["training"]["seed"])
    return jax.random.split(jax.random.fold_in(base, epoch), n_batches)


def evaluation_key(cfg: dict) -> jnp.ndarray:
    """rbm.toml's fixed evaluation randomness: same keys at every epoch and for
    both models."""
    return jax.random.key(cfg["evaluation"]["seed"])


def make_epoch(cfg: dict, chain_length: int):
    """A jitted training epoch: 600 batch updates of plain SGD on the surrogate."""
    learning_rate = cfg["training"]["learning_rate"]
    batch_size = cfg["training"]["batch_size"]

    def epoch(params: Params, v_data: jnp.ndarray, keys: jnp.ndarray):
        n_batches = v_data.shape[0] // batch_size
        batches = v_data[: n_batches * batch_size].reshape(
            n_batches, batch_size, N_VISIBLE
        )

        def batch_step(params, carry):
            v0, key = carry
            vk = gibbs(params, v0, chain_length, key)
            value, grads = cd_gradient(params, v0, vk)
            params = jax.tree.map(lambda w, g: w - learning_rate * g, params, grads)
            return params, value

        params, values = jax.lax.scan(batch_step, params, (batches, keys))
        return params, values

    return jax.jit(epoch)


def train(
    cfg: dict,
    v_train: jnp.ndarray,
    v_test: jnp.ndarray,
    chain_length: int,
    label: str,
) -> dict:
    """Train one model and return its history, parameters and wall-clock time."""
    images = int(cfg["training"]["images"])
    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    v_data = v_train[:images]
    n_batches = images // batch_size
    pixel_frequency = jnp.mean(v_data, axis=0)
    params = init_params(cfg, pixel_frequency)

    eval_key = evaluation_key(cfg)
    recon = jax.jit(reconstruction_error)

    history = {
        "epoch": [0],
        "surrogate": [float("nan")],
        "reconstruction_error": [float(recon(params, v_test, eval_key))],
    }
    started = time.time()
    epoch_fn = make_epoch(cfg, chain_length)
    for epoch in range(epochs):
        base = jax.random.key(cfg["training"]["seed"])
        permutation = jax.random.permutation(jax.random.fold_in(base, epoch), images)
        params, values = epoch_fn(params, v_data[permutation], training_keys(
            cfg, epoch, n_batches
        ))
        history["epoch"].append(epoch + 1)
        history["surrogate"].append(float(jnp.mean(values)))
        history["reconstruction_error"].append(float(recon(params, v_test, eval_key)))
    elapsed = time.time() - started
    print(
        f"{label}: {chain_length} Gibbs step(s), {epochs} epochs, "
        f"reconstruction error {history['reconstruction_error'][0]:.4f} at "
        f"initialization -> {history['reconstruction_error'][-1]:.4f} at the end, "
        f"{elapsed:.1f} s"
    )
    return {"history": history, "params": params, "seconds": elapsed}


def params_to_numpy(params: Params) -> dict:
    return {"W": np.asarray(params.W), "a": np.asarray(params.a), "b": np.asarray(params.b)}


def numpy_to_params(data) -> Params:
    return Params(
        W=jnp.asarray(data["W"]), a=jnp.asarray(data["a"]), b=jnp.asarray(data["b"])
    )


def write_settings(path: str, cfg: dict, runs: dict) -> None:
    """Record the TOML settings, the resolved values and the wall-clock times."""
    import jax as _jax

    record = {
        "rbm_toml": cfg,
        "environment": {
            "jax": _jax.__version__,
            "numpy": np.__version__,
            "platform": _jax.devices()[0].platform,
        },
        "model": {"visible": N_VISIBLE, "hidden": N_HIDDEN},
        "runs": {
            label: {"chain_length": run["chain_length"], "seconds": run["seconds"],
                    "history": run["history"]}
            for label, run in runs.items()
        },
    }
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)
