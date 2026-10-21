"""Small JAX implementation shared by the Week 7 diffusion experiments."""

from __future__ import annotations

import json
import time
import tomllib
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

ROOT = Path(__file__).resolve().parents[1]
CENTERS = jnp.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]])
T = 1000
BETAS = jnp.linspace(1e-4, 0.02, T, dtype=jnp.float32)
ALPHAS = 1.0 - BETAS
ALPHA_BARS = jnp.concatenate([jnp.ones((1,), jnp.float32), jnp.cumprod(ALPHAS)])


def load_toml(name: str) -> dict:
    with (ROOT / name).open("rb") as handle:
        return tomllib.load(handle)


def _sigmoid_free_silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def time_features(
    time: jax.Array, *, time_scale: float = 1.0, offset: float = 0.0
) -> jax.Array:
    """Sine/cosine features ordered sin(freq 0..15), cos(freq 0..15)."""
    frequencies = jnp.power(10_000.0, -jnp.arange(16, dtype=jnp.float32) / 16.0)
    phase = jnp.atleast_1d((time + offset) * time_scale)
    angles = phase[:, None] * frequencies[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def init_mlp(cfg: dict, key: jax.Array, *, flow: bool = False) -> dict[str, jax.Array]:
    widths = [2 + int(cfg["model"]["time_features"])] + list(
        map(int, cfg["model"]["hidden_widths"])
    ) + [2]
    params: dict[str, jax.Array] = {}
    for layer, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:])):
        key, subkey = jax.random.split(key)
        params[f"w{layer}"] = jax.random.normal(subkey, (n_in, n_out), jnp.float32) / jnp.sqrt(
            float(n_in)
        )
        params[f"b{layer}"] = jnp.zeros((n_out,), jnp.float32)
    return params


def mlp_forward(
    params: dict[str, jax.Array],
    x: jax.Array,
    time: jax.Array,
) -> jax.Array:
    features = time_features(time, offset=-1.0)
    if features.shape[0] == 1 and x.shape[0] != 1:
        features = jnp.broadcast_to(features, (x.shape[0], features.shape[-1]))
    h = jnp.concatenate([x, features], axis=-1)
    n_layers = sum(name.startswith("w") for name in params)
    for layer in range(n_layers):
        h = h @ params[f"w{layer}"] + params[f"b{layer}"]
        if layer + 1 < n_layers:
            h = _sigmoid_free_silu(h)
    return h


@partial(jax.jit, static_argnums=1)
def sample_target(key: jax.Array, n: int) -> jax.Array:
    key_cluster, key_noise = jax.random.split(key)
    labels = jax.random.randint(key_cluster, (n,), 0, 4)
    return CENTERS[labels] + 0.3 * jax.random.normal(key_noise, (n, 2), jnp.float32)


@partial(jax.jit, static_argnums=3)
def diffusion_batch(
    key_x: jax.Array, key_t: jax.Array, key_eps: jax.Array, batch_size: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    x0 = sample_target(key_x, batch_size)
    levels = jax.random.randint(key_t, (batch_size,), 1, T + 1)
    eps = jax.random.normal(key_eps, (batch_size, 2), jnp.float32)
    ab = ALPHA_BARS[levels][:, None]
    xt = jnp.sqrt(ab) * x0 + jnp.sqrt(1.0 - ab) * eps
    return x0, levels.astype(jnp.float32), xt, eps


def init_adam(params: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        "m": jax.tree.map(jnp.zeros_like, params),
        "v": jax.tree.map(jnp.zeros_like, params),
        "count": jnp.array(0, jnp.int32),
    }


def cosine_lr(cfg: dict, step: jax.Array) -> jax.Array:
    training = cfg["training"]
    updates = float(training["updates"])
    start = float(training["learning_rate"])
    final = float(training["final_learning_rate"])
    fraction = jnp.minimum(step / updates, 1.0)
    return final + (start - final) * 0.5 * (1.0 + jnp.cos(jnp.pi * fraction))


def adam_update(
    params: dict[str, jax.Array],
    state: dict[str, jax.Array],
    grads: dict[str, jax.Array],
    lr: jax.Array,
    cfg: dict,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    training = cfg["training"]
    b1 = float(training["beta1"])
    b2 = float(training["beta2"])
    eps = float(training["epsilon"])
    count = state["count"] + 1
    m = jax.tree.map(lambda old, g: b1 * old + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda old, g: b2 * old + (1.0 - b2) * g * g, state["v"], grads)
    mh = jax.tree.map(lambda value: value / (1.0 - b1**count), m)
    vh = jax.tree.map(lambda value: value / (1.0 - b2**count), v)
    updated = jax.tree.map(
        lambda p, mm, vv: p - lr * mm / (jnp.sqrt(vv) + eps), params, mh, vh
    )
    return updated, {"m": m, "v": v, "count": count}


def train_noise(cfg: dict) -> tuple[dict, np.ndarray, dict[str, dict], dict]:
    training = cfg["training"]
    seed = int(training["seed"])
    root = jax.random.key(seed)
    key_init, key_train = jax.random.split(root)
    params = init_mlp(cfg, key_init)
    state = init_adam(params)
    checkpoints = set(map(int, training["checkpoint_updates"]))
    saved: dict[str, dict] = {"0": jax.tree.map(np.asarray, params)}
    loss_history = np.empty((int(training["updates"]),), np.float32)
    started = time.perf_counter()

    def loss_fn(current, batch_x, levels, eps):
        prediction = mlp_forward(current, batch_x, levels)
        return jnp.mean(jnp.sum((prediction - eps) ** 2, axis=-1)) / 2.0

    @jax.jit
    def step(current, adam_state, key, update_index):
        key_x, key_t, key_eps = jax.random.split(key, 3)
        _, levels, xt, eps = diffusion_batch(
            key_x, key_t, key_eps, int(training["batch_size"])
        )
        loss, grads = jax.value_and_grad(loss_fn)(current, xt, levels, eps)
        current, adam_state = adam_update(
            current, adam_state, grads, cosine_lr(cfg, update_index), cfg
        )
        return current, adam_state, loss

    losses = []
    for update in range(1, int(training["updates"]) + 1):
        key_train, subkey = jax.random.split(key_train)
        params, state, loss = step(params, state, subkey, jnp.array(update - 1))
        losses.append(float(loss))
        if update in checkpoints:
            saved[str(update)] = jax.tree.map(np.asarray, params)
    loss_history[:] = losses
    elapsed = time.perf_counter() - started
    run = {
        "task": "week7 diffusion",
        "seed": seed,
        "updates": int(training["updates"]),
        "batch_size": int(training["batch_size"]),
        "device": str(jax.devices()[0]),
        "precision": "float32",
        "elapsed_seconds": elapsed,
        "final_loss": losses[-1],
        "checkpoint_updates": sorted(checkpoints),
    }
    return params, loss_history, saved, run


def _decode_diffusion_params(flat: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {name: jnp.asarray(value) for name, value in flat.items()}


def ddpm_sample(
    params: dict[str, jax.Array],
    x_start: jax.Array,
    levels: np.ndarray | jax.Array,
    key: jax.Array,
    *,
    return_path: bool = False,
) -> jax.Array:
    levels = jnp.asarray(levels, jnp.int32)
    n_steps = levels.shape[0] - 1
    noise = jax.random.normal(key, (n_steps,) + x_start.shape, jnp.float32)

    @jax.jit
    def run(x0, eps_seq):
        def body(x, step_data):
            t, tp, z = step_data
            eps = mlp_forward(params, x, t.astype(jnp.float32))
            ab_t, ab_tp = ALPHA_BARS[t], ALPHA_BARS[tp]
            a = ab_t / ab_tp
            mean = (x - (1.0 - a) / jnp.sqrt(1.0 - ab_t) * eps) / jnp.sqrt(a)
            variance = jnp.where(tp > 0, 1.0 - a, 0.0)
            next_x = mean + jnp.sqrt(variance) * z
            return next_x, next_x

        x, path = jax.lax.scan(body, x0, (levels[:-1], levels[1:], eps_seq))
        return (x, path) if return_path else x

    return run(x_start, noise)


def summarize(samples: np.ndarray) -> dict:
    samples = np.asarray(samples)
    centers = np.asarray(CENTERS)
    distances = np.sum((samples[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    labels = np.argmin(distances, axis=-1)
    populations, means, widths = [], [], []
    for cluster in range(4):
        group = samples[labels == cluster]
        populations.append(float(len(group) / len(samples)))
        means.append(np.mean(group, axis=0).tolist())
        widths.append(np.std(group, axis=0).tolist())
    return {
        "population": populations,
        "centre": means,
        "width": widths,
        "max_population_error": float(np.max(np.abs(np.asarray(populations) - 0.25))),
        "max_centre_error": float(
            np.max(np.linalg.norm(np.asarray(means) - centers, axis=1))
        ),
        "max_width_error": float(np.max(np.abs(np.asarray(widths) - 0.3))),
    }


def save_params(path: Path, params: dict[str, np.ndarray]) -> None:
    np.savez(path, **params)


def load_params(path: Path) -> dict[str, jax.Array]:
    with np.load(path) as data:
        return _decode_diffusion_params({name: data[name] for name in data.files})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
