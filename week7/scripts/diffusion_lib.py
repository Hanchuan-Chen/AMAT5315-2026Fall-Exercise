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
    *,
    flow: bool,
) -> jax.Array:
    scale = float(load_toml("flow.toml")["model"]["time_scale"]) if flow else 1.0
    offset = 0.0 if flow else -1.0
    features = time_features(time, time_scale=scale, offset=offset)
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
        prediction = mlp_forward(current, batch_x, levels, flow=False)
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


def train_flow(cfg: dict) -> tuple[dict, np.ndarray, dict]:
    training = cfg["training"]
    seed = int(training["seed"])
    key_init, key_train = jax.random.split(jax.random.key(seed))
    params = init_mlp(cfg, key_init, flow=True)
    state = init_adam(params)
    losses = np.empty((int(training["updates"]),), np.float32)
    started = time.perf_counter()

    def loss_fn(current, xs, s, target_velocity):
        prediction = mlp_forward(current, xs, s, flow=True)
        return jnp.mean(jnp.sum((prediction - target_velocity) ** 2, axis=-1)) / 2.0

    @jax.jit
    def step(current, adam_state, key, update_index):
        key_x, key_z, key_s = jax.random.split(key, 3)
        x0 = sample_target(key_x, int(training["batch_size"]))
        z = jax.random.normal(
            key_z, (int(training["batch_size"]), 2), jnp.float32
        )
        s = jax.random.uniform(
            key_s, (int(training["batch_size"]),), minval=0.0, maxval=1.0
        )
        xs = (1.0 - s[:, None]) * x0 + s[:, None] * z
        target = z - x0
        loss, grads = jax.value_and_grad(loss_fn)(current, xs, s, target)
        current, adam_state = adam_update(
            current, adam_state, grads, cosine_lr(cfg, update_index), cfg
        )
        return current, adam_state, loss

    for update in range(int(training["updates"])):
        key_train, subkey = jax.random.split(key_train)
        params, state, loss = step(params, state, subkey, jnp.array(update))
        losses[update] = float(loss)
    run = {
        "task": "week7 flow matching",
        "seed": seed,
        "updates": int(training["updates"]),
        "batch_size": int(training["batch_size"]),
        "device": str(jax.devices()[0]),
        "precision": "float32",
        "elapsed_seconds": time.perf_counter() - started,
        "final_loss": float(losses[-1]),
    }
    return params, losses, run


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
            eps = mlp_forward(params, x, t.astype(jnp.float32), flow=False)
            ab_t, ab_tp = ALPHA_BARS[t], ALPHA_BARS[tp]
            a = ab_t / ab_tp
            mean = (x - (1.0 - a) / jnp.sqrt(1.0 - ab_t) * eps) / jnp.sqrt(a)
            variance = jnp.where(tp > 0, 1.0 - a, 0.0)
            next_x = mean + jnp.sqrt(variance) * z
            return next_x, next_x

        x, path = jax.lax.scan(body, x0, (levels[:-1], levels[1:], eps_seq))
        return (x, path) if return_path else x

    return run(x_start, noise)


def ddim_sample(
    params: dict[str, jax.Array],
    x_start: jax.Array,
    levels: np.ndarray | jax.Array,
    *,
    exact: bool = False,
    return_path: bool = False,
) -> jax.Array:
    levels = jnp.asarray(levels, jnp.int32)

    @jax.jit
    def run(x0):
        def body(x, pair):
            t, tp = pair
            eps = (
                exact_noise(x, t.astype(jnp.float32))
                if exact
                else mlp_forward(params, x, t.astype(jnp.float32), flow=False)
            )
            ab_t = ALPHA_BARS[t]
            x0_hat = (x - jnp.sqrt(1.0 - ab_t) * eps) / jnp.sqrt(ab_t)
            ab_tp = ALPHA_BARS[tp]
            next_x = jnp.sqrt(ab_tp) * x0_hat + jnp.sqrt(1.0 - ab_tp) * eps
            return next_x, next_x

        x, path = jax.lax.scan(body, x0, (levels[:-1], levels[1:]))
        return (x, path) if return_path else x

    return run(x_start)


def exact_noise(x: jax.Array, level: jax.Array) -> jax.Array:
    ab = jnp.atleast_1d(ALPHA_BARS[level.astype(jnp.int32)])
    mean = jnp.sqrt(ab)[:, None, None] * CENTERS[None, :, :]
    variance = ab * 0.09 + (1.0 - ab)
    d2 = jnp.sum((x[:, None, :] - mean) ** 2, axis=-1)
    logits = -0.5 * d2 / variance[:, None]
    weights = jax.nn.softmax(logits, axis=-1)
    centers = CENTERS[None, :, :]
    x0_mean = centers + (
        jnp.sqrt(ab) * 0.09 / variance
    )[:, None, None] * (x[:, None, :] - mean)
    component_noise = (
        x[:, None, :] - jnp.sqrt(ab)[:, None, None] * x0_mean
    ) / jnp.sqrt(1.0 - ab)[:, None, None]
    return jnp.sum(weights[:, :, None] * component_noise, axis=1)


def exact_velocity(x: jax.Array, s: jax.Array) -> jax.Array:
    s = jnp.asarray(s, jnp.float32)
    mean = (1.0 - s) * CENTERS[None, :, :]
    variance = (1.0 - s) ** 2 * 0.09 + s**2
    centered = x[:, None, :] - mean
    weights = jax.nn.softmax(-0.5 * jnp.sum(centered**2, -1) / variance, axis=-1)
    component_x0 = CENTERS[None, :, :] + (
        0.09 * (1.0 - s) / variance
    ) * centered
    component_z = (s / variance) * centered
    return jnp.sum(weights[:, :, None] * (component_z - component_x0), axis=1)


def flow_sample(params: dict[str, jax.Array], x_start: jax.Array, steps: int) -> jax.Array:
    grid = jnp.linspace(1.0, 0.0, steps + 1, dtype=jnp.float32)

    @jax.jit
    def run(x0):
        def body(x, pair):
            s, sp = pair
            v = mlp_forward(params, x, s[None].repeat(x.shape[0]), flow=True)
            return x - (s - sp) * v, None

        x, _ = jax.lax.scan(body, x0, (grid[:-1], grid[1:]))
        return x

    return run(x_start)


def full_grid(steps: int) -> np.ndarray:
    return np.rint(np.linspace(T, 0, steps + 1)).astype(np.int32)


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


def energy_distance(x: np.ndarray, y: np.ndarray, chunk: int = 512) -> float:
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)

    def cross_mean(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        for start in range(0, len(a), chunk):
            block = a[start : start + chunk]
            distances = np.sqrt(
                np.maximum(
                    np.sum(block * block, axis=1)[:, None]
                    + np.sum(b * b, axis=1)[None, :]
                    - 2.0 * block @ b.T,
                    0.0,
                )
            )
            total += float(np.sum(distances))
        return total / (len(a) * len(b))

    def within_mean(a: np.ndarray) -> float:
        total = 0.0
        for start in range(0, len(a), chunk):
            block = a[start : start + chunk]
            distances = np.sqrt(
                np.maximum(
                    np.sum(block * block, axis=1)[:, None]
                    + np.sum(a * a, axis=1)[None, :]
                    - 2.0 * block @ a.T,
                    0.0,
                )
            )
            if start == 0:
                np.fill_diagonal(distances, 0.0)
            total += float(np.sum(distances))
        return total / (len(a) * (len(a) - 1))

    return 2.0 * cross_mean(x, y) - within_mean(x) - within_mean(y)


def save_params(path: Path, params: dict[str, np.ndarray]) -> None:
    np.savez(path, **params)


def load_params(path: Path) -> dict[str, jax.Array]:
    with np.load(path) as data:
        return _decode_diffusion_params({name: data[name] for name in data.files})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
