"""Small JAX implementation shared by the Week 7 diffusion experiments."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

ROOT = Path(__file__).resolve().parents[1]
CENTERS = jnp.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]])
T = 1000
BETAS = jnp.linspace(1e-4, 0.02, T, dtype=jnp.float32)
ALPHAS = 1.0 - BETAS
ALPHA_BARS = jnp.concatenate([jnp.ones((1,), jnp.float32), jnp.cumprod(ALPHAS)])


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
