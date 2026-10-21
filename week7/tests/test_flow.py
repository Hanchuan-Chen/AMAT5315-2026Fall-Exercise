"""Focused checks for flow matching and its exact marginal velocity."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diffusion_lib import (
    exact_velocity,
    flow_sample,
    init_mlp,
    load_toml,
    mlp_forward,
)


def test_flow_config_is_the_sheet_contract():
    cfg = load_toml("flow.toml")
    assert cfg["model"]["time_scale"] == 1000.0
    assert cfg["training"]["seed"] == 2029
    assert cfg["training"]["updates"] == 25000
    assert cfg["training"]["batch_size"] == 512


def test_flow_features_use_scaled_continuous_time():
    cfg = load_toml("flow.toml")
    params = init_mlp(cfg, jax.random.key(11), flow=True)
    x = jnp.array([[0.2, -0.4], [1.0, 2.0]], jnp.float32)
    output = mlp_forward(params, x, jnp.array([0.0, 1.0], jnp.float32), flow=True)
    assert output.shape == x.shape


def test_exact_velocity_approaches_identity_at_noise_endpoint():
    x = jax.random.normal(jax.random.key(12), (64, 2))
    velocity = exact_velocity(x, 1.0)
    assert np.allclose(np.asarray(velocity), np.asarray(x), atol=2e-6)


def test_flow_euler_sampler_is_deterministic():
    cfg = load_toml("flow.toml")
    params = init_mlp(cfg, jax.random.key(13), flow=True)
    starts = jax.random.normal(jax.random.key(14), (12, 2))
    first = flow_sample(params, starts, 5)
    second = flow_sample(params, starts, 5)
    assert np.array_equal(np.asarray(first), np.asarray(second))
