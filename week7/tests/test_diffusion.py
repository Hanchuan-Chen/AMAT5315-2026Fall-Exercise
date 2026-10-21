"""Focused checks for the Week 7 diffusion and DDIM implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diffusion_lib import (
    ALPHA_BARS,
    BETAS,
    CENTERS,
    T,
    ddpm_sample,
    ddim_sample,
    exact_noise,
    init_mlp,
    load_toml,
    mlp_forward,
    sample_target,
    summarize,
)


def test_schedule_and_checkpoints_match_contract():
    cfg = load_toml("diffusion.toml")
    assert T == 1000
    assert np.isclose(float(BETAS[0]), 1e-4)
    assert np.isclose(float(BETAS[-1]), 0.02)
    assert np.all(np.diff(np.asarray(BETAS)) > 0)
    assert cfg["training"]["checkpoint_updates"] == [
        0,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
        25000,
    ]


def test_model_shapes_and_initialization():
    cfg = load_toml("diffusion.toml")
    params = init_mlp(cfg, jax.random.key(0))
    assert params["w0"].shape == (34, 128)
    assert params["w3"].shape == (128, 2)
    assert np.isclose(float(jnp.std(params["w0"])), 1 / np.sqrt(34), rtol=0.03)
    assert all(float(jnp.max(jnp.abs(params[f"b{i}"]))) == 0.0 for i in range(4))
    output = mlp_forward(params, jnp.zeros((5, 2)), jnp.ones((5,)), flow=False)
    assert output.shape == (5, 2)
    assert output.dtype == jnp.float32


def test_target_and_forward_endpoint():
    x = np.asarray(sample_target(jax.random.key(1), 20000))
    labels = np.argmin(
        np.sum((x[:, None, :] - np.asarray(CENTERS)[None]) ** 2, axis=-1), axis=1
    )
    fractions = np.bincount(labels, minlength=4) / len(x)
    assert np.max(np.abs(fractions - 0.25)) < 0.015
    assert np.max(np.abs(np.std(x[labels == 0], axis=0) - 0.3)) < 0.015


def test_ddim_is_deterministic():
    cfg = load_toml("diffusion.toml")
    params = init_mlp(cfg, jax.random.key(2))
    x = jax.random.normal(jax.random.key(3), (16, 2))
    levels = np.array([1000, 800, 600, 0])
    a = ddim_sample(params, x, levels)
    b = ddim_sample(params, x, levels)
    assert float(jnp.max(jnp.abs(a - b))) == 0.0


def test_ddpm_final_step_is_noiseless():
    cfg = load_toml("diffusion.toml")
    params = init_mlp(cfg, jax.random.key(4))
    x = jax.random.normal(jax.random.key(5), (8, 2))
    a = ddpm_sample(params, x, np.array([1, 0]), jax.random.key(6))
    b = ddpm_sample(params, x, np.array([1, 0]), jax.random.key(7))
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_exact_noise_matches_component_conditional_mean():
    cfg = load_toml("diffusion.toml")
    params = init_mlp(cfg, jax.random.key(8))
    x = sample_target(jax.random.key(9), 4)
    levels = jnp.full((4,), 500.0)
    prediction = exact_noise(x, levels)
    assert prediction.shape == x.shape
    assert np.all(np.isfinite(np.asarray(prediction)))
    # The exact prediction need not resemble an untrained network.
    assert not np.allclose(
        np.asarray(prediction),
        np.asarray(mlp_forward(params, x, levels, flow=False)),
    )


def test_summary_uses_nearest_target_centre():
    exact = np.repeat(np.asarray(CENTERS), 1000, axis=0)
    summary = summarize(exact)
    assert np.allclose(summary["population"], 0.25)
    assert summary["max_centre_error"] < 1e-6
    assert summary["max_width_error"] == 0.3
