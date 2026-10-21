"""Checks for the pure-JAX translation of the downloaded UNet checkpoint."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from unet_jax import _downsample, load_unet_params, unet_forward


MODEL = Path(__file__).resolve().parents[1] / "inputs" / "cifar-model"


def test_asymmetric_downsample_matches_diffusers_padding():
    params = {
        "probe.conv.weight": jnp.ones((3, 3, 1, 1), jnp.float32),
        "probe.conv.bias": jnp.zeros((1,), jnp.float32),
    }
    x = jnp.ones((1, 32, 32, 1), jnp.float32)
    y = _downsample(params, "probe", x)
    assert y.shape == (1, 16, 16, 1)


@pytest.mark.skipif(
    not (MODEL / "diffusion_pytorch_model.safetensors").exists(),
    reason="pinned CIFAR-10 checkpoint has not been downloaded",
)
def test_fixed_predictor_is_finite_and_accepts_checkpoint_indices():
    params, config = load_unet_params(MODEL)
    assert len(params) == 328
    assert config["sample_size"] == 32
    x = jax.random.normal(jax.random.key(21), (1, 32, 32, 3), jnp.float32)
    first = unet_forward(params, x, jnp.array([499]))
    second = unet_forward(params, x, jnp.array([499]))
    assert first.shape == x.shape
    assert np.all(np.isfinite(np.asarray(first)))
    assert np.array_equal(np.asarray(first), np.asarray(second))
