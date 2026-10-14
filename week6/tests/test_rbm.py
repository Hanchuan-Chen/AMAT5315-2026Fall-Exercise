"""Contract checks for Part 2's restricted Boltzmann machine."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rbm_lib


@pytest.fixture
def setup():
    cfg = rbm_lib.load_config()
    key = jax.random.key(0)
    v = (jax.random.uniform(key, (64, rbm_lib.N_VISIBLE)) < 0.2).astype(jnp.float32)
    params = rbm_lib.init_params(cfg, jnp.mean(v, axis=0))
    return cfg, v, params


def test_binarize_at_half_intensity():
    images = np.array([[[0, 127], [128, 255]]], dtype=np.uint8)
    assert np.array_equal(np.asarray(rbm_lib.binarize(images)), [[0, 0, 1, 1]])


def test_init_params_follow_the_toml(setup):
    cfg, v, params = setup
    assert params.W.shape == (rbm_lib.N_VISIBLE, rbm_lib.N_HIDDEN)
    assert abs(float(jnp.std(params.W)) - cfg["initialization"]["weight_std"]) < 2e-4
    assert np.allclose(np.asarray(params.b), 0.0)
    low, high = cfg["initialization"]["pixel_frequency_clip"]
    frequency = np.clip(np.asarray(jnp.mean(v, axis=0)), low, high)
    expected = np.log(frequency / (1 - frequency))
    assert np.allclose(np.asarray(params.a), expected, atol=1e-6)
    assert float(jnp.max(jnp.abs(params.a))) < 20


def test_free_energy_sums_over_the_hidden_layer(setup):
    _, v, params = setup
    width = 4
    sliced = rbm_lib.Params(W=params.W[:, :width], a=params.a, b=params.b[:width])
    images = v[:3]
    hidden = jnp.array(list(itertools.product([0.0, 1.0], repeat=width)))
    energies = rbm_lib.energy(sliced, images[:, None, :], hidden[None, :, :])
    manual = -jax.scipy.special.logsumexp(-energies, axis=1)
    assert np.allclose(
        np.asarray(manual),
        np.asarray(rbm_lib.free_energy(sliced, images)),
        atol=1e-4,
    )


def test_cd_gradient_matches_the_explicit_update(setup):
    _, v, params = setup
    images = v[:8]
    vk = rbm_lib.gibbs(params, images, 3, jax.random.key(1))
    _, grads = rbm_lib.cd_gradient(params, images, vk)
    batch = images.shape[0]
    w_expected = -(
        images.T @ rbm_lib.hidden_probability(params, images)
        - vk.T @ rbm_lib.hidden_probability(params, vk)
    ) / batch
    b_expected = -(
        jnp.mean(rbm_lib.hidden_probability(params, images), axis=0)
        - jnp.mean(rbm_lib.hidden_probability(params, vk), axis=0)
    )
    a_expected = -(jnp.mean(images, axis=0) - jnp.mean(vk, axis=0))
    assert np.allclose(np.asarray(grads.W), np.asarray(w_expected), atol=1e-5)
    assert np.allclose(np.asarray(grads.b), np.asarray(b_expected), atol=1e-5)
    assert np.allclose(np.asarray(grads.a), np.asarray(a_expected), atol=1e-5)


def test_gibbs_draws_match_the_conditional_probabilities(setup):
    _, v, params = setup
    image = v[:1]
    draws = rbm_lib.sample_hidden(
        params, jnp.broadcast_to(image, (4000, rbm_lib.N_VISIBLE)), jax.random.key(2)
    )
    expected = rbm_lib.hidden_probability(params, image)[0]
    standard_error = jnp.sqrt(expected * (1 - expected) / 4000)
    spread = jnp.max(jnp.abs(jnp.mean(draws, axis=0) - expected) / standard_error)
    assert float(spread) < 5


def test_reconstruction_error_is_the_changed_fraction(setup):
    _, v, params = setup
    images = v[:16]
    error = rbm_lib.reconstruction_error(params, images, jax.random.key(3))
    drawn = rbm_lib.gibbs_step(params, images, jax.random.key(3))
    assert np.isclose(float(error), float(jnp.mean(drawn != images)))


def test_training_uses_the_same_randomness_for_both_models(setup):
    cfg, _, _ = setup
    first = rbm_lib.training_keys(cfg, 0, 6)
    second = rbm_lib.training_keys(cfg, 0, 6)
    first_data, second_data = jax.random.key_data(first), jax.random.key_data(second)
    assert np.array_equal(np.asarray(first_data), np.asarray(second_data))
    assert not np.array_equal(np.asarray(first_data[0]), np.asarray(first_data[1]))
