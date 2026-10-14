"""Contract checks for Part 4's modern Hopfield network."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import modern_lib


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    x = (2 * rng.integers(0, 2, size=(modern_lib.N_PIXELS, 30)) - 1).astype(
        np.float32
    )
    state = (2 * rng.integers(0, 2, size=(modern_lib.N_PIXELS, 5)) - 1).astype(
        np.float32
    )
    return jnp.asarray(x), jnp.asarray(state)


@pytest.mark.parametrize("beta", [0.001, 0.01, 0.3, 1.0])
def test_modern_update_equals_softmax_attention(setup, beta):
    x, state = setup
    update = modern_lib.modern_update(x, state, beta)
    attention = modern_lib.attention_update(x, state, beta)
    assert float(jnp.max(jnp.abs(update - attention))) < 1e-5


def test_the_default_attention_scale_is_not_beta(setup):
    x, state = setup
    keys = values = x.T[None, :, None, :]

    def one(query):
        out = jax.nn.dot_product_attention(query[None, None, None, :], keys, values)
        return out[0, 0, 0]

    default = jax.vmap(one)(state.T).T
    update = modern_lib.modern_update(x, state, 0.3)
    assert float(jnp.max(jnp.abs(update - default))) > 0.5


def test_softmax_weights_are_a_distribution(setup):
    x, state = setup
    weights = modern_lib.softmax_weights(x, state, 0.03)
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-6)
    assert np.all(weights >= 0.0)


def test_retrieval_success_needs_the_nearest_stored_digit(setup):
    x, _ = setup
    targets = np.array([0, 1], dtype=np.int32)
    exact = x[:, jnp.asarray(targets)]
    assert modern_lib.retrieval_success(x, exact, targets).all()
    wrong = x[:, jnp.array([0, 2])]
    success = modern_lib.retrieval_success(x, wrong, targets)
    assert bool(success[0]) and not bool(success[1])


def test_corrupted_queries_keep_the_observed_upper_half():
    rng = np.random.default_rng(1)
    x = jnp.asarray(
        (2 * rng.integers(0, 2, size=(modern_lib.N_PIXELS, 4)) - 1).astype(
            np.float32
        )
    )
    targets = jnp.arange(4)
    queries = modern_lib.corrupted_queries(x, targets, jax.random.key(0))
    assert np.array_equal(
        np.asarray(queries[: modern_lib.HALF]), np.asarray(x[: modern_lib.HALF])
    )
    lower = np.asarray(queries[modern_lib.HALF :])
    assert set(np.unique(lower)).issubset({-1.0, 1.0})
