"""Part 4's modern Hopfield network: the one-step update and its attention form.

The first ``P`` MNIST training images in stored order, binarized at half
intensity and turned into spins, are the columns of ``X``.  The modern update of
Eq. 17 is ``x <- X softmax(beta X^T x)``, a softmax-weighted average of the
stored patterns; with ``x`` as the query and the patterns as keys and values it
is exactly softmax attention of Eq. 18, which JAX computes with
``jax.nn.dot_product_attention(..., scale=beta)``.  The classical comparison
uses Part 3's Hebbian recall on the same patterns.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import datasets
import hopfield_lib
import rbm_lib

N_PIXELS = 784
HALF = 28 * 14
BETAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
PS = (3, 10, 30, 100, 300, 1000)
QUERIES = 100
SUCCESS_OVERLAP = 0.95


def stored_patterns(p: int) -> np.ndarray:
    """The first ``p`` training images, in stored order, as spin columns."""
    x_train, _, _, _ = datasets.read_mnist()
    visible = rbm_lib.binarize(x_train[:p])
    return np.asarray(rbm_lib.spins_of(visible)).T.astype(np.float32)


def corrupted_queries(x: np.ndarray, targets: np.ndarray, key) -> np.ndarray:
    """Cycle through the stored digits, replacing each lower half by random spins."""
    upper = x[:HALF, targets]
    lower = 2.0 * jax.random.bernoulli(key, 0.5, (HALF, targets.size)) - 1.0
    return jnp.concatenate([upper, lower], axis=0)


def modern_update(x: np.ndarray, state: jnp.ndarray, beta: float) -> jnp.ndarray:
    """Eq. 17: x <- X softmax(beta X^T x), one update, no iteration."""
    weights = jax.nn.softmax(beta * (x.T @ state), axis=0)
    return x @ weights


def attention_update(x: np.ndarray, state: jnp.ndarray, beta: float) -> jnp.ndarray:
    """The same update as JAX softmax attention with keys and values = patterns."""
    keys = values = x.T[None, :, None, :]  # (batch, source, heads=1, head dim)

    def one(query):
        out = jax.nn.dot_product_attention(
            query[None, None, None, :], keys, values, scale=beta
        )
        return out[0, 0, 0]

    return jax.vmap(one)(state.T).T


def softmax_weights(x: np.ndarray, state: jnp.ndarray, beta: float) -> np.ndarray:
    return np.asarray(jax.nn.softmax(beta * (x.T @ state), axis=0))


def retrieval_success(
    x: np.ndarray, result: jnp.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Overlap at least 0.95 with the target and the target is the nearest
    stored digit to the result."""
    n_patterns = x.shape[1]
    overlaps = (x.T @ result) / N_PIXELS
    target_overlap = overlaps[targets, jnp.arange(targets.size)]
    nearest = jnp.argmax(overlaps, axis=0)
    return np.asarray(
        (target_overlap >= SUCCESS_OVERLAP) & (nearest == targets)
    )


def classical_recall(x: np.ndarray, starts: jnp.ndarray, key) -> jnp.ndarray:
    """Part 3's recall on the same patterns, from the same corrupted queries."""
    couplings = hopfield_lib.hebbian(x)
    final, _ = hopfield_lib.recall_with_couplings(couplings, starts.T, key)
    return final.T
