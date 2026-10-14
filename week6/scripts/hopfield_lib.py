"""Part 3's classical Hopfield network: Hebbian couplings, zero-temperature
asynchronous recall and the unstable-bit fraction.

``J_ij = (1/N) sum_mu xi_i^mu xi_j^mu`` with ``J_ii = 0`` (Eq. 12), the local
field is ``h_i = sum_j J_ij s_j``, and one asynchronous step sets
``s_i <- sign(h_i)`` in a random order, leaving ``s_i`` unchanged when
``h_i = 0``.  Recall stops when a whole sweep flips nothing.

The fields are evaluated without materializing ``J``: with ``J = (X X^T - P I)/N``
for the ``N x P`` pattern matrix ``X``, ``h_i = ((X u)_i - P s_i)/N`` where
``u = X^T s``.  Updating one spin changes ``u`` by one column, so a sweep costs
``O(N P)`` work per chain instead of ``O(N^2)``.  ``tests/test_hopfield.py``
checks the couplings, the field identity, the fixed point and the local-minimum
property of the recalled states.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def hebbian(patterns: np.ndarray) -> np.ndarray:
    """Eq. 12's couplings, ``(1/N) X X^T`` with a zero diagonal."""
    patterns = np.asarray(patterns, dtype=np.float32)
    n = patterns.shape[0]
    J = patterns @ patterns.T / n
    np.fill_diagonal(J, 0.0)
    return J


def unstable_fraction(patterns: np.ndarray) -> tuple[float, np.ndarray]:
    """Fraction of bits with ``xi_i h_i < 0`` over all stored patterns.

    Returns the fraction and the per-pattern fractions.  Uses the Gram matrix
    ``X^T X`` so the cost is ``O(N P^2)`` rather than ``O(P N^2)``.
    """
    patterns = jnp.asarray(patterns, dtype=jnp.float32)
    n, p = patterns.shape
    h = (patterns @ (patterns.T @ patterns) - p * patterns) / n
    per_pattern = jnp.mean(patterns * h < 0, axis=0)
    return float(jnp.mean(per_pattern)), np.asarray(per_pattern)


def recall(
    patterns: np.ndarray,
    starts: np.ndarray,
    key: np.ndarray,
    max_sweeps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-temperature asynchronous recall of every start at once.

    ``patterns`` is ``(N, P)`` for one pattern set shared by every chain, or
    ``(B, N, P)`` for one set per chain; ``starts`` is ``(B, N)``.  Each sweep
    visits the spins in a fresh random order and a chain stops once one of its
    sweeps flips nothing.  Returns the final states and the sweeps per chain.
    """
    x = jnp.asarray(patterns, dtype=jnp.float32)
    s0 = jnp.asarray(starts, dtype=jnp.float32)
    batch = jax.jit(lambda x, s0, key: _recall(x, s0, key, max_sweeps))
    return batch(x, s0, key)


def _recall(x, s0, key, max_sweeps):
    per_chain = x.ndim == 3
    n, p = x.shape[-2:]
    b = s0.shape[0]
    u = jnp.einsum("bnp,bn->bp", x, s0) if per_chain else s0 @ x  # (B, P)

    def sweep(carry, _unused=None):
        s, u, live, key, count = carry
        key, sub = jax.random.split(key)
        order = jax.random.permutation(sub, n)

        def spin_step(carry, index):
            s, u, flipped = carry
            row = x[:, index, :] if per_chain else x[index]  # (B, P) or (P,)
            dot = jnp.sum(u * row, axis=1) if per_chain else u @ row
            field = (dot - p * s[:, index]) / n
            new = jnp.where(field == 0, s[:, index], jnp.sign(field))
            new = jnp.where(live, new, s[:, index])
            delta = new - s[:, index]
            u = u + delta[:, None] * row
            s = s.at[:, index].set(new)
            return (s, u, flipped | (delta != 0)), None

        (s, u, flipped), _ = jax.lax.scan(
            spin_step, (s, u, jnp.zeros(b, dtype=bool)), order
        )
        return (s, u, flipped, key, count + live.astype(jnp.int32))

    init = (s0, u, jnp.ones(b, dtype=bool), key, jnp.zeros(b, dtype=jnp.int32))

    def cond(carry):
        return jnp.any(carry[2] & (carry[4] < max_sweeps))

    state = jax.lax.while_loop(cond, sweep, init)
    s, _, _, _, count = state
    return s, count
