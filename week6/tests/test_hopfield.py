"""Contract checks for Part 3's classical Hopfield network."""

from __future__ import annotations

import jax
import numpy as np

import hopfield_lib


def patterns(n: int, p: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (2 * rng.integers(0, 2, size=(n, p)) - 1).astype(np.float32)


def test_hebbian_is_symmetric_with_zero_diagonal():
    x = patterns(40, 5)
    j = hopfield_lib.hebbian(x)
    assert np.allclose(j, j.T)
    assert np.allclose(np.diag(j), 0.0)
    assert np.allclose(j, (x @ x.T - 5 * np.eye(40)) / 40, atol=1e-6)


def test_unstable_fraction_matches_explicit_couplings():
    x = patterns(60, 7, seed=3)
    j = hopfield_lib.hebbian(x)
    fields = j @ x
    fraction, per_pattern = hopfield_lib.unstable_fraction(x)
    assert np.isclose(fraction, np.mean(x * fields < 0), atol=1e-6)
    assert np.allclose(
        per_pattern, np.mean(x * fields < 0, axis=0), atol=1e-6
    )


def test_a_stored_pattern_has_no_unstable_bits():
    fraction, _ = hopfield_lib.unstable_fraction(patterns(50, 1, seed=5))
    assert fraction == 0.0


def test_recall_reaches_a_fixed_point_and_matches_the_dense_form():
    x = patterns(50, 5, seed=7)
    starts = x[:, :3].T.copy()
    starts[0, 4] *= -1
    starts[1, 9] *= -1
    key = jax.random.key(11)
    final, sweeps = hopfield_lib.recall(x, starts, key)
    dense, dense_sweeps = hopfield_lib.recall_with_couplings(
        hopfield_lib.hebbian(x), starts, key
    )
    assert np.array_equal(np.asarray(final), np.asarray(dense))
    assert np.array_equal(np.asarray(sweeps), np.asarray(dense_sweeps))
    assert np.all(np.abs(final) == 1)
    assert np.all(np.asarray(sweeps) >= 1)


def test_recall_leaves_a_zero_field_state_unchanged():
    ties = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=np.float32)
    final, _ = hopfield_lib.recall(np.ones((3, 1), dtype=np.float32), ties,
                                   jax.random.key(0))
    assert np.array_equal(np.asarray(final), ties)


def test_overlaps_are_one_for_the_patterns_themselves():
    x = patterns(30, 4, seed=9)
    gram = hopfield_lib.overlaps(x, x.T)
    assert np.allclose(np.diag(gram), 1.0)
    assert np.allclose(gram, (x.T @ x) / 30, atol=1e-6)
