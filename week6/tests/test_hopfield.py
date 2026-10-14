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


def test_recall_reaches_a_fixed_point():
    x = patterns(50, 5, seed=7)
    starts = x[:, :3].T.copy()
    starts[0, 4] *= -1
    starts[1, 9] *= -1
    final, sweeps = hopfield_lib.recall(x, starts, jax.random.key(11))
    assert np.all(np.abs(final) == 1)
    assert np.all(np.asarray(sweeps) >= 1)
    assert np.all(np.asarray(sweeps) < 200)
    # One more sweep in any order flips nothing, so the state is a fixed point.
    field = hopfield_lib.hebbian(x) @ np.asarray(final).T
    assert np.all(field * np.asarray(final).T >= 0)


def test_recall_lands_in_a_local_minimum_of_the_energy():
    x = patterns(60, 6, seed=13)
    starts = x[:, :4].T.copy()
    starts[0, [3, 11]] *= -1
    starts[2, 7] *= -1
    final, _ = hopfield_lib.recall(x, starts, jax.random.key(12))
    j = hopfield_lib.hebbian(x)
    final = np.asarray(final)
    for state in final:
        energy = -0.5 * state @ j @ state
        for i in range(state.size):
            flipped = state.copy()
            flipped[i] *= -1
            assert -0.5 * flipped @ j @ flipped >= energy - 1e-4


def test_recall_leaves_a_zero_field_state_unchanged():
    ties = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=np.float32)
    final, _ = hopfield_lib.recall(np.ones((3, 1), dtype=np.float32), ties,
                                   jax.random.key(0))
    assert np.array_equal(np.asarray(final), ties)


def test_recall_recovers_a_corrupted_pattern_at_low_load():
    x = patterns(200, 4, seed=9)
    starts = x.T.copy()
    starts[0, :20] *= -1
    final, _ = hopfield_lib.recall(x, starts, jax.random.key(4))
    final = np.asarray(final)
    assert final.shape == starts.shape
    assert np.all(np.abs(final) == 1)
    overlaps = np.mean(final * x.T, axis=1)
    assert overlaps[0] > 0.9
    assert np.all(overlaps[1:] > 0.9)
