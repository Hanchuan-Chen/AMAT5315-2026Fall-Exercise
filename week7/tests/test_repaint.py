"""Formula and scheduling checks for RePaint."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from run_repaint import LEVELS, proposal_mean_variance, steps


def test_repaint_visits_the_exact_declared_grid_and_evaluation_count():
    assert LEVELS[0] == 1000
    assert LEVELS[-1] == 0
    assert np.array_equal(np.diff(LEVELS), np.full(50, -20))
    times, next_times, retries = steps()
    assert len(times) == 99
    assert np.count_nonzero(retries) == 49
    assert times[-1] == 20 and next_times[-1] == 0
    assert not retries[-1]


def test_final_proposal_reduces_to_clipped_clean_estimate():
    current = jnp.array([0.3, -0.2], jnp.float32)
    clean = jnp.array([0.8, -0.7], jnp.float32)
    mean, variance = proposal_mean_variance(
        current, clean, jnp.array(20), jnp.array(0)
    )
    assert np.allclose(np.asarray(mean), np.asarray(clean), atol=1e-7)
    assert float(variance) == 0.0


def test_intermediate_proposal_keeps_state_and_clean_terms_in_forward_direction():
    current = jnp.ones((1,), jnp.float32)
    clean = -jnp.ones((1,), jnp.float32)
    mean, variance = proposal_mean_variance(
        current, clean, jnp.array(1000), jnp.array(980)
    )
    assert np.isfinite(np.asarray(mean)).all()
    assert 0.0 < float(variance) < 1.0
