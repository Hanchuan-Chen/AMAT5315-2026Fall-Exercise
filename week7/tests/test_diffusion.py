"""Focused checks for the target and forward diffusion."""

from __future__ import annotations

import jax
import numpy as np

from diffusion_lib import BETAS, CENTERS, T, sample_target


def test_schedule_matches_contract():
    assert T == 1000
    assert np.isclose(float(BETAS[0]), 1e-4)
    assert np.isclose(float(BETAS[-1]), 0.02)
    assert np.all(np.diff(np.asarray(BETAS)) > 0)


def test_target_has_equal_clusters_and_prescribed_width():
    x = np.asarray(sample_target(jax.random.key(1), 20000))
    labels = np.argmin(
        np.sum((x[:, None, :] - np.asarray(CENTERS)[None]) ** 2, axis=-1), axis=1
    )
    fractions = np.bincount(labels, minlength=4) / len(x)
    assert np.max(np.abs(fractions - 0.25)) < 0.015
    assert np.max(np.abs(np.std(x[labels == 0], axis=0) - 0.3)) < 0.015
