"""Part 1, check (1): the hand-written modes are the sheet's numbers.

The artifact-level checks live in `scripts/ad_modes.py`; this test imports the
same functions so a regression fails before the figures are redrawn.
"""

import sys
from pathlib import Path

import numpy as np

WEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK / "scripts"))

import ad_modes  # noqa: E402


def test_hand_written_modes_match_the_analytic_derivative():
    forward = ad_modes.hand_forward(1.3)
    reverse = ad_modes.hand_reverse(1.3)
    exact = float(ad_modes.analytic(1.3))
    assert abs(forward["energy"] - (-0.6570169144600471)) < 1e-15
    assert abs(forward["tangents"]["U"] - exact) < 1e-12
    assert abs(reverse["adjoints"]["r"] - exact) < 1e-12
    assert abs(reverse["adjoints"]["a"] - (-2.3425903117359734)) < 1e-12


def test_sweep_errors_beat_the_finite_difference():
    _, _, forward, reverse, exact, finite_difference = ad_modes.sweep(601, 0.95, 2.5, 1e-6)
    forward_error = float(np.max(np.abs(forward - exact)))
    reverse_error = float(np.max(np.abs(reverse - exact)))
    finite_error = float(np.max(np.abs(finite_difference - exact)))
    assert forward_error < 1e-12 and reverse_error < 1e-12
    assert finite_error > max(forward_error, reverse_error)
