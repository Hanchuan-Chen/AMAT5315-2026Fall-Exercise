#!/usr/bin/env python3
"""Tests for the Part 3 uncertainty analysis in `errors.py` and `chi_bootstrap.py`.

The tests never touch `week3/artifacts/`: every fixture is synthetic, with an
autocorrelation function or a susceptibility that is known in closed form, so
the estimator conventions are pinned independently of the 2.8 million measured
rows. In particular the truncation rule is checked against sums worked out by
hand, because that rule is what a cross-check with the course's own
`week3/checker/tau` diagnostic reproduces.

Run from `week3/`:

    .venv/bin/python -m unittest scripts.test_part3 -v
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chi_bootstrap  # noqa: E402
import errors  # noqa: E402


def direct_acf(a: np.ndarray) -> np.ndarray:
    """The checker's autocorrelation, written as the direct double sum.

    `rho(t) = sum_k x_k x_{k+t} / (n v)` with `x` demeaned and `v` the
    population variance. This is the reference the FFT estimator must match.
    """
    x = np.asarray(a, dtype=float)
    x = x - x.mean()
    n = x.size
    v = float((x * x).mean())
    if v == 0.0:
        return np.zeros(n)
    rho = np.empty(n)
    for t in range(n):
        rho[t] = float((x[: n - t] * x[t:]).sum()) / (v * n)
    return rho


class ErrorBarTests(unittest.TestCase):
    def test_naive_stderr_is_the_sample_sd_over_root_n(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = float(np.std(a, ddof=1) / math.sqrt(5))
        self.assertAlmostEqual(errors.naive_stderr(a), expected, places=15)
        # std(ddof=1) of [1,2,3,4,5] is sqrt(2.5), so the error is sqrt(0.5).
        self.assertAlmostEqual(errors.naive_stderr(a), math.sqrt(0.5), places=15)

    def test_block_stderr_is_the_sem_of_fifty_means(self) -> None:
        # 100 samples in 50 pairs: block means are 1.5, 3.5, ..., 99.5, an
        # arithmetic sequence whose standard error is known in closed form.
        a = np.arange(1.0, 101.0)
        blocks = a.reshape(50, 2).mean(axis=1)
        expected = float(np.std(blocks, ddof=1) / math.sqrt(50))
        self.assertAlmostEqual(errors.block_stderr(a, 50), expected, places=15)
        # The block means 1.5, 3.5, ... 99.5 are the odd offsets -49 .. 49
        # around 50.5, so their sample variance is 2 * sum(odd^2) / 49 = 850
        # and their SEM is sqrt(850 / 50) = sqrt(17).
        self.assertAlmostEqual(expected, math.sqrt(17.0), places=15)

    def test_block_stderr_of_two_constant_blocks_is_one(self) -> None:
        # Block means 0 and 2 have sd sqrt(2), so their SEM over two blocks is 1.
        self.assertAlmostEqual(errors.block_stderr(np.array([0.0, 0.0, 2.0, 2.0]), 2), 1.0, places=15)

    def test_block_stderr_with_one_block_per_sample_is_the_naive_error(self) -> None:
        a = np.array([0.4, 0.9, 0.1, 0.7, 0.5, 0.2])
        self.assertAlmostEqual(
            errors.block_stderr(a, a.size), errors.naive_stderr(a), places=15
        )

    def test_block_stderr_of_a_constant_series_is_zero(self) -> None:
        self.assertAlmostEqual(errors.block_stderr(np.full(100, 0.42), 50), 0.0, places=14)

    def test_binning_curve_starts_at_the_naive_error(self) -> None:
        # An AR(1) series with phi = 0.9 has tau_int near 9.5, so 100-sweep
        # blocks are past the correlation time and the estimate must have
        # grown well past the naive one.
        rng = np.random.default_rng(11)
        noise = rng.normal(size=20000)
        a = np.empty(20000)
        a[0] = noise[0]
        for k in range(1, 20000):
            a[k] = 0.9 * a[k - 1] + noise[k]
        curve = errors.binning_curve(a, [1, 2, 4, 100])
        self.assertEqual([b for b, _, _ in curve], [1, 2, 4, 100])
        self.assertEqual([nb for _, nb, _ in curve], [20000, 10000, 5000, 200])
        self.assertAlmostEqual(curve[0][2], errors.naive_stderr(a), places=15)
        self.assertLess(curve[0][2], curve[-1][2])
        self.assertGreater(curve[3][2] / curve[0][2], 2.0)
        # The growth is monotone from block length one to four.
        self.assertLess(curve[0][2], curve[1][2])
        self.assertLess(curve[1][2], curve[2][2])


class AutocorrelationTests(unittest.TestCase):
    def test_rho_zero_is_exactly_one(self) -> None:
        rng = np.random.default_rng(3)
        rho = errors.autocorrelation(rng.normal(size=4000))
        self.assertAlmostEqual(float(rho[0]), 1.0, places=14)

    def test_fft_autocorrelation_matches_the_direct_sum(self) -> None:
        rng = np.random.default_rng(5)
        a = np.cumsum(rng.normal(size=500))
        rho = errors.autocorrelation(a)
        reference = direct_acf(a)
        np.testing.assert_allclose(rho[:10], reference[:10], rtol=0, atol=1e-12)
        np.testing.assert_allclose(rho[17], reference[17], rtol=0, atol=1e-12)

    def test_autocorrelation_is_capped_at_a_quarter_of_the_series(self) -> None:
        # The checker never looks past lag min(n // 4, 5000) - 1.
        rho = errors.autocorrelation(np.arange(200.0))
        self.assertEqual(rho.size, 50)
        rho = errors.autocorrelation(np.arange(40000.0))
        self.assertEqual(rho.size, 5000)

    def test_autocorrelation_of_a_constant_series_is_degenerate(self) -> None:
        rho = errors.autocorrelation(np.full(100, 1.0))
        self.assertEqual(rho.size, 25)
        self.assertEqual(float(rho[0]), 0.0)
        # A zero tail leaves tau_int at the 1/2 of Equation 13.
        self.assertAlmostEqual(errors.tau_int_from_rho(rho), 0.5, places=14)


class TauIntTruncationTests(unittest.TestCase):
    def test_geometric_rho_sums_to_the_closed_form(self) -> None:
        # rho(t) = phi^t, phi = 0.9, so the untruncated tau_int would be
        # 1/2 + phi/(1 - phi) = 9.5. The six-times rule fires first: after
        # t terms the running total is 0.5 + 9 (1 - 0.9^t), and
        # 6 * (0.5 + 9 (1 - 0.9^56)) = 56.7 > 56 while
        # 6 * (0.5 + 9 (1 - 0.9^57)) = 56.87 <= 57, so the loop stops at
        # t = 57 and the answer is 1/2 + sum_{t=1}^{57} 0.9^t.
        phi = 0.9
        rho = np.array([1.0] + [phi**t for t in range(1, 80)])
        expected = 0.5 + sum(phi**t for t in range(1, 58))
        self.assertAlmostEqual(errors.tau_int_from_rho(rho), expected, places=12)
        self.assertLess(errors.tau_int_from_rho(rho), 9.5)
        self.assertGreater(errors.tau_int_from_rho(rho), 9.47)

    def test_constant_positive_tail_trips_the_six_times_rule(self) -> None:
        # rho(t) = 0.02 for every t >= 1. After adding t terms the running
        # total is 0.5 + 0.02 t, and the loop stops at the first t with
        # t >= 6 (0.5 + 0.02 t), that is 0.88 t >= 3, so t = 4:
        # tau_int = 0.5 + 4 * 0.02.
        rho = np.array([1.0] + [0.02] * 50)
        self.assertAlmostEqual(errors.tau_int_from_rho(rho), 0.58, places=14)

    def test_a_negative_rho_stops_the_sum_early(self) -> None:
        # 0.5 + 0.4 + 0.3 = 1.2 and the six-times rule has not fired at t = 2
        # (2 < 7.2); the negative term at t = 3 is not added.
        rho = np.array([1.0, 0.4, 0.3, -0.05, 0.5, 0.5])
        self.assertAlmostEqual(errors.tau_int_from_rho(rho), 1.2, places=14)

    def test_a_negative_first_lag_leaves_tau_at_one_half(self) -> None:
        rho = np.array([1.0, -0.2, 0.5])
        self.assertAlmostEqual(errors.tau_int_from_rho(rho), 0.5, places=14)

    def test_ar1_series_recovers_the_closed_form_tau(self) -> None:
        # x_{k+1} = phi x_k + noise has rho(t) = phi^t, so tau_int = 9.5 for
        # phi = 0.9. The sampled autocorrelation of 20000 points turns negative
        # near lag 34, before the six-times rule fires at 6 * 9.5 = 57, so the
        # truncated estimate lands about ten percent low; the test pins that
        # the estimator is near the closed form, and the hand-computed cases
        # above pin the truncation rule itself.
        rng = np.random.default_rng(2026)
        phi = 0.9
        noise = rng.normal(size=20000)
        x = np.empty(20000)
        x[0] = noise[0]
        for k in range(1, 20000):
            x[k] = phi * x[k - 1] + noise[k]
        self.assertAlmostEqual(errors.tau_int(x), 9.5, delta=1.0)
        self.assertGreater(errors.tau_int(x), 2.0 * errors.tau_int(rng.normal(size=20000)))


class MergeTests(unittest.TestCase):
    def row(self, t: float, tag: float) -> errors.Row:
        return errors.Row(
            L=8,
            T=t,
            mean_abs_m=tag,
            n=1,
            naive=0.0,
            blocked=0.0,
            ratio=0.0,
            tau=0.0,
        )

    def test_window_rows_win_inside_the_window(self) -> None:
        coarse = [self.row(1.5, 1.0), self.row(2.0, 2.0), self.row(2.3, 3.0), self.row(3.5, 4.0)]
        window = [self.row(2.0, 20.0), self.row(2.3, 30.0), self.row(2.6, 40.0)]
        merged = errors.merge_rows(coarse, window)
        self.assertEqual([r.T for r in merged], [1.5, 2.0, 2.3, 2.6, 3.5])
        self.assertEqual([r.mean_abs_m for r in merged], [1.0, 20.0, 30.0, 40.0, 4.0])

    def test_the_sheet_grid_merges_to_twenty_seven_temperatures(self) -> None:
        coarse = [self.row(round(1.5 + 0.1 * k, 2), 1.0) for k in range(21)]
        window = [self.row(round(2.0 + 0.05 * k, 2), 2.0) for k in range(13)]
        self.assertEqual(len(errors.merge_rows(coarse, window)), 27)


def exact_window(
    lattice: int, vertex: float, curvature: float, peak: float, n: int
) -> dict[float, np.ndarray]:
    """A window grid whose susceptibility is exactly `peak + curvature*(T - vertex)^2`.

    Every series alternates `0, 2u, 0, 2u, ...`, so `|m|` has mean `u` and mean
    square `2u^2` whatever the phase: the variance is `u^2` and
    `chi = L^2 u^2 / T`. Choosing `u^2 = chi(T) T / L^2` therefore lands the
    sampled `chi` on the requested parabola, and every whole block of the same
    alternating pattern has the same mean, so a block bootstrap reproduces it
    exactly.
    """
    series: dict[float, np.ndarray] = {}
    for k in range(13):
        t = round(2.0 + 0.05 * k, 2)
        chi = peak + curvature * (t - vertex) ** 2
        if chi < 0.0:
            raise ValueError("the fixture parabola must stay positive on the grid")
        u = math.sqrt(chi * t / (lattice * lattice))
        block = np.array([0.0, 2.0 * u])
        series[t] = np.tile(block, n // 2)
    return series


class BootstrapTests(unittest.TestCase):
    def test_resampling_constant_blocks_leaves_chi_alone(self) -> None:
        rng = np.random.default_rng(7)
        abs_blocks = np.full(10, 0.4)
        sq_blocks = np.full(10, 0.2)
        for _ in range(5):
            chi = chi_bootstrap.resample_chi(abs_blocks, sq_blocks, 64, 2.3, rng)
            self.assertAlmostEqual(chi, 64 * 64 * (0.2 - 0.4**2) / 2.3, places=12)

    def test_an_exact_parabola_window_reproduces_its_vertex(self) -> None:
        n = 20000
        window = {
            32: exact_window(32, 2.35, -40.0, 30.0, n),
            64: exact_window(64, 2.31, -50.0, 60.0, n),
        }
        result = chi_bootstrap.bootstrap_tc(window, block_length=2000, replicates=20, seed=99)
        self.assertEqual(result.failures, 0)
        self.assertEqual(result.successes, 20)
        self.assertAlmostEqual(result.central_tc, 2.0 * 2.31 - 2.35, places=9)
        np.testing.assert_allclose(result.tc_values, result.central_tc, atol=1e-9)
        self.assertAlmostEqual(result.error, 0.0, places=9)

    def test_a_parabola_that_does_not_bend_downward_is_a_failed_fit(self) -> None:
        n = 20000
        # A positive curvature is a parabola that does not bend downward, the
        # first kind of failed fit the sheet names.
        window = {
            32: exact_window(32, 2.35, -40.0, 30.0, n),
            64: exact_window(64, 2.31, +50.0, 60.0, n),
        }
        result = chi_bootstrap.bootstrap_tc(window, block_length=2000, replicates=5, seed=5)
        self.assertEqual(result.successes, 0)
        self.assertEqual(result.failures, 5)
        self.assertTrue(math.isnan(result.error))

    def test_a_vertex_outside_the_fitted_five_points_is_a_failed_fit(self) -> None:
        self.assertFalse(
            chi_bootstrap.fit_is_acceptable(
                chi_bootstrap.FitOutcome(t_peak=2.95, t_min=2.20, t_max=2.40, curvature=-1.0)
            )
        )
        self.assertTrue(
            chi_bootstrap.fit_is_acceptable(
                chi_bootstrap.FitOutcome(t_peak=2.31, t_min=2.20, t_max=2.40, curvature=-1.0)
            )
        )
        self.assertFalse(
            chi_bootstrap.fit_is_acceptable(
                chi_bootstrap.FitOutcome(t_peak=2.31, t_min=2.20, t_max=2.40, curvature=+1.0)
            )
        )


if __name__ == "__main__":
    unittest.main()
