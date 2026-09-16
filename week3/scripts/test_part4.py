#!/usr/bin/env python3
"""Tests for the Part 4 cluster analysis in `magnetization_compare.py` and `compare.py`.

The tests never touch `week3/artifacts/`. The sampler-agreement statistic and
the verdict rule are pinned on hand values, the block bootstrap on synthetic
series whose resampling distribution is known in closed form, Equation 17 on
the sheet's reference numbers, and the cluster reader on a synthetic Wolff run
folder laid out exactly as the contract writes it.

Run from `week3/`:

    .venv/bin/python -m unittest scripts.test_part4 -v
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chi_bootstrap  # noqa: E402
import compare  # noqa: E402
import errors  # noqa: E402
import magnetization_compare  # noqa: E402


class AgreementTests(unittest.TestCase):
    def test_d_statistic_is_the_difference_over_the_combined_error(self) -> None:
        # 1.0 apart with errors 0.6 and 0.8: sqrt(0.36 + 0.64) = 1.
        self.assertAlmostEqual(
            magnetization_compare.d_statistic(1.0, 0.0, 0.6, 0.8), 1.0, places=15
        )
        # The sheet's reference numbers: 0.4310 against 0.4215, d = 0.45.
        d = magnetization_compare.d_statistic(0.4310, 0.4215, 0.0210, 0.0015)
        self.assertAlmostEqual(d, 0.4513, places=3)
        self.assertAlmostEqual(
            magnetization_compare.d_statistic(0.5, 0.5, 0.1, 0.2), 0.0, places=15
        )
        # Two exact zeros and a difference is infinitely many sigma.
        self.assertEqual(
            magnetization_compare.d_statistic(0.2, 0.1, 0.0, 0.0), float("inf")
        )
        self.assertEqual(
            magnetization_compare.d_statistic(0.2, 0.2, 0.0, 0.0), 0.0
        )

    def test_agreement_verdict_follows_the_sheet_rule(self) -> None:
        stable = {"metropolis": True, "wolff": True}
        self.assertTrue(magnetization_compare.agreement_verdict(0.45, stable).startswith("agreement"))
        self.assertNotIn(
            "provisional", magnetization_compare.agreement_verdict(0.45, stable)
        )
        # A block-length-sensitive error makes the comparison provisional even
        # when d is small, which is Part 3's finding at L = 64, T = 2.3.
        self.assertIn(
            "provisional",
            magnetization_compare.agreement_verdict(
                0.45, {"metropolis": False, "wolff": True}
            ),
        )
        self.assertIn(
            "provisional",
            magnetization_compare.agreement_verdict(
                0.45, {"metropolis": True, "wolff": False}
            ),
        )
        # A discrepancy is reported whatever the errors do.
        self.assertIn(
            "discrepancy", magnetization_compare.agreement_verdict(4.2, stable)
        )
        self.assertIn(
            "discrepancy",
            magnetization_compare.agreement_verdict(
                4.2, {"metropolis": False, "wolff": False}
            ),
        )
        # Right at the boundary the sheet's d <= 3 still counts as agreement.
        self.assertNotIn(
            "discrepancy", magnetization_compare.agreement_verdict(3.0, stable)
        )

    def test_agreement_verdict_names_the_unstable_side(self) -> None:
        text = magnetization_compare.agreement_verdict(
            0.45, {"metropolis": False, "wolff": True}
        )
        self.assertIn("metropolis", text)

    def test_stability_needs_the_three_block_lengths_to_agree(self) -> None:
        stable, verdict = chi_bootstrap.stability_verdict(
            {2000: 0.0096, 4000: 0.0095, 8000: 0.0092}
        )
        self.assertTrue(stable, verdict)
        unstable, verdict = chi_bootstrap.stability_verdict(
            {2000: 0.0185, 4000: 0.0240, 8000: 0.0300}
        )
        self.assertFalse(unstable, verdict)
        self.assertIn("unresolved", verdict)


class BootstrapTests(unittest.TestCase):
    def test_an_independent_series_reproduces_the_naive_error(self) -> None:
        rng = np.random.default_rng(2026)
        series = np.abs(rng.normal(0.4, 0.15, 20000))
        naive = errors.naive_stderr(series)
        # 1000 blocks of 20: the sample sd of the block means carries about
        # 2% noise, so the two estimates agree to a few percent.
        small = magnetization_compare.block_bootstrap_mean_error(
            series, block_length=20, replicates=3000, rng=np.random.default_rng(7)
        )
        self.assertAlmostEqual(small / naive, 1.0, delta=0.05)
        # 200 blocks of 100: the same estimate on a realization whose block
        # means happen to sit a few percent low, which is the sample-size
        # noise a single series has and not a bias of the resampling.
        large = magnetization_compare.block_bootstrap_mean_error(
            series, block_length=100, replicates=3000, rng=np.random.default_rng(7)
        )
        self.assertAlmostEqual(large / naive, 1.0, delta=0.10)

    def test_two_constant_blocks_have_the_exact_resampling_spread(self) -> None:
        # Two blocks of 500, means 0 and 2. Drawing two blocks with
        # replacement makes the resampled mean 0, 1 or 2 with probabilities
        # 1/4, 1/2, 1/4, whose standard deviation is sqrt(1/2).
        series = np.repeat([0.0, 2.0], 500)
        error = magnetization_compare.block_bootstrap_mean_error(
            series, block_length=500, replicates=20000, rng=np.random.default_rng(11)
        )
        self.assertAlmostEqual(error, math.sqrt(0.5), delta=0.01)
        # The naive error pretends the 1000 rows are independent, and the
        # blocks are not: sqrt(0.5) is far above 2 / sqrt(1000).
        self.assertGreater(error, 10.0 * errors.naive_stderr(series))

    def test_a_single_block_series_is_a_point_mass(self) -> None:
        series = np.full(500, 0.25)
        error = magnetization_compare.block_bootstrap_mean_error(
            series, block_length=100, replicates=200, rng=np.random.default_rng(3)
        )
        self.assertEqual(error, 0.0)

    def test_block_errors_reports_one_value_per_block_length(self) -> None:
        rng = np.random.default_rng(42)
        series = np.abs(rng.normal(0.4, 0.15, 20000))
        table = magnetization_compare.block_errors(
            series,
            block_lengths=[2000, 4000, 8000],
            replicates=200,
            seed=2026,
        )
        self.assertEqual(sorted(table), [2000, 4000, 8000])
        self.assertTrue(all(value > 0.0 for value in table.values()))


class WorkTests(unittest.TestCase):
    def test_equation_17_on_the_sheet_reference_numbers(self) -> None:
        tau_work = compare.tau_work(4.736, 921.6, 64)
        self.assertAlmostEqual(tau_work, 1.0656, places=3)
        self.assertAlmostEqual(compare.work_ratio(701.8, tau_work), 658.6, places=1)

    def test_a_whole_lattice_cluster_costs_one_sweep(self) -> None:
        self.assertAlmostEqual(compare.tau_work(2.0, 4096.0, 64), 2.0, places=12)

    def test_a_single_spin_cluster_costs_one_l_squaredth_of_a_sweep(self) -> None:
        self.assertAlmostEqual(compare.tau_work(4.0, 1.0, 64), 4.0 / 4096.0, places=12)


class ClusterReaderTests(unittest.TestCase):
    def make_run(self, directory: Path) -> None:
        (directory / "run.json").write_text(
            json.dumps(
                {
                    "L": 4,
                    "update": "wolff",
                    "t_grid": [2.0],
                    "discard": 1,
                    "measure": 3,
                    "seed": 42,
                    "sample_every": 1,
                    "time_unit": "cluster_flip",
                }
            )
        )
        rows = [
            {"L": 4, "T": 2.0, "sweep": 1, "M": 0.5, "E": -1.5, "cluster_size": 7},
            {"L": 4, "T": 2.0, "sweep": 2, "M": -0.25, "E": -1.25, "cluster_size": 4},
            {"L": 4, "T": 2.0, "sweep": 3, "M": 0.75, "E": -1.75, "cluster_size": 11},
        ]
        with (directory / "series.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_reads_the_signed_magnetization_and_the_cluster_size(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.make_run(directory)
            lattice, series = errors.read_wolff_series(directory)
            self.assertEqual(lattice, 4)
            self.assertEqual(len(series), 1)
            temperature, m, cluster = series[0]
            self.assertEqual(temperature, 2.0)
            np.testing.assert_allclose(m, [0.5, -0.25, 0.75])
            np.testing.assert_allclose(cluster, [7.0, 4.0, 11.0])

    def test_the_work_normalized_time_uses_the_read_cluster_size(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.make_run(directory)
            _, series = errors.read_wolff_series(directory)
            _, m, cluster = series[0]
            tau_moves = errors.tau_int(np.abs(m))
            tau_work = compare.tau_work(tau_moves, float(cluster.mean()), 4)
            self.assertAlmostEqual(
                tau_work, tau_moves * (7.0 + 4.0 + 11.0) / 3.0 / 16.0, places=12
            )

    def test_read_series_energy_returns_the_energy_per_site(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.make_run(directory)
            lattice, series = errors.read_series_energy(directory)
            self.assertEqual(lattice, 4)
            temperature, m, energy = series[0]
            self.assertEqual(temperature, 2.0)
            np.testing.assert_allclose(m, [0.5, -0.25, 0.75])
            np.testing.assert_allclose(energy, [-1.5, -1.25, -1.75])
            # The two-part reader is the same pass as the one-column reader.
            _, plain = errors.read_series(directory)
            np.testing.assert_allclose(plain[0][1], m)


if __name__ == "__main__":
    unittest.main()
