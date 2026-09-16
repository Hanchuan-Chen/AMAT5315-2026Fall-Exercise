#!/usr/bin/env python3
"""Tests for the Part 2 susceptibility analysis in `peaks.py`.

The tests never touch `week3/artifacts/`: they build a synthetic run folder in
a temporary directory with magnetization columns whose means are known in
closed form, so the contract of the analysis is pinned independently of the
160 MB of measured rows.

Run from `week3/`:

    .venv/bin/python -m unittest scripts.test_peaks -v
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import peaks  # noqa: E402


def write_run(folder: Path, l: int, columns: dict[float, list[float]], seed: int = 7) -> None:
    """Write a minimal `run.json` and `series.jsonl` with given M columns."""
    folder.mkdir(parents=True, exist_ok=True)
    grid = [t for t in sorted(columns)]
    (folder / "run.json").write_text(
        json.dumps(
            {
                "L": l,
                "update": "metropolis",
                "t_grid": grid,
                "discard": 2000,
                "measure": max(len(v) for v in columns.values()),
                "seed": seed,
                "sample_every": 1,
                "time_unit": "sweep",
            }
        )
    )
    with (folder / "series.jsonl").open("w") as handle:
        for t in grid:
            for k, m in enumerate(columns[t], start=1):
                handle.write(
                    json.dumps(
                        {
                            "L": l,
                            "T": t,
                            "sweep": k,
                            "M": round(m, 6),
                            "E": round(-1.5 + 0.001 * k, 6),
                        }
                    )
                    + "\n"
                )


class ReadRunTests(unittest.TestCase):
    def test_read_run_groups_by_temperature_in_ramp_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "coarse-l8"
            write_run(folder, 8, {1.5: [1.0, -1.0, 0.5, -0.5], 1.6: [1.0, 0.9, 0.8, 0.7]})
            l, stats = peaks.read_run(folder)
            self.assertEqual(l, 8)
            self.assertEqual([s.t for s in stats], [1.5, 1.6])
            self.assertEqual([s.n for s in stats], [4, 4])
            self.assertAlmostEqual(stats[0].mean_abs_m, 0.75, places=12)
            self.assertAlmostEqual(stats[0].mean_m_sq, 0.625, places=12)
            self.assertAlmostEqual(stats[1].mean_abs_m, 0.85, places=12)
            self.assertAlmostEqual(stats[1].mean_m_sq, 0.735, places=12)


class SusceptibilityTests(unittest.TestCase):
    def test_chi_is_l_squared_times_variance_of_abs_m_over_t(self) -> None:
        # |m| mean 0.75 with |m|^2 mean 0.625 -> variance 0.0625,
        # chi = 64 * 0.0625 / 1.5.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run"
            write_run(folder, 8, {1.5: [1.0, -1.0, 0.5, -0.5]})
            l, stats = peaks.read_run(folder)
            series = peaks.chi_series(l, stats)
            self.assertEqual(len(series), 1)
            t, chi = series[0]
            self.assertAlmostEqual(t, 1.5, places=12)
            self.assertAlmostEqual(chi, 64 * 0.0625 / 1.5, places=10)

    def test_chi_uses_abs_m_mean_not_signed_mean(self) -> None:
        # A perfectly ordered lattice that never flips has zero fluctuation, so
        # chi must vanish even though <m^2> = <|m|>^2 = 1.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run"
            write_run(folder, 4, {2.0: [1.0, 1.0, 1.0, 1.0]})
            l, stats = peaks.read_run(folder)
            self.assertAlmostEqual(peaks.chi_series(l, stats)[0][1], 0.0, places=12)


class FitPeakTests(unittest.TestCase):
    def test_exact_parabola_vertex_is_recovered(self) -> None:
        ts = [2.20, 2.25, 2.30, 2.35, 2.40]
        chis = [-(t - 2.31) ** 2 for t in ts]
        fit = peaks.fit_peak(ts, chis)
        self.assertAlmostEqual(fit.t_peak, 2.31, places=9)
        self.assertAlmostEqual(fit.chi_peak, 0.0, places=9)

    def test_vertex_is_not_snapped_to_the_grid(self) -> None:
        ts = [2.20, 2.25, 2.30, 2.35, 2.40]
        for vertex in (2.286, 2.3333, 2.2719):
            chis = [-3.0 * (t - vertex) ** 2 + 40.0 for t in ts]
            self.assertAlmostEqual(peaks.fit_peak(ts, chis).t_peak, vertex, places=9)

    def test_fit_uses_five_points_centred_on_the_largest_chi(self) -> None:
        ts = [2.0 + 0.05 * k for k in range(13)]
        chis = [0.0 for _ in ts]
        chis[0] = 999.0  # a huge far-away outlier must not enter the window
        chis[6] = 70.0  # T = 2.30
        chis[5], chis[7] = 60.0, 60.0
        chis[4], chis[8] = 30.0, 30.0
        chis[3], chis[9] = -10.0, -10.0
        fit = peaks.fit_peak(ts, chis)
        self.assertEqual(fit.n_points, 5)
        self.assertEqual(fit.t_min, 2.20)
        self.assertEqual(fit.t_max, 2.40)
        self.assertAlmostEqual(fit.t_peak, 2.30, places=9)

    def test_too_few_points_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            peaks.fit_peak([2.0, 2.05, 2.1], [1.0, 2.0, 1.0])


class OnsagerTests(unittest.TestCase):
    def test_matches_equation_three_below_tc(self) -> None:
        for t in (1.5, 1.8, 2.0, 2.2):
            expected = (1.0 - math.sinh(2.0 / t) ** -4) ** (1.0 / 8)
            self.assertAlmostEqual(peaks.onsager_abs_m(t), expected, places=12)

    def test_matches_the_sheets_reference_values(self) -> None:
        self.assertAlmostEqual(peaks.onsager_abs_m(1.8), 0.9569, places=4)
        self.assertAlmostEqual(peaks.onsager_abs_m(1.5), 0.9861, places=4)

    def test_vanishes_at_and_above_tc(self) -> None:
        self.assertEqual(peaks.onsager_abs_m(peaks.T_C), 0.0)
        self.assertEqual(peaks.onsager_abs_m(3.0), 0.0)

    def test_increases_as_temperature_falls(self) -> None:
        values = [peaks.onsager_abs_m(t) for t in (1.5, 1.7, 1.9, 2.1, 2.25)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[0], values[-1])

    def test_tc_is_onsagers_exact_value(self) -> None:
        self.assertAlmostEqual(peaks.T_C, 2.269185314213022, places=12)


class MergeGridTests(unittest.TestCase):
    @staticmethod
    def stats(ts: list[float], tag: float) -> list[peaks.TemperatureStats]:
        return [peaks.TemperatureStats(t, tag, tag, 1) for t in ts]

    def test_sheet_grids_merge_to_twenty_seven_temperatures(self) -> None:
        coarse = self.stats([round(1.5 + 0.1 * k, 9) for k in range(21)], 1.0)
        window = self.stats([round(2.0 + 0.05 * k, 9) for k in range(13)], 2.0)
        merged = peaks.merge_grid(coarse, window)
        self.assertEqual(len(merged), 27)
        self.assertEqual([s.t for s in merged], sorted(s.t for s in merged))
        self.assertEqual(merged[0].t, 1.5)
        self.assertEqual(merged[-1].t, 3.5)

    def test_window_rows_win_inside_the_critical_window(self) -> None:
        coarse = self.stats([2.0, 2.1, 2.2, 2.6], 1.0)
        window = self.stats([2.0, 2.05, 2.6], 2.0)
        merged = {s.t: s for s in peaks.merge_grid(coarse, window)}
        self.assertEqual(merged[2.0].mean_abs_m, 2.0)
        self.assertEqual(merged[2.6].mean_abs_m, 2.0)
        self.assertEqual(merged[2.05].mean_abs_m, 2.0)
        self.assertEqual(merged[2.1].mean_abs_m, 1.0)
        self.assertEqual(merged[2.2].mean_abs_m, 1.0)

    def test_merge_is_exact_on_temperature_values(self) -> None:
        coarse = self.stats([2.000000001], 1.0)
        window = self.stats([2.0], 2.0)
        merged = peaks.merge_grid(coarse, window)
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
