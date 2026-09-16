#!/usr/bin/env python3
"""Error bars and autocorrelation times of |M| from the Metropolis ramps.

Part 3 of the week-3 sheet, "How much to trust Part 2". Part 2 fitted a
susceptibility peak and extrapolated a critical temperature without any error
bar; this script says how far those numbers can be trusted, by treating the
saved per-sweep rows as the correlated time series they are.

For every size and temperature it reports

    mean |M|,  s_naive = std(|M|) / sqrt(n),
    s_50block = std(50 block means) / sqrt(50),  their ratio,
    and tau_int = 1/2 + sum_{t>=1} rho(t)                          (Equation 13)

with `rho` the autocorrelation function of `|M|` (Equation 12). The error bar
the sheet wants is the block one: while a block is shorter than `tau_int` its
mean is still correlated with the next one, so the block standard error keeps
growing with block length and only a plateau is honest.

Estimator conventions are the sheet's own, and the truncation of the
`tau_int` sum is the rule the course's diagnostic (`week3/checker/tau`, and
the answer-key generator behind it) implements: add `rho(t)` for increasing
`t`, stop at the first negative `rho(t)`, and stop once the lag reaches six
times the running total, never looking past lag `min(n // 4, 5000) - 1`.
`tau_int_from_rho` is a transcription of that loop, so an independent
recomputation with the shipped checker reproduces this table.

The autocorrelation is evaluated by FFT, not by the checker's `O(n^2)` double
sum: for a 100000-sweep series the direct sum is minutes per temperature. The
two are algebraically identical, and `scripts/test_part3.py` pins the FFT
result against the direct sum.

Run:  .venv/bin/python scripts/errors.py
Writes:  week3/evidence/errors.txt  (header line first, then one row per size
         and temperature, so `awk 'NR == 1 || ($1 == 64 && ...)' works)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import peaks

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent

#: The sheet's fixed block count for the table's blocked error.
NBLOCKS = 50

#: The checker's hard cap on the lag it will sum: `min(n // 4, 5000)`.
TAU_MAX_LAG = 5000


@dataclass(frozen=True)
class Row:
    """One (L, T) line of the table."""

    L: int
    T: float
    mean_abs_m: float
    n: int
    naive: float
    blocked: float
    ratio: float
    tau: float
    source: str = ""


def read_series(folder: Path) -> tuple[int, list[tuple[float, np.ndarray]]]:
    """Return `(L, [(T, signed M series), ...])` for one run folder.

    `series.jsonl` holds one object per measured sweep in ramp order, so the
    file is streamed once, grouped by temperature, and only the columns the
    analysis needs are kept. A window run is 1.3 million rows; holding every
    parsed dict would waste several hundred megabytes for nothing.
    """
    run = json.loads((folder / "run.json").read_text())
    lattice = int(run["L"])
    order: list[float] = []
    columns: dict[float, list[float]] = {}
    with (folder / "series.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = float(row["T"])
            column = columns.get(t)
            if column is None:
                order.append(t)
                column = columns[t] = []
            column.append(row["M"])
    return lattice, [(t, np.array(columns[t], dtype=float)) for t in order]


def read_wolff_series(
    folder: Path,
) -> tuple[int, list[tuple[float, np.ndarray, np.ndarray]]]:
    """Return `(L, [(T, signed M, cluster_size), ...])` for a Wolff run.

    The streaming contract is `read_series`'s: one pass over `series.jsonl`,
    grouped by temperature, no parsed row kept. The cluster column is the
    number of spins flipped by each move, which is what Part 4's Equation 17
    needs (`tau_work = tau_moves * <c> / L^2`).
    """
    run = json.loads((folder / "run.json").read_text())
    lattice = int(run["L"])
    order: list[float] = []
    columns: dict[float, list[list[float]]] = {}
    with (folder / "series.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = float(row["T"])
            column = columns.get(t)
            if column is None:
                order.append(t)
                column = columns[t] = [[], []]
            column[0].append(row["M"])
            column[1].append(row["cluster_size"])
    return lattice, [
        (
            t,
            np.array(columns[t][0], dtype=float),
            np.array(columns[t][1], dtype=float),
        )
        for t in order
    ]


def naive_stderr(a: np.ndarray) -> float:
    """`s / sqrt(n)` with `s` the sample standard deviation (ddof = 1)."""
    a = np.asarray(a, dtype=float)
    if a.size < 2:
        return 0.0
    return float(a.std(ddof=1) / math.sqrt(a.size))


def block_stderr(a: np.ndarray, nblocks: int = NBLOCKS) -> float:
    """Standard error of `nblocks` equal block means of `a`.

    The blocks are the first `nblocks * floor(n / nblocks)` sweeps, exactly as
    the sheet's fifty-block estimate does it. With one block per sample the
    result is the naive error, which is what the binning chart's leftmost
    point has to be.
    """
    a = np.asarray(a, dtype=float)
    n = a.size
    if nblocks < 2 or n < nblocks:
        return 0.0 if n == 0 else naive_stderr(a)
    length = n // nblocks
    means = a[: length * nblocks].reshape(nblocks, length).mean(axis=1)
    return float(means.std(ddof=1) / math.sqrt(nblocks))


def binning_curve(
    a: np.ndarray, block_lengths: list[int]
) -> list[tuple[int, int, float]]:
    """`(block length, surviving blocks, blocked error)` for each block length."""
    a = np.asarray(a, dtype=float)
    curve: list[tuple[int, int, float]] = []
    for length in block_lengths:
        nblocks = a.size // length
        if nblocks < 2:
            curve.append((length, nblocks, float("nan")))
            continue
        means = a[: nblocks * length].reshape(nblocks, length).mean(axis=1)
        curve.append((length, nblocks, float(means.std(ddof=1) / math.sqrt(nblocks))))
    return curve


def autocorrelation(a: np.ndarray) -> np.ndarray:
    """`rho(0) .. rho(tmax - 1)` of `a`, `tmax = min(n // 4, 5000)`.

    The estimator is the checker's biased one,

        rho(t) = sum_k x_k x_{k+t} / (n v),   x demeaned, v the variance,

    but evaluated with an FFT: zero-pad past `2n - 1`, take `|rfft|^2` and
    transform back, then divide by `n v`. That is the same number as the
    double sum for every lag below `n`, in `O(n log n)` instead of `O(n^2)`.
    A constant series has no fluctuations and returns a zero curve, which
    leaves `tau_int` at `1/2` through `tau_int_from_rho`.
    """
    a = np.asarray(a, dtype=float)
    n = a.size
    tmax = min(n // 4, TAU_MAX_LAG)
    if n == 0 or tmax < 1:
        return np.zeros(1)
    x = a - a.mean()
    variance = float((x * x).mean())
    if variance == 0.0:
        return np.zeros(tmax)
    size = 1 << (2 * n - 1).bit_length()
    transform = np.fft.rfft(x, size)
    corr = np.fft.irfft(transform * np.conjugate(transform), size)[:tmax]
    return corr / (variance * n)


def tau_int_from_rho(rho: np.ndarray) -> float:
    """Equation 13 with the sheet's truncation rule, term for term.

    `tau = 1/2 + sum_{t>=1} rho(t)`, stopping before the first negative
    `rho(t)` and stopping once `t >= 6 tau` with the running total, which is
    the rule the course's checker uses. The array passed in already ends at
    the cap `min(n // 4, 5000) - 1`, so no separate cap is needed here.
    """
    tau = 0.5
    for t in range(1, len(rho)):
        value = float(rho[t])
        if value < 0.0:
            break
        tau += value
        if t >= 6.0 * tau:
            break
    return tau


def tau_int(a: np.ndarray) -> float:
    """Integrated autocorrelation time of `a` (Equations 12 and 13)."""
    return tau_int_from_rho(autocorrelation(a))


def analyse_series(lattice: int, t: float, m: np.ndarray, source: str = "") -> Row:
    """One table row from one `|M|` series."""
    a = np.abs(np.asarray(m, dtype=float))
    naive = naive_stderr(a)
    blocked = block_stderr(a, NBLOCKS)
    ratio = blocked / naive if naive > 0.0 else float("inf")
    return Row(
        L=lattice,
        T=t,
        mean_abs_m=float(a.mean()),
        n=int(a.size),
        naive=naive,
        blocked=blocked,
        ratio=ratio,
        tau=tau_int(a),
        source=source,
    )


def rows_for_run(folder: Path) -> list[Row]:
    """Every temperature of one run folder, in ramp order."""
    lattice, series = read_series(folder)
    return [analyse_series(lattice, t, m, folder.name) for t, m in series]


def merge_rows(coarse: list[Row], window: list[Row]) -> list[Row]:
    """The sheet's single grid: window rows inside the window, coarse outside.

    Matching is on the temperature the contract prints. The two grids share
    the endpoints `2.0` and `2.6`, and inside `[2.0, 2.6]` the window rows hold
    twenty times the sampling at the same contract temperatures, so they win.
    """
    merged: dict[float, Row] = {}
    for row in coarse:
        merged[round(row.T, 9)] = row
    for row in window:
        if peaks.WINDOW_LOW - peaks.T_TOL <= row.T <= peaks.WINDOW_HIGH + peaks.T_TOL:
            merged[round(row.T, 9)] = row
    return [merged[key] for key in sorted(merged)]


def merged_rows(artifacts: Path) -> list[Row]:
    """Both sizes on the merged grid: coarse outside `[2.0, 2.6]`, window inside."""
    rows: list[Row] = []
    for lattice in (32, 64):
        coarse = rows_for_run(artifacts / peaks.COARSE_RUNS[lattice])
        window = rows_for_run(artifacts / peaks.WINDOW_RUNS[lattice])
        rows.extend(merge_rows(coarse, window))
    return rows


def build_report(artifacts: Path) -> str:
    """The `errors.txt` body: header, the table, then the notes."""
    rows = merged_rows(artifacts)
    lines: list[str] = []
    lines.append(
        f"{'L':>4} {'T':>7} {'mean_abs_M':>12} {'naive':>12} {'blocked':>12}"
        f" {'ratio':>7} {'tau_int':>9} {'n':>8}"
    )
    for row in rows:
        lines.append(
            f"{row.L:>4} {row.T:>7.2f} {row.mean_abs_m:>12.6f} {row.naive:>12.7f}"
            f" {row.blocked:>12.7f} {row.ratio:>7.2f} {row.tau:>9.2f} {row.n:>8d}"
        )
    lines.append("")
    lines.append(
        "week3 part 3 - error bars and autocorrelation times of |M|, one row per "
        "size and temperature"
    )
    lines.append("")
    lines.append(
        "The grid is the same 27-temperature merged grid Part 2 used: window rows "
        f"({peaks.WINDOW_LOW:.2f} <= T <= {peaks.WINDOW_HIGH:.2f}, 100000 measured "
        "sweeps each) win inside the critical window, coarse rows (5000 measured "
        "sweeps each) stand alone outside it."
    )
    lines.append("")
    lines.append(
        "mean_abs_M = <|M|> over the measured sweeps; n = the measured sweeps used;"
    )
    lines.append(
        "naive = std(|M|, ddof=1) / sqrt(n), the error a reader gets by pretending "
        "successive sweeps are independent;"
    )
    lines.append(
        "blocked = std(50 block means, ddof=1) / sqrt(50), the first 50 * floor(n / 50) "
        "sweeps cut into 50 equal blocks;"
    )
    lines.append("ratio = blocked / naive, how badly the naive error understates it;")
    lines.append(
        "tau_int = 1/2 + sum_{t>=1} rho(t) of |M| (Equation 13), truncated as the "
        "sheet describes: stop before the first negative rho(t), stop once "
        "t >= 6 * the running total, never past lag min(n // 4, 5000) - 1."
    )
    lines.append(
        "rho is evaluated by FFT in O(n log n); scripts/test_part3.py pins the FFT "
        "curve against the checker's direct sum."
    )
    lines.append("")
    lines.append("Checks against the sheet, at L = 64:")
    for t in (1.5, 2.3, 3.5):
        row = next((r for r in rows if r.L == 64 and abs(r.T - t) < 1e-9), None)
        if row is None:
            lines.append(f"  T = {t:.2f}: no row")
            continue
        lines.append(
            f"  T = {t:.2f}: mean |M| = {row.mean_abs_m:.4f}, ratio = {row.ratio:.2f}, "
            f"tau_int = {row.tau:.2f} sweeps, n = {row.n}"
        )
    lines.append(
        "  required: ratio >= 10 and tau_int of order hundreds at T = 2.3; ratio about 2 "
        "and tau_int a few sweeps at T = 1.5 and T = 3.5."
    )
    lines.append(
        "  2000-sweep blocks at T = 2.3 are only about three autocorrelation times, so "
        "the 50-block error there is provisional: Equation 15 predicts a ratio of "
        "sqrt(2 tau_int), and the binning chart (evidence/acf-binning.png) decides "
        "whether a plateau exists at all."
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts")
    parser.add_argument("--out", type=Path, default=WEEK / "evidence" / "errors.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.artifacts)
    print(report, end="")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
