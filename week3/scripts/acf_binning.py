#!/usr/bin/env python3
"""Autocorrelation and binning of |m| at L = 64, T = 2.3.

Part 3 of the week-3 sheet, check (3). Two panels from the same window series:

  left   rho(t), the autocorrelation function of |M| against lag      (Eq. 12)
  right  the standard error of <|M|> against block length, logarithmic
         block-length axis, from the naive error at block length one up to
         5000-sweep blocks and beyond

Binning cuts the series into equal blocks and takes the standard error of the
block means. While a block is shorter than `tau_int` neighbouring blocks are
still correlated and the estimate is too small; once the block is several
times `tau_int` the estimate stops growing and that plateau is the honest
error bar. The sheet warns that a fixed count of fifty blocks does not
establish a plateau: at T = 2.3 a 2000-sweep block is about three
autocorrelation times. This chart is the check, and the script says whether
the curve flattens before the blocks run out.

The leftmost point of the right panel is computed with the same function that
fills the `naive` column of `evidence/errors.txt`, so the two agree to
round-off by construction; the script asserts it and prints both.

Run:  .venv/bin/python scripts/acf_binning.py
Writes:  week3/evidence/acf-binning.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import errors  # noqa: E402

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent

#: The series the sheet points at.
LATTICE = 64
T_TARGET = 2.3
SOURCE = errors.peaks.WINDOW_RUNS[LATTICE]

#: Block lengths drawn, ending where only a couple of blocks are left.
BLOCK_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5000, 10000, 20000]

#: How far the right panel's lag axis runs.
MAX_LAG_SHOWN = 3000

#: The sheet's reference: 2000-sweep blocks leave 20 blocks, and the curve is
#: still rising there. A plateau is claimed only when the longest block
#: lengths agree within this factor.
PLATEAU_TOLERANCE = 1.10


def load_series(artifacts: Path) -> np.ndarray:
    """The `|M|` series of the window run at `T = 2.3`."""
    lattice, series = errors.read_series(artifacts / SOURCE)
    if lattice != LATTICE:
        raise SystemExit(f"{SOURCE} holds L = {lattice}")
    for t, m in series:
        if abs(t - T_TARGET) < 1e-9:
            return np.abs(m)
    raise SystemExit(f"no T = {T_TARGET} in {SOURCE}")


def plateau_verdict(curve: list[tuple[int, int, float]]) -> tuple[bool, str]:
    """Is the error resolved by a plateau, or still rising?

    The tail used is every block length from 4096 sweeps up, where the sheet's
    own reference curve is still climbing. A plateau means those estimates
    agree within `PLATEAU_TOLERANCE`.
    """
    tail = [err for length, _, err in curve if length >= 4096 and err == err]
    if len(tail) < 2:
        return False, "not enough block lengths survive to look for a plateau"
    spread = max(tail) / min(tail)
    if spread <= PLATEAU_TOLERANCE:
        return True, (
            f"plateau: the errors from 4096 sweeps up agree within "
            f"{100.0 * (spread - 1.0):.1f}% of each other"
        )
    return False, (
        f"no plateau: the error is still moving by a factor {spread:.2f} between "
        f"{min(tail):.4f} and {max(tail):.4f} over the longest block lengths"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts")
    parser.add_argument("--out", type=Path, default=WEEK / "evidence" / "acf-binning.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    a = load_series(args.artifacts)
    rho = errors.autocorrelation(a)
    tau = errors.tau_int_from_rho(rho)
    curve = errors.binning_curve(a, BLOCK_LENGTHS)
    naive = errors.naive_stderr(a)

    assert abs(curve[0][2] - naive) < 1e-15, "block length one must be the naive error"

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=160)

    lags = np.arange(rho.size)
    shown = lags <= MAX_LAG_SHOWN
    left.plot(lags[shown], rho[shown], color="tab:blue", linewidth=1.2)
    left.axhline(0.0, color="0.6", linewidth=0.8)
    negatives = np.flatnonzero(rho < 0.0)
    if negatives.size:
        left.axvline(
            int(negatives[0]),
            color="0.45",
            linestyle="--",
            linewidth=1.0,
            label=f"first negative lag {int(negatives[0])}",
        )
        left.legend(loc="upper right", fontsize=9)
    left.set_xlabel("lag $t$ (sweeps)")
    left.set_ylabel(r"$\rho(t)$ of $|m|$ at $T$ = 2.3")
    left.set_xlim(0, MAX_LAG_SHOWN)
    left.set_ylim(-0.25, 1.0)
    left.grid(alpha=0.25, linewidth=0.6)
    left.set_title(r"$\tau_{int}$ = " + f"{tau:.0f} sweeps", fontsize=10)

    lengths = [length for length, _, _ in curve]
    errs = [err for _, _, err in curve]
    blocks = [nblocks for _, nblocks, _ in curve]
    right.plot(lengths, errs, "o-", color="tab:blue", markersize=3.5, linewidth=1.2)
    right.axhline(naive, color="0.45", linestyle=":", linewidth=1.0)
    right.annotate(
        f"$\\sigma_{{naive}}$ = {naive:.4f}",
        xy=(lengths[0], naive),
        xytext=(2.2, naive + 0.0022),
        fontsize=9,
    )
    for length, nblocks, err in zip(lengths, blocks, errs):
        if length in (1, 5000, 20000):
            right.annotate(
                f"{nblocks} block{'s' if nblocks != 1 else ''}",
                xy=(length, err),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    right.set_xscale("log")
    right.set_xlabel("block length (sweeps)")
    right.set_ylabel(r"error bar on $\langle |m| \rangle$")
    right.set_ylim(0.0, 0.03)
    right.grid(alpha=0.25, linewidth=0.6, which="both")
    right.set_title("binning: a plateau would resolve the error", fontsize=10)

    figure.suptitle(
        r"Autocorrelation and binning of $|m|$ at $L$ = 64, $T$ = 2.3,"
        f" from the window run ($n$ = {a.size})"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)

    resolved, verdict = plateau_verdict(curve)
    print(f"wrote {args.out}")
    print(f"  L = {LATTICE}, T = {T_TARGET}, source {SOURCE}, n = {a.size}")
    print(f"  tau_int = {tau:.2f} sweeps (Equation 13, sheet truncation)")
    if negatives.size:
        print(f"  rho first goes negative at lag {int(negatives[0])}")
    print(f"  naive error (block length one) = {naive:.6f}")
    print(f"  block length one point        = {curve[0][2]:.6f}  (equal to round-off)")
    for length, nblocks, err in curve:
        print(f"  block {length:6d} sweeps, {nblocks:6d} blocks left, error = {err:.6f}")
    print(f"  {'RESOLVED' if resolved else 'UNRESOLVED'}: {verdict}")
    if not resolved:
        print("  sampling error unresolved: report the honest error as at least the")
        print(f"  largest estimate, {max(errs):.4f}, and keep quoting it as provisional.")


if __name__ == "__main__":
    main()
