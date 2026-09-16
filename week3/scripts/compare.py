#!/usr/bin/env python3
"""Work-normalized autocorrelation times of the two samplers at L = 64.

Part 4 of the week-3 sheet, check (3). A cluster move flips `<c>` spins at
once, so in units of spin-update work -- one unit for `L^2` flipped spins, one
Metropolis sweep -- it costs

    tau_work = tau_moves * <c> / L^2                                (Equation 17)

The Metropolis curve is already in sweeps: one sweep is `L^2` proposed spins,
rejections included. Both curves come from the same 13-temperature critical
window and the same Equation 13 estimator (`errors.tau_int`), so the only
difference is the update rule.

The conversion counts spin-update work, not seconds: a cluster move's
bookkeeping costs more per spin than a single-spin proposal, so the factor is
an upper bound on the wall-clock speedup. It is still the honest count of work
per independent sample, which is what critical slowing down is about.

Run:  .venv/bin/python scripts/compare.py
Writes:  week3/evidence/tau-compare.png
         week3/evidence/tau-compare.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import errors  # noqa: E402
import peaks  # noqa: E402

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent

#: The size the sheet's work comparison uses.
LATTICE = 64
#: The temperature the work ratio is quoted at.
T_REPORT = 2.3

METROPOLIS = "metropolis"
WOLFF = "wolff"
COLORS = {METROPOLIS: "tab:blue", WOLFF: "tab:red"}


def tau_work(tau_moves: float, mean_cluster_size: float, lattice: int) -> float:
    """Equation 17: one step of `tau_moves` costs `mean_cluster_size / L^2` sweeps."""
    return tau_moves * mean_cluster_size / (lattice * lattice)


def work_ratio(metropolis_tau: float, wolff_tau_work: float) -> float:
    """How many times less spin-update work the cluster sampler spends."""
    return metropolis_tau / wolff_tau_work


def metropolis_curve(folder: Path) -> dict[float, float]:
    """`{T: tau_int of |M| in sweeps}` for a Metropolis run folder."""
    lattice, series = errors.read_series(folder)
    if lattice != LATTICE:
        raise SystemExit(f"{folder.name} holds L = {lattice}, not {LATTICE}")
    return {t: errors.tau_int(np.abs(m)) for t, m in series}


def wolff_curve(folder: Path) -> dict[float, tuple[float, float, float]]:
    """`{T: (tau_moves, mean cluster size, tau_work)}` for a Wolff run folder."""
    lattice, series = errors.read_wolff_series(folder)
    if lattice != LATTICE:
        raise SystemExit(f"{folder.name} holds L = {lattice}, not {LATTICE}")
    curve: dict[float, tuple[float, float, float]] = {}
    for t, m, cluster in series:
        moves = errors.tau_int(np.abs(m))
        mean_cluster = float(cluster.mean())
        curve[t] = (moves, mean_cluster, tau_work(moves, mean_cluster, lattice))
    return curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts")
    parser.add_argument(
        "--out", type=Path, default=WEEK / "evidence" / "tau-compare.png"
    )
    parser.add_argument(
        "--report", type=Path, default=WEEK / "evidence" / "tau-compare.txt"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metropolis = metropolis_curve(args.artifacts / peaks.WINDOW_RUNS[LATTICE])
    wolff = wolff_curve(args.artifacts / "wolff-l64")
    temperatures = sorted(set(metropolis) & set(wolff))
    if len(temperatures) != 13:
        raise SystemExit(
            f"the two runs share {len(temperatures)} temperatures, not 13"
        )

    ts = np.array(temperatures)
    sweeps = np.array([metropolis[t] for t in temperatures])
    moves = np.array([wolff[t][0] for t in temperatures])
    clusters = np.array([wolff[t][1] for t in temperatures])
    work = np.array([wolff[t][2] for t in temperatures])
    ratio = sweeps / work

    figure, axes = plt.subplots(figsize=(7.6, 4.9), dpi=160)
    axes.plot(
        ts,
        sweeps,
        "o-",
        color=COLORS[METROPOLIS],
        markersize=4.5,
        linewidth=1.3,
        label="Metropolis, one sweep per row record",
    )
    axes.plot(
        ts,
        work,
        "s-",
        color=COLORS[WOLFF],
        markersize=4.5,
        linewidth=1.3,
        label=r"Wolff, one move per row record: $\tau_{moves}\langle c\rangle/L^2$",
    )
    axes.axvline(
        peaks.T_C,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label=f"$T_c$ = {peaks.T_C:.4f}",
    )
    index = int(np.argmin(np.abs(ts - T_REPORT)))
    axes.annotate(
        f"$T$ = {ts[index]:.2f}\n"
        f"Metropolis {sweeps[index]:.1f} sweeps\n"
        f"Wolff {moves[index]:.2f} moves x $\\langle c\\rangle$ ="
        f" {clusters[index]:.1f}\n"
        f"= {work[index]:.2f} sweeps of work\n"
        f"ratio {ratio[index]:.0f}x",
        xy=(ts[index], work[index]),
        xytext=(0.05, 0.12),
        textcoords="axes fraction",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.4", "linewidth": 0.8},
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    axes.set_yscale("log")
    axes.set_xlabel("temperature $T$")
    axes.set_ylabel(r"$\tau_{work}$ (sweeps of spin-update work)")
    axes.set_xlim(min(temperatures) - 0.02, max(temperatures) + 0.02)
    axes.set_title(
        r"Work per independent sample, $L$ = 64: $\tau_{work} = \tau \langle c\rangle / L^2$"
    )
    axes.grid(alpha=0.25, linewidth=0.6, which="both")
    axes.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)

    lines: list[str] = []
    lines.append("week3 part 4 - work per independent sample at L = 64")
    lines.append("")
    lines.append(
        "Metropolis rows: artifacts/window-l64, tau_int of |M| from Equation 13,"
        " already in sweeps (100000 measured sweeps at each of the 13"
        " temperatures, 2000 discarded)."
    )
    lines.append(
        "Wolff rows: artifacts/wolff-l64, tau_moves of |M| from the same Equation"
        " 13 estimator in cluster moves, times <c> / L^2 (Equation 17; 100000"
        " measured moves at each temperature, 20000 discarded)."
    )
    lines.append("")
    lines.append(
        f"  {'T':>6} {'metropolis tau':>15} {'wolff tau (moves)':>18}"
        f" {'wolff <c>':>10} {'wolff tau_work':>15} {'ratio':>8}"
    )
    for i, t in enumerate(temperatures):
        lines.append(
            f"  {t:>6.2f} {sweeps[i]:>15.2f} {moves[i]:>18.3f}"
            f" {clusters[i]:>10.1f} {work[i]:>15.3f} {ratio[i]:>8.1f}"
        )
    lines.append("")
    lines.append(
        f"  at T = {ts[index]:.2f}: {sweeps[index]:.1f} sweeps against"
        f" {moves[index]:.2f} moves x {clusters[index]:.1f} / {LATTICE ** 2} ="
        f" {work[index]:.3f} sweeps, so the cluster sampler does {ratio[index]:.0f}"
        " times less spin-update work per independent sample"
    )
    lines.append(
        f"  the largest ratio on the window grid is {ratio.max():.0f} at"
        f" T = {ts[int(np.argmax(ratio))]:.2f}; the smallest is {ratio.min():.1f}"
        f" at T = {ts[int(np.argmin(ratio))]:.2f}"
    )
    lines.append("")
    lines.append("why more independent samples do not remove the remaining bias")
    lines.append("")
    lines.append(
        "  The cluster chain decorrelates in a few moves instead of hundreds of"
        " sweeps, so the same recorded length holds hundreds of times more"
        " independent samples and the sampling error of <|m|> and of T_c falls"
        " with 1/sqrt(n_eff). Three things do not fall: the finite-size shift,"
        " which the two-size extrapolation of Equation 11 cancels only to"
        " leading order in 1/L; the five-point parabola's own bias, because a"
        " parabola through five points of an asymmetric peak puts its vertex"
        " off the true maximum by a fixed offset that more sweeps at the same"
        " 13 temperatures cannot move; and any bias of the single random stream"
        " itself, such as residual equilibration, which a bootstrap over that"
        " stream's own rows cannot see. The bootstrap error reported in"
        " magnetization-compare.txt is therefore a sampling error only, and a"
        " lower bound on the uncertainty of T_c."
    )
    lines.append("")
    lines.append(
        "  The conversion is spin-update work, not wall clock: growing a cluster"
        " costs more per spin than proposing a single flip, and the recorded"
        " move does the bookkeeping for the whole cluster. The ratio is an upper"
        " bound on the elapsed-time speedup at the same temperature."
    )
    lines.append("")
    lines.append(f"chart {args.out}")
    report = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(report, end="")
    print(f"wrote {args.out}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
