#!/usr/bin/env python3
"""Agreement of the two samplers, and the cluster critical temperature.

Part 4 of the week-3 sheet, checks (1) and (2). Both update rules target the
same Boltzmann distribution, so `<|m|>` at the same `L` and `T` has to agree
within its sampling errors:

    d = | <|m|>_1 - <|m|>_2 | / sqrt(sigma_1^2 + sigma_2^2)      (Equation 18)

The left panel draws mean `|M|` against temperature for `L = 64` from the
Metropolis critical-window run and from the Wolff run, with block-bootstrap
error bars. The right panel draws the same absolute-magnetization
susceptibility Part 2 used,

    chi(T) = L^2 (<m^2> - <|m|>^2) / T                            (Equation 10)

from the cluster runs, with each size's five-point parabola, the fitted peaks
and the extrapolated `T_c = 2 T_peak(64) - T_peak(32)` (Equation 11), with
`2.26919` marked.

The errors are block-bootstrap standard deviations of the mean at block
lengths 2000, 4000 and 8000 steps, the same three lengths and replicate count
`chi_bootstrap.py` uses, and they get the same stability verdict: stable when
the three agree within a tenth of their mean. The sheet's rule is then

    both errors stable and d <= 3   -> report agreement
    either error still block-length sensitive -> "agreement provisional"
    d > 3                           -> a discrepancy to investigate

Part 3 already found the Metropolis error at `L = 64`, `T = 2.3` unresolved
(`evidence/acf-binning.png` has no plateau), so the verdict there is
provisional whatever `d` comes out; the report below says which side is
unstable. The cluster `T_c` bootstrap reuses `chi_bootstrap.bootstrap_tc`, so
the cluster error and the Part 3 error come from the same code.

Run:  .venv/bin/python scripts/magnetization_compare.py
Writes:  week3/evidence/magnetization-compare.png
         week3/evidence/magnetization-compare.txt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import chi_bootstrap  # noqa: E402
import errors  # noqa: E402
import peaks  # noqa: E402

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent


def display_path(path: Path) -> str:
    """Report a path relative to week3/ when it lives there, so it is portable."""
    try:
        return str(path.resolve().relative_to(WEEK))
    except ValueError:
        return str(path)

#: The update rules' names, which are also their artifact folder prefixes.
SAMPLERS = ("metropolis", "wolff")
CLUSTER_RUNS = {32: "wolff-l32", 64: "wolff-l64"}

#: The lattice size the two samplers are compared on, and where `d` is quoted.
LATTICE = 64
T_COMPARE = 2.3

#: The sheet's threshold, and the block lengths and replicate count Part 3 used.
D_LIMIT = 3.0
BLOCK_LENGTHS = chi_bootstrap.BLOCK_LENGTHS
REPLICATES = chi_bootstrap.REPLICATES

#: The block whose bootstrap error the chart draws: the longest one is the
#: closest either sample length gets to an honest bar.
ERROR_BLOCK = BLOCK_LENGTHS[-1]

COLORS = {"metropolis": "tab:blue", "wolff": "tab:red"}
SIZE_COLORS = {32: "tab:green", 64: "tab:blue"}


def d_statistic(m1: float, m2: float, s1: float, s2: float) -> float:
    """Equation 18: the difference in units of the combined standard error."""
    combined = math.sqrt(s1 * s1 + s2 * s2)
    if combined == 0.0:
        return 0.0 if m1 == m2 else float("inf")
    return abs(m1 - m2) / combined


def agreement_verdict(d: float, stable: dict[str, bool]) -> str:
    """The sheet's verdict for one `d` and the two stability flags."""
    unstable = [name for name in SAMPLERS if not stable[name]]
    suffix = ""
    if len(unstable) == 1:
        suffix = f" the {unstable[0]} error still depends on the block length"
    elif unstable:
        suffix = (
            f" the {' and '.join(unstable)} errors still depend on the block length"
        )
    if d > D_LIMIT:
        text = (
            f"discrepancy: d = {d:.2f} > {D_LIMIT:.0f}, investigate equilibration"
            " and sampling"
        )
        return f"{text};{suffix}" if suffix else text
    if suffix:
        return (
            f"agreement provisional: d = {d:.2f} <= {D_LIMIT:.0f}, but{suffix}"
        )
    return (
        f"agreement: d = {d:.2f} <= {D_LIMIT:.0f} and both errors are stable in"
        " Part 3's block-length sense"
    )


def block_bootstrap_mean_error(
    series: np.ndarray,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> float:
    """Block-bootstrap standard deviation of the mean of `|series|`.

    Blocks are drawn with replacement and the resampled mean is the mean of
    the drawn block means, exactly as `chi_bootstrap.py` resamples `chi`: the
    blocks all have the same length, so no 100000-row copy is ever made.
    """
    a = np.abs(np.asarray(series, dtype=float))
    blocks = chi_bootstrap.block_means(a, block_length)
    if blocks.size < 2:
        return 0.0
    drawn = rng.integers(0, blocks.size, (replicates, blocks.size))
    return float(blocks[drawn].mean(axis=1).std(ddof=1))


def block_errors(
    series: np.ndarray,
    block_lengths: list[int],
    replicates: int,
    seed: int,
) -> dict[int, float]:
    """`{block length: bootstrap error}` for one `|M|` series."""
    return {
        length: block_bootstrap_mean_error(
            series, length, replicates, np.random.default_rng([seed, length])
        )
        for length in block_lengths
    }


def read_run(folder: Path) -> tuple[int, dict[float, np.ndarray]]:
    """`(L, {T: signed M})` for one run folder."""
    lattice, series = errors.read_series(folder)
    return lattice, {t: m for t, m in series}


def read_energy(folder: Path) -> tuple[int, dict[float, np.ndarray]]:
    """`(L, {T: energy per site})` for one run folder."""
    lattice, series = errors.read_series_energy(folder)
    return lattice, {t: e for t, _, e in series}


def cluster_window(artifacts: Path) -> dict[int, dict[float, np.ndarray]]:
    """`{L: {T: signed M}}` of the two cluster runs, for the bootstrap."""
    window: dict[int, dict[float, np.ndarray]] = {}
    for lattice in (32, 64):
        run_lattice, series = read_run(artifacts / CLUSTER_RUNS[lattice])
        if run_lattice != lattice:
            raise SystemExit(f"{CLUSTER_RUNS[lattice]} holds L = {run_lattice}")
        window[lattice] = series
    return window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=WEEK / "artifacts")
    parser.add_argument(
        "--out", type=Path, default=WEEK / "evidence" / "magnetization-compare.png"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=WEEK / "evidence" / "magnetization-compare.txt",
    )
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _, metropolis = read_run(args.artifacts / peaks.WINDOW_RUNS[LATTICE])
    _, wolff = read_run(args.artifacts / CLUSTER_RUNS[LATTICE])
    temperatures = sorted(set(metropolis) & set(wolff))
    if len(temperatures) != 13:
        raise SystemExit(
            f"the window runs share {len(temperatures)} temperatures, not 13"
        )
    if not any(abs(t - T_COMPARE) < peaks.T_TOL for t in temperatures):
        raise SystemExit(f"T = {T_COMPARE} is not on the shared grid")

    curves = {
        "metropolis": {t: float(np.abs(metropolis[t]).mean()) for t in temperatures},
        "wolff": {t: float(np.abs(wolff[t]).mean()) for t in temperatures},
    }
    bars: dict[str, dict[float, dict[int, float]]] = {}
    stability: dict[str, dict[float, bool]] = {}
    for name, series in (("metropolis", metropolis), ("wolff", wolff)):
        bars[name] = {}
        stability[name] = {}
        for t in temperatures:
            table = block_errors(
                series[t], BLOCK_LENGTHS, args.replicates, args.seed
            )
            bars[name][t] = table
            stable, _ = chi_bootstrap.stability_verdict(table)
            stability[name][t] = stable

    index = int(np.argmin([abs(t - T_COMPARE) for t in temperatures]))
    t_compare = temperatures[index]
    m_metropolis = curves["metropolis"][t_compare]
    m_wolff = curves["wolff"][t_compare]
    s_metropolis = bars["metropolis"][t_compare][ERROR_BLOCK]
    s_wolff = bars["wolff"][t_compare][ERROR_BLOCK]
    d = d_statistic(m_metropolis, m_wolff, s_metropolis, s_wolff)
    verdict = agreement_verdict(
        d,
        {
            "metropolis": stability["metropolis"][t_compare],
            "wolff": stability["wolff"][t_compare],
        },
    )

    # A second observable, because a single matching mean does not establish
    # that the two chains sample the same distribution. The energy is not the
    # order parameter, so it is a genuinely different check on the samplers.
    _, energy_metropolis = read_energy(args.artifacts / peaks.WINDOW_RUNS[LATTICE])
    _, energy_wolff = read_energy(args.artifacts / CLUSTER_RUNS[LATTICE])
    e_metropolis = float(energy_metropolis[t_compare].mean())
    e_wolff = float(energy_wolff[t_compare].mean())
    e_bars = {
        "metropolis": block_errors(
            energy_metropolis[t_compare], BLOCK_LENGTHS, args.replicates, args.seed
        ),
        "wolff": block_errors(
            energy_wolff[t_compare], BLOCK_LENGTHS, args.replicates, args.seed
        ),
    }
    d_energy = d_statistic(
        e_metropolis,
        e_wolff,
        e_bars["metropolis"][ERROR_BLOCK],
        e_bars["wolff"][ERROR_BLOCK],
    )
    energy_stable = {
        name: chi_bootstrap.stability_verdict(e_bars[name])[0] for name in SAMPLERS
    }
    energy_verdict = agreement_verdict(d_energy, energy_stable)
    spread_table = []
    for name, m_series, e_series in (
        ("metropolis", metropolis[t_compare], energy_metropolis[t_compare]),
        ("wolff", wolff[t_compare], energy_wolff[t_compare]),
    ):
        for observable, series in (
            ("|m|", np.abs(m_series)),
            ("E", e_series),
        ):
            spread_table.append(
                (
                    name,
                    observable,
                    float(series.mean()),
                    float(series.std(ddof=1)),
                    errors.tau_int(series),
                )
            )

    # The cluster susceptibility and its peaks come from the same window rows
    # the cluster bootstrap resamples, through the same fit Part 2 used.
    window = cluster_window(args.artifacts)
    fits, central_tc = chi_bootstrap.central_fits(window)
    if not np.isfinite(central_tc):
        raise SystemExit("the cluster runs have no five-point peak fit for both sizes")
    results = [
        chi_bootstrap.bootstrap_tc(
            window, length, args.replicates, args.seed
        )
        for length in BLOCK_LENGTHS
    ]
    errors_by_length = {r.block_length: r.error for r in results}
    tc_stable, tc_verdict = chi_bootstrap.stability_verdict(errors_by_length)
    deviation = (central_tc - peaks.T_C) / peaks.T_C * 100.0

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.6, 4.8), dpi=160, gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    grid = np.array(temperatures)
    for name in SAMPLERS:
        means = np.array([curves[name][t] for t in temperatures])
        sigma = np.array([bars[name][t][ERROR_BLOCK] for t in temperatures])
        left.errorbar(
            grid,
            means,
            yerr=sigma,
            fmt="o-",
            color=COLORS[name],
            markersize=4.5,
            linewidth=1.3,
            capsize=3.0,
            # Open circles for Metropolis so the two curves stay readable
            # where they coincide above T = 2.35.
            markerfacecolor="white" if name == "metropolis" else COLORS[name],
            label=(
                f"{name.capitalize()}, $L$ = {LATTICE}:"
                f" error bars from {ERROR_BLOCK}-step blocks"
            ),
        )
    left.axvline(
        peaks.T_C,
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label=f"$T_c$ = {peaks.T_C:.5f}",
    )
    left.set_xlabel("temperature $T$")
    left.set_ylabel(r"$\langle |m| \rangle$")
    left.set_xlim(min(temperatures) - 0.02, max(temperatures) + 0.02)
    left.set_ylim(0.0, 1.02)
    left.set_title("The two samplers at $L$ = 64", fontsize=10)
    left.grid(alpha=0.25, linewidth=0.6)
    left.legend(loc="lower left", fontsize=8)
    left.annotate(
        f"$T$ = {t_compare:.2f}: metropolis {m_metropolis:.4f}, wolff {m_wolff:.4f}\n"
        f"$d$ = {d:.2f} ({ERROR_BLOCK}-step errors) -> {verdict.split(':')[0]}",
        xy=(0.02, 0.28),
        xycoords="axes fraction",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    for lattice in (32, 64):
        ts = sorted(window[lattice])
        chis = np.array(
            [
                peaks.chi_at(
                    lattice,
                    peaks.TemperatureStats(
                        t,
                        float(np.abs(window[lattice][t]).mean()),
                        float((window[lattice][t] ** 2).mean()),
                        window[lattice][t].size,
                    ),
                )[1]
                for t in ts
            ]
        )
        fit = fits[lattice]
        right.plot(
            ts,
            chis,
            "o",
            color=SIZE_COLORS[lattice],
            markersize=4.5,
            label=f"$L$ = {lattice}: $\\chi(T)$",
        )
        fine = np.linspace(fit.t_min, fit.t_max, 101)
        right.plot(
            fine,
            fit.chi_peak + fit.curvature * (fine - fit.t_peak) ** 2,
            color=SIZE_COLORS[lattice],
            linewidth=1.3,
            label=f"$L$ = {lattice}: fit, $T_{{peak}}$ = {fit.t_peak:.4f}",
        )
        right.axvline(
            fit.t_peak, color=SIZE_COLORS[lattice], linestyle=":", linewidth=1.0
        )
    right.axvline(
        peaks.T_C,
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label=f"$T_c$ = {peaks.T_C:.5f}",
    )
    right.axvline(
        central_tc,
        color="0.35",
        linestyle="-.",
        linewidth=1.1,
        label=f"cluster $T_c$ = {central_tc:.4f} ({deviation:+.2f}%)",
    )
    right.set_xlabel("temperature $T$")
    right.set_ylabel(r"$\chi(T)$")
    right.set_xlim(min(temperatures) - 0.02, max(temperatures) + 0.02)
    right.set_title("Cluster susceptibility and the extrapolated $T_c$", fontsize=10)
    right.grid(alpha=0.25, linewidth=0.6)
    right.legend(loc="upper right", fontsize=8)

    figure.suptitle(
        "Sampler agreement and the cluster critical temperature, "
        f"{args.replicates} bootstrap replicates"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)

    lines: list[str] = []
    lines.append("week3 part 4 - the two samplers and the cluster critical temperature")
    lines.append("")
    lines.append(
        f"L = {LATTICE}, critical window {temperatures[0]:.2f} .. "
        f"{temperatures[-1]:.2f} ({len(temperatures)} temperatures). Metropolis rows"
        f" come from artifacts/{peaks.WINDOW_RUNS[LATTICE]} (2000 discarded sweeps,"
        " 100000 measured sweeps per temperature); cluster rows come from"
        f" artifacts/{CLUSTER_RUNS[LATTICE]} and artifacts/{CLUSTER_RUNS[32]}"
        " (20000 discarded moves, 100000 measured moves per temperature)."
    )
    lines.append("")
    lines.append("(1) sampler agreement, Equation 18")
    lines.append("")
    lines.append(
        f"  block-bootstrap error of <|m|> at T = {t_compare:.2f}, "
        f"{args.replicates} replicates per block length"
    )
    lines.append(
        f"    {'block':>7}  {'metropolis (sweeps)':>20}  {'wolff (moves)':>16}  stable?"
    )
    for length in BLOCK_LENGTHS:
        lines.append(
            f"    {length:>7}  {bars['metropolis'][t_compare][length]:>20.6f}"
            f"  {bars['wolff'][t_compare][length]:>16.6f}"
            f"  metropolis {_flag(stability['metropolis'][t_compare])},"
            f" wolff {_flag(stability['wolff'][t_compare])}"
        )
    lines.append("")
    lines.append(
        f"  mean |m| at T = {t_compare:.2f}: metropolis {m_metropolis:.4f},"
        f" wolff {m_wolff:.4f} (difference {m_metropolis - m_wolff:+.4f})"
    )
    lines.append(
        f"  errors used: {ERROR_BLOCK}-step blocks, metropolis"
        f" {s_metropolis:.6f}, wolff {s_wolff:.6f}"
    )
    lines.append(f"  d = |{m_metropolis:.4f} - {m_wolff:.4f}| / sqrt({s_metropolis:.6f}^2"
                 f" + {s_wolff:.6f}^2) = {d:.2f}")
    lines.append(f"  verdict: {verdict}")
    lines.append("")
    lines.append(
        "  The sheet requires both errors to be stable in Part 3's block-length"
        " sense before agreement is reported as settled. Part 3 found the"
        " metropolis error at L = 64, T = 2.3 unresolved (the binning curve of"
        " evidence/acf-binning.png still rises at 20000-sweep blocks); if the"
        " metropolis error is still unstable here the verdict is provisional"
        " whatever d is, and a d above 3 would be a discrepancy to investigate"
        " rather than a sampler failure."
    )
    lines.append("")
    lines.append("(1b) a second observable at the same temperature: energy per site")
    lines.append("")
    lines.append(
        f"  machine-readable mean(E): metropolis {e_metropolis:+.4f},"
        f" wolff {e_wolff:+.4f} (difference {e_metropolis - e_wolff:+.4f})"
    )
    lines.append(
        "    "
        + "  ".join(
            f"{length}-step: metropolis {e_bars['metropolis'][length]:.5f},"
            f" wolff {e_bars['wolff'][length]:.5f}"
            for length in BLOCK_LENGTHS
        )
    )
    lines.append(
        f"  d(E) = {d_energy:.2f} with the {ERROR_BLOCK}-step errors;"
        f" verdict: {energy_verdict}"
    )
    lines.append(
        "  A single matching mean does not establish that the two chains sample"
        " the same distribution, so the energy -- not the order parameter -- is"
        " checked as well. The energy's error bar is the smaller one mostly"
        " because its spread is smaller; the two samplers' own clocks are in the"
        " table below, where a Wolff row counts moves and a Metropolis row counts"
        " sweeps."
    )
    lines.append(
        f"    {'sampler':>11}  {'observable':>10}  {'mean':>10}  {'sd':>8}"
        f"  {'tau_int':>9}"
    )
    for name, observable, mean, spread, tau in spread_table:
        lines.append(
            f"    {name:>11}  {observable:>10}  {mean:>10.4f}  {spread:>8.4f}"
            f"  {tau:>9.3f}"
        )
    lines.append("")
    lines.append("(2) the cluster critical temperature")
    lines.append("")
    for lattice in (32, 64):
        fit = fits[lattice]
        lines.append(
            f"  L = {lattice}: largest chi at T = {fit.grid_peak_t:.2f}"
            f" (chi = {fit.grid_peak_chi:.4f}); five-point fit"
            f" T_peak = {fit.t_peak:.4f} over T = {fit.t_min:.2f} .. {fit.t_max:.2f}"
        )
    lines.append(
        f"  T_c = 2 T_peak(64) - T_peak(32) = {central_tc:.4f}; exact"
        f" {peaks.T_C:.5f}; deviation {deviation:+.2f}% (gate: within 2%)"
    )
    lines.append("")
    lines.append("  block-bootstrap error of the cluster T_c")
    lines.append(f"    {'block':>7}  {'successful':>10}  {'failed':>6}  {'error':>8}")
    for result in results:
        lines.append(
            f"    {result.block_length:>7}  {result.successes:>7}/{result.replicates}"
            f"  {result.failures:>6}  {result.error:>8.4f}"
        )
    lines.append(f"  stability: {tc_verdict}")
    lines.append(
        "  The cluster error is a sampling error only: the finite-size shift of"
        " Equation 11 and the five-point parabola's own bias are outside it, so"
        " it is a lower bound on the uncertainty of T_c."
    )
    lines.append("")
    lines.append("chart")
    lines.append(f"  {display_path(args.out)}")
    report = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(report, end="")
    print(f"wrote {args.out}")
    print(f"wrote {args.report}")


def _flag(stable: bool) -> str:
    return "STABLE" if stable else "unstable"


if __name__ == "__main__":
    main()
