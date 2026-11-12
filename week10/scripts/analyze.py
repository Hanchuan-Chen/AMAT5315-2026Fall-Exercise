#!/usr/bin/env python3
"""Recompute errors, peaks, checks, tables, and the three required figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ONSAGER = 2.0 / math.log(1.0 + math.sqrt(2.0))
COARSE_K = np.round(np.arange(0.40, 0.5000001, 0.0025), 8)
FINE_H = 0.00125
COARSE_H = 0.0025


def run_dir(method: str, length: int, chi: int, k: float) -> Path:
    return RUNS / method / f"L{length}" / f"chi{chi}" / f"K{k:.5f}"


def load_run(method: str, length: int, chi: int, k: float) -> dict:
    return json.loads(
        (run_dir(method, length, chi, k) / "evidence.json").read_text()
    )


def c_curve(ks: np.ndarray, gs: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    values = (ks[1:-1] ** 2) * (gs[2:] - 2.0 * gs[1:-1] + gs[:-2]) / h**2
    return ks[1:-1], values


def check_c_curve_regression() -> dict:
    ks = np.arange(0.40, 0.5000001, 0.0025)
    k0 = 0.4475
    curvature = 3.0
    gs = 2.0 + 0.5 * curvature * (ks - k0) ** 2
    centers, measured = c_curve(ks, gs, 0.0025)
    expected = centers**2 * curvature
    center_index = int(np.argmin(np.abs(centers - k0)))
    actual_at_k0 = float(measured[center_index])
    expected_at_k0 = float(k0**2 * curvature)
    return {
        "synthetic_function": "g(K)=2+1.5*(K-0.4475)^2",
        "center_points_match": bool(np.array_equal(centers, ks[1:-1])),
        "measured_at_k0": actual_at_k0,
        "expected_at_k0": expected_at_k0,
        "abs_error_at_k0": abs(actual_at_k0 - expected_at_k0),
        "max_abs_error": float(np.max(np.abs(measured - expected))),
        "matches_expected": bool(
            np.allclose(measured, expected, rtol=1e-10, atol=1e-12)
        ),
    }


def parabola_vertex(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    i = int(np.argmax(ys))
    if i == 0 or i == len(xs) - 1:
        raise ValueError("peak is not bracketed by three points")
    step = xs[i + 1] - xs[i]
    offset = 0.5 * step * (ys[i - 1] - ys[i + 1]) / (
        ys[i - 1] - 2.0 * ys[i] + ys[i + 1]
    )
    x_peak = xs[i] + offset
    y_peak = ys[i] - 0.25 * (ys[i - 1] - ys[i + 1]) * offset / step
    return float(x_peak), float(y_peak)


def coarse_peak(method: str, length: int) -> float:
    gs = np.array([load_run(method, length, 16, float(k))["g"] for k in COARSE_K])
    ks, cvs = c_curve(COARSE_K, gs, COARSE_H)
    return float(ks[int(np.argmax(cvs))])


def refinement_k(k0: float) -> np.ndarray:
    return np.round(k0 + FINE_H * np.arange(-4, 5), 8)


def analyze_svd() -> dict:
    evidence = json.loads((RUNS / "svd" / "evidence.json").read_text())
    n = evidence["size"]
    matrix = np.asarray(evidence["matrix"], dtype=float).reshape(n, n)
    singular = np.asarray(evidence["singular_values"], dtype=float)
    ranks = evidence["ranks"]
    original_norm = np.linalg.norm(matrix)
    measured = []
    predicted = []
    for rank, flat in zip(ranks, evidence["reconstructions"]):
        reconstruction = np.asarray(flat, dtype=float).reshape(n, n)
        measured.append(float(np.linalg.norm(matrix - reconstruction) / original_norm))
        predicted.append(float(np.sqrt(np.sum(singular[rank:] ** 2) / np.sum(singular**2))))
    measured = np.asarray(measured)
    predicted = np.asarray(predicted)

    all_ranks = np.arange(1, n + 1)
    tail_squared = np.cumsum((singular**2)[::-1])[::-1]
    error_at_rank = np.sqrt(
        np.concatenate((tail_squared[1:], [0.0])) / np.sum(singular**2)
    )
    crossing_10 = int(np.flatnonzero(error_at_rank <= 0.10)[0])
    equality_rank = int(math.ceil(n * n / (2 * n + 1)))
    summary = {
        "seed": evidence["seed"],
        "rng": evidence["rng"],
        "distribution": evidence["distribution"],
        "ranks": ranks,
        "measured_errors": measured.tolist(),
        "tail_predictions": predicted.tolist(),
        "max_measurement_prediction_difference": float(
            np.max(np.abs(measured - predicted))
        ),
        "full_rank_error": float(measured[-1]),
        "storage_equality_rank": equality_rank,
        "error_10_percent_rank": crossing_10,
        "storage_ratio_at_10_percent": float(
            crossing_10 * (2 * n + 1) / (n * n)
        ),
    }

    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(7.6, 3.35))
    ax_s.plot(np.arange(1, n + 1), singular, color="#2b6f9f", lw=1.7)
    ax_s.set(
        title="(a) Singular spectrum",
        xlabel="Singular-value index",
        ylabel="Singular value",
        xlim=(0, n),
        ylim=(0, None),
    )
    ax_e.axvspan(equality_rank, n, color="#dfeef8", alpha=0.85, lw=0)
    ax_e.plot(all_ranks, error_at_rank, color="#555555", ls="--", lw=1.2,
              label="SVD tail prediction")
    ax_e.scatter(ranks, measured, color="#d97706", s=18, zorder=3,
                 label="Reconstruction")
    ax_e.axhline(0.10, color="#555555", lw=0.8, ls=":")
    ax_e.axvline(crossing_10, color="#2b6f9f", lw=0.8, ls=":")
    ax_e.axvline(equality_rank, color="#2b6f9f", lw=0.8, ls=":")
    ax_e.text(crossing_10 + 8, 0.13, rf"$\chi={crossing_10}$", color="#2b6f9f")
    ax_e.text(
        equality_rank + 8,
        0.56,
        rf"$\chi={equality_rank}$",
        color="#2b6f9f",
    )
    ax_e.text(n - 34, 0.025, "10%", ha="right")
    ax_e.text(
        (equality_rank + n) / 2, 0.72, "no storage\nsaving",
        ha="center", color="#2b6f9f",
    )
    ax_e.set(
        title="(b) Error versus retained rank",
        xlabel=r"Retained rank $\chi$",
        ylabel="Relative Frobenius error",
        xlim=(0, n),
        ylim=(0, 1.02),
    )
    ax_e.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "svd.png", dpi=180)
    plt.close(fig)
    return summary


def rank_peak(method: str, length: int, chi: int, ks: np.ndarray) -> dict:
    gs = np.array([load_run(method, length, chi, float(k))["g"] for k in ks])
    center, values = c_curve(ks, gs, FINE_H)
    k_peak, height = parabola_vertex(center, values)
    every_other = ks[::2]
    gs_wide = gs[::2]
    center_wide, values_wide = c_curve(every_other, gs_wide, COARSE_H)
    k_wide, height_wide = parabola_vertex(center_wide, values_wide)
    return {
        "k_peak": k_peak,
        "T_peak": 1.0 / k_peak,
        "height": height,
        "wider_k_peak": k_wide,
        "wider_T_peak": 1.0 / k_wide,
        "wider_height": height_wide,
    }


def linear_fit(lengths: list[int], temps: list[float]) -> dict:
    inv = np.asarray([1.0 / length for length in lengths])
    temp = np.asarray(temps)
    slope, intercept = np.polyfit(inv, temp, 1)
    return {
        "sizes": lengths,
        "slope": float(slope),
        "Tc": float(intercept),
        "relative_error_vs_onsager": float(abs(intercept - ONSAGER) / ONSAGER),
    }


def analyze_critical(method: str, lengths: list[int], figure: str) -> dict:
    peaks = {}
    curves = {}
    for length in lengths:
        k0 = coarse_peak(method, length)
        fine_ks = refinement_k(k0)
        per_rank = {
            str(chi): rank_peak(method, length, chi, fine_ks) for chi in (16, 32)
        }
        peaks[str(length)] = {
            "coarse_nearest_K": k0,
            "nearest_T": 1.0 / k0,
            **per_rank,
        }
        gs = np.array(
            [load_run(method, length, 16, float(k))["g"] for k in COARSE_K]
        )
        center, cvs = c_curve(COARSE_K, gs, COARSE_H)
        order = np.argsort(1.0 / center)
        curves[str(length)] = {
            "T": (1.0 / center)[order].tolist(),
            "cV": cvs[order].tolist(),
        }

    final_temps = [peaks[str(length)]["32"]["T_peak"] for length in lengths]
    fits = {
        "all_sizes": linear_fit(lengths, final_temps),
        "largest_two": linear_fit(lengths[1:], final_temps[1:]),
    }
    stability = {}
    for length in lengths:
        data = peaks[str(length)]
        stability[str(length)] = {
            "rank_relative_change": abs(data["32"]["T_peak"] / data["16"]["T_peak"] - 1.0),
            "step_relative_change": abs(
                data["32"]["T_peak"] / data["32"]["wider_T_peak"] - 1.0
            ),
        }

    colors = {16: "#5c6470", 32: "#d97706", 64: "#247ba0"}
    if method == "trg":
        colors = {64: "#247ba0", 256: "#756bb1", 1024: "#3d8b63"}
    markers = {16: "s", 32: "^", 64: "o", 256: "s", 1024: "^"}

    fig, (ax_curve, ax_fit) = plt.subplots(1, 2, figsize=(9.2, 3.7))
    fine_colors = {16: "none", 32: "#d95f02"}
    fine_markers = {16: "s", 32: "o"}
    for length in lengths:
        curve = curves[str(length)]
        label = f"L = {length}"
        ax_curve.plot(curve["T"], curve["cV"], color=colors[length], lw=1.35,
                      label=label)
        for chi in (16, 32):
            ks = refinement_k(coarse_peak(method, length))
            gs = np.array(
                [load_run(method, length, chi, float(k))["g"] for k in ks]
            )
            tt, refined = c_curve(ks, gs, FINE_H)
            ax_curve.scatter(
                1.0 / tt, refined, s=13, marker=fine_markers[chi],
                facecolors=fine_colors[chi], edgecolors=colors[length],
                linewidths=0.8, zorder=3,
            )
    ax_curve.axvline(ONSAGER, color="#555555", lw=0.8, ls=":")
    ax_curve.text(ONSAGER + 0.003, 0.03, r"$T_c$", transform=ax_curve.get_xaxis_transform())
    ax_curve.set(
        title="(a) Open specific-heat peaks" if method == "boundary"
        else "(a) Periodic specific-heat peaks",
        xlabel="Temperature T",
        ylabel="Specific heat per spin",
        ylim=(0, None),
    )
    ax_curve.plot([], [], color="#555555", marker="s", ms=3.5, mfc="none",
                  lw=0, label=r"$\chi=16$, h=0.0025")
    ax_curve.plot([], [], color="#d95f02", marker="o", ms=3.5, mfc="none",
                  lw=0, label=r"$\chi=32$, h=0.00125")
    ax_curve.legend(frameon=False, fontsize=8)

    inv = np.array([1.0 / length for length in lengths])
    ax_fit.scatter(inv, final_temps, c=[colors[length] for length in lengths],
                   marker="D", s=28, zorder=3)
    for x, y, length in zip(inv, final_temps, lengths):
        ax_fit.annotate(f"L = {length}", (x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=8,
                        color=colors[length])
    grid = np.linspace(0, max(inv) * 1.04, 50)
    for key, style, label in (
        ("all_sizes", "-", "all sizes"),
        ("largest_two", "--", "largest two"),
    ):
        fit = fits[key]
        ax_fit.plot(grid, fit["Tc"] + fit["slope"] * grid, color="#555555",
                    ls=style, lw=1.1, label=label)
    ax_fit.axhline(ONSAGER, color="#555555", lw=0.8, ls=":")
    ax_fit.text(max(inv) * 1.015, ONSAGER, "Onsager", va="bottom", ha="right",
                fontsize=8)
    ax_fit.set(
        title="(b) Extrapolate the peak",
        xlabel="Inverse side length 1/L",
        ylabel="Peak temperature",
        xlim=(0, max(inv) * 1.04),
    )
    ax_fit.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / figure, dpi=180)
    plt.close(fig)
    return {"peaks": peaks, "fits": fits, "stability": stability, "curves": curves}


def generic_evidence(path: Path) -> dict:
    return json.loads((path / "evidence.json").read_text())


def analyze_checks() -> dict:
    checks = {"analyzer": {"c_curve_quadratic": check_c_curve_regression()},
              "boundary": {}, "trg": {}}
    for k in (0.0, 0.44, 0.7):
        exact = generic_evidence(RUNS / "checks" / "exact" / "open4" / f"K{k:.5f}")
        numeric = generic_evidence(RUNS / "checks" / "boundary" / "open4" / f"K{k:.5f}")
        checks["boundary"][f"K={k:.2f}"] = {
            "exact_g": exact["g"],
            "mps_g": numeric["g"],
            "abs_error": abs(numeric["g"] - exact["g"]),
        }
    boundary_check = [
        generic_evidence(
            RUNS / "checks" / "boundary" / "open4" / f"K{k:.5f}"
        )["g"]
        for k in (0.4375, 0.44, 0.4425)
    ]
    boundary_c = 0.44**2 * (
        boundary_check[2] - 2.0 * boundary_check[1] + boundary_check[0]
    ) / 0.0025**2
    exact_mid = generic_evidence(RUNS / "checks" / "exact" / "open4" / "K0.44000")
    checks["boundary"]["specific_heat"] = {
        "mps_cV": boundary_c,
        "exact_cV": exact_mid["specific_heat"],
        "abs_error": abs(boundary_c - exact_mid["specific_heat"]),
    }

    checks["trg"]["L=2"] = {}
    exact2 = generic_evidence(RUNS / "checks" / "exact" / "periodic2" / "K0.44000")
    numeric2 = generic_evidence(RUNS / "checks" / "trg" / "periodic2" / "K0.44000")
    checks["trg"]["L=2"]["g"] = {
        "exact_g": exact2["g"],
        "trg_g": numeric2["g"],
        "abs_error": abs(numeric2["g"] - exact2["g"]),
    }
    trg_check = [
        generic_evidence(
            RUNS / "checks" / "trg" / "periodic2" / f"K{k:.5f}"
        )["g"]
        for k in (0.4375, 0.44, 0.4425)
    ]
    trg_c = 0.44**2 * (
        trg_check[2] - 2.0 * trg_check[1] + trg_check[0]
    ) / 0.0025**2
    checks["trg"]["L=2"]["specific_heat"] = {
        "trg_cV": trg_c,
        "exact_cV": exact2["specific_heat"],
        "abs_error": abs(trg_c - exact2["specific_heat"]),
    }
    exact4 = generic_evidence(RUNS / "checks" / "exact" / "periodic4" / "K0.44000")
    numeric4 = generic_evidence(RUNS / "checks" / "trg" / "periodic4" / "K0.44000")
    checks["trg"]["L=4"] = {
        "exact_g": exact4["g"],
        "trg_g": numeric4["g"],
        "abs_error": abs(numeric4["g"] - exact4["g"]),
    }
    boundary_saved = generic_evidence(RUNS / "boundary64")
    vector = np.ones(1)
    for core in boundary_saved["final_tensors"]:
        left, _, right = core["shape"]
        data = np.asarray(core["data"]).reshape(left, 2, right)
        vector = np.einsum(
            "a,apb,p->b", vector, data, boundary_saved["recomputation"]["missing_bond"]
        )
    boundary_log_z = boundary_saved["log_scale"] + math.log(float(vector[0]))
    checks["boundary"]["saved_evidence_recontraction"] = {
        "saved_log_z": boundary_saved["log_z"],
        "recomputed_log_z": boundary_log_z,
        "abs_error": abs(boundary_log_z - boundary_saved["log_z"]),
    }

    trg_saved = generic_evidence(RUNS / "trg1024")
    dim = trg_saved["final_tensor"]["shape"][0]
    trg_tensor = np.asarray(trg_saved["final_tensor"]["data"]).reshape(
        dim, dim, dim, dim
    )
    closure = float(np.einsum("ijij->", trg_tensor))
    trg_log_z = trg_saved["log_scale"] + math.log(closure)
    checks["trg"]["saved_evidence_reclosure"] = {
        "saved_log_z": trg_saved["log_z"],
        "recomputed_log_z": trg_log_z,
        "abs_error": abs(trg_log_z - trg_saved["log_z"]),
    }
    return checks


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_tables(critical: dict) -> None:
    for method in ("boundary", "trg"):
        rows = []
        lengths = (16, 32, 64) if method == "boundary" else (64, 256, 1024)
        for length in lengths:
            for k in COARSE_K:
                evidence = load_run(method, length, 16, float(k))
                rows.append([f"{k:.5f}", evidence["g"], evidence["solver_time_ms"]])
        write_csv(DATA / f"{method}-g.csv",
                  ["K", "g", "solver_time_ms"], rows)

        rows = []
        for length in lengths:
            data = critical[method]["peaks"][str(length)]
            solve_time_s = load_run(
                method, length, 32, data["coarse_nearest_K"]
            )["solver_time_ms"] / 1000.0
            for chi in ("16", "32"):
                fit = data[chi]
                rows.append([
                    length, chi, fit["k_peak"], fit["T_peak"], fit["height"],
                    fit["wider_T_peak"],
                    abs(fit["T_peak"] / fit["wider_T_peak"] - 1.0),
                    solve_time_s,
                ])
        write_csv(
            DATA / f"{method}-refinement.csv",
            ["L", "chi", "K_peak", "T_peak", "peak_cV",
             "T_peak_wider_step", "relative_step_change",
             "chi32_near_peak_solver_time_s"],
            rows,
        )


def validate(summary: dict) -> dict:
    svd = summary["svd"]
    b = summary["boundary"]
    t = summary["trg"]
    checks = summary["checks"]
    results = {
        "analyzer_c_curve_quadratic": (
            checks["analyzer"]["c_curve_quadratic"]["center_points_match"]
            and checks["analyzer"]["c_curve_quadratic"]["matches_expected"]
        ),
        "svd_measurement_matches_tail": svd["max_measurement_prediction_difference"] < 1e-8,
        "svd_full_rank": svd["full_rank_error"] < 1e-8,
        "boundary_enumeration": all(
            item["abs_error"] < 1e-8
            for key, item in checks["boundary"].items()
            if key not in ("specific_heat", "saved_evidence_recontraction")
        ),
        "boundary_saved_evidence": checks["boundary"]["saved_evidence_recontraction"][
            "abs_error"
        ] < 1e-12,
        "boundary_variance": checks["boundary"]["specific_heat"]["abs_error"] < 1e-3,
        "trg_L2_enumeration": checks["trg"]["L=2"]["g"]["abs_error"] < 1e-8,
        "trg_L4_enumeration": checks["trg"]["L=4"]["abs_error"] < 2e-3,
        "trg_L2_variance": checks["trg"]["L=2"]["specific_heat"]["abs_error"] < 1e-3,
        "trg_saved_evidence": checks["trg"]["saved_evidence_reclosure"][
            "abs_error"
        ] < 1e-12,
        "boundary_rank_stability": all(
            value["rank_relative_change"] < 0.01
            for value in b["stability"].values()
        ),
        "boundary_step_stability": all(
            value["step_relative_change"] < 0.01
            for value in b["stability"].values()
        ),
        "boundary_Tc_all": b["fits"]["all_sizes"]["relative_error_vs_onsager"] < 0.01,
        "boundary_Tc_largest": b["fits"]["largest_two"]["relative_error_vs_onsager"] < 0.01,
        "trg_rank_stability": all(
            value["rank_relative_change"] < 0.01
            for value in t["stability"].values()
        ),
        "trg_step_stability": all(
            value["step_relative_change"] < 0.01
            for value in t["stability"].values()
        ),
        "trg_Tc_all": t["fits"]["all_sizes"]["relative_error_vs_onsager"] < 0.01,
        "trg_Tc_largest": t["fits"]["largest_two"]["relative_error_vs_onsager"] < 0.01,
    }
    results["all_pass"] = all(results.values())
    return results


def analyze_all() -> dict:
    DATA.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    summary = {
        "onsager_Tc": ONSAGER,
        "svd": analyze_svd(),
        "boundary": analyze_critical("boundary", [16, 32, 64], "boundary-critical.png"),
        "trg": analyze_critical("trg", [64, 256, 1024], "trg-critical.png"),
        "checks": analyze_checks(),
    }
    summary["validation"] = validate(summary)
    write_tables(summary)
    (DATA / "checks.json").write_text(json.dumps(summary["checks"], indent=2) + "\n")
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit nonzero if a requirement fails")
    args = parser.parse_args()
    summary = analyze_all()
    print(json.dumps(summary["validation"], indent=2))
    if args.check and not summary["validation"]["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

